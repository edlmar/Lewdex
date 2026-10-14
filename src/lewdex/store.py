"""SQLite (WAL) store: probability cache, file hashes, write provenance, migration state, runs.

The probability cache is keyed by file *content* (xxh3-128) plus the model key (repo revision,
precision, preprocessing version) plus the frame-sampling signature, so renaming or moving an
item never forces re-inference, and changing thresholds, aliases or merge rules only re-reads it.

Vectors are stored in full: probabilities quantised to uint16 (1/65535 steps), values below
``PROB_FLOOR`` zeroed (every threshold is far above it), byte-shuffled and zstd-compressed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

SCHEMA_VERSION = 1
PROB_FLOOR = 0.002
_FLOOR_Q = int(PROB_FLOOR * 65535)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
  item_id TEXT PRIMARY KEY,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  hash BLOB NOT NULL,
  media TEXT NOT NULL,
  updated REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS probs (
  hash BLOB NOT NULL,
  model TEXT NOT NULL,
  sampling TEXT NOT NULL,
  n_frames INTEGER NOT NULL,
  mask BLOB,
  data BLOB NOT NULL,
  created REAL NOT NULL,
  PRIMARY KEY (hash, model, sampling)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS probs_model ON probs (model, sampling);
CREATE TABLE IF NOT EXISTS managed (
  item_id TEXT PRIMARY KEY,
  tags TEXT NOT NULL,
  tagset TEXT NOT NULL,
  updated REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS backup (
  item_id TEXT NOT NULL,
  run_id INTEGER NOT NULL,
  removed TEXT NOT NULL,
  ts REAL NOT NULL,
  PRIMARY KEY (item_id, run_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY,
  started REAL NOT NULL,
  finished REAL,
  status TEXT,
  config TEXT,
  stats TEXT
);
"""

_tls = threading.local()


def _zc() -> zstandard.ZstdCompressor:
    c = getattr(_tls, "zc", None)
    if c is None:
        c = _tls.zc = zstandard.ZstdCompressor(level=3)
    return c


def _zd() -> zstandard.ZstdDecompressor:
    d = getattr(_tls, "zd", None)
    if d is None:
        d = _tls.zd = zstandard.ZstdDecompressor()
    return d


def encode_probs(arr: np.ndarray) -> bytes:
    """(rows, n_tags) probabilities in [0, 1] -> compressed bytes."""
    arr = np.atleast_2d(arr)
    q = np.rint(np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)
    q[q < _FLOOR_Q] = 0
    rows, cols = q.shape
    shuffled = q.view(np.uint8).reshape(-1, 2).T.copy()  # all low bytes, then all high bytes
    header = np.array([rows, cols], dtype=np.uint32).tobytes()
    return _zc().compress(header + shuffled.tobytes())


def decode_probs(blob: bytes) -> np.ndarray:
    raw = _zd().decompress(blob)
    rows, cols = np.frombuffer(raw[:8], dtype=np.uint32)
    planes = np.frombuffer(raw[8:], dtype=np.uint8).reshape(2, -1)
    q = planes.T.copy().view(np.uint16).reshape(int(rows), int(cols))
    return q.astype(np.float32) * np.float32(1.0 / 65535.0)


def mask_to_bytes(mask: Sequence[bool] | None) -> bytes | None:
    if mask is None or all(mask):
        return None
    return np.packbits(np.asarray(mask, dtype=bool)).tobytes()


def mask_from_bytes(blob: bytes | None, n_frames: int) -> np.ndarray:
    if blob is None:
        return np.ones(n_frames, dtype=bool)
    return np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:n_frames].astype(bool)


@dataclass(frozen=True)
class CachedProbs:
    n_frames: int
    mask: np.ndarray  # (n_frames,) bool: frames this model has rows for
    probs: np.ndarray  # (mask.sum(), n_tags) float32


class Store:
    """One SQLite file; one connection per thread; WAL so readers never block the writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        conn = self._conn()
        with conn:
            conn.executescript(_SCHEMA)
        row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is None:
            with conn:
                conn.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
        elif int(row[0]) > SCHEMA_VERSION:
            raise RuntimeError(f"{path} was written by a newer lewdex (schema {row[0]})")

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA cache_size=-65536")
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.execute("PRAGMA wal_checkpoint(PASSIVE)")
                c.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()

    # -- meta ------------------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self._conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def put_meta(self, key: str, value: str) -> None:
        conn = self._conn()
        with conn:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # -- files -----------------------------------------------------------------------------
    def all_files(self) -> dict[str, tuple[int, int, bytes, str]]:
        """item_id -> (size, mtime_ns, content hash, media kind "img"|"anim"|"vid")."""
        cur = self._conn().execute("SELECT item_id, size, mtime_ns, hash, media FROM files")
        return {r[0]: (int(r[1]), int(r[2]), bytes(r[3]), str(r[4])) for r in cur}

    def put_files(self, rows: Iterable[tuple[str, int, int, bytes, str]]) -> None:
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
                             [(i, s, m, h, media, now) for i, s, m, h, media in rows])

    # -- probability cache ----------------------------------------------------------------
    def cached_index(self, model: str) -> dict[tuple[bytes, str], tuple[int, bytes | None]]:
        """(hash, sampling) -> (n_frames, mask) for every cached entry of ``model``."""
        cur = self._conn().execute(
            "SELECT hash, sampling, n_frames, mask FROM probs WHERE model=?", (model,))
        return {(bytes(r[0]), str(r[1])): (int(r[2]), None if r[3] is None else bytes(r[3]))
                for r in cur}

    def get_probs(self, content_hash: bytes, model: str, sampling: str) -> CachedProbs | None:
        row = self._conn().execute(
            "SELECT n_frames, mask, data FROM probs WHERE hash=? AND model=? AND sampling=?",
            (content_hash, model, sampling)).fetchone()
        if row is None:
            return None
        n = int(row[0])
        return CachedProbs(n, mask_from_bytes(row[1], n), decode_probs(row[2]))

    def put_probs(self, rows: Iterable[tuple[bytes, str, str, int, bytes | None, bytes]]) -> None:
        """rows: (hash, model, sampling, n_frames, mask_bytes, encoded_data)."""
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany("INSERT OR REPLACE INTO probs VALUES (?, ?, ?, ?, ?, ?, ?)",
                             [(*r, now) for r in rows])

    def cache_stats(self) -> list[tuple[str, int, int]]:
        cur = self._conn().execute(
            "SELECT model, COUNT(*), SUM(LENGTH(data)) FROM probs GROUP BY model")
        return [(str(r[0]), int(r[1]), int(r[2] or 0)) for r in cur]

    # -- write provenance ------------------------------------------------------------------
    def get_managed(self, ids: Iterable[str]) -> dict[str, tuple[list[str], str]]:
        out: dict[str, tuple[list[str], str]] = {}
        conn = self._conn()
        batch = list(ids)
        for i in range(0, len(batch), 900):
            chunk = batch[i:i + 900]
            marks = ",".join("?" * len(chunk))
            q = f"SELECT item_id, tags, tagset FROM managed WHERE item_id IN ({marks})"
            for r in conn.execute(q, chunk):
                out[str(r[0])] = (json.loads(r[1]), str(r[2]))
        return out

    def all_tagsets(self) -> dict[str, str]:
        cur = self._conn().execute("SELECT item_id, tagset FROM managed")
        return {str(r[0]): str(r[1]) for r in cur}

    def record_writes(self, run_id: int, rows: Sequence[WriteRecord]) -> None:
        if not rows:
            return
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO managed VALUES (?, ?, ?, ?)",
                [(r.item_id, json.dumps(r.managed, ensure_ascii=False), r.tagset, now)
                 for r in rows])
            removed = [(r.item_id, run_id, json.dumps(r.removed, ensure_ascii=False), now)
                       for r in rows if r.removed]
            if removed:
                conn.executemany("INSERT OR REPLACE INTO backup VALUES (?, ?, ?, ?)", removed)

    def backup_runs(self) -> list[tuple[int, int, float]]:
        """(run_id, items, time) for every run that removed tags, newest first."""
        cur = self._conn().execute(
            "SELECT run_id, COUNT(*), MAX(ts) FROM backup GROUP BY run_id ORDER BY run_id DESC")
        return [(int(r[0]), int(r[1]), float(r[2])) for r in cur]

    def backup_rows(self, run_id: int) -> list[tuple[str, list[str]]]:
        cur = self._conn().execute("SELECT item_id, removed FROM backup WHERE run_id=?", (run_id,))
        return [(str(r[0]), [str(t) for t in json.loads(r[1])]) for r in cur]

    def record_wipe(self, run_id: int, rows: Sequence[tuple[str, list[str]]],
                    item_ids: Sequence[str]) -> None:
        """Back up wiped tags; the items lose their provenance record."""
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany("INSERT OR REPLACE INTO backup VALUES (?, ?, ?, ?)",
                             [(iid, run_id, json.dumps(tags, ensure_ascii=False), now)
                              for iid, tags in rows if tags])
            conn.executemany("DELETE FROM managed WHERE item_id=?", [(i,) for i in item_ids])

    # -- runs --------------------------------------------------------------------------------
    def start_run(self, config: dict[str, Any]) -> int:
        conn = self._conn()
        with conn:
            cur = conn.execute("INSERT INTO runs (started, status, config) "
                               "VALUES (?, 'running', ?)",
                               (time.time(), json.dumps(config, default=str)))
        return int(cur.lastrowid or 0)

    def last_tagging_run(self) -> dict[str, Any] | None:
        """The newest finished tagging run (not wipe commands, not killed runs), if any."""
        row = self._conn().execute(
            "SELECT run_id, started, finished, status, stats FROM runs "
            "WHERE finished IS NOT NULL AND config NOT LIKE '{\"command\"%' "
            "ORDER BY run_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return {"run_id": int(row[0]), "started": float(row[1]),
                "finished": None if row[2] is None else float(row[2]), "status": row[3],
                "stats": json.loads(row[4]) if row[4] else {}}

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any]) -> None:
        conn = self._conn()
        with conn:
            conn.execute("UPDATE runs SET finished=?, status=?, stats=? WHERE run_id=?",
                         (time.time(), status, json.dumps(stats, default=str), run_id))


@dataclass(frozen=True)
class WriteRecord:
    item_id: str
    managed: list[str]  # tags this tool is responsible for on the item after the write
    tagset: str  # fingerprint of the computed tag list that was applied
    removed: list[str]  # tags removed by this write (backed up)
