"""Logging: human-readable console, JSON-lines file, and an in-memory ring for the GUI.

Structured fields go in ``extra=`` (``log.info("wrote tags", extra={"item": iid, "n": 12})``);
the JSON file keeps them as keys, the console appends them as ``key=value``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

log = logging.getLogger("lewdex")

_STD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _STD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        extra = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        text = f"{stamp} {record.levelname[:4]:<4} {record.getMessage()}"
        if extra:
            text += f"  [{extra}]"
        if record.exc_info and record.levelno >= logging.ERROR:
            text += "\n" + self.formatException(record.exc_info)
        return text


class RingBuffer(logging.Handler):
    """Last ``maxlen`` formatted records with sequence numbers, read by the GUI timer."""

    def __init__(self, maxlen: int = 2000) -> None:
        super().__init__(level=logging.INFO)
        self._lines: deque[tuple[int, int, str]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self.setFormatter(ConsoleFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, record.levelno, line))

    def since(self, seq: int) -> list[tuple[int, int, str]]:
        with self._lock:
            return [entry for entry in self._lines if entry[0] > seq]


_configured = False


def setup_logging(log_dir: Path | None, *, console: bool = True, verbose: bool = False,
                  ring: RingBuffer | None = None) -> Path | None:
    """Configure the ``lewdex`` logger once per process; returns the JSON log path."""
    global _configured
    if _configured:
        if ring is not None:
            log.addHandler(ring)
        return None
    _configured = True
    log.setLevel(logging.DEBUG)
    log.propagate = False
    path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "lewdex.jsonl"
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=20_000_000, backupCount=5,
                                                  encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        log.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(ConsoleFormatter())
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        log.addHandler(sh)
    if ring is not None:
        log.addHandler(ring)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "PIL", "timm", "torch._dynamo",
                  "torch._inductor", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
