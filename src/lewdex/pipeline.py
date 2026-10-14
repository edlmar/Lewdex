"""The tagging engine.

    Eagle v2 listing (scope) -> plan (file stats + content-hash cache lookups)
      -> cached items ------------------------------------------------------------+
      -> decode pool (read + xxh3 + pyvips/PyAV + per-model preprocessing)          |
         -> frame queue -> GPU thread (gate -> PixAI -> Hydra, bounded batches)    |
            -> post thread (assemble items, cache vectors, merge tags) <-----------+
               -> async Eagle writer (fresh read -> union -> update, rate-limited)

Queues are bounded (or drained by a faster stage), so the GPU never waits on I/O while
decoding keeps up (24 CPU threads vs ~10 GPU frames/s). Pause stops feeding and the GPU;
stop lets the in-flight GPU step finish, stores and writes what is complete, then exits.
Anything unfinished is picked up by the next run: the cache is the resume state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import sys
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from . import runtime
from .config import Config
from .eagle import EagleClient, Item, Scope
from .errors import TaggerError, VideoBackendError
from .log import log
from .media import FileData, decode_file, media_kind, read_and_hash, sampling_signature
from .models import build_gate, build_taggers
from .models.base import Adapter
from .stats import Counters, Progress, Rate, StageTimes, summary_lines
from .store import Store, WriteRecord, decode_probs, encode_probs, mask_to_bytes
from .tags import (
    AliasMap,
    Blacklist,
    Candidate,
    ModelDecoder,
    aggregate,
    load_alias_pairs,
    merge,
    normalize,
    plan_final_tags,
    tagset_fingerprint,
)

T = TypeVar("T")
ProbRow = tuple[bytes, str, str, int, bytes | None, bytes]


@dataclass
class RunOptions:
    scope: Scope = field(default_factory=Scope)
    models: tuple[str, ...] | None = None  # None: as configured
    device: str | None = None
    video: bool | None = None
    dry_run: bool = False
    limit: int = 0
    item_ids: tuple[str, ...] = ()
    library_path: Path | None = None  # offline dry run without Eagle
    write_mode: str | None = None
    ignore_cache: bool = False  # benchmarks: re-infer even when cached


class Control:
    """Pause/stop shared by every stage; waits poll so both take effect within ~0.2 s."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._running = threading.Event()
        self._running.set()

    def request_stop(self) -> None:
        self.stop_event.set()
        self._running.set()

    def pause(self, on: bool) -> None:
        if on:
            self._running.clear()
        else:
            self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def stopping(self) -> bool:
        return self.stop_event.is_set()

    def wait(self) -> bool:
        """Block while paused; False once stop was requested."""
        while not self.stop_event.is_set():
            if self._running.wait(0.2):
                return not self.stop_event.is_set()
        return False


@dataclass(eq=False)
class Work:
    idx: int
    item: Item
    path: Path
    kind: str  # "image" | "video"
    content_hash: bytes = b""
    media: str = "img"
    sampling: str = "img"
    n_frames: int = 0
    force: bool = False  # re-run the models even if something is cached
    run_models: tuple[str, ...] = ()  # taggers run on the GPU for this item
    gate_known: np.ndarray | None = None  # cached P(real) per frame
    results: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    new_gate: dict[int, float] = field(default_factory=dict)
    pending: int = 0


@dataclass
class FrameJob:
    work: Work
    frame: int
    inputs: dict[str, Any]


@dataclass
class WriteJob:
    item_id: str
    tags: list[str]
    fingerprint: str


_DONE = object()


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while a run is active (released when ``on`` is False)."""
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


async def wipe_tags(client: EagleClient, store: Store, ids: Sequence[str], run_id: int, *,
                    dry: bool, report: Callable[[int, int, int], None] | None = None,
                    stop: Callable[[], bool] | None = None) -> tuple[int, int]:
    """Remove every tag from ``ids``; returns (items changed, tags removed).

    Removed tags go to the ``backup`` table under ``run_id``. The items also lose their
    ``managed`` record, so the next write re-sends the tool's tags instead of skipping them.
    """
    changed = removed = 0
    for i in range(0, len(ids), 500):
        if stop is not None and stop():
            break
        chunk = list(ids[i:i + 500])
        current = await client.get_tags(chunk)
        rows: list[tuple[str, list[str]]] = []
        failed: set[str] = set()

        async def one(iid: str, cur: list[str], rows: list[tuple[str, list[str]]] = rows,
                      failed: set[str] = failed) -> None:
            if not dry:
                try:
                    await client.set_tags(iid, [])
                except TaggerError as exc:
                    log.warning("Removing the tags of %s failed: %s", iid, exc,
                                extra={"item": iid})
                    failed.add(iid)
                    return
            rows.append((iid, cur))

        await asyncio.gather(*(one(iid, cur) for iid, cur in current.items() if cur))
        if not dry:
            store.record_wipe(run_id, rows, [iid for iid in current if iid not in failed])
        changed += len(rows)
        removed += sum(len(tags) for _, tags in rows)
        if report is not None:
            report(i + len(chunk), changed, removed)
    return changed, removed


def offline_items(library: Path) -> list[Item]:
    """Items from ``<library>/images/*.info/metadata.json`` (dry runs without Eagle)."""
    images = library / "images"
    if not images.is_dir():
        raise TaggerError(f"{library} does not look like an Eagle library (no images folder)")
    items: list[Item] = []
    for entry in images.iterdir():
        meta = entry / "metadata.json"
        if not entry.name.endswith(".info") or not meta.is_file():
            continue
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("isDeleted"):
            continue
        items.append(Item(str(d.get("id") or entry.name[:-5]), str(d.get("name", "")),
                          str(d.get("ext", "")), int(d.get("size") or 0),
                          tuple(str(t) for t in d.get("tags") or ())))
    return items


def decide_tags(cfg: Config, decoders: dict[str, ModelDecoder], blacklist: Blacklist,
                probs: dict[str, tuple[np.ndarray, np.ndarray]], gate_scores: np.ndarray | None,
                gated: set[str]) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """Per-model frame vectors -> the item's tag list (and each model's own candidates).

    ``probs`` maps model key -> (frame mask, rows for the masked frames). Frames the photo gate
    classifies as photos are ignored for gated models; frames are then aggregated, thresholded
    and merged.
    """
    per_model: dict[str, list[Candidate]] = {}
    for key, dec in decoders.items():
        if key not in probs:
            continue
        mask, rows = probs[key]
        if rows.shape[0] == 0:
            continue
        if key in gated and gate_scores is not None and gate_scores.shape[0] == mask.shape[0]:
            rows = rows[gate_scores[np.flatnonzero(mask)] < cfg.gate.threshold]
            if rows.shape[0] == 0:
                continue
        per_model[key] = dec.decode(aggregate(rows, cfg.video.aggregate, cfg.video.pmean_power))
    merged = merge(list(per_model.values()), max_tags=cfg.merge.max_tags,
                   prefixes=cfg.merge.category_prefixes, blacklist=blacklist)
    return merged, per_model


class Engine:
    PROGRESS_INTERVAL = 0.5

    def __init__(self, cfg: Config, opts: RunOptions | None = None,
                 on_progress: Callable[[Progress], None] | None = None) -> None:
        self.opts = opts or RunOptions()
        self.cfg = self._effective_config(cfg, self.opts)
        self.control = Control()
        self.on_progress = on_progress or (lambda _p: None)
        self.counters = Counters()
        self.stages = StageTimes()
        self.item_rate = Rate(90.0)
        self.frame_rate = Rate(90.0)
        self.infer_rate = Rate(120.0)  # items that went through the GPU
        self.planned_decode = 0
        self.failed_writes: list[str] = []  # item ids whose Eagle update failed
        self.phase = "idle"
        self.message = ""
        self.summary: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._t0 = 0.0
        self._gpu_t0 = 0.0
        self._gpu_t1 = 0.0
        self._decoding = 0
        self._gpu_thread: threading.Thread | None = None
        self.client: EagleClient | None = None
        self.write_q: asyncio.Queue[Any] | None = None
        self.library: Path | None = None

    # -- public control ------------------------------------------------------------------------
    def pause(self, on: bool) -> None:
        self.control.pause(on)
        log.info("Paused" if on else "Resumed")
        self.on_progress(self.progress())

    def stop(self) -> None:
        if not self.control.stopping:
            log.warning("Stop requested: finishing the current GPU step and pending writes")
        self.control.request_stop()

    def progress(self) -> Progress:
        c = self.counters
        with self._lock:
            elapsed = time.monotonic() - self._t0 if self._t0 else 0.0
            ips, fps = self.item_rate.per_second(), self.frame_rate.per_second()
            remaining = max(0, c.total - c.done)
            gpu_left = max(0, self.planned_decode - c.inferred - c.errors)
            irate = self.infer_rate.per_second()
            if gpu_left and irate > 0:  # the GPU-bound part dominates the time left
                ips, eta = irate, gpu_left / irate
            else:
                eta = remaining / ips if ips > 0 else 0.0
            if self.phase != "tagging":
                eta = 0.0
            return Progress(self.phase, self.message, c.total, c.done, c.inferred, c.cached,
                            c.skipped, c.errors, c.frames, c.written, c.unchanged, c.write_errors,
                            ips, fps, eta, elapsed, self.control.paused)

    def _set_phase(self, phase: str, message: str = "") -> None:
        with self._lock:
            self.phase, self.message = phase, message
        if message:
            log.info(message)
        self.on_progress(self.progress())

    def _count(self, **deltas: int) -> None:
        with self._lock:
            for k, v in deltas.items():
                setattr(self.counters, k, getattr(self.counters, k) + v)

    @staticmethod
    def _effective_config(cfg: Config, o: RunOptions) -> Config:
        c = cfg.copy()
        if o.models is not None:
            for k, m in c.models.items():
                m.enabled = k in o.models
        if o.device is not None:
            c.runtime.device = o.device
        if o.video is not None:
            c.video.enabled = o.video
        if o.write_mode is not None:
            c.write.mode = o.write_mode  # type: ignore[assignment]
        c.validate()
        return c

    # -- run -------------------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Blocking; returns the summary. User-facing failures raise TaggerError."""
        self._t0 = time.monotonic()
        store: Store | None = None
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, name="eagle-io", daemon=True)
        loop_thread.start()
        self.loop = loop
        keep_awake(True)
        try:
            runtime.configure(self.cfg.cache_dir)
            store = Store(self.cfg.db_path)
            self.store = store
            self._run(store)
            return self.summary
        except TaggerError as exc:
            self._set_phase("error", str(exc))
            raise
        except Exception as exc:
            log.exception("Run failed")
            self._set_phase("error", f"Unexpected error: {exc}")
            raise
        finally:
            if self.client is not None:
                with contextlib.suppress(Exception):  # shutdown: never mask the real error
                    self._call(self.client.__aexit__(None, None, None), 10)
            keep_awake(False)
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(5)
            if store is not None:
                store.close()
            runtime.free_cuda()

    def _call(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def _run(self, store: Store) -> None:
        cfg, opts = self.cfg, self.opts
        dry = opts.dry_run
        self.run_id = store.start_run({"options": asdict(opts), "config": asdict(cfg)})

        # 1. Items: from Eagle, or offline from the library folder
        if opts.library_path is not None:
            if not dry:
                raise TaggerError("--library-path only works together with --dry-run")
            library = opts.library_path
            self._set_phase("listing", f"Reading the library offline from {library}")
            items = offline_items(library)
        else:
            client = EagleClient(cfg.eagle)
            self._call(client.__aenter__())
            self.client = client
            self._set_phase("listing", "Connecting to Eagle")
            library, name, version = self._call(client.library_info(), 30)
            log.info("Eagle %s, library %r at %s", version, name, library)
            self._set_phase("listing", f"Listing items ({opts.scope.describe()})")
            items = self._call(client.list_items(opts.scope), 600)
        if opts.item_ids:
            wanted = set(opts.item_ids)
            items = [it for it in items if it.id in wanted]
        if opts.limit:
            items = items[: opts.limit]
        self.library = library
        self._set_phase("listing", f"{len(items):,} items in scope")

        # 2. Models (downloaded on first use) and decision rules
        if cfg.video.enabled:
            try:
                import av  # noqa: F401
            except Exception as exc:
                log.error("%s", VideoBackendError(str(exc)))
                cfg.video.enabled = False
        self._set_phase("loading", "Checking model files")
        self.taggers: dict[str, Adapter] = build_taggers(cfg)
        self.gate = build_gate(cfg)
        for a in [*self.taggers.values(), *([self.gate] if self.gate else [])]:
            a.fetch()
        aliases = AliasMap((load_alias_pairs() if cfg.merge.builtin_aliases else [])
                           + list(cfg.merge.aliases.items()))
        self.blacklist = Blacklist(cfg.merge.blacklist)
        self.decoders = {
            key: ModelDecoder(key, a.label_space(), categories=cfg.models[key].categories,
                              implications=cfg.models[key].implications, aliases=aliases,
                              blacklist=self.blacklist)
            for key, a in self.taggers.items()}
        log.info("Models: %s; photo gate %s; %d aliases; video %s",
                 ", ".join(a.title for a in self.taggers.values()),
                 "on" if self.gate else "off", len(aliases), "on" if cfg.video.enabled else "off")

        # 3. Plan
        self._set_phase("planning", "Checking which items need the GPU")
        self.index = {k: store.cached_index(a.cache_key) for k, a in self.taggers.items()}
        works_cached, works_decode, skipped = self._plan(store, items)
        self.planned_decode = len(works_decode)
        self._count(total=len(items), skipped=len(skipped), done=len(skipped))
        log.info("Plan: %d from cache, %d need decoding/inference, %d skipped "
                 "(unsupported type or video off)", len(works_cached), len(works_decode),
                 len(skipped))

        # 4. Stages
        self.device = runtime.select_device(cfg.runtime.device)
        self.frame_q: queue.Queue[FrameJob] = queue.Queue(maxsize=cfg.runtime.frames_per_step * 3)
        self.post_q: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.finished = threading.Event()
        self.post_exit = threading.Event()
        self.remaining = len(works_cached) + len(works_decode)
        writer_future = None
        if not dry and self.client is not None:
            self.write_q = self._call(self._make_queue())
            writer_future = asyncio.run_coroutine_threadsafe(self._writer(store), self.loop)
        n_workers = cfg.runtime.decode_workers or 16
        self.pool = ThreadPoolExecutor(n_workers, thread_name_prefix="decode")
        self._slots = threading.Semaphore(n_workers * 2)
        post = threading.Thread(target=self._post_main, args=(store,), name="post", daemon=True)
        beat = threading.Thread(target=self._heartbeat, name="progress", daemon=True)
        post.start()
        beat.start()
        if works_decode:
            self._ensure_gpu()
        self._set_phase("tagging", "Tagging")
        for w in works_cached:
            self.post_q.put(("cached", w))
        feeder = threading.Thread(target=self._feed, args=(works_decode,), name="feeder",
                                  daemon=True)
        feeder.start()
        if self.remaining == 0:
            self.finished.set()
        while not self.finished.wait(0.25):
            if self._fatal is not None or self.control.stopping:
                break
        # Orderly drain: feeder -> decoders -> GPU -> post-processing -> writer.
        if not self.finished.is_set():
            self.control.request_stop()
        feeder.join(30)
        deadline = time.monotonic() + 60
        while self._decoding and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._gpu_thread is not None:
            self._gpu_thread.join(120)
        self.post_exit.set()
        post.join(60)
        self.pool.shutdown(wait=False, cancel_futures=True)
        if writer_future is not None and self.write_q is not None:
            self._call(self.write_q.put(_DONE))
            try:
                writer_future.result(900)
            except Exception as exc:
                log.error("Eagle writer ended with an error: %s", exc)
        self.finished.set()
        beat.join(5)
        if self._fatal is not None:
            raise self._fatal
        c = self.counters
        status = "stopped" if c.done < c.total else "done"
        self._finish(store, status)

    # -- planning --------------------------------------------------------------------------------
    def _plan(self, store: Store, items: list[Item]) -> tuple[list[Work], list[Work], list[Item]]:
        cfg = self.cfg
        library = self.library
        assert library is not None
        known = store.all_files()
        works: list[Work] = []
        skipped: list[Item] = []
        for idx, it in enumerate(items):
            kind = media_kind(it.ext)
            if kind is None or (kind == "video" and not cfg.video.enabled):
                skipped.append(it)
            else:
                works.append(Work(idx, it, it.path(library), kind))

        def stat(w: Work) -> tuple[int, int] | None:
            try:
                st = w.path.stat()
                return st.st_size, st.st_mtime_ns
            except OSError:
                return None

        with ThreadPoolExecutor(16, thread_name_prefix="stat") as pool:
            stats = list(pool.map(stat, works, chunksize=256))
        cached: list[Work] = []
        decode: list[Work] = []
        for w, st in zip(works, stats, strict=True):
            prev = known.get(w.item.id)
            if (st is not None and prev is not None and (prev[0], prev[1]) == st
                    and not self.opts.ignore_cache):
                w.content_hash, w.media = prev[2], prev[3]
                w.sampling = self._sampling(w.media)
                if self._all_cached(w):
                    cached.append(w)
                    continue
            decode.append(w)
        return cached, decode, skipped

    def _sampling(self, media: str) -> str:
        if media == "anim" and not self.cfg.video.enabled:
            media = "img"
        return sampling_signature(media, self.cfg.video)

    def _all_cached(self, w: Work) -> bool:
        return all((w.content_hash, w.sampling) in self.index[k] for k in self.taggers)

    # -- feeding and decoding (worker threads) ------------------------------------------------
    def _feed(self, works: list[Work]) -> None:
        try:
            for w in works:
                if not self.control.wait():
                    break
                while not self._slots.acquire(timeout=0.2):
                    if self.control.stopping:
                        return
                self._submit(w)
        except Exception as exc:
            log.exception("Feeder failed")
            self._fatal = exc
            self.control.request_stop()

    def _submit(self, w: Work, slot: bool = True) -> None:
        with self._lock:
            self._decoding += 1

        def done(_f: Any) -> None:
            with self._lock:
                self._decoding -= 1
            if slot:
                self._slots.release()

        self.pool.submit(self._decode_one, w).add_done_callback(done)

    def _decode_one(self, w: Work) -> None:
        if not self.control.wait():
            return
        cfg = self.cfg
        try:
            t0 = time.perf_counter()
            fd: FileData = read_and_hash(w.path)
            w.content_hash = fd.content_hash
            self.stages.add("read_hash", time.perf_counter() - t0)
            t1 = time.perf_counter()
            decoded = decode_file(w.path, fd, w.kind, cfg.video)
            w.media = decoded.media
            w.sampling = self._sampling(w.media)
            self.post_q.put(("file", (w.item.id, fd.size, fd.mtime_ns, fd.content_hash, w.media)))
            frames = decoded.frames
            if not frames:
                raise ValueError("no decodable frames")
            w.n_frames = len(frames)
            if not w.force and not self.opts.ignore_cache and self._all_cached(w):
                self.stages.add("decode", time.perf_counter() - t1)
                self.post_q.put(("cached", w))
                return
            missing = tuple(k for k in self.taggers
                            if w.force or self.opts.ignore_cache
                            or (w.content_hash, w.sampling) not in self.index[k])
            w.run_models = missing
            gated = {k for k in missing if cfg.models[k].gated} if self.gate is not None else set()
            if gated and self.gate is not None and not (w.force or self.opts.ignore_cache):
                g = self.store.get_probs(w.content_hash, self.gate.cache_key, w.sampling)
                if g is not None and g.probs.shape == (1, w.n_frames):
                    w.gate_known = g.probs[0]
            jobs: list[FrameJob] = []
            for fi, frame in enumerate(frames):
                photo = w.gate_known is not None and w.gate_known[fi] >= cfg.gate.threshold
                inputs: dict[str, Any] = {k: self.taggers[k].prepare(frame)
                                          for k in missing if not (k in gated and photo)}
                if gated and w.gate_known is None and self.gate is not None:
                    inputs["gate"] = self.gate.prepare(frame)
                jobs.append(FrameJob(w, fi, inputs))
            self.stages.add("decode", time.perf_counter() - t1, len(frames))
            self._count(dedup_dropped=decoded.dropped)
            w.pending = len(jobs)
            for job in jobs:
                while True:
                    try:
                        self.frame_q.put(job, timeout=0.2)
                        break
                    except queue.Full:
                        if self.control.stopping:
                            return
        except TaggerError as exc:
            self.post_q.put(("failed", (w, str(exc))))
        except Exception as exc:
            self.post_q.put(("failed", (w, f"{type(exc).__name__}: {exc}")))

    # -- GPU ---------------------------------------------------------------------------------------
    def _ensure_gpu(self) -> None:
        with self._lock:
            if self._gpu_thread is None:
                self._gpu_thread = threading.Thread(target=self._gpu_main, name="gpu", daemon=True)
                self._gpu_thread.start()

    def _gpu_main(self) -> None:
        cfg = self.cfg
        adapters: dict[str, Adapter] = dict(self.taggers)
        if self.gate is not None:
            adapters["gate"] = self.gate
        try:
            self._set_phase("loading", f"Loading models on {self.device.name}")
            for a in adapters.values():
                a.load(self.device.device)
            self.batch = {k: self._initial_batch(k, a) for k, a in adapters.items()}
            self._warmup(adapters)  # on this thread: cuDNN/compile state is per thread
            self._set_phase("tagging", "Tagging")
            self._gpu_t0 = time.monotonic()
            per_step = cfg.runtime.frames_per_step
            while not self.finished.is_set():
                if not self.control.wait():
                    break
                t0 = time.perf_counter()
                try:
                    jobs = [self.frame_q.get(timeout=0.2)]
                except queue.Empty:
                    self.stages.add("gpu_starved", time.perf_counter() - t0, 0)
                    continue
                while len(jobs) < per_step:
                    try:
                        jobs.append(self.frame_q.get(timeout=0.01))
                    except queue.Empty:
                        break
                self.stages.add("gpu_starved", time.perf_counter() - t0, 0)
                self._gpu_step(jobs, adapters)
                self._gpu_t1 = time.monotonic()
        except Exception as exc:
            log.exception("GPU worker failed")
            self._fatal = exc
            self.control.request_stop()
        finally:
            for a in adapters.values():
                a.unload()
            runtime.free_cuda()

    def _batch_key(self, a: Adapter) -> str:
        return f"batch:{self.device.name}:{a.cache_key}:{int(a.compile)}"

    def _initial_batch(self, key: str, a: Adapter) -> int:
        mc = self.cfg.models.get(key)
        if mc is not None and mc.batch_size:
            return mc.batch_size
        saved = self.store.get_meta(self._batch_key(a))
        return int(saved) if saved else a.default_batch

    def _warmup(self, adapters: dict[str, Adapter]) -> None:
        """First call per model: compiles (if enabled), measures VRAM, keeps the batch in budget."""
        import torch
        for key, a in adapters.items():
            if a.compile:
                self._set_phase("loading", f"Compiling {a.title} (first run takes a few minutes, "
                                           "then it is cached)")
            while True:
                bs = self.batch[key]
                try:
                    t0 = time.perf_counter()
                    base = 0
                    if self.device.cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    a.forward(a.example_input(bs))
                    if self.device.cuda:
                        torch.cuda.synchronize()
                        per_frame = max(1, (torch.cuda.max_memory_allocated() - base) // bs)
                        free, total = torch.cuda.mem_get_info()
                        budget = total * self.cfg.runtime.vram_budget - (total - free)
                        cap = int(budget // per_frame)
                        if 0 < cap < bs:
                            new = 1 << (cap.bit_length() - 1)
                            log.info("%s: batch %d -> %d to stay inside the VRAM budget",
                                     a.title, bs, new)
                            self.batch[key] = new
                            continue
                    log.info("%s ready: batch %d (%.1fs)", a.title, bs, time.perf_counter() - t0)
                    break
                except Exception as exc:
                    if runtime.is_oom(exc) and bs > 1:
                        runtime.free_cuda()
                        self._shrink(key, a)
                        continue
                    if a.compile:  # Inductor/Triton unusable here: slower, but still correct
                        log.warning("torch.compile failed for %s (%s); running it without "
                                    "compilation", a.title, str(exc).splitlines()[0][:200])
                        a.use_eager()
                        runtime.free_cuda()
                        continue
                    raise

    def _shrink(self, key: str, a: Adapter) -> None:
        new = max(1, self.batch[key] // 2)
        log.warning("CUDA out of memory in %s: batch %d -> %d (remembered for this GPU)",
                    a.title, self.batch[key], new)
        self.batch[key] = new
        self.store.put_meta(self._batch_key(a), str(new))

    def _run_model(self, key: str, a: Adapter, inputs: list[Any]) -> np.ndarray:
        outs: list[np.ndarray] = []
        i = 0
        while i < len(inputs):
            bs = self.batch[key]
            chunk = inputs[i:i + bs]
            try:
                batch = a.collate(chunk, pad_to=bs if a.compile else None)
                t0 = time.perf_counter()
                probs = a.forward(batch)[: len(chunk)].cpu().numpy()
                self.stages.add(f"gpu_{key}", time.perf_counter() - t0, len(chunk))
            except Exception as exc:
                if runtime.is_oom(exc) and bs > 1:
                    runtime.free_cuda()
                    self._shrink(key, a)
                    continue
                raise
            outs.append(probs)
            i += len(chunk)
        return np.concatenate(outs) if len(outs) > 1 else outs[0]

    def _gpu_step(self, jobs: list[FrameJob], adapters: dict[str, Adapter]) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        scores: dict[int, float] = {}
        if "gate" in adapters:
            sel = [i for i, j in enumerate(jobs) if "gate" in j.inputs]
            if sel:
                g = self._run_model("gate", adapters["gate"], [jobs[i].inputs["gate"] for i in sel])
                scores = {i: float(s) for i, s in zip(sel, g.reshape(-1), strict=True)}
        outputs: list[dict[str, np.ndarray]] = [{} for _ in jobs]
        photos = 0
        for key, a in self.taggers.items():
            sel = []
            for i, j in enumerate(jobs):
                if key not in j.inputs:
                    continue
                s = scores.get(i)
                if cfg.models[key].gated and s is not None and s >= cfg.gate.threshold:
                    photos += 1
                    continue
                sel.append(i)
            if sel:
                p = self._run_model(key, a, [jobs[i].inputs[key] for i in sel])
                for i, row in zip(sel, p, strict=True):
                    outputs[i][key] = row
        self.stages.add("gpu_step", time.perf_counter() - t0, len(jobs))
        self._count(frames=len(jobs), photo_frames=photos)
        self.frame_rate.add(len(jobs))
        for i, j in enumerate(jobs):
            self.post_q.put(("frame", (j, outputs[i], scores.get(i))))

    # -- post-processing (one thread) -------------------------------------------------------------
    def _post_main(self, store: Store) -> None:
        prob_rows: list[ProbRow] = []
        file_rows: list[tuple[str, int, int, bytes, str]] = []
        last_flush = time.monotonic()

        def flush() -> None:
            nonlocal last_flush
            if prob_rows:
                with self.stages.time("db_write", len(prob_rows)):
                    store.put_probs(prob_rows)
                prob_rows.clear()
            if file_rows:
                store.put_files(file_rows)
                file_rows.clear()
            last_flush = time.monotonic()

        try:
            while True:
                try:
                    kind, payload = self.post_q.get(timeout=0.25)
                except queue.Empty:
                    kind, payload = "", None
                if kind == "file":
                    file_rows.append(payload)
                elif kind == "frame":
                    job, probs, gscore = payload
                    w = job.work
                    for key, row in probs.items():
                        w.results.setdefault(key, {})[job.frame] = row
                    if gscore is not None:
                        w.new_gate[job.frame] = gscore
                    w.pending -= 1
                    if w.pending == 0:
                        self._finalize(w, prob_rows)
                elif kind == "cached":
                    self._finalize(payload, prob_rows)
                elif kind == "failed":
                    w, err = payload
                    log.warning("Skipped %s (%s): %s", w.item.id, w.path.name, err,
                                extra={"item": w.item.id})
                    self._count(errors=1, done=1)
                    self.item_rate.add()
                    self._emit_tags(w, [], failed=True)
                    self._item_done()
                if len(prob_rows) >= 64 or (kind == "" and (prob_rows or file_rows)) \
                        or time.monotonic() - last_flush > 2.0:
                    flush()
                if self.finished.is_set() or (self.post_exit.is_set() and self.post_q.empty()):
                    break
        except Exception as exc:
            log.exception("Post-processing failed")
            self._fatal = exc
            self.control.request_stop()
        finally:
            flush()

    def _item_done(self) -> None:
        with self._lock:
            self.remaining -= 1
            done = self.remaining <= 0
        if done:
            self.finished.set()

    def _finalize(self, w: Work, prob_rows: list[ProbRow]) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        fresh = bool(w.run_models)
        probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # key -> (frame mask, rows)
        gate_scores = w.gate_known
        if fresh:
            n = w.n_frames
            if w.new_gate and self.gate is not None:
                raw = np.array([w.new_gate.get(i, 0.0) for i in range(n)], dtype=np.float32)
                blob = encode_probs(raw[None, :])
                prob_rows.append((w.content_hash, self.gate.cache_key, w.sampling, n, None, blob))
                gate_scores = decode_probs(blob)[0]
            for key in w.run_models:
                frames = w.results.get(key, {})
                mask = np.array([i in frames for i in range(n)], dtype=bool)
                width = self.decoders[key].space.thresholds.shape[0]
                rows = (np.stack([frames[i] for i in range(n) if i in frames]) if frames
                        else np.zeros((0, width), dtype=np.float32))
                blob = encode_probs(rows)
                prob_rows.append((w.content_hash, self.taggers[key].cache_key, w.sampling, n,
                                  mask_to_bytes(mask.tolist()), blob))
                # Merge from the stored (quantised) values so re-runs decide identically.
                probs[key] = (mask, decode_probs(blob))
                self.index[key][(w.content_hash, w.sampling)] = (n, None)
        rerun: list[str] = []
        for key, a in self.taggers.items():
            if key in probs:
                continue
            c = self.store.get_probs(w.content_hash, a.cache_key, w.sampling)
            if c is None:
                rerun.append(key)
                continue
            probs[key] = (c.mask, c.probs)
            w.n_frames = c.n_frames
        gated_keys = [k for k in probs if cfg.models[k].gated] if self.gate is not None else []
        if gated_keys and gate_scores is None and self.gate is not None:
            g = self.store.get_probs(w.content_hash, self.gate.cache_key, w.sampling)
            gate_scores = g.probs[0] if g is not None else None
        for key in gated_keys:  # a gated model must cover every frame the gate calls "not a photo"
            mask = probs[key][0]
            if not mask.all() and (gate_scores is None or gate_scores.shape[0] != mask.shape[0]
                                   or bool(np.any(~mask & (gate_scores < cfg.gate.threshold)))):
                rerun.append(key)
        if rerun and not fresh:
            log.debug("Cache incomplete for %s under current settings (%s); re-running",
                      w.item.id, ", ".join(rerun))
            w.force, w.results, w.new_gate, w.gate_known = True, {}, {}, None
            self._ensure_gpu()
            self._submit(w, slot=False)
            return
        merged, _ = decide_tags(cfg, self.decoders, self.blacklist, probs, gate_scores,
                                set(gated_keys))
        self.stages.add("merge", time.perf_counter() - t0)
        self._count(done=1, inferred=int(fresh), cached=int(not fresh))
        self.item_rate.add()
        if fresh:
            self.infer_rate.add()
        self._emit_tags(w, [c.name for c in merged])
        self._item_done()

    def _emit_tags(self, w: Work, tags: list[str], failed: bool = False) -> None:
        cfg = self.cfg
        if self.write_q is None:  # dry run: count what would change, write nothing
            current = list(w.item.tags)
            final, added, removed = plan_final_tags(
                current, tags, managed=[], mode=cfg.write.mode)
            if final != current:
                self._count(written=1, tags_added=len(added), tags_removed=len(removed))
            else:
                self._count(unchanged=1)
            return
        if failed:
            return  # never write a partial result
        fingerprint = tagset_fingerprint(tags, cfg.write.mode)
        asyncio.run_coroutine_threadsafe(
            self.write_q.put(WriteJob(w.item.id, tags, fingerprint)), self.loop)

    # -- Eagle writer (event-loop thread) ----------------------------------------------------------
    async def _make_queue(self) -> asyncio.Queue[Any]:
        return asyncio.Queue()

    async def _writer(self, store: Store) -> None:
        assert self.write_q is not None
        tagsets = store.all_tagsets()
        buf: list[WriteJob] = []
        first_at = 0.0
        while True:
            try:
                job = await asyncio.wait_for(self.write_q.get(), 0.5)
            except TimeoutError:
                job = None
            if job is _DONE:
                if buf:
                    await self._flush_writes(store, buf, tagsets)
                return
            if job is not None:
                if (self.cfg.write.mode != "replace"
                        and tagsets.get(job.item_id) == job.fingerprint):
                    self._count(unchanged=1)  # same tags as last written: nothing to send
                    continue
                if not buf:
                    first_at = time.monotonic()
                buf.append(job)
            if buf and (len(buf) >= 64 or time.monotonic() - first_at > 1.0):
                await self._flush_writes(store, buf, tagsets)
                buf = []

    async def _flush_writes(self, store: Store, jobs: list[WriteJob],
                            tagsets: dict[str, str]) -> None:
        client = self.client
        assert client is not None
        cfg = self.cfg
        t0 = time.perf_counter()
        try:
            current = await client.get_tags([j.item_id for j in jobs])
        except TaggerError as exc:
            log.error("Could not read current tags from Eagle (%s); %d writes will be redone "
                      "by the next run", exc, len(jobs))
            self._count(write_errors=len(jobs))
            self.failed_writes += [j.item_id for j in jobs]
            return
        managed = store.get_managed([j.item_id for j in jobs])
        records: list[WriteRecord] = []

        async def one(j: WriteJob) -> None:
            cur = current.get(j.item_id)
            if cur is None:
                return  # deleted in Eagle meanwhile
            prev_managed = managed.get(j.item_id, ([], ""))[0]
            final, added, removed = plan_final_tags(cur, j.tags, managed=prev_managed,
                                                    mode=cfg.write.mode)
            before = {normalize(t) for t in cur}
            owned = {normalize(t) for t in prev_managed}
            new_n = {normalize(t) for t in j.tags}
            manage = [t for t in final if normalize(t) in new_n
                      and (normalize(t) not in before or normalize(t) in owned
                           or cfg.write.mode == "replace")]
            if final != cur:
                try:
                    await client.set_tags(j.item_id, final)
                except TaggerError as exc:
                    log.warning("Eagle update failed for %s: %s", j.item_id, exc,
                                extra={"item": j.item_id})
                    self._count(write_errors=1)
                    self.failed_writes.append(j.item_id)
                    return
                self._count(written=1, tags_added=len(added), tags_removed=len(removed))
            else:
                self._count(unchanged=1)
            records.append(WriteRecord(j.item_id, manage, j.fingerprint, removed))
            tagsets[j.item_id] = j.fingerprint

        await asyncio.gather(*(one(j) for j in jobs))
        store.record_writes(self.run_id, records)
        self.stages.add("eagle_write", time.perf_counter() - t0, len(jobs))

    # -- progress & summary --------------------------------------------------------------------
    def _heartbeat(self) -> None:
        while not self.finished.wait(self.PROGRESS_INTERVAL):
            self.on_progress(self.progress())
        self.on_progress(self.progress())

    def _finish(self, store: Store, status: str) -> None:
        import psutil
        import torch
        wall = time.monotonic() - self._t0
        c = self.counters
        stages = self.stages.snapshot()
        gpu_wall = (self._gpu_t1 - self._gpu_t0) if self._gpu_t0 and self._gpu_t1 else 0.0
        gpu_busy = sum(v["total_s"] for k, v in stages.items()
                       if k.startswith("gpu_") and k not in ("gpu_step", "gpu_starved"))
        summary: dict[str, Any] = {
            "status": status,
            "dry_run": self.opts.dry_run,
            "wall_s": wall,
            "items_per_s": c.done / wall if wall else 0.0,
            "gpu_items_per_s": c.inferred / gpu_wall if gpu_wall else 0.0,
            "gpu_frames_per_s": c.frames / gpu_wall if gpu_wall else 0.0,
            "gpu_busy_pct": 100.0 * min(1.0, gpu_busy / gpu_wall) if gpu_wall else 0.0,
            "gpu_starved_s": stages.get("gpu_starved", {}).get("total_s", 0.0),
            "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3
                             if torch.cuda.is_available() else 0.0),
            "peak_rss_gb": psutil.Process().memory_info().rss / 1024**3,
            "counts": asdict(c),
            "failed_writes": self.failed_writes,
            "stages": stages,
            "batch": getattr(self, "batch", {}),
        }
        self.summary = summary
        store.finish_run(self.run_id, status, summary)
        for line in summary_lines(summary):
            log.info(line)
        what = "Dry run" if self.opts.dry_run else "Run"
        self._set_phase("done" if status == "done" else "stopped",
                        f"{what} {status}: {c.done:,} of {c.total:,} items")
