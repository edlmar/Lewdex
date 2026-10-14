"""Run metrics: stage timers, windowed throughput, ETA, the progress snapshot and the summary."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class StageTimes:
    """Thread-safe per-stage totals (count, seconds, max) plus a sample window for p95."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, list[float]] = {}  # name -> [count, total, max]
        self._samples: dict[str, deque[float]] = {}

    def add(self, name: str, seconds: float, count: int = 1) -> None:
        with self._lock:
            d = self._data.setdefault(name, [0.0, 0.0, 0.0])
            d[0] += count
            d[1] += seconds
            d[2] = max(d[2], seconds)
            self._samples.setdefault(name, deque(maxlen=2048)).append(seconds / max(1, count))

    @contextmanager
    def time(self, name: str, count: int = 1) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0, count)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for name, (count, total, mx) in self._data.items():
                s = sorted(self._samples.get(name, ()))
                p95 = s[int(0.95 * (len(s) - 1))] if s else 0.0
                out[name] = {"count": count, "total_s": total, "mean_ms": total / count * 1000
                             if count else 0.0, "p95_ms": p95 * 1000, "max_ms": mx * 1000}
            return out

    def total(self, name: str) -> float:
        with self._lock:
            return self._data.get(name, [0.0, 0.0, 0.0])[1]


class Rate:
    """Events per second over the last ``window`` seconds."""

    def __init__(self, window: float = 60.0) -> None:
        self.window = window
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def add(self, n: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, n))
            self._trim(now)

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window:
            self._events.popleft()

    def per_second(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if not self._events:
                return 0.0
            span = max(1.0, now - self._events[0][0])
            return sum(n for _, n in self._events) / span


@dataclass(frozen=True)
class Progress:
    phase: str = "idle"
    message: str = ""
    total: int = 0
    done: int = 0
    inferred: int = 0
    cached: int = 0
    skipped: int = 0
    errors: int = 0
    frames: int = 0
    written: int = 0
    unchanged: int = 0
    write_errors: int = 0
    items_per_s: float = 0.0
    frames_per_s: float = 0.0
    eta_s: float = 0.0
    elapsed_s: float = 0.0
    paused: bool = False


@dataclass
class Counters:
    total: int = 0
    done: int = 0
    inferred: int = 0
    cached: int = 0
    skipped: int = 0
    errors: int = 0
    frames: int = 0
    gpu_frames_pending: int = 0
    written: int = 0
    unchanged: int = 0
    write_errors: int = 0
    tags_added: int = 0
    tags_removed: int = 0
    photo_frames: int = 0
    dedup_dropped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def fmt_duration(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds > 7 * 86400:
        return "--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def summary_lines(summary: dict[str, Any]) -> list[str]:
    """Human-readable benchmark summary (the same dict is stored with the run)."""
    c = summary["counts"]
    s = summary
    if s.get("dry_run"):
        eagle = (f"  Eagle (dry run, nothing written): {c['written']:,} items would change"
                 f" (+{c['tags_added']:,} / -{c['tags_removed']:,} tags),"
                 f" {c['unchanged']:,} unchanged")
    else:
        eagle = (f"  Eagle: {c['written']:,} items updated"
                 f" (+{c['tags_added']:,} / -{c['tags_removed']:,} tags),"
                 f" {c['unchanged']:,} unchanged, {c['write_errors']:,} failed")
    lines = [
        f"Run {s['status']}: {c['done']:,}/{c['total']:,} items in {fmt_duration(s['wall_s'])}"
        f" ({s['items_per_s']:.2f} items/s overall)",
        f"  inferred {c['inferred']:,} items / {c['frames']:,} frames"
        f" -> {s['gpu_items_per_s']:.2f} items/s, {s['gpu_frames_per_s']:.2f} frames/s"
        f" while the GPU was working",
        f"  from cache {c['cached']:,}, skipped {c['skipped']:,}, errors {c['errors']:,},"
        f" photo frames (Hydra skipped) {c['photo_frames']:,},"
        f" duplicate frames dropped {c['dedup_dropped']:,}",
        eagle,
        f"  GPU busy {s['gpu_busy_pct']:.0f}% of inference time,"
        f" starved {s['gpu_starved_s']:.1f}s, peak VRAM {s['peak_vram_gb']:.1f} GB,"
        f" peak RAM {s['peak_rss_gb']:.1f} GB",
    ]
    for name, st in summary["stages"].items():
        lines.append(f"  {name:<16} n={int(st['count']):>7}  mean {st['mean_ms']:8.1f} ms"
                     f"  p95 {st['p95_ms']:8.1f} ms  total {st['total_s']:8.1f} s")
    return lines
