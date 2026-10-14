"""Command line: ``lewdex <command>``; ``lewdex --help`` lists the commands."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, load_config
from .eagle import Scope
from .errors import TaggerError
from .log import log, setup_logging
from .stats import Progress, fmt_duration


def _add_run_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to tag (always the whole Eagle library)")
    g.add_argument("--limit", type=int, default=0, help="only the first N items (testing)")
    g = p.add_argument_group("how")
    g.add_argument("--models", help="comma-separated: pixai,hydra (default: as configured)")
    g.add_argument("--device", help="auto, cuda, cuda:0 or cpu (default: as configured)")
    g.add_argument("--video", dest="video", action="store_true", default=None,
                   help="tag videos (keyframes) and animations")
    g.add_argument("--no-video", dest="video", action="store_false",
                   help="skip videos; animations use their first frame")
    g.add_argument("--dry-run", action="store_true",
                   help="compute and cache everything, write nothing to Eagle")
    g.add_argument("--write-mode", choices=("replace", "add", "sync"),
                   help="replace (default): an item's tags become exactly the new tags. add: only "
                        "add tags. sync: also remove tags this tool added earlier that it no "
                        "longer predicts")
    g.add_argument("--library-path", type=Path,
                   help="read the library folder directly (dry run without Eagle)")
    g.add_argument("--ignore-cache", action="store_true",
                   help="re-run the models even for cached items (benchmarking)")
    g.add_argument("--summary-json", type=Path, help="also write the run summary to this file")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lewdex",
                                description="GPU auto-tagger for Eagle 4 (PixAI v1.0 + Hydra 3.5)")
    p.add_argument("--config", type=Path, help="config file (default: lewdex.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"lewdex {__version__}")
    sub = p.add_subparsers(dest="command")
    _add_run_options(sub.add_parser("run", help="tag items headless"))
    gui = sub.add_parser("gui", help="open the window")
    gui.add_argument("--browser", action="store_true",
                     help="serve the window in the default browser instead of a native window")
    smoke = sub.add_parser("smoke", help="end-to-end test on ~20 items (images + 1 video), dry run")
    smoke.add_argument("--n", type=int, default=20)
    smoke.add_argument("--write", action="store_true", help="actually write to Eagle")
    smoke.add_argument("--models", help="comma-separated model keys")
    bench = sub.add_parser("bench", help="GPU throughput per model on synthetic batches")
    bench.add_argument("--batches", default="4,8,16", help="batch sizes to try")
    bench.add_argument("--seconds", type=float, default=8.0, help="per measurement")
    bench.add_argument("--models", help="comma-separated model keys")
    sub.add_parser("doctor", help="check GPU, models, Eagle and video support")
    explain = sub.add_parser("explain", help="show the tags computed for items (from the cache)")
    explain.add_argument("item_ids", nargs="+", metavar="ITEM_ID")
    restore = sub.add_parser("restore", help="put back tags a run removed (from the backup table)")
    restore.add_argument("--run", type=int, help="run id (omit to list runs that removed tags)")
    restore.add_argument("--remove-managed", action="store_true",
                         help="also remove the tags this tool added (full rollback)")
    restore.add_argument("--dry-run", action="store_true", help="only report what would change")
    wipe = sub.add_parser("wipe-tags",
                          help="remove ALL tags from the whole library now (backed up)")
    wipe.add_argument("--rate", type=float,
                      help="max Eagle updates per second (default: eagle.max_requests_per_second)")
    wipe.add_argument("--dry-run", action="store_true", help="only report what would change")
    return p


class ConsoleProgress:
    """One status line on a terminal, a log line every 30 s otherwise."""

    def __init__(self) -> None:
        self.tty = sys.stderr.isatty()
        self._last = 0.0
        self._lock = threading.Lock()

    def __call__(self, p: Progress) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last < (1.0 if self.tty else 30.0) and p.phase == "tagging":
                return
            self._last = now
        if p.phase != "tagging" or not p.total:
            return
        pct = 100.0 * p.done / p.total if p.total else 0.0
        line = (f"[{p.done:,}/{p.total:,} {pct:5.1f}%] {p.items_per_s:5.2f} items/s "
                f"{p.frames_per_s:5.2f} frames/s  ETA {fmt_duration(p.eta_s)}  "
                f"cached {p.cached:,} errors {p.errors:,} written {p.written:,}"
                + ("  PAUSED" if p.paused else ""))
        if self.tty:
            sys.stderr.write("\r" + line[:160].ljust(160))
            sys.stderr.flush()
        else:
            log.info(line)


def _models(value: str | None) -> tuple[str, ...] | None:
    return tuple(m.strip() for m in value.split(",") if m.strip()) if value else None


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from .pipeline import Engine, RunOptions
    opts = RunOptions(models=_models(args.models), device=args.device,
                      video=args.video, dry_run=args.dry_run, limit=args.limit,
                      library_path=args.library_path, write_mode=args.write_mode,
                      ignore_cache=args.ignore_cache)
    engine = Engine(cfg, opts, ConsoleProgress())
    try:
        summary = engine.run()
    except KeyboardInterrupt:
        engine.stop()
        return 130
    finally:
        if sys.stderr.isatty():
            sys.stderr.write("\n")
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0 if summary.get("status") == "done" else 1


def cmd_smoke(cfg: Config, args: argparse.Namespace) -> int:
    """Pick ~n items of mixed formats (at least one video) and run the full pipeline on them."""
    from .eagle import EagleClient
    from .pipeline import Engine, RunOptions

    async def sample() -> list[str]:
        async with EagleClient(cfg.eagle) as client:
            items = await client.list_items(Scope())
        by_ext: dict[str, list[str]] = {}
        for it in sorted(items, key=lambda i: i.id):
            by_ext.setdefault(it.ext.lower(), []).append(it.id)
        videos = [e for e in by_ext if e in ("mp4", "webm", "mov", "mkv", "m4v")]
        picks: list[str] = []
        if videos:
            picks.append(by_ext[videos[0]][0])
        images = [e for e in ("jpg", "jpeg", "png", "webp", "gif", "avif", "bmp") if e in by_ext]
        i = 0
        while len(picks) < args.n and images:
            ext = images[i % len(images)]
            pool = by_ext[ext]
            k = i // len(images)
            if k < len(pool):
                picks.append(pool[k * 997 % len(pool)])
            elif all(i // len(images) >= len(by_ext[e]) for e in images):
                break
            i += 1
        return list(dict.fromkeys(picks))

    import asyncio
    ids = asyncio.run(sample())
    mode = "WRITES to Eagle" if args.write else "dry run"
    log.info("Smoke test on %d items (%s)", len(ids), mode)
    opts = RunOptions(item_ids=tuple(ids), dry_run=not args.write, models=_models(args.models))
    engine = Engine(cfg, opts, ConsoleProgress())
    summary = engine.run()
    c = summary["counts"]
    ok = c["errors"] == 0 and c["done"] == len(ids)
    log.info("Smoke test %s: %d items, %d errors", "PASSED" if ok else "FAILED", c["done"],
             c["errors"])
    return 0 if ok else 1


def cmd_bench(cfg: Config, args: argparse.Namespace) -> int:
    import torch

    from . import runtime
    from .models import build_gate, build_taggers
    runtime.configure(cfg.cache_dir)
    dev = runtime.select_device(cfg.runtime.device)
    wanted = _models(args.models)
    adapters = dict(build_taggers(cfg))
    gate = build_gate(cfg)
    if gate is not None:
        adapters["gate"] = gate
    if wanted:
        adapters = {k: v for k, v in adapters.items() if k in wanted}
    log.info("Benchmark on %s (torch %s, CUDA %s)", dev.name, torch.__version__, torch.version.cuda)
    for a in adapters.values():
        a.fetch()
        a.load(dev.device)
        for bs in [int(b) for b in args.batches.split(",")]:
            x = a.example_input(bs)
            t_first = runtime.measure(functools.partial(a.forward, x), iters=1)
            n = 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.seconds:
                a.forward(x)
                n += 1
            if dev.cuda:
                torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / n / bs
            peak = torch.cuda.max_memory_allocated() / 1024**3 if dev.cuda else 0.0
            log.info("%-26s batch %2d: %7.2f frames/s (%6.1f ms/frame), peak VRAM %.1f GB, "
                     "first call %.1fs", a.title, bs, 1 / per, per * 1000, peak, t_first)
        a.unload()
        runtime.free_cuda()
    return 0


def cmd_doctor(cfg: Config) -> int:
    ok = True

    def check(name: str, fn: Any) -> None:
        nonlocal ok
        try:
            detail = fn()
            log.info("OK    %-14s %s", name, detail)
        except Exception as exc:
            ok = False
            log.error("FAIL  %-14s %s", name, exc)

    def gpu() -> str:
        import torch

        from . import runtime
        d = runtime.select_device(cfg.runtime.device)
        sm = f"sm_{d.capability[0]}{d.capability[1]}"
        return (f"{d.name}, {d.vram_bytes / 1024**3:.0f} GB, {sm}, "
                f"torch {torch.__version__} (CUDA {torch.version.cuda})")

    def triton() -> str:
        import triton
        return f"triton {triton.__version__} (torch.compile available)"

    def video() -> str:
        import av
        return f"PyAV {av.__version__}, FFmpeg libavcodec {av.library_versions['libavcodec']}"

    def vips() -> str:
        import pyvips
        return f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}"

    def eagle() -> str:
        from .eagle import probe_sync
        path, name, version = probe_sync(cfg.eagle)
        return f"Eagle {version}, library {name!r} at {path}"

    def models() -> str:
        from .models import build_gate, build_taggers
        out = []
        for a in [*build_taggers(cfg).values(), build_gate(cfg)]:
            if a is not None:
                a.fetch()
                out.append(a.title)
        return ", ".join(out)

    for name, fn in (("GPU", gpu), ("torch.compile", triton), ("video", video), ("images", vips),
                     ("Eagle", eagle), ("models", models)):
        check(name, fn)
    return 0 if ok else 1


def cmd_explain(cfg: Config, ids: list[str]) -> int:
    """Per-model and merged tags with confidences for cached items; no GPU, no Eagle needed."""
    from .media import sampling_signature
    from .models import build_gate, build_taggers
    from .pipeline import decide_tags
    from .store import Store
    from .tags import AliasMap, Blacklist, ModelDecoder, load_alias_pairs

    store = Store(cfg.db_path)
    files = store.all_files()
    taggers = build_taggers(cfg)
    gate = build_gate(cfg)
    aliases = AliasMap((load_alias_pairs() if cfg.merge.builtin_aliases else [])
                       + list(cfg.merge.aliases.items()))
    blacklist = Blacklist(cfg.merge.blacklist)
    decoders = {}
    for key, a in taggers.items():
        a.fetch()
        decoders[key] = ModelDecoder(key, a.label_space(), categories=cfg.models[key].categories,
                                     implications=cfg.models[key].implications, aliases=aliases,
                                     blacklist=blacklist)
    status = 0
    for iid in ids:
        info = files.get(iid)
        if info is None:
            print(f"{iid}: not processed yet (run `lewdex run` first)")
            status = 1
            continue
        _size, _mtime, content_hash, media = info
        sampling = sampling_signature(media if cfg.video.enabled or media != "anim" else "img",
                                      cfg.video)
        probs = {}
        for key, a in taggers.items():
            c = store.get_probs(content_hash, a.cache_key, sampling)
            if c is not None:
                probs[key] = (c.mask, c.probs)
        scores = None
        if gate is not None:
            g = store.get_probs(content_hash, gate.cache_key, sampling)
            scores = g.probs[0] if g is not None else None
        gated = {k for k in taggers if cfg.models[k].gated} if gate is not None else set()
        merged, per_model = decide_tags(cfg, decoders, blacklist, probs, scores, gated)
        frames = next(iter(probs.values()))[0].shape[0] if probs else 0
        photo = "" if scores is None else "  P(photo) " + " ".join(f"{x:.2f}" for x in scores)
        print()
        print(f"{iid}  ({media}, {frames} frame(s)){photo}")
        for key, cands in per_model.items():
            ordered = sorted(cands, key=lambda c: -c.confidence)
            print(f"  {taggers[key].title}: {len(ordered)} tags")
            print("    " + ", ".join(f"{c.name} {c.confidence:.2f}" for c in ordered))
        print(f"  merged ({len(merged)}): " + ", ".join(c.name for c in merged))
    store.close()
    return status


def cmd_restore(cfg: Config, run_id: int | None, remove_managed: bool, dry: bool) -> int:
    """Re-add the tags a run removed; optionally drop this tool's own tags (rollback)."""
    from .store import Store

    if run_id is None:
        store = Store(cfg.db_path)
        try:
            runs = store.backup_runs()
        finally:
            store.close()
        if not runs:
            print("No run has removed tags; nothing to restore.")
        for rid, n, ts in runs:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            print(f"run {rid}: {n:,} items, {when}")
        return 0
    restore_tags(cfg, run_id, remove_managed=remove_managed, dry=dry)
    return 0


def restore_tags(cfg: Config, run_id: int, *, remove_managed: bool = False,
                 dry: bool = False) -> tuple[int, int]:
    """Put back the tags run ``run_id`` removed; returns (items changed, tags put back)."""
    import asyncio

    from .eagle import EagleClient
    from .store import Store
    from .tags import normalize

    store = Store(cfg.db_path)
    try:
        rows = store.backup_rows(run_id)
        managed = store.get_managed([iid for iid, _ in rows]) if remove_managed else {}
    finally:
        store.close()

    async def go() -> tuple[int, int]:
        changed = tags_back = 0
        async with EagleClient(cfg.eagle) as client:
            for i in range(0, len(rows), 500):
                chunk = rows[i:i + 500]
                current = await client.get_tags([iid for iid, _ in chunk])
                for iid, removed in chunk:
                    cur = current.get(iid)
                    if cur is None:
                        continue
                    drop = {normalize(t) for t in managed.get(iid, ([], ""))[0]}
                    final = [t for t in cur if normalize(t) not in drop]
                    have = {normalize(t) for t in final}
                    back = [t for t in removed if normalize(t) not in have]
                    final += back
                    if final != cur:
                        changed += 1
                        tags_back += len(back)
                        if not dry:
                            await client.set_tags(iid, final)
        return changed, tags_back

    changed, back = asyncio.run(go())
    verb = "would change" if dry else "restored"
    log.info("Run %d: %s %d items (%d tags put back%s)", run_id, verb, changed, back,
             ", tool tags removed" if remove_managed else "")
    return changed, back


def _other_tagger_running() -> bool:
    import os

    import psutil
    me = os.getpid()
    parents = {p.pid for p in psutil.Process(me).parents()}
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me or p.info["pid"] in parents:
            continue
        cmd = " ".join(p.info["cmdline"] or ())
        if "lewdex" in cmd:
            return True
    return False


def cmd_wipe_tags(cfg: Config, rate: float | None, dry: bool) -> int:
    """Remove every tag of every item in the library (backed up, restorable)."""
    import asyncio

    from .eagle import EagleClient
    from .pipeline import wipe_tags
    from .store import Store

    if not dry and _other_tagger_running():
        raise TaggerError("another Lewdex (window or run) is running; stop it first, both "
                          "would rewrite the same items")
    if rate is not None:
        cfg.eagle.max_requests_per_second = rate
    store = Store(cfg.db_path)
    run_id = 0 if dry else store.start_run({"command": "wipe-tags"})
    t0 = time.monotonic()

    async def go() -> tuple[int, int, int]:
        async with EagleClient(cfg.eagle) as client:
            ids = [it.id for it in await client.list_items(Scope())]
            log.info("wipe-tags: %d items in the library", len(ids))

            def report(seen: int, changed: int, removed: int) -> None:
                eta = (len(ids) - seen) / (seen / max(time.monotonic() - t0, 1e-6))
                log.info("[%s/%s] %s items changed, %s tags removed, ETA %s", f"{seen:,}",
                         f"{len(ids):,}", f"{changed:,}", f"{removed:,}", fmt_duration(eta))

            changed, removed = await wipe_tags(client, store, ids, run_id, dry=dry,
                                               report=report)
            return len(ids), changed, removed

    status = "stopped"
    try:
        n, changed, removed = asyncio.run(go())
        status = "done"
    finally:
        if not dry:
            store.finish_run(run_id, status, {} if status != "done" else
                             {"items": n, "changed": changed, "tags_removed": removed})
        store.close()
    verb = "would remove" if dry else "removed"
    log.info("wipe-tags: %s %d tags from %d of %d items%s", verb, removed, changed, n,
             "" if dry else f"; undo with: lewdex restore --run {run_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # plain `lewdex` opens the window
        args.command, args.browser = "gui", False
    try:
        cfg = load_config(args.config)
    except TaggerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "gui":
        from .gui import run_gui
        return run_gui(cfg, native=not args.browser)
    setup_logging(cfg.log_dir, console=True, verbose=args.verbose)
    try:
        if args.command == "run":
            return cmd_run(cfg, args)
        if args.command == "smoke":
            return cmd_smoke(cfg, args)
        if args.command == "bench":
            return cmd_bench(cfg, args)
        if args.command == "doctor":
            return cmd_doctor(cfg)
        if args.command == "explain":
            return cmd_explain(cfg, args.item_ids)
        if args.command == "restore":
            return cmd_restore(cfg, args.run, args.remove_managed, args.dry_run)
        if args.command == "wipe-tags":
            return cmd_wipe_tags(cfg, args.rate, args.dry_run)
    except TaggerError as exc:
        log.error("%s", exc)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
