"""Lewdex window: a compact Windows 11 (Fluent) page in pywebview's WebView2 window.

Layout, top to bottom: the run card (progress ring, status, the buttons that make sense now),
a banner only when something needs attention, the last run's result as tiles, the log (open,
collapsible) and a status bar (Eagle on the left, the graphics card on the right). Settings open
as a dialog laid out like the Windows Settings app.

Always the whole Eagle library, videos included. The engine runs on a worker thread and never
touches the UI; a 0.5 s ``ui.timer`` reads ``engine.progress()`` and paints. The page follows the
Windows light/dark theme and accent colour; in the native window ``winnative`` adds taskbar
progress, a notification when a run ends and a confirmation before closing during a run.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from nicegui import app, ui

from . import __version__
from .config import CONFIG_TEMPLATE, DEFAULT_CONFIG_FILE, Config
from .eagle import EagleClient
from .errors import TaggerError
from .log import RingBuffer, log, setup_logging
from .stats import Progress, fmt_duration
from .winnative import NativeWindow, system_theme

DATA = Path(__file__).with_name("data")
ICON = DATA / "lewdex.ico"
MODELS = {"pixai": ("PixAI Tagger v1.0", "Danbooru tags: characters, series, art style"),
          "hydra": ("Hydra 3.5", "e621 tags: species and furry art")}
WRITE_MODES = {"replace": "Replace them", "add": "Keep them, add new",
               "sync": "Replace only Lewdex's"}
WRITE_HELP = {"replace": "Each file gets exactly the new tags.",
              "add": "Nothing is removed; new tags are added.",
              "sync": "Your own tags stay; tags Lewdex added earlier are updated."}
Kind = Literal["positive", "negative", "warning", "info"]

CSS = """
:root { --bg:#f3f3f3; --card:#fbfbfb; --stroke:#e5e5e5; --text:#1a1a1a; --muted:#5f5f5f;
  --ctl:#ffffff; --ctl-hover:#f6f6f6; --log:#f9f9f9; --track:#e1e1e1; --bar:#ebebeb;
  --on-accent:#ffffff; --ok:#0f7b0f; --warn:#9d5d00; --err:#c42b1c;
  --ok-bg:#dff6dd; --warn-bg:#fff4ce; --err-bg:#fde7e9; }
.body--dark { --bg:#202020; --card:#2b2b2b; --stroke:#1d1d1d; --text:#ffffff; --muted:#c5c5c5;
  --ctl:#2d2d2d; --ctl-hover:#323232; --log:#1f1f1f; --track:#3d3d3d; --bar:#1c1c1c;
  --on-accent:#000000; --ok:#6ccb5f; --warn:#fce100; --err:#ff99a4;
  --ok-bg:#393d1b; --warn-bg:#433519; --err-bg:#442726; }
body { background:var(--bg) !important; color:var(--text);
  font-family:'Segoe UI Variable Text','Segoe UI',system-ui,sans-serif; font-size:14px;
  user-select:none; }
.display { font-family:'Segoe UI Variable Display','Segoe UI',sans-serif; font-weight:600; }
.muted { color:var(--muted); }
.num { font-variant-numeric:tabular-nums; }
.card { background:var(--card) !important; border:1px solid var(--stroke); border-radius:8px;
  box-shadow:none; color:var(--text); }
.section { font-size:14px; font-weight:600; }
/* buttons: Fluent accent / standard / subtle */
.q-btn { text-transform:none; letter-spacing:0; font-weight:400; border-radius:4px;
  min-height:32px; }
.q-btn.btn-accent { background:var(--q-primary) !important; color:var(--on-accent) !important;
  padding:0 18px; }
.q-btn.btn-std { background:var(--ctl) !important; color:var(--text) !important;
  border:1px solid var(--stroke); padding:0 14px; }
.q-btn.btn-subtle { color:var(--text) !important; }
.q-btn.btn-link { color:var(--q-primary) !important; min-height:22px; padding:0 4px; }
.q-btn.disabled { opacity:.4 !important; }
/* progress ring */
.runring { color:var(--q-primary); }
.runring .q-circular-progress__track { stroke:var(--track); opacity:1; }
.runring.ok { color:var(--ok); } .runring.warn { color:var(--warn); }
.runring.err { color:var(--err); }
/* idle: no ring at all, just the app icon; the ring appears once a run starts */
.runring.idle .q-circular-progress__track, .runring.idle .q-circular-progress__circle {
  stroke:transparent; }
.runring-pct { font-size:21px; color:var(--text); }
.metric { color:var(--muted); font-size:13px; }
.metric .q-icon { font-size:16px; }
/* banner (Fluent InfoBar) */
.banner { border:1px solid var(--stroke); border-radius:6px; padding:8px 12px; }
.banner.warn { background:var(--warn-bg); } .banner.err { background:var(--err-bg); }
.banner.info { background:var(--card); }
.ok-fg { color:var(--ok); } .warn-fg { color:var(--warn); } .err-fg { color:var(--err); }
/* result tiles */
.tile { background:var(--card); border:1px solid var(--stroke); border-radius:8px;
  padding:9px 14px; flex:1 1 0; min-width:0; }
.tile-value { font-size:20px; line-height:1.2; white-space:nowrap; }
.tile-label { font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.tile.warn .tile-value { color:var(--warn); }
.tile-icon { width:34px; height:34px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; background:var(--ctl); border:1px solid var(--stroke);
  color:var(--q-primary); }
.tile.warn .tile-icon { color:var(--warn); }
/* log */
.logbox { background:var(--log) !important; border-top:1px solid var(--stroke);
  border-radius:0 0 8px 8px; font-family:'Cascadia Mono',Consolas,monospace; font-size:12px;
  color:var(--text); user-select:text; flex:1 1 0; min-height:60px; padding:6px 12px; }
/* status bar */
.statusbar { background:var(--bar); border-top:1px solid var(--stroke); min-height:30px;
  font-size:12px; color:var(--muted); }
.statusbar .strong { color:var(--text); }
.minibar { width:46px; }
.minibar .q-linear-progress__track { background:var(--track) !important; opacity:1 !important; }
.minibar { color:var(--q-primary); }
/* settings (Windows Settings rows) */
.q-dialog .q-card { background:var(--bg) !important; }
.setrow { background:var(--card); border:1px solid var(--stroke); border-radius:6px;
  padding:10px 14px; min-height:62px; }
.q-field--outlined .q-field__control { border-radius:4px; }
"""


class GpuMonitor:
    """Utilisation, VRAM, temperature and power of GPU 0 through NVML (absent: nothing shown)."""

    def __init__(self) -> None:
        self.handle: Any = None
        self.name = ""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = str(pynvml.nvmlDeviceGetName(self.handle))
        except Exception as exc:
            log.debug("NVML unavailable: %s", exc)

    def read(self) -> dict[str, float] | None:
        if self.handle is None:
            return None
        import pynvml
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None
        return {"util": float(util.gpu), "mem_used": mem.used / 1024**3,
                "mem_total": mem.total / 1024**3, "temp": float(temp), "power": power}


class Controller:
    """Process-wide run state shared by the page (one window, one engine at a time)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.engine: Any = None
        self.thread: threading.Thread | None = None
        self.progress = Progress()
        self.error = ""
        self.eagle_state = "Connecting to Eagle…"
        self.eagle_ok: bool | None = None
        self._probing = False
        self.models = {k: cfg.models[k].enabled for k in MODELS}
        self.write_mode: str = cfg.write.mode
        self.devices: dict[str, str] = {"auto": "Automatic"}
        self.device = "auto"
        self.gpu = GpuMonitor()
        self.last_run = self._last_run()
        self.restoring = False
        self.notices: list[tuple[str, Kind]] = []  # (message, type) for the page to show
        self.shell: NativeWindow | None = None
        self.probe()
        threading.Thread(target=self._list_devices, name="devices", daemon=True).start()

    # -- background probes --------------------------------------------------------------------
    def probe(self) -> None:
        """Check Eagle again (non-blocking); the page shows the result."""
        if self._probing:
            return
        self._probing = True
        threading.Thread(target=self._probe, name="eagle-probe", daemon=True).start()

    def _probe(self) -> None:
        async def go() -> tuple[Any, str, str]:
            async with EagleClient(self.cfg.eagle) as client:
                return await client.library_info()

        try:
            _path, name, version = asyncio.run(go())
            self.eagle_state, self.eagle_ok = f"Eagle {version} · {name}", True
        except TaggerError as exc:
            self.eagle_state, self.eagle_ok = exc.message, False
        except Exception as exc:
            self.eagle_state, self.eagle_ok = f"Eagle check failed: {exc}", False
        finally:
            self._probing = False

    def _list_devices(self) -> None:
        with contextlib.suppress(Exception):  # torch failed to load: the run reports it
            from .runtime import list_devices
            self.devices = {"auto": "Automatic", **dict(list_devices())}

    def _last_run(self) -> dict[str, Any] | None:
        from .store import Store
        try:
            store = Store(self.cfg.db_path)
            try:
                return store.last_tagging_run()
            finally:
                store.close()
        except Exception:
            return None

    def backup_runs(self) -> list[tuple[int, int, float]]:
        from .store import Store
        try:
            store = Store(self.cfg.db_path)
            try:
                return store.backup_runs()
            finally:
                store.close()
        except Exception:
            return []

    # -- control --------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def stopping(self) -> bool:
        return self.running and bool(self.engine.control.stopping)

    @property
    def busy(self) -> bool:
        return self.running or self.restoring

    def start(self, item_ids: tuple[str, ...] = ()) -> None:
        if self.busy:
            return
        from .pipeline import Engine, RunOptions
        models = tuple(k for k, on in self.models.items() if on)
        if not models:
            self.notices.append(("Turn on at least one tagger in Settings", "warning"))
            return
        device = None if self.device == "auto" else self.device
        self.engine = Engine(self.cfg, RunOptions(models=models, device=device,
                                                  item_ids=item_ids, write_mode=self.write_mode))
        self.error = ""
        self.progress = Progress(phase="listing", message="Starting")
        log.info("Start: models=%s device=%s mode=%s%s", ",".join(models), self.device,
                 self.write_mode, f" ({len(item_ids)} files to retry)" if item_ids else "")
        self.thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.engine.run()
        except TaggerError as exc:
            self.error = str(exc)
        except Exception as exc:  # full trace is in the log file
            self.error = f"Unexpected error: {exc}"
        finally:
            self.progress = self.engine.progress()
            self.last_run = self._last_run()
            self._notify_end()

    def _notify_end(self) -> None:
        if self.shell is None:
            return
        if self.error:
            self.shell.notify("Lewdex stopped with an error", self.error, warning=True)
            return
        c = (self.last_run or {}).get("stats", {}).get("counts", {})
        text = f"{c.get('written', 0):,} files updated, {c.get('unchanged', 0):,} unchanged."
        failed = c.get("write_errors", 0)
        if failed:
            text += f" {failed:,} could not be written to Eagle."
        done = (self.last_run or {}).get("status") == "done"
        self.shell.notify("Tagging finished" if done else "Tagging stopped", text,
                          warning=bool(failed))

    def toggle_pause(self) -> None:
        if self.running:
            self.engine.pause(not self.engine.control.paused)

    def stop(self) -> None:
        if self.running:
            self.engine.stop()

    def restore(self, run_id: int) -> None:
        if self.busy:
            return
        self.restoring = True

        def go() -> None:
            from .cli import restore_tags
            try:
                changed, back = restore_tags(self.cfg, run_id)
                self.notices.append((f"Put back {back:,} tags on {changed:,} files", "positive"))
            except TaggerError as exc:
                self.notices.append((f"Restore failed: {exc.message}", "negative"))
            except Exception as exc:
                log.exception("restore failed")
                self.notices.append((f"Restore failed: {exc}", "negative"))
            finally:
                self.restoring = False

        log.info("Restoring the tags removed by run %d", run_id)
        threading.Thread(target=go, name="restore", daemon=True).start()

    def set_write_mode(self, mode: str) -> None:
        self.write_mode = mode
        try:
            save_write_mode(self.cfg, mode)
        except OSError as exc:
            self.notices.append((f"Could not save the setting: {exc}", "warning"))

    def confirm_close(self) -> bool:
        """Closing the window: ask first if a run or a restore is in progress."""
        if not self.busy or self.shell is None:
            return True
        what = "Tagging" if self.running else "Restoring tags"
        if self.shell.ask("Lewdex", f"{what} is still in progress.\n\nStop it and close Lewdex? "
                          "Files that are finished are already saved; the next Start "
                          "continues where it stopped."):
            self.stop()
            return True
        return False

    def shutdown(self) -> None:
        if self.running:
            log.warning("Window closing: stopping the run first")
            self.engine.stop()
            assert self.thread is not None
            self.thread.join(60)

    def tick(self) -> Progress:
        if self.engine is not None:
            self.progress = self.engine.progress()
        return self.progress


def save_write_mode(cfg: Config, mode: str) -> None:
    """Set ``[write] mode`` in the user's lewdex.toml, keeping its comments and layout."""
    path = cfg.source if cfg.source and cfg.source != CONFIG_TEMPLATE.resolve() \
        else DEFAULT_CONFIG_FILE
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    line = f'mode = "{mode}"'
    pattern = re.compile(r'(^\[write\][ \t]*\r?\n(?:(?!\[).*\r?\n)*?)mode\s*=\s*"[a-z]*"', re.M)
    if pattern.search(text):
        text = pattern.sub(lambda m: m.group(1) + line, text, count=1)
    elif re.search(r"^\[write\][ \t]*$", text, re.M):
        text = re.sub(r"^\[write\][ \t]*$", "[write]\n" + line, text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + f"\n\n[write]\n{line}\n"
    path.write_text(text, encoding="utf-8")
    cfg.write.mode = mode  # type: ignore[assignment]


def _btn(*args: Any, **kwargs: Any) -> ui.button:
    """A button without NiceGUI's default colour (it would add bg-primary/text-white); the
    Fluent look comes from the btn-accent / btn-std classes."""
    return ui.button(*args, color=None, **kwargs)


def _ago(ts: float) -> str:
    d = max(0.0, time.time() - ts)
    if d < 90:
        return "just now"
    if d < 5400:
        return f"{d / 60:.0f} min ago"
    if d < 129600:
        return f"{d / 3600:.0f} h ago"
    return f"{d / 86400:.0f} days ago"


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.0f}k" if n >= 10_000 else f"{n:,}"


@contextlib.contextmanager
def _setting(icon: str, title: str, help_text: str = "") -> Iterator[ui.label]:
    """A Windows Settings style row: icon, title and help on the left, controls on the right.
    Yields the help label (some rows change it)."""
    with ui.row().classes("setrow w-full items-center no-wrap gap-4"):
        ui.icon(icon, size="20px").classes("muted")
        with ui.column().classes("gap-0 grow min-w-0"):
            ui.label(title)
            help_lbl = ui.label(help_text).classes("text-xs muted")
        with ui.row().classes("items-center gap-2 no-wrap shrink-0"):
            yield help_lbl


def _tile(label: str, icon: str) -> tuple[ui.element, ui.label]:
    with ui.row().classes("tile items-center no-wrap gap-3") as box:
        with ui.element("div").classes("tile-icon shrink-0"):
            ui.icon(icon, size="18px")
        with ui.column().classes("gap-0 min-w-0"):
            value = ui.label("0").classes("tile-value display num")
            ui.label(label).classes("tile-label")
    return box, value


def build(ctl: Controller, ring_buffer: RingBuffer, accent: tuple[str, str]) -> None:
    # NiceGUI serves static files as immutable: put the file's mtime in the URL so a new icon shows.
    logo = f"/lewdex-{int((DATA / 'lewdex.png').stat().st_mtime)}.png"
    app.add_static_file(local_file=DATA / "lewdex.png", url_path=logo)

    @ui.page("/")
    def page() -> None:
        # NiceGUI sets --q-primary inline on <body>; !important custom properties win over it.
        ui.add_css(CSS + f"body{{--q-primary:{accent[0]} !important}} "
                         f"body.body--dark{{--q-primary:{accent[1]} !important}}")
        ui.query(".nicegui-content").classes("p-0")
        state: dict[str, Any] = {"log_seq": 0, "log_open": True}

        with ui.column().classes("w-full h-screen no-wrap gap-0"):
            with ui.column().classes("w-full grow min-h-0 no-wrap gap-3 px-4 pt-4 pb-3"):
                # -- run card ---------------------------------------------------------------
                with ui.row().classes("card w-full items-center no-wrap gap-5 px-4 py-3"):
                    ring = ui.circular_progress(0, min=0, max=100, show_value=False,
                                                size="84px", color=None).props(
                        "thickness=0.1 instant-feedback").classes("runring shrink-0")
                    with ring:
                        ring_pct = ui.label().classes("runring-pct display num")
                        ring_logo = ui.image(logo).classes("w-9 h-9")
                        ring_icon = ui.icon("check", size="32px")
                    with ui.column().classes("gap-0.5 grow min-w-0"):
                        title_lbl = ui.label("Ready").classes("display text-[22px] leading-tight")
                        sub_lbl = ui.label().classes("text-[13px] muted ellipsis w-full")
                        with ui.row().classes("items-center gap-4 mt-1 no-wrap") as metrics:
                            with ui.row().classes("metric items-center gap-1 no-wrap"):
                                ui.icon("speed")
                                rate_lbl = ui.label().classes("num")
                            with ui.row().classes("metric items-center gap-1 no-wrap") as eta_box:
                                ui.icon("schedule")
                                eta_lbl = ui.label().classes("num")
                    # Actions in one row, vertically centred: settings (secondary) then the
                    # buttons that apply now (primary on the right, Windows' button order).
                    with ui.row().classes("items-center gap-2 no-wrap shrink-0"):
                        settings_btn = _btn("Settings", icon="settings").props(
                            "unelevated no-caps").classes("btn-std")
                        start_btn = _btn("Start", icon="play_arrow",
                                         on_click=lambda: ctl.start()).props(
                            "unelevated no-caps").classes("btn-accent")
                        pause_btn = _btn("Pause", icon="pause",
                                         on_click=ctl.toggle_pause).props(
                            "unelevated no-caps").classes("btn-std")
                        stop_btn = _btn("Stop", icon="stop", on_click=ctl.stop).props(
                            "unelevated no-caps").classes("btn-std")

                # -- banner: only when something needs attention -----------------------------
                with ui.row().classes("banner w-full items-center no-wrap gap-3") as banner:
                    banner_icon = ui.icon("warning", size="18px")
                    banner_lbl = ui.label().classes("text-[13px] grow min-w-0")
                    banner_btn = _btn("Try again").props("unelevated no-caps dense").classes(
                        "btn-std")

                # -- last run ------------------------------------------------------------------
                with ui.column().classes("w-full gap-1.5") as result:
                    result_lbl = ui.label().classes("text-[13px] muted px-0.5")
                    with ui.row().classes("w-full gap-2 no-wrap"):
                        _b, t_updated = _tile("files updated", "edit_note")
                        _b, t_same = _tile("unchanged", "check")
                        _b, t_tags = _tile("tags added / removed", "sell")

                # -- log -----------------------------------------------------------------------
                with ui.column().classes("card w-full grow min-h-0 no-wrap gap-0 relative"
                                         ) as log_card:
                    with ui.row().classes("w-full items-center no-wrap gap-1.5 px-3 py-1.5 "
                                          "cursor-pointer") as log_head:
                        chevron = ui.icon("expand_more", size="20px").classes("muted")
                        ui.label("Log").classes("section")
                        ui.space()
                        folder_btn = _btn(icon="folder_open").props(
                            "flat round dense size=sm").classes("btn-subtle")
                        folder_btn.tooltip("Open the log folder")
                    logbox = ui.log(max_lines=1500).classes("w-full logbox")
                    log_hint = ui.label("Nothing yet. What Lewdex does appears here.").classes(
                        "muted text-[13px] absolute left-4 top-[46px]")

            # -- status bar ----------------------------------------------------------------------
            with ui.row().classes("statusbar w-full items-center no-wrap gap-3 px-4"):
                eagle_dot = ui.icon("circle", size="8px")
                eagle_lbl = ui.label().classes("ellipsis min-w-0")
                ui.space()
                with ui.row().classes("items-center gap-3 no-wrap shrink-0") as gpu_box:
                    ui.label(ctl.gpu.name.replace("NVIDIA GeForce ", "")).classes("strong")
                    with ui.row().classes("items-center gap-1.5 no-wrap"):
                        gpu_bar = ui.linear_progress(0, show_value=False, size="4px",
                                                     color=None).props(
                            "rounded instant-feedback").classes("minibar")
                        gpu_util = ui.label().classes("num w-[34px]")
                    vram_lbl = ui.label().classes("num")
                    temp_lbl = ui.label().classes("num")
                    power_lbl = ui.label().classes("num")
                gpu_box.tooltip("Graphics card: load, video memory, temperature, power")

        # -- settings -----------------------------------------------------------------------------
        with ui.dialog() as settings, ui.card().classes("card p-0 gap-0 w-[560px] max-w-[92vw]"):
            with ui.row().classes("w-full items-center no-wrap px-5 pt-4 pb-2"):
                ui.label("Settings").classes("display text-[20px]")
                ui.space()
                _btn(icon="close", on_click=settings.close).props(
                    "flat round dense").classes("btn-subtle")
            with ui.column().classes("w-full gap-1.5 px-5 pb-5"):
                ui.label("Tags").classes("section mt-1 mb-0.5")
                with _setting("sell", "Tags a file already has",
                              WRITE_HELP.get(ctl.write_mode, "")) as mode_help:
                    def set_mode(e: Any) -> None:
                        ctl.set_write_mode(e.value)
                        mode_help.set_text(WRITE_HELP.get(e.value, ""))
                    ui.select(WRITE_MODES, value=ctl.write_mode, on_change=set_mode).props(
                        "dense outlined options-dense").classes("w-[190px]")
                with _setting("history", "Put back old tags",
                              "Old tags are saved before every change."):
                    runs_select = ui.select({}, label="Run").props(
                        "dense outlined options-dense").classes("w-[220px]")
                    restore_btn = _btn("Put back").props("unelevated no-caps").classes(
                        "btn-std")

                ui.label("Taggers").classes("section mt-3 mb-0.5")
                for key, (name, what) in MODELS.items():
                    with _setting("auto_awesome", name, what):
                        ui.switch(value=ctl.models[key],
                                  on_change=lambda e, k=key: ctl.models.__setitem__(
                                      k, bool(e.value))).props("dense")
                with _setting("memory", "Graphics card", "Where the taggers run."):
                    dev_select = ui.select(ctl.devices, value=ctl.device,
                                           on_change=lambda e: setattr(ctl, "device", e.value)
                                           ).props("dense outlined options-dense").classes(
                        "w-[190px]")

                ui.label("More").classes("section mt-3 mb-0.5")
                with _setting("tune", "Advanced settings",
                              "Thresholds, tag categories, the blacklist."):
                    _btn("Open lewdex.toml", on_click=lambda: _edit(ctl.cfg)).props(
                        "unelevated no-caps").classes("btn-std")
                ui.label(f"Lewdex {__version__}").classes("text-xs muted mt-2")

        with ui.dialog() as confirm, ui.card().classes("card p-6 gap-4 w-[420px]"):
            ui.label("Put back old tags?").classes("display text-[18px]")
            confirm_lbl = ui.label().classes("text-[14px]")
            with ui.row().classes("w-full justify-end gap-2"):
                _btn("Put back", on_click=lambda: confirm.submit(True)).props(
                    "unelevated no-caps").classes("btn-accent")
                _btn("Cancel", on_click=lambda: confirm.submit(False)).props(
                    "unelevated no-caps").classes("btn-std")

        # -- actions ------------------------------------------------------------------------------
        def open_settings() -> None:
            dev_select.set_options(ctl.devices, value=ctl.device)
            runs = ctl.backup_runs()
            runs_select.set_options(
                {rid: f"{time.strftime('%d %b %Y', time.localtime(ts))} · {n:,} "
                      f"file{'' if n == 1 else 's'}" for rid, n, ts in runs},
                value=runs[0][0] if runs else None)
            restore_btn.set_enabled(bool(runs))
            settings.open()

        async def ask_restore() -> None:
            rid = runs_select.value
            if rid is None or ctl.busy:
                return
            confirm_lbl.set_text(f"The tags removed by the run of {runs_select.options[rid]} "
                                 "are added back. Tags added since then stay.")
            if await confirm:
                settings.close()
                ctl.restore(int(rid))

        def retry_failed() -> None:
            ctl.start(item_ids=tuple(_failed_ids(ctl)))

        def toggle_log() -> None:
            state["log_open"] = not state["log_open"]
            logbox.set_visibility(state["log_open"])
            log_hint.set_visibility(state["log_open"] and not state["log_seq"])
            chevron.props(f"name={'expand_more' if state['log_open'] else 'chevron_right'}")
            if state["log_open"]:
                log_card.classes(add="grow")
            else:
                log_card.classes(remove="grow")

        def banner_action() -> None:
            if ctl.error:
                _open_folder(ctl.cfg.log_dir)
            elif ctl.eagle_ok is False:
                ctl.probe()
            elif _failed_ids(ctl):
                retry_failed()
            else:
                _open_folder(ctl.cfg.log_dir)

        settings_btn.on_click(open_settings)
        restore_btn.on_click(ask_restore)
        log_head.on("click", toggle_log)
        folder_btn.on("click.stop", lambda: _open_folder(ctl.cfg.log_dir))
        banner_btn.on_click(banner_action)

        # -- painting -----------------------------------------------------------------------------
        def paint_run(p: Progress) -> None:
            running, restoring = ctl.running, ctl.restoring
            paused = running and p.paused
            stopping = ctl.stopping
            frac = min(1.0, p.done / p.total) if p.total else 0.0
            finished = not running and ctl.engine is not None and not ctl.error
            # ring
            look, centre, icon = "", "logo", "check"  # centre: logo, pct or icon
            spin = False
            if ctl.error:
                look, centre, icon = "err", "icon", "priority_high"
            elif restoring or (running and not p.total):
                spin = True
            elif running:
                look, centre = ("warn" if paused or stopping else ""), "pct"
            elif finished and p.phase == "done":
                look, centre, icon = "ok", "icon", "check"
            elif finished and p.total:
                look, centre = "idle" if not p.done else "warn", "pct"
            else:
                look = "idle"
            ring.classes(remove="idle ok warn err", add=look)
            if spin:
                ring.props("indeterminate")
            else:
                ring.props(remove="indeterminate")
            ring.set_value(100 if centre == "icon" else 100 * frac)
            ring_pct.set_visibility(centre == "pct")
            ring_pct.set_text(f"{frac * 100:.0f}%")
            ring_logo.set_visibility(centre == "logo")
            ring_icon.set_visibility(centre == "icon")
            if centre == "icon":
                ring_icon.props(f"name={icon}")
                ring_icon.classes(replace="err-fg" if look == "err" else "ok-fg")

            # title, subtitle, metrics
            counts = f"{p.done:,} of {p.total:,} files"
            show_metrics = False
            if ctl.error:
                title, sub = "Stopped by an error", "Details are below and in the log."
            elif restoring:
                title, sub = "Putting back old tags", "Writing the saved tags back to Eagle…"
            elif stopping:
                title, sub = "Stopping", "Finishing the current step and saving finished files…"
            elif paused:
                title, sub = "Paused", counts
            elif running and p.phase == "tagging" and p.total:
                title, sub, show_metrics = "Tagging", counts, True
            elif running:
                title = {"listing": "Reading the library", "loading": "Loading the taggers",
                         "planning": "Preparing"}.get(p.phase, "Starting")
                sub = p.message
            elif finished and p.phase == "done":
                title, sub = "Done", f"{p.total:,} files checked in {fmt_duration(p.elapsed_s)}"
            elif finished:
                title, sub = "Stopped", f"{counts} · Start continues where it stopped"
            elif ctl.eagle_ok is False:
                title, sub = "Waiting for Eagle", "Start Eagle and open your library."
            else:
                title = "Ready"
                counts = (ctl.last_run or {}).get("stats", {}).get("counts", {})
                total = int(counts.get("total", 0))
                sub = (f"{total:,} files at the last run; files already tagged are skipped."
                       if total else "Tags every image, animation and video in your Eagle library.")
            title_lbl.set_text(title)
            sub_lbl.set_text(sub)
            metrics.set_visibility(show_metrics)
            if show_metrics:
                rate_lbl.set_text(f"{p.items_per_s:.1f} files/s")
                left = fmt_duration(p.eta_s)
                eta_box.set_visibility(left != "--")
                eta_lbl.set_text(f"{left} left")

            # buttons: only the ones that make sense now
            start_btn.set_visibility(not running)
            start_btn.set_enabled(not ctl.busy and ctl.eagle_ok is not False)
            pause_btn.set_visibility(running and not stopping)
            stop_btn.set_visibility(running and not stopping)
            pause_btn.set_text("Resume" if paused else "Pause")
            pause_btn.props(f"icon={'play_arrow' if paused else 'pause'}")
            settings_btn.set_visibility(not ctl.busy)

            # banner
            if ctl.error:
                banner.classes(remove="warn info", add="err")
                banner_icon.props("name=error").classes(replace="err-fg")
                banner_lbl.set_text(ctl.error)
                banner_btn.set_text("Open log folder")
                banner.set_visibility(True)
            elif ctl.eagle_ok is False and not running:
                banner.classes(remove="err info", add="warn")
                banner_icon.props("name=warning").classes(replace="warn-fg")
                banner_lbl.set_text(ctl.eagle_state)
                banner_btn.set_text("Try again")
                banner_btn.set_enabled(not ctl._probing)
                banner.set_visibility(True)
            elif not running and not ctl.restoring and _problems(ctl):
                banner.classes(remove="err info", add="warn")
                banner_icon.props("name=warning").classes(replace="warn-fg")
                banner_lbl.set_text(_problems(ctl))
                banner_btn.set_text("Retry these files" if _failed_ids(ctl) else "Open log folder")
                banner_btn.set_enabled(True)
                banner.set_visibility(True)
            else:
                banner.set_visibility(False)

        def paint_result() -> None:
            lr = ctl.last_run
            if ctl.running or lr is None:
                result.set_visibility(False)
                return
            st = lr.get("stats", {})
            c = st.get("counts", {})
            took = fmt_duration(float(st.get("wall_s", 0)))
            result_lbl.set_text(f"Last run {lr['status']} {_ago(lr['finished'] or lr['started'])}"
                                f"{'' if took == '--' else ' · took ' + took}")
            t_updated.set_text(f"{c.get('written', 0):,}")
            t_same.set_text(f"{c.get('unchanged', 0):,}")
            t_tags.set_text(f"+{_compact(c.get('tags_added', 0))} / "
                            f"−{_compact(c.get('tags_removed', 0))}")  # noqa: RUF001
            result.set_visibility(True)

        def paint_status() -> None:
            eagle_lbl.set_text(ctl.eagle_state)
            eagle_dot.classes(replace={True: "ok-fg", False: "err-fg", None: "warn-fg"}[
                ctl.eagle_ok])
            g = ctl.gpu.read()
            gpu_box.set_visibility(g is not None)
            if g is not None:
                gpu_bar.set_value(g["util"] / 100)
                gpu_util.set_text(f"{g['util']:.0f}%")
                vram_lbl.set_text(f"{g['mem_used']:.1f} / {g['mem_total']:.0f} GB")
                temp_lbl.set_text(f"{g['temp']:.0f} °C")
                temp_lbl.classes(replace="num warn-fg" if g["temp"] >= 85 else "num")
                power_lbl.set_text(f"{g['power']:.0f} W")

        def paint_taskbar(p: Progress) -> None:
            shell = ctl.shell
            if shell is None:
                return
            if ctl.running and (p.paused or ctl.stopping):
                shell.progress("paused", p.done, p.total)
            elif ctl.running and p.total:
                shell.progress("normal", p.done, p.total)
            elif ctl.running or ctl.restoring:
                shell.progress("busy")
            elif ctl.error:
                shell.progress("error", 1, 1)
            else:
                shell.progress("none")

        def paint() -> None:
            p = ctl.tick()
            for message, kind in ctl.notices[:]:
                ui.notify(message, type=kind, position="bottom-right")
                ctl.notices.remove((message, kind))
            paint_run(p)
            paint_result()
            paint_status()
            paint_taskbar(p)
            if ctl.shell is not None:
                running = ctl.running and p.total and not p.paused
                ctl.shell.title(f"Lewdex — {p.done / p.total:.0%}" if running else "Lewdex")
            fresh = ring_buffer.since(state["log_seq"])
            if fresh:
                state["log_seq"] = fresh[-1][0]
                log_hint.set_visibility(False)
                for _seq, level, line in fresh[-300:]:
                    logbox.push(line, classes="err-fg" if level >= 40 else
                                "warn-fg" if level >= 30 else "")

        def on_key(e: Any) -> None:
            if not e.action.keydown or e.action.repeat:
                return
            if e.key.name == "Enter" and e.modifiers.ctrl and not ctl.busy:
                ctl.start()
            elif e.key.space and ctl.running:
                ctl.toggle_pause()

        ui.keyboard(on_key=on_key)
        start_btn.tooltip("Start (Ctrl+Enter)")
        pause_btn.tooltip("Pause or resume (Space)")
        stop_btn.tooltip("Finish the current step and stop")
        paint()
        ui.timer(0.5, paint)


def _failed_ids(ctl: Controller) -> list[str]:
    return list((ctl.last_run or {}).get("stats", {}).get("failed_writes") or [])


def _problems(ctl: Controller) -> str:
    """What went wrong in the last run, as one sentence ('' if nothing)."""
    c = (ctl.last_run or {}).get("stats", {}).get("counts", {})
    failed, unread = int(c.get("write_errors", 0)), int(c.get("errors", 0))
    parts = []
    if failed:
        parts.append(f"{failed:,} file{'' if failed == 1 else 's'} could not be written to Eagle")
    if unread:
        parts.append(f"{unread:,} file{'' if unread == 1 else 's'} could not be read")
    return " and ".join(parts) + " in the last run." if parts else ""


def _edit(cfg: Config) -> None:
    """Open the user's lewdex.toml in Notepad (created from the template if missing)."""
    from .config import prepare_project_root
    prepare_project_root()
    path = cfg.source if cfg.source and cfg.source != CONFIG_TEMPLATE.resolve() \
        else DEFAULT_CONFIG_FILE
    subprocess.Popen(["notepad.exe", str(path)])


def _open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.startfile(path)  # Windows only, like the window


def run_gui(cfg: Config, *, native: bool = True) -> int:
    ring = RingBuffer()
    setup_logging(cfg.log_dir, console=False, ring=ring)
    ctl = Controller(cfg)
    dark, accent = system_theme()
    build(ctl, ring, accent)
    from nicegui import native as nicegui_native
    port = nicegui_native.find_open_port(8710, 8790)
    common: dict[str, Any] = {"title": "Lewdex", "favicon": ICON, "dark": None,
                              "show_welcome_message": False}
    if not native:
        app.on_shutdown(ctl.shutdown)
        ui.run(host="127.0.0.1", port=port, reload=False, show=True,
               uvicorn_logging_level="warning", **common)
        return 0

    # Native: the page server runs on a thread and pywebview owns the main thread, so the
    # window (taskbar, notifications, closing) is reachable from this process.
    import uvicorn
    import webview
    from fastapi import FastAPI

    api = FastAPI()
    ui.run_with(api, **common)
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port,
                                           log_level="warning"))
    server_thread = threading.Thread(target=server.run, name="ui-server", daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 30
    while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)

    window = webview.create_window("Lewdex", f"http://127.0.0.1:{port}/", width=860,
                                   height=600, min_size=(680, 480),
                                   background_color="#202020" if dark else "#f3f3f3")
    ctl.shell = NativeWindow(window, ICON, dark, ctl.confirm_close)
    try:
        webview.start()
    finally:
        ctl.shell.close()
        ctl.shell = None
        ctl.shutdown()
        server.should_exit = True
        server_thread.join(5)
    return 0
