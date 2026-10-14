"""Windows 11 integration of the Lewdex window: theme and accent colour, dark title bar, taskbar
progress, a notification when a run ends, and a confirmation before closing during a run.

Everything here is best effort: when a call fails (or on another OS) the window simply works
without it. The window is pywebview's WinForms form, in this process.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any

from .log import log

WIN = sys.platform == "win32"

# Windows 11 default accent ("blue") as Fluent uses it: darker on light, lighter on dark.
DEFAULT_ACCENT = ("#005fb8", "#60cdff")


def system_theme() -> tuple[bool, tuple[str, str]]:
    """(apps use dark theme, (accent for light theme, accent for dark theme)) from the registry."""
    dark, accent = True, DEFAULT_ACCENT
    if not WIN:
        return dark, accent
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            dark = winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent") as k:
            pal = bytes(winreg.QueryValueEx(k, "AccentPalette")[0])
        # 8 RGBA entries: light3, light2, light1, base, dark1, dark2, dark3, -.
        # Fluent: accent buttons use dark1 on the light theme and light2 on the dark theme.
        def hexc(i: int) -> str:
            return "#{:02x}{:02x}{:02x}".format(*pal[i * 4:i * 4 + 3])
        accent = (hexc(4), hexc(1))
    except (OSError, IndexError):
        pass
    return dark, accent


class _TaskbarList:
    """ITaskbarList3 through raw COM vtable calls (no comtypes/pywin32 needed)."""

    NOPROGRESS, INDETERMINATE, NORMAL, ERROR, PAUSED = 0, 1, 2, 4, 8

    def __init__(self) -> None:
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)
        clsid = _guid("{56FDF344-FD6D-11d0-958A-006097C9A090}")
        iid = _guid("{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}")
        self.ptr = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid),
                                    ctypes.byref(self.ptr))
        if hr != 0:
            raise OSError(f"CoCreateInstance(TaskbarList) failed: {hr:#x}")
        self._call(3)  # HrInit

    def _call(self, index: int, *args: Any, argtypes: tuple[Any, ...] = ()) -> None:
        vtbl = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
        proto(vtbl[index])(self.ptr, *args)

    def set_value(self, hwnd: int, done: int, total: int) -> None:
        self._call(9, hwnd, done, total,
                   argtypes=(wintypes.HWND, ctypes.c_ulonglong, ctypes.c_ulonglong))

    def set_state(self, hwnd: int, state: int) -> None:
        self._call(10, hwnd, state, argtypes=(wintypes.HWND, ctypes.c_int))


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort),
                ("d4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    g = _GUID()
    ctypes.windll.ole32.CLSIDFromString(text, ctypes.byref(g))
    return g


class NativeWindow:
    """Wraps the pywebview window once it is shown; every method is a no-op until then."""

    def __init__(self, window: Any, icon: Path, dark: bool,
                 confirm_close: Callable[[], bool]) -> None:
        self.window = window
        self.icon = icon
        self.dark = dark
        self.hwnd = 0
        self._taskbar: _TaskbarList | None = None
        self._taskbar_thread: int | None = None
        self._last: tuple[int, int] | None = None
        self._taskbar_failed = False
        self._tray: Any = None
        self._confirm_close = confirm_close
        window.events.shown += self._on_shown
        window.events.closing += self._on_closing

    # -- window lifetime ------------------------------------------------------------------------
    def _on_shown(self) -> None:
        try:
            form = self.window.native
            self.hwnd = int(form.Handle.ToInt64())
            from nicegui.native import window_icon
            window_icon.apply_icon(self.hwnd, "Lewdex", str(self.icon))
            # Dark title bar on the dark theme (DWMWA_USE_IMMERSIVE_DARK_MODE = 20).
            on = ctypes.c_int(1 if self.dark else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                self.hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
            # Rounded corners stay the system default; refresh the frame for the title bar.
            ctypes.windll.user32.SetWindowPos(
                self.hwnd, 0, 0, 0, 0, 0, 0x0027)  # NOMOVE|NOSIZE|NOZORDER|FRAMECHANGED
        except Exception as exc:
            log.debug("window setup: %s", exc)

    def _on_closing(self) -> bool:
        """Runs on the UI thread; returning False keeps the window open."""
        try:
            return self._confirm_close()
        except Exception as exc:
            log.debug("close confirmation: %s", exc)
            return True

    def ask(self, title: str, text: str) -> bool:
        """Yes/No message box owned by the window (warning icon, No is the default)."""
        if not (WIN and self.hwnd):
            return True
        flags = 0x4 | 0x30 | 0x100  # MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2
        return bool(ctypes.windll.user32.MessageBoxW(
            self.hwnd, text, title, flags) == 6)  # IDYES

    def title(self, text: str) -> None:
        """Window title (shown in Alt+Tab and the taskbar preview); only sent when it changes."""
        if text != getattr(self, "_title", "Lewdex") and self.hwnd:
            self._title = text
            with contextlib.suppress(Exception):
                self.window.set_title(text)

    # -- taskbar progress -----------------------------------------------------------------------
    def progress(self, state: str, done: int = 0, total: int = 0) -> None:
        """state: none, busy (indeterminate), normal, paused, error."""
        if not (WIN and self.hwnd) or self._taskbar_failed:
            return
        flags = {"none": _TaskbarList.NOPROGRESS, "busy": _TaskbarList.INDETERMINATE,
                 "normal": _TaskbarList.NORMAL, "paused": _TaskbarList.PAUSED,
                 "error": _TaskbarList.ERROR}[state]
        # Repaint only on a visible change (the bar has ~200 px: 1/1000 steps are enough).
        key = (flags, int(1000 * done / total) if total else 0)
        if key == self._last:
            return
        try:
            # COM objects belong to the thread that made them; the GUI timer always runs on
            # the same event-loop thread, but guard anyway.
            if self._taskbar is None or self._taskbar_thread != threading.get_ident():
                self._taskbar, self._taskbar_thread = _TaskbarList(), threading.get_ident()
            if flags in (_TaskbarList.NORMAL, _TaskbarList.PAUSED, _TaskbarList.ERROR):
                self._taskbar.set_value(self.hwnd, done if total else 1, total or 1)
            self._taskbar.set_state(self.hwnd, flags)
            self._last = key
        except Exception as exc:
            log.debug("taskbar progress unavailable: %s", exc)
            self._taskbar_failed = True

    # -- notification ---------------------------------------------------------------------------
    def notify(self, title: str, text: str, *, warning: bool = False) -> None:
        """A Windows notification, unless the window is in front (the user can see it)."""
        if not (WIN and self.hwnd):
            return
        if ctypes.windll.user32.GetForegroundWindow() == self.hwnd:
            return
        try:
            from System import Action  # pythonnet, already loaded by pywebview
            form = self.window.native
            form.Invoke(Action(lambda: self._show_balloon(title, text, warning)))
        except Exception as exc:
            log.debug("notification: %s", exc)

    def _show_balloon(self, title: str, text: str, warning: bool) -> None:
        from System.Drawing import Icon
        from System.Windows.Forms import NotifyIcon, ToolTipIcon
        if self._tray is None:
            tray = NotifyIcon()
            tray.Icon = Icon(str(self.icon))
            tray.Text = "Lewdex"

            def bring_front(_s: Any, _e: Any) -> None:
                self.window.restore()
                self.window.native.Activate()

            tray.BalloonTipClicked += bring_front
            tray.Click += bring_front
            self._tray = tray
        self._tray.Visible = True
        self._tray.ShowBalloonTip(8000, title, text,
                                  ToolTipIcon.Warning if warning else ToolTipIcon.Info)

    def close(self) -> None:
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.Visible = False
                self._tray.Dispose()
            self._tray = None

