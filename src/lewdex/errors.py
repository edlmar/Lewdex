"""User-facing errors. Every message says what went wrong and what to do about it."""

from __future__ import annotations


class TaggerError(Exception):
    """Base class: ``str(err)`` is shown to the user as-is (CLI and GUI)."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  -> {self.hint}" if self.hint else self.message


class ConfigError(TaggerError):
    pass


class EagleUnavailable(TaggerError):
    def __init__(self, url: str, detail: str = "") -> None:
        super().__init__(
            f"Eagle is not reachable at {url}" + (f" ({detail})" if detail else ""),
            "Start Eagle 4 (4.0 Build 21 or newer) and keep it open while tagging. "
            "Use --library-path with --dry-run to run offline.",
        )


class EagleApiError(TaggerError):
    pass


class ModelDownloadError(TaggerError):
    pass


class HuggingFaceAuthError(ModelDownloadError):
    def __init__(self, repo: str) -> None:
        super().__init__(
            f"Hugging Face refused access to {repo} (authentication required or not granted)",
            "Accept the model's terms on huggingface.co, then run `pixi run hf auth login` "
            "or set the HF_TOKEN environment variable.",
        )


class CudaError(TaggerError):
    pass


class VideoBackendError(TaggerError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            f"Video decoding is unavailable: {detail}",
            "PyAV/FFmpeg failed to load. Reinstall the environment with `pixi install`, "
            "or turn video off (--no-video / the Video checkbox).",
        )
