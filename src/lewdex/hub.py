"""Pinned Hugging Face downloads with readable errors (auth, missing revision, offline)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import errors as hf_errors
from huggingface_hub import hf_hub_download

from .errors import HuggingFaceAuthError, ModelDownloadError
from .log import log


def fetch(repo: str, filename: str, revision: str, cache_dir: Path) -> Path:
    """Return the local path of ``repo/filename@revision``, downloading it if needed."""
    try:
        return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=str(cache_dir),
                                    local_files_only=True))
    except (hf_errors.LocalEntryNotFoundError, FileNotFoundError):
        pass
    log.info("Downloading %s/%s @ %s", repo, filename, revision[:10])
    try:
        return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=str(cache_dir)))
    except hf_errors.GatedRepoError as exc:
        raise HuggingFaceAuthError(repo) from exc
    except hf_errors.RepositoryNotFoundError as exc:
        # The hub answers 401 as "not found" for private/gated repos without a token.
        raise HuggingFaceAuthError(repo) from exc
    except hf_errors.RevisionNotFoundError as exc:
        raise ModelDownloadError(f"{repo} has no revision {revision}",
                                 "Check models.<name>.revision in lewdex.toml.") from exc
    except hf_errors.EntryNotFoundError as exc:
        raise ModelDownloadError(f"{repo}@{revision[:10]} has no file {filename}") from exc
    except (hf_errors.HfHubHTTPError, OSError) as exc:
        raise ModelDownloadError(
            f"Could not download {repo}/{filename}: {exc}",
            "Check the internet connection; files already downloaded are reused offline.") from exc
