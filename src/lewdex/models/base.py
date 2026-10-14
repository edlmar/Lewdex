"""Model adapter contract.

An adapter owns everything model-specific: which files to fetch, the label space and default
decision rule, CPU preprocessing of one frame, batching (collate) and the GPU forward pass.
The pipeline only ever sees this interface, so adding a model means adding one adapter class
and registering it in ``models/__init__.py``.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, ClassVar

import pyvips
import torch

from ..tags import LabelSpace


class Adapter(abc.ABC):
    key: ClassVar[str]  # config / cache name, e.g. "pixai"
    title: ClassVar[str]  # shown in the GUI and logs
    repo: ClassVar[str]
    preprocess_version: ClassVar[int] = 1  # bump when prepare() output changes
    default_batch: ClassVar[int] = 8
    batch_candidates: ClassVar[tuple[int, ...]] = (4, 8, 16)

    def __init__(self, *, revision: str, models_dir: Path, cache_dir: Path,
                 precision: str = "bf16", compile: bool = False) -> None:
        self.revision = revision
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.precision = precision
        self.compile = compile
        self.device: torch.device | None = None

    @property
    def cache_key(self) -> str:
        """Identifies the numbers this adapter produces (model, revision, precision, preproc)."""
        return f"{self.key}@{self.revision[:12]}/{self.precision}/p{self.preprocess_version}"

    @abc.abstractmethod
    def fetch(self) -> None:
        """Make sure all files are present locally (download if needed)."""

    def label_space(self) -> LabelSpace:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self, frame: pyvips.Image) -> Any:
        """CPU: one upright 8-bit sRGB frame -> this model's input (numpy)."""

    @abc.abstractmethod
    def collate(self, inputs: list[Any], pad_to: int | None = None) -> Any:
        """CPU: stack prepared inputs into pinned tensors (optionally padded to a fixed batch)."""

    @abc.abstractmethod
    def load(self, device: torch.device) -> None:
        """Build the network on ``device`` (weights in bf16 unless configured otherwise)."""

    def use_eager(self) -> None:
        """Drop torch.compile and run the plain module (when compiling fails, e.g. no Triton)."""
        self.compile = False

    @abc.abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        """GPU: collated batch -> float32 probabilities, shape (B, n_tags) or (B,) for the gate."""

    def example_input(self, n: int) -> Any:
        """A synthetic collated batch for warm-up and batch-size probing."""
        raise NotImplementedError

    def unload(self) -> None:
        self.device = None

    @staticmethod
    def pinned(t: torch.Tensor) -> torch.Tensor:
        return t.pin_memory() if torch.cuda.is_available() else t
