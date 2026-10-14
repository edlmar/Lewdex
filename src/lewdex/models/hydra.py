"""Hydra 3.5 (RedRocket): SigLIP2 so400m NaFlex encoder + per-tag cross-attention head.

8,886 e621 tags. Inputs keep their aspect ratio: each frame is resized (MKS2013 kernel in
linear light, exactly as Hydra's own loader) to at most 1024 patches of 16x16 px, then padded
to 1024 patches with a validity mask. Thresholds come from the model's embedded validation
statistics (default metric F1 with a 0.1 precision floor), implications from its label graph.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyvips
import torch

from .. import hub
from ..log import log
from ..media import vips_to_numpy
from ..runtime import attention_backend, compile_module
from ..tags import LabelSpace, normalize
from .base import Adapter
from .vendor.hydra.classification import parse_metric
from .vendor.hydra.image import _LUT_RGB16_SRGB, _LUT_SRGB_RGB16
from .vendor.hydra.model import get_image_size_for_seq, load_model

PATCH = 16
SEQ = 1024
WEIGHTS = "models/hydra-3.5.safetensors"


class HydraAdapter(Adapter):
    key = "hydra"
    title = "Hydra 3.5 (e621)"
    repo = "RedRocket/Hydra"
    preprocess_version = 1
    default_batch = 8

    def __init__(self, *, calibration: str, threshold_overrides: dict[str, float],
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.calibration = calibration or "f1.0@0.1"
        self.overrides = {normalize(k): v for k, v in threshold_overrides.items()}
        self._path: str | None = None
        self._net: Any = None
        self._fwd: Any = None

    def fetch(self) -> None:
        self._path = str(hub.fetch(self.repo, WEIGHTS, self.revision, self.models_dir))

    def label_space(self) -> LabelSpace:
        assert self._path is not None, "fetch() first"
        meta = load_model(self._path, metadata_only=True)
        labels = meta.labels
        index = {lbl.label: i for i, lbl in enumerate(labels)}
        implies = [[index[t] for t in lbl.implies if t in index] for lbl in labels]
        thresholds = np.array(meta.calibrate(parse_metric(self.calibration)).as_thresholds(),
                              dtype=np.float32)
        names = [lbl.label for lbl in labels]
        for i, n in enumerate(names):
            thr = self.overrides.get(normalize(n))
            if thr is not None:
                thresholds[i] = thr
        cats = [lbl.category for lbl in labels]
        return LabelSpace(names=names, categories=cats, thresholds=thresholds, implies=implies)

    # -- CPU side ----------------------------------------------------------------------------
    def prepare(self, frame: pyvips.Image) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = get_image_size_for_seq((frame.height, frame.width), PATCH, SEQ)
        img = frame
        if (h, w) != (frame.height, frame.width):
            img = frame.maplut(_LUT_SRGB_RGB16)
            img = img.resize(w / frame.width, vscale=h / frame.height, kernel="mks2013", gap=3.0)
            img = img.maplut(_LUT_RGB16_SRGB)
            if (img.height, img.width) != (h, w):
                img = img.embed(0, 0, w, h, extend="copy")
        arr = vips_to_numpy(img)
        gh, gw = h // PATCH, w // PATCH
        patches = (arr.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 1, 3, 4)
                   .reshape(gh * gw, PATCH * PATCH * 3))
        return np.ascontiguousarray(patches), (gh, gw)

    def collate(self, inputs: list[Any],
                pad_to: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = max(len(inputs), pad_to or 0)
        patches = torch.zeros((n, SEQ, PATCH * PATCH * 3), dtype=torch.uint8)
        sizes = torch.ones((n, 2), dtype=torch.int64)
        arr = patches.numpy()
        for i, (p, (gh, gw)) in enumerate(inputs):
            arr[i, : p.shape[0]] = p
            sizes[i, 0], sizes[i, 1] = gh, gw
        return self.pinned(patches), sizes

    def example_input(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        patches = torch.randint(0, 256, (n, SEQ, PATCH * PATCH * 3), dtype=torch.uint8)
        sizes = torch.full((n, 2), 32, dtype=torch.int64)
        return self.pinned(patches), sizes

    # -- GPU side ----------------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        assert self._path is not None, "fetch() first"
        t0 = time.perf_counter()
        net = load_model(self._path)  # bf16 weights, eval mode, sigmoid outputs
        if device.type != "cuda":
            net = net.float()
        net = net.to(device)
        net.embeds.enable_cache()  # interpolated position embeddings, per patch-grid shape
        self._net = net
        compiled = self.compile and device.type == "cuda"
        self._fwd = compile_module(net.forward, self.title) if compiled else net.forward
        self.device = device
        log.info("%s loaded%s in %.1fs", self.title, " (compiled)" if compiled else "",
                 time.perf_counter() - t0)

    def use_eager(self) -> None:
        super().use_eager()
        if self._net is not None:
            self._fwd = self._net.forward

    @torch.inference_mode()
    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert self._net is not None and self.device is not None
        patches, sizes = batch
        x = self._net.from_srgb(patches.to(self.device, non_blocking=True))
        with attention_backend():
            probs: torch.Tensor = self._fwd(x, sizes)
        return probs.float()

    def unload(self) -> None:
        self._net = None
        self._fwd = None
        super().unload()
