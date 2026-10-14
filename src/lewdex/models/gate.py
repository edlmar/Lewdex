"""Photo gate: deepghs anime/real classifier (CAFormer-S36 v1.4, 99.1 % accuracy, 384 px).

Hydra is trained on e621 art only; on photographs it produces furry tags that are wrong. Frames
the gate classifies as real photos skip the models configured with ``gated = true``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvips
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .. import hub
from ..errors import ModelDownloadError
from ..log import log
from ..media import vips_to_numpy
from .base import Adapter

SIZE = 384
# The checkpoint is a full pickle (it embeds PIL plots), so it is only unpickled after its
# SHA-256 matches the LFS object id of the pinned revision; then it is kept as safetensors.
PINNED_SHA256 = {
    ("097a6c1e9e62866a435810c883a45a7cb2c7077d", "caformer_s36_v1.4"):
        "a794a59cda11d79d93a3ada0f2b53dd091f4f7b04f9e0ce2c935693762411588",
}
TIMM_NAMES = {"caformer_s36": "caformer_s36", "mobilenetv3": "mobilenetv3_large_100"}
REAL_INDEX = 1  # labels: ["anime", "real"]


class GateAdapter(Adapter):
    key = "gate"
    title = "Photo gate (anime/real v1.4)"
    repo = "deepghs/anime_real_cls"
    preprocess_version = 2  # 2: PIL bilinear, as in dghs-imgutils
    default_batch = 32

    def __init__(self, *, variant: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.variant = variant
        self._st_path: Path | None = None
        self._net: torch.nn.Module | None = None

    @property
    def cache_key(self) -> str:
        return f"gate:{self.variant}@{self.revision[:12]}/p{self.preprocess_version}"

    def fetch(self) -> None:
        name = f"anime_real_cls-{self.variant}-{self.revision[:12]}.safetensors"
        out = self.cache_dir / "models" / name
        if not out.is_file():
            ckpt = hub.fetch(self.repo, f"{self.variant}/model.ckpt", self.revision,
                             self.models_dir)
            expected = PINNED_SHA256.get((self.revision, self.variant))
            digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            if expected is None or digest != expected:
                raise ModelDownloadError(
                    f"Refusing to unpickle {ckpt.name}: SHA-256 {digest[:16]}... is not pinned",
                    "Add the verified hash to PINNED_SHA256 in models/gate.py for this revision.")
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
            profiling = ("total_ops", "total_params")  # thop counters, not weights
            state = {k: v.contiguous() for k, v in state.items()
                     if isinstance(v, torch.Tensor) and not k.endswith(profiling)}
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(state, str(out))
            log.info("Converted %s to %s", ckpt.name, out.name)
        self._st_path = out

    def _arch(self) -> str:
        for prefix, name in TIMM_NAMES.items():
            if self.variant.startswith(prefix):
                return name
        raise ModelDownloadError(f"Unknown photo-gate variant {self.variant!r}")

    def prepare(self, frame: pyvips.Image) -> np.ndarray:
        """As dghs-imgutils does it: stretch to 384x384 with PIL's bilinear filter."""
        img = Image.fromarray(vips_to_numpy(frame)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def collate(self, inputs: list[Any], pad_to: int | None = None) -> torch.Tensor:
        n = max(len(inputs), pad_to or 0)
        out = torch.zeros((n, SIZE, SIZE, 3), dtype=torch.uint8)
        arr = out.numpy()
        for i, x in enumerate(inputs):
            arr[i] = x
        return self.pinned(out)

    def example_input(self, n: int) -> torch.Tensor:
        return self.pinned(torch.randint(0, 256, (n, SIZE, SIZE, 3), dtype=torch.uint8))

    def load(self, device: torch.device) -> None:
        import timm
        assert self._st_path is not None, "fetch() first"
        t0 = time.perf_counter()
        net = timm.create_model(self._arch(), num_classes=2, pretrained=False)
        net.load_state_dict(load_file(str(self._st_path)), strict=True)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self._net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device = device
        log.info("%s loaded in %.1fs", self.title, time.perf_counter() - t0)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._net is not None and self.device is not None
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        x = batch.to(self.device, non_blocking=True).permute(0, 3, 1, 2).to(dtype)
        x = x.mul_(1.0 / 127.5).sub_(1.0)
        return torch.softmax(self._net(x).float(), dim=-1)[:, REAL_INDEX]

    def unload(self) -> None:
        self._net = None
        super().unload()
