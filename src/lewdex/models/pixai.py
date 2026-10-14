"""PixAI Tagger v1.0: SAM3 ViTDet backbone at 1008x1008, 30,877 Danbooru tags in 6 categories."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvips
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .. import hub
from ..log import log
from ..media import vips_to_numpy
from ..runtime import Fp8Linear, attention_backend, compile_module, fp8_supported
from ..tags import LabelSpace
from .base import Adapter
from .vendor.pixai_vitdet import ViTDetCls, ViTDetConfig

SIZE = 1008


class PixaiAdapter(Adapter):
    key = "pixai"
    title = "PixAI Tagger v1.0"
    repo = "pixai-labs/pixai-tagger-v1.0"
    preprocess_version = 2  # 2: reference torchvision resize, bf16 staging
    default_batch = 8

    def __init__(self, *, thresholds: dict[str, float], **kw: Any) -> None:
        super().__init__(**kw)
        self.thresholds = thresholds
        self._cfg: dict[str, Any] | None = None
        self._weights: Path | None = None
        self._net: torch.nn.Module | None = None
        self._fwd: Any = None

    def fetch(self) -> None:
        cfg_path = hub.fetch(self.repo, "config.json", self.revision, self.models_dir)
        self._cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self._weights = hub.fetch(self.repo, "model.safetensors", self.revision, self.models_dir)

    def label_space(self) -> LabelSpace:
        assert self._cfg is not None, "fetch() first"
        names = [str(t) for t in self._cfg["tags"]]
        cats: list[str] = []
        for cat, n in self._cfg["tags_split"]:
            cats += [str(cat)] * int(n)
        default = {str(k): float(v) for k, v in self._cfg["category_best_threshold"].items()}
        per_cat = {**default, **self.thresholds}
        thr = np.array([per_cat.get(c, np.inf) for c in cats], dtype=np.float32)
        return LabelSpace(names=names, categories=cats, thresholds=thr)

    # -- CPU side ----------------------------------------------------------------------------
    def prepare(self, frame: pyvips.Image) -> torch.Tensor:
        """Exactly the reference RescalePadProcessor: float [0,1] -> antialiased bilinear resize
        to int(h*r) x int(w*r) -> centred zero padding -> (x - 0.5) / 0.5 -> bf16 (the model's
        input dtype), so the network sees bit-identical input to the published pipeline."""
        arr = vips_to_numpy(frame)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32).div_(255.0)
        h, w = int(t.shape[1]), int(t.shape[2])
        r = min(SIZE / h, SIZE / w)
        nh, nw = max(1, int(h * r)), max(1, int(w * r))
        if (nh, nw) != (h, w):
            t = F.interpolate(t[None], size=(nh, nw), mode="bilinear", align_corners=False,
                              antialias=True)[0]
        out = torch.full((3, SIZE, SIZE), -1.0, dtype=torch.float32)  # padding 0 -> -1
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        out[:, top:top + nh, left:left + nw] = t.sub_(0.5).div_(0.5)
        return out.to(torch.bfloat16)

    def collate(self, inputs: list[Any], pad_to: int | None = None) -> torch.Tensor:
        n = max(len(inputs), pad_to or 0)
        out = torch.empty((n, 3, SIZE, SIZE), dtype=torch.bfloat16)
        for i, x in enumerate(inputs):
            out[i].copy_(x)
        if n > len(inputs):
            out[len(inputs):].fill_(-1.0)
        return self.pinned(out)

    def example_input(self, n: int) -> torch.Tensor:
        return self.pinned(torch.rand((n, 3, SIZE, SIZE), dtype=torch.float32)
                           .mul_(2).sub_(1).to(torch.bfloat16))

    # -- GPU side ----------------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        assert self._cfg is not None and self._weights is not None, "fetch() first"
        t0 = time.perf_counter()
        net = ViTDetCls(ViTDetConfig.from_hf(self._cfg))
        net.load_state_dict(load_file(str(self._weights), device="cpu"), strict=True)
        net.finalize()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
        for m in net.modules():  # RoPE tables stay float32, like the original complex64 path
            for name in ("rope_cos", "rope_sin"):
                if hasattr(m, name):
                    setattr(m, name, getattr(m, name).float())
        if self.precision == "fp8":
            if fp8_supported(device):
                for blk in net.blocks:
                    blk.attn.qkv = Fp8Linear(blk.attn.qkv)
                    blk.attn.proj = Fp8Linear(blk.attn.proj)
                    blk.mlp.fc1 = Fp8Linear(blk.mlp.fc1)
                    blk.mlp.fc2 = Fp8Linear(blk.mlp.fc2)
            else:
                log.warning("FP8 needs an sm_89+ GPU; %s runs in bf16", self.title)
                self.precision = "bf16"
        self._net = net
        compiled = self.compile and device.type == "cuda"
        self._fwd = compile_module(net, self.title) if compiled else net
        self.device = device
        log.info("%s loaded (%s%s) in %.1fs", self.title, self.precision,
                 ", compiled" if self._fwd is not net else "", time.perf_counter() - t0)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._fwd is not None and self.device is not None
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        x = batch.to(self.device, non_blocking=True).to(dtype)  # already normalised (prepare)
        with attention_backend():
            logits = self._fwd(x)
        return torch.sigmoid(logits.float())

    def use_eager(self) -> None:
        super().use_eager()
        self._fwd = self._net

    def unload(self) -> None:
        self._net = None
        self._fwd = None
        super().unload()
