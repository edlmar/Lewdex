"""Torch runtime: device selection, attention backend, compilation, FP8, batch sizing, OOM backoff.

Measured on an RTX 5090 Laptop (sm_120, torch 2.14+cu130, Windows):
* the Windows wheels have no FlashAttention; cuDNN attention is ~2x faster than the
  memory-efficient kernel, so it is put first;
* torch.compile (Inductor + triton-windows) makes PixAI v1.0 1.65x faster (8.1 -> 13.4 img/s);
  Hydra gains ~5 %, so it stays eager by default;
* FP8 GEMMs run ~2x the bf16 rate, but per-row scaling is not available on sm_120, and FP8
  PixAI changes ~3.5 % of tag decisions, so FP8 is opt-in.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .errors import CudaError
from .log import log


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    vram_bytes: int
    capability: tuple[int, int]

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda"


def list_devices() -> list[tuple[str, str]]:
    """(value, label) pairs for the device selector."""
    out: list[tuple[str, str]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append((f"cuda:{i}", f"{p.name} ({p.total_memory / 1024**3:.0f} GB)"))
    out.append(("cpu", "CPU (very slow)"))
    return out


def select_device(pref: str) -> DeviceInfo:
    if pref == "cpu":
        return DeviceInfo(torch.device("cpu"), "CPU", 0, (0, 0))
    if not torch.cuda.is_available():
        if pref == "auto":
            log.warning("CUDA is not available; running on the CPU (expect ~100x slower)")
            return DeviceInfo(torch.device("cpu"), "CPU", 0, (0, 0))
        raise CudaError(
            f"CUDA device {pref!r} requested but PyTorch sees no CUDA GPU "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})",
            "Update the NVIDIA driver (R580+ for CUDA 13) and reinstall with `pixi install`.")
    dev = torch.device("cuda:0" if pref in ("auto", "cuda") else pref)
    idx = dev.index or 0
    if idx >= torch.cuda.device_count():
        raise CudaError(f"No CUDA device {pref}; found {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(idx)
    arch = f"sm_{props.major}{props.minor}"
    if arch not in torch.cuda.get_arch_list():
        raise CudaError(
            f"This PyTorch build has no kernels for {props.name} ({arch}); "
            f"it supports {', '.join(torch.cuda.get_arch_list())}",
            "Blackwell GPUs need a CUDA 12.8+ build; pixi.toml installs torch from the cu130 "
            "index.")
    torch.cuda.set_device(idx)
    return DeviceInfo(dev, props.name, props.total_memory, (props.major, props.minor))


def configure(cache_dir: Path) -> None:
    """Process-wide torch settings. Call once, before any model is compiled."""
    inductor = cache_dir / "inductor"
    inductor.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    # Decode workers run small CPU tensor ops (PixAI's reference resize) in parallel threads;
    # a small intra-op pool avoids oversubscribing the cores.
    torch.set_num_threads(max(2, (os.cpu_count() or 8) // 8))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # static shapes; autotune only costs first batches
    try:
        import torch._inductor.config as ic
        ic.fx_graph_cache = True
        ic.autotune_local_cache = True
    except (ImportError, AttributeError):
        pass


@contextlib.contextmanager
def attention_backend() -> Iterator[None]:
    """Prefer cuDNN attention (FlashAttention is not compiled into the Windows wheels)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    try:
        ctx = sdpa_kernel(order, set_priority=True)
    except TypeError:  # older torch without set_priority
        ctx = sdpa_kernel(order)
    with ctx:
        yield


def compile_module(module: Callable[..., Any], name: str) -> Callable[..., Any]:
    """torch.compile with Inductor; returns the eager callable if Triton is unusable."""
    try:
        import triton  # noqa: F401
    except ImportError:
        log.warning("Triton is not installed; %s runs eagerly (install triton-windows)", name)
        return module
    return torch.compile(module, mode="max-autotune-no-cudagraphs", dynamic=False)


FP8_MAX = 448.0


class Fp8Linear(nn.Module):
    """Linear layer with FP8 (e4m3) GEMM: per-token activation and per-channel weight scales.

    sm_120 has no per-row scaled_mm in torch 2.14, so the GEMM runs with unit scales on
    pre-scaled operands and the row/column scales are applied to the bf16 result (the same
    arithmetic as row-wise scaling). Only worth it under torch.compile, which fuses the
    quantisation and rescaling around the GEMM.
    """

    w8: torch.Tensor
    ws: torch.Tensor
    one: torch.Tensor

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        w = lin.weight.detach().float()
        ws = w.abs().amax(dim=1).clamp(min=1e-12) / FP8_MAX
        self.register_buffer("w8", (w / ws[:, None]).to(torch.float8_e4m3fn))
        self.register_buffer("ws", ws)
        self.register_buffer("one", torch.ones((), dtype=torch.float32, device=w.device))
        self.bias: nn.Parameter | None = None
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=False)
        self.out_features = lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xs = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
        x8 = (x2.float() / xs).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=self.one, scale_b=self.one,
                             out_dtype=torch.bfloat16)
        out = y.float() * xs * self.ws
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(torch.bfloat16).reshape(*shape[:-1], self.out_features)


def fp8_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure(fn: Callable[[], Any], iters: int = 3) -> float:
    """Seconds per call after one warm-up call (synchronised)."""
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters
