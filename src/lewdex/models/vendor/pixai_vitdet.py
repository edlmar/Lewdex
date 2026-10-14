# Copyright PixAI Labs / Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
#
# Vendored from pixai-labs/pixai-tagger-v1.0 @ 9fe10addf9326e292da8a85a98ea74cd91b41771
# (tagger_pipeline.py, itself adapted from facebookresearch/sam3 sam3/model/vitdet.py).
#
# Changes from the original, none of which alter the numbers it computes:
# * plain nn.Module instead of transformers.PreTrainedModel (no transformers dependency);
# * inference-only: drop-path, dropout, layer-scale, activation checkpointing,
#   relative position embeddings and the class-token path are gone (the released
#   config disables all of them);
# * 2-D RoPE uses real cos/sin arithmetic instead of complex tensors, so
#   torch.compile and exporters can handle it. Rotation is still computed in
#   float32 and cast back, exactly like the complex original;
# * the tiled absolute position embedding for the fixed 1008x1008 input is
#   precomputed once (it is a pure function of the weights).
# Parameter names are unchanged, so the published safetensors load with strict=True.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ViTDetConfig:
    num_classes: int = 30877
    img_size: int = 1008
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 32
    num_heads: int = 16
    mlp_ratio: float = 4.625
    window_size: int = 24
    global_att_blocks: tuple[int, ...] = (7, 15, 23, 31)
    pretrain_img_size: int = 336
    rope_theta: float = 10000.0

    @classmethod
    def from_hf(cls, cfg: dict[str, object]) -> ViTDetConfig:
        return cls(
            num_classes=int(cfg["num_classes"]),  # type: ignore[call-overload]
            img_size=int(cfg["img_size"]),  # type: ignore[call-overload]
            patch_size=int(cfg["patch_size"]),  # type: ignore[call-overload]
            embed_dim=int(cfg["embed_dim"]),  # type: ignore[call-overload]
            depth=int(cfg["depth"]),  # type: ignore[call-overload]
            num_heads=int(cfg["num_heads"]),  # type: ignore[call-overload]
            mlp_ratio=float(cfg["mlp_ratio"]),  # type: ignore[arg-type]
            window_size=int(cfg["window_size"]),  # type: ignore[call-overload]
            global_att_blocks=tuple(int(i) for i in cfg["global_att_blocks"]),  # type: ignore[attr-defined]
            pretrain_img_size=int(cfg["pretrain_img_size"]),  # type: ignore[call-overload]
        )

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size


def _axial_rope(head_dim: int, end_x: int, end_y: int, theta: float,
                scale_pos: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables equal to the original ``compute_axial_cis`` (float32, (L, head_dim/2))."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float() * scale_pos
    t_y = torch.div(t, end_x, rounding_mode="floor").float() * scale_pos
    angles = torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], dim=-1)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate adjacent pairs (x[2i], x[2i+1]) by the per-position angle, in float32."""
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # B H W C


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, input_size: tuple[int, int],
                 rope_pt_size: tuple[int, int], theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # use_interp_rope=True in the released config: positions are scaled by pt/input.
        scale_pos = rope_pt_size[0] / input_size[0]
        cos, sin = _axial_rope(self.head_dim, input_size[0], input_size[1], theta, scale_pos)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, _ = x.shape
        seq = h * w
        qkv = self.qkv(x).reshape(b, seq, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, L, head_dim)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(x)


def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, tuple[int, int]]:
    b, h, w, c = x.shape
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // ws, ws, wp // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c), (hp, wp)


def _window_unpartition(win: Tensor, ws: int, pad_hw: tuple[int, int],
                        hw: tuple[int, int]) -> Tensor:
    hp, wp = pad_hw
    h, w = hw
    b = win.shape[0] // (hp * wp // ws // ws)
    x = win.reshape(b, hp // ws, wp // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, :h, :w, :] if (hp > h or wp > w) else x


class Block(nn.Module):
    def __init__(self, cfg: ViTDetConfig, window_size: int) -> None:
        super().__init__()
        dim = cfg.embed_dim
        grid = cfg.grid
        self.window_size = window_size
        input_size = (window_size, window_size) if window_size else (grid, grid)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, cfg.num_heads, input_size,
                              (cfg.window_size, cfg.window_size), cfg.rope_theta)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, int(dim * cfg.mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
            x = self.attn(x)
            x = _window_unpartition(x, self.window_size, pad_hw, (h, w))
        else:
            x = self.attn(x)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class MHAttnPool(nn.Module):
    """Single learned query attending over all patch tokens."""

    def __init__(self, dim: int, num_heads: int, num_patches: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Parameter(torch.empty(dim))
        self.kv = nn.Linear(dim, dim * 2)
        self.pos_embed = nn.Parameter(torch.empty(num_patches, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        x = self.norm(x) + self.pos_embed[None]
        q = self.q[None, None].expand(b, -1, -1)
        k, v = self.kv(x).chunk(2, dim=-1)
        q = q.view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, 1, d)
        return self.proj(out).squeeze(1)


class ViTDetCls(nn.Module):
    """PixAI Tagger v1.0 backbone + attention-pool classification head. Returns logits."""

    def __init__(self, cfg: ViTDetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg.patch_size, dim)
        n_pre = (cfg.pretrain_img_size // cfg.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_pre + 1, dim))  # +1: pretraining cls slot
        self.ln_pre = nn.LayerNorm(dim, eps=1e-5)
        self.blocks = nn.ModuleList(
            Block(cfg, 0 if i in cfg.global_att_blocks else cfg.window_size)
            for i in range(cfg.depth)
        )
        self.head_pool = MHAttnPool(dim, cfg.num_heads, cfg.grid**2)
        self.head = nn.Linear(dim, cfg.num_classes)
        self._last = cfg.global_att_blocks[-1]
        self.register_buffer("abs_pos", torch.empty(0), persistent=False)

    def _tiled_abs_pos(self, h: int, w: int) -> Tensor:
        pos = self.pos_embed[:, 1:]  # drop the cls-token slot
        size = int(math.sqrt(pos.shape[1]))
        if (size, size) == (h, w):
            return pos.reshape(1, h, w, -1)
        grid = pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        grid = grid.tile([1, 1, h // size + 1, w // size + 1])[:, :, :h, :w]
        return grid.permute(0, 2, 3, 1)

    def finalize(self) -> None:
        """Precompute the tiled absolute position embedding (call after loading weights)."""
        g = self.cfg.grid
        with torch.no_grad():
            self.abs_pos = self._tiled_abs_pos(g, g).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        abs_pos = self.abs_pos if self.abs_pos.numel() else self._tiled_abs_pos(
            x.shape[1], x.shape[2])
        x = self.ln_pre(x + abs_pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self._last:
                break
        x = x.flatten(1, 2)  # B, H*W, C (same token order as the original NCHW flatten)
        return self.head(self.head_pool(x))
