"""Model registry. To add a tagger: write an Adapter subclass and add it to ``TAGGERS``."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import Adapter
from .gate import GateAdapter
from .hydra import HydraAdapter
from .pixai import PixaiAdapter

TAGGERS: dict[str, type[Adapter]] = {"pixai": PixaiAdapter, "hydra": HydraAdapter}


def build_taggers(cfg: Config) -> dict[str, Adapter]:
    out: dict[str, Adapter] = {}
    for key in cfg.enabled_models():
        mc = cfg.models[key]
        common = dict(revision=mc.revision, models_dir=cfg.models_dir, cache_dir=cfg.cache_dir,
                      precision=mc.precision, compile=mc.compile)
        if key == "pixai":
            out[key] = PixaiAdapter(thresholds=mc.thresholds, **common)
        elif key == "hydra":
            out[key] = HydraAdapter(calibration=mc.calibration,
                                    threshold_overrides=mc.thresholds, **common)
    return out


def build_gate(cfg: Config) -> GateAdapter | None:
    if not cfg.gate.enabled or not any(cfg.models[k].gated for k in cfg.enabled_models()):
        return None
    return GateAdapter(variant=cfg.gate.variant, revision=cfg.gate.revision,
                       models_dir=cfg.models_dir, cache_dir=Path(cfg.cache_dir))


__all__ = ["TAGGERS", "Adapter", "GateAdapter", "build_gate", "build_taggers"]
