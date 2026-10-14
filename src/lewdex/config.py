"""Configuration: typed defaults, overridden by ``lewdex.toml`` and then by CLI/GUI choices.

Unknown keys are an error (typos should not silently fall back to defaults). Relative paths
resolve against the directory of the config file (or the project root without one).
"""

from __future__ import annotations

import contextlib
import copy
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "lewdex.toml"
CONFIG_TEMPLATE = Path(__file__).with_name("data") / "lewdex.toml"
def prepare_project_root(root: Path = PROJECT_ROOT) -> None:
    """Create the editable ``lewdex.toml`` from the shipped template if it is missing."""
    cfg_file = root / "lewdex.toml"
    if not cfg_file.exists() and CONFIG_TEMPLATE.is_file():
        with contextlib.suppress(OSError):  # read-only folder: the built-in defaults apply
            cfg_file.write_text(CONFIG_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

Precision = Literal["bf16", "fp8"]
Aggregate = Literal["mean", "max", "pmean"]
WriteMode = Literal["replace", "add", "sync"]
Implications = Literal["off", "inherit", "preserve", "collapse"]


@dataclass
class EagleConfig:
    url: str = "http://localhost:41595"
    # Only needed when Eagle runs on another machine; local requests are trusted by Eagle.
    token: str = ""
    timeout: float = 30.0
    # Parallel item/update requests and a ceiling on their rate, so Eagle stays responsive.
    concurrency: int = 4
    max_requests_per_second: float = 100.0
    retries: int = 4


@dataclass
class ModelConfig:
    enabled: bool = True
    revision: str = ""
    precision: Precision = "bf16"
    compile: bool = True
    batch_size: int = 0  # 0 = pick automatically (throughput probe within the VRAM budget)
    categories: list[str] = field(default_factory=list)  # categories written to Eagle
    # PixAI: threshold per category. Hydra: per-tag overrides on top of its calibration.
    thresholds: dict[str, float] = field(default_factory=dict)
    calibration: str = ""  # Hydra: metric used to derive per-tag thresholds
    implications: Implications = "off"  # Hydra: how e621 tag implications are applied
    gated: bool = False  # skip this model on frames the photo gate classifies as real photos


@dataclass
class GateConfig:
    enabled: bool = True
    revision: str = "097a6c1e9e62866a435810c883a45a7cb2c7077d"
    variant: str = "caformer_s36_v1.4"
    threshold: float = 0.65  # P(real photo) at or above this skips gated models


@dataclass
class VideoConfig:
    enabled: bool = True
    max_frames: int = 8
    aggregate: Aggregate = "mean"
    pmean_power: float = 4.0
    dedupe: bool = True
    dedupe_threshold: float = 3.0  # mean |diff| of 32x32 grey thumbnails, 0-255 scale


# e621 counting tags ("5 fingers", "biped"): describe anatomy the image obviously has.
COUNT_TAGS = ("3 fingers", "4 fingers", "5 fingers", "3 toes", "4 toes", "5 toes", "biped")


@dataclass
class MergeConfig:
    builtin_aliases: bool = True  # e621 -> Danbooru synonyms shipped in data/aliases.tsv
    aliases: dict[str, str] = field(default_factory=dict)
    blacklist: list[str] = field(default_factory=lambda: ["__has_tags__", *COUNT_TAGS])
    category_prefixes: bool = False
    max_tags: int = 0  # 0 = unlimited


@dataclass
class WriteConfig:
    mode: WriteMode = "replace"


@dataclass
class RuntimeConfig:
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    decode_workers: int = 0  # 0 = automatic
    frames_per_step: int = 16  # frames gathered per GPU step (models chunk it themselves)
    vram_budget: float = 0.85  # fraction of VRAM the batch-size probe may plan for


@dataclass
class PathsConfig:
    database: str = "lewdex.db"
    models: str = "models"
    cache: str = "cache"
    logs: str = "logs"


def _default_models() -> dict[str, ModelConfig]:
    return {
        "pixai": ModelConfig(
            revision="9fe10addf9326e292da8a85a98ea74cd91b41771",
            categories=["general", "character", "copyright", "style"],
            # Published recommended (per-category macro-F1) thresholds from the model card.
            thresholds={"general": 0.17, "character": 0.27, "style": 0.15, "copyright": 0.24,
                        "meta": 0.17, "rating": 0.41},
        ),
        "hydra": ModelConfig(
            revision="cfa9b0a1ffcf2b8df8553be7673210fd60fba23b",
            compile=False,
            categories=["general", "species", "character", "copyright"],
            calibration="f0.5@0.2",
            implications="collapse",
            gated=True,
        ),
    }


@dataclass
class Config:
    eagle: EagleConfig = field(default_factory=EagleConfig)
    models: dict[str, ModelConfig] = field(default_factory=_default_models)
    gate: GateConfig = field(default_factory=GateConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    base_dir: Path = PROJECT_ROOT
    source: Path | None = None

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def db_path(self) -> Path:
        return self.path(self.paths.database)

    @property
    def models_dir(self) -> Path:
        return self.path(self.paths.models)

    @property
    def cache_dir(self) -> Path:
        return self.path(self.paths.cache)

    @property
    def log_dir(self) -> Path:
        return self.path(self.paths.logs)

    def enabled_models(self) -> list[str]:
        return [k for k, m in self.models.items() if m.enabled]

    def copy(self) -> Config:
        return copy.deepcopy(self)

    def validate(self) -> None:
        problems: list[str] = []
        if not self.eagle.url.startswith(("http://", "https://")):
            problems.append(f"eagle.url must be an http(s) URL, got {self.eagle.url!r}")
        if self.eagle.concurrency < 1 or self.eagle.max_requests_per_second <= 0:
            problems.append("eagle.concurrency must be >= 1 and max_requests_per_second > 0")
        unknown = set(self.models) - {"pixai", "hydra"}
        if unknown:
            problems.append(f"unknown model(s) in [models]: {', '.join(sorted(unknown))}")
        if not self.enabled_models():
            problems.append("enable at least one tagging model")
        for name, m in self.models.items():
            for cat, thr in m.thresholds.items():
                if not 0.0 < thr <= 1.0:
                    problems.append(f"models.{name}.thresholds.{cat} must be in (0, 1], got {thr}")
            if m.batch_size < 0:
                problems.append(f"models.{name}.batch_size must be >= 0")
        if not 0.0 < self.gate.threshold <= 1.0:
            problems.append("gate.threshold must be in (0, 1]")
        if self.video.max_frames < 1:
            problems.append("video.max_frames must be >= 1")
        if self.video.pmean_power < 1.0:
            problems.append("video.pmean_power must be >= 1")
        if self.merge.max_tags < 0:
            problems.append("merge.max_tags must be >= 0")
        if not 0.1 <= self.runtime.vram_budget <= 0.98:
            problems.append("runtime.vram_budget must be between 0.1 and 0.98")
        if self.runtime.frames_per_step < 1:
            problems.append("runtime.frames_per_step must be >= 1")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def _check_type(value: Any, hint: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin is Literal:
        if value not in get_args(hint):
            raise ConfigError(f"{where}: expected one of {get_args(hint)}, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if origin is list:
        (item,) = get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_check_type(v, item, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        _, vt = get_args(hint)
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table, got {value!r}")
        return {str(k): _check_type(v, vt, f"{where}.{k}") for k, v in value.items()}
    return value


def _apply(obj: Any, data: dict[str, Any], where: str) -> None:
    hints = get_type_hints(type(obj))
    known = {f.name for f in fields(obj)} - {"base_dir", "source"}
    extra = set(data) - known
    if extra:
        raise ConfigError(f"Unknown setting(s) in [{where}]: {', '.join(sorted(extra))}")
    for key, value in data.items():
        hint = hints[key]
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{where}.{key}" if where else key)
        elif key == "models" and isinstance(value, dict):
            for mname, mdata in value.items():
                if not isinstance(mdata, dict):
                    raise ConfigError(f"[models.{mname}] must be a table")
                target = current.get(mname)
                if target is None:
                    raise ConfigError(f"Unknown model [models.{mname}] (known: pixai, hydra)")
                if "thresholds" in mdata and isinstance(mdata["thresholds"], dict):
                    merged = dict(target.thresholds)
                    merged.update(mdata["thresholds"])
                    mdata = {**mdata, "thresholds": merged}
                _apply(target, mdata, f"models.{mname}")
        else:
            setattr(obj, key, _check_type(value, hint, f"{where}.{key}" if where else key))


def load_config(path: Path | None = None) -> Config:
    """Defaults, overridden by ``path`` (or ``lewdex.toml`` next to the project, if any)."""
    cfg = Config()
    src = path
    if src is None:
        prepare_project_root()
    if src is None and DEFAULT_CONFIG_FILE.is_file():
        src = DEFAULT_CONFIG_FILE
    if src is not None:
        if not src.is_file():
            raise ConfigError(f"Config file not found: {src}")
        try:
            data = tomllib.loads(src.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{src} is not valid TOML: {exc}") from exc
        _apply(cfg, data, "")
        # The shipped template lives inside the package; its relative paths mean the project.
        is_template = src.resolve() == CONFIG_TEMPLATE.resolve()
        cfg.base_dir = PROJECT_ROOT if is_template else src.resolve().parent
        cfg.source = src.resolve()
    cfg.validate()
    return cfg
