"""Turning per-model probabilities into one tag list per item.

Order of operations for every item (all pure CPU work, nothing here touches the GPU):

1. aggregate each model's per-frame vectors (videos/animations) into one vector;
2. per model: threshold (per category for PixAI, calibrated per tag for Hydra), apply e621
   implications for Hydra, keep only the configured categories;
3. canonicalise names: lowercase, underscores -> spaces, e621 -> Danbooru aliases;
4. drop blacklisted names, union across models keeping the highest confidence, sort, cap.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources

import numpy as np

_WS = re.compile(r"\s+")

# Danbooru emoticon tags keep their underscores ("^_^" would otherwise become "^ ^").
KAOMOJI = frozenset({"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
                     "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"})


def normalize(tag: str) -> str:
    """Canonical form, used both to compare and to write: lowercase, spaces for underscores."""
    t = tag.strip().lower()
    if t in KAOMOJI:
        return t
    return _WS.sub(" ", t.replace("_", " ")).strip()


class AliasMap:
    """Normalized name -> canonical name, with chains resolved (a->b, b->c gives a->c)."""

    def __init__(self, pairs: Iterable[tuple[str, str]] = ()) -> None:
        raw: dict[str, str] = {}
        for src, dst in pairs:
            s, d = normalize(src), normalize(dst)
            if s and d and s != d:
                raw[s] = d
        resolved: dict[str, str] = {}
        for s in raw:
            seen = {s}
            d = raw[s]
            while d in raw and d not in seen:
                seen.add(d)
                d = raw[d]
            resolved[s] = d
        self._map = resolved

    def __len__(self) -> int:
        return len(self._map)

    def __call__(self, name: str) -> str:
        return self._map.get(name, name)

    @classmethod
    def builtin(cls) -> AliasMap:
        return cls(load_alias_pairs())


def load_alias_pairs() -> list[tuple[str, str]]:
    """Shipped e621 -> Danbooru synonyms (data/aliases.tsv, built by scripts/build_aliases.py)."""
    try:
        text = resources.files("lewdex").joinpath("data/aliases.tsv").read_text("utf-8")
    except FileNotFoundError:
        return []
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


class Blacklist:
    """Exact names and glob patterns (``rating:*``), compared on normalized names."""

    def __init__(self, entries: Iterable[str]) -> None:
        self.exact: set[str] = set()
        patterns: list[str] = []
        for e in entries:
            n = normalize(e)
            if not n:
                continue
            if any(ch in n for ch in "*?["):
                patterns.append(fnmatch.translate(n))
            else:
                self.exact.add(n)
        self._rx = re.compile("|".join(patterns)) if patterns else None

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        n = normalize(name)
        return n in self.exact or (self._rx is not None and self._rx.match(n) is not None)


@dataclass
class LabelSpace:
    """A model's vocabulary and default decision rule."""

    names: list[str]  # raw model tag names (underscores)
    categories: list[str]  # model-native category per tag
    thresholds: np.ndarray  # float32 per tag; inf means the tag is never emitted
    implies: list[list[int]] | None = None  # direct implications (indices), Hydra only

    def __post_init__(self) -> None:
        n = len(self.names)
        if len(self.categories) != n or len(self.thresholds) != n:
            raise ValueError("label space arrays differ in length")

    def closure(self) -> list[np.ndarray]:
        """Transitive implication closure per tag (including the tag itself)."""
        assert self.implies is not None
        memo: dict[int, frozenset[int]] = {}

        def visit(i: int, stack: frozenset[int]) -> frozenset[int]:
            if i in memo:
                return memo[i]
            acc = {i}
            for j in self.implies[i]:  # type: ignore[index]
                if j not in stack:
                    acc |= visit(j, stack | {i})
            result = frozenset(acc)
            memo[i] = result
            return result

        return [np.fromiter(sorted(visit(i, frozenset())), dtype=np.int32)
                for i in range(len(self.names))]


@dataclass(frozen=True)
class Candidate:
    name: str  # canonical display name
    confidence: float
    category: str
    model: str


class ModelDecoder:
    """Threshold + implications + category filter + canonical names for one model."""

    def __init__(self, model: str, space: LabelSpace, *, categories: Sequence[str],
                 implications: str, aliases: AliasMap, blacklist: Blacklist) -> None:
        self.model = model
        self.space = space
        self.thresholds = space.thresholds.astype(np.float32)
        self.display = [aliases(normalize(n)) for n in space.names]
        raw_norm = [normalize(n) for n in space.names]
        cats = set(categories)
        self.allowed = np.array(
            [c in cats and d not in blacklist and r not in blacklist
             for c, d, r in zip(space.categories, self.display, raw_norm, strict=True)],
            dtype=bool)
        if implications not in ("off", "inherit", "preserve", "collapse"):
            raise ValueError(f"unsupported implications mode {implications!r}")
        self.implications = implications if space.implies is not None else "off"
        self._closure = space.closure() if self.implications != "off" else None

    def decode(self, probs: np.ndarray) -> list[Candidate]:
        passing = np.flatnonzero(probs >= self.thresholds)
        if self._closure is not None and self.implications == "collapse":
            # Most specific only: drop a tag when another written tag implies it
            # ("wolf" makes "canine", "canid" and "mammal" redundant).
            keep = passing[self.allowed[passing]].tolist()
            implied: set[int] = set()
            for i in keep:
                implied.update(j for j in self._closure[i].tolist() if j != i)
            return [Candidate(self.display[i], float(probs[i]), self.space.categories[i],
                              self.model) for i in keep if i not in implied]
        if self._closure is None:
            keep = passing[self.allowed[passing]]
            return [Candidate(self.display[i], float(probs[i]), self.space.categories[i],
                              self.model) for i in keep.tolist()]
        # Implications (Hydra's "inherit"): every tag implied by a passing tag qualifies;
        # excluded categories are dropped; with "inherit", implied tags take the highest
        # confidence of the qualified tags that imply them (this matches hydra.classification).
        qualified: set[int] = set()
        for i in passing.tolist():
            qualified.update(self._closure[i].tolist())
        kept = [j for j in qualified if self.allowed[j]]
        conf = {j: float(probs[j]) for j in kept}
        if self.implications == "inherit":
            for i in kept:
                p = float(probs[i])
                for j in self._closure[i].tolist():
                    if j in conf and conf[j] < p:
                        conf[j] = p
        return [Candidate(self.display[j], conf[j], self.space.categories[j], self.model)
                for j in kept]


def aggregate(frames: np.ndarray, mode: str, power: float = 4.0) -> np.ndarray:
    """(n_frames, n_tags) -> (n_tags,): mean, max, or power mean (between the two)."""
    if frames.ndim == 1:
        return frames
    if frames.shape[0] == 1:
        return np.asarray(frames[0])
    if mode == "max":
        return np.asarray(frames.max(axis=0))
    if mode == "pmean":
        return np.asarray(np.power(np.mean(np.power(frames, power), axis=0), 1.0 / power))
    return np.asarray(frames.mean(axis=0))


PREFIX_NAMES = {"style": "artist"}


def merge(per_model: Sequence[list[Candidate]], *, max_tags: int = 0,
          prefixes: bool = False, blacklist: Blacklist | None = None) -> list[Candidate]:
    """Union across models by canonical name, highest confidence wins; sorted, capped."""
    best: dict[str, Candidate] = {}
    for candidates in per_model:
        for c in candidates:
            if blacklist is not None and c.name in blacklist:
                continue
            prev = best.get(c.name)
            if prev is None:
                best[c.name] = c
            elif c.confidence > prev.confidence:
                # Keep the first model's category (Danbooru before e621) for stable prefixes.
                best[c.name] = Candidate(c.name, c.confidence, prev.category, c.model)
    ordered = sorted(best.values(), key=lambda c: (-c.confidence, c.name))
    if max_tags:
        ordered = ordered[:max_tags]
    if prefixes:
        ordered = [Candidate(format_tag(c, True), c.confidence, c.category, c.model)
                   for c in ordered]
    return ordered


def format_tag(c: Candidate, prefixes: bool) -> str:
    if not prefixes or c.category == "general":
        return c.name
    return f"{PREFIX_NAMES.get(c.category, c.category)}:{c.name}"


def tagset_fingerprint(tags: Sequence[str], extra: str = "") -> str:
    h = hashlib.blake2b(digest_size=12)
    for t in sorted(tags):
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    h.update(extra.encode("utf-8"))
    return h.hexdigest()


def plan_final_tags(current: Sequence[str], new: Sequence[str], *, managed: Iterable[str],
                    mode: str) -> tuple[list[str], list[str], list[str]]:
    """Compute the tag list to write back to Eagle.

    Returns ``(final, added, removed)``. Existing tags keep their order; new ones are appended.
    ``mode="replace"`` makes the list exactly the new tags (everything else is removed).
    ``mode="add"`` never removes anything. ``mode="sync"`` also removes tags this tool added
    earlier that it no longer predicts; tags a user added are never removed.
    """
    new_norm = {normalize(t): t for t in new}
    remove: set[str] = set()
    if mode == "sync":
        remove |= {normalize(t) for t in managed} - set(new_norm)
    elif mode == "replace":
        remove |= {normalize(t) for t in current}
    remove -= set(new_norm)  # a predicted tag is never removed
    final: list[str] = []
    seen: set[str] = set()
    removed: list[str] = []
    for t in current:
        n = normalize(t)
        if n in remove:
            removed.append(t)
            continue
        if n in seen:
            continue
        seen.add(n)
        final.append(t)
    added: list[str] = []
    for n, t in new_norm.items():
        if n not in seen:
            seen.add(n)
            final.append(t)
            added.append(t)
    return final, added, removed
