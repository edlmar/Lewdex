"""Async client for the Eagle 4 Web API v2 (http://localhost:41595/api/v2/...).

Verified against Eagle 4.0.0 (build 20260401):
* ``POST /api/v2/item/get`` with a JSON body; ``fields`` must be a JSON array (the documented
  comma-separated ``GET ?fields=`` fails in this build with "fields.forEach is not a function").
  ``offset`` counts rows (unlike v1, where it was a page index), ``limit`` is at most 1000.
* ``isSelected: true`` returns the current selection, ``folders: [...]`` filters by folder.
* ``POST /api/v2/item/update`` replaces the item's whole tag list; there is no batch update.
Requests from localhost need no token; ``token`` is appended when configured (remote Eagle).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import EagleConfig
from .errors import EagleApiError, EagleUnavailable
from .log import log

ITEM_FIELDS = ["id", "name", "ext", "size", "tags", "isDeleted"]


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    ext: str
    size: int
    tags: tuple[str, ...]

    def path(self, library: Path) -> Path:
        ext = self.ext.lower()
        return library / "images" / f"{self.id}.info" / (f"{self.name}.{ext}" if ext else self.name)


@dataclass(frozen=True)
class Folder:
    id: str
    name: str
    path: str  # "Parent/Child"
    children: tuple[str, ...]  # descendant folder ids, including this one
    image_count: int


@dataclass(frozen=True)
class Scope:
    kind: str = "library"  # "library" | "folder" | "selected"
    folders: tuple[str, ...] = ()  # folder ids or names/paths for kind == "folder"
    subfolders: bool = True

    def describe(self) -> str:
        if self.kind == "folder":
            sub = " (+subfolders)" if self.subfolders else ""
            return f"folder {', '.join(self.folders)}{sub}"
        return "current Eagle selection" if self.kind == "selected" else "whole library"


class RateLimiter:
    """Token bucket: at most ``rate`` acquisitions per second, bursts up to ``burst``."""

    def __init__(self, rate: float, burst: int = 8) -> None:
        self.rate = rate
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.stamp = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
                self.stamp = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.rate)


class EagleClient:
    def __init__(self, cfg: EagleConfig,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.cfg = cfg
        self.base = cfg.url.rstrip("/")
        self._transport = transport  # tests inject a mock transport
        self._client: httpx.AsyncClient | None = None
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._limiter = RateLimiter(cfg.max_requests_per_second)

    async def __aenter__(self) -> EagleClient:
        limits = httpx.Limits(max_connections=self.cfg.concurrency + 4,
                              max_keepalive_connections=self.cfg.concurrency + 4)
        params = {"token": self.cfg.token} if self.cfg.token else None
        self._client = httpx.AsyncClient(base_url=self.base, timeout=self.cfg.timeout,
                                         limits=limits, params=params, transport=self._transport)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- transport ---------------------------------------------------------------------------
    async def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                       *, retries: int | None = None) -> Any:
        assert self._client is not None, "use 'async with EagleClient(...)'"
        attempts = (self.cfg.retries if retries is None else retries) + 1
        delay = 0.25
        for attempt in range(attempts):
            try:
                if method == "GET":
                    resp = await self._client.get(path)
                else:
                    resp = await self._client.post(path, json=body or {})
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request,
                                                response=resp)
                if resp.status_code >= 400:  # client errors are not worth retrying
                    raise EagleApiError(f"Eagle API {path} answered HTTP {resp.status_code}: "
                                        f"{resp.text[:200]}")
                payload = resp.json()
            except httpx.ConnectError as exc:
                if attempt + 1 >= attempts:
                    raise EagleUnavailable(self.base, "connection refused") from exc
            except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError,
                    httpx.HTTPStatusError) as exc:
                if attempt + 1 >= attempts:
                    raise EagleApiError(
                        f"Eagle API {path} failed after {attempts} attempts: {exc}",
                        "Eagle may be busy; the write is retried on the next run.") from exc
            else:
                if isinstance(payload, dict) and payload.get("status") == "error":
                    raise EagleApiError(f"Eagle API {path} returned an error: "
                                        f"{payload.get('message') or payload}")
                return payload.get("data") if isinstance(payload, dict) else payload
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)
        raise AssertionError("unreachable")

    # -- library -----------------------------------------------------------------------------
    async def library_info(self) -> tuple[Path, str, str]:
        """(library folder, library name, Eagle version)."""
        info = await self._request("GET", "/api/v2/library/info", retries=1)
        app = await self._request("GET", "/api/v2/app/info", retries=1)
        path = info.get("path") or (info.get("library") or {}).get("path")
        if not path:
            raise EagleApiError("Eagle answered but reported no open library",
                                "Open a library in Eagle and try again.")
        version = f"{app.get('version', '?')} build {app.get('buildVersion', '?')}"
        return Path(path), str(info.get("name", "")), version

    async def folders(self) -> list[Folder]:
        data = await self._request("POST", "/api/v2/folder/get", {"offset": 0, "limit": 1000})
        roots = data.get("data", []) if isinstance(data, dict) else data or []
        out: list[Folder] = []

        def walk(node: dict[str, Any], prefix: str) -> list[str]:
            path = f"{prefix}/{node['name']}" if prefix else str(node["name"])
            ids = [str(node["id"])]
            for child in node.get("children") or []:
                ids += walk(child, path)
            out.append(Folder(str(node["id"]), str(node["name"]), path, tuple(ids),
                              int(node.get("imageCount") or 0)))
            return ids

        for root in roots:
            walk(root, "")
        return sorted(out, key=lambda f: f.path.lower())

    async def resolve_folders(self, scope: Scope) -> list[str]:
        all_folders = await self.folders()
        by_key: dict[str, Folder] = {}
        for folder in all_folders:
            by_key[folder.id] = folder
            by_key.setdefault(folder.path.lower(), folder)
            by_key.setdefault(folder.name.lower(), folder)
        ids: list[str] = []
        for want in scope.folders:
            f = by_key.get(want) or by_key.get(want.lower())
            if f is None:
                raise EagleApiError(f"No Eagle folder named or with id {want!r}")
            ids += list(f.children) if scope.subfolders else [f.id]
        return list(dict.fromkeys(ids))

    async def list_items(self, scope: Scope,
                         progress: Callable[[int, int], None] | None = None) -> list[Item]:
        body: dict[str, Any] = {"fields": ITEM_FIELDS}
        if scope.kind == "selected":
            body["isSelected"] = True
        elif scope.kind == "folder":
            body["folders"] = await self.resolve_folders(scope)
            if not body["folders"]:
                return []
        first = await self._request("POST", "/api/v2/item/get",
                                    {**body, "offset": 0, "limit": 1000})
        rows: list[dict[str, Any]] = list(first.get("data", []))
        total = int(first.get("total", len(rows)))
        if progress:
            progress(len(rows), total)
        offsets = list(range(len(rows), total, 1000)) if rows else []

        async def page(off: int) -> list[dict[str, Any]]:
            async with self._sem:
                d = await self._request("POST", "/api/v2/item/get",
                                        {**body, "offset": off, "limit": 1000})
            return list(d.get("data", []))

        for i in range(0, len(offsets), 8):
            for chunk in await asyncio.gather(*(page(o) for o in offsets[i:i + 8])):
                rows += chunk
            if progress:
                progress(len(rows), total)
        seen: set[str] = set()
        items: list[Item] = []
        for r in rows:
            iid = str(r.get("id", ""))
            if not iid or iid in seen or r.get("isDeleted"):
                continue
            seen.add(iid)
            items.append(Item(iid, str(r.get("name", "")), str(r.get("ext", "")),
                              int(r.get("size") or 0), tuple(str(t) for t in r.get("tags") or ())))
        return items

    # -- tags --------------------------------------------------------------------------------
    async def get_tags(self, ids: Sequence[str]) -> dict[str, list[str]]:
        """Current tags straight from Eagle (not the possibly stale listing)."""
        out: dict[str, list[str]] = {}
        for i in range(0, len(ids), 1000):
            chunk = list(ids[i:i + 1000])
            async with self._sem:
                d = await self._request("POST", "/api/v2/item/get",
                                        {"ids": chunk, "fields": ["id", "tags", "isDeleted"],
                                         "limit": 1000})
            for r in d.get("data", []):
                if not r.get("isDeleted"):
                    out[str(r["id"])] = [str(t) for t in r.get("tags") or ()]
        return out

    async def set_tags(self, item_id: str, tags: Sequence[str]) -> None:
        await self._limiter.acquire()
        async with self._sem:
            await self._request("POST", "/api/v2/item/update", {"id": item_id, "tags": list(tags)})


async def probe(cfg: EagleConfig) -> tuple[Path, str, str]:
    async with EagleClient(cfg) as client:
        return await client.library_info()


def probe_sync(cfg: EagleConfig) -> tuple[Path, str, str]:
    return asyncio.run(probe(cfg))


def log_library(path: Path, name: str, version: str) -> None:
    log.info("Eagle %s, library %r at %s", version, name, path)
