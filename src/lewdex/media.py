"""Reading, hashing and decoding media into upright 8-bit sRGB frames (pyvips + PyAV).

Every frame handed to a model adapter is a 3-band uchar sRGB ``pyvips.Image`` held in memory:
embedded ICC profiles are honoured (relative intent + black-point compensation, as Hydra's own
loader does), transparency is flattened onto white (both taggers were trained that way) and
EXIF orientation is applied. Each adapter then does its own resize from that frame.

Animations (GIF, animated WebP/PNG/AVIF) contribute up to ``max_frames`` evenly spaced frames;
videos contribute up to ``max_frames`` *keyframes* (the decoder skips everything else), spread
evenly over the duration. Near-duplicate frames are dropped before they reach the GPU.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyvips
import xxhash

from .config import VideoConfig

# Hydra's reference loader disables the operation cache; with many worker threads the cache
# only costs memory. Each worker is a thread, so keep libvips' own thread pool small.
pyvips.cache_set_max(0)
pyvips.concurrency_set(max(1, int(os.environ.get("LEWDEX_VIPS_THREADS", "2"))))

IMAGE_EXTS = frozenset({"jpg", "jpeg", "jfif", "jpe", "png", "apng", "webp", "bmp", "tif", "tiff",
                        "avif", "heic", "heif", "jxl", "gif"})
VIDEO_EXTS = frozenset({"mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg",
                        "ts", "m2ts", "3gp", "ogv"})
FULL_HASH_LIMIT = 256 * 1024 * 1024  # larger files are hashed from head + tail + size
SAMPLE_BYTES = 16 * 1024 * 1024
MAX_VIDEO_EDGE = 1536  # keyframes are downscaled to this long edge right after decoding


def media_kind(ext: str) -> str | None:
    e = ext.lower().lstrip(".")
    if e in IMAGE_EXTS:
        return "image"
    if e in VIDEO_EXTS:
        return "video"
    return None


FRAME_PIPELINE_VERSION = 1  # bump when frame selection/decoding for animations/videos changes


def sampling_signature(media: str, video: VideoConfig) -> str:
    """Cache-key component describing how frames were chosen (stills are always ``img``)."""
    if media == "img":
        return "img"
    tag = "anim" if media == "anim" else "vid"
    return (f"{tag}{FRAME_PIPELINE_VERSION}/n{video.max_frames}"
            f"/d{int(video.dedupe)}{video.dedupe_threshold:g}")


@dataclass
class FileData:
    data: bytes | None  # full contents when the file was read completely
    content_hash: bytes  # xxh3-128
    size: int
    mtime_ns: int


def read_and_hash(path: Path) -> FileData:
    st = path.stat()
    if st.st_size <= FULL_HASH_LIMIT:
        data = path.read_bytes()
        return FileData(data, xxhash.xxh3_128_digest(data), st.st_size, st.st_mtime_ns)
    h = xxhash.xxh3_128()
    with path.open("rb") as fh:
        h.update(fh.read(SAMPLE_BYTES))
        fh.seek(-SAMPLE_BYTES, os.SEEK_END)
        h.update(fh.read(SAMPLE_BYTES))
    h.update(str(st.st_size).encode())
    return FileData(None, b"S" + h.digest()[1:], st.st_size, st.st_mtime_ns)


def to_srgb(img: pyvips.Image) -> pyvips.Image:
    """Upright, alpha-flattened-on-white, 3-band uchar sRGB copy held in memory."""
    try:
        img = img.icc_transform("srgb", embedded=True, intent="relative",
                                black_point_compensation=True, depth=8)
    except pyvips.Error:
        img = img.colourspace("srgb")
    if img.hasalpha():
        img = img.flatten(background=255)
    img = img.autorot()
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands > 3:
        img = img.extract_band(0, n=3)
    return img.copy_memory()


def _thumb(img: pyvips.Image) -> np.ndarray:
    grey = img.colourspace("b-w") if img.bands >= 3 else img
    small = grey.resize(32 / grey.width, vscale=32 / grey.height, kernel="linear")
    arr = np.ndarray(buffer=small.write_to_memory(), dtype=np.uint8,
                     shape=(small.height, small.width, small.bands))
    return arr[..., 0].astype(np.float32)


def dedupe(frames: list[pyvips.Image], threshold: float) -> list[pyvips.Image]:
    kept: list[pyvips.Image] = []
    thumbs: list[np.ndarray] = []
    for f in frames:
        t = _thumb(f)
        if any(np.abs(t - k).mean() < threshold for k in thumbs):
            continue
        kept.append(f)
        thumbs.append(t)
    return kept


def _even_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [n // 2]


@dataclass
class Decoded:
    frames: list[pyvips.Image]
    media: str  # "img" | "anim" | "vid"
    dropped: int = 0  # near-duplicate frames removed


def _decode_damaged(data: bytes, exc: Exception) -> pyvips.Image:
    """Pillow fallback for files libvips rejects (truncated or corrupt JPEGs): keep what decodes."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(data)) as im:
            rgb = np.asarray(im.convert("RGB"))
    except Exception:
        raise exc from None
    h, w, _ = rgb.shape
    return pyvips.Image.new_from_memory(np.ascontiguousarray(rgb).data, w, h, 3, "uchar")


def looks_like_video(data: bytes) -> bool:
    """MP4/MOV, Matroska/WebM or AVI content, whatever the file extension says."""
    head = data[:16]
    return (head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3"
            or (head[:4] == b"RIFF" and head[8:12] == b"AVI "))


def decode_image(data: bytes, video: VideoConfig) -> Decoded:
    try:
        img = pyvips.Image.new_from_buffer(data, "")
        img = img.copy_memory()  # decode now: corrupt data fails here, not later
    except pyvips.Error as exc:
        return Decoded([to_srgb(_decode_damaged(data, exc))], "img")
    pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
    if pages <= 1:
        return Decoded([to_srgb(img)], "img")
    if not video.enabled:
        return Decoded([to_srgb(img)], "img")  # first frame only, treated like a still
    strip = pyvips.Image.new_from_buffer(data, "", n=-1)
    page_h = strip.get("page-height") if strip.get_typeof("page-height") else strip.height
    n = max(1, strip.height // page_h)
    frames = [to_srgb(strip.crop(0, i * page_h, strip.width, page_h))
              for i in _even_indices(n, video.max_frames)]
    before = len(frames)
    if video.dedupe:
        frames = dedupe(frames, video.dedupe_threshold)
    return Decoded(frames, "anim", before - len(frames))


def _av() -> Any:
    try:
        import av
    except Exception as exc:  # DLL load failures surface as ImportError/OSError
        from .errors import VideoBackendError
        raise VideoBackendError(str(exc)) from exc
    return av


def _frame_to_vips(frame: Any) -> pyvips.Image:
    w, h = frame.width, frame.height
    scale = min(1.0, MAX_VIDEO_EDGE / max(w, h))
    if scale < 1.0:
        frame = frame.reformat(width=max(2, int(w * scale)) // 2 * 2,
                               height=max(2, int(h * scale)) // 2 * 2, format="rgb24",
                               interpolation="AREA")
    else:
        frame = frame.reformat(format="rgb24")
    arr = frame.to_ndarray()
    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
    if rotation:
        arr = np.rot90(arr, k=rotation // 90).copy()  # display matrix rotation (phone videos)
    img = pyvips.Image.new_from_memory(np.ascontiguousarray(arr).data, arr.shape[1], arr.shape[0],
                                       3, "uchar")
    return img.copy(interpretation="srgb").copy_memory()


def _open_video(av: Any, source: Path | bytes) -> Any:
    container = av.open(io.BytesIO(source) if isinstance(source, bytes) else str(source), mode="r")
    if container.streams.video:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "NONKEY"  # the decoder only produces keyframes
    return container


def _seek_keyframes(av: Any, container: Any, k: int) -> dict[int, pyvips.Image]:
    """One keyframe at or before each of ``k`` evenly spaced timestamps."""
    stream = container.streams.video[0]
    tb = stream.time_base or Fraction(1, 1000)
    start = stream.start_time or 0
    if stream.duration:
        duration = float(stream.duration * tb)
    elif container.duration:
        duration = container.duration / 1_000_000
    else:
        return {}
    picked: dict[int, pyvips.Image] = {}
    for i in range(k):
        target = start + int(((i + 0.5) / k * duration) / tb)
        container.seek(target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            key = frame.pts if frame.pts is not None else -1 - i
            if key not in picked:
                picked[key] = _frame_to_vips(frame)
            break
    return picked


def _scan_keyframes(container: Any, k: int) -> dict[int, pyvips.Image]:
    """Fallback for unseekable streams: decode keyframes in order, keep ``k`` spread out."""
    stream = container.streams.video[0]
    keyframes: list[Any] = []
    for frame in container.decode(stream):
        keyframes.append(frame)
        if len(keyframes) >= 256:
            break
    return {(f.pts if f.pts is not None else i): _frame_to_vips(f)
            for i, f in ((i, keyframes[i]) for i in _even_indices(len(keyframes), k))}


def decode_video(source: Path | bytes, video: VideoConfig) -> Decoded:
    av = _av()
    k = video.max_frames
    picked: dict[int, pyvips.Image] = {}
    with _open_video(av, source) as container:
        if not container.streams.video:
            return Decoded([], "vid")
        try:
            picked = _seek_keyframes(av, container, k)
        except (av.error.FFmpegError, ValueError):
            picked = {}
    if not picked:
        with _open_video(av, source) as container:
            picked = _scan_keyframes(container, k)
    frames = [picked[p] for p in sorted(picked)]
    before = len(frames)
    if video.dedupe:
        frames = dedupe(frames, video.dedupe_threshold)
    return Decoded(frames, "vid", before - len(frames))


def decode_file(path: Path, fd: FileData, kind: str, video: VideoConfig) -> Decoded:
    if kind == "video":
        return decode_video(fd.data if fd.data is not None else path, video)
    data = fd.data if fd.data is not None else path.read_bytes()
    if looks_like_video(data):  # e.g. an MP4 saved as .gif
        return decode_video(data, video)
    return decode_image(data, video)


def vips_to_numpy(img: pyvips.Image) -> np.ndarray:
    return np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                      shape=(img.height, img.width, img.bands))
