"""Frame extraction and pre-flight validation.

Extraction strategy:

* one frame  -> single-seek ffmpeg call (``-ss`` before ``-i``);
* a range    -> *one* ffmpeg call for the whole range using the ``fps``
  filter, which is about an order of magnitude faster than looping.

Both paths funnel into the same validation, because a frame that cannot be
trusted must never be handed to a client.  If any frame in a batch fails
validation the whole call fails — a partial batch looks like a complete one
to the caller, which is exactly the silent degradation this server exists
to avoid.
"""

from __future__ import annotations

import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import FfmpegError, InputError, LimitError
from .ffmpeg import Tools, run_ffmpeg

#: File-header magic numbers, used to prove the bytes match the declared
#: MIME type.  A truncated or mislabelled frame is caught here, not by the
#: client.
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MIME_TYPES = {"jpeg": "image/jpeg", "png": "image/png"}

#: ffmpeg's mjpeg/png encoders write sequentially, so these bounds are
#: generous and only intended to stop a hostile/degenerate header.
MAX_SANE_DIMENSION = 65535

#: Smallest PNG the format allows: signature + IHDR + one IDAT + IEND.
#: Anything shorter than this is not an image, whatever its first bytes say.
MIN_PLAUSIBLE_FRAME_BYTES = 67


@dataclass(frozen=True)
class FrameFile:
    """A validated frame on disk."""

    n: int
    t: float
    path: Path
    data: bytes
    mime_type: str
    width: int | None
    height: int | None

    @property
    def size(self) -> int:
        return len(self.data)


def mime_for(image_format: str) -> str:
    try:
        return MIME_TYPES[image_format.lower()]
    except KeyError as exc:  # pragma: no cover - callers validate first
        raise InputError(
            f"Unknown image format {image_format!r}. "
            f"Valid values: {', '.join(sorted(MIME_TYPES))}."
        ) from exc


def sniff_format(data: bytes) -> str | None:
    """Identify image bytes by magic number; ``None`` if unrecognised."""
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(PNG_MAGIC):
        return "png"
    return None


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read width/height from a PNG IHDR chunk."""
    if len(data) < 24 or not data.startswith(PNG_MAGIC):
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG segments to the start-of-frame marker.

    Returns ``None`` when no SOF segment is found, which is the signature of a
    truncated or stub file that merely starts with the right magic bytes.
    """
    if not data.startswith(JPEG_MAGIC):
        return None
    index = 2
    end = len(data)
    while index + 4 <= end:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker == 0xD9 or marker == 0xDA:
            # End of image, or start of scan (image data) reached without a
            # frame header: not a decodable still image.
            return None
        segment_length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if segment_length < 2:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 9 > end:
                return None
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        index += 2 + segment_length
    return None


def image_dimensions(data: bytes, image_format: str) -> tuple[int, int] | None:
    if image_format == "png":
        return png_dimensions(data)
    if image_format == "jpeg":
        return jpeg_dimensions(data)
    return None


def validate_frame_file(
    path: Path,
    *,
    image_format: str,
    max_image_bytes: int,
    max_image_dimension: int,
    timestamp: float | None = None,
) -> FrameFile:
    """Validate one extracted frame, or raise with the exact reason.

    Checks, in order: file exists and is non-empty; header magic matches the
    declared MIME type; byte size within budget; pixel dimensions within
    budget.  Every failure names the file and the number that failed.
    """
    label = f"Frame {path.name}" + (
        f" (t={timestamp:g}s)" if timestamp is not None else ""
    )
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise FfmpegError(
            f"{label} does not exist: {path}. ffmpeg produced no such frame."
        ) from exc
    except OSError as exc:
        raise FfmpegError(f"{label} cannot be read: {exc} ({path}).") from exc

    if not data:
        raise FfmpegError(
            f"{label} is an empty file (0 bytes): {path}. Extraction failed; "
            f"no partial batch is returned."
        )

    detected = sniff_format(data)
    expected = mime_for(image_format)
    if detected is None:
        head = " ".join(f"{b:02X}" for b in data[:8])
        raise FfmpegError(
            f"{label} does not start with a JPEG/PNG header (first 8 bytes: "
            f"{head}): {path}. The file is corrupt or truncated; no partial "
            f"batch is returned."
        )
    if detected != image_format:
        raise FfmpegError(
            f"{label} is declared {expected} but its header is "
            f"{MIME_TYPES[detected]}: {path}. No partial batch is returned."
        )

    if len(data) > max_image_bytes:
        raise LimitError(
            f"{label} is {len(data) / (1024 * 1024):.1f} MB, over the "
            f"per-frame limit of {max_image_bytes / (1024 * 1024):.1f} MB: "
            f"{path}. Lower max_edge (this frame is "
            f"{_dims_text(data, image_format)}) or use format=jpeg."
        )

    if len(data) < MIN_PLAUSIBLE_FRAME_BYTES:
        raise FfmpegError(
            f"{label} is only {len(data)} bytes, too small to be a complete "
            f"frame: {path}. The file is truncated or not a real image; no "
            f"partial batch is returned."
        )

    dims = image_dimensions(data, image_format)
    if dims is None:
        raise FfmpegError(
            f"{label} has an incomplete image header (no dimensions found): "
            f"{path}. The file is truncated or corrupt; no partial batch is "
            f"returned."
        )
    width, height = dims
    if width > max_image_dimension or height > max_image_dimension:
        raise LimitError(
            f"{label} is {width}x{height}, over the per-frame edge limit of "
            f"{max_image_dimension} pixels: {path}. Lower max_edge."
        )
    if (
        width == 0
        or height == 0
        or width > MAX_SANE_DIMENSION
        or height > MAX_SANE_DIMENSION
    ):
        raise FfmpegError(
            f"{label} reports untrustworthy dimensions ({width}x{height}): "
            f"{path}. No partial batch is returned."
        )
    return FrameFile(
        n=-1,
        t=timestamp if timestamp is not None else 0.0,
        path=path,
        data=data,
        mime_type=expected,
        width=width,
        height=height,
    )


def _dims_text(data: bytes, image_format: str) -> str:
    dims = image_dimensions(data, image_format)
    return f"{dims[0]}x{dims[1]}" if dims else "unknown"


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------
def scale_filter(max_edge: int) -> str:
    """Scale so the long edge is at most ``max_edge``, height even.

    ``-2`` keeps the aspect ratio and rounds to an even number, which the
    JPEG encoder requires.
    """
    return f"scale='min({max_edge},iw)':-2"


def encode_args(image_format: str, quality: int) -> list[str]:
    if image_format == "png":
        return ["-compression_level", "6"]
    from .budget import qscale_for_quality

    return ["-q:v", str(qscale_for_quality(quality))]


def extract_range(
    tools: Tools,
    *,
    video: Path,
    start: float,
    interval: float,
    count: int,
    max_edge: int,
    image_format: str,
    quality: int,
    out_dir: Path,
    timeout: float | None = None,
) -> list[Path]:
    """Extract a whole range in one ffmpeg call.

    Produces up to ``count`` frames at ``start + i * interval``.  The count is
    verified against what appeared on disk: the timestamps this server reports
    are computed from the grid, and that convention is only trustworthy if the
    number of files matches the number of grid points claimed.

    Fewer files than requested is legitimate only when the range runs to the
    very end of the input, where the decoder stops producing samples.  The
    caller (:mod:`mcp_video_frames.core`) reconciles the plan with the actual
    count and reports the frames that exist; a *larger* count is always an
    error, because it would mean the grid itself is wrong.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if image_format == "png" else "jpg"
    # Measured on real builds: `-ss S -t T -vf fps=1/I` (phase 0) emits samples
    # at S, S+I, ... while the sample time is *below* T, so a window of exactly
    # (count-1)*I drops the last requested sample.  One extra frame plus a
    # millisecond of slack forces the final grid point into range;
    # `-frames:v count` then caps the batch, and the produced file count is
    # verified below rather than assumed.
    span = interval * max(count, 1) + 1e-3
    args = [
        "-ss",
        f"{start:.6f}",
        "-i",
        str(video),
        "-t",
        f"{span:.6f}",
        "-an",
        "-vf",
        f"fps=1/{interval:.6f},{scale_filter(max_edge)}",
        "-frames:v",
        str(count),
        *encode_args(image_format, quality),
        "-start_number",
        "0",
        str(out_dir / f"%05d.{ext}"),
    ]
    run_ffmpeg(tools, args, timeout=timeout)
    produced = sorted(out_dir.glob(f"*.{ext}"))
    if len(produced) > count:
        raise FfmpegError(
            f"Expected at most {count} frames but got {len(produced)} "
            f"({video.name}, start {start:g}s, interval {interval:g}s). Extra "
            f"frames mean the sampling grid does not match the plan, which "
            f"makes the timestamps untrustworthy, so the whole call fails "
            f"rather than being truncated.",
            detail=f"Output directory: {out_dir}",
        )
    if not produced:
        raise FfmpegError(
            f"No frame was extracted from {start:g}s on at interval "
            f"{interval:g}s ({video.name}). The range may lie beyond the "
            f"video's duration, or the file may carry no video stream.",
            detail=f"Output directory: {out_dir}",
        )
    return produced


def extract_single(
    tools: Tools,
    *,
    video: Path,
    t: float,
    max_edge: int,
    image_format: str,
    quality: int,
    out_dir: Path,
    timeout: float | None = None,
) -> Path:
    """Extract one frame.  ``-ss`` precedes ``-i`` for a fast seek."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if image_format == "png" else "jpg"
    target = out_dir / f"single.{ext}"
    args = [
        "-ss",
        f"{t:.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-vf",
        scale_filter(max_edge),
        *encode_args(image_format, quality),
        str(target),
    ]
    run_ffmpeg(tools, args, timeout=timeout)
    if not target.is_file():
        raise FfmpegError(
            f"No frame was extracted at t={t:g}s ({video.name}). That moment "
            f"may lie beyond the video's duration, or the file may carry no "
            f"video stream.",
            detail=f"Output path: {target}",
        )
    return target


def extraction_workspace(parent: Path, tag: str) -> Path:
    """Scratch directory for one extraction, inside the cache entry.

    Created explicitly (``mkdir``) rather than via ``mkdtemp`` so the path is
    predictable; the per-video lock already guarantees only one writer, and
    the directory is removed in a ``finally``.
    """
    parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}"
    path = parent / f"extract-{tag}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_workspace(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def check_video_readable(path: Path) -> None:
    if not path.exists():
        raise InputError(
            f"Video file does not exist: {path}. Pass the absolute path of a "
            f"video file."
        )
    if not path.is_file():
        raise InputError(
            f"Path is not a file: {path}. Pass a video file, not a directory."
        )
    if not os.access(path, os.R_OK):
        raise InputError(f"Video file is not readable (permission denied): {path}.")
    try:
        if path.stat().st_size == 0:
            raise InputError(f"Video file is empty (0 bytes): {path}.")
    except OSError as exc:
        raise InputError(f"Cannot read video file information: {path} ({exc}).") from exc
