"""Container metadata via ffprobe.

``basic`` detail reads headers only and is effectively instantaneous; it is
what the MCP tool returns by default and what the CLI ``info`` subcommand
prints.  Both call :func:`probe_basic`, so they cannot drift apart.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import FfmpegError
from .ffmpeg import Tools, run_ffprobe

#: ffprobe asks for these with ``-show_entries``; the rest (chapters,
#: format tags) is cheap enough to read wholesale.
PROBE_TIMEOUT = 120.0

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
HDR_PRIMARIES = {"bt2020"}


def _fraction_to_float(value: Any) -> float:
    if value in (None, "", "N/A", "0/0"):
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _tags_language(tags: Any) -> str:
    if not isinstance(tags, dict):
        return ""
    for key in ("language", "LANGUAGE", "lang"):
        value = tags.get(key)
        if value:
            return str(value)
    return ""


def probe_raw(tools: Tools, video: Path) -> dict[str, Any]:
    """Run ffprobe and return its JSON document."""
    if not tools.has_ffprobe:
        raise FfmpegError(
            "ffprobe is not available in this environment, so JSON metadata "
            "probing is not possible. Use the banner fallback path instead."
        )
    output = run_ffprobe(
        tools,
        [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(video),
        ],
        timeout=PROBE_TIMEOUT,
    )
    try:
        document = json.loads(output)
    except ValueError as exc:
        raise FfmpegError(
            f"ffprobe output is not valid JSON, so the metadata for "
            f"{video.name} cannot be parsed.",
            detail=output[:2000],
        ) from exc
    if not isinstance(document, dict):
        raise FfmpegError(
            f"ffprobe output has an unexpected shape (top level is not an "
            f"object), so {video.name} cannot be parsed.",
            detail=output[:2000],
        )
    return document


def _best_video_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        return None
    # Prefer attached pictures never; prefer real video streams by area.
    real = [s for s in videos if s.get("disposition", {}).get("attached_pic") != 1]
    pool = real or videos
    return max(pool, key=lambda s: _to_int(s.get("width")) * _to_int(s.get("height")))


def _duration_of(document: dict[str, Any], video_stream: dict[str, Any] | None) -> float:
    fmt = document.get("format") or {}
    duration = _to_float(fmt.get("duration"))
    if duration > 0:
        return duration
    if video_stream is not None:
        duration = _to_float(video_stream.get("duration"))
        if duration > 0:
            return duration
    # Last resort: derive from the container's DURATION tag (HH:MM:SS.mmm).
    tags = fmt.get("tags") or {}
    raw = tags.get("DURATION") or tags.get("duration")
    if raw:
        return _parse_clock(str(raw))
    return 0.0


def _parse_clock(value: str) -> float:
    parts = value.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def probe_basic(tools: Tools, video: Path) -> dict[str, Any]:
    """Container-level metadata.  No decoded frames are touched.

    Uses ffprobe when available; otherwise falls back to ffmpeg's banner, and
    the returned payload records which source was used in
    ``metadata_source``.  A caller that needs a specific field can therefore
    tell "absent because the source cannot report it" from "the file lacks
    it".
    """
    if not tools.has_ffprobe:
        from .banner import probe_basic_without_ffprobe

        return probe_basic_without_ffprobe(tools, video)

    document = probe_raw(tools, video)
    streams = document.get("streams") or []
    if not isinstance(streams, list):
        streams = []
    video_stream = _best_video_stream(streams)
    if video_stream is None:
        raise FfmpegError(
            f"{video.name} carries no video stream, so there is no picture to "
            f"look at. Check that this is a video file and not audio-only."
        )

    fps = _fraction_to_float(
        video_stream.get("avg_frame_rate")
    ) or _fraction_to_float(video_stream.get("r_frame_rate"))

    color_transfer = str(video_stream.get("color_transfer") or "").lower()
    color_primaries = str(video_stream.get("color_primaries") or "").lower()
    pix_fmt = str(video_stream.get("pix_fmt") or "")
    is_hdr = color_transfer in HDR_TRANSFERS or color_primaries in HDR_PRIMARIES
    if not is_hdr and pix_fmt.startswith(("yuv420p10", "yuv420p12", "yuv422p10")):
        # 10/12-bit 4:2:0 is almost always an HDR master.
        is_hdr = color_transfer in HDR_TRANSFERS or color_primaries in HDR_PRIMARIES

    audio_tracks = [
        {
            "index": _to_int(stream.get("index")),
            "codec": str(stream.get("codec_name") or ""),
            "sample_rate": _to_int(stream.get("sample_rate")),
            "channels": _to_int(stream.get("channels")),
            "language": _tags_language(stream.get("tags")),
        }
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]
    subtitle_tracks = [
        {
            "index": _to_int(stream.get("index")),
            "codec": str(stream.get("codec_name") or ""),
            "language": _tags_language(stream.get("tags")),
            "title": str((stream.get("tags") or {}).get("title") or ""),
        }
        for stream in streams
        if stream.get("codec_type") == "subtitle"
    ]
    chapters = [
        {
            "index": index,
            "start": _to_float(chapter.get("start_time")),
            "end": _to_float(chapter.get("end_time")),
            "title": str((chapter.get("tags") or {}).get("title") or ""),
        }
        for index, chapter in enumerate(document.get("chapters") or [])
    ]

    return {
        "duration": round(_duration_of(document, video_stream), 3),
        "fps": round(fps, 6),
        "width": _to_int(video_stream.get("width")),
        "height": _to_int(video_stream.get("height")),
        "video_codec": str(video_stream.get("codec_name") or ""),
        "pix_fmt": pix_fmt,
        "color_range": str(video_stream.get("color_range") or ""),
        "color_space": str(video_stream.get("color_space") or ""),
        "is_hdr": bool(is_hdr),
        "audio_tracks": audio_tracks,
        "subtitle_tracks": subtitle_tracks,
        "chapters": chapters,
        "metadata_source": "ffprobe",
    }
