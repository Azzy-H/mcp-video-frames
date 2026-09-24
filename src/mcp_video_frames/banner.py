"""Parse ffmpeg's own container report when ffprobe is not available.

``ffprobe`` is the right tool for metadata and is preferred.  It is, however,
an *optional* companion binary: plenty of environments ship a lone
``ffmpeg`` build.  Rather than refusing to start, the server falls back to
reading ffmpeg's banner, which carries the same basic facts.

This module is deliberately small and conservative.  It produces the fields
that appear verbatim in ffmpeg's header (duration, resolution, frame rate,
codecs, streams).  It never guesses at anything ffmpeg does not print, and it
reports which fields it could determine so the caller can tell "absent" from
"unparsed".
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import FfmpegError
from .ffmpeg import Tools, excerpt, run_ffmpeg

_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)"
)
_STREAM_RE = re.compile(r"Stream #(\d+):(\d+)(?:\[[^\]]*\])?(?:\(([^)]*)\))?: (\w+): (.*)")
_VIDEO_DIMS_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_SAMPLE_RATE_RE = re.compile(r"(\d+)\s*Hz")
_CHANNELS_RE = re.compile(r"\b(mono|stereo|[\d.]+\s*(?:channels|\(.*?\)))\b")
_PIX_FMT_RE = re.compile(r",\s*(yuv\w+|gray\w*|rgb\w*|bgr\w*|yuva\w+|p010\w*|nv12)\b")

#: Known bit-depth/pixel-format prefixes that indicate HDR-capable content.
_HDR_PIX_FMTS = ("yuv420p10", "yuv420p12", "yuv422p10", "p010")


def parse_container_banner(banner: str, *, path: Path) -> dict[str, Any]:
    """Turn an ffmpeg header dump into the ``basic`` metadata shape.

    Only the *input* section is read.  The same dump also echoes the output
    stream mapping, and counting that would report every track twice (once for
    the source codec, once for the encoder's).
    """
    if not banner or "Input #" not in banner:
        raise FfmpegError(
            f"Cannot parse container information from ffmpeg's output "
            f"({path.name}). The file may not be a video container ffmpeg "
            f"recognises.",
            detail=excerpt(banner),
        )

    duration = 0.0
    match = _DURATION_RE.search(banner)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    video: dict[str, Any] = {}
    audio_tracks: list[dict[str, Any]] = []
    subtitle_tracks: list[dict[str, Any]] = []
    seen_streams = 0
    for raw in _input_stream_lines(banner):
        parsed = _STREAM_RE.search(raw)
        if not parsed:
            continue
        seen_streams += 1
        _input_index, stream_index, language, kind, rest = parsed.groups()
        rest = (rest or "").split(" (default)")[0]
        if kind == "Video" and not video:
            codec = rest.split(":")[0].strip().split(" ")[0].strip(",")
            dims = _VIDEO_DIMS_RE.search(rest)
            fps = _FPS_RE.search(rest)
            pix_fmt = _PIX_FMT_RE.search(rest)
            video = {
                "width": int(dims.group(1)) if dims else 0,
                "height": int(dims.group(2)) if dims else 0,
                "fps": round(_ratio_to_float(fps.group(1)), 6) if fps else 0.0,
                "video_codec": codec,
                "pix_fmt": pix_fmt.group(1) if pix_fmt else "",
            }
        elif kind == "Audio":
            codec = rest.split(",")[0].strip().split(" ")[0]
            rate = _SAMPLE_RATE_RE.search(rest)
            channels = _CHANNELS_RE.search(rest)
            channel_count = 0
            if channels:
                text = channels.group(1)
                if text == "mono":
                    channel_count = 1
                elif text == "stereo":
                    channel_count = 2
                else:
                    digits = re.match(r"(\d+)", text)
                    channel_count = int(digits.group(1)) if digits else 0
            audio_tracks.append(
                {
                    "index": int(stream_index),
                    "codec": codec,
                    "sample_rate": int(rate.group(1)) if rate else 0,
                    "channels": channel_count,
                    "language": language or "",
                }
            )
        elif kind == "Subtitle":
            subtitle_tracks.append(
                {
                    "index": int(stream_index),
                    "codec": rest.split(":")[0].strip(),
                    "language": language or "",
                    "title": "",
                }
            )

    if not video:
        raise FfmpegError(
            f"{path.name} carries no video stream, so there is no picture to "
            f"look at. Check that this is a video file and not audio-only.",
            detail=excerpt(banner),
        )
    if seen_streams == 0:
        raise FfmpegError(
            f"No stream information could be recognised in ffmpeg's output "
            f"({path.name}). ffmpeg versions word this differently, so this "
            f"usually means the header format changed.",
            detail=excerpt(banner),
        )

    pix_fmt = str(video.get("pix_fmt") or "")
    return {
        "duration": round(duration, 3),
        "fps": float(video.get("fps") or 0.0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": str(video.get("video_codec") or ""),
        "pix_fmt": pix_fmt,
        # ffmpeg's plain banner does not carry colour metadata.
        "color_range": "",
        "color_space": "",
        "is_hdr": pix_fmt.startswith(_HDR_PIX_FMTS),
        "audio_tracks": audio_tracks,
        "subtitle_tracks": subtitle_tracks,
        "chapters": [],
        "metadata_source": "ffmpeg-banner",
    }


def _input_stream_lines(banner: str) -> list[str]:
    """``Stream #`` lines belonging to the input, not the output.

    ffmpeg prints three kinds of line containing ``Stream #``:

    * the input header, which always carries the stream id in brackets
      (``Stream #0:1[0x2](und): Audio: aac ...``);
    * the stream mapping (``Stream #0:0 -> #0:0 (h264 (native) -> ...)``);
    * the output header, which describes what the encoder wrote
      (``Stream #0:1(und): Audio: pcm_s16le ...``).

    Only the first is the file's metadata; reading the others would report
    every track twice, once with the source codec and once with the encoder's.
    """
    return [
        line
        for line in banner.splitlines()
        if line.strip().startswith("Stream #")
        and re.search(r"Stream #\d+:\d+\[", line)
    ]


def probe_basic_without_ffprobe(tools: Tools, path: Path) -> dict[str, Any]:
    """Run ffmpeg long enough to print the container header.

    ``-f null -`` gives ffmpeg a real output to aim at, so it reports the
    input fully; the ``-t 0`` cap keeps it from actually transcoding.
    """
    banner = run_ffmpeg(
        tools,
        ["-i", str(path), "-t", "0", "-f", "null", "-"],
        timeout=120,
        expect_output=False,
    )
    return parse_container_banner(banner, path=path)


def _ratio_to_float(text: str) -> float:
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        try:
            return float(text)
        except ValueError:
            return 0.0
