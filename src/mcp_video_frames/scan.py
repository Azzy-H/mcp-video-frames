"""Whole-file scans: scene changes, silence, loudness — plus their parsers.

The parsers are separated from the scanning so they can be tested against
fixed ffmpeg output samples on machines without ffmpeg, and so a wording
change across ffmpeg versions shows up as a parser test failure rather than
as a silent empty list at runtime.

That distinction is the point: for ffmpeg, *no scene cuts found* and *I
could not understand the output* both print nothing useful.  A parser that
returns ``[]`` on unparsable input makes those two cases indistinguishable,
so every parser here raises on input it cannot interpret and returns ``[]``
only for input it positively understands to mean "none".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import SCAN_PHASE_TIMEOUT
from .errors import FfmpegError
from .ffmpeg import Tools, excerpt, run_ffmpeg

#: A whole-file scan is allowed to take a while; this is a backstop against
#: a hung process, not a performance target.  Kept as separate names because
#: each pass is timed independently, but all three come from one constant so
#: the per-video lock timeout in :mod:`mcp_video_frames.config` cannot drift
#: away from them again.
SCENE_TIMEOUT = SCAN_PHASE_TIMEOUT
SILENCE_TIMEOUT = SCAN_PHASE_TIMEOUT
LOUDNESS_TIMEOUT = SCAN_PHASE_TIMEOUT

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\|\s*silence_duration:\s*([0-9]+(?:\.[0-9]+)?)"
)
# ebur128 prints: "    I:         -23.0 LUFS" and "    Peak:       -1.2 dBFS"
_EBUR128_I_RE = re.compile(r"\bI:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*LUFS")
_EBUR128_LRA_RE = re.compile(r"\bLRA:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*LU")
_EBUR128_PEAK_RE = re.compile(r"\bPeak:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dBFS")

#: Markers that prove a scan actually ran to completion.
_SILENCE_SUMMARY_RE = re.compile(r"silence_(start|end)")
_SCENE_PROGRESS_RE = re.compile(r"frame=|n:")
#: Printed once per stream mapping, whatever the filter reports.
_STREAM_MAPPING_RE = re.compile(r"Stream #\d+:\d+.*->")


def _filter_ran(stderr: str) -> bool:
    """Did a filter graph actually execute?

    ffmpeg prints one ``Stream #0:x -> #0:y`` line per mapped stream and a
    ``frame=``/``size=`` progress line for every run.  Either is proof the
    filter ran.  Without such proof, "no events in the output" and "I did not
    understand the output" are indistinguishable, and the caller deserves to
    be told which one it is.
    """
    return bool(
        _STREAM_MAPPING_RE.search(stderr)
        or _SCENE_PROGRESS_RE.search(stderr)
        or _SILENCE_SUMMARY_RE.search(stderr)
        or re.search(r"size=\S*\s+time=", stderr)
    )


def _looks_like_audio_absent(stderr: str) -> bool:
    """ffmpeg's own wording when an audio-only filter has no audio stream."""
    lowered = stderr.lower()
    return (
        "does not contain any stream" in lowered
        or "matches no streams" in lowered
        or "output file does not contain any stream" in lowered
    )


def parse_scene_cuts(stderr: str, *, duration: float | None = None) -> list[float]:
    """Extract scene-change timestamps from ``showinfo`` output.

    ffmpeg exits 0 whether or not it found cuts, so the completion check is
    structural: the filter is expected to emit ``showinfo`` lines at all.
    A run with no ``pts_time`` line and no frame progress is treated as a
    parse failure, because "no cuts" and "the filter never ran" look the
    same in that case.
    """
    if stderr is None:
        raise FfmpegError(
            "The scene scan produced no ffmpeg output at all, so the result "
            "cannot be determined."
        )
    cuts: list[float] = []
    seen_showinfo = False
    for line in stderr.splitlines():
        if "showinfo" not in line:
            continue
        seen_showinfo = True
        match = _PTS_TIME_RE.search(line)
        if match:
            t = float(match.group(1))
            if duration is not None and t > duration + 0.5:
                continue
            cuts.append(round(t, 3))
    deduped: list[float] = []
    for value in cuts:
        if not deduped or abs(value - deduped[-1]) > 1e-6:
            deduped.append(value)
    if not seen_showinfo and not _filter_ran(stderr):
        raise FfmpegError(
            "Cannot parse scene cuts from ffmpeg's output: there are no "
            "showinfo records and no sign that the filter graph ran.",
            detail=excerpt(stderr),
        )
    return deduped


def parse_silences(stderr: str) -> list[dict[str, float | None]]:
    """Extract silence intervals from ``silencedetect`` output.

    ``silence_end`` may arrive without a matching ``silence_start`` (the
    filter was flushed mid-silence); such an end is reported with an explicit
    ``None`` duration rather than paired with a guess.
    """
    if stderr is None:
        raise FfmpegError(
            "The silence scan produced no ffmpeg output at all, so the result "
            "cannot be determined."
        )
    silences: list[dict[str, float | None]] = []
    pending: float | None = None
    saw_marker = False
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            saw_marker = True
            pending = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match:
            saw_marker = True
            end = float(end_match.group(1))
            length = float(end_match.group(2))
            start = pending if pending is not None else end - length
            silences.append(
                {
                    "start": round(max(start, 0.0), 3),
                    "end": round(end, 3),
                    "duration": round(length, 3),
                }
            )
            pending = None
    if pending is not None:
        # Unclosed silence: the interval started and never ended before the
        # scan stopped.
        silences.append(
            {"start": round(max(pending, 0.0), 3), "end": None, "duration": None}
        )
    if not saw_marker:
        if _looks_like_audio_absent(stderr):
            return []
        if _filter_ran(stderr):
            # The filter ran to completion and reported nothing: that is the
            # genuine answer "this track has no silence long enough".
            return []
        raise FfmpegError(
            "Cannot parse silence intervals from ffmpeg's output: there are "
            "no silencedetect records and no sign that the filter graph ran.",
            detail=excerpt(stderr),
        )
    cleaned = [
        {
            "start": item["start"],
            "end": item["end"],
            "duration": item["duration"],
        }
        for item in silences
        if item["end"] is None or item["end"] > item["start"]
    ]
    return cleaned


def parse_loudness(stderr: str) -> dict[str, float | None]:
    """Extract integrated loudness, true peak and LRA from ebur128 output."""
    if stderr is None:
        raise FfmpegError(
            "The loudness scan produced no ffmpeg output at all, so the result "
            "cannot be determined."
        )
    integrated = _last_match(_EBUR128_I_RE, stderr)
    peak = _last_match(_EBUR128_PEAK_RE, stderr)
    lra = _last_match(_EBUR128_LRA_RE, stderr)
    if integrated is None and peak is None and lra is None:
        if _looks_like_audio_absent(stderr):
            return {}
        raise FfmpegError(
            "Cannot parse loudness from ffmpeg's output: no Integrated "
            "loudness or Peak record was found. ffmpeg versions word this "
            "differently, so this usually means the ebur128 filter did not run.",
            detail=excerpt(stderr),
        )
    if integrated is None and re.search(r"\bI:\s*-?inf\s*LUFS", stderr):
        # ebur128 prints "I: -inf LUFS" for pure digital silence; the regex
        # cannot match that, so recognise it explicitly instead of failing.
        integrated = float("-inf")
    return {
        "integrated_lufs": integrated,
        "true_peak_dbfs": peak,
        "lra": lra,
    }


def _last_match(pattern: re.Pattern[str], text: str) -> float | None:
    """ebur128 prints a progress line plus a summary; take the last value.

    The summary block is emitted at the very end, so the last match is the
    whole-file figure rather than a partial one.
    """
    value: float | None = None
    for match in pattern.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:  # pragma: no cover - regex only matches floats
            continue
    return value


# ----------------------------------------------------------------------
# runners
# ----------------------------------------------------------------------
def scan_scene_cuts(
    tools: Tools, video: Path, *, threshold: float, duration: float | None = None
) -> list[float]:
    """Scene-change timestamps for the whole file.

    ``select`` compares every decoded frame against the previous one, so
    this is exhaustive by construction — and therefore slow on long files,
    which is why its result is cached with the highest retention priority.
    """
    output = run_ffmpeg(
        tools,
        [
            "-i",
            str(video),
            "-an",
            "-sn",
            "-vf",
            f"select='gt(scene,{threshold:g})',showinfo",
            "-f",
            "null",
            "-",
        ],
        timeout=SCENE_TIMEOUT,
    )
    return parse_scene_cuts(output, duration=duration)


def scan_silences(
    tools: Tools, video: Path, *, noise_db: float = -35.0, min_duration: float = 0.6
) -> list[dict[str, float]]:
    """Silence intervals across the whole audio track."""
    output = run_ffmpeg(
        tools,
        [
            "-i",
            str(video),
            "-vn",
            "-sn",
            "-af",
            f"silencedetect=noise={noise_db:g}dB:d={min_duration:g}",
            "-f",
            "null",
            "-",
        ],
        timeout=SILENCE_TIMEOUT,
        expect_output=False,
    )
    return parse_silences(output)


def scan_loudness(tools: Tools, video: Path) -> dict[str, float]:
    """EBU R128 integrated loudness, true peak and loudness range."""
    output = run_ffmpeg(
        tools,
        [
            "-i",
            str(video),
            "-vn",
            "-sn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        timeout=LOUDNESS_TIMEOUT,
        expect_output=False,
    )
    return parse_loudness(output)


def run_full_scan(
    tools: Tools,
    video: Path,
    *,
    threshold: float = 0.3,
    duration: float | None = None,
) -> dict[str, Any]:
    """All three whole-file scans, in one place.

    The CLI ``scan`` subcommand and the MCP ``video_info(detail="full")``
    path both call this.  There is no second implementation to drift.
    """
    return {
        "scene_cuts": scan_scene_cuts(
            tools, video, threshold=threshold, duration=duration
        ),
        "silences": scan_silences(tools, video),
        "loudness": scan_loudness(tools, video),
    }
