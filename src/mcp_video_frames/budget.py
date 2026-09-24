"""Frame-count and byte-budget arithmetic.

Everything here is pure: no ffmpeg, no filesystem, no MCP.  It is the part
of the server most likely to be wrong in a subtle way, so it is kept
isolated and unit tested on its own.

Two rules from the design are enforced here:

* frame count is decided *before* any work happens, and exceeding a limit
  is an error, never a truncation;
* clamping is reported, never silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import (
    DEFAULT_MAX_FRAMES,
    MAX_INTERVAL,
    MAX_MAX_EDGE,
    MAX_QUALITY,
    MIN_INTERVAL,
    MIN_MAX_EDGE,
    MIN_MAX_FRAMES,
    MIN_QUALITY,
)
from .errors import InputError, LimitError

#: Timestamps closer than this are treated as identical.  ffmpeg timecodes
#: carry millisecond precision in our cache filenames, so this only absorbs
#: float round-off from arithmetic like ``0.1 * 3``.
TIME_EPSILON = 1e-9

#: JPEG bytes per pixel, measured conservatively on photographic frames at
#: q≈4 (``-q:v 4``).  Used for the pre-flight estimate only; the real check
#: happens after extraction, on the actual files.
JPEG_BYTES_PER_PIXEL = 0.16

#: PNG bytes per pixel for the same content.  PNG is lossless, so this is
#: roughly an order of magnitude above JPEG — which is exactly why range
#: requests default to jpeg.
PNG_BYTES_PER_PIXEL = 1.0

#: base64 expands 3 bytes to 4 characters.
BASE64_OVERHEAD = 4.0 / 3.0

#: Past this many chunks, enumerating the calls is noise rather than advice.
MAX_ENUMERATED_CHUNKS = 24

VALID_FORMATS = ("jpeg", "png")


@dataclass(frozen=True)
class FrameSpec:
    """One frame to fetch: ordinal position plus its absolute timestamp."""

    n: int
    t: float


@dataclass(frozen=True)
class FramesPlan:
    """Outcome of planning a ``view_frames`` call.

    ``clamped`` and the ``*_requested`` fields exist so the summary can
    state the original values alongside the ones actually used.
    """

    start: float
    end: float
    interval: float
    count: int
    frames: tuple[FrameSpec, ...]
    max_edge: int
    image_format: str
    quality: int
    clamped: bool = False
    requested_start: float | None = None
    requested_end: float | None = None
    duration: float | None = None

    @property
    def range(self) -> tuple[float, float]:
        if self.count <= 1:
            return (self.start, self.start)
        return (self.frames[0].t, self.frames[-1].t)

    @property
    def is_single_frame(self) -> bool:
        return self.count == 1


def frame_timestamp(start: float, interval: float, index: int) -> float:
    """Absolute timestamp of frame ``index`` in a range extraction.

    The grid is anchored at ``start`` and stepped by ``interval``.  The
    ffmpeg range command samples by fps, which yields exactly ``count``
    frames; the count is verified on disk, which is what makes this
    convention trustworthy.
    """
    return start + index * interval


def compute_frame_count(start: float, end: float, interval: float) -> int:
    """Frames a ``[start, end]`` request can yield.

    ``start < end``  -> ``floor((end - start) / interval) + 1``
    ``start == end`` -> exactly one frame

    This is the number a caller reasons with, and the number the over-limit
    message quotes: asking for "0 to 10 seconds" is asking about ten seconds
    of video, and the cost of that question is what belongs in the error.
    """
    if end <= start + TIME_EPSILON:
        return 1
    return int(math.floor((end - start) / interval + TIME_EPSILON)) + 1


def sampled_count(start: float, end: float, interval: float) -> int:
    """Upper bound on the frames a ``[start, end]`` request will produce.

    The sample grid is ``S, S+I, S+2I, ...`` and the last sample is emitted
    only when its timestamp falls inside the clip, so this is
    ``floor((end - start) / interval) + 1`` — the same number
    :func:`compute_frame_count` reports, deliberately.

    Over-counting is the safe direction: the extraction path verifies how many
    files ffmpeg actually wrote and the reported frame list is truncated to
    that, so a plan that expects one frame too many loses that frame rather
    than inventing it.  Under-counting would be a silent loss of a frame the
    caller asked for, which is exactly what this design must not do.
    """
    return compute_frame_count(start, end, interval)


def default_format(start: float, end: float) -> str:
    """Range requests default to jpeg; single-frame requests to png."""
    return "png" if end <= start + TIME_EPSILON else "jpeg"


def default_end(start: float, interval: float) -> float:
    return start + interval


def limit_error(
    *,
    start: float,
    end: float,
    interval: float,
    count: int,
    limit: int,
    clamped_start: float | None = None,
    clamped_end: float | None = None,
) -> None:
    """Raise the over-limit error, with concrete options.

    The message must contain a workable ``interval`` suggestion and a concrete
    split into calls, because "too many frames" alone leaves the caller with
    nothing to do.

    ``start``/``end``/``count`` are the request as the caller made it: the
    headline number should describe their question, not a reduced version of
    it.  ``clamped_start``/``clamped_end`` are the range that will actually be
    sampled once the video's duration is applied, and the suggestions are
    computed against *that* — advising a wider interval for a range that no
    longer exists would be useless.

    Every suggestion is checked with :func:`compute_frame_count` before it is
    printed: a suggestion that does not fit is worse than none, because the
    caller will try it and fail again.
    """
    action_start = start if clamped_start is None else clamped_start
    action_end = end if clamped_end is None else clamped_end
    span = max(action_end - action_start, interval)
    options: list[str] = []

    # A grid of `limit` samples covers the span when the interval is this
    # wide; round up to a value a caller can actually type, then verify it.
    needed = span / max(limit - 1, 1)
    if needed > interval:
        digits = max(0, 2 - int(math.floor(math.log10(needed))))
        candidate = math.ceil(needed * 10**digits) / 10**digits
        candidate = round(max(candidate, MIN_INTERVAL, interval), 6)
        while (
            compute_frame_count(action_start, action_start + span, candidate) > limit
            and candidate < MAX_INTERVAL
        ):
            candidate = round(candidate + 10 ** (-digits), 6)
        if compute_frame_count(action_start, action_start + span, candidate) <= limit:
            options.append(f"raise interval to >= {candidate:g}")

    # The split is derived from the user-visible count, so each chunk stays
    # inside the limit by construction.  Past a couple of dozen chunks the
    # enumeration stops being advice and starts being noise, so it is
    # summarised instead.
    visible = compute_frame_count(action_start, action_end, interval)
    if action_end > action_start and visible > 1:
        chunks = max(2, math.ceil(visible / limit))
        if chunks <= MAX_ENUMERATED_CHUNKS:
            edges = [action_start]
            for i in range(1, chunks):
                edges.append(action_start + (action_end - action_start) * i / chunks)
            edges.append(action_end)
            rounded = [round(edge, 1) for edge in edges]
            spans = ", ".join(
                f"{rounded[i]:g}-{rounded[i + 1]:g}" for i in range(chunks)
            )
            options.append(f"split into {chunks} calls: {spans}")
        else:
            step = max(interval, span / limit)
            options.append(
                f"split into {chunks} calls of about {step:g}s each"
                f" (take a coarse pass with a larger interval first)"
            )

    options.append(
        "change the server configuration (MCP_VIDEO_MAX_IMAGES sets the "
        "per-call image limit)"
    )

    lines = [
        f"Request {start:g}-{end:g}s at interval {interval:g} yields {count} "
        f"frames, over the per-call limit of {limit}.",
    ]
    if clamped_start is not None and clamped_end is not None and (
        abs(clamped_start - start) > 1e-3 or abs(clamped_end - end) > 1e-3
    ):
        lines.append(
            f"(the request {start:g}-{end:g}s was clamped to the video's "
            f"{clamped_start:g}-{clamped_end:g}s; the options below use the "
            f"clamped range)"
        )
    lines.append("Options:")
    lines.extend(f"  - {option}" for option in options)
    raise LimitError("\n".join(lines))


def resolve_interval(interval: float) -> float:
    """Validate an interval, clamping float noise back into range.

    Values within 1e-9 of a bound are snapped to the bound; anything
    outside is rejected rather than quietly adjusted.
    """
    if math.isnan(interval) or math.isinf(interval):
        raise InputError(f"interval must be a finite number, got {interval!r}.")
    for bound in (MIN_INTERVAL, MAX_INTERVAL):
        if abs(interval - bound) <= 1e-9:
            return bound
    if interval < MIN_INTERVAL:
        raise InputError(
            f"interval={interval:g}s is below the minimum of {MIN_INTERVAL:g}s. "
            f"Set it to >= {MIN_INTERVAL:g}."
        )
    if interval > MAX_INTERVAL:
        raise InputError(
            f"interval={interval:g}s is above the maximum of {MAX_INTERVAL:g}s. "
            f"Set it to <= {MAX_INTERVAL:g}."
        )
    return interval


def validate_max_frames(max_frames: int, max_images_per_call: int) -> int:
    if max_frames < MIN_MAX_FRAMES:
        raise InputError(f"max_frames={max_frames} must be >= {MIN_MAX_FRAMES}.")
    if max_frames > max_images_per_call:
        raise InputError(
            f"max_frames={max_frames} is over the per-call image limit of "
            f"{max_images_per_call}. Set it to <= {max_images_per_call}, or "
            f"raise MCP_VIDEO_MAX_IMAGES (which must stay within your client's "
            f"own image limit)."
        )
    return max_frames


def validate_max_edge(max_edge: int) -> int:
    if max_edge < MIN_MAX_EDGE:
        raise InputError(
            f"max_edge={max_edge} is below the minimum of {MIN_MAX_EDGE} pixels."
        )
    if max_edge > MAX_MAX_EDGE:
        raise InputError(
            f"max_edge={max_edge} is above the maximum of {MAX_MAX_EDGE} pixels. "
            f"Set it to <= {MAX_MAX_EDGE}."
        )
    return max_edge


def validate_quality(quality: int) -> int:
    if quality < MIN_QUALITY or quality > MAX_QUALITY:
        raise InputError(
            f"quality={quality} is outside the range {MIN_QUALITY}-{MAX_QUALITY}."
        )
    return quality


def validate_format(image_format: str) -> str:
    normalized = image_format.strip().lower()
    if normalized not in VALID_FORMATS:
        raise InputError(
            f"format={image_format!r} is not available. "
            f"Valid values: {', '.join(VALID_FORMATS)}."
        )
    return normalized


def qscale_for_quality(quality: int) -> int:
    """Map a 1–100 quality onto ffmpeg's mjpeg ``-q:v`` 2–31 scale.

    Lower is better.  Linear interpolation is deliberately used instead of
    a table: exactness is not the point, monotonicity is.
    """
    validate_quality(quality)
    if quality >= 100:
        return 2
    if quality <= 1:
        return 31
    return int(round(31 - (quality - 1) * (31 - 2) / (100 - 1)))


def estimate_frame_bytes(
    *, width: int, height: int, max_edge: int, image_format: str
) -> int:
    """Estimate the encoded size of one scaled frame, in bytes.

    Only used to fail fast on absurd requests.  It over-estimates on
    purpose; the authoritative per-frame and total checks run against the
    real files after extraction.
    """
    long_edge = max(width, height)
    scale = 1.0 if long_edge <= 0 or long_edge <= max_edge else max_edge / long_edge
    pixels = max(1.0, width * scale) * max(1.0, height * scale)
    per_pixel = PNG_BYTES_PER_PIXEL if image_format == "png" else JPEG_BYTES_PER_PIXEL
    return int(pixels * per_pixel) + 1024


def estimate_payload_bytes(frame_bytes: int, frame_count: int) -> int:
    """Total response size for ``frame_count`` frames, base64 included."""
    return int(frame_bytes * frame_count * BASE64_OVERHEAD)


def check_total_budget(
    *,
    frame_bytes: int,
    frame_count: int,
    max_total_bytes: int,
    max_edge: int,
    image_format: str,
    interval: float,
    start: float,
    end: float,
) -> None:
    """Fail before extraction when the response cannot possibly fit."""
    estimated = estimate_payload_bytes(frame_bytes, frame_count)
    if estimated <= max_total_bytes:
        return
    alt_edge = max(MIN_MAX_EDGE, int(max_edge * math.sqrt(max_total_bytes / estimated)))
    suggestions = [
        f"lower max_edge to <= {alt_edge} (now {max_edge}, estimated "
        f"{_mib(estimated)} in total)",
    ]
    if image_format != "jpeg":
        suggestions.append(
            "or use format=jpeg (about an order of magnitude smaller at the "
            "same edge length)"
        )
    suggestions.append(
        f"or request fewer frames (now {frame_count} frames at interval "
        f"{interval:g}s, covering {start:g}-{end:g}s)"
    )
    lines = [
        f"This result is estimated at {_mib(estimated)}, over the per-call "
        f"total limit of {_mib(max_total_bytes)} "
        f"({frame_count} frames x about {_mib(frame_bytes)} each).",
        "Options:",
    ]
    lines.extend(f"  - {s}" for s in suggestions)
    raise LimitError("\n".join(lines))


def _mib(num_bytes: float) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


@dataclass
class PlanRequest:
    """Raw, already type-checked arguments for one ``view_frames`` call."""

    start: float = 0.0
    end: float | None = None
    interval: float = 1.0
    max_frames: int | None = None
    max_edge: int = 768
    image_format: str | None = None
    quality: int = 85
    duration: float | None = field(default=None)


def plan_frames(
    request: PlanRequest,
    *,
    max_images_per_call: int,
    duration: float | None = None,
    max_image_bytes: int | None = None,
    max_total_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> FramesPlan:
    """Turn a request into a concrete, validated frame plan.

    Raises ``InputError`` for bad arguments and ``LimitError`` for anything
    that would need truncation.  Never silently reduces what was asked for.
    """
    interval = resolve_interval(float(request.interval))
    start = float(request.start)
    if math.isnan(start) or math.isinf(start):
        raise InputError(f"start must be a finite number, got {request.start!r}.")
    if start < 0:
        raise InputError(f"start={start:g}s cannot be negative. Set it to >= 0.")

    end = None if request.end is None else float(request.end)
    if end is None:
        end = default_end(start, interval)
    if math.isnan(end) or math.isinf(end):
        raise InputError(f"end must be a finite number, got {request.end!r}.")
    if end < start:
        raise InputError(
            f"end={end:g}s is before start={start:g}s. To look at a single "
            f"moment, set start and end to the same value."
        )

    max_edge = validate_max_edge(int(request.max_edge))
    quality = validate_quality(int(request.quality))
    image_format = (
        default_format(start, end)
        if request.image_format is None
        else validate_format(request.image_format)
    )

    requested_start, requested_end = start, end
    # The count the caller's range implies, computed before clamping touches
    # the endpoints; the error message quotes this, not a reduced version.
    requested_count = compute_frame_count(requested_start, requested_end, interval)
    clamped = False
    if duration is not None and duration > 0:
        clamped_start = min(max(start, 0.0), duration)
        clamped_end = min(max(end, 0.0), duration)
        if abs(clamped_start - start) > TIME_EPSILON or abs(
            clamped_end - end
        ) > TIME_EPSILON:
            clamped = True
        # A point query at a moment past the end legitimately collapses to
        # the last available moment.  A *range* that collapses because it lay
        # entirely outside the video does not: reporting a frame from t=0
        # would answer a question that was never asked.
        if end > start + TIME_EPSILON and clamped_end <= clamped_start + TIME_EPSILON:
            raise InputError(
                f"The range {requested_start:g}-{requested_end:g}s falls "
                f"entirely outside the video's {duration:g}s, leaving no time "
                f"point after clamping. Put the range within 0-{duration:g}s."
            )
        start, end = clamped_start, clamped_end
        if end < start:
            end = start
        # A range ending at (or past) the end of the clip has no frame at that
        # last grid point: the final frame *starts* before the clip ends.
        # Pulling the end a microsecond inside the duration makes the reported
        # count equal the frames that actually exist.  A point query is left
        # alone — asking what the last instant looks like is a real question.
        if end > start + TIME_EPSILON and abs(end - duration) <= TIME_EPSILON:
            end = max(start, duration - 1e-6)
    elif duration is not None and duration <= 0:
        raise InputError(
            f"The video reports a duration of {duration:g}s, so no frame can "
            f"be extracted. Check that this file contains a video stream."
        )

    # The limit is evaluated against the frames that will actually be sent,
    # after clamping.  Raising for the pre-clamp request would reject calls
    # that fit perfectly well once the range is trimmed to the video.
    effective_limit = min(
        max_images_per_call,
        DEFAULT_MAX_FRAMES if request.max_frames is None else int(request.max_frames),
    )
    if request.max_frames is not None:
        validate_max_frames(int(request.max_frames), max_images_per_call)

    count = sampled_count(start, end, interval)
    if count == 0:
        raise LimitError(
            f"The range {start:g}-{end:g}s at interval {interval:g}s yields 0 "
            f"frames after clamping. Check that start/end fall within the "
            f"video's duration."
        )
    if count > effective_limit:
        # The headline number is what the caller asked about; the suggestions
        # are computed against the range that will actually be sampled.
        limit_error(
            start=requested_start,
            end=requested_end,
            interval=interval,
            count=requested_count,
            limit=effective_limit,
            clamped_start=start,
            clamped_end=end,
        )

    frames = tuple(
        FrameSpec(n=i, t=frame_timestamp(start, interval, i)) for i in range(count)
    )

    if (
        max_image_bytes is not None
        and max_total_bytes is not None
        and width
        and height
    ):
        frame_bytes = estimate_frame_bytes(
            width=width, height=height, max_edge=max_edge, image_format=image_format
        )
        check_total_budget(
            frame_bytes=frame_bytes,
            frame_count=count,
            max_total_bytes=max_total_bytes,
            max_edge=max_edge,
            image_format=image_format,
            interval=interval,
            start=start,
            end=end,
        )

    return FramesPlan(
        start=start,
        end=end,
        interval=interval,
        count=count,
        frames=frames,
        max_edge=max_edge,
        image_format=image_format,
        quality=quality,
        clamped=clamped,
        requested_start=requested_start,
        requested_end=requested_end,
        duration=duration,
    )
