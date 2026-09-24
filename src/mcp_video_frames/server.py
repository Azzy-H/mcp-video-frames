"""mcp-video-frames — MCP server exposing video frames and metadata.

Scope is deliberately narrow: this module answers "what does the video look
like at time t" and "what are this video's measurable properties".

It deliberately does NOT track which ranges a caller has viewed. Frame
delivery is not evidence of comprehension, so such a record would be a proxy
metric masquerading as coverage. Callers that need coverage tracking should
maintain it in their own artifacts.

The protocol layer is thin on purpose: it validates argument types, calls
:mod:`mcp_video_frames.core`, and turns the result into an ordered list of
``text``/``image`` content blocks.  Every rule that matters lives in the
core library so the CLI and the tests can exercise it without MCP.

Two properties are load-bearing and must not be "simplified" away:

* **block order** — each frame is a timecode text block immediately followed
  by its image, and the summary text block comes last.  A model that
  receives images without their timecodes cannot reason about time.
* **no silent degradation** — anything over a limit raises, and the whole
  call fails.  Truncating a batch would look like success to the caller.

The SDK is imported through :func:`_load_sdk`, which accepts both published
API generations: ``mcp.server.fastmcp.FastMCP`` (mcp 1.x) and
``mcp.server.mcpserver.MCPServer`` (mcp 2.x, where FastMCP was renamed).
Everything below uses only the small surface both provide.
"""

from __future__ import annotations

import importlib
import sys
from typing import Annotated, Any, Literal

from pydantic import Field

from .budget import (
    MAX_INTERVAL,
    MAX_MAX_EDGE,
    MAX_QUALITY,
    MIN_INTERVAL,
    MIN_MAX_EDGE,
    MIN_MAX_FRAMES,
    MIN_QUALITY,
)
from .config import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_EDGE,
    DEFAULT_QUALITY,
    Config,
    default_cache_root,
)
from .core import Core
from .errors import InputError, VideoFramesError

#: Argument bounds carried into the generated tool schema.
#:
#: These are the same constants :mod:`mcp_video_frames.budget` enforces, so the
#: schema and the runtime check cannot drift apart.  Without them a caller sees
#: a bare ``number``: it has no way to learn that 0.05 is the floor on
#: ``interval``, and "want more frames, shrink the interval" — the obvious
#: inference — walks straight into a rejection.
IntervalSeconds = Annotated[float, Field(ge=MIN_INTERVAL, le=MAX_INTERVAL)]
MaxEdgePixels = Annotated[int, Field(ge=MIN_MAX_EDGE, le=MAX_MAX_EDGE)]
JpegQuality = Annotated[int, Field(ge=MIN_QUALITY, le=MAX_QUALITY)]
#: ``max_frames``' ceiling is the deployment limit, which varies; only its
#: floor is fixed, so only the floor is declared here.
MaxFrames = Annotated[int, Field(ge=MIN_MAX_FRAMES)]
#: The accepted values, spelled out so the schema offers an ``enum`` instead of
#: an unconstrained string.
ImageFormat = Literal["jpeg", "png"]
InfoDetail = Literal["basic", "full"]

#: Install paths shown when the SDK is missing or too old.
_MCP_INSTALL_HINT = (
    "  pip install 'mcp>=1.2'\n"
    "  (this project supports both mcp 1.x FastMCP and 2.x MCPServer)\n"
    "  or run pip install -e . inside the project directory"
)


def _load_sdk() -> tuple[Any, Any, Any, Any]:
    """Return ``(sdk, server_class, content_types, tool_error)``.

    mcp 2.x renamed ``FastMCP`` to ``MCPServer`` and moved it to
    ``mcp.server.mcpserver``; the 1.x name is kept only as a stub that raises
    on import.  Both expose ``tool(...)`` and ``run(transport=...)`` with the
    same meaning, so one class reference covers both.

    ``tool_error`` matters more than it looks.  Raising an arbitrary exception
    makes the SDK treat it as a crash: mcp 2.x replaces the message with
    ``"Error executing tool <name>"`` and logs the original to stderr only.
    Our error text carries the numbers and the suggested fix, so it has to
    survive to the client — which means raising the SDK's own ``ToolError``.
    Falling back to ``RuntimeError`` keeps older SDKs working at the cost of
    the message being wrapped rather than passed through.
    """
    tool_error: Any = RuntimeError
    for module_name in (
        "mcp.server.fastmcp.exceptions",
        "mcp.server.mcpserver.exceptions",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "ToolError"):
            tool_error = module.ToolError
            break

    try:
        from mcp.server.fastmcp import FastMCP as server_class
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer as server_class
        except ImportError as exc:
            raise ImportError(
                "No usable mcp server API found (neither "
                "mcp.server.fastmcp.FastMCP nor mcp.server.mcpserver.MCPServer)."
            ) from exc
    try:
        from mcp.types import ImageContent, TextContent
    except ImportError as exc:  # pragma: no cover - SDK too old
        raise ImportError("TextContent / ImageContent not found in mcp.types.") from exc
    import mcp

    return mcp, server_class, (TextContent, ImageContent), tool_error


try:
    _MCP, _SERVER_CLASS, (_TEXT_CONTENT, _IMAGE_CONTENT), _TOOL_ERROR = _load_sdk()
except ImportError as exc:  # pragma: no cover - exercised only when uninstalled
    raise SystemExit(f"No usable mcp SDK: {exc}\n{_MCP_INSTALL_HINT}") from exc

TextContent = _TEXT_CONTENT
ImageContent = _IMAGE_CONTENT


INSTRUCTIONS = """\
This server gives image-only models a way to look at video—an input modality
they do not natively support. If a model already accepts video, this server
is unnecessary.

Practical advice:

Start with video_info(detail="basic"), then call view_frames for the moments
worth looking at. For a long video, take a coarse pass first and use the
returned timecodes to pick the seconds worth a closer look.

Overlapping calls are cheap: ask for the range you want without worrying about
having covered part of it already.
"""


def _json_ok(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _limits_note(config: Config) -> str:
    """The effective hard limits, for the tool descriptions.

    A model sees only tool descriptions, parameter schemas and results — never
    the generated client config or the startup log.  Without these numbers in
    the description it can only discover a limit by hitting it, which costs a
    round trip and looks like a failure rather than a rule.  Stating them also
    tells the caller that the server refuses rather than truncates, so the
    error is expected rather than surprising.
    """
    return (
        "Server limits for this deployment: at most "
        f"{config.max_images_per_call} images per call, each at most "
        f"{config.max_image_bytes // (1024 * 1024)} MB, "
        f"{config.max_total_bytes // (1024 * 1024)} MB per result. "
        "Frames are never silently dropped or resampled, so the interval you "
        "ask for is the interval you get. Requests past a limit fail with the "
        "numbers and concrete alternatives — the server never truncates a "
        "range or coarsens your interval."
    )


def _requests_note() -> str:
    """How the per-request arguments interact with the deployment limits.

    These are server-side rules, so they hold for every deployment and are
    worth stating in full.  The *numbers* they interact with are not: those
    come from configuration and stay in :func:`_limits_note`.
    """
    return (
        "max_frames defaults to 12; raising it up to the image limit trades "
        "away detail — lower max_edge or use format=\"jpeg\" to afford more "
        "frames. All of these limits are set by whoever deployed this server, "
        "not by you."
    )


def _view_frames_description(config: Config) -> str:
    return (
        "Sample a video and return frames as images, each preceded by its "
        "timecode — the only thing tying an image to a moment. Set start == "
        "end to look at a single moment (returns a lossless PNG); the interval "
        "is ignored in that case. Range requests default to jpeg, which is "
        "about an order of magnitude smaller and fits far more frames in the "
        "same budget. Frames are cached by absolute timestamp, so overlapping "
        "ranges reuse work. "
        + _requests_note()
        + " "
        + _limits_note(config)
    )


def _video_info_description(config: Config) -> str:
    return (
        'Metadata for a video. detail="basic" reads container headers only and '
        "returns immediately. detail=\"full\" additionally scans the whole file "
        "for scene cuts, silence intervals and EBU R128 loudness; that scan is "
        "exhaustive and therefore slow on long files, but it is cached and "
        "reused. Use force=true to recompute. This tool returns no images, so "
        "the image limits do not apply to it, but your client's per-call time "
        "limit does: a full scan of a long video takes time proportional to "
        "its length, and may exceed the limit. Raise that limit if your client "
        "allows it."
    )


def build_server(core: Core | None = None) -> Any:
    """Create the MCP server.  ``core`` is injectable for tests."""
    config = (core.config if core is not None else Config.from_env())
    _log_startup(config)

    mcp = _SERVER_CLASS("mcp-video-frames", instructions=INSTRUCTIONS)
    _core = core

    def get_core() -> Core:
        nonlocal _core
        if _core is None:
            _core = Core(config)
        return _core

    # Run the self check once, here, so a missing ffmpeg or an unwritable
    # cache is reported at startup instead of on the first tool call.
    startup_check = getattr(get_core(), "startup_check", None)
    if startup_check is not None:
        for problem in startup_check():
            print(
                f"mcp-video-frames: self-check failed\n{problem}",
                file=sys.stderr,
                flush=True,
            )

    @mcp.tool(
        name="view_frames",
        description=_view_frames_description(config),
    )
    def view_frames(
        video: str,
        start: float = 0.0,
        end: float | None = None,
        interval: IntervalSeconds = DEFAULT_INTERVAL,
        max_frames: MaxFrames | None = None,
        max_edge: MaxEdgePixels = DEFAULT_MAX_EDGE,
        format: ImageFormat | None = None,
        quality: JpegQuality = DEFAULT_QUALITY,
    ) -> list[Any]:
        """Extract frames from ``video`` between ``start`` and ``end`` seconds.

        Args:
            video: Absolute path to the video file.
            start: Start time in seconds (>= 0). Default 0.
            end: End time in seconds (>= start). Defaults to start + interval.
            interval: Sampling interval in seconds, 0.05-600. Default 1.0.
            max_frames: Maximum frames for this call, 1 to the server limit
                (MCP_VIDEO_MAX_IMAGES, default 20). Default 12.
            max_edge: Long-edge size of each frame in pixels, 64-4096.
                Default 768.
            format: "jpeg" or "png". Defaults to jpeg for a range and png for
                a single frame.
            quality: jpeg quality, 1-100. Default 85.
        """
        try:
            result = get_core().view_frames(
                video,
                start=start,
                end=end,
                interval=interval,
                max_frames=max_frames,
                max_edge=max_edge,
                image_format=format,
                quality=quality,
            )
        except VideoFramesError as exc:
            raise _tool_error(exc) from exc
        return _content_blocks(result)

    @mcp.tool(
        name="video_info",
        description=_video_info_description(config),
    )
    def video_info(
        video: str, detail: InfoDetail = "basic", force: bool = False
    ) -> str:
        """Return video metadata as JSON.

        Args:
            video: Absolute path to the video file.
            detail: "basic" (container headers only) or "full" (adds a
                whole-file scene/silence/loudness scan). Default "basic".
            force: Ignore cached results and recompute. Default false.
        """
        wanted = (detail or "basic").strip().lower()
        if wanted not in ("basic", "full"):
            return _error_payload(
                InputError(
                    f"detail={detail!r} is not available. Valid values: basic, "
                    f"full. basic reads container information only; full adds a "
                    f"whole-file scan."
                )
            )
        try:
            core_ = get_core()
            if wanted == "full":
                payload = core_.video_full_info(video, force=force)
            else:
                payload, _cached = core_.video_basic_info(video, force=force)
        except VideoFramesError as exc:
            return _error_payload(exc)
        return _json_ok(payload)

    return mcp


def _tool_error(exc: VideoFramesError) -> Exception:
    """Turn a core error into what the client sees: message + guidance.

    No stack traces: the caller is a model or a human operator, and both
    need "what happened / which numbers / what to do next".

    The SDK's own ``ToolError`` is used when available so the text is passed
    through verbatim.  An arbitrary exception is treated as a crash by mcp 2.x
    and replaced with a generic "Error executing tool ..." — which would throw
    away exactly the information this server exists to provide.
    """
    text = str(exc)
    if exc.detail:
        text = f"{text}\n\n{exc.detail}"
    return _TOOL_ERROR(text)


def _error_payload(exc: VideoFramesError) -> str:
    """Same information as :func:`_tool_error`, as machine-readable JSON.

    ``video_info`` returns text, so its failures travel as a JSON object with
    ``error`` and ``kind`` rather than a bare sentence: a caller that wants to
    branch on the failure should not have to parse prose.
    """
    import json

    payload = {
        "error": exc.message,
        "kind": type(exc).__name__,
    }
    if exc.detail:
        payload["detail"] = exc.detail
    stderr = getattr(exc, "stderr", None)
    if stderr:
        payload["ffmpeg_stderr"] = stderr
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _content_blocks(result) -> list[Any]:
    """Assemble ``[text, image] × n + [text]``.

    The SDK is asked to preserve this order; if a future SDK reordered or
    dropped blocks, the design would silently break, so the count is asserted
    here rather than trusted.

    ``ImageContent`` carries the MIME type under the protocol's camelCase
    field.  A pydantic model rejects unknown fields, so a spelling change in a
    future SDK surfaces as a ``TypeError`` here and is retried with the
    snake_case alias instead of crashing the tool.
    """
    from .core import build_content_blocks

    blocks: list[Any] = []
    for block in build_content_blocks(result):
        if block["type"] == "text":
            blocks.append(TextContent(type="text", text=block["text"]))
            continue
        try:
            blocks.append(
                ImageContent(
                    type="image", data=block["data"], mimeType=block["mimeType"]
                )
            )
        except TypeError:  # pragma: no cover - older/newer SDK field name
            blocks.append(
                ImageContent(
                    type="image", data=block["data"], mime_type=block["mimeType"]
                )
            )
    expected = 2 * len(result.frames) + 1
    if len(blocks) != expected:
        raise RuntimeError(
            f"Internal error: {len(result.frames)} frames must produce "
            f"{expected} content blocks, but produced {len(blocks)}."
        )
    return blocks


def _log_startup(config: Config) -> None:
    """Record the resolved paths on stderr.

    stdout belongs to the JSON-RPC stream, so diagnostics go to stderr.
    Printing the *actual* cache path is the difference between a five-minute
    and a fifty-minute debugging session later.
    """
    from .cli import ensure_utf8_stdio

    ensure_utf8_stdio()
    # Show the resolved path, not the raw setting: a relative cache directory
    # (which resolves against the working directory) is otherwise impossible
    # to locate from a log line, and that is exactly when someone reads it.
    cache_display = config.cache_root
    try:
        cache_display = config.cache_root.resolve()
    except OSError:  # pragma: no cover - unreadable parent
        pass
    lines = [
        f"mcp-video-frames: cache dir = {cache_display}",
        f"mcp-video-frames: max images/call = {config.max_images_per_call}"
        " (set MCP_VIDEO_MAX_IMAGES to match your client)",
    ]
    for line in lines:
        print(line, file=sys.stderr, flush=True)


def serve_stdio(core: Core | None = None) -> int:
    """Run the server over stdio (the MCP transport).

    The transport is named explicitly so the behaviour cannot change with a
    newer SDK default; older SDKs whose ``run`` takes no transport fall back
    to their own default, which is stdio.
    """
    server = build_server(core)
    run = server.run
    try:
        run(transport="stdio")
    except TypeError:
        run()
    return 0


def default_cache_hint() -> str:
    return str(default_cache_root())
