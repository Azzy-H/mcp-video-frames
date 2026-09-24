#!/usr/bin/env python3
"""Test the server end to end without an MCP client.

An MCP client is just a program that speaks JSON-RPC over the server's
stdin/stdout.  This script does the same thing using the SDK's own client, so
you can confirm the whole chain works *before* wiring the server into any
desktop application.

Usage::

    # 1. the server starts and reports its environment
    .venv/bin/python test_server.py doctor

    # 2. the client connects, lists tools, and calls one
    .venv/bin/python test_server.py connect <video>

Exit code is 0 only if every step passed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
LAUNCHER = PROJECT_ROOT / "run_server.py"

PASS = "  [OK]"
FAIL = "  [!!]"


def _fail(message: str) -> None:
    print(FAIL + " " + message, file=sys.stderr)


def _is_error_result(result) -> bool:
    """Whether a tool result is flagged as an error, on either SDK generation.

    The flag is spelled ``is_error`` on mcp 2.x's pydantic model and
    ``isError`` on 1.x, and the pydantic model raises AttributeError rather
    than returning a default for the name it does not have — so the attribute
    has to be probed, not assumed.  This is the same camelCase/snake_case split
    the server handles for ``ImageContent.mimeType``.
    """
    for attribute in ("is_error", "isError"):
        if hasattr(result, attribute):
            return bool(getattr(result, attribute))
    return False


def _server_env() -> dict[str, str]:
    """Environment for the server child process.

    Returning a copy rather than ``None`` matters: the SDK starts the server
    with ONLY what is passed here plus a few OS essentials, so ``env=None``
    would drop the PATH and any FFMPEG_PATH the caller had set.  Locating
    ffmpeg itself is the server's job, not this script's -- it looks beside its
    own interpreter, which is where ``setup_mcp.py`` installs it.
    """
    return dict(os.environ)


def run_doctor() -> int:
    """Launch run_server.py exactly as a client would, and look at the output."""
    import subprocess

    print(f"Launching: {sys.executable} {LAUNCHER}")
    print("(simulating a client: stdin is a pipe, not a terminal)")
    print()

    # stdin must be a real *pipe*, which is what an MCP client hands over.
    # ``subprocess.DEVNULL`` is NOT equivalent on Windows: it opens the NUL
    # character device, and NUL reports ``isatty() == True``, so the server
    # would take the "a human is at a terminal" branch and print help.  An
    # empty pipe reads EOF immediately, so the server starts, sees EOF, and
    # exits — and we only care about what it wrote to stderr meanwhile.
    # The same environment the connect path uses: without it a venv-local
    # ffmpeg (which is where setup_mcp.py puts it) is invisible to the child,
    # and this check reports "ffmpeg not found" for a perfectly good install.
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER)],
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=_server_env(),
    )
    print("--- server stderr ---")
    print(completed.stderr.strip() or "(empty)")
    print("--- server stdout (the MCP protocol stream; must be empty here) ---")
    print(completed.stdout.strip() or "(empty)")
    print()

    problems = []
    if completed.returncode not in (0, 1):
        problems.append(f"unexpected exit code: {completed.returncode}")
    if "cache dir = " not in completed.stderr:
        problems.append("the cache directory was not logged (startup did not run?)")
    if "self-check failed" in completed.stderr:
        problems.append("the self-check failed; see stderr above")
    if completed.stdout.strip():
        problems.append("stdout must be empty: it would corrupt the JSON-RPC stream")

    if problems:
        for problem in problems:
            _fail(problem)
        print()
        print("Failed. Fix the environment using the output above, then run connect.")
        print("(the server looks for ffmpeg beside its own interpreter as well as")
        print(" on PATH, so installing it into the venv is enough -- no environment")
        print(" variable needs to be set beforehand.)")
        return 1

    print(PASS + " the server starts")
    print(PASS + " startup self-check passed, stdout is clean")
    print()
    print("Next:")
    print(f"    {sys.executable} {Path(__file__).name} connect <video file path>")
    return 0


async def _connect(video: Path) -> int:
    """Speak real MCP to the server over stdio and check what comes back."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        _fail(f"the mcp SDK is not available in this interpreter: {exc}")
        print("Run this script with the interpreter from .venv.", file=sys.stderr)
        return 1

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(LAUNCHER)],
        env=_server_env(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(PASS + " initialize handshake succeeded")

            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            print(f"  tools: {names}")
            if names != ["video_info", "view_frames"]:
                _fail("wrong tool list, expected ['video_info', 'view_frames']")
                return 1
            print(PASS + " the exposed tools are exactly view_frames and video_info")

            # --- video_info -------------------------------------------------
            # The payload is inspected *before* its fields are read.  A failure
            # here arrives as a JSON body carrying "error" rather than as a
            # flagged result, so indexing straight into it would replay the
            # mistake as a KeyError and hide the message that says what
            # actually went wrong.
            info = await session.call_tool("video_info", {"video": str(video)})
            raw = info.content[0].text
            try:
                payload = json.loads(raw)
            except ValueError:
                _fail(f"video_info did not return JSON: {raw[:200]}")
                return 1
            if _is_error_result(info) or "error" in payload:
                _fail(
                    "video_info failed: "
                    f"{payload.get('error') or raw[:200]}"
                )
                detail = payload.get("detail")
                if detail:
                    print(f"    {detail}", file=sys.stderr)
                return 1
            print(f"  video_info -> {payload['duration']}s "
                  f"{payload['width']}x{payload['height']} {payload['video_codec']}")
            for key in ("duration", "fps", "width", "height", "sha256"):
                if key not in payload:
                    _fail(f"video_info is missing the field {key}")
                    return 1
            print(PASS + " video_info(basic) has every expected field")

            # --- view_frames: content-block order ---------------------------
            result = await session.call_tool(
                "view_frames",
                {"video": str(video), "start": 0.0, "end": 3.0, "interval": 1.0},
            )
            kinds = [block.type for block in result.content]
            print(f"  content blocks ({len(kinds)}): {kinds}")
            expected = ["text", "image"] * 4 + ["text"]
            if kinds != expected:
                _fail(f"wrong content-block order, expected {expected}")
                return 1
            print(PASS + " content-block order = [text, image] xN + [text]")

            timecodes, images = [], []
            for block in result.content:
                if block.type == "text":
                    text = block.text
                    if text.startswith('{"n"'):
                        timecodes.append(json.loads(text))
                    else:
                        summary = json.loads(text)
                else:
                    images.append(getattr(block, "mimeType", None) or block.mime_type)
            print(f"  timecodes: {timecodes}")
            print(f"  image MIME types: {set(images)}")
            if [tc["n"] for tc in timecodes] != list(range(4)):
                _fail("timecode numbering is not contiguous")
                return 1
            if set(images) != {"image/jpeg"}:
                _fail(f"wrong image MIME type: {set(images)}")
                return 1
            if len(images) != 4:
                _fail(f"wrong image count: {len(images)}")
                return 1
            print(PASS + " every timecode pairs with its image, MIME types correct")
            print(f"  summary: {summary}")

            # --- a single frame must come back as png ----------------------
            single = await session.call_tool(
                "view_frames", {"video": str(video), "start": 2.0, "end": 2.0}
            )
            single_kinds = [block.type for block in single.content]
            single_mime = [
                getattr(b, "mimeType", None) or b.mime_type
                for b in single.content
                if b.type == "image"
            ]
            if single_kinds != ["text", "image", "text"]:
                _fail(f"a single-frame request must return 3 blocks, got {single_kinds}")
                return 1
            if single_mime != ["image/png"]:
                _fail(f"a single frame must default to png, got {single_mime}")
                return 1
            print(PASS + " a single-frame request (start==end) returns one PNG")

            # --- over the limit must fail, never truncate ------------------
            over = await session.call_tool(
                "view_frames",
                {"video": str(video), "start": 0.0, "end": 600.0, "interval": 0.25},
            )
            if not _is_error_result(over):
                _fail("an over-limit request did not return an error -- silent degradation!")
                return 1
            message = over.content[0].text
            print("  over-limit error text (first line): " + message.splitlines()[0])
            if "interval" not in message:
                _fail("the over-limit error carries no actionable interval advice")
                return 1
            print(PASS + " an over-limit request is flagged as an error, with interval advice")

    print()
    print("All checks passed: an MCP client can drive this server.")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command = args[0]
    if command == "doctor":
        return run_doctor()
    if command == "connect":
        if len(args) < 2:
            _fail("usage: test_server.py connect <video file path>")
            return 2
        video = Path(args[1]).expanduser()
        if not video.is_file():
            _fail(f"video file does not exist: {video}")
            return 2
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        return asyncio.run(_connect(video))
    _fail(f"unknown command: {command}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
