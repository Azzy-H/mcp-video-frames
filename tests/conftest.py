"""Shared test fixtures.

The tests import the package from ``src/`` without installing it, so the
unit tests run anywhere — including CI jobs that only want the pure-logic
checks and have no ffmpeg.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Temporary space for tests.  Kept inside the project because some sandboxes
#: deny writes elsewhere — and a suite that cannot create a temp file reports
#: noise, not signal.  Names are deterministic so the same sandbox can reason
#: about them.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_BASE = _PROJECT_ROOT / ".test-tmp"

_counter = itertools.count()


@pytest.fixture(scope="session", autouse=True)
def _prepare_tmp_base() -> Iterator[None]:
    os.makedirs(TMP_BASE, exist_ok=True)
    yield


@pytest.fixture(scope="session")
def tmp_path_factory():
    """Unused stand-in.

    pytest's own factory deletes and recreates ``--basetemp`` at session
    start, which fails under restrictive sandboxes.  The fixtures below
    create their directories directly instead.
    """
    return None


@pytest.fixture()
def tmp_path(tmp_path_factory) -> Iterator[Path]:
    path = TMP_BASE / f"case-{os.getpid()}-{next(_counter):04d}"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    yield path


@pytest.fixture()
def tmp_cache_dir(tmp_path, monkeypatch):
    """Point the cache at a fresh directory for the duration of a test."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("MCP_VIDEO_CACHE_DIR", str(cache_dir))
    return cache_dir


@pytest.fixture()
def config(tmp_cache_dir):
    from mcp_video_frames.config import Config

    return Config.from_env(
        {
            "MCP_VIDEO_CACHE_DIR": str(tmp_cache_dir),
            "MCP_VIDEO_MAX_IMAGES": "20",
        }
    )


@pytest.fixture()
def cache(config):
    from mcp_video_frames.cache import Cache

    return Cache(config)


# ----------------------------------------------------------------------
# Minimal stand-ins for the parts of the MCP SDK the server touches
# ----------------------------------------------------------------------
class FakeTextContent:
    def __init__(self, type="text", text="", **kwargs):
        self.type = type
        self.text = text


class FakeImageContent:
    def __init__(self, type="image", data="", mimeType="", **kwargs):
        self.type = type
        self.data = data
        self.mimeType = mimeType


class FakeToolError(Exception):
    """Stand-in for the SDK's ``ToolError``.

    Its presence matters: raising *this* is what makes the SDK pass our error
    text through verbatim.  Raising anything else is treated as a crash, and
    mcp 2.x then replaces the message with a generic "Error executing tool".
    """


class FakeFastMCP:
    """Records registered tools and exposes them as plain callables.

    Enough surface for the server to build itself; nothing that could start
    a transport, because a test that starts a server is not a unit test.
    """

    def __init__(self, name, instructions=None, **kwargs):
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, dict] = {}

    def tool(self, name=None, description=None, **kwargs):
        def decorator(func):
            import inspect

            annotations = dict(getattr(func, "__annotations__", {}))
            annotations.pop("return", None)
            self.tools[name or func.__name__] = {
                "func": func,
                "description": description or "",
                "schema": annotations,
                "signature": inspect.signature(func),
            }
            return func

        return decorator

    def run(self, transport="stdio", **kwargs):  # pragma: no cover - never used
        raise AssertionError("tests must not start a real server")


def install_fake_mcp_sdk(monkeypatch, *, with_tool_error: bool = True) -> None:
    """Make ``import mcp...`` resolve to the stand-ins above.

    The server module is dropped from ``sys.modules`` so the next import
    re-binds against these objects.

    ``with_tool_error=False`` simulates an SDK that has no ``ToolError``, which
    is the case the server must still start under.
    """
    import types as _types

    mcp_module = _types.ModuleType("mcp")
    types_module = _types.ModuleType("mcp.types")
    types_module.TextContent = FakeTextContent
    types_module.ImageContent = FakeImageContent
    server_module = _types.ModuleType("mcp.server")
    fastmcp_module = _types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    exceptions_module = _types.ModuleType("mcp.server.fastmcp.exceptions")
    if with_tool_error:
        exceptions_module.ToolError = FakeToolError
    mcp_module.types = types_module
    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    fastmcp_module.exceptions = exceptions_module
    for name, module in [
        ("mcp", mcp_module),
        ("mcp.types", types_module),
        ("mcp.server", server_module),
        ("mcp.server.fastmcp", fastmcp_module),
        ("mcp.server.fastmcp.exceptions", exceptions_module),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "mcp_video_frames.server", raising=False)


def _ffmpeg_binary() -> str | None:
    """ffmpeg to use in tests: FFMPEG_PATH first, then PATH."""
    explicit = os.environ.get("FFMPEG_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffmpeg")


def _ffprobe_binary() -> str | None:
    explicit = os.environ.get("FFPROBE_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffprobe")


def make_png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A genuine, decodable PNG of the requested size.

    Built rather than pasted: a hand-written blob with one wrong character
    produces a confusing failure in an unrelated test.
    """
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x40\x80\xc0\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    """A minimal but structurally valid baseline JPEG.

    Long enough to clear the server's "too small to be a real frame" guard,
    so it exercises the parser rather than the size check.
    """
    import struct

    def segment(marker: int, payload: bytes) -> bytes:
        return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload

    sof0 = (
        bytes([8])
        + struct.pack(">HH", height, width)
        + bytes([3])
        + bytes([1, 0x22, 0])
        + bytes([2, 0x11, 1])
        + bytes([3, 0x11, 1])
    )
    padding = bytes(range(1, 256)) * 2  # keeps the file above the size floor
    return (
        b"\xff\xd8"
        + segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
        + segment(0xDB, bytes([0]) + bytes([1] * 64))
        + segment(0xC0, sof0)
        + segment(0xC4, bytes([0]) + bytes([0] * 16) + bytes([0]))
        + segment(0xDA, bytes([3, 1, 0, 2, 0, 3, 0, 0, 63, 0]))
        + padding
        + b"\xff\xd9"
    )


def _has_ffmpeg() -> bool:
    return _ffmpeg_binary() is not None and _ffprobe_binary() is not None


requires_ffmpeg = pytest.mark.skipif(
    not _has_ffmpeg(), reason="ffmpeg/ffprobe not found (PATH or FFMPEG_PATH)"
)


@pytest.fixture(scope="session")
def test_video() -> Path:
    """A 10 s synthetic clip with a hard scene change at 5 s.

    Generated by ffmpeg itself, so the fixture needs no binary assets in the
    repository and stays honest about what it contains.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    out = TMP_BASE / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    out = out / "test.mp4"
    if out.is_file():
        return out
    # Two 5 s sources concatenated: testsrc for movement, a solid colour for
    # the second half.  The join at t=5 gives exactly one unambiguous scene
    # cut, which is what the scan assertions look for.
    filter_complex = "[0:v][1:v]concat=n=2:v=1:a=0[v]"
    command = [
        _ffmpeg_binary() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=5:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:duration=5:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=10",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-t",
        "10",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Older ffmpeg builds may lack libx264; fall back to mpeg4.
        command[command.index("libx264")] = "mpeg4"
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not generate test video: {result.stderr[-500:]}")
    return out
