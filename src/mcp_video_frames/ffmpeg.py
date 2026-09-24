"""ffmpeg / ffprobe: location, invocation, version and error wrapping.

ffmpeg is an *external binary*, deliberately not a Python dependency.  This
module is the only place that knows how to find it and how to run it, so
that a wrong path or a missing filter is reported once, clearly, instead of
turning into a mystery in the middle of a scan.

Nothing here is MCP-aware.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import FfmpegError, FfmpegNotFoundError

#: Filters this server depends on.  Checked at startup so a build without
#: them fails loudly instead of returning wrong results.
REQUIRED_FILTERS = ("select", "silencedetect", "ebur128", "showinfo", "scale", "fps")

#: How much stderr we are willing to carry inside an error message.
STDERR_EXCERPT_CHARS = 4000

#: Matches the version token in either banner: "ffmpeg version 6.1.1",
#: "ffprobe version 6.1.1 Copyright (c) ...".
_VERSION_RE = re.compile(r"ff(?:mpeg|probe) version (\S+)")


@dataclass(frozen=True)
class Tools:
    """Resolved external tools and their versions.

    ``ffprobe`` is optional: when it is missing, container metadata comes
    from ffmpeg's own banner instead (see :mod:`mcp_video_frames.banner`),
    and ``warnings`` explains the degraded path rather than failing silently.
    """

    ffmpeg: str
    ffprobe: str
    ffmpeg_version: str
    ffprobe_version: str
    ffmpeg_filters: frozenset[str]
    warnings: tuple[str, ...] = ()

    @property
    def has_ffprobe(self) -> bool:
        return bool(self.ffprobe)

    def has_filter(self, name: str) -> bool:
        return name.lower() in self.ffmpeg_filters


def _candidate_names(base: str) -> list[str]:
    names = [base]
    if os.name == "nt":
        names.extend([base + ".exe", base + ".cmd", base + ".bat"])
    return names


def _beside_interpreter(base: str) -> str | None:
    """Look for a binary in the directory holding the running interpreter.

    This is where a venv-local install lands: ``setup_mcp.py`` unpacks ffmpeg
    into ``.venv/Scripts`` (Windows) or ``.venv/bin``, so the server's own
    interpreter sits right next to it.  Without this step that ffmpeg is
    invisible until someone exports ``FFMPEG_PATH`` -- which is exactly the
    state a fresh deployment is in, and ``doctor`` would report a broken
    environment for a perfectly good one.  Only the interpreter's own directory
    is searched: not the whole PATH, and not the working directory.
    """
    here = Path(sys.executable).resolve().parent
    for name in _candidate_names(base):
        candidate = here / name
        if candidate.is_file():
            return str(candidate)
    return None


def find_binary(explicit: str | None, base: str) -> str | None:
    """Locate a binary: explicit setting, then beside the interpreter, then PATH.

    An explicit path that does not exist is *not* silently ignored; the
    caller gets a clear error instead of a surprise fallback.
    """
    if explicit:
        expanded = os.path.expanduser(explicit)
        candidate = Path(expanded)
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            for name in _candidate_names(base):
                inner = candidate / name
                if inner.is_file():
                    return str(inner)
        found = shutil.which(expanded)
        if found:
            return found
        raise FfmpegNotFoundError(
            f"{base.upper()}_PATH={explicit!r} does not point to an existing "
            f"file. Fix that environment variable, or remove it so the server "
            f"looks up {base} in its own environment and on PATH."
        )
    return _beside_interpreter(base) or shutil.which(base)


def locate(
    *,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    check_filters: bool = True,
) -> Tools:
    """Resolve ffmpeg (required) and ffprobe (preferred) and read versions.

    ffmpeg is mandatory.  ffprobe is not: a missing ffprobe downgrades
    container metadata to parsing ffmpeg's own banner, and the downgrade is
    recorded in ``Tools.warnings`` so it can be surfaced instead of being a
    silent behaviour change.

    Raises ``FfmpegNotFoundError`` when ffmpeg is missing, with install
    guidance rather than a stack trace.
    """
    ffmpeg = find_binary(ffmpeg_path, "ffmpeg")
    if ffmpeg is None:
        raise FfmpegNotFoundError(_missing_message("ffmpeg", "FFMPEG_PATH"))
    warnings: list[str] = []

    try:
        ffprobe = find_binary(ffprobe_path, "ffprobe")
    except FfmpegNotFoundError:
        raise
    if ffprobe is None:
        warnings.append(
            "ffprobe not found: container metadata falls back to parsing "
            "ffmpeg's own header output. Fewer fields are available (no "
            "chapters, no colour metadata). Installing a complete ffmpeg "
            "distribution restores them."
        )

    ffmpeg_version = _version_of(ffmpeg)
    ffprobe_version = _version_of(ffprobe) if ffprobe else ""
    filters = _filters_of(ffmpeg) if check_filters else frozenset()

    tools = Tools(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe or "",
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
        ffmpeg_filters=filters,
        warnings=tuple(warnings),
    )
    if check_filters:
        missing = [f for f in REQUIRED_FILTERS if not tools.has_filter(f)]
        if missing:
            raise FfmpegError(
                f"ffmpeg is missing required filters: {', '.join(missing)}. "
                f"Install an ffmpeg build with the full filter set (an "
                f"official build or a distribution's complete package).",
                detail=f"ffmpeg: {ffmpeg}\nversion: {ffmpeg_version}",
            )
    return tools


def _missing_message(binary: str, env_name: str) -> str:
    return (
        f"{binary} not found. This server needs {binary} to read video.\n"
        f"Looked in this project's virtual environment "
        f"({Path(sys.executable).parent}) and on PATH.\n"
        f"Install ffmpeg (it ships with ffprobe):\n"
        f"  - Windows: winget install Gyan.FFmpeg   or choco install ffmpeg\n"
        f"  - macOS:   brew install ffmpeg\n"
        f"  - Linux:   sudo apt-get install ffmpeg\n"
        f"Afterwards run `mcp-video-frames doctor` to re-check; if {binary} is "
        f"neither in the virtual environment nor on PATH, set {env_name} to its "
        f"full path."
    )


def _run(
    command: list[str],
    *,
    timeout: float | None = None,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=stdin_text,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FfmpegNotFoundError(
            f"Cannot execute {command[0]}: no such file. Check FFMPEG_PATH / "
            f"FFPROBE_PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(
            f"{Path(command[0]).name} was still running after {timeout:g}s and "
            f"was aborted. Shorten the range, or run scan first to pre-build "
            f"the cache for a long video.",
            command=list(command),
        ) from exc
    except OSError as exc:
        raise FfmpegError(
            f"Cannot execute {Path(command[0]).name}: {exc}.",
            command=list(command),
        ) from exc


def _version_of(binary: str) -> str:
    result = _run([binary, "-version"], timeout=30)
    text = (result.stdout or "") + (result.stderr or "")
    match = _VERSION_RE.search(text)
    if match:
        return match.group(1)
    first = text.strip().splitlines()
    if result.returncode == 0 and first:
        return first[0].strip()
    raise FfmpegError(
        f"Cannot read version information from {Path(binary).name} (exit code "
        f"{result.returncode}). Check that it is a working ffmpeg/ffprobe "
        f"binary.",
        command=[binary, "-version"],
        stderr=excerpt(text),
    )


def _filters_of(ffmpeg: str) -> frozenset[str]:
    result = _run([ffmpeg, "-hide_banner", "-filters"], timeout=60)
    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and not text.strip():
        raise FfmpegError(
            f"Cannot list ffmpeg filters (exit code {result.returncode}).",
            command=[ffmpeg, "-filters"],
            stderr=excerpt(text),
        )
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        # Rows look like: " TSC scale   V->V  Scale the input video size..."
        if len(parts) >= 2 and len(parts[0]) <= 3 and not parts[0].isdigit():
            names.add(parts[1].lower())
    return frozenset(names)


def excerpt(text: str | None, limit: int = STDERR_EXCERPT_CHARS) -> str:
    """Trim ffmpeg output for inclusion in an error message.

    ffmpeg's own diagnostics are the single most useful thing to include,
    but unbounded they swamp the message.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n... ({omitted} characters omitted) ...\n{text[-half:]}"


def run_ffmpeg(
    tools: Tools,
    args: list[str],
    *,
    timeout: float | None = None,
    expect_output: bool = True,
) -> str:
    """Run ffmpeg with ``-hide_banner -nostdin`` and return combined output.

    ``expect_output=False`` is for probes whose *goal* is a failing run
    (ffmpeg reports "no such stream" that way).  Any other non-zero exit is
    an error carrying the real stderr.
    """
    command = [tools.ffmpeg, "-hide_banner", "-nostdin", *args]
    result = _run(command, timeout=timeout)
    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and expect_output:
        raise FfmpegError(
            f"ffmpeg failed (exit code {result.returncode}). "
            f"Command: {' '.join(command)}",
            command=command,
            stderr=excerpt(text),
        )
    return text


def run_ffprobe(
    tools: Tools,
    args: list[str],
    *,
    timeout: float | None = None,
) -> str:
    command = [tools.ffprobe, "-hide_banner", *args]
    result = _run(command, timeout=timeout)
    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise FfmpegError(
            f"ffprobe failed (exit code {result.returncode}). "
            f"Command: {' '.join(command)}",
            command=command,
            stderr=excerpt(text),
        )
    return text


def check_cache_dir_writable(cache_root: Path) -> None:
    """Prove the cache directory is usable, without leaving a file behind."""
    from .errors import ConfigError

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(
            f"Cannot create the cache directory {cache_root}: {exc}. Set "
            f"MCP_VIDEO_CACHE_DIR to a writable directory."
        ) from exc
    probe = cache_root / ".write-test"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise ConfigError(
            f"The cache directory {cache_root} is not writable: {exc}. Set "
            f"MCP_VIDEO_CACHE_DIR to a writable directory."
        ) from exc
