"""Error types shared by the core library and both front ends.

Rationale: this server must never silently degrade.  Anything that is
truncated, clamped or dropped has to surface as an error carrying three
things: what happened, the concrete numbers involved, and a usable next
step.  ``VideoFramesError`` is the single error type the front ends catch
and turn into an MCP ``isError: true`` result or a CLI message + exit code.
"""

from __future__ import annotations


class VideoFramesError(Exception):
    """Base class for every error this package raises on purpose.

    ``message`` is written for a caller (model or human), not for a
    traceback: no stack, concrete numbers, actionable advice.
    """

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.detail:
            return f"{self.message}\n\n{self.detail}"
        return self.message


class ConfigError(VideoFramesError):
    """Invalid configuration (bad environment variable, unwritable cache)."""


class InputError(VideoFramesError):
    """Bad tool/subcommand arguments or a missing/unreadable input file."""


class CacheError(VideoFramesError):
    """The cache directory is unusable, or a lock could not be acquired."""


class LimitError(VideoFramesError):
    """A hard limit was exceeded.

    Raised instead of truncating, clamping the sampling grid, or returning
    a partial batch.
    """


class FfmpegNotFoundError(VideoFramesError):
    """ffmpeg or ffprobe could not be located."""


class FfmpegError(VideoFramesError):
    """ffmpeg/ffprobe ran and failed, or produced unparsable output."""

    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        command: list[str] | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message, detail=detail)
        self.command = command
        self.stderr = stderr
