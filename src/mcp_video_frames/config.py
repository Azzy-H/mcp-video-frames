"""Configuration: environment variables over built-in defaults.

Every knob is read from the environment so a single deployment can be
tuned per MCP client without touching code.  Nothing here talks to MCP.

``MCP_VIDEO_MAX_IMAGES`` is the most important setting: it must be less
than or equal to the image-count limit of the client you run.  Different
clients differ; too low is safe, too high makes the whole batch unusable.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

#: Frame-count defaults (see IMPLEMENTATION section four).
DEFAULT_INTERVAL = 1.0
DEFAULT_MAX_FRAMES = 12
DEFAULT_MAX_EDGE = 768
DEFAULT_QUALITY = 85

MIN_INTERVAL = 0.05
MAX_INTERVAL = 600.0
MIN_MAX_EDGE = 64
MAX_MAX_EDGE = 4096
MIN_QUALITY = 1
MAX_QUALITY = 100
MIN_MAX_FRAMES = 1

DEFAULT_MAX_IMAGES = 20
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_IMAGE_DIM = 8192
DEFAULT_CACHE_MAX_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_FRAME_MAX_AGE_DAYS = 14
DEFAULT_SCENE_THRESHOLD = 0.3

DEFAULT_CACHE_DIRNAME = "mcp-video-frames-cache"
CACHE_DIR_ENV = "MCP_VIDEO_CACHE_DIR"

_ENV_PREFIX = "MCP_VIDEO_"


def default_cache_root() -> Path:
    """Resolve the cache root without loading full configuration.

    Used by ``--help``-style output and by the self check, which must be
    able to report the path even when the rest of the config is broken.
    """
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIRNAME


def _env_int(raw: str | None, name: str, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{name}={raw!r} is not an integer. "
            f"Set it to an integer or remove it to use the default ({default})."
        ) from exc
    return value


def _env_float(raw: str | None, name: str, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{name}={raw!r} is not a number. "
            f"Set it to a number or remove it to use the default ({default})."
        ) from exc
    return value


def _env_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the process configuration."""

    cache_root: Path
    ffmpeg_path: str | None
    ffprobe_path: str | None
    max_images_per_call: int = DEFAULT_MAX_IMAGES
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIM
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES
    frame_max_age_days: float = DEFAULT_FRAME_MAX_AGE_DAYS
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    #: Names of environment variables that were set, for logging.
    env_seen: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        """Build a config from the environment.

        Pass ``env`` (a plain mapping) to override the process environment
        wholesale — useful in tests.  Only names this package understands
        are consulted.
        """
        source: dict[str, str] = dict(os.environ) if env is None else dict(env)

        def int_of(name: str, default: int) -> int:
            return _env_int(source.get(name), name, default)

        def float_of(name: str, default: float) -> float:
            return _env_float(source.get(name), name, default)

        def str_of(name: str) -> str | None:
            return _env_str(source.get(name))

        cache_dir = str_of(CACHE_DIR_ENV)
        return cls(
            cache_root=(
                Path(cache_dir).expanduser() if cache_dir else default_cache_root()
            ),
            ffmpeg_path=str_of("FFMPEG_PATH"),
            ffprobe_path=str_of("FFPROBE_PATH"),
            max_images_per_call=int_of(_ENV_PREFIX + "MAX_IMAGES", DEFAULT_MAX_IMAGES),
            max_image_bytes=int_of(
                _ENV_PREFIX + "MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES
            ),
            max_total_bytes=int_of(
                _ENV_PREFIX + "MAX_TOTAL_BYTES", DEFAULT_MAX_TOTAL_BYTES
            ),
            max_image_dimension=int_of(
                _ENV_PREFIX + "MAX_IMAGE_DIM", DEFAULT_MAX_IMAGE_DIM
            ),
            cache_max_bytes=int_of(
                _ENV_PREFIX + "CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES
            ),
            frame_max_age_days=float_of(
                _ENV_PREFIX + "FRAME_MAX_AGE_DAYS", DEFAULT_FRAME_MAX_AGE_DAYS
            ),
            scene_threshold=float_of(
                _ENV_PREFIX + "SCENE_THRESHOLD", DEFAULT_SCENE_THRESHOLD
            ),
            env_seen=tuple(sorted(k for k in source if k in _KNOWN_ENV)),
        )

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------
    @property
    def cache_videos_dir(self) -> Path:
        return self.cache_root / "videos"

    @property
    def index_path(self) -> Path:
        return self.cache_root / "index.json"

    @property
    def frame_max_age_seconds(self) -> float:
        return max(0.0, float(self.frame_max_age_days)) * 86400.0

    def describe(self) -> dict[str, object]:
        """Human/machine readable summary used by ``doctor`` and startup logs."""
        return {
            "cache_root": str(self.cache_root),
            "ffmpeg_path": self.ffmpeg_path,
            "ffprobe_path": self.ffprobe_path,
            "max_images_per_call": self.max_images_per_call,
            "max_image_bytes": self.max_image_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_image_dimension": self.max_image_dimension,
            "cache_max_bytes": self.cache_max_bytes,
            "frame_max_age_days": self.frame_max_age_days,
            "scene_threshold": self.scene_threshold,
        }


_KNOWN_ENV: tuple[str, ...] = (
    CACHE_DIR_ENV,
    "FFMPEG_PATH",
    "FFPROBE_PATH",
    _ENV_PREFIX + "MAX_IMAGES",
    _ENV_PREFIX + "MAX_IMAGE_BYTES",
    _ENV_PREFIX + "MAX_TOTAL_BYTES",
    _ENV_PREFIX + "MAX_IMAGE_DIM",
    _ENV_PREFIX + "CACHE_MAX_BYTES",
    _ENV_PREFIX + "FRAME_MAX_AGE_DAYS",
    _ENV_PREFIX + "SCENE_THRESHOLD",
)


def known_env_names() -> tuple[str, ...]:
    """The environment variables this package reads, for docs and doctor."""
    return _KNOWN_ENV
