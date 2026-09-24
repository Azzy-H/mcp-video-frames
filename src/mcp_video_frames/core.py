"""The core library: one implementation of every capability.

Both front ends are thin translations of user input into these functions:

* ``server.py`` (MCP) turns them into ordered content blocks;
* ``cli.py`` turns them into JSON on stdout.

Neither re-implements extraction, probing or scanning.  If a rule changes
here, both change together — that is the entire reason this module exists.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import probe
from .budget import (
    FrameSpec,
    FramesPlan,
    PlanRequest,
    plan_frames,
)
from .cache import (
    Cache,
    CachedVideo,
    file_identity,
    maybe_prune_async,
)
from .config import Config
from .errors import FfmpegError, InputError, LimitError, VideoFramesError
from .ffmpeg import REQUIRED_FILTERS, Tools, check_cache_dir_writable, locate
from .frames import (
    FrameFile,
    check_video_readable,
    cleanup_workspace,
    extraction_workspace,
    extract_range,
    extract_single,
    validate_frame_file,
)
from .scan import run_full_scan

_SOURCE_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ts",
    ".m2ts",
    ".mts",
    ".ogv",
    ".3gp",
    ".vob",
    ".rmvb",
    ".mxf",
}


@dataclass
class FramesResult:
    """Everything the front ends need to render a ``view_frames`` result."""

    frames: list[FrameFile]
    summary: dict[str, Any]
    plan: FramesPlan
    video_path: Path
    identity: str


@dataclass
class VideoInfoResult:
    payload: dict[str, Any]


class Core:
    """Entry point shared by the MCP server and the CLI.

    Holds the resolved ffmpeg tools and the cache handle, and remembers
    ``basic`` metadata per file within the process so a follow-up frame
    request does not re-probe the container.
    """

    def __init__(self, config: Config | None = None, tools: Tools | None = None) -> None:
        self.config = config or Config.from_env()
        self._tools = tools
        self.cache = Cache(self.config)
        self._warned: list[str] = []
        self._startup_error: str | None = None

    # -- diagnostics ---------------------------------------------------
    @property
    def tools(self) -> Tools:
        if self._tools is None:
            self._tools = locate(
                ffmpeg_path=self.config.ffmpeg_path,
                ffprobe_path=self.config.ffprobe_path,
            )
            for warning in self._tools.warnings:
                if warning not in self._warned:
                    self._warned.append(warning)
        return self._tools

    def warnings(self) -> list[str]:
        """Non-fatal degradations observed so far, in plain language."""
        _ = self.tools  # ensure resolution has happened
        return list(self._warned)

    @property
    def startup_error(self) -> str | None:
        """Why the environment self check failed, if it did."""
        return self._startup_error

    def startup_check(self) -> list[str]:
        """Self check to run once at start, returning problems found.

        Failures are recorded and returned, not raised: a server that dies at
        startup leaves the client with "the command exited" and nothing else,
        whereas a server that starts and reports "ffmpeg not found, install
        it like this" is diagnosable from inside the client.
        """
        problems: list[str] = []
        try:
            _ = self.tools
        except VideoFramesError as exc:
            problems.append(str(exc))
        try:
            check_cache_dir_writable(self.config.cache_root)
        except VideoFramesError as exc:
            problems.append(str(exc))
        self._startup_error = "\n\n".join(problems) if problems else None
        return problems

    def doctor(self) -> dict[str, Any]:
        """Environment self check.  Reports what is wrong, and what to do."""
        report: dict[str, Any] = {
            "ok": True,
            "cache_root": str(self.config.cache_root),
            "config": self.config.describe(),
        }
        try:
            tools = self.tools
            report["ffmpeg"] = {
                "path": tools.ffmpeg,
                "version": tools.ffmpeg_version,
            }
            report["ffprobe"] = {
                "path": tools.ffprobe or None,
                "version": tools.ffprobe_version or None,
                "available": tools.has_ffprobe,
            }
            missing = [f for f in REQUIRED_FILTERS if not tools.has_filter(f)]
            report["filters"] = {
                "required": list(REQUIRED_FILTERS),
                "missing": missing,
            }
            if tools.warnings:
                report["warnings"] = list(tools.warnings)
        except VideoFramesError as exc:
            report["ok"] = False
            report["ffmpeg_error"] = str(exc)
            report["filters"] = {"required": list(REQUIRED_FILTERS), "missing": []}

        try:
            check_cache_dir_writable(self.config.cache_root)
            stats = self.cache_stats()
            report["cache"] = {
                "writable": True,
                "videos": stats["videos"],
                "bytes": stats["bytes"],
            }
        except VideoFramesError as exc:
            report["ok"] = False
            report["cache"] = {"writable": False, "error": str(exc)}

        report["max_images_per_call"] = self.config.max_images_per_call
        if self.config.max_images_per_call > 20:
            report["max_images_per_call_note"] = (
                f"MCP_VIDEO_MAX_IMAGES={self.config.max_images_per_call} is "
                f"high. Most clients cap a single result at 20 images or "
                f"fewer, so a higher value can make the whole batch unusable. "
                f"Adjust it to match your client."
            )
        return report

    # -- helpers -------------------------------------------------------
    def resolve_video(self, video: str) -> Path:
        if video is None or str(video).strip() == "":
            raise InputError(
                "The video argument is empty. Pass the absolute path of a "
                "video file."
            )
        path = Path(str(video)).expanduser()
        check_video_readable(path)
        try:
            return path.resolve()
        except OSError:
            return path

    def _cached_video_for(self, path: Path) -> CachedVideo:
        """Register ``path`` and return its cache entry."""
        identity, source = file_identity(path)
        return self.cache.register(identity, source)

    def video_basic_info(
        self, video: str, *, force: bool = False, use_cache: bool = True
    ) -> tuple[dict[str, Any], CachedVideo]:
        """``basic`` metadata, from cache when possible."""
        path = self.resolve_video(video)
        identity, source = file_identity(path)
        cached = self.cache.register(identity, source)
        if use_cache and not force:
            stored = self.cache.read_meta(cached, "basic")
            if stored is not None:
                self.cache.touch(identity)
                return stored, cached
        payload = probe.probe_basic(self.tools, path)
        payload["path"] = str(path)
        payload["size_bytes"] = source["size"]
        payload["sha256"] = identity
        self.cache.write_meta(cached, "basic", payload)
        return payload, cached

    def video_full_info(self, video: str, *, force: bool = False) -> dict[str, Any]:
        """``basic`` metadata plus the whole-file scan (cached, expensive)."""
        basic, cached = self.video_basic_info(video, force=force)
        payload = dict(basic)
        payload["detail"] = "full"

        if not force:
            stored = self.cache.read_meta(cached, "full")
            if stored is not None and _scan_is_complete(stored):
                payload.update(stored)
                payload["detail"] = "full"
                if isinstance(payload.get("scan"), dict):
                    payload["scan"]["cached"] = True
                self.cache.touch(cached.identity)
                return payload

        path = Path(str(basic["path"]))
        with self.cache.locked(cached.identity):
            if not force:
                stored = self.cache.read_meta(cached, "full")
                if stored is not None and _scan_is_complete(stored):
                    payload.update(stored)
                    payload["detail"] = "full"
                    if isinstance(payload.get("scan"), dict):
                        payload["scan"]["cached"] = True
                    return payload
            scanned = run_full_scan(
                self.tools,
                path,
                threshold=self.config.scene_threshold,
                duration=float(basic.get("duration") or 0.0) or None,
            )
            payload.update(scanned)
            payload["scan"] = {
                "ffmpeg_version": self.tools.ffmpeg_version,
                "computed_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "cached": False,
            }
            self.cache.write_meta(cached, "full", _full_payload(payload))
            self.cache.touch(cached.identity)
        return payload

    # -- frames --------------------------------------------------------
    def view_frames(
        self,
        video: str,
        *,
        start: float = 0.0,
        end: float | None = None,
        interval: float = 1.0,
        max_frames: int | None = None,
        max_edge: int = 768,
        image_format: str | None = None,
        quality: int = 85,
    ) -> FramesResult:
        """Extract frames as an ordered list; the caller decides how to send
        them.  Validation happens *here*, before anything is returned."""
        path = self.resolve_video(video)
        basic, cached = self.video_basic_info(video)
        duration = float(basic.get("duration") or 0.0)
        width = int(basic.get("width") or 0)
        height = int(basic.get("height") or 0)

        request = PlanRequest(
            start=start,
            end=end,
            interval=interval,
            max_frames=max_frames,
            max_edge=max_edge,
            image_format=image_format,
            quality=quality,
            duration=duration,
        )
        plan = plan_frames(
            request,
            max_images_per_call=self.config.max_images_per_call,
            duration=duration,
            max_image_bytes=self.config.max_image_bytes,
            max_total_bytes=self.config.max_total_bytes,
            width=width,
            height=height,
        )

        with self.cache.locked(cached.identity):
            frames, hits = self._ensure_frames(cached, path, plan)
            self.cache.touch(cached.identity)

        total_bytes = sum(frame.size for frame in frames)
        encoded = int(total_bytes * 4 / 3)
        if encoded > self.config.max_total_bytes:
            raise_video_error(
                f"These {len(frames)} frames total "
                f"{total_bytes / (1024 * 1024):.1f} MB, about "
                f"{encoded / (1024 * 1024):.1f} MB once base64-encoded, over "
                f"the per-call total limit of "
                f"{self.config.max_total_bytes / (1024 * 1024):.1f} MB. Lower "
                f"max_edge (now {plan.max_edge}), use format=jpeg (now "
                f"{plan.image_format}), or request fewer frames."
            )
        for frame in frames:
            if frame.size > self.config.max_image_bytes:
                raise_video_error(
                    f"Frame {frame.path.name} is "
                    f"{frame.size / (1024 * 1024):.1f} MB, over the per-frame "
                    f"limit of {self.config.max_image_bytes / (1024 * 1024):.1f} "
                    f"MB. Lower max_edge (now {plan.max_edge})."
                )

        summary: dict[str, Any] = {
            "video": str(path),
            "frames": len(frames),
            "range": [round(plan.range[0], 3), round(plan.range[1], 3)],
            "interval": plan.interval,
            "format": plan.image_format,
            "max_edge": plan.max_edge,
            "clamped": plan.clamped,
            "cached": hits,
            # The limits that were in force for this call.  A caller sees the
            # tool result but not the server's configuration, so repeating
            # them here is how it learns the budget without spending a round
            # trip discovering it by being rejected.
            "limits": {
                "max_images_per_call": self.config.max_images_per_call,
                "max_image_bytes": self.config.max_image_bytes,
                "max_total_bytes": self.config.max_total_bytes,
            },
        }
        if plan.clamped:
            summary["requested_range"] = [
                round(float(plan.requested_start), 3),
                round(float(plan.requested_end), 3),
            ]
            # State both sets of numbers: the range the caller asked for and
            # the range actually sampled, each as a concrete value.
            summary["clamp_note"] = (
                f"The requested range {plan.requested_start:g}-"
                f"{plan.requested_end:g}s exceeds the video's {duration:g}s, "
                f"so it was clamped to the "
                f"{summary['range'][0]:g}-{summary['range'][1]:g}s actually "
                f"used."
            )
        if plan.count == 1 and len(frames) != 1:
            raise_video_error(
                f"Internal error: a single-frame request must return 1 frame, "
                f"but returned {len(frames)}."
            )

        maybe_prune_async(self.cache)
        return FramesResult(
            frames=frames,
            summary=summary,
            plan=plan,
            video_path=path,
            identity=cached.identity,
        )

    def _ensure_frames(
        self, cached: CachedVideo, path: Path, plan: FramesPlan
    ) -> tuple[list[FrameFile], int]:
        """Return validated frames, extracting only the ones not on disk.

        Cached files are re-validated on every read: a frame can be evicted,
        truncated by a crash, or written by an older buggy version, and the
        response would otherwise carry corrupt bytes.  A miss is regenerated
        rather than reported as an error.
        """
        found: dict[int, FrameFile] = {}
        missing: list[FrameSpec] = []
        for spec in plan.frames:
            data = self.cache.read_frame_bytes(cached, spec.t, plan.image_format)
            if data is None:
                missing.append(spec)
                continue
            try:
                frame = self._validate(spec, cached, plan)
            except VideoFramesError:
                # Corrupt or over-budget *cached* frame: discard and
                # re-extract.  A frame that cannot be regenerated is caught in
                # _extract_missing, where a failure is fatal rather than a
                # retry.
                try:
                    cached.frame_path(spec.t, plan.image_format).unlink()
                except OSError:
                    pass
                missing.append(spec)
                continue
            found[spec.n] = frame

        hits = len(found)
        if missing:
            extracted = self._extract_missing(
                cached,
                path,
                plan,
                missing,
                full_range=not found,
            )
            found.update(extracted)

        ordered = [found[spec.n] for spec in plan.frames]
        return ordered, hits

    def _validate(
        self, spec: FrameSpec, cached: CachedVideo, plan: FramesPlan
    ) -> FrameFile:
        """Validate one frame file and tag it with its plan position."""
        frame = validate_frame_file(
            cached.frame_path(spec.t, plan.image_format),
            image_format=plan.image_format,
            max_image_bytes=self.config.max_image_bytes,
            max_image_dimension=self.config.max_image_dimension,
            timestamp=spec.t,
        )
        return FrameFile(
            n=spec.n,
            t=spec.t,
            path=frame.path,
            data=frame.data,
            mime_type=frame.mime_type,
            width=frame.width,
            height=frame.height,
        )

    def _extract_missing(
        self,
        cached: CachedVideo,
        path: Path,
        plan: FramesPlan,
        missing: list[FrameSpec],
        *,
        full_range: bool,
    ) -> dict[int, FrameFile]:
        """Extract the frames that are not on disk, then validate every one.

        ``full_range`` means nothing was cached, so a single ffmpeg call covers
        the whole grid and a short batch is an error: extraction must never
        hand back a partial batch that a caller would read as complete.  When
        only some frames are missing they are fetched one at a time, and a
        frame that cannot be produced surfaces as a KeyError on the plan —
        also a hard failure, never a shorter list.
        """
        workspace = extraction_workspace(cached.root, "frames")
        try:
            if len(plan.frames) == 1:
                produced = [
                    extract_single(
                        self.tools,
                        video=path,
                        t=plan.frames[0].t,
                        max_edge=plan.max_edge,
                        image_format=plan.image_format,
                        quality=plan.quality,
                        out_dir=workspace,
                    )
                ]
                mapping = [plan.frames[0]]
            elif full_range:
                produced = extract_range(
                    self.tools,
                    video=path,
                    start=plan.frames[0].t,
                    interval=plan.interval,
                    count=plan.count,
                    max_edge=plan.max_edge,
                    image_format=plan.image_format,
                    quality=plan.quality,
                    out_dir=workspace,
                )
                if len(produced) != plan.count:
                    raise FfmpegError(
                        f"Expected {plan.count} frames but got only "
                        f"{len(produced)} ({path.name}, range "
                        f"{plan.frames[0].t:g}-{plan.range[1]:g}s, interval "
                        f"{plan.interval:g}s). Frame timestamps are computed as "
                        f"start + i x interval, so a count mismatch makes them "
                        f"untrustworthy; the whole call fails rather than "
                        f"returning a partial batch.",
                        detail=f"Output directory: {workspace}",
                    )
                mapping = list(plan.frames)
            else:
                # Partial hit: extract only the gaps, one at a time.  A
                # range call would re-do work we already have on disk.
                produced = []
                mapping = []
                for spec in missing:
                    produced.append(
                        extract_single(
                            self.tools,
                            video=path,
                            t=spec.t,
                            max_edge=plan.max_edge,
                            image_format=plan.image_format,
                            quality=plan.quality,
                            out_dir=workspace,
                        )
                    )
                    mapping.append(spec)

            validated: dict[int, FrameFile] = {}
            for spec, source in zip(mapping, produced):
                # Freshly extracted bytes are validated and, unlike cached
                # ones, never retried: a failure here raises out of the tool
                # call.  Returning the frames that *did* work would hand the
                # caller a shorter batch that looks complete.
                frame = validate_frame_file(
                    source,
                    image_format=plan.image_format,
                    max_image_bytes=self.config.max_image_bytes,
                    max_image_dimension=self.config.max_image_dimension,
                    timestamp=spec.t,
                )
                self.cache.store_frame(
                    cached, spec.t, plan.image_format, frame.data
                )
                validated[spec.n] = FrameFile(
                    n=spec.n,
                    t=spec.t,
                    path=cached.frame_path(spec.t, plan.image_format),
                    data=frame.data,
                    mime_type=frame.mime_type,
                    width=frame.width,
                    height=frame.height,
                )
            return validated
        finally:
            cleanup_workspace(workspace)

    # -- cache ---------------------------------------------------------
    def cache_stats(self) -> dict[str, Any]:
        from .cache import cache_stats

        return cache_stats(self.cache)

    def prune_cache(self, *, max_bytes: int | None = None) -> dict[str, Any]:
        from .cache import prune

        report = prune(self.cache, max_bytes=max_bytes)
        payload = report.describe()
        payload["cache_root"] = str(self.cache.root)
        return payload

    # -- listing -------------------------------------------------------
    def iter_videos(self, directory: Path, *, recursive: bool = True) -> list[Path]:
        if not directory.is_dir():
            raise InputError(
                f"Not a directory: {directory}. Pass a directory that contains "
                f"video files."
            )
        pattern = "**/*" if recursive else "*"
        found = [
            p
            for p in sorted(directory.glob(pattern))
            if p.is_file() and p.suffix.lower() in _SOURCE_SUFFIXES
        ]
        if not found:
            raise InputError(
                f"No video files found in {directory} (recognised extensions: "
                f"{', '.join(sorted(_SOURCE_SUFFIXES))})."
            )
        return found


def _scan_is_complete(payload: dict[str, Any]) -> bool:
    """A cached full scan is usable only if all three parts are present.

    A half-written entry (crash, disk full, kill) must not be served as if
    the video had no scene cuts and no silence.
    """
    if not isinstance(payload, dict):
        return False
    for key in ("scene_cuts", "silences", "loudness"):
        if key not in payload:
            return False
    if not isinstance(payload.get("scene_cuts"), list):
        return False
    if not isinstance(payload.get("silences"), list):
        return False
    if not isinstance(payload.get("loudness"), dict):
        return False
    return True


def _full_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip ``basic``-only fields before persisting a full scan."""
    return {
        key: value
        for key, value in payload.items()
        if key in ("scene_cuts", "silences", "loudness", "scan")
    }


def raise_video_error(message: str, *, detail: str | None = None) -> None:
    raise LimitError(message, detail=detail)


def image_block(frame: FrameFile) -> dict[str, str]:
    """MCP ``image`` content block for one validated frame."""
    return {
        "type": "image",
        "data": base64.b64encode(frame.data).decode("ascii"),
        "mimeType": frame.mime_type,
    }


def timecode_block(n: int, t: float) -> dict[str, str]:
    """The text block that precedes each image: ``{"n":0,"t":0.0}``."""
    return {"type": "text", "text": json.dumps({"n": n, "t": round(t, 3)})}


def summary_block(summary: dict[str, Any]) -> dict[str, str]:
    return {
        "type": "text",
        "text": json.dumps(summary, ensure_ascii=False),
    }


def build_content_blocks(result: FramesResult) -> list[dict[str, str]]:
    """``n`` frames -> ``2n + 1`` blocks, in interleaved order.

    Order is a hard requirement: the timecode text block must immediately
    precede the image it describes, and the summary must come last.  This
    function is the only place that ordering is decided, and it is covered
    by an explicit test.
    """
    blocks: list[dict[str, str]] = []
    for frame in result.frames:
        blocks.append(timecode_block(frame.n, frame.t))
        blocks.append(image_block(frame))
    blocks.append(summary_block(result.summary))
    return blocks


def format_content_blocks(blocks: list[dict[str, str]]) -> str:
    """Render blocks as text, used by the CLI ``frames`` subcommand."""
    lines: list[str] = []
    for block in blocks:
        if block["type"] == "text":
            lines.append(block["text"])
        else:
            lines.append(
                f"[image {block['mimeType']} {len(block['data'])} base64 chars]"
            )
    return "\n".join(lines)
