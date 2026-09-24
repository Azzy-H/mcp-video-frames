"""Command line front end.

The CLI exists so a human can diagnose and batch-prepare the same core
library the MCP server uses.  It calls exactly the same functions — no
ffmpeg command is assembled here.  Duplicating extraction or scanning logic
is how the two front ends drift apart, and drift is how "it works from the
terminal but not from the model" happens.

Model-visible surface is unaffected: the CLI is invisible to MCP clients,
so the tool list stays at exactly ``view_frames`` and ``video_info``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import default_cache_root, known_env_names
from .core import Core
from .errors import FfmpegNotFoundError, InputError, VideoFramesError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

PROGRAM = "mcp-video-frames"


def ensure_utf8_stdio() -> None:
    """Force UTF-8 for our own stdout/stderr.

    JSON is emitted on stdout and messages on stderr, and both can carry
    non-ASCII text — a video's path, for instance.  On a Windows console with a
    legacy code page (cp1252, cp936, ...) those would be mangled into
    unreadable bytes, and a redirected stdout would be written as UTF-16, which
    no JSON parser accepts.  Both front ends call this, so the output is the
    same regardless of platform default.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - exotic streams
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Let models without video input look at video.\n"
            "Run with no arguments to start the MCP server (stdio); "
            "run with a subcommand to use the CLI below."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The cache directory defaults to "
            "<system temp>/mcp-video-frames-cache and can be overridden with "
            "MCP_VIDEO_CACHE_DIR; deleting it is always safe.\n\n"
            f"Recognised environment variables: {', '.join(known_env_names())}"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROGRAM} {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser(
        "doctor",
        help="self-check: ffmpeg/ffprobe paths and versions, required filters, cache writability",
        description="Self-check. Whatever is missing is named, with how to fix it.",
    )

    info = sub.add_parser(
        "info",
        help="basic metadata (same as video_info detail=\"basic\")",
        description=(
            "Read container information and print JSON. Equivalent to the MCP "
            "tool video_info(detail=\"basic\")."
        ),
    )
    info.add_argument("video", help="path to a video file")
    info.add_argument(
        "--force", action="store_true", help="ignore the cache and re-read"
    )

    scan = sub.add_parser(
        "scan",
        help="pre-build whole-file scan results (scene cuts / silence / loudness)",
        description=(
            "Exhaustively scan the whole file and write the result to the "
            "cache, so a later video_info(detail=\"full\") hits it directly. "
            "This is a whole-file operation and is naturally batched."
        ),
    )
    scan.add_argument("videos", nargs="*", help="one or more video file paths")
    scan.add_argument(
        "--dir", dest="directory", help="run over every video file in a directory"
    )
    scan.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subdirectories of --dir"
    )
    scan.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    # Accepted for interface stability. Slow/network storage can make a
    # sequential full read faster than random seeks, but the scans below are
    # already single-pass whole-file operations, so the flag currently
    # changes nothing and is therefore hidden from --help.
    scan.add_argument(
        "--sequential-read", action="store_true", help=argparse.SUPPRESS
    )

    frames = sub.add_parser(
        "frames",
        help="extract frames (same code path as view_frames) and emit content-block JSON",
        description=(
            "Extract frames and emit exactly the content-block sequence the MCP "
            "tool view_frames produces. With --images-dir the images are written "
            "to files instead and the sequence is wrapped in an envelope that "
            "says so, because a block whose data holds a path is not a valid "
            "content block."
        ),
    )
    frames.add_argument("video", help="path to a video file")
    frames.add_argument("--start", type=float, default=0.0, help="start time in seconds (default 0)")
    frames.add_argument("--end", type=float, default=None, help="end time in seconds")
    frames.add_argument(
        "--interval", type=float, default=1.0, help="sampling interval in seconds (default 1.0)"
    )
    frames.add_argument("--max-frames", type=int, default=None, help="maximum frames for this call")
    frames.add_argument("--max-edge", type=int, default=768, help="long-edge size in pixels")
    frames.add_argument(
        "--format", dest="image_format", default=None, help="jpeg or png"
    )
    frames.add_argument("--quality", type=int, default=85, help="jpeg quality, 1-100")
    frames.add_argument(
        "--images-dir",
        default=None,
        help=(
            "write the images to this directory; the output becomes "
            "{'paths': true, 'blocks': [...]} where each image block's data is "
            "a file: reference rather than base64"
        ),
    )
    frames.add_argument(
        "--no-content-blocks",
        action="store_true",
        help="print the summary JSON only, without per-frame content blocks",
    )

    cache = sub.add_parser(
        "cache",
        help="cache usage statistics / manual eviction",
        description="Everything in the cache is reproducible, so deleting it is always safe.",
    )
    cache.add_argument(
        "--stats", action="store_true", help="print cache usage statistics"
    )
    cache.add_argument("--prune", action="store_true", help="run eviction manually")
    cache.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="target size to evict down to (defaults to MCP_VIDEO_CACHE_MAX_BYTES)",
    )

    return parser


def print_help() -> None:
    build_parser().print_help()
    print()
    print(f"Cache directory: {default_cache_root()}")


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    ensure_utf8_stdio()
    parser = build_parser()
    if not args_list:
        print_help()
        return EXIT_OK

    args = parser.parse_args(args_list)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    try:
        core = Core()
        if args.command == "doctor":
            return cmd_doctor(core)
        if args.command == "info":
            return cmd_info(core, args)
        if args.command == "scan":
            return cmd_scan(core, args)
        if args.command == "frames":
            return cmd_frames(core, args)
        if args.command == "cache":
            return cmd_cache(core, args)
    except FfmpegNotFoundError as exc:
        _fail(exc)
        return EXIT_ERROR
    except VideoFramesError as exc:
        _fail(exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrupted.", file=sys.stderr)
        return 130
    parser.print_help()
    return EXIT_USAGE


def _fail(exc: VideoFramesError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.detail:
        print(exc.detail, file=sys.stderr)
    hint = getattr(exc, "stderr", None)
    if hint:
        print("--- ffmpeg output ---", file=sys.stderr)
        print(hint, file=sys.stderr)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------
def cmd_doctor(core: Core) -> int:
    report = core.doctor()
    _print_json(report)
    if report.get("ok"):
        print(
            "\nSelf-check passed: ffmpeg/ffprobe available, required filters "
            "present, cache directory writable."
        )
        return EXIT_OK
    print(
        "\nSelf-check failed; fix the problems reported above.", file=sys.stderr
    )
    return EXIT_ERROR


def cmd_info(core: Core, args: argparse.Namespace) -> int:
    payload, _cached = core.video_basic_info(args.video, force=args.force)
    _print_json(payload)
    return EXIT_OK


def cmd_scan(core: Core, args: argparse.Namespace) -> int:
    targets: list[Path] = [Path(v) for v in (args.videos or [])]
    if args.directory:
        targets.extend(
            core.iter_videos(Path(args.directory), recursive=not args.no_recursive)
        )
    if not targets:
        raise InputError(
            "No video to scan. Give one or more file paths, or use --dir "
            "<directory> to process a whole directory."
        )

    failures = 0
    for target in targets:
        try:
            payload = core.video_full_info(str(target), force=args.force)
            scan_meta = payload.get("scan") or {}
            print(
                json.dumps(
                    {
                        "video": str(target),
                        "scene_cuts": len(payload.get("scene_cuts") or []),
                        "silences": len(payload.get("silences") or []),
                        "loudness": payload.get("loudness") or {},
                        "cached": bool(scan_meta.get("cached")),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except VideoFramesError as exc:
            failures += 1
            print(f"Error: {target}: {exc.message}", file=sys.stderr, flush=True)
            if exc.detail:
                print(exc.detail, file=sys.stderr, flush=True)
    return EXIT_ERROR if failures else EXIT_OK


#: Prefix marking an image block's ``data`` as a file reference, not base64.
PATH_SCHEME = "file:"


def _write_frame_files(
    result, blocks: list[dict[str, str]], out_dir: Path
) -> list[dict[str, str]]:
    """Write each frame to disk and point its block at the file.

    The result is a *document*, not the content-block sequence: an MCP image
    block's ``data`` field **is** the base64 payload, so a block whose ``data``
    holds a path is not a valid one — a consumer that trusts the shape decodes
    a filename and fails, and ``mimeType`` describes bytes that are not there.
    Emitting that shape anyway is worse than emitting a different one, because
    the difference is invisible.

    So the block sequence is kept intact — same count, same order, same ``type``
    and ``mimeType``, so the output still lines up with what ``view_frames``
    returns — and only the image blocks' ``data`` becomes a ``file:`` reference.
    The envelope's ``paths`` key says so once, at the top.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if result.plan.image_format == "png" else "jpg"
    by_index = {frame.n: frame for frame in result.frames}
    document: list[dict[str, str]] = []
    # Image blocks sit at odd indices: [text, image] * n + [text].
    for index, block in enumerate(blocks):
        if index % 2 == 0 or index >= 2 * len(result.frames):
            document.append(block)
            continue
        frame = by_index[index // 2]
        target = out_dir / f"n{frame.n:04d}_t{frame.t:09.3f}.{ext}"
        target.write_bytes(frame.data)
        replaced = dict(block)
        replaced["data"] = f"{PATH_SCHEME}{target}"
        document.append(replaced)
    return document


def cmd_frames(core: Core, args: argparse.Namespace) -> int:
    result = core.view_frames(
        args.video,
        start=args.start,
        end=args.end,
        interval=args.interval,
        max_frames=args.max_frames,
        max_edge=args.max_edge,
        image_format=args.image_format,
        quality=args.quality,
    )
    from .core import build_content_blocks

    blocks = build_content_blocks(result)

    if args.images_dir:
        out_dir = Path(args.images_dir)
        blocks = _write_frame_files(result, blocks, out_dir)

    if args.no_content_blocks:
        _print_json({"summary": result.summary, "blocks": len(blocks)})
    elif args.images_dir:
        # Wrapped, because the block sequence alone would be indistinguishable
        # from real content blocks.  `paths` is the one signal a consumer needs.
        _print_json({"paths": True, "blocks": blocks})
        print(json.dumps(result.summary, ensure_ascii=False), file=sys.stderr)
    else:
        _print_json(blocks)
        print(json.dumps(result.summary, ensure_ascii=False), file=sys.stderr)
    return EXIT_OK


def cmd_cache(core: Core, args: argparse.Namespace) -> int:
    if not args.stats and not args.prune:
        raise InputError("Specify --stats or --prune.")
    if args.stats:
        _print_json(core.cache_stats())
    if args.prune:
        _print_json(core.prune_cache(max_bytes=args.max_bytes))
    return EXIT_OK


__all__ = [
    "main",
    "build_parser",
    "print_help",
    "ensure_utf8_stdio",
    "cmd_doctor",
    "cmd_frames",
]
