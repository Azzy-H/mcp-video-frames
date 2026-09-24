# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

First release.

### Added

- `view_frames` MCP tool: samples a time range and returns each frame as an
  image preceded by its timecode, as `2n + 1` ordered content blocks.
- `video_info` MCP tool: `basic` container metadata, or `full` for a
  whole-file scan producing scene cuts, silence intervals and EBU R128
  loudness.
- Content-addressed cache with per-timestamp frame files, atomic writes,
  per-video file locks, layered eviction (frames first, whole-file scans
  last) and orphan collection. Cleanup is implemented here rather than
  delegated to the operating system, whose temporary-directory policy differs
  per platform.
- Pre-flight validation of every frame before it is returned: magic bytes
  against the declared MIME type, per-frame size, and pixel dimensions. A
  frame that fails anywhere fails the whole call — never a partial batch.
- `mcp-video-frames` CLI sharing one core library with the MCP server:
  `doctor`, `info`, `scan` (including `--dir`), `frames`, and
  `cache --stats|--prune`.
- Local deployment without installing the project: `setup_mcp.py` creates a
  private `.venv/` inside the checkout, installs only the runtime dependencies
  into it, and writes `mcp-config.json` with absolute paths. `run_server.py`
  puts `src/` on `sys.path`, so the client configuration needs no
  `PYTHONPATH`. Nothing is written outside the project directory, and deleting
  `.venv/` uninstalls it.
- `setup_mcp.py` asks what to do when ffmpeg is missing: download a prebuilt
  build into `.venv/`, accept a path you paste, or skip and write the config
  anyway. Non-interactive runs never block on the prompt. Re-running is
  idempotent — the virtual environment is reused and an ffmpeg you already
  confirmed is remembered rather than forgotten.
- The generated client entry puts the frame cache in the project's own
  `.cache/` and pins `cwd`, so a local deployment keeps everything in one
  deletable directory instead of the system temp directory that Windows never
  cleans. A relative cache path is only ever emitted together with `cwd`,
  because it resolves against the client's working directory. The package
  default stays the temp directory, which is the right choice for an installed
  distribution.
- Startup self check: ffmpeg and ffprobe discovery, version reporting, filter
  availability, and cache writability.
- ffprobe-optional metadata path: when only ffmpeg is installed, container
  metadata is parsed from ffmpeg's own banner and the degradation is reported
  rather than hidden.
- Both published MCP SDK generations: `FastMCP` (mcp 1.x) and the renamed
  `MCPServer` (mcp 2.x).
- Tool errors are raised as the SDK's `ToolError` when it exists, so the
  message reaches the client intact. mcp 2.x treats an arbitrary exception as
  a crash and replaces the text with a generic "Error executing tool ...",
  which would discard the numbers and the suggested fix that this server
  exists to provide.

[Unreleased]: https://github.com/Azzy-H/mcp-video-frames/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azzy-H/mcp-video-frames/releases/tag/v0.1.0
