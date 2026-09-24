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

### Fixed

- The per-video lock now outlasts the work it guards. It defaulted to 120
  seconds while one whole-file scan pass may run for 12 hours, so a second
  client touching a video another client was scanning waited two minutes and
  then failed — which contradicted the documented promise that clients share a
  cache. The lock timeout, the three scan timeouts and the fallback lock's
  staleness threshold now all derive from one constant, so they cannot drift
  apart again.
- A file ffmpeg cannot read no longer publishes a cache entry. The entry was
  registered before the probe ran, so pointing the tool at a text file or a
  corrupt download left a directory, a `source.json` and an index listing
  behind — it looked like a real video to `cache --stats`, inflated the totals,
  and could not be collected as an orphan because the source file still
  existed and still matched.
- Index updates are no longer lost to a concurrent writer. `read_index()` and
  `write_index()` each took the lock, but the read-modify-write *between* them
  did not, so a writer that had read before another committed would install its
  stale snapshot and delete the newer entry — a video registered while the
  background prune was running simply vanished from the index. Every mutation
  (`register`, `touch`, `prune`) now runs inside one `index_transaction()` that
  holds the lock across the whole sequence. The transaction is deliberately not
  reentrant: nesting would commit whichever in-memory copy finished last, so it
  raises instead, and the call paths keep mutations disjoint.
- `os.replace` is retried when the destination is momentarily busy. Registering
  a video writes `source.json` before it touches the index, so several clients
  asking for the same video at once rename onto the same path — and Windows
  fails that with `PermissionError` when the destination has a concurrent
  handle. Measured on a 12-thread stampede onto one entry, 7 writers failed
  without a retry and 0 failed with one, needing at most three attempts. Only
  `PermissionError` is retried, and only a bounded number of times: when the
  retries are exhausted the error is raised rather than swallowed, so a write
  that did not happen can never look like one that did. The uncontended path is
  unchanged (a try/except, no sleep).
- A file that exists but is not video now produces the same kind of message as
  a missing path, instead of handing the caller an ffprobe command line as the
  headline. ffmpeg's own output is kept in the error's detail.
- `cache.py` orphan collection carries a comment explaining why an entry whose
  `source.json` is unreadable is retained while its directory still exists:
  dropping it would desynchronise the index from the bytes on disk.
- `frames.extract_range`'s docstring no longer claims that a short batch is
  "legitimate only at the end of the input". It returns however many frames
  ffmpeg produced, and the caller decides; the caller treats a short batch as
  fatal.
- Removed a dead branch in `probe.py` that recomputed `is_hdr` from the same
  two inputs inside `if not is_hdr`, so it could never change the value. The
  10/12-bit pixel-format check it was paired with had no effect on any result.
- Removed the unused `use_cache` parameter from `Core.video_basic_info`, and a
  `cli.py` docstring that still described the messages as Chinese.

### Changed

- `frames --images-dir` no longer emits blocks that claim to be content blocks.
  It used to put the file path in an image block's `data` field — which per MCP
  **is** the base64 payload — keep `mimeType` as `image/jpeg`, and add an
  invented `dataEncoding` key. A consumer that trusted the shape decoded a
  filename and failed, and nothing at the top level said otherwise. The output
  is now `{"paths": true, "blocks": [...]}` with each image block's `data` a
  `file:` reference: the sequence, its order and `mimeType` are unchanged, so it
  still lines up with what `view_frames` returns, but the one thing that
  differs is now stated once instead of implied. (No released version emitted
  the old shape.)

[Unreleased]: https://github.com/Azzy-H/mcp-video-frames/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azzy-H/mcp-video-frames/releases/tag/v0.1.0
