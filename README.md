# mcp-video-frames — Video MCP Server

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![ffmpeg](https://img.shields.io/badge/ffmpeg-%23007808.svg?style=flat&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/) [![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-1f6feb.svg?style=flat)](https://modelcontextprotocol.io/)

An MCP server that lets image-only models look at video: frames with timecodes, plus measurable video properties.

## Available Tools

The server provides these tools:

- `view_frames` - Sample a time range and return frames as images, each preceded by its timecode
- `video_info` - Container metadata, optionally with a whole-file scene/silence/loudness scan

> The model you are talking to **must accept image input**. Without it, the returned frames are dropped or degraded to text.

This is a local server: it is not hosted anywhere, and a client has to be pointed at it. See **Self-Hosted** below.

## Self-Hosted

Installs nothing into your system Python: dependencies go into the project's own `.venv/`, and the client config points at that interpreter directly.

1. Clone the repository

```sh
git clone https://github.com/Azzy-H/mcp-video-frames.git
cd mcp-video-frames
```

2. Generate the virtual environment and the client config

```sh
python setup_mcp.py
```

`setup_mcp.py` only does three things:

1. creates `.venv/` in this directory (run it with any Python 3.10+ interpreter);
2. installs the two runtime dependencies from `requirements.txt` into that `.venv/`;
3. writes `mcp-config.json`, with the paths already filled in.

Options:

```sh
python setup_mcp.py --max-images 12      # match your client's image limit
python setup_mcp.py --print              # print the config only; write nothing
python setup_mcp.py --ffmpeg <path>      # ffmpeg is not on PATH
```

If `ffmpeg` is missing the script offers to download a prebuilt build into `.venv/` (Windows), to point at one you installed, or to skip. Nothing is ever installed system-wide.

3. Check the environment before wiring it up

```sh
.venv/Scripts/python run_server.py doctor     # Windows
.venv/bin/python     run_server.py doctor     # macOS / Linux
```

This reports the ffmpeg/ffprobe paths and versions, whether the required filters are present, and whether the cache directory is writable — naming whatever is missing rather than leaving you to guess. It finds a venv-local ffmpeg on its own, so nothing has to be exported first.

`test_server.py` goes further and speaks real MCP to the server over stdio, so the whole path can be verified before any client is configured:

```sh
.venv/Scripts/python test_server.py connect <video file path>   # Windows
.venv/bin/python     test_server.py connect <video file path>   # macOS / Linux
```

It checks the handshake, the tool list, the `2n + 1` content-block ordering, whether a single-frame request comes back as PNG, and that an over-limit request is refused — and that stdout stays clean for the JSON-RPC stream.

4. Add this MCP server

Merge the generated `mcp-config.json` into your client's config. If an `mcpServers` section already exists, just add the `video-frames` entry to it. `run_server.py` puts `src/` on `sys.path` itself, so **no `PYTHONPATH` is needed** and nothing has to be `pip install`ed anywhere.

**Replace** `/path/to/mcp-video-frames` with the actual path to this repository on your system. On Windows the interpreter is `.venv\Scripts\python.exe`.

**Uninstall**: delete `.venv/` and remove that config entry. Nothing else was written.

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe`. `setup_mcp.py` installs (or downloads) them into the project's own `.venv/`, and the server looks for them **beside its own interpreter first, then on `PATH`** — so a venv-local install needs no `FFMPEG_PATH` to be exported. `ffmpeg` is required; without `ffprobe`, metadata falls back to parsing ffmpeg's own header output (fewer fields), and `doctor` says so explicitly
- A client model that accepts image input

## Tools

### `view_frames`

Samples a time range and returns an interleaved `timecode + image` sequence. Set `start` and `end` to the same value to look at a single moment.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `video` | string | — | absolute path (required) |
| `start` | number | 0 | start in seconds |
| `end` | number | `start + interval` | end in seconds |
| `interval` | number | 1.0 | sampling interval, 0.05–600 s |
| `max_frames` | integer | 12 | frames for this call; may be raised up to the image limit |
| `max_edge` | integer | 768 | long-edge size in pixels, 64–4096 |
| `format` | `jpeg` \| `png` | `jpeg` for a range, `png` for one frame | |
| `quality` | integer | 85 | jpeg quality, 1–100 |

The result is `2n + 1` content blocks:

```
text:  {"n":0,"t":0.0}
image: <frame 0>
text:  {"n":1,"t":1.0}
image: <frame 1>
...
text:  {"video":"...","frames":12,"range":[0.0,11.0],"interval":1.0,
        "format":"jpeg","max_edge":768,"clamped":false,"cached":9,
        "limits":{"max_images_per_call":20,"max_image_bytes":10485760,
                  "max_total_bytes":41943040}}
```

The block order is load-bearing: an image without its timecode cannot be reasoned about in time. `clamped` reports whether the range was trimmed to the video's length, `cached` how many frames came from disk, and `limits` repeats the budget in force so the numbers do not have to be discovered by being rejected.

**A range defaults to jpeg** because at equal edge length it is about an order of magnitude smaller, which fits many more frames inside a client's byte budget. Ask for a single frame as png when detail matters.

**Over a limit the call fails: nothing is truncated and the interval is never widened.** Silently coarsening the sampling grid would invalidate the caller's assumption about time precision, so the whole call fails instead. If any single frame fails validation, no frames are returned.

### `video_info`

Returns video metadata. `detail="basic"` reads container headers only and returns immediately; `detail="full"` adds an exhaustive whole-file scan for scene cuts, silences and EBU R128 loudness.

```json
{
  "path": "", "sha256": "", "size_bytes": 0,
  "duration": 0.0, "fps": 0.0, "width": 0, "height": 0,
  "video_codec": "", "pix_fmt": "",
  "color_range": "", "color_space": "", "is_hdr": false,
  "audio_tracks": [], "subtitle_tracks": [], "chapters": [],
  "detail": "full",
  "scene_cuts": [1.23, 5.67],
  "silences": [{"start": 12.0, "end": 14.5, "duration": 2.5}],
  "loudness": {"integrated_lufs": -23.0, "true_peak_dbfs": -1.2, "lra": 5.0},
  "scan": {"ffmpeg_version": "", "computed_at": "", "cached": false}
}
```

With `detail="basic"` the `scene_cuts` / `silences` / `loudness` / `scan` fields are **absent** rather than empty — "no value" and "not scanned" mean different things. The first `full` call scans the whole file; later calls reuse the cache. `force=true` recomputes.

## Configuration

Environment variables take precedence over defaults.

| Variable | Default | Purpose |
|---|---|---|
| `MCP_VIDEO_MAX_IMAGES` | 20 | **maximum images per call** (the one that matters most) |
| `MCP_VIDEO_CACHE_DIR` | `.cache/` as generated by `setup_mcp.py`; `<system temp>/mcp-video-frames-cache` otherwise | cache root |
| `FFMPEG_PATH` / `FFPROBE_PATH` | auto-detected | checked before `PATH` |
| `MCP_VIDEO_MAX_IMAGE_BYTES` | 10485760 | per-frame byte limit |
| `MCP_VIDEO_MAX_TOTAL_BYTES` | 41943040 | per-result total byte limit |
| `MCP_VIDEO_MAX_IMAGE_DIM` | 8192 | per-frame edge limit |
| `MCP_VIDEO_CACHE_MAX_BYTES` | 5368709120 | cache size limit |
| `MCP_VIDEO_FRAME_MAX_AGE_DAYS` | 14 | frame age limit |
| `MCP_VIDEO_SCENE_THRESHOLD` | 0.3 | scene-change sensitivity |

`MCP_VIDEO_MAX_IMAGES` caps how many frames one call may return. **Set it to match your client**: limits differ between clients, and while too low is merely conservative, too high makes the entire batch unusable. `max_frames` defaults to 12 and may be raised only up to this limit.

The effective limits are also stated in the tool descriptions and repeated in every `view_frames` result, so a caller can see them without being rejected first.

## CLI

The same package ships a CLI for troubleshooting and batch work:

```bash
mcp-video-frames doctor               # environment self-check
mcp-video-frames info video.mp4       # basic metadata
mcp-video-frames scan video.mp4       # pre-build the full scan
mcp-video-frames scan --dir ./videos  # process a whole directory
mcp-video-frames frames video.mp4 --start 0 --end 5   # frames, as content-block JSON
mcp-video-frames cache --stats        # cache usage
mcp-video-frames cache --prune        # evict cache entries
```

Without installing the project (**Self-Hosted** above), use the launcher for the same subcommands:

```bash
.venv/Scripts/python run_server.py doctor            # Windows
.venv/bin/python     run_server.py info video.mp4    # macOS / Linux
```

Running it with no arguments prints help; as an MCP server it takes no arguments. When stdin is a pipe (as an MCP client provides) a bare invocation starts the server, which is how nothing has to be configured for the stdio case.

`doctor` prints its JSON report and then a one-line human verdict ("Self-check passed" / "Self-check failed"), so it is the one subcommand whose stdout is not pure JSON. The CLI and the MCP tools call the **same core functions**, so behaviour cannot drift between what works in a terminal and what works for a model.

## Cache

**Location**: depends on how the server was started.

| Deployment | Cache root |
|---|---|
| `setup_mcp.py` (the recommended one) | `.cache/` **inside this project** |
| A plain package install with no `MCP_VIDEO_CACHE_DIR` | `<system temp>/mcp-video-frames-cache` |

The setup script configures the project-local `.cache/` deliberately, together with `cwd`. A relative path resolves against the *client's* working directory, so without `cwd` the cache would land wherever the client happened to start — less predictable than the temp default, not more. It is already in `.gitignore`.

`MCP_VIDEO_CACHE_DIR` overrides either default, and the resolved path is printed to stderr at startup.

Everything in the cache is reproducible — deleting the directory is always safe, and only costs a re-probe.

Entries are keyed by content hash (size + mtime + the first and last 1 MB), and frames are stored individually by **absolute timestamp**, so overlapping views reuse work: look at 0–20 s, then 10–30 s, and 10–20 s is already on disk.

**If you left the cache in the system temp directory, do not rely on the operating system to clean it up** — the platforms differ too much:

| Platform | Temp-directory cleanup |
|---|---|
| Windows | Essentially none. Storage Sense runs only under disk pressure and skips files in use |
| Linux | `systemd-tmpfiles` ages out `/tmp` (~10 days); `/tmp` is often tmpfs and is cleared on reboot |
| macOS | `$TMPDIR` is cleaned by the system at unspecified times |

So the server cleans up after itself, evicting whole video entries by LRU once `MCP_VIDEO_CACHE_MAX_BYTES` (default 5 GB) is exceeded, plus a 14-day frame age limit (`MCP_VIDEO_FRAME_MAX_AGE_DAYS`).

The tiers follow **recomputation cost**: frames are cheap and go first, while whole-file scan results (`video_info(detail="full")`) are expensive to rebuild and are kept preferentially.

Eviction runs asynchronously after a tool call returns — it never blocks a response, and there is no full scan at startup.

Multiple clients share one cache. Reads and writes are guarded by a file lock, writes go through a temp file plus `os.replace`, and a frame that has been evicted between listing and reading is regenerated rather than reported as an error.

## License

MIT — see [LICENSE](LICENSE).
