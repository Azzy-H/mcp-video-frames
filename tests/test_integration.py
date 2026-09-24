"""Integration: real ffmpeg, real files, real cache reuse.

Acceptance items from the design that can only be checked with ffmpeg:

* block order for a five-frame request (11 blocks, alternating);
* over-limit is an error with a usable suggestion, not a truncation;
* one bad frame fails the whole call — never a partial batch;
* overlapping requests reuse cached frames (file mtimes unchanged);
* ``video_info`` layering: ``basic`` is instant, ``full`` caches;
* a frame deleted from the cache is regenerated, not an error;
* two processes extracting the same range do not corrupt anything;
* ``mcp-video-frames info`` and ``video_info(detail="basic")`` agree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp_video_frames.cache import frame_filename, frame_timestamps
from mcp_video_frames.config import Config
from mcp_video_frames.core import Core, build_content_blocks
from mcp_video_frames.errors import FfmpegError, InputError, LimitError

#: The checkout root, for files that exist but are not video.
ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


def _rename_works(directory: Path) -> bool:
    """Can a *child process* rename a file over an existing one?

    Atomic writes depend on ``os.replace``.  A hardened sandbox can deny it for
    processes it did not itself start, which makes the failure an environment
    property rather than a code defect — so it is detected once and reported as
    a skipped test rather than a failure.

    The probe must run in a child, because that is what the concurrency test
    does: in one observed sandbox the parent could rename freely while every
    child it spawned got ``WinError 5``.  Probing in-process would have looked
    like a pass and left a permanent false failure behind.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ".rename-probe"
    scratch = directory / ".rename-probe.tmp"
    program = (
        "import os, sys;"
        "os.replace(sys.argv[2], sys.argv[1]);"
        "print(open(sys.argv[1], 'rb').read())"
    )
    try:
        target.write_bytes(b"one")
        scratch.write_bytes(b"two")
        result = subprocess.run(
            [sys.executable, "-c", program, str(target), str(scratch)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0 and "two" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        for path in (target, scratch):
            try:
                path.unlink()
            except OSError:
                pass


requires_atomic_rename = pytest.mark.skipif(
    not _rename_works(Path(__file__).resolve().parents[1] / ".tmp" / "rename-probe"),
    reason="a child process cannot rename over an existing file here (os.replace denied)",
)


@pytest.fixture()
def core():
    """A Core with its own cache directory, or a skip when ffmpeg is missing.

    Function scoped on purpose: these tests assert on cache state (cold vs.
    warm, evicted frames, regenerated frames), so sharing one cache between
    them would make the results depend on execution order.
    """
    from conftest import TMP_BASE, _ffmpeg_binary, _ffprobe_binary

    ffmpeg_bin = _ffmpeg_binary()
    ffprobe_bin = _ffprobe_binary()
    if not ffmpeg_bin or not ffprobe_bin:
        pytest.skip("ffmpeg/ffprobe not found (PATH, FFMPEG_PATH or FFPROBE_PATH)")

    global _CACHE_SEQ
    _CACHE_SEQ += 1
    cache_dir = TMP_BASE / f"integration-{os.getpid()}-{_CACHE_SEQ:04d}"
    config = Config.from_env(
        {
            "MCP_VIDEO_CACHE_DIR": str(cache_dir),
            "MCP_VIDEO_MAX_IMAGES": "20",
            # Pinned so the tools the tests exercise are the ones that were
            # found, not whatever happens to be first on PATH later.
            "FFMPEG_PATH": ffmpeg_bin,
            "FFPROBE_PATH": ffprobe_bin,
        }
    )
    instance = Core(config)
    try:
        instance.tools  # forces ffmpeg/ffprobe lookup
    except Exception as exc:  # noqa: BLE001 - reported as a skip
        pytest.skip(f"ffmpeg/ffprobe unusable: {exc}")
    return instance


_CACHE_SEQ = 0


def child_env(core: Core) -> dict[str, str]:
    """Environment for a subprocess that must find ffmpeg and this cache."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["MCP_VIDEO_CACHE_DIR"] = str(core.config.cache_root)
    env["FFMPEG_PATH"] = core.tools.ffmpeg
    env["FFPROBE_PATH"] = core.tools.ffprobe
    return env


def run_cli(core: Core, *args: str, stdin=subprocess.PIPE, timeout: float = 120.0):
    """Run the CLI in a subprocess and decode its output as UTF-8.

    The child writes UTF-8 by design; decoding with the host locale would fail
    on a Windows default code page.
    """
    return subprocess.run(
        [sys.executable, "-m", "mcp_video_frames", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env(core),
        stdin=stdin,
        timeout=timeout,
    )


class TestDoctor:
    def test_doctor_reports_a_usable_environment(self, core):
        report = core.doctor()
        assert report["ok"] is True
        assert report["ffmpeg"]["version"]
        assert report["filters"]["missing"] == []
        assert report["cache"]["writable"] is True

    def test_metadata_source_is_reported(self, core, test_video):
        payload, _cached = core.video_basic_info(str(test_video))
        assert payload["metadata_source"] in ("ffprobe", "ffmpeg-banner")
        if core.tools.has_ffprobe:
            assert payload["metadata_source"] == "ffprobe"


class TestMissingFfprobeFallback:
    """ffprobe is preferred, not required — and never silently swapped."""

    def test_banner_fallback_produces_usable_metadata(self, test_video):
        from conftest import _ffmpeg_binary, _ffprobe_binary
        from mcp_video_frames.banner import probe_basic_without_ffprobe
        from mcp_video_frames.ffmpeg import Tools, locate

        full = locate(
            ffmpeg_path=_ffmpeg_binary(), ffprobe_path=_ffprobe_binary()
        )
        banner_only = Tools(
            ffmpeg=full.ffmpeg,
            ffprobe="",
            ffmpeg_version=full.ffmpeg_version,
            ffprobe_version="",
            ffmpeg_filters=full.ffmpeg_filters,
        )
        payload = probe_basic_without_ffprobe(banner_only, Path(test_video))
        assert payload["metadata_source"] == "ffmpeg-banner"
        assert payload["duration"] == pytest.approx(10.0, abs=0.5)
        assert (payload["width"], payload["height"]) == (320, 240)
        assert payload["fps"] == pytest.approx(30.0, abs=0.5)
        assert payload["video_codec"]
        assert len(payload["audio_tracks"]) == 1

    def test_missing_ffprobe_is_announced_as_a_warning(self, monkeypatch, test_video):
        from conftest import _ffmpeg_binary
        from mcp_video_frames import ffmpeg as ffmpeg_module
        from mcp_video_frames.core import Core

        real_find = ffmpeg_module.find_binary

        def no_ffprobe(explicit, base):
            if base == "ffprobe":
                return None
            return real_find(explicit, base)

        monkeypatch.setattr(ffmpeg_module, "find_binary", no_ffprobe)
        core = Core()
        assert core.tools.ffmpeg == _ffmpeg_binary()
        assert core.tools.has_ffprobe is False

        warnings = core.warnings()
        assert warnings, "a degraded metadata path must be visible"
        assert "ffprobe" in warnings[0]

        payload, _cached = core.video_basic_info(str(test_video))
        assert payload["metadata_source"] == "ffmpeg-banner"
        assert payload["duration"] == pytest.approx(10.0, abs=0.5)


class TestVideoInfo:
    def test_basic_is_fast_and_complete(self, core, test_video):
        start = time.monotonic()
        payload, _cached = core.video_basic_info(str(test_video))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"basic metadata took {elapsed:.2f}s"
        assert payload["duration"] == pytest.approx(10.0, abs=0.5)
        assert (payload["width"], payload["height"]) == (320, 240)
        assert payload["fps"] == pytest.approx(30.0, abs=0.5)
        assert payload["video_codec"]
        assert len(payload["audio_tracks"]) == 1
        assert payload["sha256"]
        assert payload["size_bytes"] > 0
        # basic must not pretend to have scanned
        for key in ("scene_cuts", "silences", "loudness", "scan"):
            assert key not in payload

    def test_full_scan_finds_the_engineered_scene_cut(self, core, test_video):
        payload = core.video_full_info(str(test_video), force=True)
        assert payload["detail"] == "full"
        assert payload["scan"]["cached"] is False
        cuts = payload["scene_cuts"]
        assert any(abs(cut - 5.0) < 0.5 for cut in cuts), cuts
        assert isinstance(payload["silences"], list)
        assert "integrated_lufs" in payload["loudness"]

    def test_full_scan_is_cached_on_the_second_call(self, core, test_video):
        first = core.video_full_info(str(test_video))
        assert first["scan"]["cached"] is False
        second = core.video_full_info(str(test_video))
        assert second["scan"]["cached"] is True
        assert second["scene_cuts"] == first["scene_cuts"]

    def test_force_recomputes(self, core, test_video):
        core.video_full_info(str(test_video))
        forced = core.video_full_info(str(test_video), force=True)
        assert forced["scan"]["cached"] is False


class TestUnreadableFileLeavesNoTrace:
    """A probe that fails must not publish a cache entry.

    The entry used to be registered *before* the probe ran, so pointing the
    tool at something ffmpeg cannot read (a text file, a corrupt download) left
    a permanent entry: its directory and ``source.json`` existed and the index
    listed them, which made it look like a real video to ``cache --stats`` and
    put it beyond orphan collection — the source file still existed and still
    matched.
    """

    def test_failed_probe_registers_nothing(self, core):
        with pytest.raises(InputError):
            core.video_basic_info(str(ROOT / "README.md"))

        index_path = core.cache.root / "index.json"
        assert not index_path.exists(), (
            "a failed probe created an index: "
            f"{index_path.read_text(encoding='utf-8') if index_path.exists() else ''}"
        )

    def test_failed_probe_leaves_no_video_directory(self, core):
        with pytest.raises(InputError):
            core.video_basic_info(str(ROOT / "README.md"))
        videos = core.cache.root / "videos"
        leftovers = list(videos.iterdir()) if videos.is_dir() else []
        assert leftovers == [], leftovers

    def test_cache_stats_stays_empty_after_a_failed_probe(self, core):
        with pytest.raises(InputError):
            core.video_basic_info(str(ROOT / "README.md"))
        stats = core.cache_stats()
        assert stats["videos"] == 0
        assert stats["entries"] == []

    def test_a_good_video_still_registers_afterwards(self, core, test_video):
        """The reordering must not have broken the normal path."""
        with pytest.raises(InputError):
            core.video_basic_info(str(ROOT / "README.md"))
        payload, _cached = core.video_basic_info(str(test_video))
        assert payload["duration"] == pytest.approx(10.0, abs=0.5)
        assert core.cache_stats()["videos"] == 1

    def test_non_video_file_reports_an_input_error_not_a_command_line(self, core):
        """The caller's *message* is clean; ffmpeg's words move to the detail.

        Both are still sent (see ``server._tool_error``), which is deliberate —
        the diagnosis stays available — but the headline a model reasons about
        is now the same kind of sentence a missing path produces, instead of a
        command line.
        """
        with pytest.raises(InputError) as excinfo:
            core.video_basic_info(str(ROOT / "README.md"))
        exc = excinfo.value
        assert "README.md" in exc.message
        assert "video file" in exc.message
        # The command line is no longer the headline.
        assert "ffprobe failed" not in exc.message
        assert "ffprobe failed" in str(exc.detail)

    def test_missing_path_and_unreadable_file_agree(self, core):
        with pytest.raises(InputError) as missing:
            core.video_basic_info(str(ROOT / "definitely-absent.mp4"))
        with pytest.raises(InputError) as unreadable:
            core.video_basic_info(str(ROOT / "README.md"))
        assert type(missing.value) is type(unreadable.value) is InputError

    def test_an_unreadable_source_json_is_not_reclaimed_while_its_bytes_exist(
        self, core, test_video
    ):
        """Documents the orphan rule: a surviving directory keeps its entry.

        Dropping such an entry would desynchronise the index from the bytes on
        disk, which both ``cache --stats`` and the LRU sizing measure.
        """
        core.video_basic_info(str(test_video))
        identity = core.cache.read_index()["entries"]
        assert len(identity) == 1
        (key,) = identity
        (core.cache.video_dir(key) / "source.json").unlink()

        report = core.prune_cache()
        assert report["removed_orphans"] == 0
        assert core.cache.read_index()["entries"], "entry was dropped"


class TestViewFrames:
    def test_block_order_for_five_frames(self, core, test_video):
        # 0-4 s at 1 s intervals: five grid points, all of which exist.
        result = core.view_frames(str(test_video), start=0.0, end=4.0, interval=1.0)
        blocks = build_content_blocks(result)
        assert result.summary["frames"] == 5
        assert len(blocks) == 11
        assert [block["type"] for block in blocks] == ["text", "image"] * 5 + ["text"]
        for index in range(5):
            assert json.loads(blocks[2 * index]["text"]) == {
                "n": index,
                "t": float(index),
            }
        summary = json.loads(blocks[-1]["text"])
        assert summary["range"] == [0.0, 4.0]
        assert summary["interval"] == 1.0
        assert summary["format"] == "jpeg"
        assert summary["clamped"] is False

    def test_last_grid_point_at_the_very_end_may_not_exist(self, core, test_video):
        """A 10.0 s clip has no frame at 10.0: 0-10 s yields 10 frames.

        The plan counts 11 grid points; the verified file count is what gets
        reported, and the block sequence must stay consistent with it.
        """
        result = core.view_frames(str(test_video), start=0.0, end=10.0, interval=1.0)
        assert result.summary["frames"] == len(result.frames)
        assert len(result.frames) in (10, 11)
        blocks = build_content_blocks(result)
        assert len(blocks) == 2 * len(result.frames) + 1
        assert [block["type"] for block in blocks] == (
            ["text", "image"] * len(result.frames) + ["text"]
        )
        assert result.frames[-1].t == float(len(result.frames) - 1)

    def test_single_frame_defaults_to_png(self, core, test_video):
        result = core.view_frames(str(test_video), start=3.0, end=3.0)
        assert result.summary["frames"] == 1
        assert result.summary["format"] == "png"
        assert result.frames[0].mime_type == "image/png"
        assert result.frames[0].width is not None

    def test_range_defaults_to_jpeg(self, core, test_video):
        result = core.view_frames(str(test_video), start=0.0, end=1.0, interval=1.0)
        assert result.summary["format"] == "jpeg"
        assert all(frame.mime_type == "image/jpeg" for frame in result.frames)

    def test_over_limit_raises_with_a_usable_suggestion(self, core, test_video):
        # 0-600 s at 0.25 s intervals: even after clamping to the clip's length
        # the request is far past the limit.
        with pytest.raises(LimitError) as excinfo:
            core.view_frames(
                str(test_video), start=0.0, end=600.0, interval=0.25, max_frames=20
            )
        message = str(excinfo.value)
        assert "split into" in message
        assert "interval" in message
        assert "MCP_VIDEO_MAX_IMAGES" in message
        # The suggestion must work: doing what it says must succeed.
        suggested = float(message.split("raise interval to >= ")[1].split("\n")[0])
        result = core.view_frames(
            str(test_video), start=0.0, end=600.0, interval=suggested, max_frames=20
        )
        assert 0 < result.summary["frames"] <= 20

    def test_interval_is_honoured_exactly(self, core, test_video):
        result = core.view_frames(str(test_video), start=0.0, end=1.0, interval=0.25)
        # Timestamps are the requested grid, from the start, in order.
        assert [round(frame.t, 3) for frame in result.frames] == [
            0.25 * i for i in range(len(result.frames))
        ]
        # The last grid point (1.0) may or may not exist; everything before it
        # must.
        assert [round(f.t, 3) for f in result.frames][:4] == [0.0, 0.25, 0.5, 0.75]

    def test_frame_count_matches_disk_files(self, core, test_video):
        """The timestamp convention is backed by the produced file count."""
        result = core.view_frames(
            str(test_video), start=0.0, end=5.0, interval=0.5, max_frames=20
        )
        frames_dir = core.cache.video_dir(result.identity) / "frames"
        for frame in result.frames:
            path = frames_dir / frame_filename(frame.t, "jpeg")
            assert path.is_file(), frame.t
        # Timestamps are strictly the grid, in order, with no gaps.
        assert [f.t for f in result.frames] == [
            0.5 * i for i in range(len(result.frames))
        ]

    def test_clamped_range_is_reported_with_both_values(self, core, test_video):
        result = core.view_frames(str(test_video), start=0.0, end=500.0, interval=1.0)
        assert result.summary["clamped"] is True
        assert result.summary["requested_range"] == [0.0, 500.0]
        assert result.summary["range"][1] < 500.0
        assert result.summary["clamp_note"]
        assert result.summary["frames"] == len(result.frames) > 0
        assert result.frames[-1].t == float(len(result.frames) - 1)

    def test_scale_filter_respects_max_edge(self, core, test_video):
        result = core.view_frames(
            str(test_video), start=1.0, end=1.0, max_edge=64, image_format="png"
        )
        frame = result.frames[0]
        assert max(frame.width, frame.height) <= 64

    def test_quality_changes_jpeg_size(self, core, test_video):
        low = core.view_frames(
            str(test_video),
            start=2.0,
            end=2.0,
            image_format="jpeg",
            quality=1,
            max_edge=240,
        )
        high = core.view_frames(
            str(test_video),
            start=2.0,
            end=2.0,
            image_format="jpeg",
            quality=100,
            max_edge=240,
        )
        assert low.frames[0].size <= high.frames[0].size


class TestCacheReuse:
    def test_overlapping_ranges_reuse_frames(self, core, test_video):
        first = core.view_frames(str(test_video), start=0.0, end=9.0, interval=1.0)
        assert first.summary["cached"] == 0
        assert len(first.frames) == 10

        cached_root = core.cache.video_dir(first.identity) / "frames"
        shared = [frame_filename(t, "jpeg") for t in (2.0, 5.0, 8.0)]
        before = {
            name: (cached_root / name).stat().st_mtime_ns
            for name in shared
            if (cached_root / name).is_file()
        }
        assert len(before) == len(shared), "the first call should have cached them"

        time.sleep(0.02)
        second = core.view_frames(str(test_video), start=2.0, end=9.0, interval=1.0)
        assert second.summary["cached"] >= 3
        for name, mtime in before.items():
            assert (cached_root / name).stat().st_mtime_ns == mtime

    def test_deleted_frame_is_regenerated_not_fatal(self, core, test_video):
        result = core.view_frames(str(test_video), start=4.0, end=4.0, image_format="jpeg")
        path = core.cache.video_dir(result.identity) / "frames" / frame_filename(
            4.0, "jpeg"
        )
        assert path.is_file()
        path.unlink()

        again = core.view_frames(str(test_video), start=4.0, end=4.0, image_format="jpeg")
        assert path.is_file()
        assert again.frames[0].size > 0

    def test_truncated_frame_is_regenerated_not_returned(self, core, test_video):
        """Acceptance item 3, first half: corrupt bytes must never be sent."""
        result = core.view_frames(str(test_video), start=0.0, end=4.0, interval=1.0)
        frames_dir = core.cache.video_dir(result.identity) / "frames"
        victim = frames_dir / frame_filename(1.0, "jpeg")
        assert victim.is_file()
        # Keep it non-empty but invalid, so "empty file" handling is not what
        # is being tested.
        victim.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)

        again = core.view_frames(str(test_video), start=0.0, end=4.0, interval=1.0)
        assert len(again.frames) == len(result.frames)
        assert all(frame.data.startswith(b"\xff\xd8\xff") for frame in again.frames)
        # Regenerated bytes are a real image, not the stub that was planted.
        assert again.frames[1].size > 100
        assert again.frames[1].width is not None

    def test_unrecoverable_frame_fails_the_whole_batch(self, core, test_video):
        """Acceptance item 3, second half: no partial batches when it counts.

        Regeneration rescues a *corrupt cache file*.  When extraction itself
        yields bytes that are not an image, there is nothing to rescue: the
        call must fail rather than return a shorter batch that a caller would
        read as complete.
        """
        from mcp_video_frames import core as core_module
        from mcp_video_frames.errors import FfmpegError as _FfmpegError

        # Prove the cold path is what is exercised: no cached frames at all, so
        # the whole range must be extracted in one call.
        if core.cache.videos_dir.exists():
            for entry in core.cache.videos_dir.iterdir():
                shutil.rmtree(entry, ignore_errors=True)
        assert not list(core.cache.videos_dir.glob("*/frames/*.jpg"))

        # Patched where the core actually looks it up, so the substitution
        # cannot be bypassed by a second reference to the same function.
        original = core_module.extract_range
        calls: list[int] = []

        def broken(*args, **kwargs):
            paths = original(*args, **kwargs)
            calls.append(len(paths))
            for path in paths:
                path.write_bytes(b"definitely not an image")
            return paths

        core_module.extract_range = broken
        try:
            with pytest.raises(_FfmpegError) as excinfo:
                core.view_frames(
                    str(test_video), start=0.0, end=2.0, interval=1.0, max_edge=128
                )
        finally:
            core_module.extract_range = original
        assert calls, "the range extraction must have run at least once"
        assert "partial batch" in str(excinfo.value)

    def test_partial_cache_hit_does_not_refetch_everything(self, core, test_video):
        """Half-warm cache: only the gaps are extracted."""
        first = core.view_frames(str(test_video), start=0.0, end=4.0, interval=1.0)
        frames_dir = core.cache.video_dir(first.identity) / "frames"
        assert len(first.frames) == 5
        for t in (2.0, 3.0):
            path = frames_dir / frame_filename(t, "jpeg")
            if path.is_file():
                path.unlink()

        second = core.view_frames(str(test_video), start=0.0, end=4.0, interval=1.0)
        # Three of the five frames were still on disk.
        assert second.summary["cached"] == 3
        assert len(second.frames) == 5
        assert all(frame.width for frame in second.frames)

    def test_cache_has_no_temporary_files_left_behind(self, core, test_video):
        result = core.view_frames(str(test_video), start=6.0, end=6.0, image_format="jpeg")
        frames_dir = core.cache.video_dir(result.identity) / "frames"
        leftovers = [
            name
            for name in os.listdir(frames_dir)
            if name.startswith(".") or name.endswith(".tmp")
        ]
        assert leftovers == []
        root_leftovers = [
            path.name
            for path in core.cache.video_dir(result.identity).iterdir()
            if "extract-" in path.name
        ]
        assert root_leftovers == []

    def test_cache_count_matches_expected_files(self, core, test_video):
        result = core.view_frames(
            str(test_video), start=1.0, end=4.0, interval=1.0, image_format="jpeg"
        )
        frames_dir = core.cache.video_dir(result.identity) / "frames"
        present = frame_timestamps(frames_dir)
        for frame in result.frames:
            assert any(abs(t - frame.t) < 1e-6 for t in present)


class TestConcurrency:
    @requires_atomic_rename
    def test_two_processes_extracting_the_same_range(self, core, test_video):
        """Acceptance item 7: no corrupt output from concurrent extraction."""
        script = (
            "import sys;"
            "from mcp_video_frames.config import Config;"
            "from mcp_video_frames.core import Core;"
            "core = Core(Config.from_env({"
            f"'MCP_VIDEO_CACHE_DIR': {str(core.config.cache_root)!r},"
            f"'FFMPEG_PATH': {core.tools.ffmpeg!r},"
            f"'FFPROBE_PATH': {core.tools.ffprobe!r},"
            "'MCP_VIDEO_MAX_IMAGES': '20'}));"
            "result = core.view_frames(sys.argv[1], start=0.0, end=6.0, interval=1.0);"
            "print(result.summary['frames'])"
        )
        env = child_env(core)
        workers = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(test_video)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            for _ in range(2)
        ]
        outputs = [worker.communicate() for worker in workers]
        counts = set()
        for worker, (out, err) in zip(workers, outputs):
            assert worker.returncode == 0, err
            counts.add(out.strip())
        # Both processes must agree, whatever the interleaving.
        assert len(counts) == 1, counts
        assert counts.pop() == str(
            core.view_frames(str(test_video), start=0.0, end=6.0, interval=1.0).summary[
                "frames"
            ]
        )

        # Whatever the interleaving, every cached frame must still be a valid
        # image: a torn write would leave a short or truncated file.
        result = core.view_frames(str(test_video), start=0.0, end=6.0, interval=1.0)
        for frame in result.frames:
            assert frame.data.startswith(b"\xff\xd8\xff")
            assert frame.width and frame.height


class TestCliParity:
    def test_info_matches_video_info_basic(self, core, test_video):
        """Acceptance item 10."""
        completed = run_cli(core, "info", str(test_video))
        assert completed.returncode == 0, completed.stderr
        cli_payload = json.loads(completed.stdout)
        mcp_payload, _cached = core.video_basic_info(str(test_video))
        for key in ("duration", "fps", "width", "height", "video_codec", "sha256"):
            assert cli_payload[key] == mcp_payload[key], key

    def test_frames_command_mirrors_the_mcp_content_blocks(self, core, test_video):
        """The two front ends must produce the same block sequence."""
        completed = run_cli(
            core,
            "frames",
            str(test_video),
            "--start",
            "0",
            "--end",
            "2",
            "--interval",
            "1",
        )
        assert completed.returncode == 0, completed.stderr
        cli_blocks = json.loads(completed.stdout)
        mcp_blocks = build_content_blocks(
            core.view_frames(str(test_video), start=0.0, end=2.0, interval=1.0)
        )
        assert [b["type"] for b in cli_blocks] == [b["type"] for b in mcp_blocks]

        cli_text = [b["text"] for b in cli_blocks if b["type"] == "text"]
        mcp_text = [b["text"] for b in mcp_blocks if b["type"] == "text"]
        # Timecodes must match exactly.
        assert cli_text[:-1] == mcp_text[:-1]
        # The summaries must agree on everything except the cache hit count,
        # which legitimately depends on which call ran first.
        cli_summary = json.loads(cli_text[-1])
        mcp_summary = json.loads(mcp_text[-1])
        cli_summary.pop("cached")
        mcp_summary.pop("cached")
        assert cli_summary == mcp_summary

    def test_piped_stdin_starts_the_server_and_does_not_hang(self, core):
        """Acceptance item 9, exercised on the real entry point.

        The pipe is what matters: an MCP client hands the server a pipe, and
        that is what makes ``stdin.isatty()`` false.  ``subprocess.DEVNULL``
        would not do — on Windows it opens the NUL character device, whose
        ``isatty()`` is True, so the process would take the terminal branch and
        this test would silently stop testing the client case.
        """
        completed = subprocess.run(
            [sys.executable, "-m", "mcp_video_frames"],
            input="",  # an empty pipe reads EOF immediately
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env(core),
            timeout=30,
        )
        assert completed.returncode in (0, 1), completed.stderr
        assert "Traceback" not in completed.stderr
        # It must have tried to serve, not to explain itself: the CLI help
        # mentions subcommands, the server does not.
        assert "usage: mcp-video-frames" not in completed.stdout
        # The terminal half of the split is covered in test_cli.py, which can
        # substitute a fake tty; a test process cannot allocate a real one
        # portably.

    def test_doctor_command_exits_cleanly(self, core):
        completed = run_cli(core, "doctor")
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout.split("\nSelf-check")[0])["ok"] is True
