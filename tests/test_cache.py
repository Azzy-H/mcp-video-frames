"""Cache layout, keying, atomic writes, LRU eviction, orphan collection.

Everything here runs without ffmpeg: the cache is a filesystem concern, and
its failure modes (partial writes, stale entries, frames evicted under a
reader) are exactly the ones that never show up in a happy-path integration
test.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from mcp_video_frames.cache import (
    Cache,
    cache_stats,
    entry_bytes,
    evict_frames,
    file_identity,
    frame_filename,
    frame_timestamps,
    parse_frame_filename,
    prune,
    read_json,
    source_still_matches,
    atomic_write_json,
)
from mcp_video_frames.config import Config

def _png_bytes() -> bytes:
    """A real 1x1 PNG, built rather than pasted.

    Hand-written base64/hex blobs are a maintenance trap: a single wrong
    character turns into a confusing failure in an unrelated test.
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

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw = b"\x00\x00\x00\x00\x00"  # one filter byte + one RGBA pixel
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


# Cache tests do not decode images, so the payload only has to be a plausible
# frame: the magic-number checks are what matter here.
JPEG_BYTES = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")
PNG_BYTES = _png_bytes()


def fake_video(path: Path, size: int = 4096, mtime_ns: int | None = None) -> Path:
    payload = bytes((i * 7) % 256 for i in range(size))
    path.write_bytes(payload)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


class TestFrameFilenames:
    def test_format_uses_the_documented_printf_spec(self):
        # The spec pins the name to t%09.3f; %09.3f is a *minimum* width of
        # nine, so five integer digits are kept for every timestamp.
        assert frame_filename(12.5, "jpeg") == "t00012.500.jpg"
        assert frame_filename(0.0, "jpeg") == "t00000.000.jpg"
        assert frame_filename(0.0, "png") == "t00000.000.png"
        assert frame_filename(1.0, "jpeg") == f"t{1.0:09.3f}.jpg"

    def test_constant_width_across_the_whole_range(self):
        names = [frame_filename(t, "jpeg") for t in (0.0, 1.0, 99999.999)]
        assert {len(name) for name in names} == {len("t00000.000.jpg")}

    def test_roundtrip(self):
        for t in (0.0, 0.05, 1.0, 12.5, 3599.999, 7200.0):
            for fmt in ("jpeg", "png"):
                parsed = parse_frame_filename(frame_filename(t, fmt))
                assert parsed is not None
                assert parsed[1] == fmt
                assert parsed[0] == pytest.approx(t, abs=1e-3)

    def test_lexicographic_order_is_chronological(self):
        times = [0.0, 1.0, 2.0, 9.999, 10.0, 10.001, 100.0, 1000.0, 99999.999]
        names = sorted(frame_filename(t, "jpeg") for t in times)
        parsed = [parse_frame_filename(n)[0] for n in names]
        assert parsed == sorted(times)

    def test_rejects_foreign_names(self):
        for name in (
            "index.json",
            "t12.jpg",  # not the fixed-width form
            "t000012.500.webp",
            "frame.jpg",
            "jpg",
            "t000012500.jpg",
            "t00012.500.jpg.tmp",
        ):
            assert parse_frame_filename(name) is None

    def test_frame_timestamps_of_missing_dir(self, tmp_path):
        assert frame_timestamps(tmp_path / "nope") == []

    def test_frame_timestamps_sorted(self, tmp_path):
        for name in ("t00010.000.jpg", "t00001.000.jpg", "garbage.txt"):
            (tmp_path / name).write_bytes(JPEG_BYTES)
        assert frame_timestamps(tmp_path) == [1.0, 10.0]


class TestIdentity:
    def test_stable_for_same_file(self, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        assert file_identity(video)[0] == file_identity(video)[0]

    def test_changes_when_content_changes(self, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        first = file_identity(video)[0]
        video.write_bytes(b"x" * 5000)
        assert file_identity(video)[0] != first

    def test_changes_when_mtime_changes(self, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        first = file_identity(video)[0]
        stat = video.stat()
        os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert file_identity(video)[0] != first

    def test_content_change_beyond_the_sample_window_still_changes_mtime(self, tmp_path):
        # The sampling window is a deliberate trade-off; the mtime component
        # is what catches edits in the untouched middle of a large file.
        video = fake_video(tmp_path / "big.mp4", size=3 * 1024 * 1024)
        first = file_identity(video)[0]
        before_ns = video.stat().st_mtime_ns
        with video.open("r+b") as handle:
            handle.seek(1024 * 1024 + 512)
            handle.write(b"\x00" * 16)
        if video.stat().st_mtime_ns == before_ns:
            pytest.skip("filesystem timestamp resolution is too coarse for this check")
        assert file_identity(video)[0] != first

    def test_does_not_hash_the_whole_file(self, tmp_path):
        video = fake_video(tmp_path / "v.mp4", size=6 * 1024 * 1024)
        identity, record = file_identity(video)
        assert record["sampled_bytes"] == 1024 * 1024
        assert record["size"] == 6 * 1024 * 1024
        assert len(identity) == 64

    def test_empty_file_is_an_error(self, tmp_path):
        from mcp_video_frames.errors import CacheError

        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(CacheError):
            file_identity(empty)


class TestAtomicWrites:
    def test_json_write_and_read(self, tmp_path):
        target = tmp_path / "sub" / "x.json"
        # Non-ASCII is the point: the round trip must survive whatever the host
        # platform's default encoding is, so the value stays deliberately
        # non-Latin rather than being an English placeholder.
        atomic_write_json(target, {"a": 1, "中文": "值"})
        assert read_json(target) == {"a": 1, "中文": "值"}

    def test_no_temp_files_left_behind(self, tmp_path):
        target = tmp_path / "x.json"
        atomic_write_json(target, {"a": 1})
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "x.json"]
        assert leftovers == []

    def test_corrupt_json_reads_as_default(self, tmp_path):
        target = tmp_path / "index.json"
        target.write_text("{not json", encoding="utf-8")
        assert read_json(target, default={"fallback": True}) == {"fallback": True}

    def test_overwrite_is_atomic_from_a_readers_perspective(self, tmp_path):
        target = tmp_path / "x.json"
        atomic_write_json(target, {"v": 1})
        for value in range(2, 30):
            atomic_write_json(target, {"v": value})
            # A reader either sees the old file or the new one, never a
            # half-written document.
            assert read_json(target)["v"] in (value - 1, value)


class TestRegistration:
    def test_register_creates_layout(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        assert cached.frames_dir.is_dir()
        assert (cached.root / "source.json").is_file()
        assert cache.index_path.is_file()
        index = cache.read_index()
        assert identity in index["entries"]

    def test_first_seen_is_preserved_across_registrations(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cache.register(identity, source)
        first = json.loads((cache.video_dir(identity) / "source.json").read_text())
        time.sleep(0.01)
        cache.register(identity, source)
        second = json.loads((cache.video_dir(identity) / "source.json").read_text())
        assert second["first_seen"] == first["first_seen"]
        assert second["last_access"] >= first["last_access"]

    def test_store_and_read_frame(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        cache.store_frame(cached, 12.5, "jpeg", JPEG_BYTES)
        assert cache.frame_exists(cached, 12.5, "jpeg")
        assert cache.read_frame_bytes(cached, 12.5, "jpeg") == JPEG_BYTES
        assert (cached.frames_dir / "t00012.500.jpg").is_file()

    def test_missing_frame_reads_as_none_not_an_exception(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        assert cache.read_frame_bytes(cached, 99.0, "jpeg") is None

    def test_video_dirs_are_namespaced_by_identity_prefix(self, cache: Cache, tmp_path):
        names = set()
        for index in range(3):
            video = fake_video(tmp_path / f"v{index}.mp4", size=2048 + index)
            identity, source = file_identity(video)
            cached = cache.register(identity, source)
            names.add(cached.root.name)
            assert cached.root.name == identity[:16]
        assert len(names) == 3


class TestIndexAtomicity:
    """Reading the index and writing it back is not atomic on its own.

    ``register``, ``touch`` and the background ``prune`` all mutate the index.
    When the read and the write were separate calls, a writer that read before
    a concurrent writer committed would overwrite it with its stale snapshot,
    and the newer entry disappeared from the index for good.  Every mutation
    therefore goes through ``index_transaction``, which holds one lock across
    the whole sequence.
    """

    def test_concurrent_registrations_all_survive(self, cache: Cache, tmp_path):
        videos = [fake_video(tmp_path / f"v{i}.mp4", size=1024 + i) for i in range(6)]
        pairs = [file_identity(video) for video in videos]
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(pairs))

        def worker(pair) -> None:
            try:
                barrier.wait(timeout=10)
                cache.register(*pair)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(pair,)) for pair in pairs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not errors, errors
        recorded = set(cache.read_index()["entries"])
        expected = {identity for identity, _source in pairs}
        assert recorded == expected

    def test_nesting_is_refused_not_silently_lost(self, cache: Cache):
        """Nesting would commit whichever in-memory copy finished last."""
        with cache.index_transaction():
            with pytest.raises(RuntimeError, match="already held"):
                with cache.index_transaction():
                    pass
        # The refusal must not leave the guard stuck on.
        with cache.index_transaction() as index:
            index["entries"]["after"] = {"last_access": 1.0}
        assert "after" in cache.read_index()["entries"]

    def test_prune_does_not_drop_a_video_registered_during_the_pass(
        self, cache: Cache, tmp_path
    ):
        """The background prune is the other writer; it used to clobber."""
        original = fake_video(tmp_path / "original.mp4", size=4096)
        cache.register(*file_identity(original))
        entrant = fake_video(tmp_path / "entrant.mp4", size=8192)

        started = threading.Event()
        may_finish = threading.Event()
        errors: list[BaseException] = []

        def registrar() -> None:
            try:
                started.set()
                cache.register(*file_identity(entrant))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                may_finish.set()

        thread = threading.Thread(target=registrar)
        # Run the prune concurrently with the registration.
        thread.start()
        started.wait(timeout=10)
        prune(cache)
        may_finish.wait(timeout=10)
        thread.join(timeout=20)

        assert not errors, errors
        recorded = set(cache.read_index()["entries"])
        assert file_identity(entrant)[0] in recorded, (
            "the video registered during the prune was dropped from the index"
        )

    def test_a_snapshot_cannot_be_written_back_by_accident(
        self, cache: Cache, tmp_path
    ):
        """``read_index`` re-reads the file, so a mutated snapshot is inert."""
        video = fake_video(tmp_path / "v.mp4")
        cache.register(*file_identity(video))
        snapshot = cache.read_index()
        snapshot["entries"]["bogus"] = {"last_access": 0}
        assert "bogus" not in cache.read_index()["entries"]


class TestAtomicReplaceUnderContention:
    """``os.replace`` onto a busy destination, which Windows refuses.

    Registering a video writes ``source.json`` before touching the index, and
    several clients can ask for the same video at once.  Windows fails the
    rename with ``PermissionError`` when the destination has a concurrent
    handle, so writers stampeding one entry used to fail outright — and unlike
    the index case there was no retry to fall back on.
    """

    def test_many_threads_registering_one_video_all_succeed(
        self, cache: Cache, tmp_path
    ):
        video = fake_video(tmp_path / "shared.mp4", size=4096)
        identity, source = file_identity(video)
        threads_wanted = 12
        barrier = threading.Barrier(threads_wanted)
        errors: list[BaseException] = []
        succeeded = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal succeeded
            try:
                barrier.wait(timeout=20)
                cache.register(identity, source)
                with lock:
                    succeeded += 1
            except BaseException as exc:  # noqa: BLE001 - asserted below
                with lock:
                    errors.append(exc)

        pool = [threading.Thread(target=worker) for _ in range(threads_wanted)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join(timeout=30)

        assert not errors, errors[:3]
        assert succeeded == threads_wanted

    def test_no_temp_files_are_left_by_contention(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "shared.mp4", size=4096)
        identity, source = file_identity(video)
        barrier = threading.Barrier(6)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=20)
                cache.register(identity, source)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        pool = [threading.Thread(target=worker) for _ in range(6)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join(timeout=30)

        assert not errors, errors[:3]
        leftovers = list(cache.root.rglob("*.tmp"))
        assert leftovers == [], leftovers

    def test_exhausted_retries_still_raise(self, monkeypatch, cache: Cache, tmp_path):
        """A write that did not happen must not look like one that did."""
        from mcp_video_frames import cache as cache_module

        calls = {"n": 0}

        def always_busy(_src, _dst, *args, **kwargs):
            calls["n"] += 1
            raise PermissionError(5, "Access is denied")

        monkeypatch.setattr(cache_module.os, "replace", always_busy)
        monkeypatch.setattr(cache_module, "_REPLACE_BACKOFF_SECONDS", 0.0)
        with pytest.raises(PermissionError):
            atomic_write_json(tmp_path / "target.json", {"a": 1})
        assert calls["n"] == cache_module._REPLACE_RETRIES, calls
        assert not (tmp_path / "target.json").exists()

    def test_a_non_permission_error_is_not_retried(self, monkeypatch, tmp_path):
        """Only the transient Windows contention error is worth retrying."""
        from mcp_video_frames import cache as cache_module

        calls = {"n": 0}

        def boom(_src, _dst, *args, **kwargs):
            calls["n"] += 1
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(cache_module.os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_json(tmp_path / "target.json", {"a": 1})
        assert calls["n"] == 1, "a non-contention error must fail on the first try"

    def test_a_transient_failure_is_retried_and_succeeds(self, monkeypatch, tmp_path):
        """The real case: contended for a moment, then accepted."""
        from mcp_video_frames import cache as cache_module

        real_replace = cache_module.os.replace
        state = {"failures": 3}

        def flaky(src, dst, *args, **kwargs):
            if state["failures"] > 0:
                state["failures"] -= 1
                raise PermissionError(5, "Access is denied")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(cache_module.os, "replace", flaky)
        monkeypatch.setattr(cache_module, "_REPLACE_BACKOFF_SECONDS", 0.0)
        target = tmp_path / "eventual.json"
        atomic_write_json(target, {"a": 1})
        assert state["failures"] == 0
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


class TestMetaLayering:
    def test_detail_selects_the_file(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        cache.write_meta(cached, "basic", {"duration": 10.0})
        cache.write_meta(cached, "full", {"scene_cuts": [1.0]})
        assert cache.read_meta(cached, "basic") == {"duration": 10.0}
        assert cache.read_meta(cached, "full") == {"scene_cuts": [1.0]}
        assert cache.read_meta(cached, "full")["scene_cuts"] == [1.0]

    def test_missing_meta_reads_as_none(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        assert cache.read_meta(cached, "basic") is None


class TestLocking:
    def test_lock_is_exclusive_across_instances(self, tmp_path):
        from mcp_video_frames._locks import lock_for

        target = tmp_path / "l.lock"
        with lock_for(target, timeout=1.0):
            with pytest.raises(Exception):
                with lock_for(target, timeout=0.1):
                    pass

    def test_lock_is_released_after_use(self, tmp_path):
        from mcp_video_frames._locks import lock_for

        target = tmp_path / "l.lock"
        with lock_for(target, timeout=1.0):
            pass
        with lock_for(target, timeout=1.0):
            pass

    def test_cache_locked_context_manager(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        with cache.locked(identity):
            assert cache.entry_dir_exists(identity)


class TestEviction:
    def _populate(self, cache: Cache, tmp_path, count=3, frames=4):
        entries = []
        for index in range(count):
            video = fake_video(tmp_path / f"v{index}.mp4", size=2048 + index * 16)
            identity, source = file_identity(video)
            cached = cache.register(identity, source)
            for n in range(frames):
                cache.store_frame(cached, float(n), "jpeg", JPEG_BYTES * 8)
            entries.append((video, identity, cached))
        return entries

    def test_lru_order_evicts_the_least_recently_used_first(self, cache: Cache, tmp_path):
        entries = self._populate(cache, tmp_path)
        index = cache.read_index()
        now = time.time()
        # v0 oldest, v2 newest.
        for offset, (_video, identity, _cached) in enumerate(entries):
            index["entries"][identity]["last_access"] = now - 1000 + offset
        cache.write_index(index)

        sizes = {
            identity: entry_bytes(cache.video_dir(identity))
            for _v, identity, _c in entries
        }
        budget = max(sizes.values()) + 1  # room for about one entry
        report = prune(cache, max_bytes=budget, max_age_seconds=10**9)
        remaining = set(cache.read_index()["entries"])
        assert entries[0][1] not in remaining
        assert entries[2][1] in remaining
        assert report.removed_entries >= 1

    def test_meta_full_survives_size_pressure(self, cache: Cache, tmp_path):
        entries = self._populate(cache, tmp_path)
        # Entry 0 owns the expensive full scan and is also the most recently
        # used, so LRU reaches the other two first.
        _video, identity, cached = entries[0]
        cache.write_meta(
            cached, "full", {"scene_cuts": [], "silences": [], "loudness": {}}
        )
        index = cache.read_index()
        now = time.time()
        for offset, (_v, other, _c) in enumerate(entries):
            index["entries"][other]["last_access"] = now + offset
        cache.write_index(index)

        prune(cache, max_bytes=1, max_age_seconds=10**9)
        # Cheaper entries are dropped outright; the expensive scan survives.
        assert (cache.video_dir(identity) / "meta_full.json").is_file()
        remaining = set(cache.read_index()["entries"])
        assert identity in remaining

    def test_frames_are_dropped_before_a_full_scan(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        cache.write_meta(
            cached, "full", {"scene_cuts": [], "silences": [], "loudness": {}}
        )
        for n in range(4):
            cache.store_frame(cached, float(n), "jpeg", JPEG_BYTES * 64)
        assert frame_timestamps(cached.frames_dir) != []

        # Budget too small to hold the entry: frames go first, the scan stays.
        prune(cache, max_bytes=1, max_age_seconds=10**9)
        assert not (cached.frames_dir / frame_filename(0.0, "jpeg")).exists()
        assert (cached.root / "meta_full.json").is_file()

    def test_age_limit_evicts_old_frames_only(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        cache.write_meta(cached, "full", {"scene_cuts": [], "silences": [], "loudness": {}})
        cache.store_frame(cached, 0.0, "jpeg", JPEG_BYTES)
        cache.store_frame(cached, 1.0, "jpeg", JPEG_BYTES)
        # Age one frame by 30 days.
        old = cached.frames_dir / frame_filename(0.0, "jpeg")
        stale = time.time() - 30 * 86400
        os.utime(old, (stale, stale))

        report = prune(cache, max_bytes=10**12, max_age_seconds=14 * 86400)
        assert report.removed_frames == 1
        assert not old.exists()
        assert (cached.frames_dir / frame_filename(1.0, "jpeg")).is_file()
        # The full scan is never age-pruned: recomputing it costs a whole scan.
        assert (cached.root / "meta_full.json").is_file()

    def test_orphan_collection_removes_missing_sources(self, cache: Cache, tmp_path):
        entries = self._populate(cache, tmp_path, count=2)
        entries[0][0].unlink()
        report = prune(cache, max_bytes=10**12, max_age_seconds=10**9)
        assert report.removed_orphans == 1
        remaining = set(cache.read_index()["entries"])
        assert entries[0][1] not in remaining
        assert entries[1][1] in remaining
        assert not cache.video_dir(entries[0][1]).exists()

    def test_unavailable_volume_is_not_treated_as_deleted(self, cache: Cache, tmp_path):
        # An unplugged drive or sleeping NAS must not wipe a valid cache.
        source = {
            "path": str(tmp_path / "gone" / "v.mp4"),
            "size": 10,
            "mtime_ns": 1,
        }
        assert source_still_matches(source) is True

    def test_changed_source_is_an_orphan(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cache.register(identity, source)
        video.write_bytes(b"different content entirely")
        prune(cache, max_bytes=10**12, max_age_seconds=10**9)
        assert identity not in cache.read_index()["entries"]

    def test_evict_frames_respects_age_boundary(self, tmp_path):
        frames = tmp_path / "frames"
        frames.mkdir()
        fresh = frames / "t000000.000.jpg"
        fresh.write_bytes(JPEG_BYTES)
        count, freed = evict_frames(tmp_path, max_age_seconds=3600)
        assert (count, freed) == (0, 0)
        assert fresh.is_file()

    def test_evict_frames_ignores_non_frame_files(self, tmp_path):
        frames = tmp_path / "frames"
        frames.mkdir()
        keep = frames / "notes.txt"
        keep.write_bytes(b"hello")
        old = time.time() - 10**7
        os.utime(keep, (old, old))
        count, _freed = evict_frames(tmp_path, max_age_seconds=1)
        assert count == 0
        assert keep.is_file()


class TestStats:
    def test_stats_shape(self, cache: Cache, tmp_path):
        video = fake_video(tmp_path / "v.mp4")
        identity, source = file_identity(video)
        cached = cache.register(identity, source)
        cache.store_frame(cached, 0.0, "jpeg", JPEG_BYTES)
        cache.write_meta(cached, "full", {"scene_cuts": [], "silences": [], "loudness": {}})
        stats = cache_stats(cache)
        assert stats["videos"] == 1
        assert stats["full_scans"] == 1
        assert stats["bytes"] > 0
        assert stats["frame_bytes"] > 0

    def test_stats_on_empty_cache(self, cache: Cache):
        stats = cache_stats(cache)
        assert stats["videos"] == 0
        assert stats["bytes"] == 0


class TestConfigOverrides:
    def test_cache_dir_env_override(self, tmp_path):
        config = Config.from_env({"MCP_VIDEO_CACHE_DIR": str(tmp_path / "custom")})
        assert config.cache_root == tmp_path / "custom"

    def test_limits_are_read_from_env(self):
        config = Config.from_env(
            {
                "MCP_VIDEO_MAX_IMAGES": "7",
                "MCP_VIDEO_MAX_IMAGE_BYTES": "1234",
                "MCP_VIDEO_MAX_TOTAL_BYTES": "5678",
                "MCP_VIDEO_MAX_IMAGE_DIM": "4096",
                "MCP_VIDEO_CACHE_MAX_BYTES": "999",
                "MCP_VIDEO_FRAME_MAX_AGE_DAYS": "3",
                "MCP_VIDEO_SCENE_THRESHOLD": "0.5",
            }
        )
        assert config.max_images_per_call == 7
        assert config.max_image_bytes == 1234
        assert config.max_total_bytes == 5678
        assert config.max_image_dimension == 4096
        assert config.cache_max_bytes == 999
        assert config.frame_max_age_days == 3
        assert config.scene_threshold == 0.5
        assert config.frame_max_age_seconds == 3 * 86400

    def test_non_numeric_env_is_a_clear_error(self):
        from mcp_video_frames.errors import ConfigError

        with pytest.raises(ConfigError) as excinfo:
            Config.from_env({"MCP_VIDEO_MAX_IMAGES": "twenty"})
        assert "MCP_VIDEO_MAX_IMAGES" in str(excinfo.value)

    def test_defaults_apply_when_env_is_absent(self):
        config = Config.from_env({})
        assert config.max_images_per_call == 20
        assert config.cache_max_bytes == 5 * 1024 * 1024 * 1024
        assert config.frame_max_age_days == 14
