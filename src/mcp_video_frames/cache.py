"""Cache layout, keying, locking, LRU eviction and atomic writes.

The cache is infrastructure, not a tool: everything in it can be
regenerated.  It is never mentioned in the tool list, and deleting the
whole directory is always safe.

Layout::

    <cache_root>/
      index.json                      # identity -> {path, size, first_seen, last_access}
      videos/<identity16>/
        source.json
        meta_basic.json
        meta_full.json
        frames/
          t000012.500.jpg             # name = timestamp, lexicographic == chronological
          .lock

Key design choices:

* identity is a *sampled* hash (size + mtime + first 1 MB + last 1 MB), so
  hashing a 50 GB file is not part of answering a question about it;
* frames are cached per *absolute timestamp*, never per requested range, so
  overlapping requests reuse work (0–20 s then 10–30 s hits 10–20 s);
* every writer holds a lock and writes through a temporary file plus
  ``os.replace``, so a second server instance on the same cache directory
  can never observe a half-written frame;
* the operating system is not asked to clean this up.  Windows effectively
  never does, Linux/macOS do it unpredictably.  We implement our own
  layered eviction, ordered by *recompute cost* rather than by
  regenerability: frames are cheap and evicted first, ``meta_full.json``
  costs a whole-file scan and is kept preferentially.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ._locks import FileLock, lock_for
from .config import VIDEO_LOCK_TIMEOUT, Config
from .errors import CacheError

# ----------------------------------------------------------------------
# JSON helpers (atomic write / tolerant read)
# ----------------------------------------------------------------------
#: How many times to retry the final ``os.replace``, and the first backoff.
#:
#: Windows fails the replace when the *destination* has a concurrent handle —
#: another thread or process reading it, or renaming onto it — with
#: ``PermissionError``.  That condition is transient: measured on a 12-thread
#: stampede onto one ``source.json``, seven writers failed without a retry and
#: none did with one, needing at most three attempts.  POSIX replaces
#: unconditionally, so this costs nothing there.
_REPLACE_RETRIES = 20
_REPLACE_BACKOFF_SECONDS = 0.005
_REPLACE_BACKOFF_CAP = 0.1


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """``os.replace``, retrying only the transient Windows contention error.

    Anything else propagates on the first attempt.  When the retries are
    exhausted the original ``PermissionError`` is raised rather than swallowed:
    a write that did not happen must never look like one that did.
    """
    delay = _REPLACE_BACKOFF_SECONDS
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _REPLACE_BACKOFF_CAP)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Same-directory temp file, then ``os.replace``.  Same directory because
    replace is only atomic within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(
        path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    )


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when absent *or* unreadable.

    A corrupt cache entry must never break a tool call; the entry is simply
    treated as a miss and regenerated.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


# ----------------------------------------------------------------------
# identity
# ----------------------------------------------------------------------
#: How much of each end of the file feeds the identity hash.
SAMPLE_BYTES = 1024 * 1024


def file_identity(path: Path) -> tuple[str, dict[str, Any]]:
    """Return ``(identity, source_record)`` for a video file.

    The identity is a SHA-256 over size, mtime (ns) and the first and last
    1 MB.  It is stable across moves but changes when the file changes,
    which is what the cache needs; it is not a cryptographic digest of the
    whole file and is not presented as one.
    """
    try:
        stat = path.stat()
    except OSError as exc:
        raise CacheError(
            f"Cannot read video file information for {path}: {exc}. Check "
            f"that the path exists and is accessible."
        ) from exc
    if stat.st_size == 0:
        raise CacheError(f"Video file {path} is empty (0 bytes).")

    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as handle:
        head = handle.read(SAMPLE_BYTES)
        digest.update(head)
        if stat.st_size > SAMPLE_BYTES:
            handle.seek(max(0, stat.st_size - SAMPLE_BYTES))
            digest.update(handle.read(SAMPLE_BYTES))
    identity = digest.hexdigest()
    record = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "identity": identity,
        "sampled_bytes": SAMPLE_BYTES,
    }
    return identity, record


#: ``t%09.3f`` plus an extension.  ``%09.3f`` means a *minimum* field width
#: of nine characters, of which three are the fraction, so timestamps up to
#: 99999.999 s keep a constant digit count — which is what makes
#: lexicographic order match chronological order.  The pattern is part of the
#: contract, so it is enforced rather than inferred.
_FRAME_NAME_RE = re.compile(r"^t(\d{5}\.\d{3})\.(jpg|jpeg|png)$", re.IGNORECASE)


def frame_filename(t: float, image_format: str) -> str:
    """Cache filename for an absolute timestamp.

    ``t%09.3f`` yields e.g. ``t00012.500.jpg``; the zero padding is what
    makes lexicographic order equal chronological order.
    """
    suffix = "png" if image_format.lower() == "png" else "jpg"
    return f"t{t:09.3f}.{suffix}"


def parse_frame_filename(name: str) -> tuple[float, str] | None:
    """Inverse of :func:`frame_filename`; ``None`` for anything else."""
    match = _FRAME_NAME_RE.match(name)
    if match is None:
        return None
    return float(match.group(1)), (
        "png" if match.group(2).lower() == "png" else "jpeg"
    )


def frame_timestamps(directory: Path) -> list[float]:
    """Sorted timestamps present in a frames directory (missing dir -> [])."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    out: list[float] = []
    for name in names:
        parsed = parse_frame_filename(name)
        if parsed is not None:
            out.append(parsed[0])
    return sorted(out)


# ----------------------------------------------------------------------
# view of the cache for one video
# ----------------------------------------------------------------------
@dataclass
class CachedVideo:
    identity: str
    root: Path
    source: dict[str, Any]

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def basic_path(self) -> Path:
        return self.root / "meta_basic.json"

    @property
    def full_path(self) -> Path:
        return self.root / "meta_full.json"

    @property
    def lock_path(self) -> Path:
        """Lock file for this entry, per the documented layout.

        It lives inside ``frames/`` and eviction ignores anything that is not
        a frame filename, so the lock is never collected out from under a
        running extraction — and a second process extracts into its own
        scratch directory anyway.
        """
        return self.frames_dir / ".lock"

    def frame_path(self, t: float, image_format: str) -> Path:
        return self.frames_dir / frame_filename(t, image_format)


class Cache:
    """Filesystem cache rooted at ``config.cache_root``.

    One instance per process is enough; all mutating operations take the
    per-video lock, so several processes may share the directory.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = Path(config.cache_root)
        self.videos_dir = self.root / "videos"
        self.index_path = self.root / "index.json"
        #: Cross-process mutual exclusion for index mutations.
        self._index_lock = FileLock(str(self.root / "index.lock"))
        #: In-process counterpart.  Both are needed: the file lock excludes other
        #: processes, this excludes other threads of this one.  They are
        #: separate locks rather than one because a transaction may nest (a
        #: ``prune`` triggered from inside another mutation), and re-acquiring a
        #: ``FileLock`` already held by this thread deadlocks.
        self._index_guard = threading.RLock()
        #: Nesting depth per thread, so an inner transaction joins the outer
        #: one instead of trying to take the file lock twice.
        self._index_depth = threading.local()
        self._prune_checked_at = 0.0
        self._prune_min_interval = 60.0

    # -- setup ---------------------------------------------------------
    def ensure_root(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def video_dir(self, identity: str) -> Path:
        return self.videos_dir / identity[:16]

    def lock_path_for(self, identity: str) -> Path:
        """Lock file guarding one video entry.

        Placed inside the entry so it travels with it, and inside ``frames/``
        to match the documented layout.  Eviction only considers names that
        parse as frame timestamps, so it never deletes this file.
        """
        return self.video_dir(identity) / "frames" / ".lock"

    @contextmanager
    def locked(
        self, identity: str, timeout: float = VIDEO_LOCK_TIMEOUT
    ) -> Iterator[None]:
        """Hold the per-video lock.

        Extraction, metadata writes and eviction all take it, so a second
        process never observes a half-populated entry.  The default timeout
        covers a whole-file scan, which is the longest legitimate hold; see
        :data:`mcp_video_frames.config.VIDEO_LOCK_TIMEOUT`.
        """
        self.ensure_root()
        (self.video_dir(identity) / "frames").mkdir(parents=True, exist_ok=True)
        with lock_for(self.lock_path_for(identity), timeout=timeout):
            yield

    # -- index ---------------------------------------------------------
    def _read_index_unlocked(self) -> dict[str, Any]:
        data = read_json(self.index_path, default=None)
        if not isinstance(data, dict):
            return {"version": 1, "entries": {}}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            data["entries"] = {}
        data.setdefault("version", 1)
        return data

    def read_index(self) -> dict[str, Any]:
        """A snapshot of the index.  Read-only; safe to call anywhere.

        A snapshot is stale the moment it is returned, so it must not be
        modified and written back on its own — that is exactly the lost update
        :meth:`index_transaction` exists to prevent.
        """
        return self._read_index_unlocked()

    @contextmanager
    def index_transaction(self) -> Iterator[dict[str, Any]]:
        """Read-modify-write the index under one lock.

        Use this for *every* mutation, including the background prune.  The
        lock is held across the whole sequence rather than just the write, so a
        writer cannot install a snapshot taken before a concurrent writer added
        an entry — which is how a newly registered video used to vanish from the
        index permanently.

        Not reentrant, and nesting is refused rather than nested: the file lock
        cannot be taken twice by one thread, and joining an outer transaction
        would mean committing whichever of the two in-memory copies finished
        last.  Callers keep mutations disjoint instead — the per-video lock,
        which is what serialises work on a video, is held *outside* any index
        transaction.
        """
        if getattr(self._index_depth, "held", False):
            raise RuntimeError(
                "index_transaction() is already held by this thread; nesting "
                "would lose one of the two in-memory copies"
            )
        with self._index_guard, self._index_lock:
            self._index_depth.held = True
            try:
                index = self._read_index_unlocked()
                yield index
                atomic_write_json(self.index_path, index)
            finally:
                self._index_depth.held = False

    def write_index(self, index: dict[str, Any]) -> None:
        """Replace the index wholesale.

        Only correct for a caller that has just read it under the same lock;
        everything else should use :meth:`index_transaction`.
        """
        with self._index_guard:
            self.ensure_root()
            atomic_write_json(self.index_path, index)

    def register(self, identity: str, source: dict[str, Any]) -> CachedVideo:
        """Create/refresh the cache entry for a video and return its view."""
        root = self.video_dir(identity)
        (root / "frames").mkdir(parents=True, exist_ok=True)
        now = time.time()
        source_path = root / "source.json"
        previous = read_json(source_path, default={}) or {}
        record = {
            **source,
            "identity": identity,
            "first_seen": previous.get("first_seen", now),
            "last_access": now,
        }
        atomic_write_json(source_path, record)

        with self.index_transaction() as index:
            entry = index["entries"].get(identity, {})
            entry.update(
                {
                    "path": source.get("path"),
                    "size": source.get("size"),
                    "mtime_ns": source.get("mtime_ns"),
                    "first_seen": entry.get("first_seen", now),
                    "last_access": now,
                }
            )
            index["entries"][identity] = entry
        return CachedVideo(identity=identity, root=root, source=record)

    def touch(self, identity: str) -> None:
        """Update ``last_access`` in the index (used by LRU ordering)."""
        with self.index_transaction() as index:
            entry = index["entries"].get(identity)
            if entry is None:
                return
            entry["last_access"] = time.time()

    def get(self, identity: str) -> CachedVideo | None:
        root = self.video_dir(identity)
        source = read_json(root / "source.json", default=None)
        if not isinstance(source, dict):
            return None
        return CachedVideo(identity=identity, root=root, source=source)

    def entry_dir_exists(self, identity: str) -> bool:
        return self.video_dir(identity).is_dir()

    # -- frames --------------------------------------------------------
    def frame_exists(self, cached: CachedVideo, t: float, image_format: str) -> bool:
        try:
            return cached.frame_path(t, image_format).is_file()
        except OSError:
            return False

    def read_frame_bytes(
        self, cached: CachedVideo, t: float, image_format: str
    ) -> bytes | None:
        path = cached.frame_path(t, image_format)
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return data or None

    def store_frame(
        self, cached: CachedVideo, t: float, image_format: str, data: bytes
    ) -> Path:
        target = cached.frame_path(t, image_format)
        _atomic_write_bytes(target, data)
        return target

    def frame_bytes_total(self, cached: CachedVideo) -> int:
        total = 0
        try:
            with os.scandir(cached.frames_dir) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        try:
                            total += entry.stat().st_size
                        except OSError:
                            continue
        except OSError:
            return 0
        return total

    # -- metadata ------------------------------------------------------
    def read_meta(self, cached: CachedVideo, detail: str) -> dict[str, Any] | None:
        path = cached.full_path if detail == "full" else cached.basic_path
        data = read_json(path, default=None)
        return data if isinstance(data, dict) else None

    def write_meta(self, cached: CachedVideo, detail: str, payload: dict[str, Any]) -> None:
        path = cached.full_path if detail == "full" else cached.basic_path
        atomic_write_json(path, payload)


def source_still_matches(source: dict[str, Any]) -> bool:
    """Cheap check that a cached entry still describes the file on disk.

    Deliberately forgiving when the *volume* is unavailable (unplugged drive,
    unmounted share): a missing parent directory means "cannot tell", not
    "the file is gone", and we do not want to delete a valid cache because a
    NAS was asleep.
    """
    raw = source.get("path")
    if not raw:
        return False
    path = Path(raw)
    parent = path.parent
    if not parent.exists():
        return True
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size == source.get("size") and stat.st_mtime_ns == source.get(
        "mtime_ns"
    )


# ----------------------------------------------------------------------
# eviction
# ----------------------------------------------------------------------
@dataclass
class PruneReport:
    removed_entries: int = 0
    removed_orphans: int = 0
    removed_frames: int = 0
    removed_bytes: int = 0
    kept_full_scans: int = 0
    total_bytes: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "removed_entries": self.removed_entries,
            "removed_orphans": self.removed_orphans,
            "removed_frames": self.removed_frames,
            "removed_bytes": self.removed_bytes,
            "kept_full_scans": self.kept_full_scans,
            "total_bytes": self.total_bytes,
        }


def entry_bytes(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def evict_frames(directory: Path, *, max_age_seconds: float) -> tuple[int, int]:
    """Delete frame files older than ``max_age_seconds``.

    Applies to ``frames/`` only — never to ``meta_full.json``, whose
    recompute cost is a full-file scan.  Returns ``(count, bytes)``.
    """
    frames_dir = directory / "frames"
    if not frames_dir.is_dir():
        return 0, 0
    cutoff = time.time() - max_age_seconds
    count = 0
    freed = 0
    with os.scandir(frames_dir) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            if parse_frame_filename(entry.name) is None:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if stat.st_mtime > cutoff:
                continue
            try:
                os.unlink(entry.path)
            except OSError:
                continue
            count += 1
            freed += stat.st_size
    return count, freed


def prune(
    cache: Cache,
    *,
    max_bytes: int | None = None,
    max_age_seconds: float | None = None,
    now: float | None = None,
) -> PruneReport:
    """Run orphan collection, age-based frame pruning and size-based LRU.

    Order matters: drop orphaned entries first (they can never be reused),
    then old frames, then whole-entry LRU until the size budget is met.

    ``meta_full.json`` is the most expensive thing in the cache — recomputing
    it means scanning the entire file — so it is evicted last.  In practice
    that means surviving even a wildly small budget as long as it can be
    stored at all; the spec's layered policy is explicit that frames are the
    cheap tier.

    The whole pass runs inside one index transaction.  It runs on a background
    thread while requests are still registering videos, so reading the index
    first and writing it back at the end would drop anything registered in
    between — the entries are deleted from ``entries`` in place instead.
    """
    config = cache.config
    max_bytes = config.cache_max_bytes if max_bytes is None else max_bytes
    max_age_seconds = (
        config.frame_max_age_seconds if max_age_seconds is None else max_age_seconds
    )
    now = time.time() if now is None else now
    report = PruneReport()

    cache.ensure_root()
    with cache.index_transaction() as index:
        entries: dict[str, Any] = index.setdefault("entries", {})
        _prune_locked(
            cache,
            entries,
            report,
            max_bytes=max_bytes,
            max_age_seconds=max_age_seconds,
            now=now,
        )
    return report


def _prune_locked(
    cache: Cache,
    entries: dict[str, Any],
    report: PruneReport,
    *,
    max_bytes: int,
    max_age_seconds: float,
    now: float,
) -> None:
    """The eviction passes.  ``entries`` is the live index, mutated in place."""
    # --- 1. orphans ---------------------------------------------------
    removed_ids: list[str] = []
    for identity, entry in list(entries.items()):
        directory = cache.video_dir(identity)
        source = read_json(directory / "source.json", default=None)
        if not isinstance(source, dict):
            # Unreadable source.json.  That alone does not make the entry
            # garbage: if the directory is still there its bytes are still on
            # disk, and the index has to keep counting them or the LRU sizing
            # and `cache --stats` totals drift.  Only a directory that has also
            # disappeared is unreachable, and that is what gets reclaimed.
            if not directory.exists():
                removed_ids.append(identity)
            continue
        if directory.exists() and not source_still_matches(source):
            removed_ids.append(identity)
    for identity in removed_ids:
        directory = cache.video_dir(identity)
        freed = entry_bytes(directory)
        _rmtree(directory)
        entries.pop(identity, None)
        report.removed_orphans += 1
        report.removed_bytes += freed

    # --- 2. frame age -------------------------------------------------
    for identity in list(entries.keys()):
        directory = cache.video_dir(identity)
        if not directory.is_dir():
            continue
        count, freed = evict_frames(directory, max_age_seconds=max_age_seconds)
        report.removed_frames += count
        report.removed_bytes += freed

    # --- 3. size budget, LRU over whole entries -----------------------
    sizes = {identity: entry_bytes(cache.video_dir(identity)) for identity in entries}
    total = sum(sizes.values())
    ordered = sorted(
        entries.items(), key=lambda kv: float(kv[1].get("last_access") or 0.0)
    )
    for identity, entry in ordered:
        if total <= max_bytes:
            break
        directory = cache.video_dir(identity)
        if (directory / "meta_full.json").is_file():
            # Keep the expensive artifact, drop the cheap frames with it.
            report.kept_full_scans += 1
            count, frame_bytes = evict_frames(directory, max_age_seconds=0.0)
            report.removed_frames += count
            report.removed_bytes += frame_bytes
            total -= frame_bytes
            continue
        freed = sizes.get(identity, 0)
        _rmtree(directory)
        entries.pop(identity, None)
        report.removed_entries += 1
        report.removed_bytes += freed
        total -= freed

    report.total_bytes = max(total, 0)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def cache_stats(cache: Cache) -> dict[str, Any]:
    """Occupancy summary for ``cache --stats`` and the prune log."""
    cache.ensure_root()
    index = cache.read_index()
    entries = index.get("entries", {})
    frames_bytes = 0
    meta_bytes = 0
    full_scans = 0
    per_video: list[dict[str, Any]] = []
    for identity, entry in entries.items():
        directory = cache.video_dir(identity)
        if not directory.is_dir():
            continue
        fbytes = cache.frame_bytes_total(
            CachedVideo(identity=identity, root=directory, source={})
        )
        total = entry_bytes(directory)
        frames_bytes += fbytes
        meta_bytes += total - fbytes
        has_full = (directory / "meta_full.json").is_file()
        if has_full:
            full_scans += 1
        per_video.append(
            {
                "identity": identity[:16],
                "path": entry.get("path"),
                "bytes": total,
                "frame_bytes": fbytes,
                "last_access": entry.get("last_access"),
                "has_full_scan": has_full,
            }
        )
    per_video.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "cache_root": str(cache.root),
        "videos": len(per_video),
        "bytes": frames_bytes + meta_bytes,
        "frame_bytes": frames_bytes,
        "meta_bytes": meta_bytes,
        "full_scans": full_scans,
        "max_bytes": cache.config.cache_max_bytes,
        "frame_max_age_days": cache.config.frame_max_age_days,
        "entries": per_video,
    }


def maybe_prune_async(cache: Cache) -> None:
    """Kick off eviction *after* a response, never before.

    No startup full scan (that would slow every launch), no timers: a cheap
    size check decides whether a background thread is worth starting, and
    at most one prune per ``_prune_min_interval`` seconds.
    """
    now = time.time()
    if now - cache._prune_checked_at < cache._prune_min_interval:
        return
    cache._prune_checked_at = now
    try:
        if _total_cache_bytes(cache) <= cache.config.cache_max_bytes:
            return
    except OSError:
        return
    import threading

    thread = threading.Thread(
        target=_safe_prune, args=(cache,), name="mcp-video-frames-prune", daemon=True
    )
    thread.start()


def _safe_prune(cache: Cache) -> None:
    try:
        prune(cache)
    except Exception:  # pragma: no cover - background best effort
        # Eviction is housekeeping: it must never take the server down.
        pass


def _total_cache_bytes(cache: Cache) -> int:
    index = cache.read_index()
    total = 0
    for identity in index.get("entries", {}):
        directory = cache.video_dir(identity)
        if directory.is_dir():
            total += entry_bytes(directory)
    return total
