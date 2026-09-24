"""File locking with a portable fallback.

``filelock`` is the preferred implementation: it is genuinely cross-platform
and handles the platform quirks (Windows' mandatory locking, NFS caveats)
that are easy to get wrong.

It is imported opportunistically rather than required.  A server that cannot
start because a lock library is missing is worse than a server that starts
with a slightly less battle-tested lock, and the fallback below is enough for
the one thing we need it for: mutual exclusion between two server processes
sharing a cache directory.

Both implementations expose the same minimal surface used by the cache::

    with lock_for(path, timeout=...):
        ...
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import VIDEO_LOCK_TIMEOUT

try:  # pragma: no cover - depends on the environment
    from filelock import FileLock, Timeout

    HAVE_FILELOCK = True
except ImportError:  # pragma: no cover - exercised only without filelock
    HAVE_FILELOCK = False

    class Timeout(Exception):  # type: ignore[no-redef]
        """Raised when a lock cannot be acquired in time."""

    class FileLock:  # type: ignore[no-redef]
        """Minimal ``O_CREAT | O_EXCL`` lock file.

        Not reentrant, and it assumes a local filesystem — which is what a
        cache directory is.  Stale locks left by a killed process are
        detected by age and removed.
        """

        #: Derived from the longest legitimate hold, so a lock taken for a
        #: whole-file scan is never mistaken for one left behind by a crash.
        #: A fixed value here (600 s was the original) is shorter than a scan
        #: is allowed to run, which would let a second process in to repeat it.
        STALE_SECONDS = VIDEO_LOCK_TIMEOUT + 600.0

        def __init__(self, lock_file: str, timeout: float = -1) -> None:
            self.lock_file = lock_file
            self.timeout = timeout
            self._fd: int | None = None

        def acquire(self, timeout: float | None = None, poll_interval: float = 0.05):
            wait = self.timeout if timeout is None else timeout
            deadline = None if wait is None or wait < 0 else time.time() + wait
            path = Path(self.lock_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    self._fd = os.open(
                        str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR
                    )
                    os.write(self._fd, f"{os.getpid()}\n".encode())
                    return self
                except FileExistsError:
                    self._clear_stale(path)
                    if deadline is not None and time.time() >= deadline:
                        raise Timeout(str(path)) from None
                    time.sleep(poll_interval)
                except OSError:
                    raise

        def _clear_stale(self, path: Path) -> None:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                return
            if age > self.STALE_SECONDS:
                try:
                    path.unlink()
                except OSError:
                    pass

        def release(self) -> None:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
            try:
                Path(self.lock_file).unlink()
            except OSError:
                pass

        def __enter__(self):
            return self.acquire()

        def __exit__(self, *exc_info) -> None:
            self.release()


@contextmanager
def lock_for(path: Path | str, timeout: float = 120.0) -> Iterator[None]:
    """Acquire ``path`` as a lock, releasing it on exit."""
    lock = FileLock(str(path))
    try:
        with lock.acquire(timeout=timeout):
            yield
    except Timeout as exc:
        from .errors import CacheError

        raise CacheError(
            f"Timed out after {timeout:g}s waiting for the file lock: {path}. "
            f"Another instance may be processing the same video. Retry shortly."
        ) from exc


__all__ = ["HAVE_FILELOCK", "FileLock", "Timeout", "lock_for"]
