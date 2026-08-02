"""Small cross-platform advisory file-lock helpers."""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def acquire_file_lock(
    path: str | Path,
    *,
    exclusive: bool,
    blocking: bool,
) -> BinaryIO | None:
    """Open and lock ``path``, returning ``None`` only for contention."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            import msvcrt

            if lock_path.stat().st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                # Windows has no shared mode here. Serializing readers is
                # slower, but preserves the lifecycle safety contract.
                msvcrt.locking(lock_file.fileno(), mode, 1)
            except OSError as error:
                if not blocking and error.errno in {errno.EACCES, errno.EDEADLK}:
                    lock_file.close()
                    return None
                raise
        else:
            try:
                import fcntl
            except ImportError as error:  # pragma: no cover - unusual platform
                raise RuntimeError(
                    "This platform has no supported advisory file locking"
                ) from error

            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(lock_file.fileno(), operation)
            except BlockingIOError:
                lock_file.close()
                return None
        return lock_file
    except BaseException:
        if not lock_file.closed:
            lock_file.close()
        raise


def release_file_lock(lock_file: BinaryIO) -> None:
    """Release and close a handle returned by :func:`acquire_file_lock`."""
    if lock_file.closed:
        return
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


@contextmanager
def advisory_file_lock(
    path: str | Path,
    *,
    exclusive: bool,
    blocking: bool = True,
) -> Iterator[BinaryIO | None]:
    """Hold an advisory lock for the duration of the context."""
    lock_file = acquire_file_lock(
        path,
        exclusive=exclusive,
        blocking=blocking,
    )
    try:
        yield lock_file
    finally:
        if lock_file is not None:
            release_file_lock(lock_file)
