"""Crash-resistant helpers for small persisted state files."""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def fsync_directory(path: str | Path) -> None:
    """Persist directory-entry changes where the platform supports it."""
    if os.name == "nt":  # pragma: no cover - Windows cannot fsync directories
        return
    try:
        directory_fd = os.open(Path(path), os.O_RDONLY)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
            return
        raise
    try:
        try:
            os.fsync(directory_fd)
        except OSError as error:
            if error.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(directory_fd)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    mode: int = 0o600,
) -> None:
    """Durably replace a UTF-8 text file without exposing partial contents."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        destination_mode = mode

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            os.chmod(temporary, destination_mode)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_lines(
    path: str | Path,
    lines: Iterable[str],
    *,
    mode: int = 0o600,
) -> None:
    """Durably replace a text file while streaming its contents."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        destination_mode = mode

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            os.chmod(temporary, destination_mode)
            temporary_file.writelines(lines)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = None,
    mode: int = 0o600,
) -> None:
    """Serialize JSON through :func:`atomic_write_text`."""
    atomic_write_text(
        path,
        json.dumps(value, indent=indent),
        mode=mode,
    )


def durable_unlink(path: str | Path) -> None:
    """Remove a file and persist the directory-entry change."""
    target = Path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        return
    fsync_directory(target.parent)
