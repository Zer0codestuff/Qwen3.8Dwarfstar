from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO


class RuntimeBusyError(ValueError):
    pass


def lock_path() -> Path:
    override = os.environ.get("DWARFSTAR_RUNTIME_LOCK")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / ".dwarfstar" / "runtime.lock"


def acquire_runtime_lock() -> IO[str]:
    """Prevent two 12+ GB model processes from sharing a 16 GB Mac."""

    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeBusyError(
            "another DwarfStar model process is already running; stop the "
            "chat, server or benchmark before starting a second one"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle
