from __future__ import annotations

import os
from pathlib import Path

from .config import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MTP_MODEL,
    DEFAULT_MTP_MODEL_REVISION,
)


PINNED_REVISIONS = {
    DEFAULT_MODEL: DEFAULT_MODEL_REVISION,
    DEFAULT_MTP_MODEL: DEFAULT_MTP_MODEL_REVISION,
}


def pinned_revision(reference: str) -> str | None:
    return PINNED_REVISIONS.get(reference)


def materialize_model(reference: str, *, local_files_only: bool = False) -> Path | str:
    """Resolve default Hub models at immutable revisions.

    Custom local paths and custom repositories are intentionally returned as-is.
    The two shipped defaults are always resolved to a snapshot directory so the
    runtime cannot silently move when a Hub repository updates its main branch.
    """

    candidate = Path(reference).expanduser()
    if candidate.exists():
        return candidate.resolve()
    revision = pinned_revision(reference)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    if not local_files_only:
        try:
            return Path(
                snapshot_download(
                    reference,
                    revision=revision,
                    local_files_only=True,
                )
            )
        except Exception:
            pass

    return Path(
        snapshot_download(
            reference,
            revision=revision,
            local_files_only=local_files_only,
            max_workers=3,
        )
    )


def weight_bytes(path: Path | str) -> int:
    candidate = Path(path)
    if not candidate.is_dir():
        return 0
    return sum(item.stat().st_size for item in candidate.glob("*.safetensors"))
