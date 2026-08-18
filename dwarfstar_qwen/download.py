from __future__ import annotations

import os
from pathlib import Path

from .config import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_WEIGHT_BYTES,
    DEFAULT_MTP_MODEL,
    DEFAULT_MTP_MODEL_WEIGHT_BYTES,
)
from .model_store import materialize_model, weight_bytes


def download_models(*, model: str, mtp_model: str, include_mtp: bool) -> list[Path]:
    # Xet can stall on large anonymous shard downloads on some macOS networks.
    # The regular Hub downloader is resumable and gives us a predictable setup.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    repos = [model]
    if include_mtp:
        repos.append(mtp_model)
    paths = []
    expected_sizes = {
        DEFAULT_MODEL: DEFAULT_MODEL_WEIGHT_BYTES,
        DEFAULT_MTP_MODEL: DEFAULT_MTP_MODEL_WEIGHT_BYTES,
    }
    for repo in repos:
        print(f"Downloading {repo} ...", flush=True)
        path = Path(materialize_model(repo))
        paths.append(path)
        size = weight_bytes(path)
        expected = expected_sizes.get(repo)
        if expected is not None and size != expected:
            raise RuntimeError(
                f"weight size mismatch for {repo}: expected {expected}, found {size}"
            )
        print(f"Ready: {path} ({size / 1e9:.2f} GB of weights)")
    return paths


def download_defaults(*, include_mtp: bool = True) -> list[Path]:
    return download_models(
        model=DEFAULT_MODEL,
        mtp_model=DEFAULT_MTP_MODEL,
        include_mtp=include_mtp,
    )
