from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MTP_MODEL_REVISION,
    DEFAULT_PREFILL_STEP_SIZE,
)


def profile_path() -> Path:
    override = os.environ.get("DWARFSTAR_PROFILE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / ".dwarfstar" / "profile.json"


def read_profile() -> dict[str, Any] | None:
    try:
        value = json.loads(profile_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def runtime_fingerprint(
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    mtp_block_size: int = 3,
) -> dict[str, Any]:
    try:
        import mlx.core as mx

        device = mx.device_info()
    except Exception:
        device = {}
    versions = {}
    for distribution in ("mlx", "mlx-vlm", "transformers", "tokenizers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "machine": platform.machine(),
        "device_name": device.get("device_name"),
        "device_architecture": device.get("architecture"),
        "recommended_working_set_bytes": int(
            device.get("max_recommended_working_set_size", 0)
        ),
        "versions": versions,
        "context_size": context_size,
        "prefill_step_size": prefill_step_size,
        "mtp_block_size": mtp_block_size,
    }


def recommended_mtp(
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    mtp_block_size: int = 3,
) -> bool:
    profile = read_profile()
    if not profile:
        return False
    if profile.get("target_revision") != DEFAULT_MODEL_REVISION:
        return False
    if profile.get("mtp_revision") != DEFAULT_MTP_MODEL_REVISION:
        return False
    expected_fingerprint = runtime_fingerprint(
        context_size=context_size,
        prefill_step_size=prefill_step_size,
        mtp_block_size=mtp_block_size,
    )
    if profile.get("runtime_fingerprint") != expected_fingerprint:
        return False
    return profile.get("recommended_backend") == "mtp"


def write_profile(profile: dict[str, Any]) -> Path:
    path = profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
