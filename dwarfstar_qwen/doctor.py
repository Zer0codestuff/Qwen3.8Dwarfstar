from __future__ import annotations

import json
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_WEIGHT_BYTES,
    DEFAULT_MTP_MODEL,
    DEFAULT_MTP_MODEL_WEIGHT_BYTES,
)
from .model_store import materialize_model, weight_bytes


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _sysctl_int(name: str) -> int:
    try:
        value = subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        )
        return int(value.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


def _cached_snapshot(repo_id: str) -> Path | None:
    try:
        return Path(materialize_model(repo_id, local_files_only=True))
    except Exception:
        return None


def run_checks() -> list[Check]:
    checks = [
        Check(
            "Apple Silicon",
            sys.platform == "darwin" and platform.machine() == "arm64",
            f"{platform.system()} {platform.machine()}",
        ),
        Check(
            "Unified memory",
            _sysctl_int("hw.memsize") >= 16 * 1024**3,
            f"{_sysctl_int('hw.memsize') / 1024**3:.1f} GiB",
        ),
    ]
    distribution_names = {"mlx": "mlx", "mlx_vlm": "mlx-vlm"}
    for module_name in ("mlx", "mlx_vlm"):
        try:
            __import__(module_name)
            detail = importlib.metadata.version(distribution_names[module_name])
            checks.append(Check(module_name, True, detail))
        except Exception as exc:
            checks.append(Check(module_name, False, type(exc).__name__))

    try:
        import mlx.core as mx

        device = mx.device_info()
        checks.append(Check("MLX Metal", True, str(device.get("device_name", device))))
        recommended = int(device.get("max_recommended_working_set_size", 0))
        checks.append(
            Check(
                "Metal working set",
                recommended >= 12_000_000_000,
                f"{recommended / 1e9:.2f} GB recommended",
            )
        )
    except Exception as exc:
        checks.append(Check("MLX Metal", False, str(exc)))

    target = _cached_snapshot(DEFAULT_MODEL)
    draft = _cached_snapshot(DEFAULT_MTP_MODEL)
    target_bytes = weight_bytes(target) if target else 0
    draft_bytes = weight_bytes(draft) if draft else 0
    checks.extend(
        [
            Check(
                "Qwen target",
                target is not None and target_bytes == DEFAULT_MODEL_WEIGHT_BYTES,
                f"{target_bytes / 1e9:.2f} GB"
                if target
                else "not downloaded",
            ),
            Check(
                "Qwen MTP head",
                draft is None or draft_bytes == DEFAULT_MTP_MODEL_WEIGHT_BYTES,
                f"{draft_bytes / 1e9:.2f} GB"
                if draft
                else "optional; not downloaded",
            ),
        ]
    )
    return checks


def print_report(*, as_json: bool = False) -> int:
    checks = run_checks()
    if as_json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            marker = "OK" if check.ok else "FAIL"
            print(f"[{marker:4}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1
