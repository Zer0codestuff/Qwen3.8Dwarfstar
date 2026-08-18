from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
DEFAULT_MODEL_REVISION = "c98bba5926f51fec1c8d8737e577221673f524d7"
DEFAULT_MODEL_WEIGHT_BYTES = 11_771_374_457
DEFAULT_MTP_MODEL = "lukaskremla/Qwen3.8-27B-MTP-3bit-MLX"
DEFAULT_MTP_MODEL_REVISION = "9d061a0661258e75b401a11ac9fa22fc648e039d"
DEFAULT_MTP_MODEL_WEIGHT_BYTES = 185_849_974
DEFAULT_CONTEXT_SIZE = 1024
DEFAULT_PREFILL_STEP_SIZE = 256
SAFE_MAX_CONTEXT_SIZE = 2048
SAFE_MTP_MAX_CONTEXT_SIZE = 1024


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    mtp_model: str
    context_size: int
    prefill_step_size: int
    use_mtp: bool
    mtp_block_size: int


def runtime_config(
    *,
    model: str | None = None,
    mtp_model: str | None = None,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    use_mtp: bool = False,
    mtp_block_size: int = 3,
) -> RuntimeConfig:
    if context_size < 64:
        raise ValueError("context size must be at least 64 tokens")
    if prefill_step_size < 32:
        raise ValueError("prefill step size must be at least 32 tokens")
    if not 2 <= mtp_block_size <= 8:
        raise ValueError("MTP block size must be between 2 and 8")
    unsafe_context = os.environ.get("DWARFSTAR_ALLOW_UNSAFE_CONTEXT") == "1"
    if context_size > SAFE_MAX_CONTEXT_SIZE and not unsafe_context:
        raise ValueError(
            f"context sizes above {SAFE_MAX_CONTEXT_SIZE} are unsafe on a 16 GB Mac; "
            "set DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1 to override"
        )
    if use_mtp and context_size > SAFE_MTP_MAX_CONTEXT_SIZE and not unsafe_context:
        raise ValueError(
            f"MTP context sizes above {SAFE_MTP_MAX_CONTEXT_SIZE} are unsafe on a 16 GB Mac; "
            "disable MTP or set DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1"
        )
    if prefill_step_size > context_size:
        prefill_step_size = context_size
    return RuntimeConfig(
        model=model or os.environ.get("DWARFSTAR_MODEL", DEFAULT_MODEL),
        mtp_model=mtp_model
        or os.environ.get("DWARFSTAR_MTP_MODEL", DEFAULT_MTP_MODEL),
        context_size=context_size,
        prefill_step_size=prefill_step_size,
        use_mtp=use_mtp,
        mtp_block_size=mtp_block_size,
    )
