from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
DEFAULT_MODEL_REVISION = "c98bba5926f51fec1c8d8737e577221673f524d7"
DEFAULT_MODEL_WEIGHT_BYTES = 11_771_374_457
DEFAULT_MTP_MODEL = "lukaskremla/Qwen3.8-27B-MTP-3bit-MLX"
DEFAULT_MTP_MODEL_REVISION = "9d061a0661258e75b401a11ac9fa22fc648e039d"
DEFAULT_MTP_MODEL_WEIGHT_BYTES = 185_849_974
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_KV_BITS = 8.0
DEFAULT_PREFILL_STEP_SIZE = 256

# Context caps measured on the M4 16 GB validation machine (2026-08-18).
# Only 16 of the 64 layers keep a KV cache (the rest are GatedDeltaNet with
# constant state), so KV quantization is what turns "theoretical context"
# into context that actually prefills without a Metal command-buffer OOM:
#   - bf16 KV OOMs on a 3.3K-token prompt even at prefill chunk 128;
#   - 8-bit KV completes 4096 at chunk 128 (peak 12.49 GB);
#   - 4-bit KV completes 8192 at chunk 64 (peak 12.40 GB);
#   - 8-bit KV at 8192 still OOMs mid-prefill.
SAFE_MAX_CONTEXT_SIZE = 2048
SAFE_KV8_MAX_CONTEXT_SIZE = 4096
SAFE_KV4_MAX_CONTEXT_SIZE = 8192
SAFE_MTP_MAX_CONTEXT_SIZE = 1024


def safe_max_context_size(kv_bits: float) -> int:
    if not kv_bits:
        return SAFE_MAX_CONTEXT_SIZE
    if kv_bits < 6:
        return SAFE_KV4_MAX_CONTEXT_SIZE
    return SAFE_KV8_MAX_CONTEXT_SIZE


def safe_max_prefill_step(context_size: int) -> int:
    """Largest prefill chunk that survived long-prompt prefill at this size."""

    if context_size <= 2048:
        return 256
    if context_size <= 4096:
        return 128
    return 64


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    mtp_model: str
    context_size: int
    prefill_step_size: int
    kv_bits: float
    use_mtp: bool
    mtp_block_size: int


def runtime_config(
    *,
    model: str | None = None,
    mtp_model: str | None = None,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    kv_bits: float = DEFAULT_KV_BITS,
    use_mtp: bool = False,
    mtp_block_size: int = 3,
) -> RuntimeConfig:
    kv_bits = float(kv_bits)
    if context_size < 64:
        raise ValueError("context size must be at least 64 tokens")
    if prefill_step_size < 32:
        raise ValueError("prefill step size must be at least 32 tokens")
    if kv_bits and not 2 <= kv_bits <= 8:
        raise ValueError("kv-bits must be zero (bf16 KV) or between 2 and 8")
    if not 2 <= mtp_block_size <= 8:
        raise ValueError("MTP block size must be between 2 and 8")
    unsafe_context = os.environ.get("DWARFSTAR_ALLOW_UNSAFE_CONTEXT") == "1"
    max_context = safe_max_context_size(kv_bits)
    if context_size > max_context and not unsafe_context:
        hint = (
            "quantize the KV cache (e.g. --kv-bits 4)"
            if not kv_bits or kv_bits >= 6
            else "this is the measured 16 GB ceiling"
        )
        raise ValueError(
            f"context sizes above {max_context} are unsafe on a 16 GB Mac with "
            f"kv-bits={kv_bits:g}; {hint}, or set DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1 "
            "for diagnostics"
        )
    if use_mtp and context_size > SAFE_MTP_MAX_CONTEXT_SIZE and not unsafe_context:
        raise ValueError(
            f"MTP context sizes above {SAFE_MTP_MAX_CONTEXT_SIZE} are unsafe on a 16 GB Mac; "
            "disable MTP or set DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1"
        )
    if not unsafe_context:
        prefill_step_size = min(prefill_step_size, safe_max_prefill_step(context_size))
    if prefill_step_size > context_size:
        prefill_step_size = context_size
    return RuntimeConfig(
        model=model or os.environ.get("DWARFSTAR_MODEL", DEFAULT_MODEL),
        mtp_model=mtp_model
        or os.environ.get("DWARFSTAR_MTP_MODEL", DEFAULT_MTP_MODEL),
        context_size=context_size,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        use_mtp=use_mtp,
        mtp_block_size=mtp_block_size,
    )
