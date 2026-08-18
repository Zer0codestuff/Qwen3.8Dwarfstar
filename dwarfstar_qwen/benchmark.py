from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MTP_MODEL,
    DEFAULT_MTP_MODEL_REVISION,
    runtime_config,
)
from .model_store import materialize_model
from .profile import runtime_fingerprint, write_profile
from .runner import apply_low_memory_profile
from .runtime_lock import acquire_runtime_lock


COMPLEX_PROMPTS = (
    (
        "reasoning",
        """A box contains 12 red, 10 blue, and 8 green balls. Without replacement, six
balls are drawn. Explain a rigorous dynamic-programming method to compute the
probability that every color appears at least once, then give the reduced exact
fraction. Keep the derivation auditable, avoid introductory material, and answer
in at most 140 words.""",
    ),
    (
        "code",
        """Review this Python function. Identify the concurrency bug, explain an
interleaving that triggers it, and provide a minimal corrected implementation:

async def get_or_create(key):
    if key not in cache:
        cache[key] = await expensive_fetch(key)
    return cache[key]

Assume many tasks can request the same key and cancellation is possible. Include
the corrected code and keep the whole answer under 180 words.""",
    ),
    (
        "technical_synthesis",
        """Write a short technical note deciding whether a 27B model quantized to
3 bits is suitable for a Mac with 16 GB. Distinguish weight memory, KV cache,
recurrent state, temporary buffers and quality. Conclude with three measurable
acceptance criteria, no slogans. Limit: 170 words.""",
    ),
)


@dataclass
class Trial:
    prompt: str
    mode: str
    repeat: int
    prompt_tokens: int
    generated_tokens: int
    prompt_tps: float
    generation_tps: float
    elapsed_seconds: float
    peak_memory_gb: float
    finish_reason: str | None
    text: str
    accepted_draft_tokens: int | None = None
    proposed_draft_tokens: int | None = None


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _render_prompt(processor: Any, config: Any, user_prompt: str) -> str:
    from mlx_vlm import apply_chat_template

    return apply_chat_template(
        processor,
        config,
        [{"role": "user", "content": user_prompt}],
        enable_thinking=False,
        thinking_mode="disabled",
    )


def _draft_counts(draft_model: Any) -> tuple[int | None, int | None]:
    if draft_model is None:
        return None, None
    accepted = getattr(draft_model, "accept_lens", None)
    proposed = getattr(draft_model, "draft_lens", None)
    if not isinstance(accepted, list) or not isinstance(proposed, list):
        return None, None
    return sum(int(value) for value in accepted), sum(int(value) for value in proposed)


def _run_trial(
    *,
    model: Any,
    processor: Any,
    draft_model: Any,
    draft_kind: str | None,
    prompt_name: str,
    prompt: str,
    mode: str,
    repeat: int,
    max_tokens: int,
    context_size: int,
    prefill_step_size: int,
    mtp_block_size: int,
) -> Trial:
    import mlx.core as mx
    from mlx_vlm import generate

    rendered = _render_prompt(processor, model.config, prompt)
    mx.reset_peak_memory()
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "max_kv_size": context_size,
        "prefill_step_size": prefill_step_size,
        "skip_special_tokens": False,
        "verbose": False,
    }
    if draft_model is not None:
        kwargs.update(
            draft_model=draft_model,
            draft_kind=draft_kind,
            draft_block_size=mtp_block_size,
        )

    started = time.perf_counter()
    result = generate(model, processor, rendered, **kwargs)
    elapsed = time.perf_counter() - started
    accepted, proposed = _draft_counts(draft_model)
    return Trial(
        prompt=prompt_name,
        mode=mode,
        repeat=repeat,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generation_tokens,
        prompt_tps=result.prompt_tps,
        generation_tps=result.generation_tps,
        elapsed_seconds=elapsed,
        peak_memory_gb=result.peak_memory,
        finish_reason=result.finish_reason,
        text=result.text,
        accepted_draft_tokens=accepted,
        proposed_draft_tokens=proposed,
    )


def _median(trials: list[Trial], mode: str, field: str) -> float | None:
    values = [float(getattr(trial, field)) for trial in trials if trial.mode == mode]
    return statistics.median(values) if values else None


def _parity(trials: list[Trial]) -> tuple[bool | None, list[str]]:
    serial = {(trial.prompt, trial.repeat): trial.text for trial in trials if trial.mode == "serial"}
    mtp = {(trial.prompt, trial.repeat): trial.text for trial in trials if trial.mode == "mtp"}
    shared = sorted(serial.keys() & mtp.keys())
    if not shared:
        return None, []
    mismatches = [f"{name}#{repeat}" for name, repeat in shared if serial[(name, repeat)] != mtp[(name, repeat)]]
    return not mismatches, mismatches


def _load_runtime(model_reference: str, mtp_reference: str, include_mtp: bool):
    from mlx_vlm import load

    model_path = materialize_model(model_reference)
    model, processor = load(str(model_path))
    if not include_mtp:
        return model, processor, None, None

    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

    draft_path = materialize_model(mtp_reference)
    draft_model, draft_kind = load_drafter(str(draft_path), kind="mtp")
    validate_drafter_compatibility(model, draft_model, draft_kind)
    return model, processor, draft_model, draft_kind


def run_cli_benchmark(args: Any) -> int:
    if args.max_tokens < 1 or args.repeats < 1:
        raise ValueError("max tokens and repeats must be positive")
    # The benchmark measures the plain serial/MTP decode paths without KV
    # quantization, so its context cap is the bf16 one.
    config = runtime_config(
        model=args.model,
        mtp_model=args.mtp_model,
        context_size=args.ctx_size,
        prefill_step_size=args.prefill_step_size,
        kv_bits=0.0,
        use_mtp=args.mode in {"mtp", "both"},
        mtp_block_size=args.mtp_block_size,
    )
    apply_low_memory_profile()
    _runtime_lock = acquire_runtime_lock()

    include_mtp = args.mode in {"mtp", "both"}
    print("Loading the pinned 3-bit Qwen target...", flush=True)
    model, processor, draft_model, draft_kind = _load_runtime(
        config.model, config.mtp_model, include_mtp
    )

    modes = ["serial", "mtp"] if args.mode == "both" else [args.mode]
    backend_errors: dict[str, str] = {}

    def is_memory_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("insufficient memory", "out of memory", "memory limit")
        )

    def disable_mtp(exc: BaseException) -> None:
        nonlocal draft_model, draft_kind
        backend_errors["mtp"] = f"{type(exc).__name__}: {exc}"
        action = (
            "no MTP trials will run"
            if args.mode == "mtp"
            else "continuing with serial decoding"
        )
        print(
            f"MTP unavailable on this memory profile; {action} "
            f"({backend_errors['mtp']}).",
            flush=True,
        )
        # An MTP verification failure may leave both the small draft head and
        # failed command-buffer temporaries resident.  Drop every reference,
        # collect Python containers and then release MLX's reusable cache before
        # asking Metal to execute another target-only pass.
        draft_model = None
        draft_kind = None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()

    if not args.no_warmup:
        print("Warming Metal kernels...", flush=True)
        for mode in modes:
            if mode in backend_errors:
                continue
            try:
                _run_trial(
                    model=model,
                    processor=processor,
                    draft_model=draft_model if mode == "mtp" else None,
                    draft_kind=draft_kind if mode == "mtp" else None,
                    prompt_name="warmup",
                    prompt="Reply with exactly one short sentence about local inference.",
                    mode=mode,
                    repeat=-1,
                    max_tokens=min(4, args.max_tokens),
                    context_size=config.context_size,
                    prefill_step_size=config.prefill_step_size,
                    mtp_block_size=config.mtp_block_size,
                )
            except RuntimeError as exc:
                if mode != "mtp" or not is_memory_error(exc):
                    raise
                disable_mtp(exc)

    trials: list[Trial] = []
    for repeat in range(args.repeats):
        for prompt_name, prompt in COMPLEX_PROMPTS:
            for mode in modes:
                if mode in backend_errors:
                    continue
                print(f"[{mode}] {prompt_name} (run {repeat + 1}/{args.repeats})", flush=True)
                try:
                    trial = _run_trial(
                        model=model,
                        processor=processor,
                        draft_model=draft_model if mode == "mtp" else None,
                        draft_kind=draft_kind if mode == "mtp" else None,
                        prompt_name=prompt_name,
                        prompt=prompt,
                        mode=mode,
                        repeat=repeat,
                        max_tokens=args.max_tokens,
                        context_size=config.context_size,
                        prefill_step_size=config.prefill_step_size,
                        mtp_block_size=config.mtp_block_size,
                    )
                except RuntimeError as exc:
                    if mode != "mtp" or not is_memory_error(exc):
                        raise
                    disable_mtp(exc)
                    continue
                trials.append(trial)
                print(
                    f"  {trial.generation_tps:.2f} tok/s, "
                    f"prefill {trial.prompt_tps:.1f} tok/s, "
                    f"peak {trial.peak_memory_gb:.2f} GB",
                    flush=True,
                )

    parity_ok, mismatches = _parity(trials)
    serial_tps = _median(trials, "serial", "generation_tps")
    mtp_tps = _median(trials, "mtp", "generation_tps")
    speedup = mtp_tps / serial_tps if serial_tps and mtp_tps else None
    mtp_peak = max(
        (trial.peak_memory_gb for trial in trials if trial.mode == "mtp"),
        default=0.0,
    )

    import mlx.core as mx

    device = mx.device_info()
    recommended_bytes = int(device.get("max_recommended_working_set_size", 0))
    memory_safe: bool | None
    if args.mode == "serial":
        # A target-only run contains no evidence about MTP's verification
        # workspace.  Report that honestly instead of treating a zero MTP peak
        # as a successful memory check.
        memory_safe = None
    else:
        memory_safe = "mtp" not in backend_errors and (
            not recommended_bytes or mtp_peak <= 0.98 * recommended_bytes / 1e9
        )
    recommend_mtp = bool(
        parity_ok is True
        and speedup is not None
        and speedup >= 1.03
        and memory_safe is True
    )
    profile = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_revision": DEFAULT_MODEL_REVISION,
        "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
        "recommended_backend": "mtp" if recommend_mtp else "serial",
        "parity_ok": parity_ok,
        "memory_safe": memory_safe,
        "serial_median_generation_tps": serial_tps,
        "mtp_median_generation_tps": mtp_tps,
        "mtp_speedup": speedup,
        "mtp_block_size": config.mtp_block_size,
        "backend_errors": backend_errors,
        "runtime_fingerprint": runtime_fingerprint(
            context_size=config.context_size,
            prefill_step_size=config.prefill_step_size,
            mtp_block_size=config.mtp_block_size,
        ),
    }
    default_pair = args.model == DEFAULT_MODEL and args.mtp_model == DEFAULT_MTP_MODEL
    profile_file = write_profile(profile) if default_pair else None

    payload = {
        "schema": 1,
        "created_at": profile["created_at"],
        "hardware": {
            "platform": platform.platform(),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "memory_bytes": int(_sysctl("hw.memsize")) if _sysctl("hw.memsize").isdigit() else None,
            "metal": device,
        },
        "software": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-vlm", "transformers", "tokenizers")
        },
        "configuration": {
            "model": config.model,
            "model_revision": DEFAULT_MODEL_REVISION,
            "mtp_model": config.mtp_model,
            "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
            "mode": args.mode,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "context_size": config.context_size,
            "prefill_step_size": config.prefill_step_size,
            "mtp_block_size": config.mtp_block_size,
            "thinking": "disabled",
            "temperature": 0.0,
        },
        "summary": {**profile, "mismatches": mismatches},
        "trials": [asdict(trial) for trial in trials],
    }
    output = Path(args.output) if args.output else Path("benchmark-results/qwen38-m4.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Results: {output.resolve()}")
    if profile_file is not None:
        print(f"Runtime profile: {profile_file}")
    else:
        print("Runtime profile: not updated for custom model references")
    if speedup is not None:
        print(f"MTP speedup: {speedup:.3f}x; parity: {parity_ok}; memory-safe: {memory_safe}")
    print(f"Selected default: {profile['recommended_backend']}")
    if args.mode == "mtp" and "mtp" in backend_errors:
        return 4
    return 0 if parity_ok is not False else 3
