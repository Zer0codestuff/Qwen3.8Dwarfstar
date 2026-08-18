"""Launch MLX-VLM with the memory profile required by a 16 GB Mac.

MLX-VLM's Qwen decode optimization concatenates four already-quantized GDN
projection matrices and permanently caches the concatenated copy.  For this
27B model that costs about 1.8 GB, which is a good speed trade on a large Mac
but exceeds the M4 16 GB Metal working-set budget.  The low-memory profile
keeps the original matrices and lets MLX dispatch their matmuls separately.
"""

from __future__ import annotations

import os
import sys
import traceback

from .runtime_lock import acquire_runtime_lock


DEFAULT_CACHE_LIMIT_MB = 64


def apply_low_memory_profile() -> None:
    import mlx.core as mx

    cache_limit_mb = int(
        os.environ.get("DWARFSTAR_MLX_CACHE_LIMIT_MB", DEFAULT_CACHE_LIMIT_MB)
    )
    if cache_limit_mb < 0:
        raise ValueError("DWARFSTAR_MLX_CACHE_LIMIT_MB cannot be negative")
    device = mx.device_info()
    recommended_bytes = int(device.get("max_recommended_working_set_size", 0))
    memory_limit_value = os.environ.get("DWARFSTAR_MLX_MEMORY_LIMIT_MB")
    if memory_limit_value is not None:
        memory_limit_mb = int(memory_limit_value)
        if memory_limit_mb <= 0:
            raise ValueError("DWARFSTAR_MLX_MEMORY_LIMIT_MB must be positive")
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
    else:
        memory_limit_bytes = recommended_bytes
    if memory_limit_bytes:
        # MLX otherwise permits more than Metal's recommended working set (up
        # to 1.5x, also capped by physical memory).  That is useful on larger
        # machines but permits this 16 GB profile to submit
        # command buffers which cannot fit.  Use the device-reported budget as
        # MLX's allocator/GC guideline (it is not a strict allocation ceiling;
        # advanced users may override it explicitly).
        mx.set_memory_limit(memory_limit_bytes)
    mx.set_cache_limit(cache_limit_mb * 1024 * 1024)

    if os.environ.get("DWARFSTAR_ALLOW_FUSED_GDN") == "1":
        return

    from mlx_vlm.models.qwen3_5 import language

    if not hasattr(language, "_decode_quantized_linears_fused"):
        raise RuntimeError(
            "the installed mlx-vlm release is incompatible with DwarfStar's "
            "16 GB low-memory patch; reinstall requirements-macos.txt"
        )

    if not hasattr(language, "_dwarfstar_original_fused_decode"):
        language._dwarfstar_original_fused_decode = (  # type: ignore[attr-defined]
            language._decode_quantized_linears_fused
        )

    # Returning None selects the existing, numerically identical unfused path.
    # It avoids a persistent concatenated copy of all four GDN projections.
    language._decode_quantized_linears_fused = lambda linears, x: None


def pin_default_models(args: list[str]) -> list[str]:
    from .config import DEFAULT_MODEL, DEFAULT_MTP_MODEL
    from .model_store import materialize_model

    resolved = list(args)
    for flag, default in (("--model", DEFAULT_MODEL), ("--draft-model", DEFAULT_MTP_MODEL)):
        try:
            index = resolved.index(flag)
        except ValueError:
            continue
        if index + 1 < len(resolved) and resolved[index + 1] == default:
            resolved[index + 1] = str(materialize_model(default))
    return resolved


def pop_option(args: list[str], flag: str) -> tuple[list[str], str | None]:
    remaining = list(args)
    try:
        index = remaining.index(flag)
    except ValueError:
        return remaining, None
    if index + 1 >= len(remaining):
        raise ValueError(f"{flag} requires a value")
    value = remaining[index + 1]
    del remaining[index : index + 2]
    return remaining, value


def apply_reasoning_effort(reasoning_effort: str | None) -> None:
    if reasoning_effort is None:
        return
    if reasoning_effort not in {"xhigh", "medium", "low"}:
        raise ValueError("invalid reasoning effort")

    from mlx_vlm.generate import dispatch

    original = dispatch.apply_chat_template

    def with_reasoning_effort(*args, **kwargs):
        kwargs.setdefault("reasoning_effort", reasoning_effort)
        return original(*args, **kwargs)

    dispatch.apply_chat_template = with_reasoning_effort


def option_value(args: list[str], flag: str) -> str | None:
    try:
        index = args.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(args):
        raise ValueError(f"{flag} requires a value")
    return args[index + 1]


def apply_interactive_chat_limits(args: list[str]) -> None:
    """Repair upstream chat's missing KV/context forwarding.

    The MLX-VLM 0.6.14 interactive branch does not pass max_kv_size or KV
    quantization to stream_generate.  It also drops draft-model arguments and
    cannot round-trip Qwen reasoning content correctly, so DwarfStar rejects
    those two modes and keeps this bounded serial chat path.
    """

    if "--draft-model" in args:
        raise ValueError("interactive chat does not support MTP; use --no-mtp")
    if "--enable-thinking" in args:
        raise ValueError(
            "interactive multi-turn thinking is unsupported; use one-shot generate"
        )
    max_kv_value = option_value(args, "--max-kv-size")
    if max_kv_value is None:
        raise ValueError("interactive chat requires an explicit context limit")
    max_kv_size = int(max_kv_value)
    kv_bits_value = option_value(args, "--kv-bits")
    kv_bits = float(kv_bits_value) if kv_bits_value is not None else 0.0

    from mlx_vlm.generate import dispatch

    if hasattr(dispatch, "_dwarfstar_original_chat_stream_generate"):
        return
    original = dispatch.stream_generate
    dispatch._dwarfstar_original_chat_stream_generate = original

    def bounded_stream_generate(model, processor, prompt, *positional, **kwargs):
        tokenizer = getattr(processor, "tokenizer", processor)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) >= max_kv_size:
            raise RuntimeError(
                f"interactive chat context is {len(prompt_tokens)} tokens and "
                f"exceeds the safe {max_kv_size - 1}-token prompt budget; "
                "restart the chat"
            )
        kwargs.setdefault("max_kv_size", max_kv_size)
        if kv_bits:
            kwargs.setdefault("kv_bits", kv_bits)
        yield from original(model, processor, prompt, *positional, **kwargs)

    dispatch.stream_generate = bounded_stream_generate


def normalize_server_messages(prompt):
    if not isinstance(prompt, list):
        return prompt
    instructions = []
    conversation = []
    for message in prompt:
        if isinstance(message, dict) and message.get("role") in {
            "system",
            "developer",
        }:
            instructions.append(message.get("content", ""))
        else:
            conversation.append(message)
    if not instructions:
        return prompt
    merged = "\n\n".join(str(value) for value in instructions)
    return [{"role": "system", "content": merged}, *conversation]


def allowed_model_loader(original, target_path: str):
    from fastapi import HTTPException

    from .config import DEFAULT_MODEL

    aliases = {
        DEFAULT_MODEL: target_path,
        "dwarfstar-qwen": target_path,
        "qwen3.8-27b": target_path,
        target_path: target_path,
    }

    def get_cached_model(model_path, *args, **kwargs):
        model_key = str(model_path)
        model_kind = kwargs.get("model_kind")
        if model_key not in aliases:
            raise HTTPException(
                status_code=400,
                detail="this 16 GB server only permits the pinned DwarfStar Qwen model",
            )
        if model_kind not in (None, "text_generation"):
            raise HTTPException(
                status_code=400,
                detail="this DwarfStar build serves text generation only",
            )
        return original(aliases[model_key], *args, **kwargs)

    return get_cached_model


def positive_generation_args(original):
    """Reject zero/negative API generation budgets before decode starts."""

    def build_gen_args(*args, **kwargs):
        generation_args = original(*args, **kwargs)
        if generation_args.max_tokens < 1:
            raise ValueError("max tokens must be positive")
        return generation_args

    return build_gen_args


def apply_server_model_aliases(target_path: str) -> None:
    """Keep API model names stable while the server uses a pinned local path."""

    import importlib

    from .config import DEFAULT_MODEL
    import mlx_vlm.server as server
    anthropic_module = importlib.import_module("mlx_vlm.server.anthropic")
    app_module = importlib.import_module("mlx_vlm.server.app")
    openai_module = importlib.import_module("mlx_vlm.server.openai")

    if hasattr(server, "_dwarfstar_original_get_cached_model"):
        return
    original = server.get_cached_model
    get_cached_model = allowed_model_loader(original, target_path)

    original_template = server.apply_chat_template

    def normalized_apply_chat_template(processor, config, prompt, *args, **kwargs):
        prompt = normalize_server_messages(prompt)
        return original_template(processor, config, prompt, *args, **kwargs)

    server._dwarfstar_original_get_cached_model = original
    server.get_cached_model = get_cached_model
    app_module.get_cached_model = get_cached_model
    server.apply_chat_template = normalized_apply_chat_template
    app_module.apply_chat_template = normalized_apply_chat_template

    # Upstream's request schemas accept max_tokens <= 0.  Its text endpoints
    # already translate argument-construction errors to HTTP 400, so install a
    # shared positive-budget check without changing the response contract.
    original_build_gen_args = app_module._build_gen_args
    build_gen_args = positive_generation_args(original_build_gen_args)
    app_module._dwarfstar_original_build_gen_args = original_build_gen_args
    app_module._build_gen_args = build_gen_args
    openai_module._build_gen_args = build_gen_args
    anthropic_module._build_gen_args = build_gen_args


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"generate", "server"}:
        print("runner requires 'generate' or 'server'", file=sys.stderr)
        return 2

    mode = args.pop(0)
    args, reasoning_effort = pop_option(args, "--dwarfstar-reasoning-effort")
    apply_low_memory_profile()
    _runtime_lock = acquire_runtime_lock()
    args = pin_default_models(args)
    if mode == "generate":
        apply_reasoning_effort(reasoning_effort)
        if "--chat" in args:
            apply_interactive_chat_limits(args)
    else:
        try:
            model_index = args.index("--model")
            target_path = args[model_index + 1]
        except (ValueError, IndexError):
            target_path = ""
        if target_path:
            apply_server_model_aliases(target_path)
    sys.argv = [f"dwarfstar-{mode}", *args]
    if mode == "generate":
        from mlx_vlm.generate.cli import main as upstream_main
    else:
        from mlx_vlm.server import main as upstream_main
    try:
        result = upstream_main()
    except EOFError:
        if mode == "generate" and "--chat" in args:
            # Ctrl-D is the conventional successful exit from an interactive
            # terminal.  Upstream's input() loop exposes it as an exception.
            print()
            return 0
        raise
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        exit_code = 130
    except ValueError as exc:
        print(f"dwarfstar: {exc}", file=sys.stderr)
        exit_code = 2
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # With this 27B checkpoint the Metal allocator can still own almost the
    # complete recommended working set when Python starts destroying modules.
    # mlx/mlx-vlm may then crash in interpreter finalization even though the
    # generation completed successfully.  Flush user-visible output and let
    # the OS reclaim Metal resources atomically instead of running that unsafe
    # teardown path.  This is intentionally confined to the dedicated worker
    # process; the public CLI and benchmark process retain normal cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
