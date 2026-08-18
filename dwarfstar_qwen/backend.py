from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from .config import RuntimeConfig


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


def _mtp_args(config: RuntimeConfig) -> list[str]:
    if not config.use_mtp:
        return []
    return [
        "--draft-model",
        config.mtp_model,
        "--draft-kind",
        "mtp",
        "--draft-block-size",
        str(config.mtp_block_size),
    ]


def _kv_args(config: RuntimeConfig) -> list[str]:
    if not config.kv_bits:
        return []
    # Upstream defers KV quantization until token 5000 by default, which would
    # make it a no-op inside DwarfStar's context caps.  Quantize from token 0:
    # that is the configuration the 16 GB memory profile was validated with.
    return ["--kv-bits", str(config.kv_bits), "--quantized-kv-start", "0"]


def build_generate_command(
    config: RuntimeConfig,
    *,
    prompt: str | None,
    system: str | None,
    max_tokens: int,
    sampling: SamplingConfig,
    thinking: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    chat: bool = False,
    verbose: bool = True,
) -> list[str]:
    if max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("invalid thinking mode")
    if reasoning_effort not in {None, "xhigh", "medium", "low"}:
        raise ValueError("invalid reasoning effort")
    if thinking_budget is not None and thinking_budget < 0:
        raise ValueError("thinking budget cannot be negative")

    command = [
        sys.executable,
        "-m",
        "dwarfstar_qwen.runner",
        "generate",
        "--model",
        config.model,
        "--max-tokens",
        str(max_tokens),
        "--max-kv-size",
        str(config.context_size),
        "--prefill-step-size",
        str(config.prefill_step_size),
        "--temperature",
        str(sampling.temperature),
        "--presence-penalty",
        str(sampling.presence_penalty),
        "--repetition-penalty",
        str(sampling.repetition_penalty),
        "--thinking-mode",
        thinking,
        "--skip-special-tokens",
        "--verbose" if verbose else "--no-verbose",
    ]
    if prompt is not None:
        # Use argparse's --option=value form so prompts beginning with a dash
        # (for example ``--help``) cannot be reinterpreted as runner options.
        command.append(f"--prompt={prompt}")
    if system:
        command.append(f"--system={system}")
    if chat:
        command.append("--chat")
    if thinking != "disabled":
        command.append("--enable-thinking")
    if reasoning_effort is not None:
        command.extend(["--dwarfstar-reasoning-effort", reasoning_effort])
    if thinking_budget is not None and thinking != "disabled":
        command.extend(["--thinking-budget", str(thinking_budget)])
    command.extend(_kv_args(config))
    if sampling.top_p != 1.0 or sampling.top_k or sampling.min_p:
        generation_args: dict[str, float | int] = {
            "top_p": sampling.top_p,
            "min_p": sampling.min_p,
        }
        if sampling.top_k:
            generation_args["top_k"] = sampling.top_k
        command.extend(["--gen-kwargs", json.dumps(generation_args)])
    command.extend(_mtp_args(config))
    return command


def build_session_command(
    config: RuntimeConfig,
    *,
    mode: str,
    prompt: str | None,
    system: str | None,
    max_tokens: int,
    sampling: SamplingConfig,
    thinking: str,
    reasoning_effort: str,
    thinking_budget: int | None,
    answer_reserve: int,
    show_thinking: bool = False,
    preserve_thinking: bool = True,
    show_stats: bool = False,
    quiet: bool = False,
) -> list[str]:
    if mode not in {"ask", "chat"}:
        raise ValueError("session mode must be ask or chat")
    if config.use_mtp:
        raise ValueError("the memory-safe interactive runtime is serial-only")
    if max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("invalid thinking mode")
    if reasoning_effort not in {"xhigh", "medium", "low"}:
        raise ValueError("invalid reasoning effort")
    if answer_reserve < 0 or answer_reserve >= max_tokens:
        raise ValueError("answer reserve must be non-negative and smaller than max tokens")

    command = [
        sys.executable,
        "-m",
        "dwarfstar_qwen.runner",
        mode,
        "--model",
        config.model,
        "--max-tokens",
        str(max_tokens),
        "--max-kv-size",
        str(config.context_size),
        "--prefill-step-size",
        str(config.prefill_step_size),
        "--thinking-mode",
        thinking,
        "--reasoning-effort",
        reasoning_effort,
        "--answer-reserve",
        str(answer_reserve),
        "--temperature",
        str(sampling.temperature),
        "--top-p",
        str(sampling.top_p),
        "--top-k",
        str(sampling.top_k),
        "--min-p",
        str(sampling.min_p),
        "--presence-penalty",
        str(sampling.presence_penalty),
        "--repetition-penalty",
        str(sampling.repetition_penalty),
    ]
    if prompt is not None:
        command.append(f"--prompt={prompt}")
    if system:
        command.append(f"--system={system}")
    if thinking_budget is not None and thinking != "disabled":
        command.extend(["--thinking-budget", str(thinking_budget)])
    if config.kv_bits:
        command.extend(["--kv-bits", str(config.kv_bits)])
    command.append("--show-thinking" if show_thinking else "--no-show-thinking")
    command.append(
        "--preserve-thinking" if preserve_thinking else "--no-preserve-thinking"
    )
    if show_stats:
        command.append("--stats")
    if quiet:
        command.append("--quiet")
    return command


def build_server_command(
    config: RuntimeConfig,
    *,
    host: str,
    port: int,
    max_tokens: int,
    max_sequences: int,
    thinking: bool,
    api_key: str | None,
) -> list[str]:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if max_sequences != 1 and os.environ.get("DWARFSTAR_ALLOW_UNSAFE_SEQUENCES") != "1":
        raise ValueError(
            "the 16 GB profile supports exactly one server sequence; set "
            "DWARFSTAR_ALLOW_UNSAFE_SEQUENCES=1 only for diagnostics"
        )
    command = [
        sys.executable,
        "-m",
        "dwarfstar_qwen.runner",
        "server",
        "--model",
        config.model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-tokens",
        str(max_tokens),
        "--max-kv-size",
        str(config.context_size),
        "--prefill-step-size",
        str(config.prefill_step_size),
        "--max-num-seqs",
        str(max_sequences),
    ]
    if thinking:
        command.append("--enable-thinking")
    if api_key:
        command.extend(["--api-key", api_key])
    command.extend(_kv_args(config))
    command.extend(_mtp_args(config))
    return command
