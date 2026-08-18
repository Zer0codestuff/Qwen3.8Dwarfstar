from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from .config import RuntimeConfig


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


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


def build_generate_command(
    config: RuntimeConfig,
    *,
    prompt: str | None,
    system: str | None,
    max_tokens: int,
    sampling: SamplingConfig,
    thinking: str,
    reasoning_effort: str | None = None,
    chat: bool = False,
    kv_bits: float = 0,
    verbose: bool = True,
) -> list[str]:
    if max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("invalid thinking mode")
    if reasoning_effort not in {None, "xhigh", "medium", "low"}:
        raise ValueError("invalid reasoning effort")

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
    if kv_bits:
        command.extend(["--kv-bits", str(kv_bits)])
    if sampling.top_p != 1.0 or sampling.top_k:
        generation_args: dict[str, float | int] = {"top_p": sampling.top_p}
        if sampling.top_k:
            generation_args["top_k"] = sampling.top_k
        command.extend(["--gen-kwargs", json.dumps(generation_args)])
    command.extend(_mtp_args(config))
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
    kv_bits: float = 0,
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
    if kv_bits:
        command.extend(["--kv-bits", str(kv_bits)])
    command.extend(_mtp_args(config))
    return command
