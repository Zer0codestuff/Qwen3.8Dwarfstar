from __future__ import annotations

import argparse
import os
import sys
import traceback

from .backend import (
    SamplingConfig,
    build_generate_command,
    build_server_command,
    build_session_command,
)
from .config import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_MTP_MODEL,
    DEFAULT_PREFILL_STEP_SIZE,
    runtime_config,
)
from .doctor import print_report
from .download import download_models
from .presets import PROFILES, GenerationProfile, resolve_profile


COMMANDS = {"ask", "chat", "generate", "serve", "benchmark", "doctor", "download"}


def _runtime_arguments(
    parser: argparse.ArgumentParser, *, profile_context: bool = False
) -> None:
    parser.add_argument("--model", default=None, help=f"target model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--mtp-model", default=None, help=f"MTP head (default: {DEFAULT_MTP_MODEL})"
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=None if profile_context else DEFAULT_CONTEXT_SIZE,
    )
    parser.add_argument("--prefill-step-size", type=int, default=DEFAULT_PREFILL_STEP_SIZE)
    parser.add_argument("--mtp-block-size", type=int, default=3)
    mtp = parser.add_mutually_exclusive_group()
    mtp.add_argument(
        "--mtp",
        dest="mtp",
        action="store_true",
        help="force MTP speculative decoding",
    )
    mtp.add_argument(
        "--no-mtp",
        dest="mtp",
        action="store_false",
        help="force reliable serial decoding",
    )
    mtp.add_argument(
        "--mtp-auto",
        dest="mtp",
        action="store_const",
        const=None,
        help="use the local benchmark recommendation (default)",
    )
    parser.set_defaults(mtp=None)


def _generation_arguments(parser: argparse.ArgumentParser) -> None:
    _runtime_arguments(parser, profile_context=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="deep",
        help="deep (maximum quality), balanced or quick",
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--system")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument(
        "--thinking", choices=("enabled", "disabled")
    )
    parser.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"))
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--answer-reserve", type=int)
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="KV cache quantization bits; 0 disables (default: profile value)",
    )
    parser.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also show the generated reasoning",
    )
    parser.add_argument(
        "--preserve-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the reasoning in the next-turn cache",
    )
    parser.add_argument("--stats", action="store_true", help="mostra metriche MLX")
    parser.add_argument("--quiet", action="store_true")


def _generation_profile(args: argparse.Namespace) -> GenerationProfile:
    return resolve_profile(
        args.profile,
        context_size=args.ctx_size,
        kv_bits=args.kv_bits,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        thinking_budget=args.thinking_budget,
        answer_reserve=args.answer_reserve,
    )


def _config(
    args: argparse.Namespace, settings: GenerationProfile | None = None
):
    from .profile import recommended_mtp

    if settings is None and hasattr(args, "profile"):
        settings = _generation_profile(args)
    context_size = settings.context_size if settings else args.ctx_size
    kv_bits = settings.kv_bits if settings else args.kv_bits
    use_mtp = args.mtp
    if use_mtp is None:
        effective_model = args.model or os.environ.get("DWARFSTAR_MODEL", DEFAULT_MODEL)
        effective_mtp_model = args.mtp_model or os.environ.get(
            "DWARFSTAR_MTP_MODEL", DEFAULT_MTP_MODEL
        )
        is_default_target = effective_model == DEFAULT_MODEL
        is_default_draft = effective_mtp_model == DEFAULT_MTP_MODEL
        use_mtp = is_default_target and is_default_draft and recommended_mtp(
            context_size=context_size,
            prefill_step_size=min(args.prefill_step_size, context_size),
            mtp_block_size=args.mtp_block_size,
        )
    return runtime_config(
        model=args.model,
        mtp_model=args.mtp_model,
        context_size=context_size,
        prefill_step_size=args.prefill_step_size,
        kv_bits=kv_bits,
        use_mtp=use_mtp,
        mtp_block_size=args.mtp_block_size,
    )


def _exec(command: list[str]) -> int:
    os.execv(command[0], command)
    return 127


def _normalize_argv(argv: list[str]) -> list[str]:
    """Make no command mean chat, and plain text mean a one-shot question."""

    if not argv:
        return ["chat"]
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    if argv[0].startswith("-"):
        return ["chat", *argv]
    return ["ask", *argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwarfstar",
        description="Qwen 3.8 27B inference tuned for a 16 GB Apple Silicon Mac",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check hardware, MLX and model cache")
    doctor.add_argument("--json", action="store_true")

    download = sub.add_parser("download", help="download the memory-safe target and MTP head")
    download.add_argument("--model", default=DEFAULT_MODEL)
    download.add_argument("--mtp-model", default=DEFAULT_MTP_MODEL)
    download.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=False)

    ask = sub.add_parser("ask", help="ask a question with deep reasoning")
    _generation_arguments(ask)
    ask.add_argument("prompt", nargs="*", help="question; omit to read from stdin")

    generate = sub.add_parser("generate", help="one-shot generation using the Qwen chat template")
    _generation_arguments(generate)
    generate.add_argument("prompt", nargs="?", help="user prompt; omit to read stdin")

    chat = sub.add_parser("chat", help="interactive multi-turn chat")
    _generation_arguments(chat)
    chat.add_argument("prompt", nargs="*", help="optional first message")

    serve = sub.add_parser("serve", help="OpenAI-compatible MLX server")
    _runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--max-tokens", type=int, default=3072)
    serve.add_argument("--max-sequences", type=int, default=1)
    serve.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    serve.add_argument("--api-key")
    serve.add_argument(
        "--kv-bits",
        type=float,
        default=8,
        help="KV cache quantization bits; 0 disables and caps the context at 2048",
    )

    bench = sub.add_parser(
        "benchmark", help="benchmark serial and MTP decoding on complex prompts"
    )
    bench.add_argument("--model", default=DEFAULT_MODEL)
    bench.add_argument("--mtp-model", default=DEFAULT_MTP_MODEL)
    bench.add_argument("--mode", choices=("serial", "mtp", "both"), default="both")
    bench.add_argument("--max-tokens", type=int, default=48)
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--ctx-size", type=int, default=1024)
    bench.add_argument("--prefill-step-size", type=int, default=256)
    bench.add_argument("--mtp-block-size", type=int, default=3)
    bench.add_argument("--output")
    bench.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_argv(raw_argv))
    if args.command == "doctor":
        return print_report(as_json=args.json)
    if args.command == "download":
        download_models(model=args.model, mtp_model=args.mtp_model, include_mtp=args.mtp)
        return 0
    if args.command in {"ask", "generate", "chat"}:
        settings = _generation_profile(args)
        if args.command in {"ask", "chat"}:
            if args.mtp is True:
                raise ValueError(
                    "ask/chat are serial-only on this 16 GB runtime; use --no-mtp"
                )
            args.mtp = False
        if args.command in {"ask", "chat"}:
            prompt = " ".join(args.prompt).strip() or None
        else:
            prompt = args.prompt
        if args.command == "generate" and prompt is None:
            prompt = sys.stdin.read()
        sampling = SamplingConfig(
            settings.temperature,
            settings.top_p,
            settings.top_k,
            settings.min_p,
            settings.presence_penalty,
            settings.repetition_penalty,
        )
        if args.command in {"ask", "chat"}:
            command = build_session_command(
                _config(args, settings),
                mode=args.command,
                prompt=prompt,
                system=args.system,
                max_tokens=settings.max_tokens,
                sampling=sampling,
                thinking=settings.thinking,
                reasoning_effort=settings.reasoning_effort,
                thinking_budget=settings.thinking_budget,
                answer_reserve=settings.answer_reserve,
                show_thinking=args.show_thinking,
                preserve_thinking=args.preserve_thinking,
                show_stats=args.stats,
                quiet=args.quiet,
            )
            return _exec(command)
        command = build_generate_command(
            _config(args, settings),
            prompt=prompt,
            system=args.system,
            max_tokens=settings.max_tokens,
            sampling=sampling,
            thinking=settings.thinking,
            reasoning_effort=settings.reasoning_effort,
            thinking_budget=settings.thinking_budget,
            verbose=not args.quiet,
        )
        return _exec(command)
    if args.command == "serve":
        command = build_server_command(
            _config(args),
            host=args.host,
            port=args.port,
            max_tokens=args.max_tokens,
            max_sequences=args.max_sequences,
            thinking=args.thinking,
            api_key=args.api_key,
        )
        return _exec(command)
    if args.command == "benchmark":
        from .benchmark import run_cli_benchmark

        return run_cli_benchmark(args)
    raise AssertionError(f"unhandled command: {args.command}")


def console_main() -> None:
    """Make the installed console script use the same safe module exit path."""

    os.execv(
        sys.executable,
        [sys.executable, "-m", "dwarfstar_qwen.cli", *sys.argv[1:]],
    )


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
    # The benchmark loads the 27B model in this process.  Use the same safe
    # post-flush exit as the generation worker so mlx's Metal objects are not
    # finalized piecemeal while the working set is at its 16 GB-machine limit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
