"""Memory-bounded Qwen 3.8 text sessions with exact multi-turn KV reuse."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable


THINKING_END = "</think>"
CONTEXT_GUARD_TOKENS = 8
MIN_GENERATION_TOKENS = 32


@dataclass(frozen=True)
class TurnBudget:
    max_tokens: int
    thinking_budget: int | None
    prompt_tokens: int


@dataclass(frozen=True)
class TurnResult:
    answer: str
    reasoning: str
    prompt_tokens: int
    generation_tokens: int
    cached_tokens: int
    generation_tps: float
    peak_memory_gb: float
    finish_reason: str | None


class ThinkingStream:
    """Split a streamed ``</think>`` marker even when it spans chunks."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.in_reasoning = enabled
        self.pending = ""
        self.reasoning: list[str] = []
        self.answer: list[str] = []

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self.in_reasoning:
            self.answer.append(text)
            return [("answer", text)]

        self.pending += text
        marker_at = self.pending.find(THINKING_END)
        if marker_at >= 0:
            thought = self.pending[:marker_at]
            answer = self.pending[marker_at + len(THINKING_END) :]
            self.pending = ""
            self.in_reasoning = False
            events: list[tuple[str, str]] = []
            if thought:
                self.reasoning.append(thought)
                events.append(("reasoning", thought))
            if answer:
                self.answer.append(answer)
                events.append(("answer", answer))
            return events

        # Retain only the suffix which could still be the start of the marker.
        safe = max(0, len(self.pending) - len(THINKING_END) + 1)
        if not safe:
            return []
        thought = self.pending[:safe]
        self.pending = self.pending[safe:]
        self.reasoning.append(thought)
        return [("reasoning", thought)]

    def finish(self) -> tuple[str, str, bool]:
        closed = not self.in_reasoning
        if self.pending:
            target = self.reasoning if self.in_reasoning else self.answer
            target.append(self.pending)
            self.pending = ""
        reasoning = "".join(self.reasoning).strip()
        answer = "".join(self.answer).strip()
        # A truncated thought is still more useful than returning an empty
        # response.  It is treated as the answer, but not preserved as hidden
        # reasoning, so the next chat template remains structurally valid.
        if self.enabled and not closed and not answer:
            answer, reasoning = reasoning, ""
        return reasoning, answer, closed


def allocate_turn_budget(
    *,
    prompt_tokens: int,
    context_size: int,
    max_tokens: int,
    thinking_enabled: bool,
    requested_thinking_budget: int | None,
    answer_reserve: int,
) -> TurnBudget:
    available = context_size - prompt_tokens - CONTEXT_GUARD_TOKENS
    if available < MIN_GENERATION_TOKENS:
        raise ValueError(
            f"context full ({prompt_tokens}/{context_size} tokens); "
            "use /clear to start a new conversation"
        )
    generation = min(max_tokens, available)
    thinking_budget = None
    if thinking_enabled and requested_thinking_budget is not None:
        # Shrink the reasoning budget first when a long conversation leaves
        # less room.  This preserves as much of the requested final-answer
        # reserve as the remaining context permits.
        effective_reserve = min(answer_reserve, max(0, generation - 1))
        thinking_budget = min(
            requested_thinking_budget,
            max(0, generation - effective_reserve),
        )
    return TurnBudget(generation, thinking_budget, prompt_tokens)


def _parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"dwarfstar {mode}")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--system")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-kv-size", type=int, required=True)
    parser.add_argument("--prefill-step-size", type=int, required=True)
    parser.add_argument(
        "--thinking-mode", choices=("enabled", "disabled"), required=True
    )
    parser.add_argument(
        "--reasoning-effort", choices=("xhigh", "medium", "low"), required=True
    )
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--answer-reserve", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--min-p", type=float, required=True)
    parser.add_argument("--presence-penalty", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--kv-bits", type=float, default=0)
    parser.add_argument(
        "--show-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--preserve-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    return len(encoded)


def _render_prompt(
    processor: Any,
    config: Any,
    messages: list[dict[str, str]],
    *,
    thinking_enabled: bool,
    reasoning_effort: str,
    preserve_thinking: bool,
) -> str:
    del config
    tokenizer = getattr(processor, "tokenizer", processor)
    # MLX-VLM's generic text-message normalizer copies only role/content and
    # drops Qwen's ``reasoning_content`` field.  Calling the pinned tokenizer's
    # own template preserves the prior thought verbatim, which in turn makes
    # the next prompt an exact extension of PromptCacheState instead of asking
    # the non-trimmable GDN caches to roll back.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_enabled,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
    )


def _stream_turn(
    model: Any,
    processor: Any,
    prompt: str,
    prompt_cache_state: Any,
    args: argparse.Namespace,
    *,
    emit_output: bool,
) -> TurnResult:
    import mlx.core as mx
    from mlx_vlm import stream_generate

    tokenizer = getattr(processor, "tokenizer", processor)
    thinking_enabled = args.thinking_mode == "enabled"
    budget = allocate_turn_budget(
        prompt_tokens=_token_count(tokenizer, prompt),
        context_size=args.max_kv_size,
        max_tokens=args.max_tokens,
        thinking_enabled=thinking_enabled,
        requested_thinking_budget=args.thinking_budget,
        answer_reserve=args.answer_reserve,
    )
    splitter = ThinkingStream(thinking_enabled)
    answer_started = False
    if emit_output and not args.quiet:
        if thinking_enabled:
            print(
                f"Qwen · reasoning {args.reasoning_effort} in progress…",
                flush=True,
            )
            if args.show_thinking:
                print("\nThought > ", end="", flush=True)
        else:
            print("Qwen > ", end="", flush=True)
            answer_started = True

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    kwargs: dict[str, Any] = {
        "max_tokens": budget.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_kv_size": args.max_kv_size,
        "prefill_step_size": args.prefill_step_size,
        "prompt_cache_state": prompt_cache_state,
        "enable_thinking": thinking_enabled,
        "skip_special_tokens": False,
        "verbose": False,
    }
    if budget.thinking_budget is not None:
        kwargs["thinking_budget"] = budget.thinking_budget
    if args.kv_bits:
        # MLX-VLM defaults quantization to token 5000; that would make this
        # option a no-op under DwarfStar's 4096-token safety cap.
        kwargs["kv_bits"] = args.kv_bits
        kwargs["quantized_kv_start"] = 0

    last = None
    for chunk in stream_generate(model, processor, prompt, **kwargs):
        last = chunk
        for section, text in splitter.feed(chunk.text):
            if not emit_output:
                continue
            if section == "reasoning":
                if args.show_thinking:
                    print(text, end="", flush=True)
                continue
            if not answer_started:
                print("\n\nQwen > ", end="", flush=True)
                answer_started = True
            print(text, end="", flush=True)

    reasoning, answer, closed = splitter.finish()
    if emit_output:
        if not answer_started and answer:
            print("\n\nQwen > " if thinking_enabled else "Qwen > ", end="")
            print(answer, end="")
        print(flush=True)
        if thinking_enabled and not closed:
            print(
                "[warning: budget exhausted before the reasoning block was closed]",
                file=sys.stderr,
            )
    if last is None:
        return TurnResult(answer, reasoning, budget.prompt_tokens, 0, 0, 0.0, 0.0, "length")
    return TurnResult(
        answer=answer,
        reasoning=reasoning,
        prompt_tokens=last.prompt_tokens,
        generation_tokens=last.generation_tokens,
        cached_tokens=last.cached_tokens,
        generation_tps=last.generation_tps,
        peak_memory_gb=last.peak_memory,
        finish_reason=last.finish_reason,
    )


def _print_help() -> None:
    print(
        "Commands: /clear  /stats  /effort xhigh|medium|low  "
        "/thinking on|off  /help  /exit"
    )


def _print_stats(result: TurnResult | None) -> None:
    if result is None:
        print("No generation completed yet.")
        return
    print(
        f"prompt {result.prompt_tokens} · cache reused {result.cached_tokens} · "
        f"generated {result.generation_tokens} · {result.generation_tps:.2f} tok/s · "
        f"MLX peak {result.peak_memory_gb:.2f} GB · finish {result.finish_reason}"
    )


def _new_cache_state():
    from mlx_vlm.generate import PromptCacheState

    return PromptCacheState()


def _clear_cache():
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _initial_messages(system: str | None) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}] if system else []


def _run_one(
    model: Any,
    processor: Any,
    config: Any,
    messages: list[dict[str, str]],
    cache_state: Any,
    user: str,
    args: argparse.Namespace,
) -> TurnResult:
    messages.append({"role": "user", "content": user})
    try:
        prompt = _render_prompt(
            processor,
            config,
            messages,
            thinking_enabled=args.thinking_mode == "enabled",
            reasoning_effort=args.reasoning_effort,
            preserve_thinking=args.preserve_thinking,
        )
        result = _stream_turn(
            model,
            processor,
            prompt,
            cache_state,
            args,
            emit_output=True,
        )
    except BaseException:
        messages.pop()
        raise
    messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "reasoning_content": result.reasoning,
        }
    )
    return result


def main(mode: str, argv: Iterable[str]) -> int:
    if mode not in {"ask", "chat"}:
        raise ValueError("session mode must be ask or chat")
    args = _parser(mode).parse_args(list(argv))
    if args.kv_bits and args.kv_bits < 2:
        raise ValueError("kv-bits must be zero or at least 2")

    if not args.quiet:
        print("Loading Qwen 3.8 27B…", file=sys.stderr, flush=True)
    started = time.perf_counter()
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(args.model)
    config = load_config(args.model)
    if not args.quiet:
        print(
            f"Model ready in {time.perf_counter() - started:.1f}s · "
            f"context {args.max_kv_size} · serial",
            file=sys.stderr,
            flush=True,
        )

    messages = _initial_messages(args.system)
    cache_state = _new_cache_state()
    if mode == "ask":
        user = args.prompt
        if user is None:
            user = sys.stdin.read().strip()
        if not user:
            raise ValueError("write a question or pass it via stdin")
        result = _run_one(model, processor, config, messages, cache_state, user, args)
        if args.stats:
            _print_stats(result)
        return 0

    if args.prompt:
        last: TurnResult | None = _run_one(
            model, processor, config, messages, cache_state, args.prompt, args
        )
    else:
        last = None
    if not args.quiet:
        print(
            "Chat ready. Press Enter to talk, /help for commands, Ctrl-D to exit."
        )
    while True:
        try:
            user = input("\nYou > ").strip()
        except EOFError:
            print()
            return 0
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            return 0
        if user == "/help":
            _print_help()
            continue
        if user == "/clear":
            messages = _initial_messages(args.system)
            cache_state = _new_cache_state()
            last = None
            _clear_cache()
            print("Conversation and cache cleared.")
            continue
        if user == "/stats":
            _print_stats(last)
            continue
        if user.startswith("/effort "):
            effort = user.split(maxsplit=1)[1]
            if effort not in {"xhigh", "medium", "low"}:
                print("Valid values: xhigh, medium, low")
            else:
                args.reasoning_effort = effort
                print(f"Reasoning effort set to {effort}.")
            continue
        if user.startswith("/thinking "):
            value = user.split(maxsplit=1)[1]
            if value not in {"on", "off"}:
                print("Valid values: on, off")
            else:
                args.thinking_mode = "enabled" if value == "on" else "disabled"
                print(f"Thinking {value}.")
            continue
        try:
            last = _run_one(
                model, processor, config, messages, cache_state, user, args
            )
            if args.stats:
                _print_stats(last)
        except ValueError as exc:
            print(f"dwarfstar: {exc}", file=sys.stderr)
