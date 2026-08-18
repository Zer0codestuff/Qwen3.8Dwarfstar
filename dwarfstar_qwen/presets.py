from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GenerationProfile:
    """A quality/latency policy, independent from the inference backend."""

    name: str
    context_size: int
    kv_bits: float
    max_tokens: int
    thinking: str
    reasoning_effort: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    thinking_budget: int | None
    answer_reserve: int


# ``deep`` follows Qwen 3.8's published thinking sampler.  The context and KV
# settings are the configurations measured on the M4 16 GB machine: only 16 of
# 64 layers keep a KV cache, so quantizing it is what makes room for long
# reasoning.  bf16 KV OOMs during a long prefill at 4096; 8-bit KV holds 4096
# and 4-bit KV holds 8192 (see AUDIT_QWEN38.md).  The answer reserve makes
# sure a long chain of thought cannot consume the complete decode budget
# before the model gets to its user-visible response.
PROFILES: dict[str, GenerationProfile] = {
    "deep": GenerationProfile(
        name="deep",
        context_size=8192,
        kv_bits=4.0,
        max_tokens=4096,
        thinking="enabled",
        reasoning_effort="xhigh",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_budget=3072,
        answer_reserve=1024,
    ),
    "balanced": GenerationProfile(
        name="balanced",
        context_size=4096,
        kv_bits=8.0,
        max_tokens=2048,
        thinking="enabled",
        reasoning_effort="medium",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_budget=1536,
        answer_reserve=512,
    ),
    "quick": GenerationProfile(
        name="quick",
        context_size=1024,
        kv_bits=0.0,
        max_tokens=512,
        thinking="disabled",
        reasoning_effort="low",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        thinking_budget=None,
        answer_reserve=0,
    ),
}


def profile(name: str) -> GenerationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {name!r}; choose one of: {choices}") from exc


def resolve_profile(
    name: str,
    *,
    context_size: int | None = None,
    kv_bits: float | None = None,
    max_tokens: int | None = None,
    thinking: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    thinking_budget: int | None = None,
    answer_reserve: int | None = None,
) -> GenerationProfile:
    """Return a preset with only the explicitly supplied values replaced."""

    selected = profile(name)
    overrides = {
        "context_size": context_size,
        "kv_bits": kv_bits,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "answer_reserve": answer_reserve,
    }
    values = {key: value for key, value in overrides.items() if value is not None}
    # ``None`` is meaningful for a no-limit thinking budget, so callers use a
    # separate sentinel at the CLI boundary and pass only concrete integers.
    if thinking_budget is not None:
        values["thinking_budget"] = thinking_budget
    resolved = replace(selected, **values)
    _validate(resolved)
    return resolved


def _validate(value: GenerationProfile) -> None:
    if value.context_size < 64:
        raise ValueError("context size must be at least 64 tokens")
    if value.kv_bits and not 2 <= value.kv_bits <= 8:
        raise ValueError("kv-bits must be zero (bf16 KV) or between 2 and 8")
    if value.max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if value.thinking not in {"enabled", "disabled"}:
        raise ValueError("invalid thinking mode")
    if value.reasoning_effort not in {"xhigh", "medium", "low"}:
        raise ValueError("invalid reasoning effort")
    if value.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 <= value.top_p <= 1 or not 0 <= value.min_p <= 1:
        raise ValueError("top-p and min-p must be between 0 and 1")
    if value.top_k < 0:
        raise ValueError("top-k cannot be negative")
    if value.thinking_budget is not None and value.thinking_budget < 0:
        raise ValueError("thinking budget cannot be negative")
    if value.answer_reserve < 0:
        raise ValueError("answer reserve cannot be negative")
    if value.answer_reserve >= value.max_tokens:
        raise ValueError("answer reserve must be smaller than max tokens")
