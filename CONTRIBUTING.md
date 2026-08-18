# Contributing

DwarfStar Qwen changes should be tested against the failure mode they can
realistically affect. Please include the commands you ran, the machine, the
model quant, and any notable failures in the PR or commit notes.

## Quick gate

```sh
make qwen-test
```

The unit tests cover config safety caps, command building, argv normalization,
presets, session budget and thinking-split behavior, profile revision binding,
the low-memory patch, server model allowlist and the runtime lock.

## Real validation on a 16 GB Mac

```sh
make qwen-doctor
make qwen-bench
```

Also exercise Italian outputs, code, math, Unicode, tool calls, long prompts,
OpenAI-compatible server requests (`/v1/models`, `/v1/chat/completions`, alias
allowlist, `developer` normalization, `max_tokens: 0` rejection) and the
second-process lock rejection.

The benchmark exit codes are contractual: 0 success with parity, 3 when
serial/MTP texts differ, 4 when an MTP-only run hit a backend error.

## Memory profile changes

Anything touching the memory profile, the GDN patch, context caps, KV
quantization or worker teardown needs an audit-level re-validation on the
M4 16 GB machine (peaks, serial decode, MTP attempt) before it is considered
done. See [AUDIT_QWEN38.md](AUDIT_QWEN38.md) for the measured baselines.

## Quality rules

Keep the spirit of the original engine: small, sharp, readable implementation
with no slop, elegant minimal designs, instructive comments beside the code.

- Prefer wrapping and pinning behavior at the wrapper boundary over
  reimplementing model math: MLX-VLM 0.6.14 is the provider of load,
  generation, streaming, server and MTP code, fixed by
  `requirements-macos.txt`.
- Every monkey-patch in `runner.py` must be guarded and idempotent.
- Keep the public CLI surface narrow. CLI/server code must not know tensor
  internals; the runner translates user options into `mlx-vlm` arguments.
- Keep memory-related decisions explicit in code and comments: why a context
  cap, cache limit or unfused path exists, and what it costs.
- Do not add "improvements" that contradict AUDIT_QWEN38.md without a new
  measured validation on the 16 GB machine.
