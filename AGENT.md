# Agent Notes

DwarfStar Qwen runs **Qwen 3.8 27B on a 16 GB Apple Silicon Mac** through a
small Python wrapper around MLX and MLX-VLM with a memory profile engineered
so the 27B checkpoint stays resident in the Metal working set of an M4 16 GB
machine. The project was originally derived from
[`antirez/ds4`](https://github.com/antirez/ds4) (a native C/Metal inference
engine for DeepSeek V4 Flash) but the legacy C/Metal, CUDA and ROCm engine
has been removed from this repository. The MLX wrapper is the only runtime.

## Target hardware and baseline

- Validation machine: Apple M4, 10-core GPU, 16 GiB unified memory, macOS
  26.5.2. Metal reports a recommended working set of about 12.71 GB.
- Weights: 11.771 GB (target) plus 0.186 GB (optional MTP head). That leaves
  roughly 400 MB of headroom at the measured serial peak of 12.29 GB.
- Measured on that machine: serial decode 7.35 tok/s median (6.47–7.36 tok/s
  on complex prompts), prefill 7.3–42.5 tok/s, server about 5.1 tok/s, MTP
  OOM. The full audit is in [AUDIT_QWEN38.md](AUDIT_QWEN38.md).
- Long-context measurements (2026-08-18, same machine): a 3.3K-token prompt
  with bf16 KV OOMs in prefill at both 256- and 128-token chunks; 8-bit KV
  completes 4096 at chunk 128 (peak 12.49 GB, 7.58 tok/s); 4-bit KV completes
  8192 with a 7,038-token prompt at chunk 64 (peak 12.40 GB, 7.43 tok/s);
  8-bit KV at 8192 OOMs mid-prefill. Only 16 of 64 layers carry a KV cache
  (4 KV heads × 256 dims ≈ 64 KiB/token bf16); the binding constraint is the
  transient prefill peak, not KV storage.
- The [MLXFast leaderboard](https://www.yukon.org/mlxfast) and the
  [Layr-Labs qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)
  are study material for scheduling and fusion ideas, not targets: their
  numbers come from M5 Max 128 GB machines running specialized 4-bit kernels.
  Chasing them on this Mac produces OOMs, not speedups.

## Goals

- Keep the whole-model MLX/MLX-VLM path as the only recommended Qwen backend,
  fully resident on the 16 GB profile.
- Treat the memory profile as the product: context caps, a single server
  sequence, serial decode by default, small MLX caches and disabled GDN fusion
  are reliability requirements on 16 GB, not conservative settings to relax.
- Enable MTP speculative decoding only when the local benchmark profile proves
  greedy parity, at least 3% speedup and memory safety on this exact machine
  and checkpoint revisions.
- Keep checkpoint and dependency revisions pinned; a Hugging Face `main`
  branch must never move silently under the runtime. Model downloads resolve
  to immutable snapshot directories with weight sizes verified.
- Keep the model text-only: the vision tower was removed from the checkpoint
  to make it fit; image, audio and video input are unsupported.
- Preserve correctness before speed. Logits, KV and tokenizer behavior must
  not drift without a documented reason and a measured check.

## Hard memory invariants (16 GB profile)

These are enforced in code and documented in the audit. Changing any of them
without a validation run on the M4 16 GB machine is a release blocker:

1. **One model process at a time.** `runtime_lock.py` takes an exclusive fcntl
   lock; chat, server and benchmark refuse to start while another 12+ GB model
   process runs.
2. **MLX allocator guidance** (`apply_low_memory_profile` in `runner.py`): the
   MLX memory limit is set to Metal's `max_recommended_working_set_size` and
   the free-cache limit to 64 MB. Overridable with `DWARFSTAR_MLX_MEMORY_LIMIT_MB`
   and `DWARFSTAR_MLX_CACHE_LIMIT_MB`.
3. **GDN fusion disabled.** MLX-VLM's Qwen decode optimization concatenates
   four already-quantized GDN projections and permanently caches a copy of
   about 1.8 GB. The 16 GB patch routes `_decode_quantized_linears_fused` to
   return `None`, selecting the existing numerically identical unfused path.
   This monkey-patch is version-sensitive: keep the `hasattr` guard and fail
   loudly if mlx-vlm drops the symbol.
4. **KV-aware context caps.** The caps follow the measured OOM boundary and
   depend on KV cache quantization (`safe_max_context_size` in `config.py`):
   2048 with bf16 KV, 4096 with ≥ 6-bit quantized KV (validated at 8), 8192
   with < 6-bit (validated at 4), 1024 with MTP. Quantized KV must start at
   token 0 (`--quantized-kv-start 0`; upstream defaults to token 5000, a
   no-op under these caps). The prefill chunk is clamped per context size
   (256 up to 2K, 128 up to 4K, 64 above) because the prefill transient, not
   KV storage, is what OOMs. `DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1` is diagnostic
   only: it disables both the caps and the clamp and can swap, kill the
   process or destabilize the system. The theoretical 262K context of the
   model is not reachable on 16 GB.
5. **One server sequence.** More than one `--max-sequences` is rejected unless
   `DWARFSTAR_ALLOW_UNSAFE_SEQUENCES=1` is set; each session adds recurrent
   state and KV cache.
6. **Safe worker teardown.** Generation workers and the benchmark call
   `os._exit()` after flushing stdout/stderr: with a 12+ GB working set,
   Python's piecemeal Metal-object finalization crashed with SIGSEGV (139)
   after successful generation. Use the same pattern for any new process that
   loads the model.
7. **Fused GDN stays off on 16 GB** (`DWARFSTAR_ALLOW_FUSED_GDN=1` would add
   the ~1.8 GB copy back).

## MTP policy

- MTP is MLX-VLM speculative decoding with the 3-bit MTP head
  (`draft-kind mtp`, block size 2–8; 1 is not a valid MLX-VLM draft round).
- `--mtp-auto` (the default) consults `.dwarfstar/profile.json`.
  `benchmark --mode both` writes the recommendation only when all of these
  hold: greedy text identical to serial, speedup ≥ 1.03, peak MLX memory at or
  below 98% of the recommended working set and no backend error. The profile
  is bound to the exact target/MTP revisions and a runtime fingerprint
  (chip, device, mlx/mlx-vlm/transformers versions, context, prefill step and
  block size), so it silently expires on any change.
- Empirically, MTP OOMs during warm-up on the M4 16 GB even at the minimum
  block size 2 (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). The failure is
  recorded in the benchmark JSON as a backend error (exit 4 for an MTP-only
  run) instead of crashing; the default stays serial.
- `ask` and `chat` are serial-only by design: upstream MLX-VLM 0.6.14 chat
  drops draft-model arguments and cannot round-trip Qwen reasoning content
  correctly, so MTP and multi-turn thinking are rejected there, not patched
  around.

## Quality rules

Keep the spirit of the original engine: small, sharp, readable implementation
with no slop, elegant minimal designs, instructive comments beside the code.

- Prefer wrapping and pinning behavior at the wrapper boundary over
  reimplementing model math: MLX-VLM 0.6.14 is the provider of load,
  generation, streaming, server and MTP code, fixed by
  `requirements-macos.txt`.
- Every monkey-patch in `runner.py` must be guarded and idempotent: check for
  a saved original, stash it once, decorate through it, and fail with a clear
  error if the pinned mlx-vlm shape is missing.
- Keep the public CLI surface narrow. CLI/server code must not know tensor
  internals; the runner translates user options into `mlx-vlm` arguments.
- Prefer the tokenizer's own chat template over hand-built ChatML; the Qwen
  whitespace look-ahead and `reasoning_content` handling are covered by golden
  tests and must not regress.
- Keep memory-related decisions explicit in code and comments: why a context
  cap, cache limit or unfused path exists, and what it costs. Do not add flags
  that silently relax safety.
- Do not add "improvements" that contradict AUDIT_QWEN38.md without a new
  measured validation on the 16 GB machine.

## Safety

- Never run two model processes; the fcntl lock is intentional. Close heavy
  applications before long sessions: RAM and bandwidth are unified, so a thin
  ~400 MB margin shrinks with any background load.
- Never use the unsafe context/sequence overrides in normal use.

## Layout

Primary Qwen path (Python, Apple Silicon only):

- `dwarfstar_qwen/config.py` — constants: pinned model IDs/revisions, weight
  sizes, context caps and safety limits; `RuntimeConfig` validation.
- `dwarfstar_qwen/cli.py` — argument parsing, profile resolution, safe command
  construction, smart argv normalization (no command means chat, plain text
  means ask), `os._exit` teardown after the final flush.
- `dwarfstar_qwen/backend.py` — command builders for generate/session/server
  and `SamplingConfig`; rejects unsafe configurations before a worker starts.
- `dwarfstar_qwen/runner.py` — worker entry point: applies the 16 GB memory
  profile, acquires the runtime lock, pins default models to snapshot paths,
  applies the guarded mlx-vlm patches (GDN fusion, reasoning effort, chat
  limits, server aliases) and exits safely.
- `dwarfstar_qwen/session.py` — `ask`/`chat` sessions with exact multi-turn KV
  reuse through `PromptCacheState`, thinking/answer stream splitting, and a
  per-turn budget (thinking budget plus answer reserve) computed against the
  KV cap before decode.
- `dwarfstar_qwen/presets.py` — `deep`/`balanced`/`quick` generation profiles:
  sampling parameters, thinking mode, reasoning effort, token budgets and the
  validated context/KV-quantization pairs (deep 8192/kv4, balanced 4096/kv8,
  quick 1024/bf16).
- `dwarfstar_qwen/benchmark.py` — serial vs MTP A/B on three complex prompts
  with greedy parity, memory and speed gates; writes `.dwarfstar/profile.json`
  when the run is valid and uses the default checkpoints.
- `dwarfstar_qwen/doctor.py` — environment health: Apple Silicon, unified
  memory, mlx/mlx-vlm versions, MLX Metal, working set, checkpoint presence
  and weight sizes.
- `dwarfstar_qwen/download.py` — pinned downloads with weight-size verification
  and Xet disabled for resumable anonymous downloads.
- `dwarfstar_qwen/model_store.py` — resolves the default repos to immutable
  snapshot directories; custom paths and repos pass through untouched.
- `dwarfstar_qwen/profile.py` — benchmark recommendation storage, runtime
  fingerprint and revision binding for `recommended_mtp()`.
- `dwarfstar_qwen/runtime_lock.py` — single-process fcntl lock.

Supporting files:

- `dwarfstar` — shell shim into `.venv/bin/python -m dwarfstar_qwen.cli`.
- `scripts/setup-qwen-macos.sh` — creates `.venv` and installs the pinned
  dependencies; `scripts/install-qwen-command.sh` — symlinks `dwarfstar` into
  `~/.local/bin`.
- `scripts/baseline-llamacpp.sh`, `scripts/inspect-qwen-gguf.py` — optional
  audit probes against a llama.cpp baseline (unsloth Q3_K_M GGUF); they are not
  part of the supported runtime.
- `tests/test_qwen_cli.py` — unit tests without model inference;
  `tests/test_qwen_tokenizer.py` — tokenizer/template goldens (skipped when
  the pinned checkpoint is not cached).
- `Makefile` — the `qwen-*` targets are the only targets (`qwen-help` is the
  default goal).

Docs: `README.md` is the user-facing entry, `AUDIT_QWEN38.md` is the
authoritative Qwen audit with measured numbers, `THIRD_PARTY_NOTICES.md`
lists licensing.

## Build and commands

```sh
make qwen-setup          # create .venv and install pinned MLX dependencies
make qwen-download       # download the pinned 3-bit target (+ optional MTP head)
make qwen-doctor         # check hardware, MLX and the model cache
make qwen-bench          # safe serial benchmark (default); use --mode both for MTP A/B
make qwen-chat           # interactive serial chat
make qwen-server         # OpenAI-compatible server, one sequence, 127.0.0.1:8080
make qwen-test           # Python unit tests without loading the model
```

Extra options pass through `QWEN_*_ARGS`, for example
`make qwen-bench QWEN_BENCH_ARGS='--mode both --max-tokens 64 --repeats 2'`.

## Testing

- Fast gate: `make qwen-test`. The unit tests cover config safety caps,
  command building, argv normalization, presets, session budget and
  thinking-split behavior, profile revision binding, the low-memory patch,
  server model allowlist and the runtime lock.
- Real validation on the 16 GB Mac: `make qwen-doctor`, then `make qwen-bench`.
  Also exercise code, math, Unicode, tool calls, long
  prompts, OpenAI-compatible server requests (`/v1/models`,
  `/v1/chat/completions`, alias allowlist, `developer` normalization,
  `max_tokens: 0` rejection) and the second-process lock rejection.
- The benchmark exit codes are contractual: 0 success with parity, 3 when
  serial/MTP texts differ, 4 when an MTP-only run hit a backend error.
- Anything touching the memory profile, the GDN patch, context caps or worker
  teardown needs an audit-level re-validation on the M4 16 GB machine (peaks,
  serial decode, MTP attempt) before it is considered done.

## Pinned revisions and versions

| Component | Identifier | Revision / version | Weights |
| --- | --- | --- | ---: |
| Target | `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | `c98bba5926f51fec1c8d8737e577221673f524d7` | 11.771 GB |
| MTP head | `lukaskremla/Qwen3.8-27B-MTP-3bit-MLX` | `9d061a0661258e75b401a11ac9fa22fc648e039d` | 0.186 GB |
| MLX | Apple MLX | `0.32.1` | — |
| MLX-VLM | Blaizzy | `0.6.14` | — |
| Transformers / Tokenizers | Hugging Face | `5.15.0` / `0.22.2` | — |
| Hugging Face Hub | Hugging Face | `1.27.0` | — |
| Jinja2 | — | `3.1.6` | — |

`DWARFSTAR_*` environment variables: `DWARFSTAR_MODEL`, `DWARFSTAR_MTP_MODEL`,
`DWARFSTAR_MLX_CACHE_LIMIT_MB` (default 64), `DWARFSTAR_MLX_MEMORY_LIMIT_MB`
(default: Metal recommended working set), `DWARFSTAR_PROFILE`,
`DWARFSTAR_RUNTIME_LOCK`, `DWARFSTAR_ALLOW_UNSAFE_CONTEXT`,
`DWARFSTAR_ALLOW_UNSAFE_SEQUENCES`, `DWARFSTAR_ALLOW_FUSED_GDN`. The
unsafe/fused variables are diagnostics and must not be enabled on 16 GB.
