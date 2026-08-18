# DwarfStar Qwen

Local inference of **Qwen 3.8 27B** optimized for Apple Silicon Macs with
**16 GB of unified memory**. The runtime uses MLX and MLX-VLM with a text-only
3-bit checkpoint. The project is historically derived from
[`antirez/ds4`](https://github.com/antirez/ds4), but the original C/Metal
engine is no longer included in this repository.

> Status: beta. The model fits in 16 GB only with tight margins. The
> low-memory profile, context caps and single session are reliability
> requirements, not conservative settings.

[Audit and M4 16 GB benchmarks](AUDIT_QWEN38.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Quick start

Requirements: Apple Silicon Mac, recent macOS, Python 3.10 or newer, about
13 GB of free disk space for the weights plus room for the Python cache.

```sh
make qwen-setup
make qwen-download
make qwen-doctor
make qwen-bench
make qwen-chat
```

`qwen-setup` creates `.venv` and installs pinned versions of the direct
dependencies. `qwen-download` downloads the target and the MTP head at the
immutable revisions listed below. Files are saved in the standard Hugging
Face cache and interrupted downloads can be resumed.

To install only the serial target and save about 213 MB of cache (186 MB of
which is weights):

```sh
make qwen-download QWEN_DOWNLOAD_ARGS=--no-mtp
```

List all Qwen commands:

```sh
make qwen-help
```

## Pinned models and versions

| Component | Identifier | Revision/version | Weights |
| --- | --- | --- | ---: |
| Target | `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | `c98bba5926f51fec1c8d8737e577221673f524d7` | 11.771 GB |
| MTP head | `lukaskremla/Qwen3.8-27B-MTP-3bit-MLX` | `9d061a0661258e75b401a11ac9fa22fc648e039d` | 0.186 GB |
| MLX | Apple MLX | `0.32.1` | — |
| MLX-VLM | Blaizzy MLX-VLM | `0.6.14` | — |
| Hugging Face Hub | `huggingface-hub` | `1.27.0` | — |
| Transformers / Tokenizers | Hugging Face | `5.15.0` / `0.22.2` | — |

Default revisions resolve to immutable snapshot directories: a future
`main` branch update on Hugging Face will not silently change the installed
runtime. Local models or repositories chosen with `--model` and
`--mtp-model` remain the user's responsibility.

The target is **text-only**: the vision tower was removed to make it fit on
the 16 GB Mac. Image, audio and video inputs are not supported.

## CLI

The main commands are serial sessions with reasoning and exact KV cache reuse
between turns. Running with no arguments starts the chat; plain text is a
one-shot question:

```sh
./dwarfstar "Why is the sky blue?"                # one-shot question (ask)
./dwarfstar chat --profile balanced               # intermediate profile
./dwarfstar ask --show-thinking "2^10 - 24?"      # also show the thought
```

Profiles bundle the configurations validated on the M4 16 GB Mac:

| Profile | Context | KV cache | Thinking | Thought budget |
| --- | ---: | --- | --- | ---: |
| `deep` (default) | 8,192 | 4-bit quantized | xhigh | 3,072 |
| `balanced` | 4,096 | 8-bit quantized | medium | 1,536 |
| `quick` | 1,024 | bf16 | disabled | — |

KV cache quantization covers only the 16 full-attention layers out of 64 (the
rest use recurrent GatedDeltaNet state): that is what makes long context
possible, because bf16 KV runs out of memory in prefill on a 3,300-token
prompt. Every value is overridable (`--ctx-size`, `--kv-bits`,
`--thinking-budget`, `--reasoning-effort`, `--answer-reserve`, sampling).
The answer reserve guarantees the thought cannot consume the whole decode
budget before the visible response.

In chat the commands `/clear`, `/stats`, `/effort`, `/thinking`, `/help` and
`/exit` are available. The wrapper counts the transcript tokens at every turn
and rejects the turn before it exceeds the KV limit.

One-shot generation through the upstream CLI (diagnostic overrides):

```sh
./dwarfstar generate --no-mtp \
  --system "Answer precisely and concisely." \
  "Explain the difference between concurrency and parallelism."
```

`--mtp` is a diagnostic override. On the M4 16 GB machine used for this
validation it ran out of memory even after closing the heaviest applications;
use the default `--mtp-auto`, which selects serial decoding on this
installation.

The MTP modes are:

- `--mtp-auto`: use the local recommendation saved by the benchmark; this is
  the default behavior and stays serial if no valid profile exists;
- `--no-mtp`: serial decode, the most reliable path;
- `--mtp`: force speculative decoding;
- `--mtp-block-size N`: verified total block, from 2 to 8; MLX-VLM proposes
  `N-1` draft tokens per round. The value 1 is not a valid MTP generation in
  the pinned runtime and is rejected.

Make targets accept extra options without editing the Makefile:

```sh
make qwen-chat QWEN_CHAT_ARGS='--profile balanced'
make qwen-bench QWEN_BENCH_ARGS='--mode both --max-tokens 64 --repeats 2'
```

## OpenAI-compatible server

Start a local server with a single active sequence:

```sh
make qwen-server
```

Direct equivalent:

```sh
./dwarfstar serve \
  --host 127.0.0.1 \
  --port 8080 \
  --max-sequences 1 \
  --mtp-auto
```

Example request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dwarfstar-qwen",
    "messages": [{"role": "user", "content": "Write a quicksort function in Python."}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Do not expose the server on an untrusted network without `--api-key`. On a
16 GB machine do not raise `--max-sequences`: every session adds recurrent
state and KV cache. The wrapper only accepts the pinned Qwen checkpoint and
its aliases, normalizes the OpenAI `developer` role into `system` and rejects
request-chosen remote model loads.

## Benchmark and experimental MTP

```sh
make qwen-bench
```

The default benchmark uses complex prompts and measures safe serial decode.
With `--mode both` it also compares MTP on the same machine and records
throughput, acceptance, memory and textual parity under greedy decoding;
when the result is valid it saves the local recommendation in
`.dwarfstar/profile.json`. The profile is bound to the exact revisions of the
two checkpoints, chip/Metal architecture, working set, runtime versions,
context, prefill and block size; it is not reused with a different
configuration.

MTP is experimental because the gain depends on the prompt, the accepted
draft ratio and memory pressure. The target always verifies the proposed
tokens, but draft and rollback work can make MTP slower than serial. Do not
carry the MLXFast leaderboard numbers over to this Mac: they were measured on
faster M5 hardware with a specialized runtime.

Measured on August 18, 2026 on the target M4 16 GB Mac:

| Test | Result |
| --- | ---: |
| Serial decode, median of 3 complex prompts × 256 tokens | **7.35 tok/s** |
| Decode range | 6.47–7.36 tok/s |
| Prefill of the two prompts after the first measured | 39.6–42.5 tok/s |
| MLX peak | 12.29 GB |
| 3,398-token prompt, ctx 4K, KV 8-bit, chunk 128 | 7.58 tok/s, peak 12.49 GB |
| 7,038-token prompt, ctx 8K, KV 4-bit, chunk 64 | 7.43 tok/s, peak 12.40 GB |
| 3,346-token prompt with bf16 KV (chunks 256 and 128) | OOM in prefill |
| MTP 3-bit | OOM in warm-up, serial default |

The previous C/SSD path measured about 0.04 tok/s. Raw outputs, qualitative
evaluation and the issues found are documented in
[AUDIT_QWEN38.md](AUDIT_QWEN38.md).

For a serial-only measurement:

```sh
./dwarfstar benchmark --mode serial
```

To force a longer comparison:

```sh
./dwarfstar benchmark --mode both --max-tokens 96 --repeats 2
```

## Memory limits on 16 GB

On the M4 Mac used to develop this port, Metal reports a recommended working
set of about **12.71 GB**. The target occupies 11.77 GB; with MTP the weights
alone reach about 11.96 GB. GatedDeltaNet state, KV cache, activations,
graphs and temporary buffers still need to be allocated.

Only 16 of the 64 layers keep a KV cache (about 64 KiB/token in bf16); the
others use constant-size recurrent GatedDeltaNet state. The long-context
bottleneck is not KV storage but the transient prefill peak: the measured
mitigation is quantizing the KV cache from token 0 and shrinking the prefill
chunk.

The defaults are therefore:

- `deep` profile: context 8,192, KV 4-bit, prefill chunk 64;
- `balanced` profile: context 4,096, KV 8-bit, prefill chunk 128;
- `quick` profile and benchmark: context 1,024, bf16 KV, prefill chunk 256;
- one server sequence;
- serial decode until the local benchmark recommends MTP;
- MLX free cache limited to 64 MB;
- GDN fusion, which would duplicate about 1.8 GB of weights, disabled.

Context caps depend on KV quantization and were measured on the target Mac:

| KV cache | Context cap |
| --- | ---: |
| bf16 (`--kv-bits 0`) | 2,048 |
| quantized ≥ 6 bits | 4,096 |
| quantized < 6 bits | 8,192 |
| with MTP | 1,024 |

The prefill chunk is reduced automatically (256 up to 2K, 128 up to 4K, 64
above). The `DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1` override is diagnostic: it can
swap, kill the process or destabilize the system. The model's theoretical
262K context is not reachable on a 16 GB Mac.

To reduce the OOM risk:

- close heavy applications before loading the model;
- never run two model processes at the same time;
- on OOM fall back to `--profile balanced` or `--profile quick`;
- run `make qwen-doctor` after setup and download;
- do not enable `DWARFSTAR_ALLOW_FUSED_GDN=1` on 16 GB.

Advanced variables:

| Variable | Usage |
| --- | --- |
| `DWARFSTAR_MODEL` | alternative local target or repository |
| `DWARFSTAR_MTP_MODEL` | alternative MTP head |
| `DWARFSTAR_MLX_CACHE_LIMIT_MB` | MLX free cache limit; default 64 |
| `DWARFSTAR_MLX_MEMORY_LIMIT_MB` | MLX allocator guidance; default: recommended Metal working set |
| `DWARFSTAR_PROFILE` | alternative benchmark profile path |
| `DWARFSTAR_RUNTIME_LOCK` | alternative inter-process lock path |
| `DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1` | ignore context caps, not recommended |
| `DWARFSTAR_ALLOW_UNSAFE_SEQUENCES=1` | allow multiple server sequences, diagnostic |
| `DWARFSTAR_ALLOW_FUSED_GDN=1` | re-enable high-memory GDN fusion, do not use on 16 GB |

## Tests

Fast tests do not load the model:

```sh
make qwen-test
```

Before considering a real configuration valid:

```sh
make qwen-doctor
make qwen-bench
```

Also evaluate code, math, Unicode, tool calls and long prompts. 3-bit is the
largest uniform affine checkpoint validated here that fits the profile, but
it can lose quality compared to wider quantizations.

## Credits and licenses

The project derives from the [antirez/ds4](https://github.com/antirez/ds4)
engine and uses MLX and MLX-VLM. The benchmark criteria are inspired by the
[Qwen 3.8 MLXFast challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge);
speculative decoding and rollback are provided by MLX-VLM.
Qwen and the quantized checkpoints keep the license and conditions of the
upstream model. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE](LICENSE).
