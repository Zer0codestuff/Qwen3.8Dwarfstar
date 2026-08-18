# Audit and validation — DwarfStar Qwen 3.8 27B

Test date: August 18, 2026. Hardware: Apple M4 MacBook, 10-core GPU, 16 GiB of
unified memory, macOS 26.5.2. All measurements in this document were taken on
the target machine, not extrapolated from public benchmarks.

## Outcome

The C/Metal port found in the repository was not a usable Qwen engine on this
Mac: the CPU forward pass produced plausible text, but streaming the weights
from disk reached about 0.04 tok/s, while the 13.82 GB GGUF did not fit in
the Metal working set. The supported path was therefore rebuilt on top of
MLX-VLM with a text-only affine 3-bit 11.77 GB checkpoint and a low-memory
profile specific to 16 GB.

The result is stable in serial decoding: 7.35 tok/s median on the three
complex prompts, 12.29 GB peak, and a working OpenAI-compatible server. MTP
cannot be enabled safely with this combination of hardware, 3-bit weights and
current kernels: two trials with block size 3 and one with the minimum valid
size 2, also run after closing Zen, ended with
`kIOGPUCommandBufferCallbackErrorOutOfMemory` during warm-up. The runtime
records it and keeps serial as the default.

## Issues found in the native port

The main Qwen 3.8 math was more complete than the external behavior
suggested: GatedDeltaNet, convolution, RMSNorm Q/K, GQA, gated attention and
head mapping were substantially aligned with the expected GGUF layout. The
blocking issues were mostly integration and runtime architecture.

1. **Backend incompatible with 16 GB.** The local Q3_K_M GGUF weighs 13.819 GB
   against a recommended Metal working set of 12.713 GB. Full llama.cpp
   offload ran out of memory; the native path instead read about 10–14 GB of
   layers from disk per token.
2. **Unreliable Qwen chat template.** Some inputs were tokenized as raw text
   or went through the DeepSeek renderer; the hand-built ChatML markers did
   not reproduce the official template. This substantially changes the
   distribution seen by the model.
3. **Divergent tokenizer whitespace.** The port did not implement the Qwen
   look-ahead for repeated spaces and indentation. Golden examples:
   `alpha  beta` must become `[6918, 220, 13053]`; a Python line with four
   spaces must keep the indentation according to the official tokenizer.
4. **Recurrent state not isolated per session.** The Qwen runtime owned a
   single mutable GDN/conv/KV state; interleaved sessions, rewinds and
   restores could therefore contaminate each other.
5. **Masked context.** Requests over 4,096 tokens could be silently reduced
   to 512 instead of being rejected with an explicit error.
6. **Incomplete Metal dispatch and stops.** Qwen could enter the generic
   Metal batch without an initialized Qwen graph; both official stops,
   `248046` (`<|im_end|>`) and `248044` (`<|endoftext|>`), must be handled.
7. **Persistence claimed wider than reality.** The session snapshot did not
   correctly cover all the Qwen recurrent state.

These defects were not hidden behind small local fixes: the native C remained
explicitly an experimental legacy backend. Making it Metal-resident would
also require a smaller quantization format and dedicated Qwen kernels; pure
API refactoring would not solve the physical weight limit.

## Solution implemented

- target checkpoint
  `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` pinned to revision
  `c98bba5926f51fec1c8d8737e577221673f524d7`;
- optional MTP head pinned to revision
  `9d061a0661258e75b401a11ac9fa22fc648e039d`;
- direct versions pinned: MLX 0.32.1, MLX-VLM 0.6.14,
  Transformers 5.15.0 and Tokenizers 0.22.2;
- chat template and tokenizer from the checkpoint, without a spurious BOS and
  with the two official stop tokens;
- default context 1,024, prefill chunk 256 and a single server sequence;
- MLX free cache limited to 64 MB;
- MLX allocator guidance lowered from the upstream default, which exceeded
  the recommended working set, to the value recommended by Metal; it is not
  a hard cap;
- GDN fusion disabled, which on the 3-bit target would have kept an extra
  concatenated copy of about 1.77 GB;
- downloads resolved to immutable snapshots with total weight file size
  verification;
- A/B benchmark that enables MTP only with greedy textual parity, at least
  +3% speed and memory within the limit; Metal errors become a recorded
  result, not a crash;
- safe worker teardown after the flush. With the 27B almost at the limit,
  piecemeal finalization of Metal objects caused a segfault 139 after a
  completed generation; resources are now returned atomically by the OS;
- local OpenAI-compatible server with stable aliases `dwarfstar-qwen` and
  `qwen3.8-27b`, allowlist of the pinned target only and `developer` role
  normalized to `system`;
- process lock: chat, server and benchmark cannot load two copies of the 27B
  at the same time;
- interactive chat limited to the serial/no-thinking path with a hard
  transcript count before the KV limit. Multi-turn reasoning and MTP are
  rejected because the upstream chat branch does not preserve/forward them
  correctly.

## Measured memory

| Item | Value |
| --- | ---: |
| Physical unified memory | 17.18 GB / 16 GiB |
| Recommended Metal working set | 12.713 GB |
| 3-bit target weights | 11.771 GB |
| Additional MTP weights | 0.186 GB |
| Serial peak, complex prompts | 12.248–12.294 GB |
| Margin over the worst peak | about 419 MB |
| MTP, block sizes 2 and 3 | OOM in warm-up, even with Zen closed |

The margin is enough for a single short session, but it does not make the
model indifferent to other applications: RAM and bandwidth are unified.
Closing heavy apps reduces the pressure/swap risk; it did not make MTP
runnable.

## Performance

| Path | Load | Decode | Prefill | Peak | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Initial native C | 32 tokens, simple prompt | 0.04 tok/s | 0.04 tok/s | RSS 4.18 GB | 920.91 s |
| Initial native C | arithmetic, 16 tokens | 0.04 tok/s | 0.04 tok/s | RSS 4.04 GB | 1,208.27 s, truncated response |
| Initial native C | code, 24 tokens | 0.04 tok/s | 0.04 tok/s | RSS 3.97 GB | 1,370.21 s, truncated response |
| Final MLX serial | 3 prompts × 256 tokens | **7.35 tok/s median** | 7.3–42.5 tok/s | 12.29 GB | stable |
| Final MLX serial, runtime | 3 prompts × 32 tokens | 5.31 tok/s median | 35.2–41.5 tok/s | 12.29 GB | stable |
| Warm server | short request | 5.1 tok/s | 23.5 tok/s | 12.09 GB | HTTP 200 |
| MTP 3-bit, block sizes 2/3 | 4-token warm-up | — | — | over budget | OOM |

The final complex serial decode varies between 6.47 and 7.36 tok/s: about
**184×** the old 0.04 tok/s path. The first prefill after loading is slower
due to compilation/warm-up; the following ones are significantly faster. The
final measurement was taken after closing Zen and confirms that open apps
affect the bandwidth/pressure of unified memory.
The versionable raw data are `benchmark-results/qwen38-m4-complex.json`,
`qwen38-m4-final-smoke.json` and `qwen38-m4-mtp-block2.json`.

## Quality on complex prompts

3-bit quantization is the compromise that makes loading possible, not a
guarantee of quality equivalent to a wider-precision model.

- **Math:** the model set up the method, but in no-thinking mode wrote a
  wrong value for `C(30,6)` and hit the output limit. The oracle is
  `(C(30,6)-C(18,6)-C(20,6)-C(22,6)+C(8,6)+C(10,6)+C(12,6))/C(30,6)` =
  **18520/23751**.
- **Python async:** it correctly identifies the check-then-act race, but the
  per-key lock solution remains incomplete regarding cancellation, sharing
  the in-flight task and lock cleanup.
- **Technical writing:** prose is fluent, but it attributes too large a KV
  cache and treats the architecture as pure Transformer, omitting the
  recurrent GDN state.
- **Reasoning medium (manual trial):** it correctly builds the
  inclusion-exclusion and self-corrects on the color mapping, but consumes
  all 512 tokens in the thought block without emitting a final answer.

Qualitative conclusion: good for chat, summaries and light coding with human
verification; do not use it as an unverified source for combinatorial
calculations, memory estimates or delicate concurrency. For those cases you
need a larger budget, more constrained prompts and external verification, or
a less aggressively quantized checkpoint on hardware with more memory.

## Verification performed

- `dwarfstar doctor`: hardware, versions, Metal and both checkpoint sizes OK;
- 27/27 Qwen tests, including ChatML goldens, special tokens, whitespace and
  Python indentation;
- Python compilation and `pip check` without errors;
- legacy C/Metal build and suite passed; the only tests requiring the old
  `ds4flash.gguf` are explicitly skipped when the file is absent;
- real one-shot generation, including clean shutdown after the teardown fix;
- complex serial benchmark, short benchmark on the final runtime and three
  MTP attempts (block sizes 2 and 3); MTP-only correctly returns exit 4;
- serial smoke on the final runtime with the new MLX threshold and 2K
  context: `DUEMILA OK`, exit 0, 6.86 tok/s, 12.07 GB MLX;
- real server: `GET /v1/models`, two sequential `POST /v1/chat/completions`
  with different aliases, `ALFA` and `BETA` responses, HTTP 200, no
  cross-request contamination and clean shutdown; `developer` role
  normalized with `GAMMA`/`OK` response; arbitrary remote model and
  `max_tokens: 0` rejected with HTTP 400;
- real interactive chat: `CHAT OK` response, KV limit enforced and clean
  `Ctrl-D` with exit 0; a second concurrent process rejected with exit 2.

## Addendum — long context and quantized KV (August 18, 2026)

Version 0.3.0 introduced reasoning sessions (`ask`/`chat`).
The long-prefill trials run on the same machine showed that the real limit is
not KV cache storage (only 16 of 64 layers have one, about 64 KiB/token in
bf16) but the transient prefill peak:

| Trial (real prompt, exact retrieval verified) | Result |
| --- | --- |
| 3,346 tokens, ctx 4096, bf16 KV, chunk 256 | OOM in prefill |
| 3,346 tokens, ctx 4096, bf16 KV, chunk 128 | OOM in prefill |
| 3,398 tokens, ctx 4096, KV 8-bit, chunk 128 | OK: 7.58 tok/s, peak 12.49 GB |
| 6,986 tokens, ctx 8192, KV 8-bit, chunk 64 | OOM in prefill |
| 7,038 tokens, ctx 8192, KV 4-bit, chunk 64 | OK: 7.43 tok/s, peak 12.40 GB |
| 2-turn chat, deep defaults (8192/kv4/chunk 64) | OK: 175 tokens reused from cache, 7.28 tok/s, peak 12.08 GB |

Adopted policy: context caps tied to KV quantization (bf16 → 2,048;
≥ 6 bits → 4,096; < 6 bits → 8,192; MTP → 1,024), prefill chunk reduced
automatically (256/128/64) and `--quantized-kv-start 0` mandatory when KV is
quantized (the upstream default, token 5000, would make it a no-op within
these caps). The `deep` (8192/kv4, xhigh thinking) and `balanced`
(4096/kv8, medium thinking) profiles use only measured configurations. The
2-bit checkpoint was evaluated and rejected: it would free about 3.5 GB but
degrades exactly the reasoning quality this runtime is built for.

## Remaining limits and recommended use

1. The target is text-only; images, audio and video are not supported.
2. Use the validated profiles: `deep` 8K with 4-bit KV, `balanced` 4K with
   8-bit KV, `quick` 1K bf16. Without quantized KV the cap is 2K. The
   theoretical 262K context of the model is not practical on 16 GB.
3. Keep MTP in `auto`/serial. Even the minimum valid block size (2, i.e. one
   draft token) runs out of memory. A dedicated q3 kernel or a smaller
   checkpoint remains the most promising path.
4. Never run multiple 27B processes at the same time and close heavy
   applications when the maximum margin is needed.
5. Always verify important technical answers: the observed errors come from
   the model/quantization, not only from the renderer.

Essential commands:

```sh
make qwen-setup
make qwen-download
make qwen-doctor
./dwarfstar chat --no-mtp
./dwarfstar benchmark --mode serial --max-tokens 256
./dwarfstar serve --no-mtp --host 127.0.0.1 --port 8080 --max-sequences 1
```

## Primary sources

- official model: <https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0>;
- exact 3-bit target: <https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly/tree/c98bba5926f51fec1c8d8737e577221673f524d7>;
- exact 3-bit MTP: <https://huggingface.co/lukaskremla/Qwen3.8-27B-MTP-3bit-MLX/tree/9d061a0661258e75b401a11ac9fa22fc648e039d>;
- MLX-VLM 0.6.14: <https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.14>;
- challenge and studied implementation snapshot:
  <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/tree/8dabcfb75c19dad6cbf5cc7cf9f26e1bd440a0dd>;
- live crown observed at 09:17 UTC on August 18, 2026:
  <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/tree/d56b4a0eb4e52f2fb92540ba5c6ed764176e8d33>;
- live MLXFast leaderboard (can change):
  <https://www.yukon.org/mlxfast>;
- contrasting Qwen implementation in llama.cpp:
  <https://github.com/ggml-org/llama.cpp/blob/82dbc4f017a7b005f993ac2e7af9c048ad686c04/src/models/qwen35.cpp>;
- legacy antirez/ds4 base:
  <https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28>.

At 09:17 UTC on August 18, 2026 the live crown showed 84.2 tok/s median on
eight 512-token sequences. That figure was produced on an M5 Max 128 GB with
specialized 4-bit kernels and the ranking changes over time: it is a research
reference for scheduling and fusion, not a target transferable to the
M4 16 GB.
