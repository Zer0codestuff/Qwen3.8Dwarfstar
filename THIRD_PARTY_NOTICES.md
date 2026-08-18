# Third-party notices

This file records the principal third-party projects, models and research used
by the Qwen 3.8 execution path. It is provided for attribution and does not
replace the license files shipped by each dependency or model repository.

The Python dependencies are installed into `.venv` by `qwen-setup`; their
source code is not copied into this repository unless stated otherwise.

## MLX

- Project: [ml-explore/mlx 0.32.1](https://github.com/ml-explore/mlx/tree/3a6219917e4535575ce5bce2fc2ba27a483a709b)
- Version used: `0.32.1`
- Copyright: © 2023 Apple Inc.
- License: MIT

MLX provides the Apple Silicon array runtime, allocator and Metal kernels used
for model execution.

## MLX-VLM

- Project: [Blaizzy/mlx-vlm 0.6.14](https://github.com/Blaizzy/mlx-vlm/tree/625f71fae24f0d5c5ee7f1ec747094e815393405)
- Version used: `0.6.14`
- Copyright: © 2025 Prince Canuma and contributors
- License: MIT

MLX-VLM provides the Qwen 3.8 model implementation, chat generation, server,
MTP speculative decoding and rollback machinery. DwarfStar applies a runtime-only
low-memory policy to its public Python implementation; the upstream package is
installed unchanged.

## Transformers and Tokenizers

- Projects: [huggingface/transformers 5.15.0](https://github.com/huggingface/transformers/tree/5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9)
  and [huggingface/tokenizers 0.22.2](https://github.com/huggingface/tokenizers/tree/f383101a26663708484cac0727792aad74f78234)
- Versions used: `5.15.0` and `0.22.2`
- License: Apache License 2.0

They provide the pinned Qwen tokenizer, chat-template execution and model
configuration utilities used by MLX-VLM and the golden tests.

## Qwen 3.8 27B

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- Model author: Qwen team / Alibaba Cloud
- License: Apache License 2.0, subject to the upstream model card

Default converted checkpoints:

- [lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly/tree/c98bba5926f51fec1c8d8737e577221673f524d7), revision `c98bba5926f51fec1c8d8737e577221673f524d7`;
- [lukaskremla/Qwen3.8-27B-MTP-3bit-MLX](https://huggingface.co/lukaskremla/Qwen3.8-27B-MTP-3bit-MLX/tree/9d061a0661258e75b401a11ac9fa22fc648e039d), revision `9d061a0661258e75b401a11ac9fa22fc648e039d`.

These are third-party quantized conversions of the upstream checkpoint. They
are downloaded separately and are not distributed in this repository. Their
model cards, licenses and upstream restrictions continue to apply. The MTP
checkpoint is an auxiliary drafter and is not a standalone language model.

## Hugging Face Hub

- Project: [huggingface/huggingface_hub 1.27.0](https://github.com/huggingface/huggingface_hub/tree/a7d85da5de06e8b85e94999e7cf985f6e0f9991b)
- Version used: `1.27.0`
- License: Apache License 2.0

It is used only to download and resolve the immutable model snapshots.

## Qwen 3.8 MLXFast challenge

- Project: [Layr-Labs/qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/tree/8dabcfb75c19dad6cbf5cc7cf9f26e1bd440a0dd)
- Implementation snapshot studied: `8dabcfb`
- Copyright: © 2026 Layr Labs, Inc.
- License: MIT

The challenge informed the benchmark design, scoring and correctness criteria.
Speculative decoding and cache rollback are supplied by MLX-VLM 0.6.14;
challenge source code is not copied here. Challenge model weights are not included.
Leaderboard throughput was measured on M5 Max 128 GB hardware and is not a
performance claim for this 16 GB M4 build.

## Historical derivation

This project was originally derived from
[antirez/ds4 at `84cc882`](https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28),
which in turn drew from [llama.cpp/GGML](https://github.com/ggml-org/llama.cpp).
The legacy C/Metal, CUDA and ROCm engine has been removed from this
repository; the current runtime is a Python wrapper around MLX and MLX-VLM.
See the repository [LICENSE](LICENSE) for the applicable MIT terms.

## MIT license text

For the MIT-licensed components listed above:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
