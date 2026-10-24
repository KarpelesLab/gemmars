# oxinfer

[![CI](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxinfer.svg)](https://crates.io/crates/oxinfer)
[![docs.rs](https://img.shields.io/docsrs/oxinfer)](https://docs.rs/oxinfer)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Try it in your browser](https://karpeleslab.github.io/oxinfer/)**: Qwen 3.5 0.8B downloads from
Hugging Face and runs in the tab, with WebGPU when the browser has it; MLX 4-bit Gemma 4, Gemma 3
and Qwen3 models load from the mirror too. Early days: the crate is at 0.0.x and the API and file
formats may still change.

Local LLM inference in pure Rust: Gemma 4 (text, images, audio), Gemma 3 and Qwen 3 / 3.5 / 3.8
today, more model families as they come. No BLAS, no ML framework: weights are read straight out
of the safetensors files (HuggingFace checkpoints or MLX 4-bit ones) and every operation
(matmuls, attention, norms, RoPE, sampling) is implemented here, with hand-written SIMD kernels
(NEON, AVX2, wasm simd128) and its own thread pool. The CPU path runs everywhere; GPUs (Metal,
CUDA, WebGPU in the browser) are detected at run time and used for prefill when present, with
nothing linked at build time.

Output is validated against the `transformers` float32 reference: on Gemma 4 E2B-it the
last-token logits agree to a max absolute difference of about 1e-4 and greedy decoding produces
identical tokens.

## Supported models

| Model | Params (total) | bf16 size | Status |
|---|---|---|---|
| google/gemma-4-E2B-it | 5.1B | 10.2 GB | tested, matches reference (max logit diff 1e-4) |
| google/gemma-4-E4B-it | 8.0B | 16.0 GB | tested, matches reference (max logit diff 8e-5) |
| google/gemma-4-12B-it | 12B | 24 GB | supported (k=v global attention), untested |
| google/gemma-4-31B-it | 31B | 62 GB | supported, sharded loading, untested |
| google/gemma-4-26B-A4B-it | 26B MoE | 52 GB | not yet (MoE block unimplemented) |
| Qwen/Qwen3.8-27B (and Qwen3.5/3.6 dense) | 27.8B | 55.6 GB | text only; matches the bf16 reference; see below |
| Qwen/Qwen3.6-35B-A3B and fine-tunes (PixilabAI/Zawen-v0.1-35B-A3B) | 35B MoE, 3B active | 69 GB | text only; matches the reference at full width (see below); full checkpoint not yet run |
| Qwen/Qwen3.5-0.8B | 0.8B | 1.7 GB | text only; the small one used for the wasm measurements |
| mlx-community/gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit | 5.1B / 8.0B | 3.6 / 5.2 GB (MLX 4-bit) | text, images and audio; see [MLX checkpoints](#mlx-checkpoints) |
| mlx-community/Qwen3-4B-4bit (Qwen3 dense) | 4.0B | 2.3 GB (MLX 4-bit) | text; matches the reference on the same weights (same top 10) |
| mlx-community/gemma-3-4b-it-4bit (Gemma 3) | 4.3B | 3.4 GB (MLX 4-bit) | text only; matches the reference, also past the 1024-token window |
| google/embeddinggemma-300m | 0.3B | 1.2 GB (fp32) | embeddings; matches sentence-transformers (see [Embeddings](#embeddings)) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1.2 GB | embeddings; matches sentence-transformers |
| BAAI/bge-m3 | 0.6B | 2.3 GB (fp32, `pytorch_model.bin`) | dense, sparse and ColBERT embeddings; matches the reference |
| sentence-transformers/all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2 | 23M / 118M | 90 / 470 MB | embeddings (BERT); match sentence-transformers |
| BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small, mixedbread-ai/mxbai-embed-large-v1 | 33M-335M | 0.13-1.3 GB | embeddings (BERT / XLM-R); match sentence-transformers |
| stabilityai/sd-turbo | 1.3B | 2.6 GB (fp16) | text-to-image in one step; matches diffusers (see Image generation) |
| stable-diffusion-v1-5/stable-diffusion-v1-5 (and SD 1.x / 2.x fine-tunes) | 1.1B | 2.0 GB (fp16) | text-to-image; matches diffusers |

### Qwen3.5 / Qwen3.8

The Qwen3.5-family dense models are a different architecture (`qwen.rs`): 48 Gated DeltaNet layers
(a causal depthwise conv, L2-normalised keys/queries and a 128x128 delta-rule recurrent state per
value head) interleaved with 16 gated full-attention layers (24 query / 4 KV heads of 256, partial
RoPE on 64 dims, a sigmoid output gate), SiLU MLPs and zero-centred RMSNorm. The recurrent layers
keep a fixed-size state instead of a KV cache, so memory does not grow with context for 48 of the
64 layers. The vision encoder and the MTP head in the checkpoint are not loaded yet, and
speculative decoding is not available for this family yet.

The MoE variants (`qwen3_5_moe`, Qwen3.5/3.6-35B-A3B) replace each MLP with 256 routed SiLU
experts of width 512 (softmax router, top 8, weights renormalised) plus a shared expert scaled by
a sigmoid gate. Experts load from the fused `experts.gate_up_proj` / `down_proj` tensors or from
per-expert `experts.N.*_proj` weights (as transformers saves them), and quantize like any other
matrix (`--save-quantized` included). Decode batches the 8 chosen experts' input projections into
one matmul. Prefill gathers each expert's tokens and runs every expert as one grouped matmul on
the CPU. With `--quant q4` (about 19 GB) on an M5 Max, that is about 220 tok/s of prefill and 65
tok/s of decode. On randomly initialised weights at the real width (2048 hidden, 256 experts, 2
layers, the real vocabulary), oxinfer matches transformers in float32 to a max logit difference of
7e-5, with the same 20 greedy tokens. MLX-quantized experts are not read yet.

Thinking is on by default in the Qwen template; `--think` enables it here (the trace is dimmed),
otherwise the generation prompt closes the think block.

Text, images and audio are supported (E2B/E4B have both towers). Video is not.

## Embeddings

sentence-transformers models are recognised by their `modules.json` and run in embedding mode:
the backbone (Gemma 3 with bidirectional attention, Qwen3, or a BERT / XLM-RoBERTa encoder in
`bert.rs`), then the listed modules: pooling (CLS, mean, max, last token), dense projections,
normalization. Named prompts come from `config_sentence_transformers.json`. Each input line
gives one JSON array (or `-p TEXT` one):

```sh
./target/release/oxinfer --model Qwen/Qwen3-Embedding-0.6B --prompt-name query -p "What is the capital of France?"
printf 'Which planet is the Red Planet?\nMars looks red.\nVenus is the hottest planet.\n' |
  ./target/release/oxinfer --model google/embeddinggemma-300m --prompt-name query --similarity
#  1.0000  0.8014  0.5470
#  ...
./target/release/oxinfer --model google/embeddinggemma-300m --dim 256 -p "..."   # Matryoshka
./target/release/oxinfer --model BAAI/bge-m3 --sparse -p "..."     # lexical weights {token: weight}
./target/release/oxinfer --model BAAI/bge-m3 --colbert -p "..."    # one vector per token
```

`google/embeddinggemma-300m` is gated on Hugging Face (set `HF_TOKEN`); `unsloth/embeddinggemma-300m`
is an ungated copy. BGE-M3 ships only a `pytorch_model.bin`: `torch.rs` reads `torch.save`
archives directly (a zip of raw storages plus a pickled state dict; only the handful of pickle
opcodes and constructors that state dicts use are understood, and nothing is executed), so the
same goes for its `sparse_linear.pt` / `colbert_linear.pt` heads.

Checked against sentence-transformers float32 (`tools/ref_embed.py`) on short texts in five
languages and a ~2000-token one: cosine between our vector and the reference at least 0.99999 for
EmbeddingGemma (query and document prompts, and at 256 dimensions) and 1 - 1e-6 for
Qwen3-Embedding; BGE-M3's dense vectors 0.99991 or better. fp32 checkpoints are computed with
bf16-rounded matrices; against a reference using the same rounding the vectors agree to about
1e-5, so that is the whole difference. One caveat: on the long text, BGE-M3's ColBERT vectors
have a mean cosine of 0.9995 to the fp32 reference, but one token drops to 0.40. That text is
unusually sensitive: the reference itself in fp16 already drops a token to 0.89. The sparse
weights select the same tokens, with weights within 0.013.

Any BERT, RoBERTa or XLM-RoBERTa sentence-transformers model should work the same way; besides
the three above, all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, bge-small-en-v1.5,
multilingual-e5-small and mxbai-embed-large-v1 were checked (cosine >= 0.99999 to the reference).
E5 models expect a `query: ` / `passage: ` prefix in the text itself (they define no prompts).
MPNet (all-mpnet-base-v2: BERT plus a shared T5-style relative position bias) is supported too,
cosine >= 0.999991 to the reference (0.999996 against bf16-rounded reference weights).

NomicBERT (`nomic_bert.rs`: rotary positions, fused Wqkv, SwiGLU, dynamic NTK scaling past 2048
tokens) runs nomic-embed-text-v1 and v1.5: cosine >= 0.99998 to the reference (`--trust-remote-code`)
with and without the `search_query: ` / `search_document: ` prefixes, which go in the text (the
model defines no prompts), and on a 4000-token text; 0.999999 against bf16-rounded weights.
v1.5's Matryoshka recipe layer-norms the pooled vector before truncating, which is not a
sentence-transformers module, so `--dim` alone truncates without it.

ModernBERT (alternating global and sliding-window attention) is supported too:
modernbert-embed-base and gte-modernbert-base give cosines of 0.99993 and 0.99998 against the
fp32 reference, and 1 - 1e-6 against one with the same bf16-rounded matrices. modernbert-embed
expects a `search_query: ` / `search_document: ` prefix in the text itself.
Qwen2-based embedders (gte-Qwen2, stella) are another architecture and are not supported yet.

Two details that are easy to get wrong: transformers halves EmbeddingGemma's sliding window for
bidirectional attention (512 in the config means keys less than 257 positions away on either
side), and Qwen3-Embedding pools the `<|endoftext|>` token its tokenizer appends.

From the library (`examples/embed.rs`):

```rust
let dir = oxinfer::hf::resolve("BAAI/bge-m3", None)?;
let files = oxinfer::safetensors::ModelFiles::open(&dir, false)?;
let emb = oxinfer::embed::Embedder::load(&dir, &files, oxinfer::model::Quant::Bf16)?;
let mut e = emb.embed("what is a panda?", None)?;          // or Some("query") with named prompts
e.truncate(256)?;                                         // Matryoshka (--dim)
let lexical = emb.sparse_tokens(&e);                      // BGE-M3 heads: e.sparse / e.colbert
```

## Tool calling

Both model families can call tools, in the format each was trained on: Gemma 4 declares them in
its system turn (`<|tool>declaration:...<tool|>`) and calls `<|tool_call>call:NAME{...}<tool_call|>`;
Qwen 3.5 / 3.8 lists JSON schemas in `<tools>` and answers with `<tool_call><function=...>`.
`src/tools.rs` formats the declarations, parses the calls and feeds results back the way each
template does, and the rendered prompts are byte-identical to `transformers`'
`apply_chat_template(..., tools=...)` for both families.

On the command line `--tools` enables two built-in tools, the local date and time and an exact
calculator:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --tools -p "What is 1234.5 * 6789 - 2^20?"
# [tool calculate({"expression":"1234.5 * 6789 - 2^20"}) -> 7332444.5]
# The result of 1234.5 x 6789 - 2^20 is **7332444.5**.
```

In the browser the model gets `get_datetime` and `run_javascript`, which runs the code it writes in
a Web Worker with the network and storage APIs removed and a 5 s limit (hardening, not a security
boundary). Its system prompt describes where it runs: browser, operating system, language, time
zone, local time, screen, CPU cores and GPU. Both can be changed under Settings.

## Image generation

A diffusers folder (one with `model_index.json`) switches oxinfer to text-to-image with Stable
Diffusion 1.x / 2.x, which includes SD 1.5, SD 2.1 and the one-step SD-Turbo (`src/sd/`):

```sh
./target/release/oxinfer --model models/sd-turbo -p "a red fox in a snowy forest" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 -p "a watercolor lighthouse" --negative "blurry" \
    --steps 25 --guidance 7.5 --sampler euler_a --size 512x512 --seed 7 -o lighthouse.png
```

Turbo models (recognised by name) default to 1 step without guidance, others to 25 Euler-ancestral
steps at guidance 7.5. The CLIP tokenizer and text encoder, the UNet (ResNets, spatial transformers
with GEGLU, skip connections), the VAE decoder and the Euler samplers are implemented here.
Activations are channels-last, so every convolution unfolds into the same matmul the language models
use: the SIMD kernels apply on every CPU, and GPU backends run convolutions directly as an implicit
GEMM (`Backend::conv2d`: the kernel gathers each pixel's neighbourhood itself; Metal measured, CUDA
compile-checked, WebGPU in the browser). Weights are converted from fp16 to bf16 at load.

`tools/ref_sd.py` runs diffusers on the same noise and compares every stage (`OXINFER_SD_DUMP=dir`).
With the weights rounded to bf16 on both sides the outputs agree to about 1e-6 (79 dB PSNR on
the SD-Turbo image, 75 dB on SD 1.5 after 4 guided steps), so the only difference from the fp32
reference is the bf16 weight storage (47 dB).

512x512 on the M5 Max:

| | Text | UNet pass | VAE decode | SD-Turbo image | SD 1.5, 25 steps + guidance |
|---|---|---|---|---|---|
| CPU only (18 threads) | 0.08 s | 0.91 s | 2.87 s | 3.9 s | |
| Metal | 0.08 s | 0.37 s | 0.66 s | 1.1 s | 16 s |
| Browser (Chrome, WebGPU, one CPU thread) | 0.2 s | 1.1 s | 1.9 s | 3.2 s | |

With a GPU, convolutions, attention and the group and layer norms run on it: each norm is fused
into the convolution or projection that consumes it (a statistics kernel, then normalization and
SiLU applied while the input is loaded), and attention runs its two products as GPU matmuls with a
softmax kernel between them. What stays on the CPU is element-wise: residual adds, the GEGLU gate,
upsampling and skip concatenation. `OXINFER_GPU_CHECK=1` recomputes every device op on the CPU and
prints the worst relative error per kind (about 4e-6 for attention, 1e-6 for the fused norms, on
Metal and WebGPU). SDXL, SD 3 and Flux use different architectures and are not supported.

**LoRA.** `--lora FILE[:SCALE]` (repeatable, scale 1 by default) merges a LoRA into the UNet and
text-encoder weights at load, `W += scale * alpha / rank * up @ down` in f32 before the bf16
conversion, so generation runs at the base model's speed (`src/sd/lora.rs`). Convolution LoRAs
(LoCon: a `[rank, in, 3, 3]` down and a `[out, rank, 1, 1]` up) are merged the same way. Both
common layouts are read: kohya / A1111 (`lora_unet_down_blocks_0_..._to_q.lora_down.weight`,
`.alpha`, `lora_te_text_model_...`) and diffusers / PEFT (`unet.<module>.lora_A.weight`, `.lora.down`,
`text_encoder.`, with the alpha from the `lora_adapter_metadata` header). Keys that match no weight
of the model, and LyCORIS kinds other than plain LoRA / LoCon (LoHa, LoKr, DoRA), are skipped and
reported; a file that matches nothing, or whose shapes belong to another base model, is an error.
`FILE` may also be a Hugging Face path, `org/name/file.safetensors` (or `org/name` when the repo
holds one safetensors file), fetched into the cache like models; or download it yourself:

```sh
curl -L -o lcm-lora.safetensors \
    https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
./target/release/oxinfer --model models/sd-v1-5 --lora lcm-lora.safetensors \
    --steps 4 --guidance 1 --sampler euler -p "a red fox in the snow" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 --lora KoRo8888/ufotable_artstyle_locon_v1:0.8 \
    -p "a girl standing in a forest, anime screencap" -o forest.png
```

Against diffusers' `load_lora_weights` + `fuse_lora` (`tools/ref_sd.py --lora`, bf16 weights on
both sides), on SD 1.5: LCM-LoRA (kohya keys, 278 UNet modules including convolutions), 4 Euler
steps, 79.6 dB; the ufotable LoCon (kohya, 278 UNet + 72 text-encoder modules) at 0.8, 20 steps
at guidance 7.5, 78.4 dB, and the same file re-saved by diffusers in the PEFT layout gives the
identical image; both together (LCM 1.0 + ufotable 0.6), 82.3 dB. Scale 0 reproduces the base
model bit for bit. In the library, `sd::Pipeline::load_with_loras(dir, &[LoraSpec { path, scale }],
report)` does the same (`examples/sd_lora.rs`); the browser build loads no LoRA yet.

## Image and audio input

```sh
./target/release/oxinfer --model models/gemma-4-E4B-it --quant q4 \
    --image photo.jpg --audio clip.wav --prompt "Describe the image and transcribe the audio."
```

`--image` and `--audio` can be repeated; attachments are placed before the text of the user turn
(or of the first turn in `--chat`). Files are decoded with the pure-Rust
[oxideav](https://crates.io/crates/oxideav-core) crates (PNG, JPEG, WebP, GIF, BMP; WAV, MP3, FLAC,
AAC/M4A, Ogg Vorbis/Opus, MKV audio), resampled to 16 kHz mono for audio.

- **Vision**: aspect-preserving bicubic resize to at most 280 soft tokens (`--max-soft-tokens`
  accepts 70/140/280/560/1120), 16x16 patches, a 16-layer ViT (hidden 768) with axial 2D RoPE and
  clipped linears, 3x3 patch pooling, then the vision embedder. A 640x480 image becomes 57x42
  patches and 266 soft tokens in about 2.5 s.
- **Audio**: 128-bin log-mel features (20 ms frames, 10 ms hop), two stride-2 conv layers, a
  12-layer conformer with chunked local attention and relative position bias, then the audio
  embedder: 40 ms of audio per soft token (750 max, i.e. 30 s).

Soft tokens replace the `<|image|>` / `<|audio|>` placeholder embeddings inside the text model
(the per-layer-embedding lookup uses the pad token there, as in the reference). All attention
stays causal, matching the E2B/E4B configuration.

## Getting weights

`--model` accepts a HuggingFace repo id and fetches it on first use (resumable, pure-Rust HTTP via
[rsurl](https://crates.io/crates/rsurl)) into `~/.cache/oxinfer/hub` (or `$OXINFER_CACHE`,
or `--cache-dir`):

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 -p "Hello"
./target/release/oxinfer --model Qwen/Qwen3.8-27B --download-only     # just fetch
```

Set `HF_TOKEN` for gated repos. The Gemma 4 checkpoints are Apache 2.0 and not gated; manual
downloads with plain `curl` work too:

```sh
M=gemma-4-E4B-it
mkdir -p models/$M
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o models/$M/$f https://huggingface.co/google/$M/resolve/main/$f
done
```

For 31B, fetch `model.safetensors.index.json` and the `model-0000N-of-0000M.safetensors` shards instead of `model.safetensors`.
For speculative decoding also fetch `google/gemma-4-E4B-it-assistant` (160 MB) the same way.

### The ATOnline mirror

Repo ids are looked up on the [ATOnline](https://www.atonline.com/) mirror first: its
`AI/Model:get` method (`https://ws.atonline.com/_special/rest/AI/Model:get?key=org/name`) lists a
repo's files with download URLs, small ones inlined. Repos it does not carry come from
huggingface.co. `OXINFER_HUB=mirror` or `OXINFER_HUB=hf` (`oxinfer::hf::set_hub` in the library) pins one source; the HuggingFace token is
only ever sent to huggingface.co. The browser build uses the same lookup (the mirror allows
cross-origin requests).

### MLX checkpoints

MLX repos (mlx-community and the like) store each matrix as packed 4-bit values (`.weight`, eight
per uint32) plus a bf16 scale and bias per group of 64: `w = scale * q + bias`. Since
`scale * q + bias = scale * (q - 8) + (bias + 8 * scale)`, the nibbles carry over unchanged into
the Q4 kernels' layout, and the extra term becomes one bf16 offset per group. Decoding adds
`offset * sum(x over the group)` per row after the int8 dot products, and prefill dequantizes it
into the f32 tiles (on the CPU, Metal, CUDA and WebGPU alike). Other bit widths, and quantized
tensors that are read as plain bf16 (Gemma 4's vision and audio projections), are dequantized on
load; embedding tables are dequantized one row at a time straight from the checkpoint. The MLX
names (`language_model.model.*`, `vision_tower.*`) map to the transformers ones, and channels-last
convolution weights are permuted back.

```sh
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit -p "Hello"
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit --audio clip.wav -p "Transcribe this."
```

Against transformers float32 on the dequantized MLX weights, last-token logits agree to a mean
absolute difference of 0.016 (Gemma 4 E4B), 0.026 (Qwen3-4B) and 0.055 (Gemma 3 4B), with the
same top 10; the difference to the original bf16 models is MLX's own quantization.

M5 Max, 200-token replies: Gemma 4 E2B 110 tok/s (120 for oxinfer's own Q4 file, which has no
offsets), Gemma 4 E4B 71 tok/s, Gemma 3 4B 77 tok/s, Qwen3-4B 70 tok/s. Prefill of 2153 tokens on
Gemma 3 4B takes 4.0 s with Metal (12.9 s on the CPU).

`tools/mlx_to_hf.py` dequantizes an MLX checkpoint into a float32 transformers folder, which is
the exact reference for those weights; `tools/ref_logits.py` then compares logits. LiteRT-LM
files (`.litertlm`, TFLite graphs) are not supported.

## Build and run

```sh
cargo build --release
./target/release/oxinfer --model models/gemma-4-E4B-it --prompt "Explain RMSNorm in two sentences."
./target/release/oxinfer --model models/gemma-4-E4B-it --chat          # interactive, multi-turn
./target/release/oxinfer --model models/gemma-4-E4B-it --think -p "..."  # enable thinking mode
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--temp F --top-k N --top-p F --seed N` | sampling (default is greedy, `--temp 0`) |
| `--system TEXT` | system prompt (default: a short one saying the model runs locally, with the local date, time and zone) |
| `--no-system` | no system prompt (also implied by `--raw` and `--dump-logits`) |
| `--tools` | let the model call built-in tools: `get_current_datetime` and `calculate` (see Tool calling) |
| `--print-prompt` | print the fully formatted prompt and exit |
| `--raw` | feed the prompt as-is after `<bos>`, no chat template |
| `--max-tokens N` | cap on generated tokens |
| `--threads N` | worker threads (default: all cores) |
| `--quant q8` | quantize streamed weights to int8 at load (blocks of 32, f32 scales); ~1.8x less memory traffic |
| `--quant q4` | GGUF-style Q4_0 (blocks of 32, 4-bit, f16 scale) for all streamed weights; ~3.5x less traffic |
| `--draft DIR` | speculative decoding with Google's `gemma-4-E4B-it-assistant` drafter (greedy only; output identical to plain greedy) |
| `--draft-tokens N` | max draft tokens per round (default 6; the length adapts between 2 and N) |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--save-quantized DIR` | write a pre-quantized, text-only checkpoint at `--quant` precision (see below) and exit |
| `--devices` | list detected CPU features and GPUs (Metal, CUDA, Vulkan) and exit |
| `--device D` | `auto` (default: use a GPU backend when one exists), `cpu`, `metal`, `cuda`, `vulkan` |
| `--prefill-chunk N` | tokens per prefill chunk (default 256 on the CPU, 1024 when a GPU backend is active) |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

## Performance

Decode is memory-bandwidth bound: every generated token streams all dense weights once.
On an M5 Max (18 cores, ~280 GB/s measured CPU read bandwidth):

| Model | Precision | Weights per token | Decode | Decode + drafter (prose / code) | Prefill (3k prompt) |
|---|---|---|---|---|---|
| E2B-it | bf16 | 4.6 GB | ~50 tok/s (≈230 GB/s) | n/a (no E2B drafter tested) | ~280 tok/s |
| E2B-it | `--quant q8` | 2.6 GB | ~86 tok/s | | ~255 tok/s |
| E2B-it | `--quant q4` | 1.3 GB | ~130 tok/s | | |
| E4B-it | bf16 | 9.3 GB | ~28 tok/s (≈245 GB/s) | 41 / 47 tok/s | ~135 tok/s |
| E4B-it | `--quant q8` | 5.2 GB | ~45 tok/s | 66 / 82 tok/s | ~135 tok/s |
| E4B-it | `--quant q4` | 2.6 GB | ~76 tok/s | 75 / 99 tok/s | ~135 tok/s |
| Qwen3.8-27B | bf16 | 51 GB | ~5.3 tok/s (≈270 GB/s) | n/a | ~19 tok/s |
| Qwen3.8-27B | `--quant q8` | 29 GB | ~8.8 tok/s | n/a | ~19 tok/s |
| Qwen3.8-27B | `--quant q4` | 14 GB | ~16 tok/s | n/a | ~20 tok/s |

For the 27B model `--mmap` is recommended: quantization then reads the checkpoint straight from
the page cache (about 3 s to quantize 27.8B parameters on 18 threads) instead of copying 55 GB
into the process first.

With the Metal backend (default on Apple Silicon) prefill runs on the GPU:

| Model | Precision | Prefill, CPU only | Prefill, Metal |
|---|---|---|---|
| E4B-it | `--quant q4` | ~97 tok/s | ~580 tok/s |
| Qwen3.8-27B | `--quant q4` | ~18 tok/s | ~96 tok/s (256-token chunks) |

(616-token prompt; decode speed is unchanged, see below for why.)

The matmul kernels reach the machine's bandwidth ceiling with 12 threads
(`cargo run --release --bin membench` prints the raw numbers for your machine), so decode
speed is set by bytes per token: Q8 cuts them by 1.8x and Q4 by 3.5x. Prefill is compute-bound
on an 8x8 outer-product GEMM at roughly 1.3 TFLOP/s (the kernel itself runs at ~90% of FMA peak
when operands are in L1; a 12-token tile was tried and lost to register pressure). Quantized
prefill dequantizes to f32 and uses the same kernel, so prompt processing speed is independent
of precision.

### Quantization quality

Q8 and Q4 are both measured against the bf16 output distribution (itself within 1e-4 of the
transformers float32 reference). KL divergence of the next-token distribution on a test prompt is
about 1e-5 nats for Q8 and 3e-5 nats for Q4 on E4B, with the same top-5 tokens. Google also
publishes `gemma-4-*-it-qat-q4_0-unquantized` checkpoints trained for Q4_0; they load the same
way (`--quant q4` on that directory) but did not measure better than Q4 of the regular weights
here, and they are a different fine-tune (KL 0.03 from the regular model).

### Pre-quantized checkpoints

`--quant` normally quantizes the bf16 weights at every start. `--save-quantized DIR` writes the
result instead, so later runs (and browsers) load the int8 / int4 blocks as they are:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --save-quantized models/gemma-4-E4B-it-q4
./target/release/oxinfer --model models/gemma-4-E4B-it-q4 -p "Hello"     # precision comes from the file
```

The output is ordinary safetensors: each quantized matrix `X` becomes `X.q8` (I8) + `X.q8s` (F32
scales) or `X.q4` (U8, two nibbles per byte) + `X.q4s` (F16 scales), in exactly the in-memory
layout of the kernels, so the loader borrows them without copying (`QBuf`). Only the tensors the
text model reads are written (no vision/audio towers, no Qwen MTP head); `__metadata__` records the
precision. Shards are at most 1 GiB (one allocation must stay under 2 GiB in wasm32), and large
tensors start on 16 KiB boundaries so Metal wraps them in place. The embedding tables are stored
quantized too and dequantized one row per token; tables that are only ever looked up (Gemma's
per-layer embeddings, Qwen 27B's untied input embedding) go to separate `model-tables-*` shards,
which the browser host keeps outside wasm memory. `OXINFER_TABLE_QUANT=bf16|q8|q4` overrides
their precision (default: the model's).

| Model | bf16 checkpoint | Q4 file | Load (in RAM) | KL vs bf16: Q4 at load / Q4 file |
|---|---|---|---|---|
| E2B-it | 10.2 GB | 2.6 GB | 0.40 s | 6.5e-4 / 5.9e-4 |
| E4B-it | 16.0 GB | 4.2 GB | 0.48 s (was 1.51 s) | 0.167 / 0.181 |
| Qwen3.8-27B | 55.6 GB | 15.1 GB | 2.55 s (4.6 s with `--mmap` before) | 0.130 / 0.133 (vs Q8) |

The KL column uses one fixed prompt whose last-token distribution is unusually sensitive; what it
shows is that the file adds nothing on top of Q4 itself (the per-layer embedding table at Q4 is
within noise of keeping it at higher precision). Q8 files match Q8-at-load the same way (E4B
7.1e-5 vs 6.7e-5). On Qwen3.5-0.8B, Q4 is too lossy in either form (KL ~1.3); use Q8 there.

### Speculative decoding

`--draft models/gemma-4-E4B-it-assistant` enables Google's multi-token-prediction drafter: a
4-layer, 256-wide decoder that cross-attends to the main model's shared KV caches and predicts
from the concatenation of the last token's embedding and the previous hidden state. Each round
drafts 2–6 tokens (adaptive) and verifies them in one multi-token forward pass; greedy
verification makes the output identical to plain greedy decoding. Acceptance is about 40% on prose
and 55–65% on code, giving 2–4 tokens per forward. The drafter's own drafts were checked against
the transformers implementation token for token. It pays off most where the verification pass
is bandwidth-bound (bf16, Q8); at Q4 the multi-token int8 kernels are instruction-bound, so it
only helps on predictable text.

## Platforms and accelerators

The CPU path is the reference and works everywhere; everything else is detected at run time and
loaded with [`dlsym`](https://crates.io/crates/dlsym), so one binary runs with or without the
libraries present and nothing GPU-related is needed at build time. `--devices` prints what was found:

```
CPU: aarch64 on macos, 18 logical cores
  features: neon dotprod i8mm bf16 fp16
  memory: 137.4 GB
GPU (metal #0): Apple M5 Max, memory 115.4 GB (unified with CPU)
```

| Platform | Compute | Status |
|---|---|---|
| Apple Silicon (macOS, iOS) | NEON kernels + Metal GEMM for prefill | measured, bit-identical logits to the CPU path |
| x86-64 (Linux, Windows, macOS) | AVX2 + FMA kernels (bf16, Q8 via `maddubs`, Q4), scalar fallback below AVX2 | compiles for all three; the kernel tests pass on the scalar fallback under Rosetta (which exposes no AVX), so the AVX2 kernels themselves are not yet executed anywhere |
| Linux / Windows + NVIDIA | CUDA driver API + NVRTC-compiled GEMM (`cuda.rs`) | compile-checked only, not yet run on NVIDIA hardware |
| Android (aarch64) | NEON kernels | cross-compiles (`aarch64-linux-android`) |
| Vulkan | detection only (device name and memory) | no compute yet |
| wasm32 (WASI) | simd128 kernels (bf16, Q8, Q4), threads via wasi-threads | measured: Qwen3.5-0.8B Q8 at 123 tok/s decode with 16 threads under wasmtime 30, identical output to native; see below |
| Browser (Chrome 137+) | simd128 kernels + WebGPU GEMM for prefill (WGSL, via JSPI) | measured in headless Chrome 155: Qwen3.5-0.8B Q8 prefill 34 -> 231 tok/s, identical output; Firefox/Safari run the CPU path (no JSPI) |

What the GPU does and does not do: every backend only takes prefill-sized matmuls (`t >= 16`
tokens and enough FLOPs to amortize the ~0.15 ms it costs to submit and wait for a Metal command
buffer). Weights are not copied on Apple Silicon: quantized matrices and the safetensors blob
live in page-aligned memory and are wrapped as shared `MTLBuffer`s. The Metal GEMM (64x64 tile per
threadgroup, `simdgroup_float8x8`, weights dequantized to f32 in threadgroup memory) runs at
~10 TFLOP/s for bf16 and ~7.5 for Q4, against ~1.3 TFLOP/s for the NEON GEMM. Decode stays on
the CPU on purpose: it is bandwidth-bound, the CPU already streams weights at ~280 GB/s, and the
GPU's ~450 GB/s only pays off if the entire forward pass moves over (attention, norms, the KV
cache), which is the next step if decode speed on the GPU is wanted.

### WebAssembly

`.cargo/config.toml` turns on `simd128` for the wasm targets (every current engine supports it;
without it the scalar kernels are used, ~8x slower on Q8 decode) and raises the shared-memory
maximum for the threads target to 4 GiB. The model directory is passed in as a preopened directory:

```sh
# single-threaded, runs under Node's built-in WASI (node --experimental-wasi ... on older versions)
cargo build --release --target wasm32-wasip1 --no-default-features
# threads (wasi-threads): wasmtime up to v30 runs it; v31+ dropped the wasi-threads proposal
cargo build --release --target wasm32-wasip1-threads --no-default-features
wasmtime run -W threads=y -S threads=y --dir ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B::/model \
  target/wasm32-wasip1-threads/release/oxinfer.wasm --model /model --quant q8 -j 16 -p "Hello"
```

Qwen3.5-0.8B Q8, 18-core M5 Max, 17-token prompt:

| Build | Threads | Prefill | Decode |
|---|---|---|---|
| wasm32-wasip1, scalar | 1 | 9 tok/s | 4.6 tok/s |
| wasm32-wasip1, simd128 | 1 | 29 tok/s | 40 tok/s |
| wasm32-wasip1-threads, simd128 | 16 | 327 tok/s | 123 tok/s |
| native (NEON) | 18 | 373 tok/s | 184 tok/s |

Memory is 32-bit, so a model plus its quantized copy must fit in 4 GiB (`--quant q4` on a 2B
model is fine; the 27B models are not). No file mapping exists in WASI, so the checkpoint is
always read into memory.

### Browser (WebGPU)

The site has a Chat mode and an Image mode; Image mode loads SD-Turbo (2.6 GB of fp16 weights)
from Hugging Face by default. Diffusion weights stay in JavaScript memory and the module converts
them tensor by tensor, so only the bf16 copy counts against the 4 GiB of wasm memory.

The `web` feature (`src/web/`) turns the library into a plain C-ABI wasm module, no bindgen;
`cargo build-web` builds it as a `cdylib` for `wasm32-unknown-unknown` (the crate type is given on
the command line, so native builds never link a shared library). `web/` is the site deployed to
[GitHub Pages](https://karpeleslab.github.io/oxinfer/) by `.github/workflows/pages.yml`: a Vue 3 +
Vite app around the JavaScript host `web/src/lib/oxinfer.js`; it is not a crate. Downloads go to
the browser's Cache Storage, so a model is fetched once per browser. MLX checkpoints load too
(Gemma 4 E2B and E4B, Gemma 3 4B and Qwen3-4B from the mirror were checked in headless Chrome):
the host splits each one into shards of at most 1 GiB kept in JavaScript memory as it streams in,
since a 5 GB file fits neither one allocation nor a 32-bit module, and the module converts the
matrices into its own memory, then lets the host drop every shard but the embedding tables. The host registers the model files in the module's in-memory filesystem (streamed straight from
`fetch` or a local folder into module memory) and drives a chat session token by token.

GPU work goes the other way round: the module imports `gpu_upload` / `gpu_matmul`, which the
host implements with WebGPU (WGSL GEMM, 64x64 tile per workgroup, weights dequantized to f32 in
workgroup memory; ~3.9 TFLOP/s at 4096^3 on the M5 Max, ~0.7 ms per round trip). The module's
matmul call is synchronous while the buffer readback is asynchronous, so `gpu_matmul` is wrapped
in `WebAssembly.Suspending` and the generating exports in `WebAssembly.promising` (JavaScript
Promise Integration, Chrome 137+). Without JSPI or WebGPU the same page runs on the CPU.
WebGL is not used: it has no compute shaders.

```sh
cd web
npm install
npm run wasm                                       # cargo build-web, copied to public/oxinfer.wasm
npm run smoke                                      # the JS <-> wasm bridge, headless (Node)
OXINFER_MODELS=~/.cache/oxinfer/hub npm run dev    # local folders load as "./models/<folder>"
# or from Deno directly (Deno has WebGPU, but its readback polls every ~14 ms, so --cpu is as fast there):
deno run --allow-read --allow-env scripts/run.mjs --model ../models/gemma-4-E4B-it-q4 --cpu -p "Hello"
```

Point the page (or `run.mjs`) at a pre-quantized checkpoint (`--save-quantized`, see above): it
downloads only the quantized bytes and uses them in place, and its `model-tables-*` shards stay in
JavaScript memory, read one row per token. That is what makes the larger models fit: E4B at Q4
puts 2.6 GB in wasm memory and 1.6 GB of per-layer embeddings in JavaScript. Plain bf16
checkpoints still work for small models (they are quantized in the module, which then holds both
copies).

Headless Chrome 155, Qwen3.5-0.8B Q8, 123-token prompt, single CPU thread (the browser build has
no threads):

| | Prefill | Decode |
|---|---|---|
| CPU (simd128) | 34 tok/s | 36 tok/s |
| WebGPU prefill, first turn (includes the weight upload) | 165 tok/s | 38 tok/s |
| WebGPU prefill, steady state | 231 tok/s | 38 tok/s |

Gemma 4 from pre-quantized Q4 files, same page, same 115-token prompt (second turn, weights
already on the GPU); the output text is identical with and without WebGPU, and 40 greedy tokens
from the wasm build match the native binary on the same file:

| Model | CPU prefill | WebGPU prefill | Decode |
|---|---|---|---|
| E2B-it Q4 | 10 tok/s | 124 tok/s | 7.7 tok/s |
| E4B-it Q4 | 5 tok/s | 125 tok/s | 4.0 tok/s |

Limits: decode is single-threaded CPU and bandwidth-bound (one core streams ~10 GB/s of Q4
weights), and stays off the GPU for the same reasons as on Metal. Weights in module memory must
fit in 4 GiB, and no single file can exceed 2 GiB. `web/scripts/test_gemm.mjs` checks the WGSL kernels
against a JavaScript reference.

Feature flags: `media` (oxideav image/audio decoding), `hf` (HuggingFace fetching), `gpu`
(dlsym-loaded backends), all on by default, and `web` (the browser module, off by default and only
meaningful for `wasm32-unknown-unknown`); `--no-default-features` gives the plain CPU build.
Cross-compiling needs no C toolchain: the tokenizer uses its pure-Rust regex backend.

## How it works

- `safetensors.rs`: parses the header, exposes bf16 tensors as `&[u16]` (single file or sharded).
- `config.rs`: reads `config.json` and derives per-layer facts (sliding vs global, head dims, KV sharing source, k=v).
- `platform.rs`: CPU feature and GPU detection (sysctl / procfs for memory, Metal via `objc_msgSend`,
  CUDA driver API, Vulkan loader), all through `dlsym`; `gpu.rs`: the `Backend` trait the linear layers
  dispatch through and the registry; `metal.rs`, `cuda.rs`: the backends with their shader sources.
- `fsx.rs`, `clock.rs`: file access and monotonic time behind one interface, so the loaders run on a
  real filesystem or on the browser's in-memory table; `src/web/` (feature `web`): the browser module's
  C-ABI exports and the WebGPU backend calling back into the host; `web/` at the root: the
  JavaScript host and page.
- `sd/`: Stable Diffusion: `clip.rs` (BPE tokenizer, text encoder), `unet.rs`, `vae.rs`, `sched.rs`
  (Euler samplers), `ops.rs` (channels-last convolution, group/layer norm, multi-head attention on
  the f32 GEMM tiles); `png.rs`: a dependency-free PNG writer.
- `kernels.rs`: NEON kernels (`kernels_x86.rs`, `kernels_wasm.rs`: the AVX2 and simd128 equivalents, selected by `build.rs` through the `has_simd` cfg). Decode tiles stream 1–2 weight rows with 8 accumulators (the prefetcher
  prefers few long streams); prefill packs activations into 8-token panels and runs an 8-row x 8-token
  outer-product kernel that broadcasts bf16 weights by lane. Also RMSNorm, GELU (tanh approximation via
  a fast exp), RoPE tables, softmax.
- `quant.rs`: Q8 (32 weights per f32 scale) and Q4_0 (two nibbles per byte, f16 scale) block
  quantization of weights at load time, plus per-matmul activation quantization; decode kernels use
  int8 SDOT over four blocks at a time, with multi-token variants for speculative verification.
- `drafter.rs`: the Gemma 4 assistant model (drafter) and its centroid-masked LM head.
- `vision.rs`, `audio.rs`: the two encoders plus preprocessing (resize/patching, mel features) and
  the embedders; `media.rs`: file decoding through oxideav; `mm_config.rs`: tower configs and
  placeholder token ids.
- `pool.rs`: a spin-waiting thread pool with an atomic work counter. Dispatch costs about a microsecond,
  which matters because a decode step issues ~200 small parallel regions.
- `model.rs`: the Gemma 4 decoder: scaled embeddings, per-layer embeddings (PLE), QK-norm, sliding /
  global attention with proportional RoPE on 512-dim global heads, KV-cache sharing for the top layers,
  gated-GELU MLP, post-norms, per-layer scalars, tied softcapped LM head.
- `chat.rs`, `sampler.rs`, `main.rs`: chat template, greedy / top-k / top-p sampling, streaming CLI.

## Verifying against the reference

```sh
python3 -m venv venv && ./venv/bin/pip install torch transformers safetensors numpy
printf '<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' > prompt.txt
./target/release/oxinfer --model models/gemma-4-E2B-it --prompt "What is the capital of France?" --dump-logits ours.bin
./venv/bin/python tools/ref_logits.py models/gemma-4-E2B-it prompt.txt ours.bin --generate 12
```
