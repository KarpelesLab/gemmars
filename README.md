# gemmars

Gemma 4 text inference in pure Rust on the CPU. No GPU, no BLAS, no ML framework:
bf16 weights are read straight out of the HuggingFace safetensors files and every
operation (matmuls, attention, norms, RoPE, sampling) is implemented here and run
across threads with NEON SIMD.

Output is validated against the `transformers` float32 reference: on E2B-it the
last-token logits agree to a max absolute difference of about 1e-4 and greedy
decoding produces identical tokens.

## Supported models

| Model | Params (total) | bf16 size | Status |
|---|---|---|---|
| google/gemma-4-E2B-it | 5.1B | 10.2 GB | tested, matches reference (max logit diff 1e-4) |
| google/gemma-4-E4B-it | 8.0B | 16.0 GB | tested, matches reference (max logit diff 8e-5) |
| google/gemma-4-12B-it | 12B | 24 GB | supported (k=v global attention), untested |
| google/gemma-4-31B-it | 31B | 62 GB | supported, sharded loading, untested |
| google/gemma-4-26B-A4B-it | 26B MoE | 52 GB | not yet (MoE block unimplemented) |

Only the text decoder is implemented. Vision and audio towers in the checkpoints are ignored.

## Getting weights

The Gemma 4 checkpoints are Apache 2.0 and not gated, so plain `curl` works:

```sh
M=gemma-4-E4B-it
mkdir -p models/$M
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o models/$M/$f https://huggingface.co/google/$M/resolve/main/$f
done
```

For 31B, fetch `model.safetensors.index.json` and the `model-0000N-of-0000M.safetensors` shards instead of `model.safetensors`.

## Build and run

```sh
cargo build --release
./target/release/gemmars --model models/gemma-4-E4B-it --prompt "Explain RMSNorm in two sentences."
./target/release/gemmars --model models/gemma-4-E4B-it --chat          # interactive, multi-turn
./target/release/gemmars --model models/gemma-4-E4B-it --think -p "..."  # enable thinking mode
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--temp F --top-k N --top-p F --seed N` | sampling (default is greedy, `--temp 0`) |
| `--system TEXT` | system prompt |
| `--raw` | feed the prompt as-is after `<bos>`, no chat template |
| `--max-tokens N` | cap on generated tokens |
| `--threads N` | worker threads (default: all cores) |
| `--quant q8` | quantize streamed weights to int8 at load (blocks of 32, f32 scales); ~1.8x less memory traffic |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

## Performance

Decode is memory-bandwidth bound: every generated token streams all dense weights once.
On an M5 Max (18 cores, ~280 GB/s measured CPU read bandwidth):

| Model | Precision | Weights per token | Decode | Prefill (3k-token prompt) |
|---|---|---|---|---|
| E2B-it | bf16 | 4.6 GB | ~46 tok/s (≈230 GB/s) | ~280 tok/s |
| E2B-it | `--quant q8` | 2.6 GB | ~71 tok/s | ~255 tok/s |
| E4B-it | bf16 | 9.3 GB | ~25 tok/s (≈245 GB/s) | ~135 tok/s |
| E4B-it | `--quant q8` | 5.2 GB | ~40 tok/s | ~135 tok/s |

The matmul kernels reach the machine's bandwidth ceiling with 12 threads
(`cargo run --release --bin membench` prints the raw numbers for your machine), so decode
speed is set by bytes per token: Q8 cuts them by 1.8x. Prefill is compute-bound on an 8x8
outer-product GEMM at roughly 1.3 TFLOP/s (about 55% of the measured f32 FMA peak; a 12-token
tile was tried and lost to register pressure), and Q8 prefill dequantizes to f32 and uses the
same kernel, so prompt processing speed is independent of precision.

## How it works

- `safetensors.rs`: parses the header, exposes bf16 tensors as `&[u16]` (single file or sharded).
- `config.rs`: reads `config.json` and derives per-layer facts (sliding vs global, head dims, KV sharing source, k=v).
- `kernels.rs`: NEON kernels. Decode tiles stream 1–2 weight rows with 8 accumulators (the prefetcher
  prefers few long streams); prefill packs activations into 8-token panels and runs an 8-row x 8-token
  outer-product kernel that broadcasts bf16 weights by lane. Also RMSNorm, GELU (tanh approximation via
  a fast exp), RoPE tables, softmax.
- `quant.rs`: Q8 block quantization (32 weights per f32 scale) of weights at load time and of
  activations per matmul; the decode kernel uses int8 SDOT over four blocks at a time. On E2B/E4B the
  KL divergence of the output distribution from the float32 reference is about 1e-5 nats.
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
./target/release/gemmars --model models/gemma-4-E2B-it --prompt "What is the capital of France?" --dump-logits ours.bin
./venv/bin/python tools/ref_logits.py models/gemma-4-E2B-it prompt.txt ours.bin --generate 12
```
