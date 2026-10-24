# Core ML (Apple Neural Engine) package format

`tools/coreml_convert.py` writes Hugging Face checkpoints as Core ML packages in the layout of
[ANEMLL](https://github.com/Anemll/Anemll) 0.3.5 (commit `f4ad26d`), so ANEMLL's runtimes
(`tests/chat.py`, the Swift CLI) and oxinfer's Core ML runtime load them. This document is the
runtime contract: the files, every model input, output and state, and the call sequence. Plain
transformer packages (`architecture: qwen3`) need nothing beyond ANEMLL's contract; hybrid
recurrent ones (`architecture: qwen3_5`) add states and two rules (marked **extension**).

`tools/coreml_check.py` is a reference implementation of the runtime side in Python
(coremltools); it is the code to read when something here is unclear.

## Files

```
meta.yaml                                       description + runtime parameters
<prefix>_embeddings[_lutN].mlmodelc             token ids -> hidden states
<prefix>_FFN_PF[_lutN]_chunk_01ofNN.mlmodelc    decoder layers, functions `infer` and `prefill`
...
<prefix>_FFN_PF[_lutN]_chunk_NNofNN.mlmodelc
<prefix>_lm_head[_lutN].mlmodelc                final hidden state -> logits
config.json                                     source config.json (+ tokenizer_class)
tokenizer.json, tokenizer_config.json, ...      tokenizer files copied from the checkpoint
generation_config.json, chat_template.jinja     when the checkpoint has them
```

`_lutN` names the palettization bits (`_int8` / `_int4` for linear quantization). The default
prefix is `qwen` for Qwen3 and `qwen35` for Qwen3.5. All models are ML Programs for iOS 18 /
macOS 15 or later (Core ML states), fp16 compute.

## meta.yaml

```yaml
model_info:
  name: anemll-Qwen-Qwen3.5-0.8B-ctx512
  version: 0.3.5                 # ANEMLL format version
  architecture: qwen3_5          # qwen3 | qwen3_5
  parameters:
    context_length: 512          # KV cache length = largest position + 1
    batch_size: 64               # tokens per `prefill` call
    lut_embeddings: none         # palettization bits per part, or none
    lut_ffn: 4
    lut_ffn_per_channel: 8       # output channels sharing one LUT
    lut_lmhead: 6
    lut_lmhead_per_channel: 8
    num_chunks: 1
    model_prefix: qwen35
    embeddings: qwen35_embeddings.mlmodelc
    lm_head: qwen35_lm_head_lut6.mlmodelc
    ffn: qwen35_FFN_PF_lut4_chunk_01of01.mlmodelc   # first chunk; others: _chunk_02of01 ...
    split_lm_head: 16            # number of logits outputs
    vocab_size: 248320
    lm_head_chunk_sizes: [15520, 15520, ...]          # width of logits1..logitsN
    prefill_dynamic_slice: true  # prefill writes the KV cache at current_pos (ANEMLL flag)
    prefill_last_token_output: true   # see "prefill output" below
    hidden_size: 1024
    num_layers: 24
    recommended_sampling: {do_sample: true, temperature: 0.6, top_p: 0.95, top_k: 20}
    # --- extensions, only for recurrent architectures (qwen3_5) ---
    recurrent_state: true        # states that cannot be rewound: feed every token exactly once
    prefill_full_batches_only: true
    gdn_chunk_size: 64           # informational (chunk of the in-graph delta rule)
    layer_types: [linear, linear, linear, full, ...]
    states:
      - name: model_model_kv_cache_0
        shape: [12, 2, 512, 256]
        dtype: float16
        layers: [3, 7, 11, 15, 19, 23]
      - name: model_model_gdn_state_0
        shape: [288, 128, 128]
        dtype: float16
        layers: [0, 1, 2, 4, ...]
      - name: model_model_conv_state_0
        shape: [18, 3, 6144]
        dtype: float16
        layers: [0, 1, 2, 4, ...]
```

Keys other runtimes do not know must be ignored. When `quant_ffn` (or `quant_lmhead`,
`quant_embeddings`) is present, the part uses linear weight quantization (`int8` or `int4,B`)
instead of a LUT and `lut_*` is `none`; it changes nothing at run time.

## Embeddings model

| | name | shape | type |
|---|---|---|---|
| input | `input_ids` | `[1, 1]` or `[1, batch_size]` (enumerated shapes) | int32 |
| output | `hidden_states` | `[1, T, hidden_size]` | fp16 |

A plain table lookup (no scaling).

## LM head model

| | name | shape | type |
|---|---|---|---|
| input | `hidden_states` | `[1, 1, hidden_size]` (the final-normed hidden state) | fp16 |
| outputs | `logits1` ... `logitsN` | `[1, 1, lm_head_chunk_sizes[i]]` | fp16 |

N = `split_lm_head`. The logits of the vocabulary are the concatenation of `logits1..logitsN`
in order (the first `vocab_size % N` parts are one wider when the split is uneven).

## Decoder chunks

The layers are split into `num_chunks` consecutive ranges (layer ranges are in the model
metadata `com.anemll.layers`). Every chunk is a multifunction model with two functions sharing
weights and states:

| function | `hidden_states` | `position_ids` | `causal_mask` | `current_pos` | output `output_hidden_states` |
|---|---|---|---|---|---|
| `infer` | `[1, 1, H]` fp16 | `[1]` int32 | `[1, 1, 1, ctx]` fp16 | `[1]` int32 | `[1, 1, H]` fp16 |
| `prefill` | `[1, B, H]` fp16 | `[B]` int32 | `[1, 1, B, ctx]` fp16 | `[1]` int32 | `[1, B, H]` (non-last chunk) or `[1, 1, H]` (last chunk) |

with `H = hidden_size`, `B = batch_size`, `ctx = context_length`, for a call that processes the
tokens at positions `p .. p+T-1`:

* `position_ids` = `[p, p+1, ..., p+T-1]` (RoPE positions), `current_pos` = `[p]` (where the
  KV cache is written);
* `causal_mask` = rows `p .. p+T-1` of the `ctx x ctx` causal mask: `0` where column <= row,
  `-inf` elsewhere;
* `hidden_states` of the first chunk = the embeddings output, of chunk i+1 = the output of
  chunk i (an internal representation: the residual stream).
* The last chunk's `infer` output is the final-normed hidden state of the token, ready for the
  LM head. The last chunk's `prefill` output is the final-normed hidden state of the **last**
  token of the batch (`prefill_last_token_output: true`); ANEMLL's own packages return an
  unnormalized first-token state there, which runtimes ignore.

### States

All chunks declare the same states with the full shapes (every chunk only touches the slices of
its own layers), so one state object serves the whole model: create it once with
`make_state()` on the first chunk (either function) and pass it to every `predict` call of
every chunk. A fresh state (all zeros) is an empty context.

* `model_model_kv_cache_0` `[2 * n_attn, n_kv_heads, ctx, head_dim]` fp16: keys of attention
  layer `i` (the i-th full attention layer, counting from 0) at index `i`, values at
  `n_attn + i`; position `p` lives at `[:, :, p, :]`. K is stored after its RMSNorm and RoPE.
  For qwen3 every layer is an attention layer (`n_attn = num_layers`, ANEMLL's layout).
* **extension** `model_model_gdn_state_0` `[n_linear * n_v_heads, d_k, d_v]` fp16: the
  recurrent memory of every Gated DeltaNet value head; linear layer `j` (the j-th linear
  attention layer) owns rows `j * n_v_heads .. (j+1) * n_v_heads - 1`. It holds
  `sqrt(d_k) * S` where `S` is the transformers state (`S[k, v]`), a rescaling that keeps the
  fp16 products on the Neural Engine out of the subnormal range.
* **extension** `model_model_conv_state_0` `[n_linear, K - 1, conv_dim]` fp16: the last `K-1`
  inputs of the causal depthwise convolution of layer `j` (the `in_proj_qkv` outputs, before
  the convolution), oldest first. Time-major on purpose: a trailing axis of 3 makes the ANE fail.

The names are the Core ML state names (`stateSchema` in the compiled model's `metadata.json`,
the same as ANEMLL's for the KV cache). Inference only passes the opaque state; the names matter
for reading or writing state contents (`MLState.withMultiArray(for:)`), e.g. to snapshot a
recurrent state.

## Call sequence

### Plain transformers (ANEMLL contract)

For a prompt of `n` tokens starting at position 0:

1. `p = 0`; while `p + B <= n`: embed tokens `p..p+B-1` (`[1, B]`), run `prefill` through
   all chunks, `p += B`.
2. While `p < n`: embed token `p` (`[1, 1]`), run `infer` through all chunks, `p += 1`.
3. Logits for the next token: LM head on the last chunk's output of the call that processed
   token `n-1`. (ANEMLL's runtimes instead re-run token `n-1` through `infer`, which is
   harmless with a KV cache: the same K/V are written again.)
4. Generation: for each new token at position `p`: embed, `infer` through all chunks, LM head,
   sample, `p += 1`. Stop at `ctx` (ANEMLL runtimes then shift the window and re-prefill).

Partial batches are not passed to `prefill` (they could write K/V beyond the end of the cache).

### Recurrent architectures (`recurrent_state: true`) — extension

The Gated DeltaNet state is updated by every call and cannot be rewound, so:

* **Every position is processed exactly once and in order.** In particular the ANEMLL habit of
  re-running the last prompt token through `infer` corrupts the state: take the logits from
  the call that processed token `n-1` (the `infer` output, or the last chunk's `prefill`
  output when the prompt ends on a batch boundary — `prefill_last_token_output`).
* **`prefill` only takes full batches** of `B` valid tokens (`prefill_full_batches_only`);
  padding tokens would be absorbed into the recurrent state. The remainder goes through
  `infer`.
* No rewinding or window shifting: to drop context (context full, edited history, rejected
  speculative tokens) create a new state and re-feed the tokens to keep, or restore a snapshot
  of all three states taken with `MLState.withMultiArray(for:)` at the position to return to.
* Positions still index the KV cache of the attention layers, so `ctx` remains the limit.

Everything else (inputs, masks, LM head) is as above.

## Architectures

### qwen3

Same graph as ANEMLL's Qwen3 converter: RMSNorm through `layer_norm` on `[x, -x]`, linear layers
as 1x1 convolutions, q/k RMSNorm then RoPE (rotate-half, full head dim), GQA by repeating the
cached K/V heads, softmax attention over the whole cache with the causal mask, SiLU MLP.
Differences: SiLU and sigmoid are written as `x * (0.5 + 0.5 tanh(x / 2))` because the ANE's
native sigmoid is a coarse approximation; the prefill output described above.

### qwen3_5 (Qwen3.5 / 3.6 / 3.8 dense)

* Zero-centered RMSNorm (`1 + w`, folded into the weights at conversion).
* Full attention layers (every `full_attention_interval`-th): `q_proj` produces query and a
  per-head output gate (split into two projections at conversion), q/k RMSNorm, partial RoPE
  (first `rot = head_dim * partial_rotary_factor` dims, rotate-half within them; for text the
  three M-RoPE sections share one position so it is plain RoPE, theta from `rope_parameters`),
  attention as qwen3, output multiplied by `sigmoid(gate)` before `o_proj`.
* Gated DeltaNet layers: `in_proj_qkv` -> causal depthwise conv (kernel `K`, state above) ->
  SiLU; q, k L2-normalized (repeated to `n_v_heads` when `n_v > n_k`), `beta = sigmoid(b)`,
  `g = -exp(A_log) * softplus(a + dt_bias)`; recurrence per value head
  `S = e^g S; S += k (beta (v - S^T k))^T; o = S^T q / sqrt(d_k)`; output
  `RMSNorm(o) * w * SiLU(z)` per head, `out_proj`.
  * `infer` runs the recurrence for one token.
  * `prefill` runs the chunked (WY) form of the same recurrence on chunks of
    `gdn_chunk_size` tokens (64 = the whole batch by default): decays from term-by-term sums
    of `g` (not differences of a long cumulative sum, which loses fp16 precision), the unit
    lower-triangular inverse by block doubling (`log2(C)` levels of two batched matmuls; the
    power series `(I - N)(I + N^2)...` overflows for correlated keys), then the usual
    `u = T (beta v)`, `w = T (beta k e^G)`, intra-chunk attention and state update. Every op is
    a matmul or elementwise op, all on the Neural Engine.
* MoE variants (`num_experts`) are not supported yet.

## Validation

`tools/coreml_check.py DIR --hf CHECKPOINT` greedy-generates with transformers (float32), feeds
the same tokens through the package (teacher forcing) and compares the logits at every generated
position; `--bench` measures prefill / decode speed, `--plan` lists the compute device of every
op (`MLComputePlan`).
