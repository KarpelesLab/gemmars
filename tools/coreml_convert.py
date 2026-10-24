#!/usr/bin/env python3
"""Convert a Hugging Face checkpoint into an ANEMLL-format Core ML package for the Apple Neural Engine.

usage: coreml_convert.py --model DIR --output DIR [options]

The output directory follows ANEMLL's layout (https://github.com/Anemll/Anemll, 0.3.5) so that
ANEMLL-compatible runtimes (ANEMLL's tests/chat.py and Swift CLI, oxinfer's Core ML runtime) can
load it:

  meta.yaml                                    model description and runtime parameters
  <prefix>_embeddings[_lutN].mlmodelc          input_ids [1,1] or [1,batch] -> hidden_states
  <prefix>_FFN_PF[_lutN]_chunk_XXofYY.mlmodelc decoder layers; functions `infer` (one token) and
                                               `prefill` (batch tokens) sharing weights and the
                                               Core ML state (KV cache, recurrent state)
  <prefix>_lm_head[_lutN].mlmodelc             final hidden state -> logits1..logits16 (vocab split)
  tokenizer.json, tokenizer_config.json, ...   copied from the checkpoint

Supported architectures:
  qwen3    Qwen3 dense (Qwen3ForCausalLM, Qwen3-Embedding checkpoints): same graph as ANEMLL's
  qwen3_5  Qwen3.5 / 3.6 / 3.8 dense text models: Gated DeltaNet linear attention layers keep
           their recurrent and convolution state as Core ML states next to the KV cache of the
           gated full attention layers. See docs/coreml-format.md for the exact contract.

The graph is written for the ANE the way ANEMLL does it: fp16 everywhere, linear layers as 1x1
convolutions, RMSNorm through layer_norm on [x, -x], 4-D tensors, KV cache updated in place with a
dynamic slice at `current_pos`, LM head split 16 ways. Weights are palettized (k-means LUT,
per_grouped_channel with 8 output channels per LUT, as in ANEMLL's released models: 6 bits for the
decoder layers and the LM head, embeddings unquantized); `infer` and `prefill` see byte-identical
weights, so the merged package stores them once.

Requires: coremltools >= 8 (tested with 9.0 and torch 2.14; coremltools 9.0 needs numpy < 2.3),
torch, safetensors, numpy, scikit-learn (k-means),
Xcode command line tools (`xcrun coremlcompiler`). Run on macOS.

examples:
  coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3-0.6B --output out/qwen3-0.6b
  coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/q35 \\
      --context-length 1024 --ffn-quant lut6 --lmhead-quant lut6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float16
FORMAT_VERSION = "0.3.5"  # ANEMLL meta.yaml version this layout follows
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "tokenizer.model",
]


# ---------------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------------


@dataclass
class Spec:
    """Architecture hyper-parameters, normalized across config.json variants."""

    arch: str
    model_type: str
    hidden: int
    inter: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    eps: float
    theta: float
    rot_dim: int
    vocab: int
    tied: bool
    zero_centered: bool = False  # norm weights stored as w, applied as 1 + w
    attn_gate: bool = False  # q_proj also produces a per-head sigmoid output gate
    layer_types: list = field(default_factory=list)  # "full" | "linear"
    lin_nk: int = 0
    lin_nv: int = 0
    lin_dk: int = 0
    lin_dv: int = 0
    conv_k: int = 0

    @property
    def full_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "full"]

    @property
    def linear_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "linear"]

    @property
    def conv_dim(self):
        return 2 * self.lin_nk * self.lin_dk + self.lin_nv * self.lin_dv


def load_spec(cfg: dict) -> Spec:
    top_type = cfg.get("model_type", "")
    tc = cfg.get("text_config") or cfg
    mt = tc.get("model_type", top_type)
    tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
    n_heads = tc["num_attention_heads"]
    head_dim = tc.get("head_dim") or tc["hidden_size"] // n_heads
    common = dict(
        model_type=mt,
        hidden=tc["hidden_size"],
        inter=tc["intermediate_size"],
        n_layers=tc["num_hidden_layers"],
        n_heads=n_heads,
        n_kv=tc.get("num_key_value_heads", n_heads),
        head_dim=head_dim,
        eps=float(tc.get("rms_norm_eps", 1e-6)),
        vocab=tc["vocab_size"],
        tied=tied,
    )
    rp = tc.get("rope_parameters") or {}
    theta = float(rp.get("rope_theta", tc.get("rope_theta", 10000.0)))
    if mt in ("qwen3",):
        if tc.get("rope_scaling"):
            raise SystemExit("rope_scaling is not supported")
        return Spec(arch="qwen3", theta=theta, rot_dim=head_dim, layer_types=["full"] * common["n_layers"], **common)
    if mt in ("qwen3_5", "qwen3_5_text", "qwen3_next"):
        if tc.get("num_experts"):
            raise SystemExit("Qwen3.5 MoE checkpoints are not supported yet (dense only)")
        n = common["n_layers"]
        lt = tc.get("layer_types")
        if lt is None:
            k = tc.get("full_attention_interval", 4)
            lt = ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(n)]
        frac = float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
        return Spec(
            arch="qwen3_5",
            theta=theta,
            rot_dim=int(head_dim * frac),
            zero_centered=True,
            attn_gate=bool(tc.get("attn_output_gate", True)),
            layer_types=["full" if t == "full_attention" else "linear" for t in lt],
            lin_nk=tc["linear_num_key_heads"],
            lin_nv=tc["linear_num_value_heads"],
            lin_dk=tc["linear_key_head_dim"],
            lin_dv=tc["linear_value_head_dim"],
            conv_k=tc["linear_conv_kernel_dim"],
            **common,
        )
    raise SystemExit(f"unsupported model_type {mt!r} (supported: qwen3, qwen3_5)")


class Checkpoint:
    """Lazy access to safetensors tensors by their name without the model prefix."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self.files = {}
        names = {}
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                h = safe_open(os.path.join(path, f), framework="pt")
                self.files[f] = h
                for k in h.keys():
                    names[k] = f
        if not names:
            raise SystemExit(f"no .safetensors files in {path}")
        for prefix in ("model.language_model.", "model.", ""):
            if prefix + "embed_tokens.weight" in names:
                self.prefix = prefix
                break
        else:
            raise SystemExit("cannot find embed_tokens.weight in the checkpoint")
        self.names = names

    def has(self, name: str, raw=False) -> bool:
        return (name if raw else self.prefix + name) in self.names

    def get(self, name: str, raw=False) -> torch.Tensor:
        full = name if raw else self.prefix + name
        return self.files[self.names[full]].get_tensor(full).float()


# ---------------------------------------------------------------------------------------------
# ANE building blocks
# ---------------------------------------------------------------------------------------------


def conv1x1(w: torch.Tensor) -> nn.Conv2d:
    """A linear layer `y = W x` as a 1x1 convolution over [1, in, 1, T]."""
    m = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False, dtype=DTYPE)
    m.weight.data = w.to(DTYPE).view(w.shape[0], w.shape[1], 1, 1).contiguous()
    m.weight.requires_grad_(False)
    return m


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization over the last axis via layer_norm on [x, -x] (zero mean by
    construction), which the ANE evaluates more accurately than a hand-written rsqrt."""
    d = x.shape[-1]
    return F.layer_norm(torch.cat([x, -x], dim=-1), (2 * d,), eps=eps)[..., :d]


def sigmoid(x):
    """sigmoid through tanh: the ANE's own sigmoid / SiLU is a coarse approximation (errors up
    to ~0.015 on the Gated DeltaNet conv outputs), its tanh is accurate."""
    return 0.5 + 0.5 * torch.tanh(x * 0.5)


def silu(x):
    return x * sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, w: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(w.to(DTYPE), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return rms(x, self.eps) * self.weight


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Attention(nn.Module):
    """Full causal attention over the KV cache; optionally Qwen3.5's gated variant (sigmoid
    output gate from q_proj, partial RoPE)."""

    def __init__(self, sp: Spec, ck: Checkpoint, layer: int, slot: int):
        super().__init__()
        p = f"layers.{layer}.self_attn."
        nh, hd = sp.n_heads, sp.head_dim
        self.sp, self.slot = sp, slot
        wq = ck.get(p + "q_proj.weight")
        if sp.attn_gate:
            # q_proj rows are [query_h, gate_h] per head: split into two projections.
            wq = wq.view(nh, 2, hd, sp.hidden)
            self.g_proj = conv1x1(wq[:, 1].reshape(nh * hd, sp.hidden))
            wq = wq[:, 0].reshape(nh * hd, sp.hidden)
        self.q_proj = conv1x1(wq)
        self.k_proj = conv1x1(ck.get(p + "k_proj.weight"))
        self.v_proj = conv1x1(ck.get(p + "v_proj.weight"))
        self.o_proj = conv1x1(ck.get(p + "o_proj.weight"))
        off = 1.0 if sp.zero_centered else 0.0
        self.q_norm = RMSNorm(ck.get(p + "q_norm.weight") + off, sp.eps)
        self.k_norm = RMSNorm(ck.get(p + "k_norm.weight") + off, sp.eps)

    def rope(self, x, cos, sin):
        r = self.sp.rot_dim
        if r == x.shape[-1]:
            return x * cos + rotate_half(x) * sin
        xr, xp = x[..., :r], x[..., r:]
        return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)

    def forward(self, x, cos, sin, causal_mask, current_pos, kv_cache, n_full):
        """x: normed hidden [1, T, H]; cos/sin [1, 1, T, rot]; causal_mask [1, 1, T, ctx]."""
        sp = self.sp
        nh, nkv, hd = sp.n_heads, sp.n_kv, sp.head_dim
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)  # [1, H, 1, T]
        if t == 1:
            q = self.q_proj(hs).view(1, nh, 1, hd)
            k = self.k_proj(hs).view(1, nkv, 1, hd)
            v = self.v_proj(hs).view(1, nkv, 1, hd)
        else:
            q = self.q_proj(hs).view(1, nh, hd, t).permute(0, 1, 3, 2)
            k = self.k_proj(hs).view(1, nkv, hd, t).permute(0, 1, 3, 2)
            v = self.v_proj(hs).view(1, nkv, hd, t).permute(0, 1, 3, 2)
        q = self.rope(self.q_norm(q), cos, sin)
        k = self.rope(self.k_norm(k), cos, sin)

        s = self.slot
        kv_cache[s : s + 1, :, current_pos : current_pos + t, :] = k
        kv_cache[n_full + s : n_full + s + 1, :, current_pos : current_pos + t, :] = v
        kc = kv_cache[s : s + 1].squeeze(0)  # [nkv, ctx, hd]
        vc = kv_cache[n_full + s : n_full + s + 1].squeeze(0)
        rep = nh // nkv
        if rep > 1:
            kc = kc.unsqueeze(1).repeat(1, rep, 1, 1)
            vc = vc.unsqueeze(1).repeat(1, rep, 1, 1)
        kc = kc.view(1, nh, -1, hd)
        vc = vc.view(1, nh, -1, hd)

        att = torch.matmul(q, kc.transpose(-1, -2)) * (1.0 / math.sqrt(hd)) + causal_mask
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, vc)  # [1, nh, T, hd]
        o = o.permute(0, 1, 3, 2).reshape(1, nh * hd, 1, t)  # [1, nh*hd, 1, T]
        if sp.attn_gate:
            o = o * sigmoid(self.g_proj(hs))
        return self.o_proj(o).squeeze(2).permute(0, 2, 1)  # [1, T, H]


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear attention: causal depthwise conv + SiLU over [q, k, v], L2-normed q/k and
    the gated delta rule S <- a S + beta k (v - a S^T k)^T, o = S^T q, then a per-head RMSNorm
    gated by SiLU(z).

    States (shared by all layers of the model, indexed by this layer's slot):
      gdn_state  [n_lin * n_v, d_k, d_v]  sqrt(d_k) * S for every value head
      conv_state [n_lin, K - 1, conv_dim] the last K-1 conv inputs (pre-activation), oldest first
    One token runs the recurrence directly; a prefill batch uses the chunked (WY) form with a
    block-doubling triangular inverse, so everything is a matmul.

    fp16 scaling: with unit q, k and the 1/sqrt(d_k) attention scale folded into q, the products
    in q^T S and k^T S fall into fp16 subnormals (~1e-5), which the ANE flushes (the layer output
    was off by cosine 0.97). The state is therefore stored as S' = sqrt(d_k) S and the outer
    product uses k' = sqrt(d_k) k (= rms(k)); the output o' = d_k o only feeds the per-head
    RMSNorm, so it is used as is with eps * d_k^2.
    """

    def __init__(self, sp: Spec, ck: Checkpoint, layer: int, slot: int, chunk: int):
        super().__init__()
        p = f"layers.{layer}.linear_attn."
        self.sp, self.slot, self.chunk = sp, slot, chunk
        self.in_qkv = conv1x1(ck.get(p + "in_proj_qkv.weight"))
        self.in_z = conv1x1(ck.get(p + "in_proj_z.weight"))
        self.in_b = conv1x1(ck.get(p + "in_proj_b.weight"))
        self.in_a = conv1x1(ck.get(p + "in_proj_a.weight"))
        self.out_proj = conv1x1(ck.get(p + "out_proj.weight"))
        cw = ck.get(p + "conv1d.weight")  # [conv_dim, 1, K]
        self.conv = nn.Conv2d(sp.conv_dim, sp.conv_dim, (1, sp.conv_k), groups=sp.conv_dim, bias=False, dtype=DTYPE)
        self.conv.weight.data = cw.to(DTYPE).view(sp.conv_dim, 1, 1, sp.conv_k)
        self.conv.weight.requires_grad_(False)
        nv = sp.lin_nv
        self.register_buffer("a_neg", (-ck.get(p + "A_log").exp()).to(DTYPE).view(nv, 1))
        self.register_buffer("dt_bias", ck.get(p + "dt_bias").to(DTYPE).view(nv, 1))
        self.norm = RMSNorm(ck.get(p + "norm.weight"), sp.eps * sp.lin_dk**2)  # see o' above
        c = chunk
        i = torch.arange(c)
        lower = (i[:, None] >= i[None, :]).to(DTYPE)  # j <= i
        self.register_buffer("lower", lower)
        self.register_buffer("upper_neg", (1.0 - lower) * -30000.0)
        self.levels = int(math.log2(c))
        for lvl in range(self.levels):
            b = 1 << lvl  # lower-left b x b block of each 2b x 2b diagonal block
            m = (i[:, None] // (2 * b) == i[None, :] // (2 * b)) & (i[:, None] // b > i[None, :] // b)
            self.register_buffer(f"blk{lvl}", m.to(DTYPE))
        self.register_buffer("eye", torch.eye(c, dtype=DTYPE))
        # cum[t, i] = 1 if t <= i: (g @ cum)[i] = sum_{t<=i} g_t
        self.register_buffer("cum", lower.t().contiguous())
        # seg[t, i*c + j] = 1 if j < t <= i: (g @ seg)[i, j] = sum_{j<t<=i} g_t (exact log decay
        # from token j to token i, summed term by term instead of differencing a long cumsum)
        tt, ii, jj = torch.meshgrid(i, i, i, indexing="ij")
        self.register_buffer("seg", ((jj < tt) & (tt <= ii)).to(DTYPE).reshape(c, c * c))
        # rest[t, i] = 1 if t > i: (g @ rest)[i] = sum_{t>i} g_t (decay from i to chunk end)
        self.register_buffer("rest", (i[:, None] > i[None, :]).to(DTYPE))

    def forward(self, x, gdn_state, conv_state):
        sp = self.sp
        nk, nv, dk, dv, kd = sp.lin_nk, sp.lin_nv, sp.lin_dk, sp.lin_dv, sp.lin_nk * sp.lin_dk
        cd, kk = sp.conv_dim, sp.conv_k
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)  # [1, H, 1, T]
        qkv = self.in_qkv(hs)  # [1, conv_dim, 1, T]
        s = self.slot
        cs = conv_state[s : s + 1].permute(0, 2, 1).unsqueeze(2)  # [1, cd, 1, K-1]
        win = torch.cat([cs, qkv], dim=3)  # [1, cd, 1, K-1+T]
        conv_state[s : s + 1] = win[:, :, 0, t:].permute(0, 2, 1)
        c = silu(self.conv(win))  # [1, cd, 1, T]

        q = c[:, :kd].view(nk, dk, t).permute(0, 2, 1)  # [nk, T, dk]
        k = c[:, kd : 2 * kd].view(nk, dk, t).permute(0, 2, 1)
        v = c[:, 2 * kd :].view(nv, dv, t).permute(0, 2, 1)  # [nv, T, dv]
        # l2norm(x) = rms(x) / sqrt(d): q, k unit vectors, ks = sqrt(d_k) k (see class doc).
        rd = 1.0 / math.sqrt(dk)
        q = rms(q, 1e-6) * rd
        ks = rms(k, 1e-6)
        if nv != nk:
            rep = nv // nk
            q = q.unsqueeze(1).repeat(1, rep, 1, 1).view(nv, t, dk)
            ks = ks.unsqueeze(1).repeat(1, rep, 1, 1).view(nv, t, dk)
        k = ks * rd
        beta = sigmoid(self.in_b(hs).view(nv, t))
        g = self.a_neg * F.softplus(self.in_a(hs).view(nv, t) + self.dt_bias)  # log decay <= 0
        z = self.in_z(hs).view(nv, dv, t).permute(0, 2, 1)  # [nv, T, dv]

        st = gdn_state[s * nv : (s + 1) * nv]  # [nv, dk, dv] = sqrt(dk) S
        if t == 1:
            st = st * torch.exp(g).view(nv, 1, 1)
            mem = torch.matmul(k, st) * rd  # S^T k, [nv, 1, dv]
            delta = (v - mem) * beta.view(nv, 1, 1)
            st = st + torch.matmul(ks.transpose(1, 2), delta)
            o = torch.matmul(q, st)  # d_k S^T q_scaled
        else:
            outs = []
            for c0 in range(0, t, self.chunk):
                sl = slice(c0, c0 + self.chunk)
                o_c, st = self.chunk_step(q[:, sl], k[:, sl], ks[:, sl], v[:, sl], g[:, sl], beta[:, sl], st)
                outs.append(o_c)
            o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        gdn_state[s * nv : (s + 1) * nv] = st

        o = self.norm(o) * silu(z)  # [nv, T, dv]
        o = o.permute(0, 2, 1).reshape(1, nv * dv, 1, t)
        return self.out_proj(o).squeeze(2).permute(0, 2, 1)  # [1, T, H]

    def chunk_step(self, q, k, ks, v, g, beta, st):
        """Chunked gated delta rule over C tokens (unit q, k and ks = sqrt(d_k) k [nv, C, dk],
        v [nv, C, dv], g, beta [nv, C], scaled state [nv, dk, dv]); returns (o', new state)."""
        nv, c = g.shape
        rd = 1.0 / math.sqrt(k.shape[-1])
        gc = torch.matmul(g, self.cum)  # [nv, C] cumulative log decay up to i
        seg = torch.matmul(g, self.seg).view(nv, c, c)  # log decay j -> i (0 above diagonal)
        dec = torch.exp(seg + self.upper_neg)  # [nv, C, C], zero above the diagonal
        kb = k * beta.unsqueeze(-1)
        n = torch.matmul(kb, k.transpose(1, 2)) * dec  # (I + strictly lower part)^-1 below
        # Unit lower-triangular inverse by block doubling: with X the inverse of the b-sized
        # diagonal blocks, X - X L X (L restricted to the lower-left b x b block of every 2b
        # block) inverts the 2b blocks. log2(C) levels of two matmuls, stable unlike the
        # power series (I - N)(I + N^2)(I + N^4)... which overflows for correlated keys.
        tinv = self.eye
        for lvl in range(self.levels):
            off = n * getattr(self, f"blk{lvl}")
            tinv = tinv - torch.matmul(torch.matmul(tinv, off), tinv)
        u = torch.matmul(tinv, v * beta.unsqueeze(-1))  # [nv, C, dv]
        w = torch.matmul(tinv, kb * torch.exp(gc).unsqueeze(-1))  # [nv, C, dk]
        v_new = u - torch.matmul(w, st) * rd
        att = torch.matmul(q, ks.transpose(1, 2)) * dec  # causal (incl. diagonal)
        o = torch.matmul(q * torch.exp(gc).unsqueeze(-1), st) + torch.matmul(att, v_new)
        g_last = gc[:, c - 1 :]  # [nv, 1]
        k_dec = ks * torch.exp(torch.matmul(g, self.rest)).unsqueeze(-1)  # decay i -> chunk end
        st = st * torch.exp(g_last).unsqueeze(-1) + torch.matmul(k_dec.transpose(1, 2), v_new)
        return o, st


class MLP(nn.Module):
    def __init__(self, ck: Checkpoint, layer: int):
        super().__init__()
        p = f"layers.{layer}.mlp."
        self.gate_proj = conv1x1(ck.get(p + "gate_proj.weight"))
        self.up_proj = conv1x1(ck.get(p + "up_proj.weight"))
        self.down_proj = conv1x1(ck.get(p + "down_proj.weight"))

    def forward(self, x):
        hs = x.permute(0, 2, 1).unsqueeze(2)
        y = self.down_proj(silu(self.gate_proj(hs)) * self.up_proj(hs))
        return y.squeeze(2).permute(0, 2, 1)


class Layer(nn.Module):
    def __init__(self, sp: Spec, ck: Checkpoint, i: int, gdn_chunk: int):
        super().__init__()
        off = 1.0 if sp.zero_centered else 0.0
        self.kind = sp.layer_types[i]
        self.input_layernorm = RMSNorm(ck.get(f"layers.{i}.input_layernorm.weight") + off, sp.eps)
        self.post_attention_layernorm = RMSNorm(ck.get(f"layers.{i}.post_attention_layernorm.weight") + off, sp.eps)
        if self.kind == "full":
            self.self_attn = Attention(sp, ck, i, sp.full_layers.index(i))
        else:
            self.linear_attn = GatedDeltaNet(sp, ck, i, sp.linear_layers.index(i), gdn_chunk)
        self.mlp = MLP(ck, i)


class Decoder(nn.Module):
    """The decoder layers of one chunk plus every Core ML state of the model (all chunks declare
    the full state so a single state object is shared by every chunk)."""

    def __init__(self, sp: Spec, ck: Checkpoint, start: int, end: int, ctx: int, gdn_chunk: int):
        super().__init__()
        self.sp, self.start, self.end, self.ctx = sp, start, end, ctx
        self.layers = nn.ModuleList([Layer(sp, ck, i, gdn_chunk) for i in range(start, end)])
        self.last = end == sp.n_layers
        if self.last:
            off = 1.0 if sp.zero_centered else 0.0
            self.norm = RMSNorm(ck.get("norm.weight") + off, sp.eps)
        inv = 1.0 / (sp.theta ** (torch.arange(0, sp.rot_dim, 2).float() / sp.rot_dim))
        freqs = torch.outer(torch.arange(ctx).float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_t", emb.cos().to(DTYPE))
        self.register_buffer("sin_t", emb.sin().to(DTYPE))
        for name, shape in state_shapes(sp, ctx).items():
            self.register_buffer(name, torch.zeros(shape, dtype=DTYPE))

    def forward(self, x, position_ids, causal_mask, current_pos, prefill: bool):
        t = x.shape[1]
        cos = self.cos_t[position_ids].view(1, 1, t, -1)
        sin = self.sin_t[position_ids].view(1, 1, t, -1)
        n_full = len(self.sp.full_layers)
        for layer in self.layers:
            h = layer.input_layernorm(x)
            if layer.kind == "full":
                h = layer.self_attn(h, cos, sin, causal_mask, current_pos, self.kv_cache_0, n_full)
            else:
                h = layer.linear_attn(h, self.gdn_state_0, self.conv_state_0)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Core ML rejects a declared state the graph never reads: touch the ones this chunk does
        # not use (e.g. the KV cache in a chunk of linear attention layers only).
        kinds = {layer.kind for layer in self.layers}
        unused = [n for n in state_shapes(self.sp, self.ctx) if ("kv" in n and "full" not in kinds) or ("kv" not in n and "linear" not in kinds)]
        for n in unused:
            st = getattr(self, n)
            x = x + st[(0,) * (st.dim() - 1)][0:1].view(1, 1, 1) * 0.0
        if self.last:
            # Final norm of the last token only: the prefill output gives the logits of the next
            # token when a prompt ends on a batch boundary.
            x = self.norm(x[:, t - 1 :, :])
        return x


def state_shapes(sp: Spec, ctx: int) -> dict:
    """Core ML states of the model (buffer name -> shape), all fp16."""
    out = {}
    nf = len(sp.full_layers)
    if nf:
        out["kv_cache_0"] = (2 * nf, sp.n_kv, ctx, sp.head_dim)
    nl = len(sp.linear_layers)
    if nl:
        out["gdn_state_0"] = (nl * sp.lin_nv, sp.lin_dk, sp.lin_dv)
        # time-major: a last axis of K-1 = 3 elements makes the ANE fail at run time
        out["conv_state_0"] = (nl, sp.conv_k - 1, sp.conv_dim)
    return out


class Root(nn.Module):
    """Holds the decoder at `model.model` so the Core ML state names match ANEMLL's
    (`model_model_kv_cache_0` once Core ML sanitizes `model.model.kv_cache_0`)."""

    def __init__(self, dec: Decoder):
        super().__init__()
        self.model = dec


class ChunkFn(nn.Module):
    def __init__(self, dec: Decoder, prefill: bool):
        super().__init__()
        self.model = Root(dec)
        self.prefill = prefill

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        return self.model.model(hidden_states, position_ids, causal_mask, current_pos, self.prefill)


class Embeddings(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.embed_tokens = nn.Embedding(w.shape[0], w.shape[1], dtype=DTYPE)
        self.embed_tokens.weight.data = w.to(DTYPE)
        self.embed_tokens.weight.requires_grad_(False)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LMHead(nn.Module):
    """Vocabulary split into `n` 1x1 convolutions (ANE output width limit), one output each."""

    def __init__(self, w, n: int):
        super().__init__()
        v = w.shape[0]
        self.sizes = [v // n + (1 if i < v % n else 0) for i in range(n)]
        self.heads = nn.ModuleList([conv1x1(p) for p in torch.split(w, self.sizes)])

    def forward(self, hidden_states):
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        return tuple(h(hs).squeeze(2).transpose(1, 2) for h in self.heads)


# ---------------------------------------------------------------------------------------------
# Core ML conversion
# ---------------------------------------------------------------------------------------------


@dataclass
class Quant:
    """Weight compression: none, lutN[,G] (k-means palettization, G output channels per LUT,
    0 = per tensor), int8 (per-channel linear) or int4[,B] (per-block linear, block B)."""

    kind: str = "none"
    bits: int = 0
    group: int = 0

    @staticmethod
    def parse(s: str) -> "Quant":
        s = s.strip().lower()
        if s in ("none", "fp16", ""):
            return Quant()
        if s.startswith("lut"):
            b, _, g = s[3:].partition(",")
            return Quant("lut", int(b), int(g) if g else 8)
        if s.startswith("int8"):
            return Quant("int8", 8, 0)
        if s.startswith("int4"):
            _, _, b = s.partition(",")
            return Quant("int4", 4, int(b) if b else 32)
        raise argparse.ArgumentTypeError(f"bad quantization {s!r}")

    @property
    def suffix(self):
        return {"none": "", "lut": f"_lut{self.bits}", "int8": "_int8", "int4": "_int4"}[self.kind]

    @property
    def meta(self):
        return self.bits if self.kind == "lut" else "none"

    def __str__(self):
        if self.kind == "lut":
            return f"lut{self.bits},{self.group}"
        if self.kind == "int4":
            return f"int4,{self.group}"
        return self.kind


def compress(mlmodel, q: Quant, workers: int):
    import coremltools.optimize as cto

    if q.kind == "none":
        return mlmodel
    if q.kind == "lut":
        # coremltools 9.0 closes its shared k-means pool after a call without clearing it.
        from coremltools.optimize.coreml import _quantization_passes as qp

        if qp.palettize_weights._compress_pool is not None:
            qp.palettize_weights._compress_pool.terminate()
            qp.palettize_weights._compress_pool = None
        if q.group > 0:
            op = cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=q.bits, granularity="per_grouped_channel", group_size=q.group, num_kmeans_workers=workers
            )
        else:
            op = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=q.bits, granularity="per_tensor", num_kmeans_workers=workers)
        cfg = cto.coreml.OptimizationConfig(global_config=op)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cto.coreml.palettize_weights(mlmodel, cfg)
    if q.kind == "int8":
        op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    else:
        op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=q.group)
    return cto.coreml.linear_quantize_weights(mlmodel, cto.coreml.OptimizationConfig(global_config=op))


def convert(module, example, inputs, outputs, states, enumerate_input=None):
    import coremltools as ct

    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states or None,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        skip_model_load=True,
    )


def reset_states(dec: Decoder):
    with torch.no_grad():
        for name in state_shapes(dec.sp, dec.ctx):
            getattr(dec, name).zero_()


def ct_states(sp: Spec, ctx: int):
    import coremltools as ct

    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=shape, dtype=np.float16), name=f"model.model.{name}")
        for name, shape in state_shapes(sp, ctx).items()
    ]


def set_meta(mlmodel, desc: str, extra: dict):
    mlmodel.author = "oxinfer coreml_convert.py (ANEMLL format)"
    mlmodel.short_description = desc
    for k, v in extra.items():
        mlmodel.user_defined_metadata[k] = str(v)


def compile_package(pkg: str, out_dir: str) -> str:
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir], check=True, stdout=subprocess.DEVNULL)
    return os.path.join(out_dir, os.path.basename(pkg)[: -len(".mlpackage")] + ".mlmodelc")


def chunk_ranges(n_layers: int, chunks: int):
    base, rem = divmod(n_layers, chunks)
    out, s = [], 0
    for i in range(chunks):
        e = s + base + (1 if i < rem else 0)
        out.append((s, e))
        s = e
    return out


def bits_per_weight(q: Quant) -> float:
    return {"none": 16, "lut": q.bits, "int8": 8, "int4": 4}[q.kind]


def auto_chunks(sp: Spec, q: Quant, max_mb: float) -> int:
    h, i = sp.hidden, sp.inter
    per_layer = 3 * h * i
    for t in sp.layer_types:
        if t == "full":
            qd = sp.n_heads * sp.head_dim * (2 if sp.attn_gate else 1)
            per_layer_mix = h * (qd + 2 * sp.n_kv * sp.head_dim) + sp.n_heads * sp.head_dim * h
        else:
            vd = sp.lin_nv * sp.lin_dv
            per_layer_mix = h * (sp.conv_dim + vd + 2 * sp.lin_nv) + vd * h
        per_layer += per_layer_mix / len(sp.layer_types)
    total_mb = per_layer * sp.n_layers * bits_per_weight(q) / 8 / 1e6 * 1.1
    return max(1, math.ceil(total_mb / max_mb))


def write_meta(path, args, sp: Spec, names: dict, chunks: int, lm_sizes, gen_cfg: dict):
    a = args
    lines = [
        "model_info:",
        f"  name: anemll-{a.name}-ctx{a.context_length}",
        f"  version: {FORMAT_VERSION}",
        "  description: |",
        f"    {a.name} on the Apple Neural Engine (converted by oxinfer tools/coreml_convert.py)",
        f"    Context length: {a.context_length}",
        f"    Batch size: {a.batch_size}",
        f"    Chunks: {chunks}",
        "  license: MIT",
        "  author: oxinfer",
        "  framework: Core ML",
        "  language: Python",
        f"  architecture: {sp.arch}",
        "  parameters:",
        f"    context_length: {a.context_length}",
        f"    batch_size: {a.batch_size}",
        f"    lut_embeddings: {a.embed_quant.meta}",
    ]
    for key, q in (("embeddings", a.embed_quant), ("ffn", a.ffn_quant), ("lmhead", a.lmhead_quant)):
        if key != "embeddings":
            lines.append(f"    lut_{key}: {q.meta}")
        if q.kind == "lut":
            lines.append(f"    lut_{key}_per_channel: {q.group}")
        elif q.kind != "none":
            lines.append(f"    quant_{key}: {q}")
    lines += [
        f"    num_chunks: {chunks}",
        f"    model_prefix: {a.prefix}",
        f"    embeddings: {names['embeddings']}",
        f"    lm_head: {names['lm_head']}",
        f"    ffn: {names['ffn']}",
        f"    split_lm_head: {len(lm_sizes)}",
        f"    vocab_size: {sp.vocab}",
        f"    lm_head_chunk_sizes: [{', '.join(str(s) for s in lm_sizes)}]",
        "    prefill_dynamic_slice: true",
        "    prefill_last_token_output: true",
        f"    hidden_size: {sp.hidden}",
        f"    num_layers: {sp.n_layers}",
    ]
    if gen_cfg.get("temperature") is not None:
        lines += [
            "    recommended_sampling:",
            f"      do_sample: {str(bool(gen_cfg.get('do_sample', True))).lower()}",
            f"      temperature: {gen_cfg.get('temperature')}",
            f"      top_p: {gen_cfg.get('top_p', 1.0)}",
            f"      top_k: {gen_cfg.get('top_k', 0)}",
        ]
    if sp.arch != "qwen3":
        # oxinfer extensions (docs/coreml-format.md)
        lines += [
            "    recurrent_state: true",
            "    prefill_full_batches_only: true",
            f"    gdn_chunk_size: {a.gdn_chunk}",
            f"    layer_types: [{', '.join(sp.layer_types)}]",
            "    states:",
        ]
        for name, shape in state_shapes(sp, a.context_length).items():
            lines.append(f"      - name: model_model_{name}")
            lines.append(f"        shape: [{', '.join(str(s) for s in shape)}]")
            lines.append("        dtype: float16")
            if name == "kv_cache_0":
                lines.append(f"        layers: [{', '.join(str(i) for i in sp.full_layers)}]")
            else:
                lines.append(f"        layers: [{', '.join(str(i) for i in sp.linear_layers)}]")
    lines += [
        "",
        "# Conversion parameters",
        f"# source: {a.model}",
        f"# command: {' '.join(sys.argv)}",
        f"# ffn: {a.ffn_quant}  lm_head: {a.lmhead_quant}  embeddings: {a.embed_quant}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def copy_tokenizer(src: str, dst: str, cfg: dict):
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    # config.json: the source config plus tokenizer_class (swift-transformers reads it).
    out = dict(cfg)
    tc_path = os.path.join(src, "tokenizer_config.json")
    if os.path.exists(tc_path):
        tcls = json.load(open(tc_path)).get("tokenizer_class")
        if tcls:
            out["tokenizer_class"] = tcls
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="Hugging Face checkpoint directory (config.json + safetensors)")
    ap.add_argument("--output", required=True, help="output directory (created)")
    ap.add_argument("--name", help="model name for meta.yaml (default: checkpoint directory name)")
    ap.add_argument("--prefix", help="file name prefix (default: qwen for qwen3, qwen35 for qwen3_5)")
    ap.add_argument("--context-length", type=int, default=512, help="KV cache length / max positions (default 512)")
    ap.add_argument("--batch-size", type=int, default=64, help="prefill batch (tokens per prefill call, default 64)")
    ap.add_argument("--chunks", default="auto", help="number of FFN chunks or 'auto' (<= --max-chunk-mb each)")
    ap.add_argument("--max-chunk-mb", type=float, default=950, help="chunk size target for --chunks auto (default 950)")
    ap.add_argument(
        "--ffn-quant",
        type=Quant.parse,
        default=Quant.parse("lut6"),
        help="decoder layers: none|lutN[,G]|int8|int4[,B] (default lut6 = lut6,8, what ANEMLL recommends and ships; lut4 is ~1.5x smaller but loses ~20 points of top-1 agreement on <1B models)",
    )
    ap.add_argument("--lmhead-quant", type=Quant.parse, default=Quant.parse("lut6"), help="LM head (default lut6)")
    ap.add_argument("--embed-quant", type=Quant.parse, default=Quant.parse("none"), help="embeddings (default none)")
    ap.add_argument("--no-quant", action="store_true", help="fp16 everywhere (same as all three = none)")
    ap.add_argument("--split-lm-head", type=int, default=16, help="LM head output splits (default 16)")
    ap.add_argument("--gdn-chunk", type=int, default=0, help="Gated DeltaNet prefill chunk (tokens, power of 2; default = batch size)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="k-means worker processes")
    ap.add_argument("--keep-packages", action="store_true", help="also keep the .mlpackage files")
    ap.add_argument("--parts", default="embeddings,ffn,lm_head", help="comma list of parts to (re)build")
    args = ap.parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")  # k-means worker processes
    if args.no_quant:
        args.ffn_quant = args.lmhead_quant = args.embed_quant = Quant()

    import coremltools as ct

    src = os.path.abspath(os.path.expanduser(args.model))
    cfg = json.load(open(os.path.join(src, "config.json")))
    sp = load_spec(cfg)
    args.model = src
    args.name = args.name or os.path.basename(src.rstrip("/")).replace("--", "-")
    args.prefix = args.prefix or {"qwen3": "qwen", "qwen3_5": "qwen35"}[sp.arch]
    args.gdn_chunk = args.gdn_chunk or args.batch_size
    if args.gdn_chunk & (args.gdn_chunk - 1) or args.batch_size % args.gdn_chunk:
        raise SystemExit("--gdn-chunk must be a power of two dividing --batch-size")
    ctx, bs = args.context_length, args.batch_size
    chunks = auto_chunks(sp, args.ffn_quant, args.max_chunk_mb) if args.chunks == "auto" else int(args.chunks)
    parts = set(args.parts.split(","))
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="coreml_convert_", dir=out)
    ck = Checkpoint(src)
    print(f"{sp.arch}: {sp.n_layers} layers ({len(sp.full_layers)} full attention), hidden {sp.hidden}, vocab {sp.vocab}; "
          f"ctx {ctx}, batch {bs}, {chunks} chunk(s); ffn {args.ffn_quant}, lm_head {args.lmhead_quant}, embeddings {args.embed_quant}")
    common_meta = {
        "com.anemll.context_length": ctx,
        "com.anemll.batch_size": bs,
        "com.anemll.num_chunks": chunks,
        "com.anemll.architecture": sp.arch,
        "com.anemll.info": f"oxinfer coreml_convert.py from {os.path.basename(src)}",
    }

    def finish(mlmodel, base: str, q: Quant, desc: str, extra: dict):
        set_meta(mlmodel, desc, {**common_meta, **extra, "com.anemll.lut_bits": q.meta})
        pkg = os.path.join(tmp, base + ".mlpackage")
        mlmodel.save(pkg)
        dst = os.path.join(out, base + ".mlmodelc")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        compile_package(pkg, out)
        if args.keep_packages:
            final = os.path.join(out, base + ".mlpackage")
            if os.path.exists(final):
                shutil.rmtree(final)
            shutil.move(pkg, final)
        return base + ".mlmodelc"

    names = {}
    t0 = time.time()
    # Embeddings
    emb_w = ck.get("embed_tokens.weight")
    base = f"{args.prefix}_embeddings{args.embed_quant.suffix}"
    names["embeddings"] = base + ".mlmodelc"
    if "embeddings" in parts:
        print("embeddings ...")
        shape = ct.EnumeratedShapes(shapes=[[1, 1], [1, bs]], default=[1, 1])
        m = convert(
            Embeddings(emb_w),
            (torch.zeros((1, 1), dtype=torch.int32),),
            [ct.TensorType(name="input_ids", shape=shape, dtype=np.int32)],
            [ct.TensorType(name="hidden_states", dtype=np.float16)],
            None,
        )
        m = compress(m, args.embed_quant, args.workers)
        finish(m, base, args.embed_quant, "token embeddings", {})

    # LM head
    if sp.tied or not ck.has("lm_head.weight", raw=True):
        head_w = emb_w
    else:
        head_w = ck.get("lm_head.weight", raw=True)
    head = LMHead(head_w, args.split_lm_head)
    base = f"{args.prefix}_lm_head{args.lmhead_quant.suffix}"
    names["lm_head"] = base + ".mlmodelc"
    if "lm_head" in parts:
        print("lm_head ...")
        m = convert(
            head,
            (torch.zeros((1, 1, sp.hidden), dtype=DTYPE),),
            [ct.TensorType(name="hidden_states", shape=(1, 1, sp.hidden), dtype=np.float16)],
            [ct.TensorType(name=f"logits{i + 1}", dtype=np.float16) for i in range(args.split_lm_head)],
            None,
        )
        m = compress(m, args.lmhead_quant, args.workers)
        finish(m, base, args.lmhead_quant, "LM head (vocabulary split)", {"com.anemll.split_lm_head": args.split_lm_head})
    del emb_w, head_w

    # Decoder chunks: infer and prefill are converted and compressed separately, then merged into
    # one multifunction package. Both see byte-identical fp16 weights and k-means is seeded
    # (random_state=0), so the compressed weights are identical too and stored once.
    for ci, (s, e) in enumerate(chunk_ranges(sp.n_layers, chunks)):
        base = f"{args.prefix}_FFN_PF{args.ffn_quant.suffix}_chunk_{ci + 1:02d}of{chunks:02d}"
        if ci == 0:
            names["ffn"] = base + ".mlmodelc"
        if "ffn" not in parts:
            continue
        print(f"chunk {ci + 1}/{chunks}: layers {s}..{e - 1}")
        dec = Decoder(sp, ck, s, e, ctx, args.gdn_chunk)
        fn_pkgs = {}
        for fn, t in (("infer", 1), ("prefill", bs)):
            ex = (
                torch.zeros((1, t, sp.hidden), dtype=DTYPE),
                torch.arange(t, dtype=torch.int32),
                torch.zeros((1, 1, t, ctx), dtype=DTYPE),
                torch.zeros((1,), dtype=torch.int32),
            )
            reset_states(dec)
            m = convert(
                ChunkFn(dec, fn == "prefill"),
                ex,
                [
                    ct.TensorType(name="hidden_states", shape=(1, t, sp.hidden), dtype=np.float16),
                    ct.TensorType(name="position_ids", shape=(t,), dtype=np.int32),
                    ct.TensorType(name="causal_mask", shape=(1, 1, t, ctx), dtype=np.float16),
                    ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
                ],
                [ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                ct_states(sp, ctx),
            )
            reset_states(dec)
            if args.ffn_quant.kind != "none":
                print(f"  {fn}: compressing ({args.ffn_quant}) ...")
                m = compress(m, args.ffn_quant, args.workers)
            fn_pkgs[fn] = os.path.join(tmp, f"{base}_{fn}.mlpackage")
            m.save(fn_pkgs[fn])
            del m
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(fn_pkgs["infer"], "main", "infer")
        desc.add_function(fn_pkgs["prefill"], "main", "prefill")
        desc.default_function_name = "infer"
        merged = os.path.join(tmp, f"{base}_merged.mlpackage")
        ct.utils.save_multifunction(desc, merged)
        m = ct.models.MLModel(merged, skip_model_load=True)
        finish(
            m,
            base,
            args.ffn_quant,
            f"decoder layers {s}..{e - 1} (functions infer, prefill)",
            {"com.anemll.chunk_no": ci + 1, "com.anemll.layers": f"{s}-{e - 1}"},
        )
        del dec

    lm_sizes = head.sizes
    gen_cfg = {}
    if os.path.exists(os.path.join(src, "generation_config.json")):
        gen_cfg = json.load(open(os.path.join(src, "generation_config.json")))
    write_meta(os.path.join(out, "meta.yaml"), args, sp, names, chunks, lm_sizes, gen_cfg)
    copy_tokenizer(src, out, cfg)
    shutil.rmtree(tmp)
    print(f"done in {time.time() - t0:.0f}s: {out}")


if __name__ == "__main__":
    main()
