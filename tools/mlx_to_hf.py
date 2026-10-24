#!/usr/bin/env python3
"""Dequantizes an MLX checkpoint into a transformers-loadable text-only float32 checkpoint.

usage: mlx_to_hf.py MLX_DIR OUT_DIR

Affine-quantized matrices (`.weight` uint32 + `.scales` + `.biases`) become `scale * q + bias`
in float32, so a transformers run on OUT_DIR is the exact reference for the MLX weights. Only
the language model is kept (renamed to `model.*`), with a text config, so AutoModelForCausalLM
loads Gemma 3 / Gemma 4 as their *ForCausalLM text models.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

TEXT_PREFIXES = ("language_model.model.", "model.language_model.", "model.")


def text_name(n):
    if n.startswith("language_model.lm_head.") or n.startswith("lm_head."):
        return "lm_head." + n.split("lm_head.", 1)[1]
    for p in TEXT_PREFIXES:
        if n.startswith(p):
            rest = n[len(p):]
            if p == "model." and rest.split(".")[0] in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector", "language_model"):
                return None
            return "model." + rest
    return None


def main():
    src, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    q = cfg.get("quantization") or {}
    bits, group = q.get("bits", 4), q.get("group_size", 64)
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    tensors = {}
    for f in files:
        tensors.update(load_file(os.path.join(src, f)))
    result = {}
    for name, t in tensors.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        hn = text_name(name)
        if hn is None:
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if t.dtype == torch.uint32 and base + ".scales" in tensors:
            per = 32 // bits
            shifts = torch.arange(per, dtype=torch.int64) * bits
            qv = ((t.to(torch.int64)[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(t.shape[0], -1).float()
            s = tensors[base + ".scales"].float()
            b = tensors[base + ".biases"].float()
            g = qv.shape[1] // s.shape[1]
            w = qv.reshape(t.shape[0], s.shape[1], g) * s[:, :, None] + b[:, :, None]
            result[hn] = w.reshape(t.shape[0], -1).contiguous()
        else:
            result[hn] = t.float().contiguous()
    save_file(result, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    text = dict(cfg.get("text_config", cfg))
    for k in ("quantization", "quantization_config"):
        text.pop(k, None)
        cfg.pop(k, None)
    mt = cfg.get("model_type", "")
    if mt in ("gemma3", "gemma4"):
        text["model_type"] = mt + "_text"
        text["architectures"] = [("Gemma3" if mt == "gemma3" else "Gemma4") + "ForCausalLM"]
        text.setdefault("torch_dtype", "float32")
        if mt == "gemma3":
            text.setdefault("vocab_size", result["model.embed_tokens.weight"].shape[0])
        cfg = text
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    for f in os.listdir(src):
        if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja", "generation_config.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.json"):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    print(f"wrote {len(result)} tensors to {out}")


if __name__ == "__main__":
    main()
