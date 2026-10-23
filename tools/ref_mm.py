#!/usr/bin/env python3
"""Multimodal reference: run the HF processor + model on an image and/or audio file and dump
the soft-token embeddings (after embed_vision / embed_audio) plus a greedy reply.

usage: ref_mm.py MODEL_DIR [--image F] [--audio F] [--prompt TEXT] [--out PREFIX] [--generate N]
Writes PREFIX_image.bin / PREFIX_audio.bin (f32 LE, [n_tokens, hidden]) and prints shapes,
pixel/mel feature stats, and the prompt token ids.
"""
import sys, argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--image")
ap.add_argument("--audio")
ap.add_argument("--prompt", default="Describe this.")
ap.add_argument("--out", default="/tmp/ref_mm")
ap.add_argument("--generate", type=int, default=0)
a = ap.parse_args()

from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained(a.model)
content = []
kwargs = {}
if a.image:
    from PIL import Image
    img = Image.open(a.image).convert("RGB")
    content.append({"type": "image"})
    kwargs["images"] = [img]
if a.audio:
    import wave
    w = wave.open(a.audio)
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if w.getnchannels() > 1:
        pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    assert w.getframerate() == 16000, "expects 16 kHz wav"
    content.append({"type": "audio"})
    kwargs["audio"] = [pcm]
content.append({"type": "text", "text": a.prompt})
messages = [{"role": "user", "content": content}]
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("template text:", repr(text[:200]))
inputs = proc(text=text, return_tensors="pt", **kwargs)
ids = inputs["input_ids"][0].tolist()
print("prompt ids:", len(ids), ids[:12], "...", ids[-8:])
print("image tokens:", sum(1 for t in ids if t == 258880), "audio tokens:", sum(1 for t in ids if t == 258881))
if "pixel_values" in inputs:
    pv = inputs["pixel_values"]
    pos = inputs["image_position_ids"]
    print("pixel_values", tuple(pv.shape), "min/max", float(pv.min()), float(pv.max()), "first", pv[0, 0, :6].tolist())
    print("positions first/last", pos[0, 0].tolist(), pos[0, -1].tolist(), "valid", int((pos[0] != -1).all(-1).sum()))
    np.asarray(pv[0], dtype="<f4").tofile(a.out + "_pixels.bin")
if "input_features" in inputs:
    feats = inputs["input_features"]
    print("input_features", tuple(feats.shape), "mask sum", int(inputs["input_features_mask"].sum()), "first", feats[0, 0, :5].tolist())
    np.asarray(feats[0], dtype="<f4").tofile(a.out + "_mel.bin")

model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32).eval()
with torch.no_grad():
    if "pixel_values" in inputs:
        feats = model.model.get_image_features(inputs["pixel_values"], inputs["image_position_ids"])
        emb = torch.cat(feats.pooler_output, dim=0).float().numpy()
        print("image soft tokens", emb.shape, "norm", float(np.linalg.norm(emb, axis=1).mean()), "first", emb[0, :5].tolist())
        emb.astype("<f4").tofile(a.out + "_image.bin")
    if "input_features" in inputs:
        ao = model.model.get_audio_features(inputs["input_features"], inputs["input_features_mask"])
        emb = ao.pooler_output[0][ao.attention_mask[0]].float().numpy()
        print("audio soft tokens", emb.shape, "norm", float(np.linalg.norm(emb, axis=1).mean()), "first", emb[0, :5].tolist())
        emb.astype("<f4").tofile(a.out + "_audio.bin")
    if a.generate:
        out = model.generate(**inputs, max_new_tokens=a.generate, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        print("greedy ids:", new)
        print("greedy text:", repr(proc.tokenizer.decode(new)))
