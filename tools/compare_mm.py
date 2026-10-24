#!/usr/bin/env python3
"""Compare oxinfer multimodal dumps (--dump-mm PREFIX) with tools/ref_mm.py dumps.
usage: compare_mm.py OURS_PREFIX REF_PREFIX HIDDEN
"""
import sys, numpy as np

ours, ref, hidden = sys.argv[1], sys.argv[2], int(sys.argv[3])

def load(path, cols):
    try:
        a = np.fromfile(path, dtype="<f4")
    except FileNotFoundError:
        return None
    return a.reshape(-1, cols)

def report(name, a, b):
    if a is None or b is None:
        print(f"{name}: missing ({'ours' if a is None else 'ref'})")
        return
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"{name}: row count ours {len(a)} vs ref {len(b)} (comparing first {n})")
    a, b = a[:n], b[:n]
    diff = np.abs(a - b)
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9)
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    print(f"{name}: rows {n} x {a.shape[1]} | max abs {diff.max():.4f} mean abs {diff.mean():.5f} | rel fro {rel:.4f} | cosine {cos:.6f} | ref scale {np.abs(b).mean():.4f}")
    if rel > 0.05:
        worst = diff.max(axis=1).argsort()[-3:]
        print("   worst rows:", worst.tolist(), "ours", a[worst[-1], :4].round(3).tolist(), "ref", b[worst[-1], :4].round(3).tolist())

pix_o = load(f"{ours}_pixels.bin", 768)
pix_r = load(f"{ref}_pixels.bin", 768)
if pix_o is not None and pix_r is not None:
    pix_r = pix_r[: len(pix_o)]  # reference is zero-padded to max_patches
report("pixel patches", pix_o, pix_r)
report("mel features", load(f"{ours}_mel.bin", 128), load(f"{ref}_mel.bin", 128))
report("image soft tokens", load(f"{ours}_image.bin", hidden), load(f"{ref}_image.bin", hidden))
report("audio soft tokens", load(f"{ours}_audio.bin", hidden), load(f"{ref}_audio.bin", hidden))
