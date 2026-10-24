#!/usr/bin/env python3
"""Reference check of the GGUF dequantization against gguf-py (`pip install gguf`).

usage: OXINFER_GGML_DUMP=DIR cargo test --release --lib dequantization_matches_ggml
       ggml_ref.py DIR

The test writes, for every type, the raw fixture blocks (TYPE.bin) and our dequantized values
(TYPE.ours, f32). This compares them bit for bit with `gguf.quants.dequantize` and prints the
FNV-1a hashes for `REF_HASHES` in src/ggml.rs.
"""
import sys, os
import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize


def fnv(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    d = sys.argv[1]
    ok = True
    for name in sorted(f[:-4] for f in os.listdir(d) if f.endswith(".bin")):
        raw = np.fromfile(os.path.join(d, name + ".bin"), dtype=np.uint8)
        ours = np.fromfile(os.path.join(d, name + ".ours"), dtype=np.float32)
        ref = dequantize(raw, T[name]).astype(np.float32).reshape(-1)
        same = ref.tobytes() == ours.tobytes()
        if not same:
            ok = False
            diff = np.nonzero(ref.view(np.uint32) != ours.view(np.uint32))[0]
            rel = np.max(np.abs(ref - ours) / np.maximum(np.abs(ref), 1e-30))
            print(f"# {name}: {len(diff)} of {len(ref)} differ (first at {diff[0]}: {ref[diff[0]]} vs {ours[diff[0]]}), max rel {rel:.2e}")
        print(f"        (GgmlType::{name}, 0x{fnv(ref.tobytes()):016x}),{'' if same else '  // MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
