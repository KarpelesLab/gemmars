#!/usr/bin/env python3
"""Reference check against HuggingFace transformers (float32, CPU).

usage: ref_logits.py MODEL_DIR PROMPT_TEXT_FILE OURS_LOGITS.bin [--generate N]

PROMPT_TEXT_FILE holds the exact templated text (including <bos>) that gemmars tokenized.
Prints token ids, compares our logits with the reference and optionally greedy-generates.
"""
import sys, struct, time
import numpy as np
import torch

def main():
    model_dir, text_file, ours_file = sys.argv[1:4]
    gen = 0
    if "--generate" in sys.argv:
        gen = int(sys.argv[sys.argv.index("--generate") + 1])
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(text_file).read()
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    print("ref ids:", ids[0].tolist())
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    print(f"loaded {type(model).__name__} in {time.time()-t0:.1f}s")
    with torch.no_grad():
        out = model(input_ids=ids)
    ref = out.logits[0, -1].float().numpy()
    ours = np.fromfile(ours_file, dtype="<f4")
    print("vocab", ref.shape, ours.shape)
    diff = np.abs(ref - ours)
    print(f"max abs diff {diff.max():.5f}  mean abs diff {diff.mean():.6f}")
    rt = np.argsort(-ref)[:10]; ot = np.argsort(-ours)[:10]
    print("ref  top10:", [(int(i), round(float(ref[i]), 3)) for i in rt])
    print("ours top10:", [(int(i), round(float(ours[i]), 3)) for i in ot])
    print("argmax match:", int(rt[0]) == int(ot[0]))
    if gen:
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=gen, do_sample=False)
        new = g[0, ids.shape[1]:].tolist()
        print("ref greedy ids:", new)
        print("ref greedy text:", repr(tok.decode(new)))

if __name__ == "__main__":
    main()
