#!/usr/bin/env python3
"""Reference drafting check: run the main model on a prompt, then the Gemma 4 assistant drafter
for K steps exactly as transformers' candidate generator does, and print the drafted tokens.
usage: ref_draft.py MODEL_DIR DRAFTER_DIR PROMPT_TEXT_FILE [K]
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma4AssistantForCausalLM

model_dir, draft_dir, text_file = sys.argv[1:4]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
tok = AutoTokenizer.from_pretrained(model_dir)
text = open(text_file).read()
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
print("prompt ids:", ids[0].tolist())
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
drafter = Gemma4AssistantForCausalLM.from_pretrained(draft_dir, dtype=torch.float32).eval()
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True, return_shared_kv_states=True, use_cache=True)
    t0 = int(out.logits[0, -1].argmax())
    print("first token (main argmax):", t0)
    last_hidden = out.hidden_states[-1][:, -1:]
    print("hidden_states[-1] last-token norm:", float(last_hidden.norm()), "first vals", last_hidden[0, 0, :4].tolist())
    skv = out.shared_kv_states
    print("shared kv keys:", {k: tuple(v[0].shape) for k, v in skv.items()})
    L = ids.shape[1]
    emb = model.get_input_embeddings()
    last_token = torch.tensor([[t0]])
    position_ids = torch.tensor([[L]])
    drafts = []
    for i in range(K):
        e = emb(last_token)
        if i == 0:
            print("embedding scale check: emb norm", float(e.norm()), "first vals", e[0, 0, :4].tolist())
        x = torch.cat([e, last_hidden], dim=-1)
        o = drafter(inputs_embeds=x, position_ids=position_ids, shared_kv_states=skv, use_cache=False)
        last_token = o.logits.argmax(dim=-1)
        last_hidden = o.last_hidden_state
        drafts.append(int(last_token))
        if i == 0:
            top = torch.topk(o.logits[0, -1], 5)
            print("drafter step0 top5:", top.indices.tolist(), [round(float(v), 3) for v in top.values])
            print("drafter step0 new hidden first vals", last_hidden[0, 0, :4].tolist())
    print("ref drafts:", drafts)
    # ground truth continuation (plain greedy)
    g = model.generate(ids, max_new_tokens=K + 1, do_sample=False)
    print("greedy truth:", g[0, L:].tolist())
