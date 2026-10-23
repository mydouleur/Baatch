import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, os, sys

d = r"D:\xdna_llm\models\Ornith-1.5-35B-A3B"
outdir = sys.argv[1] if len(sys.argv) > 1 else "probe_hf"
os.makedirs(outdir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
m.eval()

s = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
ids = tok(s, return_tensors="pt")
with torch.no_grad():
    out = m(**ids, output_hidden_states=True)
# hidden_states: 41 层（embed + 40 层输出），取最后位置
hs = out.hidden_states
print("n hidden:", len(hs))
for L in range(1, len(hs)):
    v = hs[L][0, -1].float().numpy()
    v.tofile(os.path.join(outdir, f"ylayer_{L-1:02d}.bin"))
logits = out.logits[0, -1]
top = torch.topk(logits, 10)
print("HF top10:", [(int(i), repr(tok.decode([int(i)])), round(float(logits[i]), 3)) for i in top.indices])
