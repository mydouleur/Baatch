import torch, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

d = r"D:\xdna_llm\models\Ornith-1.5-35B-A3B"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
m.eval()

s = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(s, return_tensors="pt")
with torch.no_grad():
    out = m(ids)
logits = out.logits[0, -1]
top = torch.topk(logits, 10)
print("HF top10:", [(int(i), repr(tok.decode([int(i)])), round(float(logits[i]), 3)) for i in top.indices])
EOF = None
