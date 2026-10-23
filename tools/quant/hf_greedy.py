import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

d = r"D:\xdna_llm\models\Ornith-1.5-35B-A3B"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
m.eval()

s = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
ids = tok(s, return_tensors="pt")
with torch.no_grad():
    gen = m.generate(**ids, max_new_tokens=10, do_sample=False)
out = gen[0, ids["input_ids"].shape[1]:].tolist()
print("HF greedy:", out)
print(repr(tok.decode(out)))
