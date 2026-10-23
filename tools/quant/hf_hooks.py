import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, os

d = r"D:\xdna_llm\models\Ornith-1.5-35B-A3B"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
m.eval()

caps = {}
def hook(name):
    def h(mod, inp, out):
        o = out
        if isinstance(o, tuple):
            o = next((t for t in o if torch.is_tensor(t) and t.dim() >= 2 and t.shape[-1] == 2048), None)
            if o is None: o = out[0]
        caps[name] = (inp[0][0, 0].float().numpy() if (len(inp) and torch.is_tensor(inp[0])) else None,
                      o[0, 0].float().numpy() if (o is not None and torch.is_tensor(o)) else None)
    return h

lay0 = m.model.layers[0]
lay0.linear_attn.register_forward_hook(hook("l0.attn"))
lay0.mlp.register_forward_hook(hook("l0.mlp"))
lay0.input_layernorm.register_forward_hook(hook("l0.innorm"))

s = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
ids = tok(s, return_tensors="pt")
with torch.no_grad():
    out = m(**ids)

os.makedirs("probe_hf_hooks", exist_ok=True)
for k, (i, o) in caps.items():
    if i is not None: np.save(f"probe_hf_hooks/{k}.in.npy", i)
    if o is not None: np.save(f"probe_hf_hooks/{k}.out.npy", o)
    print(k, "in rms", None if i is None else float(np.sqrt((i**2).mean())),
          "out rms", None if o is None else float(np.sqrt((o**2).mean())))
