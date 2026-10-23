import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np, os

d = r"D:\xdna_llm\models\Ornith-1.5-35B-A3B"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
m.eval()

caps = {}
def hook(name, pos=0):
    def h(mod, inp, out):
        o = out
        if isinstance(o, tuple):
            o = next((t for t in o if torch.is_tensor(t) and t.dim() >= 2), None)
        if o is None or not torch.is_tensor(o):
            return
        v = o[0, pos].detach().float().numpy()
        caps[name] = v.reshape(-1) if v.ndim > 1 else v
        if len(inp) and torch.is_tensor(inp[0]):
            vi = inp[0][0, pos].detach().float().numpy()
            caps[name + ".in"] = vi.reshape(-1) if vi.ndim > 1 else vi
    return h

la = m.model.layers[0].linear_attn
la.in_proj_qkv.register_forward_hook(hook("qkv"))
la.in_proj_z.register_forward_hook(hook("z"))
la.in_proj_b.register_forward_hook(hook("b"))
la.in_proj_a.register_forward_hook(hook("a"))
la.norm.register_forward_hook(hook("gatednorm"))
la.out_proj.register_forward_hook(hook("outproj"))

s = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
ids = tok(s, return_tensors="pt")
with torch.no_grad():
    m(**ids)

os.makedirs("probe_hf_gdn", exist_ok=True)
for k, v in caps.items():
    np.save(f"probe_hf_gdn/{k}.npy", v)
    print(k, v.shape, "rms=%.5f" % float(np.sqrt((v**2).mean())))
