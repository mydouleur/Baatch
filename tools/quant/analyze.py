"""M1' XBF3 v0 analysis: real measurements on Ornith-1.5-35B-A3B.

Quantizes a set of tensors (MoE experts from layers 0/20, attention q_proj,
lm_head sample) with XBF3, with and without Hadamard rotation, and measures:
  - weight-domain SNR (dB)
  - per-superblock relative error distribution
  - synthetic-activation (Gaussian) output relative error for a few experts
  - wall-clock time per matrix

Results are written as JSON to results_v0.json; the markdown report is
written from those numbers (see reports/bf3_v0_report.md).

Usage: .venv/Scripts/python.exe analyze.py [model_dir]
"""

import json
import sys
import time

import numpy as np

from safetensors_reader import ModelReader
from xbf3 import quantize, dequantize, pack3, unpack3, snr_db, per_block_stats

MODEL_DIR = r"d:\xdna_llm\models\Ornith-1.5-35B-A3B"
LM_HEAD_ROWS = 4096          # strided row sample of lm_head
ACT_SAMPLES = 256            # synthetic activation batch
SEED = 42


def expert_matrix(mr, layer, expert, kind):
    """kind in {gate, up, down}; sliced out of the fused expert tensors."""
    if kind in ("gate", "up"):
        name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        rows = slice(0, 512) if kind == "gate" else slice(512, 1024)
        return mr.read(name, (expert, rows, slice(None)))
    name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    return mr.read(name, (expert, slice(None), slice(None)))


def measure(w, rotate):
    """Quantize+dequantize one matrix; return SNR, per-block rel err, seconds."""
    t0 = time.perf_counter()
    qq = quantize(w, rotate=rotate)
    # verify packing roundtrip on this exact tensor (cheap, catches bit bugs)
    packed = pack3(qq["q"])
    assert np.array_equal(unpack3(packed, qq["q"].shape[-1]), qq["q"])
    rec = dequantize(qq)
    dt = time.perf_counter() - t0
    return snr_db(w, rec), per_block_stats(w, qq), dt, rec, qq


def act_rel_err(w, rec, rng, n=ACT_SAMPLES):
    """Relative output error with Gaussian activations: ||X(W-Wh)T||/||XWT||."""
    x = rng.standard_normal((n, w.shape[1]))
    y = x @ w.astype(np.float64).T
    yh = x @ rec.astype(np.float64).T
    return float(np.linalg.norm(y - yh) / np.linalg.norm(y))


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR
    mr = ModelReader(model_dir)
    rng = np.random.default_rng(SEED)
    experts = sorted(rng.choice(256, size=3, replace=False).tolist())
    print(f"sampled experts: {experts}", flush=True)

    results = {"model_dir": model_dir, "seed": SEED, "experts": experts,
               "lm_head_rows": LM_HEAD_ROWS, "act_samples": ACT_SAMPLES,
               "tensors": []}

    jobs = []  # (label, loader)
    for layer in (0, 20):
        for e in experts:
            for kind in ("gate", "up", "down"):
                jobs.append((f"L{layer}.expert{e}.{kind}",
                             lambda l=layer, e=e, k=kind: expert_matrix(mr, l, e, k)))
    jobs.append(("L3.self_attn.q_proj", lambda: mr.read(
        "model.language_model.layers.3.self_attn.q_proj.weight")))

    def load_lm_head():
        name = "lm_head.weight"
        _, shape, _ = mr.meta(name)
        rows = np.linspace(0, shape[0] - 1, LM_HEAD_ROWS).astype(np.int64)
        return mr.read(name, (rows, slice(None)))
    jobs.append((f"lm_head[{LM_HEAD_ROWS}rows]", load_lm_head))

    act_targets = {f"L0.expert{experts[0]}.down", f"L20.expert{experts[0]}.up",
                   f"L0.expert{experts[0]}.gate"}

    for label, loader in jobs:
        t0 = time.perf_counter()
        w = loader()
        t_load = time.perf_counter() - t0
        entry = {"label": label, "shape": list(w.shape),
                 "load_s": round(t_load, 3)}
        for rot in (False, True):
            snr, relblk, dt, rec, qq = measure(w, rot)
            tag = "rot" if rot else "norot"
            entry[tag] = {"snr_db": round(snr, 3), "quant_s": round(dt, 3),
                          "blk_rel_err_pct": [round(float(np.percentile(relblk, p)), 5)
                                              for p in (50, 90, 99, 100)]}
            if label in act_targets:
                entry[tag]["act_rel_err"] = round(act_rel_err(w, rec, rng), 6)
        entry["snr_gain_db"] = round(entry["rot"]["snr_db"]
                                     - entry["norot"]["snr_db"], 3)
        results["tensors"].append(entry)
        print(f"{label:26s} {str(entry['shape']):16s} "
              f"norot={entry['norot']['snr_db']:7.3f} dB  "
              f"rot={entry['rot']['snr_db']:7.3f} dB  "
              f"gain={entry['snr_gain_db']:+6.3f} dB  "
              f"({entry['norot']['quant_s']:.1f}s/{entry['rot']['quant_s']:.1f}s, "
              f"load {entry['load_s']:.1f}s)", flush=True)

    # aggregate per-superblock error distribution over all expert matrices
    agg = {False: [], True: []}
    for layer in (0, 20):
        for e in experts:
            for kind in ("gate", "up", "down"):
                w = expert_matrix(mr, layer, e, kind)
                for rot in (False, True):
                    qq = quantize(w, rotate=rot)
                    agg[rot].append(per_block_stats(w, qq))
    results["expert_blk_rel_err_agg"] = {
        ("rot" if rot else "norot"): {
            "n_blocks": int(sum(len(a) for a in v)),
            "percentiles": {str(p): round(float(np.percentile(np.concatenate(v), p)), 5)
                            for p in (50, 90, 99, 99.9, 100)},
            "mean": round(float(np.concatenate(v).mean()), 5),
        } for rot, v in agg.items()}

    with open("results_v0.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote results_v0.json")


if __name__ == "__main__":
    main()
