"""Confidence-Gated AdaBN (refined label-free gate).  Per (model,target,seed):
  1. run AdaBN; measure change in MEAN prediction entropy  d_ent = ent_adabn - ent_noadapt
  2. if AdaBN sharply over-confidences (d_ent < -TAU_E) -> REVERT to no-adapt
     else                                               -> keep AdaBN
Rationale: a large confidence spike signals over-commitment (the hurt regime);
a mild/positive entropy change signals healthy re-normalisation (the help regime).
Label-free (no target labels).  Evaluated per-seed on all 8 cells; compares
No-adapt | Always-AdaBN | Min-share Gated | Confidence-Gated, for several TAU_E.
Saves outputs/confidence_gate.json.
"""
import json, os, importlib, statistics
import numpy as np
import torch
import torch.nn.functional as F
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAUS_E = [0.10, 0.145, 0.20]
TAU_MIN = 0.05
CELLS = [
    ("MNet paddy>phil",  "mobilenet_v3_small", "paddy_doctor", "philippines", [42,1,2,3,4,5,6,7]),
    ("MNet kaggle>rice", "mobilenet_v3_small", "kaggle_rice",  "riceleafs",   [42,1,2,3,4]),
    ("ENet paddy>phil",  "efficientnet_b0",    "paddy_doctor", "philippines", [42,1,2,3,4,5]),
    ("ENet kaggle>rice", "efficientnet_b0",    "kaggle_rice",  "riceleafs",   [42,1,2,3,4,5]),
    ("MNet paddy>rice",  "mobilenet_v3_small", "paddy_doctor", "riceleafs",   [42,1,2,3,4,5,6,7]),
    ("ENet paddy>rice",  "efficientnet_b0",    "paddy_doctor", "riceleafs",   [42,1,2,3,4,5,6,7]),
    ("MNet kaggle>phil", "mobilenet_v3_small", "kaggle_rice",  "philippines", [42,1,2,3,4,5,6,7]),
    ("ENet kaggle>phil", "efficientnet_b0",    "kaggle_rice",  "philippines", [42,1,2,3,4,5,6,7]),
]

# Pairing-specific leakage de-dup: the canopy target (paddy_doctor->philippines)
# shares 32 near-duplicate images with the source (see cross_leakage.py). Exclude
# them from that pairing's external_test only (leaf->canopy keeps them -- not leaked).
CANOPY_EXCLUDE = str((config.OUTPUTS / "canopy_leak_exclude.txt"))


def _set_exclude(src, tgt):
    if src == "paddy_doctor" and tgt == "philippines":
        os.environ["RICE_EXTERNAL_EXCLUDE"] = CANOPY_EXCLUDE
    else:
        os.environ.pop("RICE_EXTERNAL_EXCLUDE", None)


def main():
    from models import build_model
    from dg import adabn_adapt
    rows = {}

    def stats(net, ext):
        ents, preds, ys = [], [], []
        with torch.no_grad():
            for x, y, *_ in ext:
                logit = net(x.to(DEVICE)); prob = F.softmax(logit, 1)
                ents += (-(prob*(prob+1e-9).log()).sum(1)).cpu().tolist()
                preds += logit.argmax(1).cpu().tolist(); ys += y.tolist()
        return float(np.mean(ents)), np.array(preds), np.array(ys)

    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        _set_exclude(src, tgt)
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(64)
        per = {"no": [], "always": [], "minshare": [], "dent": [], "minsh": [],
               **{f"conf_{t}": [] for t in TAUS_E}}
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
            net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(DEVICE).eval()
            e0, p0, ys = stats(net, ext)
            f0 = T._metrics(ys.tolist(), p0.tolist())["macro_f1"]
            min_share = min((p0 == c).mean() for c in range(config.NUM_CLASSES))
            adabn_adapt(net, ext, DEVICE)
            e1, p1, _ = stats(net, ext)
            f1 = T._metrics(ys.tolist(), p1.tolist())["macro_f1"]
            d_ent = e1 - e0
            per["no"].append(f0); per["always"].append(f1)
            per["dent"].append(round(float(d_ent), 4)); per["minsh"].append(round(float(min_share), 4))
            per["minshare"].append(f1 if min_share < TAU_MIN else f0)
            for t in TAUS_E:
                per[f"conf_{t}"].append(f0 if d_ent < -t else f1)   # revert on big confidence spike
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        m = lambda v: round(statistics.mean(v), 2)
        rows[name] = {k: m(v) for k, v in per.items()}
        rows[name]["n"] = len(per["no"])
        rows[name]["raw"] = {"no": per["no"], "always": per["always"],
                             "dent": per["dent"], "minsh": per["minsh"]}
        print(f"{name:16s} n={rows[name]['n']}  no {rows[name]['no']:5.1f} | always {rows[name]['always']:5.1f} "
              f"| minshare {rows[name]['minshare']:5.1f} | conf.145 {rows[name]['conf_0.145']:5.1f}")

    def mg(k): return round(statistics.mean([rows[n][k]-rows[n]["no"] for n in rows]), 2)
    def worst(k): return round(min(rows[n][k]-rows[n]["no"] for n in rows), 2)
    summary = {"cells": rows,
               "mean_gain": {k: mg(k) for k in ["always", "minshare"] + [f"conf_{t}" for t in TAUS_E]},
               "worst_cell_delta": {k: worst(k) for k in ["always", "minshare"] + [f"conf_{t}" for t in TAUS_E]}}
    json.dump(summary, open(config.OUTPUTS / "confidence_gate.json", "w"), indent=2)
    print("\n=== MEAN GAIN vs no-adapt (8 cells) ===")
    for k, v in summary["mean_gain"].items():
        print(f"  {k:12s} mean {v:+.2f}   worst-cell {summary['worst_cell_delta'][k]:+.2f}")
    print("Saved confidence_gate.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
