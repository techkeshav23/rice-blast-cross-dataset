"""Collapse-Gated AdaBN (the novel method): a label-free rule that decides WHEN to
apply test-time BatchNorm adaptation, so we keep AdaBN's large recoveries under
severe collapse while avoiding its significant degradation of a well-calibrated
baseline.

Signal (no labels): run the source-trained model on the unlabeled target, take the
predicted-class distribution, and compute the MINIMUM class share. If some class is
almost never predicted (min share < tau), the model is severely collapsed -> apply
AdaBN; otherwise leave the model unchanged.

For each (backbone x pairing x seed) we compute:
  - no-adapt external Macro-F1
  - always-AdaBN external Macro-F1
  - the label-free trigger (min predicted-class share)
  - selective-AdaBN = AdaBN if triggered else no-adapt
Saves outputs/selective_adabn.json and a comparison figure.
"""
import json, os, importlib, statistics
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

TAU = 0.05  # a class predicted for <5% of target samples => severe collapse
CELLS = [
    ("MNet paddy>phil",   "mobilenet_v3_small", "paddy_doctor", "philippines", [42,1,2,3,4,5,6,7]),
    ("MNet kaggle>rice",  "mobilenet_v3_small", "kaggle_rice",  "riceleafs",   [42,1,2,3,4]),
    ("ENet paddy>phil",   "efficientnet_b0",    "paddy_doctor", "philippines", [42,1,2,3,4,5]),
    ("ENet kaggle>rice",  "efficientnet_b0",    "kaggle_rice",  "riceleafs",   [42,1,2,3,4,5]),
    ("MNet paddy>rice",   "mobilenet_v3_small", "paddy_doctor", "riceleafs",   [42,1,2,3]),
    ("ENet paddy>rice",   "efficientnet_b0",    "paddy_doctor", "riceleafs",   [42,1,2,3]),
    ("MNet kaggle>phil",  "mobilenet_v3_small", "kaggle_rice",  "philippines", [42,1,2,3]),
    ("ENet kaggle>phil",  "efficientnet_b0",    "kaggle_rice",  "philippines", [42,1,2,3]),
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    from models import build_model
    from dg import adabn_adapt
    rows = {}
    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src
        os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T
        importlib.reload(T)
        pair = f"{src}-{tgt}"
        _, _, _, ext = T._loaders(64)

        noad, always, sel, trig, minshares = [], [], [], [], []
        for s in seeds:
            ckpt = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ckpt.exists():
                continue
            net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
            net = net.to(DEVICE).eval()

            # --- no-adapt: metrics + label-free trigger ---
            yt, yp = T._predict(net, ext, DEVICE)
            m0 = T._metrics(yt, yp)["macro_f1"]
            yp = np.array(yp)
            shares = [(yp == c).mean() for c in range(config.NUM_CLASSES)]
            min_share = float(min(shares))

            # --- always AdaBN ---
            adabn_adapt(net, ext, DEVICE)
            m1 = T._metrics(*T._predict(net, ext, DEVICE))["macro_f1"]

            apply = min_share < TAU
            noad.append(m0); always.append(m1)
            sel.append(m1 if apply else m0)
            trig.append(apply); minshares.append(round(min_share, 3))
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        def ms(v): return (round(statistics.mean(v), 2),
                           round(statistics.stdev(v), 2) if len(v) > 1 else 0.0)
        rows[name.replace("\n", " ")] = {
            "n": len(noad),
            "no_adapt": ms(noad), "always_adabn": ms(always), "selective": ms(sel),
            "triggered": sum(trig), "min_shares": minshares,
            "per_seed": {"no_adapt": [round(x,2) for x in noad],
                         "always": [round(x,2) for x in always],
                         "selective": [round(x,2) for x in sel],
                         "trigger": trig},
        }
        na, al, se = ms(noad)[0], ms(always)[0], ms(sel)[0]
        print(f"{name.replace(chr(10),' '):26s} n={len(noad)} trig={sum(trig)}/{len(trig)}  "
              f"no-adapt {na:6.2f} | always {al:6.2f} | SELECTIVE {se:6.2f}  "
              f"(min-share {min(minshares):.3f}-{max(minshares):.3f})")
    return rows


def figure(rows):
    labels = list(rows.keys())
    x = np.arange(len(labels)); w = 0.26
    na = [rows[l]["no_adapt"][0] for l in labels]
    al = [rows[l]["always_adabn"][0] for l in labels]
    se = [rows[l]["selective"][0] for l in labels]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w, na, w, label="No adaptation", color="#9e9e9e")
    ax.bar(x,     al, w, label="Always AdaBN", color="#ef9a9a")
    ax.bar(x + w, se, w, label="Collapse-Gated AdaBN (ours)", color="#2e7d32")
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8.5)
    ax.set_ylabel("External Macro-F1 (%)"); ax.set_ylim(0, 82)
    ax.set_title("Collapse-Gated AdaBN keeps the gains, avoids the harm", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    figs = config.OUTPUTS / "paper_figs"; figs.mkdir(exist_ok=True)
    fig.savefig(figs / "fig5_selective.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(figs / "fig5_selective.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  saved fig5_selective.pdf/.png")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    print(f"=== Collapse-Gated AdaBN (tau={TAU}) ===")
    rows = run()
    figure(rows)
    # aggregate: mean gain over no-adapt for each strategy
    def totgain(key): return round(sum(rows[l][key][0]-rows[l]["no_adapt"][0] for l in rows), 2)
    summary = {"tau": TAU, "cells": rows,
               "mean_gain_always": round(totgain("always_adabn")/len(rows), 2),
               "mean_gain_selective": round(totgain("selective")/len(rows), 2)}
    (config.OUTPUTS / "selective_adabn.json").write_text(json.dumps(summary, indent=2))
    print(f"\nMean gain vs no-adapt:  always-AdaBN {summary['mean_gain_always']:+.2f}  |  "
          f"Collapse-Gated {summary['mean_gain_selective']:+.2f}")
    print("Saved outputs/selective_adabn.json")
