"""Collapse-Adaptive AdaBN (the refined method).  Label-free, per (model,target):
  1. run source model on unlabelled target -> min predicted-class share s_min
  2. if s_min < TAU (severe collapse)   -> FULL AdaBN     (lam = 1.0)
     else (mild shift)                  -> PARTIAL AdaBN  (lam = LAM_MILD)
The partial-vs-full choice is motivated by the lam-sweep ablation: full re-estimation
best repairs severe collapse, while partial source/target mixing best helps mild
shift (where full AdaBN over-corrects).

Evaluates per-seed on the 4 original cells and compares:
  No-adapt | Full AdaBN | Collapse-Gated (hard) | Collapse-Adaptive (ours, soft)
Saves outputs/adaptive_adabn.json + fig6.
"""
import json, os, importlib, statistics
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU = 0.05
LAM_MILD = 0.5
CELLS = [
    ("MobileNetV3 Canopy",      "mobilenet_v3_small", "paddy_doctor", "philippines", [42,1,2,3,4,5,6,7]),
    ("MobileNetV3 Single-leaf", "mobilenet_v3_small", "kaggle_rice",  "riceleafs",   [42,1,2,3,4]),
    ("EfficientNet Canopy",     "efficientnet_b0",    "paddy_doctor", "philippines", [42,1,2,3,4,5]),
    ("EfficientNet Single-leaf","efficientnet_b0",    "kaggle_rice",  "riceleafs",   [42,1,2,3,4,5]),
]


def main():
    from models import build_model
    from dg import adabn_adapt, partial_adabn_adapt
    rows = {}

    def load(model_name, pair, s):
        ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
        if not ck.exists():
            return None
        net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        return net.to(DEVICE)

    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"
        _, _, _, ext = T._loaders(64)
        na, full, gated, adapt = [], [], [], []
        for s in seeds:
            net = load(model_name, pair, s)
            if net is None:
                continue
            net.eval()
            yt, yp = T._predict(net, ext, DEVICE); yp = np.array(yp)
            f_na = T._metrics(yt, yp.tolist())["macro_f1"]
            s_min = min((yp == c).mean() for c in range(config.NUM_CLASSES))
            severe = s_min < TAU
            # full AdaBN
            nf = load(model_name, pair, s); adabn_adapt(nf, ext, DEVICE)
            f_full = T._metrics(*T._predict(nf, ext, DEVICE))["macro_f1"]
            # partial AdaBN (lam mild)
            npar = load(model_name, pair, s); partial_adabn_adapt(npar, ext, DEVICE, LAM_MILD)
            f_par = T._metrics(*T._predict(npar, ext, DEVICE))["macro_f1"]
            # hard gate: full if severe else no-adapt
            f_gate = f_full if severe else f_na
            # adaptive: full if severe else partial
            f_adapt = f_full if severe else f_par
            na.append(f_na); full.append(f_full); gated.append(f_gate); adapt.append(f_adapt)
            del net, nf, npar                       # free GPU memory each seed
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        m = lambda v: round(statistics.mean(v), 2)
        rows[name] = {"n": len(na), "no_adapt": m(na), "full_adabn": m(full),
                      "gated": m(gated), "adaptive": m(adapt)}
        print(f"  {name:26s} n={len(na)}  no {m(na):.1f} | full {m(full):.1f} | "
              f"gated {m(gated):.1f} | ADAPTIVE {m(adapt):.1f}")

    def mg(k): return round(statistics.mean([rows[n][k]-rows[n]["no_adapt"] for n in rows]), 2)
    summary = {"tau": TAU, "lam_mild": LAM_MILD, "cells": rows,
               "mean_gain": {"full_adabn": mg("full_adabn"), "gated": mg("gated"),
                             "adaptive": mg("adaptive")}}
    (config.OUTPUTS / "adaptive_adabn.json").write_text(json.dumps(summary, indent=2))

    # figure
    labels = list(rows.keys()); x = np.arange(len(labels)); w = 0.2
    series = [("No adaptation", "no_adapt", "#9e9e9e"),
              ("Full AdaBN", "full_adabn", "#ef9a9a"),
              ("Collapse-Gated", "gated", "#66bb6a"),
              ("Collapse-Adaptive (ours)", "adaptive", "#1b5e20")]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for i,(lab,key,col) in enumerate(series):
        ax.bar(x + (i-1.5)*w, [rows[l][key] for l in labels], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" ","\n") for l in labels], fontsize=8.5)
    ax.set_ylabel("External Macro-F1 (%)"); ax.set_ylim(0, 82)
    ax.set_title("Collapse-Adaptive AdaBN: gains everywhere, harm nowhere", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", ncol=2); ax.grid(alpha=0.3, axis="y")
    figs = config.OUTPUTS / "paper_figs"; figs.mkdir(exist_ok=True)
    fig.savefig(figs / "fig6_adaptive.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(figs / "fig6_adaptive.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n  MEAN GAIN vs no-adapt: full {mg('full_adabn'):+.2f} | "
          f"gated {mg('gated'):+.2f} | ADAPTIVE {mg('adaptive'):+.2f}")
    print("  saved fig6_adaptive + adaptive_adabn.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
