"""Generate publication figures for the IEEE paper from the REAL final results.

Outputs (300 dpi PNG + PDF) into ml/outputs/paper_figs/:
  fig1_gap.png        - internal vs external Macro-F1 (the generalization gap), 4 cells
  fig2_adabn.png      - no-adapt vs AdaBN, 4 cells, with significance stars
  fig3_collapse.png   - confusion matrices: EfficientNet single-leaf, no-adapt vs AdaBN
                        (prediction collapse -> AdaBN recovery), computed live from checkpoint
All numbers come straight from ml/outputs/*.json or the trained checkpoint.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

O = config.OUTPUTS
FIGS = O / "paper_figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # Times-like math to match the IEEE (Times) body
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.dpi": 300,
})
GREEN, BLUE, RED, GREY = "#2e7d32", "#1565c0", "#c62828", "#9e9e9e"
CELLS = [
    ("MobileNetV3\nCanopy", "mobilenet_v3_small", "paddy_doctor-philippines", "MNet paddy>phil"),
    ("MobileNetV3\nSingle-leaf", "mobilenet_v3_small", "kaggle_rice-riceleafs", "MNet kaggle>rice"),
    ("EfficientNet\nCanopy", "efficientnet_b0", "paddy_doctor-philippines", "ENet paddy>phil"),
    ("EfficientNet\nSingle-leaf", "efficientnet_b0", "kaggle_rice-riceleafs", "ENet kaggle>rice"),
]

# Refreshed per-seed data (8-seed cross-regime + DE-DUPLICATED canopy target).
_CG = json.loads((O / "confidence_gate.json").read_text())["cells"]


def _save(fig, name):
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ---------- Fig 1: generalization gap ----------
def fig1():
    internal, external, ext_err = [], [], []
    for _, m, p, cgk in CELLS:
        s = json.loads((O / f"seed_summary_{m}_{p}.json").read_text())["methods"]["baseline"]
        internal.append(s["internal_mean"])
        no = _CG[cgk]["raw"]["no"]                    # cleaned-canopy baseline external
        external.append(float(np.mean(no))); ext_err.append(float(np.std(no)))
    x = np.arange(len(CELLS)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x - w/2, internal, w, label="Internal (source test)", color=GREEN)
    ax.bar(x + w/2, external, w, yerr=ext_err, capsize=4, label="External (unseen target)", color=BLUE)
    for i in range(len(CELLS)):
        ax.annotate(f"gap\n{internal[i]-external[i]:.1f}", (x[i], max(internal[i], external[i]) + 3),
                    ha="center", fontsize=8, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CELLS], fontsize=9)
    ax.set_ylabel("Macro-F1 (%)"); ax.set_ylim(0, 108)
    ax.set_title("The internal-to-external generalization gap (baseline)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig1_gap")


# ---------- Fig 2: AdaBN effect ----------
def fig2():
    labels, noad, adabn, ps = [], [], [], []
    from scipy import stats
    for name, m, p, cgk in CELLS:
        no = _CG[cgk]["raw"]["no"]; al = _CG[cgk]["raw"]["always"]
        noad.append(float(np.mean(no))); adabn.append(float(np.mean(al))); labels.append(name)
        t, pv = stats.ttest_rel(al, no)
        ps.append(pv)
    x = np.arange(len(CELLS)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x - w/2, noad, w, label="No adaptation", color=GREY)
    bars = ax.bar(x + w/2, adabn, w, label="+ AdaBN (test-time BN)", color=GREEN)
    for i in range(len(CELLS)):
        d = adabn[i] - noad[i]
        col = GREEN if d > 0 else RED
        star = "*" if ps[i] < 0.05 else ""
        ax.annotate(f"{d:+.1f}{star}", (x[i] + w/2, adabn[i] + 1.5), ha="center",
                    fontsize=8.5, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("External Macro-F1 (%)"); ax.set_ylim(0, 82)
    ax.set_title("AdaBN is a conditional remedy (* = p < 0.05, paired t-test)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig2_adabn")


# ---------- Fig 3: prediction collapse -> AdaBN recovery (confusion matrices) ----------
def fig3():
    import torch
    from models import build_model
    from dg import adabn_adapt
    from train import _loaders, _predict
    import os
    os.environ["RICE_SOURCE_DOMAIN"] = "kaggle_rice"
    os.environ["RICE_TARGET_DOMAIN"] = "riceleafs"
    # reload config domains
    import importlib
    importlib.reload(config)
    from train import _loaders as loaders2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "baseline_efficientnet_b0_kaggle_rice-riceleafs_s42"
    net = build_model("efficientnet_b0", config.NUM_CLASSES, pretrained=False)
    net.load_state_dict(torch.load(O / f"{tag}_efficientnet_b0.pt", map_location="cpu"))
    net = net.to(device)
    _, _, _, ext = loaders2(64)

    def confusion(model):
        y_true, y_pred = _predict(model, ext, device)
        cm = np.zeros((3, 3), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
    net.eval(); cm0 = confusion(net)
    adabn_adapt(net, ext, device); cm1 = confusion(net)

    classes = ["Blast", "Brown\nSpot", "Healthy"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for ax, cm, title in [(axes[0], cm0, "No adaptation (collapse)"),
                          (axes[1], cm1, "+ AdaBN (recovered)")]:
        cmn = cm / cm.sum(1, keepdims=True) * 100
        im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=100)
        ax.set_xticks(range(3)); ax.set_xticklabels(classes, fontsize=8.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(classes, fontsize=8.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i,j]:.0f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 55 else "black", fontsize=9)
    fig.suptitle("EfficientNet-B0, single-leaf target: recall % (rows sum to 100)",
                 fontsize=10.5, fontweight="bold")
    _save(fig, "fig3_collapse")


if __name__ == "__main__":
    print("Generating paper figures ->", FIGS)
    fig1(); fig2(); fig3()
    print("Done.")
