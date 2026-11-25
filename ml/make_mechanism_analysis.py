"""Deep mechanism analysis for the paper (turns the BN-mismatch CLAIM into
demonstrated evidence).  For each backbone x pairing cell (seed 42 baseline):
  - per-class recall on the external target, no-adapt vs AdaBN
  - predicted-class distribution + mean softmax entropy, no-adapt vs AdaBN
  - BN-statistic distance between SOURCE running stats and TARGET-estimated stats
Also computes Cohen's d effect sizes for the AdaBN comparisons from per-seed data,
and dataset per-class image counts.  Saves outputs/mechanism.json and a figure
outputs/paper_figs/fig4_bndist.pdf (BN distance vs AdaBN gain).
"""
import json, os, importlib, statistics, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

import config as config

CELLS = [
    ("MobileNetV3 Canopy",      "mobilenet_v3_small", "paddy_doctor", "philippines"),
    ("MobileNetV3 Single-leaf", "mobilenet_v3_small", "kaggle_rice",  "riceleafs"),
    ("EfficientNet Canopy",     "efficientnet_b0",    "paddy_doctor", "philippines"),
    ("EfficientNet Single-leaf","efficientnet_b0",    "kaggle_rice",  "riceleafs"),
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bn_layers(model):
    return [m for m in model.modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]


def snapshot_bn(model):
    return [(m.running_mean.detach().clone(), m.running_var.detach().clone())
            for m in bn_layers(model)]


def bn_distance(src, tgt):
    """Mean over layers/channels of a normalised source-target Gaussian gap."""
    ds = []
    for (ms, vs), (mt, vt) in zip(src, tgt):
        eps = 1e-5
        mean_term = (ms - mt) ** 2 / (vs + eps)
        std_term = (vs.sqrt() - vt.sqrt()) ** 2 / (vs + eps)
        ds.append((mean_term + std_term).mean().item())
    return float(np.mean(ds))


def analyse():
    from models import build_model
    from dg import adabn_adapt
    from train import _loaders, _predict

    results = {}
    for name, model_name, src, tgt in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src
        os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T
        importlib.reload(T)

        pair = f"{src}-{tgt}"
        tag = f"baseline_{model_name}_{pair}_s42"
        ckpt = config.OUTPUTS / f"{tag}_{model_name}.pt"
        if not ckpt.exists():
            print(f"  (skip {name}: no checkpoint)"); continue
        net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net = net.to(DEVICE).eval()
        _, _, _, ext = T._loaders(64)

        def recall_entropy(model):
            model.eval()
            ys, ps, ents, preds = [], [], [], []
            with torch.no_grad():
                for x, y, *_ in ext:
                    logit = model(x.to(DEVICE))
                    prob = F.softmax(logit, 1)
                    ent = -(prob * (prob + 1e-9).log()).sum(1)
                    pred = logit.argmax(1)
                    ys += y.tolist(); preds += pred.cpu().tolist()
                    ents += ent.cpu().tolist()
            ys = np.array(ys); preds = np.array(preds)
            cm = np.zeros((3, 3), int)
            for t, p in zip(ys, preds):
                cm[t, p] += 1
            recall = [round(100 * cm[c, c] / max(cm[c].sum(), 1), 1) for c in range(3)]
            pred_dist = [round(100 * (preds == c).mean(), 1) for c in range(3)]
            return recall, pred_dist, round(float(np.mean(ents)), 3), cm.tolist()

        src_bn = snapshot_bn(net)
        r0, d0, e0, cm0 = recall_entropy(net)
        adabn_adapt(net, ext, DEVICE)
        tgt_bn = snapshot_bn(net)
        r1, d1, e1, cm1 = recall_entropy(net)
        dist = bn_distance(src_bn, tgt_bn)

        results[name] = {
            "pair": pair, "model": model_name,
            "recall_noadapt": r0, "recall_adabn": r1,
            "pred_dist_noadapt": d0, "pred_dist_adabn": d1,
            "entropy_noadapt": e0, "entropy_adabn": e1,
            "bn_distance": round(dist, 3),
        }
        print(f"  {name:26s} BNdist={dist:6.3f}  recall {r0}->{r1}  ent {e0}->{e1}")
    return results


def effect_sizes():
    """Cohen's d (paired) for AdaBN comparisons from per-seed arrays."""
    out = {}
    for name, model_name, src, tgt in CELLS:
        f = config.OUTPUTS / f"adabn_baseline_{model_name}_{src}-{tgt}.json"
        if not f.exists():
            continue
        a = json.loads(f.read_text())
        diffs = [b - n for n, b in zip(a["per_seed_no_adapt"], a["per_seed_adabn"])]
        if len(diffs) > 1 and statistics.pstdev(diffs) > 0:
            d = statistics.mean(diffs) / statistics.stdev(diffs)
        else:
            d = float("nan")
        out[name] = {"mean_gain": round(statistics.mean(diffs), 2),
                     "cohens_d": round(d, 2), "n": len(diffs)}
        print(f"  {name:26s} gain={out[name]['mean_gain']:+.2f}  d={out[name]['cohens_d']}  n={out[name]['n']}")
    return out


def dataset_counts():
    """Per-class image counts from the split manifests, both pairings."""
    counts = {}
    splits = config.OUTPUTS.parent / "data" / "splits"
    for mani in sorted(splits.glob("*.json")) if splits.exists() else []:
        try:
            d = json.loads(mani.read_text())
        except Exception:
            continue
        per = {}
        items = d if isinstance(d, list) else d.get("items", d.get("samples", []))
        for it in items if isinstance(items, list) else []:
            c = it.get("label") or it.get("class")
            per[c] = per.get(c, 0) + 1
        if per:
            counts[mani.stem] = per
    return counts


def fig_bndist(mech):
    # AdaBN gains come from the REFRESHED confidence_gate.json (de-duplicated canopy,
    # 8-seed cross-regime, batch 64) -- NOT the stale adabn_baseline_*.json.
    cgkey = {
        "MobileNetV3 Canopy":      "MNet paddy>phil",
        "MobileNetV3 Single-leaf": "MNet kaggle>rice",
        "EfficientNet Canopy":     "ENet paddy>phil",
        "EfficientNet Single-leaf":"ENet kaggle>rice",
    }
    cg = json.loads((config.OUTPUTS / "confidence_gate.json").read_text())["cells"]
    xs, ys, labels = [], [], []
    for c in mech:
        r = cg[cgkey[c]]["raw"]
        gain = sum(r["always"]) / len(r["always"]) - sum(r["no"]) / len(r["no"])
        xs.append(mech[c]["bn_distance"]); ys.append(gain); labels.append(c)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for x, y, l in zip(xs, ys, labels):
        col = "#2e7d32" if y > 0 else "#c62828"
        ax.scatter(x, y, s=90, color=col, zorder=3)
        ax.annotate(l, (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("BatchNorm source$\\rightarrow$target statistic distance")
    ax.set_ylabel("AdaBN gain (external Macro-F1)")
    ax.set_title("Larger BN mismatch $\\Rightarrow$ larger AdaBN recovery", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3)
    figs = config.OUTPUTS / "paper_figs"; figs.mkdir(exist_ok=True)
    fig.savefig(figs / "fig4_bndist.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(figs / "fig4_bndist.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  saved fig4_bndist.pdf/.png")


if __name__ == "__main__":
    import sys
    os.environ.setdefault("RICE_NUM_WORKERS", "0")
    os.environ.setdefault("RICE_CACHE", "1")
    # --fig-only: regenerate just fig4 from cached mechanism.json (BN distances) +
    # refreshed confidence_gate.json gains. No GPU / no checkpoint re-run needed.
    if "--fig-only" in sys.argv:
        cached = json.loads((config.OUTPUTS / "mechanism.json").read_text())["mechanism"]
        fig_bndist(cached)
        raise SystemExit(0)
    print("== mechanism (per-class recall, entropy, BN distance) ==")
    mech = analyse()
    print("== effect sizes (Cohen's d) ==")
    es = effect_sizes()
    print("== dataset counts ==")
    dc = dataset_counts()
    if mech:
        fig_bndist(mech)
    (config.OUTPUTS / "mechanism.json").write_text(json.dumps(
        {"mechanism": mech, "effect_sizes": es, "dataset_counts": dc}, indent=2))
    print("Saved outputs/mechanism.json")
