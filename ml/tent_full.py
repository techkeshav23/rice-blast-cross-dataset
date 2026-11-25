"""Tent on ALL seeds (for a consistent full-seed baseline-comparison table).
Saves outputs/tent_full.json: per-cell mean external Macro-F1 under Tent.
"""
import json, os, importlib, statistics
import torch
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELLS = [
    ("MobileNetV3 Canopy",      "mobilenet_v3_small", "paddy_doctor", "philippines", [42,1,2,3,4,5,6,7]),
    ("MobileNetV3 Single-leaf", "mobilenet_v3_small", "kaggle_rice",  "riceleafs",   [42,1,2,3,4]),
    ("EfficientNet Canopy",     "efficientnet_b0",    "paddy_doctor", "philippines", [42,1,2,3,4,5]),
    ("EfficientNet Single-leaf","efficientnet_b0",    "kaggle_rice",  "riceleafs",   [42,1,2,3,4,5]),
    ("MNet paddy>rice",  "mobilenet_v3_small", "paddy_doctor", "riceleafs",   [42,1,2,3,4,5]),
    ("ENet paddy>rice",  "efficientnet_b0",    "paddy_doctor", "riceleafs",   [42,1,2,3,4,5]),
    ("MNet kaggle>phil", "mobilenet_v3_small", "kaggle_rice",  "philippines", [42,1,2,3,4,5]),
    ("ENet kaggle>phil", "efficientnet_b0",    "kaggle_rice",  "philippines", [42,1,2,3,4,5]),
]


def main():
    from models import build_model
    from dg import tent_adapt
    out = {}
    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(32)
        vals, novals = [], []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            def fresh():
                n = build_model(model_name, config.NUM_CLASSES, pretrained=False)
                n.load_state_dict(torch.load(ck, map_location="cpu")); return n.to(DEVICE)
            n0 = fresh(); n0.eval()
            novals.append(T._metrics(*T._predict(n0, ext, DEVICE))["macro_f1"]); del n0
            net = fresh(); tent_adapt(net, ext, DEVICE)
            vals.append(T._metrics(*T._predict(net, ext, DEVICE))["macro_f1"]); del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        out[name] = {"n": len(vals), "no": round(statistics.mean(novals), 2),
                     "tent": round(statistics.mean(vals), 2),
                     "gain": round(statistics.mean(vals) - statistics.mean(novals), 2)}
        print(f"  {name:26s} n={len(vals)}  no {out[name]['no']:.1f} Tent {out[name]['tent']:.1f} gain {out[name]['gain']:+.2f}")
    import statistics as st
    mg = round(st.mean([out[n]["gain"] for n in out]), 2)
    worst = round(min(out[n]["gain"] for n in out), 2)
    out["_summary"] = {"mean_gain": mg, "worst_cell": worst, "n_cells": len(out)}
    (config.OUTPUTS / "tent_full.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Tent (8 cells): mean gain {mg:+.2f}  worst-cell {worst:+.2f}")
    print("Saved outputs/tent_full.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
