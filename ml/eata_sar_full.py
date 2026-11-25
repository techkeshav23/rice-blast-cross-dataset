"""Run EATA and SAR (selective-TTA baselines) on all 8 cells, all seeds — a fair
empirical comparison for the paper (addresses 'cited but not compared').
Saves outputs/eata_sar_full.json: per-cell mean external Macro-F1 under EATA and SAR.
Run ALONE (one torch job).
"""
import json, os, importlib, statistics
import torch
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

CANOPY_EXCLUDE = str((config.OUTPUTS / "canopy_leak_exclude.txt"))


def _set_exclude(src, tgt):
    if src == "paddy_doctor" and tgt == "philippines":
        os.environ["RICE_EXTERNAL_EXCLUDE"] = CANOPY_EXCLUDE
    else:
        os.environ.pop("RICE_EXTERNAL_EXCLUDE", None)


def main():
    from models import build_model
    from dg import eata_adapt, sar_adapt
    from train import _loaders, _predict, _metrics
    out = {}
    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        _set_exclude(src, tgt)
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(32)   # batch 32: SAR does 2x fwd/bwd
        eata_v, sar_v, no_v = [], [], []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            def fresh():
                n = build_model(model_name, config.NUM_CLASSES, pretrained=False)
                n.load_state_dict(torch.load(ck, map_location="cpu"))
                return n.to(DEVICE)
            n0 = fresh(); n0.eval()
            no_v.append(T._metrics(*T._predict(n0, ext, DEVICE))["macro_f1"]); del n0
            ne = fresh(); eata_adapt(ne, ext, DEVICE)
            eata_v.append(T._metrics(*T._predict(ne, ext, DEVICE))["macro_f1"]); del ne
            ns = fresh(); sar_adapt(ns, ext, DEVICE)
            sar_v.append(T._metrics(*T._predict(ns, ext, DEVICE))["macro_f1"]); del ns
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        m = lambda v: round(statistics.mean(v), 2)
        out[name] = {"n": len(no_v), "no": m(no_v), "eata": m(eata_v), "sar": m(sar_v)}
        print(f"{name:16s} n={out[name]['n']}  no {out[name]['no']:5.1f} | EATA {out[name]['eata']:5.1f} | SAR {out[name]['sar']:5.1f}")

    def mg(k): return round(statistics.mean([out[n][k]-out[n]["no"] for n in out]), 2)
    def worst(k): return round(min(out[n][k]-out[n]["no"] for n in out), 2)
    summary = {"cells": out,
               "mean_gain": {"eata": mg("eata"), "sar": mg("sar")},
               "worst_cell": {"eata": worst("eata"), "sar": worst("sar")}}
    json.dump(summary, open(config.OUTPUTS / "eata_sar_full.json", "w"), indent=2)
    print(f"\n  EATA: mean {mg('eata'):+.2f}  worst {worst('eata'):+.2f}")
    print(f"  SAR : mean {mg('sar'):+.2f}  worst {worst('sar'):+.2f}")
    print("Saved eata_sar_full.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
