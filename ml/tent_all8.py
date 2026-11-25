"""Tent on ALL 8 cells in one consistent pass: correct seed counts (new cells at 8),
cleaned canopy target (paddy_doctor->philippines de-duplicated). Memory-light
(1 model load/seed, batch 16). Saves outputs/tent_all8.json with per-cell no/tent/
gain and the 8-cell mean-gain + worst-cell. Supersedes tent_full+tent_new4.
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
    from dg import tent_adapt
    bs = int(os.environ.get("RICE_TENT_BS", "16"))
    out = {}
    for name, model_name, src, tgt, seeds in CELLS:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        _set_exclude(src, tgt)
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(bs)
        no_v, tent_v = [], []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            def fresh():
                n = build_model(model_name, config.NUM_CLASSES, pretrained=False)
                n.load_state_dict(torch.load(ck, map_location="cpu")); return n.to(DEVICE)
            n0 = fresh(); n0.eval()
            no_v.append(T._metrics(*T._predict(n0, ext, DEVICE))["macro_f1"]); del n0
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            nt = fresh(); tent_adapt(nt, ext, DEVICE)
            tent_v.append(T._metrics(*T._predict(nt, ext, DEVICE))["macro_f1"]); del nt
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        m = lambda v: round(statistics.mean(v), 2)
        out[name] = {"n": len(no_v), "no": m(no_v), "tent": m(tent_v),
                     "gain": round(m(tent_v) - m(no_v), 2)}
        print(f"{name:16s} n={out[name]['n']}  no {out[name]['no']:5.1f} | Tent {out[name]['tent']:5.1f} | gain {out[name]['gain']:+.2f}")

    mg = round(statistics.mean([out[n]["gain"] for n in out]), 2)
    worst = round(min(out[n]["gain"] for n in out), 2)
    out["_summary"] = {"mean_gain": mg, "worst_cell": worst, "n_cells": len(CELLS)}
    json.dump(out, open(config.OUTPUTS / "tent_all8.json", "w"), indent=2)
    print(f"\n  TENT 8-CELL: mean gain {mg:+.2f}  worst-cell {worst:+.2f}")
    print("Saved tent_all8.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
