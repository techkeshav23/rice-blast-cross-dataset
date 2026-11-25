"""Tent on the 4 NEW-pairing cells only (memory-light: 1 model load/seed, no-adapt
reused from eata_sar_full.json). Combines with the original 4 cells (tent_full.json)
to report an honest 8-cell Tent mean-gain + worst-cell.
"""
import json, os, importlib, statistics
import torch
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEW = [
    ("MNet paddy>rice",  "mobilenet_v3_small", "paddy_doctor", "riceleafs",   [42,1,2,3,4,5]),
    ("ENet paddy>rice",  "efficientnet_b0",    "paddy_doctor", "riceleafs",   [42,1,2,3,4,5]),
    ("MNet kaggle>phil", "mobilenet_v3_small", "kaggle_rice",  "philippines", [42,1,2,3,4,5]),
    ("ENet kaggle>phil", "efficientnet_b0",    "kaggle_rice",  "philippines", [42,1,2,3,4,5]),
]


def main():
    from models import build_model
    from dg import tent_adapt
    noadapt = json.load(open(config.OUTPUTS / "eata_sar_full.json"))["cells"]  # has 'no' per cell
    gains = {}
    bs = int(os.environ.get("RICE_TENT_BS", "16"))  # smaller batch -> OOM-safe on 6GB laptop GPUs
    for name, model_name, src, tgt, seeds in NEW:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(bs)
        vals = []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
            net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(DEVICE)
            tent_adapt(net, ext, DEVICE)
            vals.append(T._metrics(*T._predict(net, ext, DEVICE))["macro_f1"])
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        tent_mean = statistics.mean(vals)
        gains[name] = round(tent_mean - noadapt[name]["no"], 2)
        print(f"  {name:16s} n={len(vals)} no {noadapt[name]['no']:.1f} Tent {tent_mean:.1f} gain {gains[name]:+.2f}")

    # combine with original-4 Tent gains (from tent_full.json: no + tent per cell)
    orig = json.load(open(config.OUTPUTS / "tent_full.json"))
    orig_gains = {k: round(v["tent"] - v["no"], 2) for k, v in orig.items()
                  if isinstance(v, dict) and "tent" in v and "no" in v}
    if not orig_gains:  # older format without 'no' — recompute from known no-adapt
        base = {"MobileNetV3 Canopy": 64.55, "MobileNetV3 Single-leaf": 32.46,
                "EfficientNet Canopy": 71.11, "EfficientNet Single-leaf": 26.30}
        orig_gains = {k: round(orig[k]["tent"] - base[k], 2) for k in base if k in orig}
    all_gains = list(gains.values()) + list(orig_gains.values())
    mg = round(statistics.mean(all_gains), 2); worst = round(min(all_gains), 2)
    print(f"\n  TENT 8-CELL: mean gain {mg:+.2f}  worst-cell {worst:+.2f}  (n_cells={len(all_gains)})")
    json.dump({"new_gains": gains, "orig_gains": orig_gains,
               "tent_8cell_mean_gain": mg, "tent_8cell_worst": worst},
              open(config.OUTPUTS / "tent_8cell.json", "w"), indent=2)
    print("Saved tent_8cell.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
