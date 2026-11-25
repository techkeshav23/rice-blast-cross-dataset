"""Probe label-free signals that might predict whether AdaBN HELPS or HURTS,
across all 8 cells. For each cell (mean over its seeds) compute:
  min_share, entropy(no-adapt), entropy(AdaBN), d_entropy = ent_adabn - ent_noadapt,
  bn_distance (source vs target running stats)
and pair with the true AdaBN delta (from adabn json).  Prints a table sorted by
delta so a separating signal is visible.
"""
import json, os, importlib, statistics, math
import numpy as np
import torch
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELLS = [
    ("MNet paddy>phil",  "mobilenet_v3_small", "paddy_doctor", "philippines", [42,1,2,3]),
    ("MNet kaggle>rice", "mobilenet_v3_small", "kaggle_rice",  "riceleafs",   [42,1,2,3]),
    ("ENet paddy>phil",  "efficientnet_b0",    "paddy_doctor", "philippines", [42,1,2,3]),
    ("ENet kaggle>rice", "efficientnet_b0",    "kaggle_rice",  "riceleafs",   [42,1,2,3]),
    ("MNet paddy>rice",  "mobilenet_v3_small", "paddy_doctor", "riceleafs",   [42,1,2,3]),
    ("ENet paddy>rice",  "efficientnet_b0",    "paddy_doctor", "riceleafs",   [42,1,2,3]),
    ("MNet kaggle>phil", "mobilenet_v3_small", "kaggle_rice",  "philippines", [42,1,2,3]),
    ("ENet kaggle>phil", "efficientnet_b0",    "kaggle_rice",  "philippines", [42,1,2,3]),
]


def ent3(dist):  # base-3 entropy of a class distribution (uniform=1.0)
    return -sum((p/100)*math.log((p/100)+1e-9, 3) for p in dist)


def main():
    from models import build_model
    from dg import adabn_adapt
    import torch.nn.functional as F
    rows = []
    for name, model_name, src, tgt, seeds in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(64)
        mins, e0s, e1s, bnds = [], [], [], []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                continue
            net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
            net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(DEVICE).eval()

            def pred_stats(m):
                ents, preds = [], []
                with torch.no_grad():
                    for x, *_ in ext:
                        logit = m(x.to(DEVICE)); prob = F.softmax(logit, 1)
                        ents += (-(prob*(prob+1e-9).log()).sum(1)).cpu().tolist()
                        preds += logit.argmax(1).cpu().tolist()
                preds = np.array(preds)
                dist = [100*(preds == c).mean() for c in range(config.NUM_CLASSES)]
                return min(d/100 for d in dist), float(np.mean(ents)), dist
            bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
            src_stat = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
            mn, e0, _ = pred_stats(net)
            adabn_adapt(net, ext, DEVICE)
            tgt_stat = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
            _, e1, _ = pred_stats(net)
            bnd = float(np.mean([((sm-tm)**2/(sv+1e-5)).mean().item()
                                 for (sm,sv),(tm,tv) in zip(src_stat, tgt_stat)]))
            mins.append(mn); e0s.append(e0); e1s.append(e1); bnds.append(bnd)
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        ab = json.load(open(config.OUTPUTS / f"adabn_baseline_{model_name}_{pair}.json"))
        rows.append({"cell": name, "delta": ab["improvement"],
                     "min_share": round(statistics.mean(mins), 3),
                     "ent0": round(statistics.mean(e0s), 3),
                     "ent1": round(statistics.mean(e1s), 3),
                     "d_ent": round(statistics.mean(e1s)-statistics.mean(e0s), 3),
                     "bn_dist": round(statistics.mean(bnds), 4)})

    rows.sort(key=lambda r: r["delta"])
    print(f"\n{'cell':16s} {'AdaBN_d':>8s} {'min_sh':>7s} {'ent0':>6s} {'ent1':>6s} {'d_ent':>7s} {'bn_dist':>8s}")
    for r in rows:
        flag = "HELP" if r["delta"] > 1 else ("HURT" if r["delta"] < -1 else "flat")
        print(f"{r['cell']:16s} {r['delta']:+8.2f} {r['min_share']:7.3f} {r['ent0']:6.3f} "
              f"{r['ent1']:6.3f} {r['d_ent']:+7.3f} {r['bn_dist']:8.4f}  {flag}")
    json.dump(rows, open(config.OUTPUTS / "signal_probe.json", "w"), indent=2)
    print("\nSaved signal_probe.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
