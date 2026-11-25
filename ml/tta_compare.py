"""Baseline comparison + ablations for the paper's method section (runs on CPU so
it does not disturb GPU training). On the 4 original cells:

  Main table (seeds 42,1,2): No-adapt | AdaBN | Tent | Collapse-Gated (ours)
  Ablation A (seed 42): Partial-AdaBN statistic mixing, lam in {0,.25,.5,.75,1}
  Ablation B: tau-sensitivity of the collapse gate {.02,.05,.10}

Saves outputs/tta_compare.json.
"""
import json, os, importlib, statistics
import numpy as np
import torch

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # run ALONE (no concurrent torch job)
CELLS = [
    ("MobileNetV3 Canopy",      "mobilenet_v3_small", "paddy_doctor", "philippines"),
    ("MobileNetV3 Single-leaf", "mobilenet_v3_small", "kaggle_rice",  "riceleafs"),
    ("EfficientNet Canopy",     "efficientnet_b0",    "paddy_doctor", "philippines"),
    ("EfficientNet Single-leaf","efficientnet_b0",    "kaggle_rice",  "riceleafs"),
]
SEEDS = [42, 1, 2]
TAU = 0.05


def load(model_name, pair, seed, build_model):
    ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{seed}_{model_name}.pt"
    if not ck.exists():
        return None
    net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
    net.load_state_dict(torch.load(ck, map_location="cpu"))
    return net.to(DEVICE)


def min_share(net, ext, T):
    yt, yp = T._predict(net, ext, DEVICE)
    yp = np.array(yp)
    return min((yp == c).mean() for c in range(config.NUM_CLASSES)), T._metrics(yt, yp)["macro_f1"]


def main():
    from models import build_model
    from dg import adabn_adapt, tent_adapt, partial_adabn_adapt
    out = {"main": {}, "partial_lam": {}, "tau": {}, "seeds": SEEDS, "tau_used": TAU}

    for name, model_name, src, tgt in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"
        _, _, _, ext = T._loaders(64)

        na, ab, tn, gt = [], [], [], []
        for s in SEEDS:
            net = load(model_name, pair, s, build_model)
            if net is None:
                continue
            net.eval()
            ms, f_na = min_share(net, ext, T)
            # AdaBN
            net_a = load(model_name, pair, s, build_model); adabn_adapt(net_a, ext, DEVICE)
            f_ab = T._metrics(*T._predict(net_a, ext, DEVICE))["macro_f1"]
            # Tent
            net_t = load(model_name, pair, s, build_model); tent_adapt(net_t, ext, DEVICE)
            f_tn = T._metrics(*T._predict(net_t, ext, DEVICE))["macro_f1"]
            # Collapse-Gated (ours)
            f_gt = f_ab if ms < TAU else f_na
            na.append(f_na); ab.append(f_ab); tn.append(f_tn); gt.append(f_gt)
            print(f"  {name:26s} s{s}: no {f_na:.1f} | AdaBN {f_ab:.1f} | Tent {f_tn:.1f} | Gated {f_gt:.1f} (min-share {ms:.3f})")

        mean = lambda v: round(statistics.mean(v), 2) if v else None
        out["main"][name] = {"no_adapt": mean(na), "adabn": mean(ab),
                             "tent": mean(tn), "gated": mean(gt)}

        # ---- Ablation A: partial-lam on seed 42 ----
        lamres = {}
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            net_p = load(model_name, pair, 42, build_model)
            if net_p is None:
                break
            partial_adabn_adapt(net_p, ext, DEVICE, lam)
            lamres[str(lam)] = round(T._metrics(*T._predict(net_p, ext, DEVICE))["macro_f1"], 2)
        out["partial_lam"][name] = lamres
        print(f"  {name:26s} partial-lam: {lamres}")

    # ---- Ablation B: tau sensitivity (uses seed-42 min-shares from partial run is not stored;
    #      recompute quickly) ----
    for name, model_name, src, tgt in CELLS:
        os.environ["RICE_SOURCE_DOMAIN"] = src; os.environ["RICE_TARGET_DOMAIN"] = tgt
        importlib.reload(config)
        import train as T; importlib.reload(T)
        pair = f"{src}-{tgt}"; _, _, _, ext = T._loaders(64)
        shares = []
        for s in SEEDS:
            net = load(model_name, pair, s, build_model)
            if net is None:
                continue
            shares.append(min_share(net, ext, T)[0])
        out["tau"][name] = {f"tau={t}": sum(1 for x in shares if x < t) for t in (0.02, 0.05, 0.10)}

    (config.OUTPUTS / "tta_compare.json").write_text(json.dumps(out, indent=2))
    print("\n=== MAIN (mean external Macro-F1) ===")
    for n, r in out["main"].items():
        print(f"  {n:26s} no {r['no_adapt']:.1f} | AdaBN {r['adabn']:.1f} | Tent {r['tent']:.1f} | Gated {r['gated']:.1f}")
    def mg(k): return round(statistics.mean([out['main'][n][k]-out['main'][n]['no_adapt'] for n in out['main']]), 2)
    print(f"  mean gain -> AdaBN {mg('adabn'):+.2f} | Tent {mg('tent'):+.2f} | Gated {mg('gated'):+.2f}")
    print("Saved outputs/tta_compare.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    main()
