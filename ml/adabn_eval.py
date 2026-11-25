"""Test AdaBN (test-time BatchNorm adaptation) on already-trained checkpoints.
Compares external Macro-F1 WITHOUT adaptation vs WITH AdaBN, for each seed of a
method, on the current SOURCE->TARGET pairing (set via env vars).

Usage (clean pairing):
  set RICE_SOURCE_DOMAIN=kaggle_rice & set RICE_TARGET_DOMAIN=riceleafs
  python adabn_eval.py --method baseline --seeds 42 1 2 3 4
"""
import argparse
import statistics

import torch

import config
from dg import adabn_adapt, to_mixstyle, to_randconv
from models import build_model
from train import _loaders, _metrics, _predict


def _load(tag, model_name, device, method):
    net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
    if method == "mixstyle":
        net = to_mixstyle(net)
    elif method == "randconv":
        net = to_randconv(net)
    ckpt = config.OUTPUTS / f"{tag}_{model_name}.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return net.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline")   # baseline | mixstyle | randconv
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pair = f"{config.SOURCE_DOMAIN}-{config.TARGET_DOMAIN}"
    _, _, _, ext = _loaders(args.batch)
    print(f"AdaBN test | {args.method} | {pair} | device={device}\n")

    plain, adapted = [], []
    for s in args.seeds:
        tag = f"{args.method}_{args.model}_{pair}_s{s}"
        net = _load(tag, args.model, device, args.method)

        net.eval()
        m0 = _metrics(*_predict(net, ext, device))
        adabn_adapt(net, ext, device)                    # in-place BN re-estimation
        m1 = _metrics(*_predict(net, ext, device))

        plain.append(m0["macro_f1"])
        adapted.append(m1["macro_f1"])
        print(f"  seed {s:2d}: external  {m0['macro_f1']:6.2f}  ->  AdaBN {m1['macro_f1']:6.2f}   "
              f"(+{m1['macro_f1']-m0['macro_f1']:.2f})")
        print(f"           recall no-adapt: {m0['per_class_recall']}")
        print(f"           recall AdaBN   : {m1['per_class_recall']}")

    def ms(v):
        return round(statistics.mean(v), 2), (round(statistics.stdev(v), 2) if len(v) > 1 else 0.0)
    pm, ps = ms(plain)
    am, as_ = ms(adapted)
    print("\n================= AdaBN SUMMARY (external Macro-F1) =================")
    print(f"  No adaptation : {pm} ± {ps}")
    print(f"  + AdaBN       : {am} ± {as_}")
    print(f"  Improvement   : {round(am - pm, 2)}")
    print("=" * 60)

    import json
    out = {
        "method": args.method, "model": args.model, "pairing": pair,
        "seeds": args.seeds,
        "no_adapt_mean": pm, "no_adapt_std": ps,
        "adabn_mean": am, "adabn_std": as_,
        "improvement": round(am - pm, 2),
        "per_seed_no_adapt": plain, "per_seed_adabn": adapted,
    }
    fn = config.OUTPUTS / f"adabn_{args.method}_{args.model}_{pair}.json"
    fn.write_text(json.dumps(out, indent=2))
    print(f"Saved {fn.name}")


if __name__ == "__main__":
    main()
