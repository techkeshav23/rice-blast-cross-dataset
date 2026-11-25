"""Train a baseline classifier and evaluate it on the internal test split AND
the unseen external (target) domain — reporting the generalization gap.

Usage:
  python train.py --model mobilenet_v3_small --epochs 8 --tag baseline
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score)
from torch.utils.data import DataLoader

import config
from data import SplitDataset
from dg import to_mixstyle, to_randconv
from models import build_model
from utils import set_seed


def _loaders(batch: int):
    src = config.SPLITS_DIR / f"{config.SOURCE_DOMAIN}.json"
    tgt = config.SPLITS_DIR / f"{config.TARGET_DOMAIN}.json"
    # Parallel data loading so the GPU is not starved by single-threaded
    # image decode. NOTE: on Windows each worker re-imports CUDA torch (~2-3 GB),
    # so keep the count low to avoid "paging file too small" (WinError 1455).
    nw = int(os.environ.get("RICE_NUM_WORKERS", "2"))
    common = dict(num_workers=nw, pin_memory=True,
                  persistent_workers=nw > 0)
    train = DataLoader(SplitDataset(src, "train", train=True),
                       batch_size=batch, shuffle=True, **common)
    val = DataLoader(SplitDataset(src, "val"), batch_size=batch, **common)
    test = DataLoader(SplitDataset(src, "test"), batch_size=batch, **common)
    ext = DataLoader(SplitDataset(tgt, "external_test"), batch_size=batch, **common)
    return train, val, test, ext


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return ys, ps


def _metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100, 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "per_class_recall": {
            c: round(float(r) * 100, 2) for c, r in zip(
                config.CLASSES,
                recall_score(y_true, y_pred, average=None,
                             labels=list(range(config.NUM_CLASSES)), zero_division=0))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(config.NUM_CLASSES))).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--dg", choices=["none", "mixstyle", "randconv"], default="none",
                    help="domain-generalization method")
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="training seed (splits stay fixed; only training varies)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {args.model} | dg: {args.dg} | epochs: {args.epochs}")

    train_dl, val_dl, test_dl, ext_dl = _loaders(args.batch)
    model = build_model(args.model, config.NUM_CLASSES)
    if args.dg == "mixstyle":
        model = to_mixstyle(model)
    elif args.dg == "randconv":
        model = to_randconv(model)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        yv, pv = _predict(model, val_dl, device)
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0) * 100
        print(f"  epoch {ep:2d}/{args.epochs}  loss={running/len(train_dl.dataset):.3f}"
              f"  val_macroF1={val_f1:.2f}  ({time.time()-t0:.1f}s)")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation: internal test vs unseen external test.
    internal = _metrics(*_predict(model, test_dl, device))
    external = _metrics(*_predict(model, ext_dl, device))
    gap = round(internal["macro_f1"] - external["macro_f1"], 2)

    print("\n================ RESULTS ================")
    print(f"  Internal (source) Macro-F1 : {internal['macro_f1']}")
    print(f"  External (unseen) Macro-F1 : {external['macro_f1']}")
    print(f"  Generalization gap         : {gap}  (lower is better)")
    print("=========================================")

    config.OUTPUTS.mkdir(exist_ok=True)
    ckpt = config.OUTPUTS / f"{args.tag}_{args.model}.pt"
    torch.save(model.state_dict(), ckpt)
    result = {
        "tag": args.tag, "model": args.model, "dg": args.dg,
        "epochs": args.epochs, "seed": args.seed,
        "internal_test": internal, "external_test": external,
        "generalization_gap_macro_f1": gap,
    }
    (config.OUTPUTS / f"metrics_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"Saved checkpoint -> {ckpt.relative_to(config.ROOT)}")
    print(f"Saved metrics    -> outputs/metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
