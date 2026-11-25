"""Run baseline vs one-or-more DG methods across multiple seeds and report
mean +/- std, with each method compared to the baseline.

Splits stay fixed (created once by prepare_data.py); only training randomness
(init, augmentation order, MixStyle/RandConv randomness) varies per seed.

Usage:
  python run_seeds.py --methods none mixstyle randconv --seeds 42 1 2 3 4 --epochs 20
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import config


def _keyname(dg: str) -> str:
    return "baseline" if dg == "none" else dg


def _run(dg: str, tag: str, seed: int, epochs: int, batch: int, model: str) -> dict:
    mfile = config.OUTPUTS / f"metrics_{tag}.json"
    if mfile.exists():
        print(f"  [resume] {tag} already trained -> reusing saved metrics")
    else:
        subprocess.run(
            [sys.executable, "train.py", "--dg", dg, "--tag", tag, "--model", model,
             "--seed", str(seed), "--epochs", str(epochs), "--batch", str(batch)],
            check=True, env={**os.environ},
        )
    m = json.loads(mfile.read_text())
    return {
        "seed": seed,
        "internal": m["internal_test"]["macro_f1"],
        "external": m["external_test"]["macro_f1"],
        "gap": m["generalization_gap_macro_f1"],
    }


def _stat(rows, key):
    vals = [r[key] for r in rows]
    mean = round(statistics.mean(vals), 2)
    std = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--methods", nargs="+", default=["none", "mixstyle"],
                    help="DG methods to run ('none' = baseline)")
    args = ap.parse_args()

    pair = f"{config.SOURCE_DOMAIN}-{config.TARGET_DOMAIN}"
    print(f"Run: model={args.model} methods={args.methods} seeds={args.seeds} "
          f"epochs={args.epochs} batch={args.batch}")
    print(f"Domains: source={config.SOURCE_DOMAIN}  target={config.TARGET_DOMAIN}\n")

    results = {_keyname(m): [] for m in args.methods}
    for s in args.seeds:
        for dg in args.methods:
            name = _keyname(dg)
            print(f"\n########## {args.model} | {pair} | {name} | seed {s} ##########")
            results[name].append(
                _run(dg, f"{name}_{args.model}_{pair}_s{s}", s, args.epochs, args.batch, args.model))

    # ---- aggregate ----
    print("\n\n================= MULTI-SEED SUMMARY (Macro-F1 %) =================")
    print(f"{'Method':16s} {'Internal':>15s} {'External':>15s} {'Gap':>14s}")
    summary = {}
    for dg in args.methods:
        name = _keyname(dg)
        rows = results[name]
        im, isd = _stat(rows, "internal")
        em, esd = _stat(rows, "external")
        gm, gsd = _stat(rows, "gap")
        summary[name] = {
            "internal_mean": im, "internal_std": isd,
            "external_mean": em, "external_std": esd,
            "gap_mean": gm, "gap_std": gsd, "per_seed": rows,
        }
        label = "Baseline" if name == "baseline" else f"+ {name}"
        print(f"{label:16s} {im:6.2f} ± {isd:<5.2f} {em:6.2f} ± {esd:<5.2f} "
              f"{gm:6.2f} ± {gsd:<5.2f}")
    print("-" * 62)

    # ---- each method vs baseline ----
    comparisons = {}
    base = summary.get("baseline")
    if base:
        for name, st in summary.items():
            if name == "baseline":
                continue
            gr = round(base["gap_mean"] - st["gap_mean"], 2)
            el = round(st["external_mean"] - base["external_mean"], 2)
            comparisons[name] = {"gap_reduction": gr, "external_lift": el}
            print(f"  {name:12s} vs baseline:  gap reduction {gr:+.2f},  external lift {el:+.2f}")
    print("=" * 62)

    out = {
        "model": args.model, "seeds": args.seeds, "epochs": args.epochs,
        "batch": args.batch, "source": config.SOURCE_DOMAIN,
        "target": config.TARGET_DOMAIN, "methods": summary,
        "vs_baseline": comparisons,
    }
    sname = f"seed_summary_{args.model}_{pair}.json"
    (config.OUTPUTS / sname).write_text(json.dumps(out, indent=2))
    print(f"\nSaved outputs/{sname}")

    # ---- push deployed backbone + official pairing to app/backend ----
    if (args.model != "mobilenet_v3_small" or config.SOURCE_DOMAIN != "paddy_doctor"
            or "mixstyle" not in summary or "baseline" not in summary):
        return
    app_metrics = {
        "primary_metric": "macro_f1",
        "note": f"Real cross-dataset, {config.SOURCE_DOMAIN} -> {config.TARGET_DOMAIN}, "
                f"mean of {len(args.seeds)} seeds (+/- std).",
        "baseline": {"in_set": base["internal_mean"], "unseen": base["external_mean"],
                     "gap": base["gap_mean"]},
        "mixstyle": {"in_set": summary["mixstyle"]["internal_mean"],
                     "unseen": summary["mixstyle"]["external_mean"],
                     "gap": summary["mixstyle"]["gap_mean"]},
        "std": {"baseline_gap": base["gap_std"],
                "mixstyle_gap": summary["mixstyle"]["gap_std"]},
        "seeds": args.seeds, "version": "v0.3-real-multiseed",
    }
    (config.OUTPUTS / "metrics.json").write_text(json.dumps(app_metrics, indent=2))
    backend = config.ROOT.parent / "backend" / "models" / "metrics.json"
    if backend.parent.exists():
        backend.write_text(json.dumps(app_metrics, indent=2))
        print(f"Updated backend metrics -> {backend}")


if __name__ == "__main__":
    main()
