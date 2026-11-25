"""Master consolidation across the WHOLE study -> outputs/final_results_v2.json.

Pulls together every experiment we ran so the thesis/paper has one authoritative
table:
  * 2 backbones     : MobileNetV3-Small, EfficientNet-B0
  * 2 pairings      : canopy (paddy_doctor->philippines),
                      single-leaf (kaggle_rice->riceleafs)
  * DG methods      : baseline, MixStyle, RandConv  (external Macro-F1, paired t vs baseline)
  * AdaBN           : test-time BN adaptation on the baseline (no-adapt vs AdaBN)

Everything is read from the per-seed JSON summaries already on disk; nothing is
re-trained here, so the numbers are exactly what the experiments produced.
"""
import json
from pathlib import Path

from scipy import stats

import config

O = config.OUTPUTS

BACKBONES = [("mobilenet_v3_small", "MobileNetV3-Small"),
             ("efficientnet_b0", "EfficientNet-B0")]
PAIRS = [("paddy_doctor-philippines", "Canopy (Paddy Doctor -> Philippines)"),
         ("kaggle_rice-riceleafs", "Single-leaf (anshulm -> riceleafs)")]
METHODS = ["baseline", "mixstyle", "randconv"]


def paired(a, b):
    """Return (mean_diff, t, p) for paired b-a, or None if not comparable."""
    if not a or not b or len(a) != len(b) or len(a) < 2:
        return None
    t, p = stats.ttest_rel(b, a)
    md = sum(bi - ai for ai, bi in zip(a, b)) / len(a)
    return round(md, 2), round(float(t), 2), round(float(p), 4)


def load_json(name):
    f = O / name
    return json.loads(f.read_text()) if f.exists() else None


def main():
    report = {"backbones": {}}
    print("=" * 90)
    print("  FULL CROSS-DATASET GENERALIZATION STUDY  --  external Macro-F1 (%)")
    print("=" * 90)

    for mtag, mname in BACKBONES:
        report["backbones"][mtag] = {"name": mname, "pairings": {}}
        for ptag, ptitle in PAIRS:
            ss = load_json(f"seed_summary_{mtag}_{ptag}.json")
            if not ss:
                continue
            s = ss["methods"]
            n_seeds = len(ss["seeds"])
            base_ext = [r["external"] for r in s["baseline"]["per_seed"]]

            print(f"\n### {mname}  |  {ptitle}   (n={n_seeds} seeds)")
            print(f"  {'method':10s} {'int':>7s} {'ext':>13s}   {'vs baseline':>28s}")
            block = {"n_seeds": n_seeds, "methods": {}}
            for m in METHODS:
                if m not in s:
                    continue
                v = s[m]
                ext = [r["external"] for r in v["per_seed"]]
                row = {"internal_mean": v["internal_mean"],
                       "external_mean": v["external_mean"],
                       "external_std": v["external_std"],
                       "gap_mean": v["gap_mean"],
                       "per_seed_external": ext}
                if m == "baseline":
                    print(f"  {m:10s} {v['internal_mean']:7.2f} "
                          f"{v['external_mean']:6.2f}+/-{v['external_std']:<5.2f}")
                else:
                    pr = paired(base_ext, ext)
                    if pr:
                        md, t, p = pr
                        sig = "SIG" if p < 0.05 else "n.s."
                        row["vs_baseline"] = {"delta": md, "t": t, "p": p, "sig": p < 0.05}
                        print(f"  {m:10s} {v['internal_mean']:7.2f} "
                              f"{v['external_mean']:6.2f}+/-{v['external_std']:<5.2f}   "
                              f"d={md:+6.2f} t={t:+5.2f} p={p:.3f} ({sig})")
                block["methods"][m] = row

            # ---- AdaBN on the baseline ----
            ab = load_json(f"adabn_baseline_{mtag}_{ptag}.json")
            if ab:
                pr = paired(ab["per_seed_no_adapt"], ab["per_seed_adabn"])
                md, t, p = pr if pr else (ab["improvement"], 0.0, 1.0)
                sig = "SIG" if p < 0.05 else "n.s."
                verdict = "HELPS" if ab["improvement"] > 0 else "HURTS"
                block["adabn_baseline"] = {
                    "no_adapt": ab["no_adapt_mean"], "adabn": ab["adabn_mean"],
                    "improvement": ab["improvement"], "t": t, "p": p,
                    "sig": p < 0.05, "verdict": verdict}
                print(f"  {'AdaBN':10s} baseline {ab['no_adapt_mean']:.2f} -> "
                      f"{ab['adabn_mean']:.2f}  "
                      f"(d={ab['improvement']:+.2f} t={t:+.2f} p={p:.3f} {sig} -> {verdict})")
            report["backbones"][mtag]["pairings"][ptag] = {"title": ptitle, **block}

    # ---- headline AdaBN grid ----
    print("\n" + "=" * 90)
    print("  AdaBN SUMMARY GRID (baseline, test-time BN adaptation)")
    print("=" * 90)
    print(f"  {'backbone':16s} {'pairing':14s} {'no-adapt':>9s} {'AdaBN':>8s} "
          f"{'delta':>8s} {'p':>8s}  verdict")
    grid = []
    for mtag, mname in BACKBONES:
        for ptag, ptitle in PAIRS:
            b = report["backbones"].get(mtag, {}).get("pairings", {}).get(ptag)
            if not b or "adabn_baseline" not in b:
                continue
            a = b["adabn_baseline"]
            short = "canopy" if "philippines" in ptag else "single-leaf"
            star = "*" if a["sig"] else " "
            print(f"  {mname:16s} {short:14s} {a['no_adapt']:9.2f} {a['adabn']:8.2f} "
                  f"{a['improvement']:+8.2f} {a['p']:8.4f}{star} {a['verdict']}")
            grid.append({"backbone": mname, "pairing": short, **a})
    report["adabn_grid"] = grid

    (O / "final_results_v2.json").write_text(json.dumps(report, indent=2))
    print("\nSaved outputs/final_results_v2.json")


if __name__ == "__main__":
    main()
