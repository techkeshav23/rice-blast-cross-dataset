"""Compile the full study into one results table with paired significance tests.

Reads:
  - seed_summary_<model>_<pair>.json   (baseline vs DG methods, per-seed)
  - adabn_<method>_<model>_<pair>.json (no-adapt vs AdaBN, per-seed)
Produces a printed report and outputs/final_results.json.
"""
import json
from pathlib import Path

from scipy import stats

import config

MODEL = "mobilenet_v3_small"
PAIRS = [("paddy_doctor-philippines", "Canopy (Paddy Doctor->Philippines)"),
         ("kaggle_rice-riceleafs", "Single-leaf (anshulm->riceleafs)")]
METHODS = ["baseline", "mixstyle", "randconv"]
O = config.OUTPUTS


def _paired(a, b):
    """Paired t-test on two equal-length lists; return (mean_diff, t, p)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    t, p = stats.ttest_rel(b, a)   # b - a
    md = sum(bi - ai for ai, bi in zip(a, b)) / len(a)
    return round(md, 2), round(float(t), 2), round(float(p), 4)


def main() -> None:
    report = {"pairings": {}}
    for pair, title in PAIRS:
        ss = O / f"seed_summary_{MODEL}_{pair}.json"
        if not ss.exists():
            print(f"(skip {pair}: no seed summary)")
            continue
        s = json.loads(ss.read_text())["methods"]
        base_ext = [r["external"] for r in s["baseline"]["per_seed"]]

        print(f"\n================ {title} ================")
        print(f"{'Method':10s} {'Ext Macro-F1':>16s}   {'vs baseline (paired t-test)':>32s}")
        block = {}
        for m in METHODS:
            if m not in s:
                continue
            ext = [r["external"] for r in s[m]["per_seed"]]
            mean, std = s[m]["external_mean"], s[m]["external_std"]
            if m == "baseline":
                print(f"{m:10s} {mean:6.2f} ± {std:<5.2f}")
                block[m] = {"external_mean": mean, "external_std": std}
            else:
                md, t, p = _paired(base_ext, ext)
                sig = "significant" if p < 0.05 else "NOT significant"
                print(f"{m:10s} {mean:6.2f} ± {std:<5.2f}   d={md:+.2f} t={t:+.2f} p={p:.3f} ({sig})")
                block[m] = {"external_mean": mean, "external_std": std,
                            "vs_baseline": {"delta": md, "t": t, "p": p}}

        # ---- AdaBN significance per method ----
        print("  -- AdaBN (test-time BN adaptation) --")
        for m in METHODS:
            af = O / f"adabn_{m}_{MODEL}_{pair}.json"
            if not af.exists():
                continue
            a = json.loads(af.read_text())
            md, t, p = _paired(a["per_seed_no_adapt"], a["per_seed_adabn"])
            sig = "significant" if p < 0.05 else "n.s."
            print(f"    {m:10s}: {a['no_adapt_mean']:.2f} -> {a['adabn_mean']:.2f} "
                  f"(+{a['improvement']:.2f}, t={t:+.2f}, p={p:.3f} {sig})")
            block.setdefault(m, {})["adabn"] = {
                "no_adapt": a["no_adapt_mean"], "adabn": a["adabn_mean"],
                "improvement": a["improvement"], "t": t, "p": p}
        report["pairings"][pair] = {"title": title, "methods": block}

    (O / "final_results.json").write_text(json.dumps(report, indent=2))
    print("\nSaved outputs/final_results.json")


if __name__ == "__main__":
    main()
