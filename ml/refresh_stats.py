"""Compile ALL refreshed paper numbers in one place after the 8-seed training +
cleaned-canopy re-runs. Pure CPU (reads JSONs). Prints a transcription sheet and
saves outputs/refreshed_numbers.json.

Sources (all must be refreshed first):
  seed_summary_<model>_<pair>.json  -- baseline/mixstyle per-seed internal+external
  confidence_gate.json              -- per-seed no/always/dent (8 cells, cleaned canopy)
  eata_sar_full.json, tent_all8.json-- TTA baselines
Canopy cells use the DE-DUPLICATED target (via the exclude env in those re-runs);
tab:dg keeps the full-target relative comparison (leakage cancels in baseline-vs-DG).
"""
import json
import statistics

from scipy import stats

import config

O = config.OUTPUTS

# display -> (model, pair, confidence_gate key)
CELLS = [
    ("MobileNetV3 Canopy (P->Ph)",  "mobilenet_v3_small", "paddy_doctor-philippines", "MNet paddy>phil"),
    ("MobileNetV3 Leaf (K->R)",     "mobilenet_v3_small", "kaggle_rice-riceleafs",    "MNet kaggle>rice"),
    ("MobileNetV3 Canopy->leaf",    "mobilenet_v3_small", "paddy_doctor-riceleafs",   "MNet paddy>rice"),
    ("MobileNetV3 Leaf->canopy",    "mobilenet_v3_small", "kaggle_rice-philippines",  "MNet kaggle>phil"),
    ("EfficientNet Canopy (P->Ph)", "efficientnet_b0",    "paddy_doctor-philippines", "ENet paddy>phil"),
    ("EfficientNet Leaf (K->R)",    "efficientnet_b0",    "kaggle_rice-riceleafs",    "ENet kaggle>rice"),
    ("EfficientNet Canopy->leaf",   "efficientnet_b0",    "paddy_doctor-riceleafs",   "ENet paddy>rice"),
    ("EfficientNet Leaf->canopy",   "efficientnet_b0",    "kaggle_rice-philippines",  "ENet kaggle>phil"),
]


def _load(name):
    p = O / name
    return json.loads(p.read_text()) if p.exists() else None


def holm(pvals):
    """Holm-Bonferroni corrected p-values, preserving input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); corr = [0.0] * m; run = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        corr[i] = min(1.0, run)
    return [round(c, 4) for c in corr]


def main():
    cg = _load("confidence_gate.json")
    es = _load("eata_sar_full.json")
    tent = _load("tent_all8.json")
    canopy = _load("canopy_dedup.json")
    out = {"baseline_table": [], "dg_table": [], "adabn_table": [],
           "tta_compare": {}, "gated_table": []}

    print("\n" + "=" * 70)
    print("TABLE: BASELINE (internal / external / gap)  [canopy=de-dup, new=8 seeds]")
    print("=" * 70)
    for disp, model, pair, cgk in CELLS:
        ss = _load(f"seed_summary_{model}_{pair}.json")
        internal = ss["methods"]["baseline"]["internal_mean"] if ss else None
        ext = cg["cells"][cgk]["no"] if cg and cgk in cg["cells"] else None
        nseed = cg["cells"][cgk]["n"] if cg and cgk in cg["cells"] else None
        gap = round(internal - ext, 2) if (internal is not None and ext is not None) else None
        out["baseline_table"].append({"cell": disp, "n": nseed, "internal": internal,
                                       "external": ext, "gap": gap})
        print(f"  {disp:30s} n={nseed}  int={internal}  ext={ext}  gap={gap}")

    print("\n" + "=" * 70)
    print("TABLE: DG (MixStyle vs baseline, external; paired t-test)  [new cells @8 seeds]")
    print("=" * 70)
    for disp, model, pair, cgk in CELLS:
        ss = _load(f"seed_summary_{model}_{pair}.json")
        if not ss or "mixstyle" not in ss["methods"]:
            continue
        b = [r["external"] for r in ss["methods"]["baseline"]["per_seed"]]
        mx = [r["external"] for r in ss["methods"]["mixstyle"]["per_seed"]]
        if len(b) == len(mx) and len(b) > 1:
            t, p = stats.ttest_rel(mx, b)
            d = round(statistics.mean(mx) - statistics.mean(b), 2)
            out["dg_table"].append({"cell": disp, "n": len(b), "delta": d, "p": round(float(p), 4)})
            print(f"  {disp:30s} n={len(b)}  MixStyle d={d:+.2f}  p={round(float(p),4)}")

    print("\n" + "=" * 70)
    print("TABLE: AdaBN (no-adapt -> AdaBN, external; paired t-test + Holm)  [8 cells]")
    print("=" * 70)
    rows, praw = [], []
    for disp, model, pair, cgk in CELLS:
        if not cg or cgk not in cg["cells"]:
            continue
        r = cg["cells"][cgk]["raw"]
        no, al = r["no"], r["always"]
        t, p = stats.ttest_rel(al, no)
        d = round(statistics.mean(al) - statistics.mean(no), 2)
        rows.append({"cell": disp, "n": len(no), "no": round(statistics.mean(no), 2),
                     "adabn": round(statistics.mean(al), 2), "delta": d, "p": round(float(p), 4)})
        praw.append(float(p))
    for row, hp in zip(rows, holm(praw)):
        row["p_holm"] = hp
        sig = "sig" if row["p"] < 0.05 else "n.s."
        hsig = "sig" if hp < 0.05 else "n.s."
        print(f"  {row['cell']:30s} {row['no']:.2f}->{row['adabn']:.2f}  d={row['delta']:+.2f} "
              f"p={row['p']:.4f}({sig}) Holm={hp:.4f}({hsig})")
    out["adabn_table"] = rows

    print("\n" + "=" * 70)
    print("TABLE: TTA COMPARISON (mean gain / worst-cell over 8 cells)")
    print("=" * 70)
    if cg:
        out["tta_compare"]["always_adabn"] = {"mean_gain": cg["mean_gain"]["always"],
                                              "worst": cg["worst_cell_delta"]["always"]}
        out["tta_compare"]["collapse_gated"] = {"mean_gain": cg["mean_gain"]["minshare"],
                                               "worst": cg["worst_cell_delta"]["minshare"]}
        out["tta_compare"]["conf_gated_0.1"] = {"mean_gain": cg["mean_gain"]["conf_0.1"],
                                              "worst": cg["worst_cell_delta"]["conf_0.1"]}
    if tent:
        out["tta_compare"]["tent"] = {"mean_gain": tent["_summary"]["mean_gain"],
                                     "worst": tent["_summary"]["worst_cell"]}
    if es:
        out["tta_compare"]["eata"] = {"mean_gain": es["mean_gain"]["eata"], "worst": es["worst_cell"]["eata"]}
        out["tta_compare"]["sar"] = {"mean_gain": es["mean_gain"]["sar"], "worst": es["worst_cell"]["sar"]}
    for k, v in out["tta_compare"].items():
        print(f"  {k:18s} mean {v['mean_gain']:+.2f}  worst {v['worst']:+.2f}")

    print("\n" + "=" * 70)
    print("TABLE: GATED (per-cell no / always / collapse / conf_0.1)")
    print("=" * 70)
    if cg:
        for disp, model, pair, cgk in CELLS:
            if cgk not in cg["cells"]:
                continue
            c = cg["cells"][cgk]
            g = {"cell": disp, "no": c["no"], "always": c["always"],
                 "collapse": c["minshare"], "conf_0.1": c["conf_0.1"]}
            out["gated_table"].append(g)
            print(f"  {disp:30s} no {c['no']:.1f} | always {c['always']:.1f} | "
                  f"coll {c['minshare']:.1f} | conf {c['conf_0.1']:.1f}")

    (O / "refreshed_numbers.json").write_text(json.dumps(out, indent=2))
    print("\nSaved outputs/refreshed_numbers.json")


if __name__ == "__main__":
    main()
