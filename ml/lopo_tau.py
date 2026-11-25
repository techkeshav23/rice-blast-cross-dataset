"""Leave-One-Pairing-Out cross-validation for the confidence-gate threshold tau_H.
Addresses the "threshold tuned on the evaluation cells" critique: for each held-out
pairing, tau_H is chosen ONLY on the other three pairings, then applied to the
held-out pairing. Reports the OUT-OF-SAMPLE mean gain and worst-cell delta.
Pure Python (reads confidence_gate.json raw per-seed data).
"""
import json, statistics
import numpy as np
import config

cells = json.load(open(config.OUTPUTS / "confidence_gate.json"))["cells"]
# group the 8 cells into 4 pairings (by target-source suffix)
PAIRINGS = {
    "P>Ph": ["MNet paddy>phil", "ENet paddy>phil"],
    "K>R":  ["MNet kaggle>rice", "ENet kaggle>rice"],
    "P>R":  ["MNet paddy>rice", "ENet paddy>rice"],
    "K>Ph": ["MNet kaggle>phil", "ENet kaggle>phil"],
}
GRID = [round(x, 3) for x in np.arange(0.05, 0.51, 0.025)]


def gated_gain(cell, tau):
    r = cells[cell]["raw"]
    no, al, de = r["no"], r["always"], r["dent"]
    gated = [al[i] if de[i] >= -tau else no[i] for i in range(len(no))]
    return statistics.mean(gated) - statistics.mean(no), statistics.mean(gated) - statistics.mean(no)


def cell_gain(cell, tau):
    r = cells[cell]["raw"]
    no, al, de = r["no"], r["always"], r["dent"]
    gated = [al[i] if de[i] >= -tau else no[i] for i in range(len(no))]
    return statistics.mean(gated) - statistics.mean(no)


oos_gains, oos_worst, chosen = [], [], {}
for held, hcells in PAIRINGS.items():
    train = [c for p, cs in PAIRINGS.items() if p != held for c in cs]
    # standard CV: choose tau maximizing TRAIN mean gain, tie-break by best worst-cell.
    best = None
    for tau in GRID:
        gains = [cell_gain(c, tau) for c in train]
        score = (round(statistics.mean(gains), 4), round(min(gains), 4))
        if best is None or score > best[0]:
            best = (score, tau)
    tau = best[1]; chosen[held] = tau
    hg = [cell_gain(c, tau) for c in hcells]
    oos_gains += hg; oos_worst += hg
    print(f"held-out {held:5s}: tau*={tau:.3f} (from other 3 pairings)  "
          f"OOS cell gains {[round(g,2) for g in hg]}")

print(f"\n=== LEAVE-ONE-PAIRING-OUT (out-of-sample) ===")
print(f"  chosen tau per fold: {chosen}")
print(f"  OOS mean gain  : {statistics.mean(oos_gains):+.2f}")
print(f"  OOS worst-cell : {min(oos_worst):+.2f}")
# compare to always-AdaBN OOS (same cells)
always = [cells[c]['always']-cells[c]['no'] for p in PAIRINGS for c in PAIRINGS[p]]
print(f"  (always-AdaBN mean {statistics.mean(always):+.2f}, worst {min(always):+.2f})")
json.dump({"chosen_tau": chosen, "oos_mean_gain": round(statistics.mean(oos_gains), 2),
           "oos_worst_cell": round(min(oos_worst), 2)},
          open(config.OUTPUTS / "lopo_tau.json", "w"), indent=2)
print("Saved lopo_tau.json")
