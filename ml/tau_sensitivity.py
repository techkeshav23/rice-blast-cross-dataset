"""Confidence-gate threshold (tau_H) sensitivity analysis.

Reviewer concern: tau_H=0.1 may be a hyper-tuned point rather than a robust
operating band. This script sweeps tau_H over a continuous range and reports the
8-cell mean gain and worst-cell delta at each value -- PURELY from the cached
per-seed data in confidence_gate.json (NO retraining, NO GPU).

The gate is applied PER SEED: for seed i, keep AdaBN if dent[i] >= -tau, else
revert to the no-adapt model. Cell gain = mean(gated) - mean(no); the 8-cell
summary is the mean gain across cells and the worst (min) per-cell delta.

Outputs:
  outputs/tau_sensitivity.json   -- grid of (tau, mean_gain, worst_cell)
  paper/figs/fig6_tau.{pdf,png}  -- sensitivity plot (also outputs/paper_figs/)
"""
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

import config

CELLS = json.load(open(config.OUTPUTS / "confidence_gate.json"))["cells"]


def cell_gain(cell: str, tau: float) -> float:
    """Per-seed gated gain for one cell at threshold tau."""
    r = CELLS[cell]["raw"]
    no, al, de = r["no"], r["always"], r["dent"]
    gated = [al[i] if de[i] >= -tau else no[i] for i in range(len(no))]
    return statistics.mean(gated) - statistics.mean(no)


def summary(tau: float):
    gains = [cell_gain(c, tau) for c in CELLS]
    return statistics.mean(gains), min(gains)


# --- always-AdaBN reference (tau -> inf: never revert) ---
always_gains = [statistics.mean(CELLS[c]["raw"]["always"]) - statistics.mean(CELLS[c]["raw"]["no"])
                for c in CELLS]
always_mean, always_worst = statistics.mean(always_gains), min(always_gains)

# --- fine continuous sweep ---
GRID = [round(t, 3) for t in np.arange(0.0, 0.501, 0.005)]
rows = []
for tau in GRID:
    mg, wc = summary(tau)
    rows.append({"tau": tau, "mean_gain": round(mg, 3), "worst_cell": round(wc, 3)})

# reviewer's requested operational band
BAND = (0.05, 0.30)
in_band = [r for r in rows if BAND[0] <= r["tau"] <= BAND[1]]
band_mean = [r["mean_gain"] for r in in_band]
band_worst = [r["worst_cell"] for r in in_band]

out = {
    "grid": rows,
    "always_adabn": {"mean_gain": round(always_mean, 3), "worst_cell": round(always_worst, 3)},
    "operational_band": {
        "range": list(BAND),
        "mean_gain_min": round(min(band_mean), 3),
        "mean_gain_max": round(max(band_mean), 3),
        "worst_cell_min": round(min(band_worst), 3),
        "worst_cell_max": round(max(band_worst), 3),
        "beats_always_on_mean_gain_everywhere": bool(min(band_mean) > always_mean),
        "beats_always_on_worst_cell_everywhere": bool(min(band_worst) > always_worst),
    },
}
(config.OUTPUTS / "tau_sensitivity.json").write_text(json.dumps(out, indent=2))

print("=== tau_H sensitivity (from cached per-seed data) ===")
print(f"always-AdaBN reference: mean gain {always_mean:+.2f}, worst-cell {always_worst:+.2f}")
print(f"operational band tau in [{BAND[0]}, {BAND[1]}]:")
print(f"  mean gain  stays in [{min(band_mean):+.2f}, {max(band_mean):+.2f}]")
print(f"  worst-cell stays in [{min(band_worst):+.2f}, {max(band_worst):+.2f}]")
print(f"  beats always-on in mean gain everywhere : {out['operational_band']['beats_always_on_mean_gain_everywhere']}")
print(f"  beats always-on in worst-cell everywhere: {out['operational_band']['beats_always_on_worst_cell_everywhere']}")
for t in (0.05, 0.1, 0.145, 0.2, 0.25, 0.3):
    mg, wc = summary(t)
    print(f"    tau={t:.3f}  mean gain {mg:+.2f}  worst-cell {wc:+.2f}")

# --- figure ---
taus = [r["tau"] for r in rows]
mg = [r["mean_gain"] for r in rows]
wc = [r["worst_cell"] for r in rows]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axvspan(BAND[0], BAND[1], color="#c8e6c9", alpha=0.45, label=f"operational band [{BAND[0]}, {BAND[1]}]")
ax.plot(taus, mg, color="#1b5e20", lw=2.2, label="Mean gain (8 cells)")
ax.plot(taus, wc, color="#c62828", lw=2.2, label="Worst-cell $\\Delta$")
ax.axhline(always_mean, color="#1b5e20", ls=":", lw=1.3, alpha=0.8,
           label=f"always-AdaBN mean ({always_mean:+.1f})")
ax.axhline(always_worst, color="#c62828", ls=":", lw=1.3, alpha=0.8,
           label=f"always-AdaBN worst ({always_worst:+.1f})")
ax.axvline(0.1, color="#37474f", ls="--", lw=1.2, alpha=0.9)
ax.axhline(0, color="k", lw=0.6, alpha=0.5)
ax.annotate("chosen\n$\\tau_H=0.1$", xy=(0.1, 2.3), xytext=(0.17, 2.1),
            fontsize=8.5, color="#37474f",
            arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0))
ax.set_xlabel("Confidence-gate threshold $\\tau_H$")
ax.set_ylabel("External Macro-F1 $\\Delta$ vs no-adapt (%)")
ax.set_xlim(0, 0.5)
ax.set_title("Confidence-gate is robust across a wide $\\tau_H$ band",
             fontweight="bold", fontsize=10.5)
ax.legend(fontsize=7.3, loc="lower right", framealpha=0.9)
ax.grid(alpha=0.3)

for d in (config.OUTPUTS / "paper_figs", config.ROOT.parent / "paper" / "figs"):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig6_tau.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(d / "fig6_tau.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("\nSaved outputs/tau_sensitivity.json and fig6_tau.{pdf,png} (paper/figs/ + outputs/paper_figs/)")
