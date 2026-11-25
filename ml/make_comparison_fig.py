"""Assemble the full-seed method-comparison figure (fig5) + printed table from
saved jsons: no-adapt/full-AdaBN/gated (adaptive_adabn.json) and Tent (tent_full.json).
"""
import json, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

ad = json.load(open(config.OUTPUTS / "adaptive_adabn.json"))["cells"]
tent = json.load(open(config.OUTPUTS / "tent_full.json"))
labels = list(ad.keys())

rows = []
for l in labels:
    rows.append({"cell": l, "no_adapt": ad[l]["no_adapt"], "full": ad[l]["full_adabn"],
                 "tent": tent[l]["tent"], "gated": ad[l]["gated"]})

def mg(k): return round(statistics.mean([r[k]-r["no_adapt"] for r in rows]), 2)
print(f"{'Cell':26s} {'no':>6s} {'Full':>6s} {'Tent':>6s} {'Gated':>6s}")
for r in rows:
    print(f"{r['cell']:26s} {r['no_adapt']:6.2f} {r['full']:6.2f} {r['tent']:6.2f} {r['gated']:6.2f}")
print(f"mean gain -> Full {mg('full'):+.2f} | Tent {mg('tent'):+.2f} | Gated {mg('gated'):+.2f}")

x = np.arange(len(labels)); w = 0.2
series = [("No adaptation", "no_adapt", "#9e9e9e"),
          ("Full AdaBN", "full", "#ef9a9a"),
          ("Tent", "tent", "#90caf9"),
          ("Collapse-Gated (ours)", "gated", "#1b5e20")]
fig, ax = plt.subplots(figsize=(7.8, 4.0))
for i, (lab, key, col) in enumerate(series):
    ax.bar(x + (i-1.5)*w, [r[key] for r in rows], w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8.5)
ax.set_ylabel("External Macro-F1 (%)"); ax.set_ylim(0, 82)
ax.set_title("Collapse-Gated AdaBN: best mean gain, and never degrades", fontweight="bold", fontsize=11)
ax.legend(fontsize=8, loc="upper center", ncol=2); ax.grid(alpha=0.3, axis="y")
figs = config.OUTPUTS / "paper_figs"; figs.mkdir(exist_ok=True)
fig.savefig(figs / "fig5_selective.pdf", bbox_inches="tight", dpi=300)
fig.savefig(figs / "fig5_selective.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig5_selective (updated with Tent, full-seed)")
