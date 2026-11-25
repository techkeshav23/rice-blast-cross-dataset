"""8-cell comparison figure: No-adapt vs Always-AdaBN vs Confidence-Gated (tau_e=0.1).
Reads confidence_gate.json. Saves fig7_confgate (spanning figure for the paper).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

d = json.load(open(config.OUTPUTS / "confidence_gate.json"))["cells"]
labels = list(d.keys())
x = np.arange(len(labels)); w = 0.26
no = [d[l]["no"] for l in labels]
al = [d[l]["always"] for l in labels]
cg = [d[l]["conf_0.1"] for l in labels]

fig, ax = plt.subplots(figsize=(10.5, 3.8))
ax.bar(x - w, no, w, label="No adaptation", color="#9e9e9e")
ax.bar(x,     al, w, label="Always AdaBN", color="#ef9a9a")
ax.bar(x + w, cg, w, label="Confidence-Gated AdaBN (ours)", color="#1b5e20")
ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
ax.set_ylabel("External Macro-F1 (%)"); ax.set_ylim(0, 82)
ax.set_title("Confidence-Gated AdaBN across 4 pairings (8 cells): best mean gain, no degradation",
             fontweight="bold", fontsize=11)
ax.legend(fontsize=9, loc="upper right", ncol=1); ax.grid(alpha=0.3, axis="y")
figs = config.OUTPUTS / "paper_figs"; figs.mkdir(exist_ok=True)
fig.savefig(figs / "fig7_confgate.pdf", bbox_inches="tight", dpi=300)
fig.savefig(figs / "fig7_confgate.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig7_confgate")
