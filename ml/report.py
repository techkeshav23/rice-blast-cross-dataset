"""Phase 4 — consolidate baseline vs MixStyle into the headline cross-dataset
result: a comparison table, bar chart, confusion-matrix plots, a combined
results.json, and a metrics.json in the shape the app/backend consume.

Usage:
  python report.py
"""
import json

import numpy as np

import config

TAGS = [("baseline", "Baseline"), ("mixstyle", "+ MixStyle (ours)")]


def _load(tag):
    p = config.OUTPUTS / f"metrics_{tag}.json"
    if not p.exists():
        raise SystemExit(f"Missing {p}. Train the '{tag}' model first.")
    return json.loads(p.read_text())


def _plot_bars(results, out):
    import matplotlib.pyplot as plt
    labels = [name for _, name in TAGS]
    internal = [results[t]["internal_test"]["macro_f1"] for t, _ in TAGS]
    external = [results[t]["external_test"]["macro_f1"] for t, _ in TAGS]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, internal, w, label="Internal (source)", color="#9CC3A6")
    ax.bar(x + w / 2, external, w, label="External (unseen)", color="#1E7A4D")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-dataset generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    for i, v in enumerate(external):
        ax.text(x[i] + w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_confusion(tag, res, out):
    import matplotlib.pyplot as plt
    cm = np.array(res["external_test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(config.CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{tag} — external confusion")
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    results = {t: _load(t) for t, _ in TAGS}
    base, dg = results["baseline"], results["mixstyle"]
    gap_reduction = round(base["generalization_gap_macro_f1"]
                          - dg["generalization_gap_macro_f1"], 2)

    # ---- console table ----
    print("\n============== CROSS-DATASET RESULTS (Macro-F1) ==============")
    print(f"{'Model':22s} {'Internal':>9s} {'External':>9s} {'Gap':>7s}")
    for t, name in TAGS:
        r = results[t]
        print(f"{name:22s} {r['internal_test']['macro_f1']:9.2f} "
              f"{r['external_test']['macro_f1']:9.2f} "
              f"{r['generalization_gap_macro_f1']:7.2f}")
    print("-" * 61)
    print(f"Gap reduction by MixStyle: {gap_reduction} Macro-F1 points")
    print(f"External Macro-F1 lift    : "
          f"{round(dg['external_test']['macro_f1'] - base['external_test']['macro_f1'], 2)}")
    print("=" * 61)

    # ---- plots (optional, needs matplotlib) ----
    try:
        _plot_bars(results, config.OUTPUTS / "comparison.png")
        _plot_confusion("baseline", base, config.OUTPUTS / "confusion_baseline.png")
        _plot_confusion("mixstyle", dg, config.OUTPUTS / "confusion_mixstyle.png")
        print("Saved plots: comparison.png, confusion_baseline.png, confusion_mixstyle.png")
    except Exception as e:
        print(f"(plots skipped: {e})")

    # ---- combined results.json ----
    combined = {
        "baseline": base, "mixstyle": dg,
        "gap_reduction_macro_f1": gap_reduction,
    }
    (config.OUTPUTS / "results.json").write_text(json.dumps(combined, indent=2))

    # ---- metrics.json in app/backend shape ----
    app_metrics = {
        "primary_metric": "macro_f1",
        "note": "Results on synthetic source→target domains (pipeline demo).",
        "baseline": {
            "in_set": base["internal_test"]["macro_f1"],
            "unseen": base["external_test"]["macro_f1"],
            "gap": base["generalization_gap_macro_f1"],
        },
        "mixstyle": {
            "in_set": dg["internal_test"]["macro_f1"],
            "unseen": dg["external_test"]["macro_f1"],
            "gap": dg["generalization_gap_macro_f1"],
        },
        "per_class_recall_unseen": dg["external_test"]["per_class_recall"],
        "version": "v0.2-synthetic",
    }
    (config.OUTPUTS / "metrics.json").write_text(json.dumps(app_metrics, indent=2))

    # also push to the backend so /metrics serves real numbers
    backend_metrics = config.ROOT.parent / "backend" / "models" / "metrics.json"
    if backend_metrics.parent.exists():
        backend_metrics.write_text(json.dumps(app_metrics, indent=2))
        print(f"Updated backend metrics -> {backend_metrics}")

    print("Saved outputs/results.json and outputs/metrics.json")


if __name__ == "__main__":
    main()
