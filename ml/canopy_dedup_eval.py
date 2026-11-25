"""Re-evaluate the CANOPY pairing (Paddy Doctor -> Philippines) on a DE-DUPLICATED
target, excluding the 32 pHash near-duplicate Philippines images that also appear in
the Paddy Doctor source (see cross_leakage.py / canopy_leak_exclude.txt).

ONLY the canopy external_test set is filtered; every other cell is untouched (its
target was already audited clean). Pure inference on the existing trained
checkpoints -- NO retraining. Needs torch + (ideally) GPU.

Outputs outputs/canopy_dedup.json: old vs de-duplicated external Macro-F1 for the
baseline and AdaBN, with a paired t-test, for both backbones. Use these numbers to
refresh the two Canopy (P->Ph) rows in paper tables tab:baseline / tab:adabn /
tab:gated / tab:ttacompare.

  python canopy_dedup_eval.py
"""
import json
import os
import statistics

import config

EXCLUDE_FILE = config.OUTPUTS / "canopy_leak_exclude.txt"

CELLS = [
    ("MobileNetV3 Canopy", "mobilenet_v3_small", [42, 1, 2, 3, 4, 5, 6, 7],
     {"no": 64.55, "adabn": 69.70}),
    ("EfficientNet Canopy", "efficientnet_b0", [42, 1, 2, 3, 4, 5],
     {"no": 71.11, "adabn": 65.15}),
]


def main():
    import torch
    from torch.utils.data import DataLoader
    from scipy.stats import ttest_rel

    from data import SplitDataset
    from models import build_model
    from dg import adabn_adapt
    import train as T

    excluded = {ln.strip() for ln in EXCLUDE_FILE.read_text().splitlines() if ln.strip()}
    print(f"Excluding {len(excluded)} leaked Philippines target images.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["RICE_SOURCE_DOMAIN"] = "paddy_doctor"
    os.environ["RICE_TARGET_DOMAIN"] = "philippines"
    import importlib; importlib.reload(config)

    class CleanExt(SplitDataset):
        """external_test split with the leaked basenames removed."""
        def __init__(self):
            super().__init__(config.SPLITS_DIR / "philippines.json", "external_test")
            before = len(self.records)
            self.records = [r for r in self.records
                            if os.path.basename(r["path"]) not in excluded]
            print(f"  external_test: {before} -> {len(self.records)} after de-dup")

    clean = CleanExt()
    bs = int(os.environ.get("RICE_TENT_BS", "32"))
    nw = int(os.environ.get("RICE_NUM_WORKERS", "0"))

    out = {}
    for name, model_name, seeds, old in CELLS:
        pair = "paddy_doctor-philippines"
        no_vals, ada_vals = [], []
        for s in seeds:
            ck = config.OUTPUTS / f"baseline_{model_name}_{pair}_s{s}_{model_name}.pt"
            if not ck.exists():
                print(f"    (missing {ck.name})"); continue

            def fresh():
                n = build_model(model_name, config.NUM_CLASSES, pretrained=False)
                n.load_state_dict(torch.load(ck, map_location="cpu"))
                return n.to(device)

            loader = DataLoader(clean, batch_size=bs, num_workers=nw)
            n0 = fresh(); n0.eval()
            no_vals.append(T._metrics(*T._predict(n0, loader, device))["macro_f1"])
            del n0
            if device == "cuda":
                torch.cuda.empty_cache()
            na = fresh()
            adabn_adapt(na, loader, device)
            ada_vals.append(T._metrics(*T._predict(na, loader, device))["macro_f1"])
            del na
            if device == "cuda":
                torch.cuda.empty_cache()

        def summ(v):
            return (round(statistics.mean(v), 2),
                    round(statistics.pstdev(v), 2) if len(v) > 1 else 0.0)
        no_m, no_sd = summ(no_vals)
        ad_m, ad_sd = summ(ada_vals)
        p = float(ttest_rel(ada_vals, no_vals).pvalue) if len(no_vals) > 1 else float("nan")
        out[name] = {
            "n": len(no_vals),
            "old_external_no_adapt": old["no"], "dedup_external_no_adapt": no_m,
            "dedup_no_adapt_std": no_sd,
            "old_external_adabn": old["adabn"], "dedup_external_adabn": ad_m,
            "dedup_adabn_std": ad_sd,
            "dedup_adabn_delta": round(ad_m - no_m, 2),
            "dedup_adabn_p": round(p, 4),
        }
        print(f"\n  {name}:")
        print(f"    no-adapt : {old['no']:.2f} (old) -> {no_m:.2f}+-{no_sd:.2f} (de-dup)")
        print(f"    AdaBN    : {old['adabn']:.2f} (old) -> {ad_m:.2f}+-{ad_sd:.2f} (de-dup), "
              f"delta {ad_m - no_m:+.2f}, p={p:.4f}")

    (config.OUTPUTS / "canopy_dedup.json").write_text(json.dumps(out, indent=2))
    print("\nSaved outputs/canopy_dedup.json")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0")
    os.environ.setdefault("RICE_CACHE", "1")
    main()
