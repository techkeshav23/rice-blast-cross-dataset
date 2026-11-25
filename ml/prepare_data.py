"""Build leakage-safe train/val/test split manifests.

Rules (from the experimental protocol):
  - Splits are made at the FILE level, BEFORE any augmentation.
  - Stratified by class, deterministic (seeded).
  - The SOURCE domain is split train/val/test.
  - The TARGET domain is held out ENTIRELY as an unseen external test set.

Outputs a manifest JSON per domain in data/splits/.

Usage:
  python prepare_data.py
"""
import json
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

import config
from utils import set_seed


def _list_by_class(domain_dir):
    by_class = defaultdict(list)
    for cls_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name.replace("_", " ")
        for f in sorted(cls_dir.glob("*.jpg")):
            by_class[cls].append(str(f.relative_to(config.ROOT)))
    return by_class


def split_source(domain: str) -> dict:
    by_class = _list_by_class(config.DATA_RAW / domain)
    records = []
    for cls, files in by_class.items():
        # train vs temp(val+test)
        train, temp = train_test_split(
            files, train_size=config.TRAIN_RATIO,
            random_state=config.SEED, shuffle=True)
        rel = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val, test = train_test_split(
            temp, train_size=rel, random_state=config.SEED, shuffle=True)
        for f in train:
            records.append({"path": f, "class": cls, "split": "train"})
        for f in val:
            records.append({"path": f, "class": cls, "split": "val"})
        for f in test:
            records.append({"path": f, "class": cls, "split": "test"})
    return {"domain": domain, "role": "source", "records": records}


def hold_out_target(domain: str) -> dict:
    by_class = _list_by_class(config.DATA_RAW / domain)
    records = [{"path": f, "class": cls, "split": "external_test"}
               for cls, files in by_class.items() for f in files]
    return {"domain": domain, "role": "external", "records": records}


def _summary(manifest: dict) -> None:
    by_split = defaultdict(Counter)
    for r in manifest["records"]:
        by_split[r["split"]][r["class"]] += 1
    print(f"\n{manifest['domain']} ({manifest['role']}):")
    for split, counter in by_split.items():
        total = sum(counter.values())
        detail = ", ".join(f"{c}:{counter[c]}" for c in config.CLASSES)
        print(f"    {split:14s} {total:4d}  ({detail})")


def _save(manifest: dict) -> None:
    out = config.SPLITS_DIR / f"{manifest['domain']}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"    saved -> {out.relative_to(config.ROOT)}")


def main() -> None:
    set_seed(config.SEED)
    print("Building leakage-safe split manifests...")

    src = split_source(config.SOURCE_DOMAIN)
    _summary(src)
    _save(src)

    tgt = hold_out_target(config.TARGET_DOMAIN)
    _summary(tgt)
    _save(tgt)

    print("\nDone. Train only on SOURCE 'train'; never touch TARGET during training.")


if __name__ == "__main__":
    main()
