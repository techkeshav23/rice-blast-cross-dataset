"""Download the REAL datasets from Kaggle and arrange them into the pipeline's
folder layout (folder-per-class), mapping each source's class folders onto the
project's three classes: Rice Blast, Brown Spot, Healthy.

  Source domain  : Paddy Doctor  (Kaggle competition 'paddy-disease-classification')
  External domain: Philippines Rice Diseases (Kaggle 'shrupyag001/philippines-rice-diseases')

PREREQUISITES
  1. pip install kaggle
  2. Put your Kaggle API token 'kaggle.json' in this ml/ folder (or ~/.kaggle/).
     (Kaggle -> Account -> Create New API Token)
  3. Accept the competition rules once in a browser:
     https://www.kaggle.com/c/paddy-disease-classification/rules
     (competition files cannot be pulled via API until the rules are accepted)

USAGE
  python fetch_real_data.py --max-per-class 600
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import config

# our_class -> keywords that a source sub-folder name may contain (lowercased)
CLASS_KEYWORDS = {
    "Rice Blast": ["blast"],          # 'blast', 'rice blast', 'leaf blast'
    "Brown Spot": ["brown"],          # 'brown spot', 'brown_spot', 'brownspot'
    "Healthy": ["healthy", "normal"], # Paddy Doctor uses 'normal'
}
# folders to never treat as a class (avoid false matches)
EXCLUDE = ["bacterial", "blight", "tungro", "smut", "scald", "hispa",
           "sheath", "streak", "mildew", "borer", "roller", "test_images"]


def _kaggle_api(creds_dir: Path):
    os.environ["KAGGLE_CONFIG_DIR"] = str(creds_dir)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        sys.exit(f"kaggle package not installed ({e}). Run: pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    return api


def _find_creds() -> Path:
    for c in [config.ROOT / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"]:
        if c.exists():
            return c.parent
    sys.exit("kaggle.json not found. Put it in ml/ or ~/.kaggle/ (see header).")


def _images_in(folder: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def _class_dirs(root: Path):
    """Yield (dir, images) for leaf-ish directories that directly hold images."""
    for d in root.rglob("*"):
        if d.is_dir():
            imgs = [p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if imgs:
                yield d, imgs


def arrange(extracted_root: Path, domain: str, max_per_class: int) -> None:
    out_root = config.DATA_RAW / domain
    print(f"\nArranging '{domain}' from {extracted_root} ...")
    counts = {c: 0 for c in config.CLASSES}
    for d, imgs in _class_dirs(extracted_root):
        name = d.name.lower()
        if any(x in name for x in EXCLUDE):
            continue
        for cls, kws in CLASS_KEYWORDS.items():
            if any(k in name for k in kws):
                dst = out_root / cls.replace(" ", "_")
                dst.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    if counts[cls] >= max_per_class:
                        break
                    target = dst / f"{domain}_{cls.replace(' ', '_')}_{counts[cls]:05d}{img.suffix.lower()}"
                    if not target.exists():
                        shutil.copy2(img, target)
                        counts[cls] += 1
                break
    for c in config.CLASSES:
        print(f"    {c:12s}: {counts[c]} images -> {out_root / c.replace(' ', '_')}")
    if sum(counts.values()) == 0:
        print("    WARNING: no images matched. Inspect the extracted folder names.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=600,
                    help="cap images per class per domain (keeps CPU training feasible)")
    ap.add_argument("--external", default="shrupyag001/philippines-rice-diseases",
                    help="Kaggle dataset slug for the external/target domain")
    ap.add_argument("--source-dataset", default="",
                    help="use a PUBLIC Kaggle dataset slug as source (skips the "
                         "competition; e.g. anshulm257/rice-disease-dataset)")
    ap.add_argument("--source-domain", default="paddy_doctor",
                    help="folder name for the arranged source domain")
    args = ap.parse_args()

    creds_dir = _find_creds()
    api = _kaggle_api(creds_dir)
    dl = config.ROOT / "data" / "downloads"
    dl.mkdir(parents=True, exist_ok=True)

    # 1) Source
    if args.source_dataset:
        print(f"Downloading public source dataset ({args.source_dataset}) ...")
        paddy_dir = dl / "source_public"
        paddy_dir.mkdir(exist_ok=True)
        try:
            api.dataset_download_files(args.source_dataset, path=str(paddy_dir), unzip=True)
        except Exception as e:
            print(f"  Source dataset download failed: {e}")
    else:
        print("Downloading Paddy Doctor (competition: paddy-disease-classification) ...")
        paddy_dir = dl / "paddy"
        paddy_dir.mkdir(exist_ok=True)
        try:
            api.competition_download_files("paddy-disease-classification", path=str(paddy_dir))
            for z in paddy_dir.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(paddy_dir)
        except Exception as e:
            print(f"  Paddy Doctor download failed: {e}")
            print("  -> Have you accepted the competition rules in a browser? (see header)")

    # 2) External: Philippines Rice Diseases (or user-chosen slug)
    print(f"Downloading external dataset ({args.external}) ...")
    ext_dir = dl / "external"
    ext_dir.mkdir(exist_ok=True)
    try:
        api.dataset_download_files(args.external, path=str(ext_dir), unzip=True)
    except Exception as e:
        print(f"  External download failed: {e}")

    # 3) Arrange both into ml/data/raw/<domain>/<Class>/
    arrange(paddy_dir, args.source_domain, args.max_per_class)
    arrange(ext_dir, "philippines", args.max_per_class)

    print("\nDone. Next (real run):")
    print('  $env:RICE_SOURCE_DOMAIN="paddy_doctor"; $env:RICE_TARGET_DOMAIN="philippines"')
    print("  python audit.py --domain paddy_doctor")
    print("  python prepare_data.py")
    print("  python train.py --dg none --tag baseline")
    print("  python train.py --dg mixstyle --tag mixstyle")
    print("  python report.py")


if __name__ == "__main__":
    main()
