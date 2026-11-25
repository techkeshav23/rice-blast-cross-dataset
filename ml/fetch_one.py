"""Download ONE public Kaggle dataset and arrange it into a named domain
(folder-per-class -> Rice Blast / Brown Spot / Healthy).

Usage:
  python fetch_one.py --dataset shayanriyaz/riceleafs --domain riceleafs --max-per-class 600
"""
import argparse
from pathlib import Path

import config
from fetch_real_data import _kaggle_api, arrange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Kaggle dataset slug")
    ap.add_argument("--domain", required=True, help="target folder name under data/raw")
    ap.add_argument("--max-per-class", type=int, default=600)
    args = ap.parse_args()

    api = _kaggle_api(Path(__file__).resolve().parent)
    d = config.ROOT / "data" / "downloads" / args.domain
    d.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.dataset} ...")
    api.dataset_download_files(args.dataset, path=str(d), unzip=True)
    arrange(d, args.domain, args.max_per_class)
    print(f"Done -> data/raw/{args.domain}")


if __name__ == "__main__":
    main()
