"""Re-arrange the already-downloaded datasets with a higher per-class cap
(use more data). Clears the old raw/<domain> folders first.

Usage:
  python rearrange.py --max-per-class 900
"""
import argparse
import shutil

import config
from fetch_real_data import arrange

# domain -> download subfolder
DOMAINS = {"paddy_doctor": "paddy", "philippines": "external"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=900)
    args = ap.parse_args()
    for dom, dl in DOMAINS.items():
        raw = config.DATA_RAW / dom
        if raw.exists():
            shutil.rmtree(raw)
        src = config.ROOT / "data" / "downloads" / dl
        if not src.exists():
            print(f"  skip {dom}: {src} not found")
            continue
        arrange(src, dom, args.max_per_class)


if __name__ == "__main__":
    main()
