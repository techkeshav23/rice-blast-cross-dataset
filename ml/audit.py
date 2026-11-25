"""Dataset audit: per-class counts, image sanity, and duplicate / near-duplicate
detection (a leakage risk called out in the research findings).

Usage:
  python audit.py --domain synthetic_source
"""
import argparse
import hashlib
from collections import Counter, defaultdict

from PIL import Image

import config


def _list_images(domain_dir):
    for cls_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name.replace("_", " ")
        for f in sorted(cls_dir.glob("*.jpg")):
            yield cls, f


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _ahash(path, hash_size: int = 8) -> str:
    """Average hash for near-duplicate detection."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size))
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p > avg else "0" for p in px)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def audit(domain: str) -> dict:
    domain_dir = config.DATA_RAW / domain
    if not domain_dir.exists():
        raise SystemExit(f"Domain not found: {domain_dir}. Run make_synthetic.py first.")

    items = list(_list_images(domain_dir))
    counts = Counter(cls for cls, _ in items)

    md5_map = defaultdict(list)
    ahash_map = defaultdict(list)
    sizes = Counter()
    for cls, f in items:
        md5_map[_md5(f)].append((cls, f.name))
        ahash_map[_ahash(f)].append((cls, f.name))
        with Image.open(f) as im:
            sizes[im.size] += 1

    exact_dups = {h: v for h, v in md5_map.items() if len(v) > 1}
    near_dups = {h: v for h, v in ahash_map.items() if len(v) > 1}
    # Cross-class duplicates are the dangerous ones (label leakage).
    cross_class = [v for v in near_dups.values()
                   if len({c for c, _ in v}) > 1]

    print(f"\n=== Audit: {domain} ===")
    print(f"Total images : {len(items)}")
    print("Per class    :")
    for cls in config.CLASSES:
        print(f"    {cls:12s}: {counts.get(cls, 0)}")
    print(f"Image sizes  : {dict(sizes)}")
    print(f"Exact duplicates (md5)      : {len(exact_dups)} group(s)")
    print(f"Near-duplicates (avg-hash)  : {len(near_dups)} group(s)")
    print(f"Cross-class near-duplicates : {len(cross_class)}  <-- leakage risk if > 0")
    if not exact_dups and not cross_class:
        print("Result       : OK — no exact dups, no cross-class leakage.")
    print("=" * (len(domain) + 12))

    return {
        "domain": domain,
        "total": len(items),
        "per_class": {c: counts.get(c, 0) for c in config.CLASSES},
        "exact_duplicate_groups": len(exact_dups),
        "near_duplicate_groups": len(near_dups),
        "cross_class_near_duplicates": len(cross_class),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=config.SOURCE_DOMAIN)
    args = ap.parse_args()
    audit(args.domain)


if __name__ == "__main__":
    main()
