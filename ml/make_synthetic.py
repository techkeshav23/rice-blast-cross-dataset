"""Generate a synthetic rice-leaf dataset so the whole pipeline runs end-to-end
before the real Paddy Doctor data is available.

Two DOMAINS are produced to simulate the cross-dataset shift the project studies:
  - synthetic_source  : clean "lab-like" images  (train + internal test)
  - synthetic_target  : shifted "field-like" images (unseen external test)

Each domain has the 3 classes with visually distinguishable, learnable patterns:
  - Healthy     : green leaf, faint veins
  - Brown Spot  : green leaf + small round brown spots
  - Rice Blast  : green leaf + elongated grey-brown (eye-shaped) lesions

Usage:
  python make_synthetic.py --per-class 60
"""
import argparse
import random

import numpy as np
from PIL import Image, ImageDraw

import config
from utils import set_seed


def _leaf_base(size: int, domain: str, rng: random.Random) -> Image.Image:
    """A green leaf background; the target domain is colour/brightness shifted."""
    if domain == config.TARGET_DOMAIN:
        # Field-like: warmer, darker, more background noise (domain shift).
        base = (rng.randint(70, 100), rng.randint(95, 130), rng.randint(40, 70))
        noise_amp = 34
        bright = 0.82
    else:
        # Lab-like: cleaner, brighter green.
        base = (rng.randint(45, 70), rng.randint(120, 160), rng.randint(50, 80))
        noise_amp = 16
        bright = 1.0

    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[:, :] = base
    arr += np.random.normal(0, noise_amp, arr.shape)
    arr *= bright
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Faint central vein for all classes.
    d = ImageDraw.Draw(img)
    d.line([(size // 2, 8), (size // 2, size - 8)],
           fill=(30, 90, 40), width=2)
    return img


def _add_brown_spots(img: Image.Image, rng: random.Random) -> None:
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 12)):
        x, y = rng.randint(20, img.width - 20), rng.randint(20, img.height - 20)
        r = rng.randint(4, 9)
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=(rng.randint(95, 130), rng.randint(55, 80), rng.randint(20, 45)),
                  outline=(60, 35, 15))


def _add_blast_lesions(img: Image.Image, rng: random.Random) -> None:
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):
        x, y = rng.randint(24, img.width - 24), rng.randint(24, img.height - 24)
        w, h = rng.randint(16, 30), rng.randint(6, 11)  # elongated / eye-shaped
        # brown border
        d.ellipse([x - w, y - h, x + w, y + h],
                  fill=(rng.randint(120, 150), rng.randint(95, 120), rng.randint(80, 105)),
                  outline=(90, 60, 40), width=2)
        # grey centre
        d.ellipse([x - w + 5, y - h + 2, x + w - 5, y + h - 2],
                  fill=(rng.randint(170, 200), rng.randint(165, 195), rng.randint(155, 185)))


def make_domain(domain: str, per_class: int) -> None:
    out_root = config.DATA_RAW / domain
    rng = random.Random(config.SEED + hash(domain) % 1000)
    for cls in config.CLASSES:
        cls_dir = out_root / cls.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = _leaf_base(config.IMG_SIZE, domain, rng)
            if cls == "Brown Spot":
                _add_brown_spots(img, rng)
            elif cls == "Rice Blast":
                _add_blast_lesions(img, rng)
            # Healthy: base only
            img.save(cls_dir / f"{cls.replace(' ', '_')}_{i:04d}.jpg", quality=90)
    total = per_class * len(config.CLASSES)
    print(f"  {domain:20s} -> {total} images  ({per_class}/class)  at {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60,
                    help="images per class per domain")
    args = ap.parse_args()

    set_seed(config.SEED)
    print("Generating synthetic dataset (2 domains):")
    make_domain(config.SOURCE_DOMAIN, args.per_class)
    make_domain(config.TARGET_DOMAIN, args.per_class)
    print("Done.")


if __name__ == "__main__":
    main()
