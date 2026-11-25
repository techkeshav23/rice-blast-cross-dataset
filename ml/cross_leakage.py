"""Cross-dataset leakage audit (reviewer concern 2a).

Checks whether the two Kaggle datasets share near-duplicate images with Paddy
Doctor (would mean the "unseen" target was partly seen in training) and whether any
source->target pairing overlaps (would inflate external Macro-F1). Uses a 64-bit
average-hash and Hamming distance -- the same near-duplicate signal as audit.py,
extended ACROSS datasets. Pure CPU, no GPU, no training.

exact-dup  : Hamming distance 0 (identical down-sampled luminance)
near-dup   : Hamming distance <= 5 bits (visually near-identical)

Outputs outputs/cross_leakage.json and prints a matrix.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

import config

DATASETS = ["paddy_doctor", "philippines", "kaggle_rice", "riceleafs"]
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
NEAR_BITS = 5  # out of 64; a strict perceptual-hash threshold (very low false-positive)

# The source->target pairings actually used in the study (external-score integrity).
PAIRINGS = [
    ("paddy_doctor", "philippines", "Canopy"),
    ("kaggle_rice", "riceleafs", "Single-leaf"),
    ("paddy_doctor", "riceleafs", "Canopy->leaf"),
    ("kaggle_rice", "philippines", "Leaf->canopy"),
]


def phash64(path: Path) -> int:
    """Perceptual hash (DCT-based, classic pHash): resize to 32x32 grayscale, take
    the top-left 8x8 low-frequency DCT block (excluding the DC term), threshold at
    the median. Far more robust to the canopy false-positives that an 8x8 average
    hash produces on visually homogeneous field images."""
    img = Image.open(path).convert("L").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32)
    d = dctn(arr, norm="ortho")
    block = d[:8, :8].flatten()
    med = np.median(block[1:])            # exclude DC (index 0) from the median
    bits = 0
    for i, v in enumerate(block):
        bits = (bits << 1) | (1 if (i != 0 and v > med) else 0)
    return bits


def load_hashes(name: str):
    root = config.DATA_RAW / name
    out = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix in EXTS:
                try:
                    out.append((phash64(f), f.name, cls_dir.name))
                except Exception:
                    pass
    return out


def compare(a, b, record_pairs=False):
    """Count images in b that have an exact / near duplicate in a.

    Returns exact, near, examples, hist where hist[d] = #target imgs whose nearest
    source image is at Hamming distance d (for a threshold sweep).
    """
    exact = near = 0
    examples = []
    hist = {}
    for hb, fnb, clsb in b:
        best = 65
        best_a = None
        for ha, fna, clsa in a:
            d = (ha ^ hb).bit_count()
            if d < best:
                best, best_a = d, (fna, clsa)
                if best == 0:
                    break
        hist[best] = hist.get(best, 0) + 1
        if best == 0:
            exact += 1
        elif best <= NEAR_BITS:
            near += 1
        if (best <= NEAR_BITS) and (record_pairs or len(examples) < 5):
            examples.append({"dist": best, "b_file": fnb, "b_class": clsb,
                             "a_file": best_a[0] if best_a else None,
                             "a_class": best_a[1] if best_a else None})
    hist = {str(k): hist[k] for k in sorted(hist)}
    return exact, near, examples, hist


def main():
    print("Loading perceptual hashes (pHash) ...")
    H = {}
    for d in DATASETS:
        H[d] = load_hashes(d)
        print(f"  {d:14s}: {len(H[d])} images")

    result = {"datasets": {d: len(H[d]) for d in DATASETS},
              "near_bits_threshold": NEAR_BITS, "pairings": {}, "all_pairs": {}}

    print("\n=== Source -> Target pairings used in the study ===")
    for src, tgt, regime in PAIRINGS:
        ex, nr, exs, hist = compare(H[src], H[tgt], record_pairs=True)
        pct = 100.0 * (ex + nr) / max(1, len(H[tgt]))
        result["pairings"][f"{src}->{tgt}"] = {
            "regime": regime, "target_n": len(H[tgt]),
            "exact_dups": ex, "near_dups": nr,
            "overlap_pct_of_target": round(pct, 3),
            "dist_histogram": hist, "examples": exs}
        flag = "OK (clean)" if (ex + nr) == 0 else f"<-- {ex} exact, {nr} near ({pct:.2f}% of target)"
        print(f"  {regime:13s} {src} -> {tgt:12s}: {flag}")
        print(f"      nearest-dist histogram (d:count): {hist}")

    print("\n=== Kaggle sets vs Paddy Doctor (provenance / leakage) ===")
    for k in ("kaggle_rice", "riceleafs"):
        ex, nr, exs, hist = compare(H["paddy_doctor"], H[k], record_pairs=True)
        pct = 100.0 * (ex + nr) / max(1, len(H[k]))
        result["all_pairs"][f"paddy_doctor<->{k}"] = {
            "exact_dups": ex, "near_dups": nr, "pct_of_kaggle": round(pct, 3),
            "dist_histogram": hist, "examples": exs}
        flag = "OK (no shared images)" if (ex + nr) == 0 else f"<-- {ex} exact, {nr} near ({pct:.2f}%)"
        print(f"  paddy_doctor vs {k:12s}: {flag}")
        print(f"      nearest-dist histogram (d:count): {hist}")

    (config.OUTPUTS / "cross_leakage.json").write_text(json.dumps(result, indent=2))
    print("\nSaved outputs/cross_leakage.json")


if __name__ == "__main__":
    main()
