"""Quantitative Grad-CAM grounding (reviewer 3a), reproducible -- NO manual boxes.

On the single-leaf target (plain white/sky background) we segment, per image:
  * LEAF (plant) vs background   -- trivial on plain backgrounds
  * LESION within the leaf        -- brown/necrotic (non-green) leaf pixels
Then, for the EfficientNet single-leaf baseline, we compute Grad-CAM for the
NO-ADAPT model and an AdaBN-adapted copy, and measure (paired across images):
  * leaf_mass   = fraction of CAM attention mass falling on the plant (vs background)
  * lesion_IoU  = IoU(top-k% CAM mask, lesion mask)
  * lesion_mass = fraction of CAM mass on lesions
  * pointing    = is the CAM peak on the plant / on a lesion
This turns the qualitative "attends to background, not lesions" claim into numbers.

  python gradcam_grounding.py --n 30 --overlays 6
  python gradcam_grounding.py --tune   # just render overlays to check segmentation
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import config

SIZE = config.IMG_SIZE
CKPT = "baseline_efficientnet_b0_kaggle_rice-riceleafs_s42_efficientnet_b0.pt"
LESION_CLASSES = ["Rice_Blast", "Brown_Spot"]


# --------------------------- segmentation (PIL HSV) --------------------------- #
def segment(pil_rgb):
    """Return (leaf_mask, lesion_mask) as bool arrays at SIZE x SIZE."""
    im = pil_rgb.resize((SIZE, SIZE))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]   # each 0..255 (H maps 0..360)

    bg_white = (V > 195) & (S < 45)                   # bright, desaturated -> paper/white
    bg_blue = (H > 140) & (H < 185) & (S > 35)        # blue sky
    background = bg_white | bg_blue
    leaf = ~background
    # clean: close small gaps, keep the largest connected component, fill holes
    leaf = ndimage.binary_closing(leaf, iterations=2)
    lab, n = ndimage.label(leaf)
    if n >= 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        leaf = lab == (1 + int(np.argmax(sizes)))
    leaf = ndimage.binary_fill_holes(leaf)

    # lesion = leaf pixels that are NOT healthy-green: brown/orange/yellow hue or dark
    green = (H >= 45) & (H <= 100)
    brownish = (H < 42) | (H > 170)
    dark = V < 75
    lesion = leaf & ((brownish & (S > 30)) | dark) & (~(green & (S > 60)))
    lesion = ndimage.binary_opening(lesion, iterations=1)   # drop single-pixel noise
    return leaf, lesion


# ------------------------------- overlays ------------------------------------- #
def save_overlay(pil_rgb, leaf, lesion, cam0, cam1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = np.asarray(pil_rgb.resize((SIZE, SIZE)))
    fig, ax = plt.subplots(1, 5, figsize=(13, 2.9))
    ax[0].imshow(base); ax[0].set_title("image", fontsize=9)
    ax[1].imshow(base); ax[1].imshow(leaf, alpha=0.4, cmap="Greens"); ax[1].set_title("leaf mask", fontsize=9)
    ax[2].imshow(base); ax[2].imshow(lesion, alpha=0.55, cmap="Reds"); ax[2].set_title("lesion mask", fontsize=9)
    ax[3].imshow(base); ax[3].imshow(cam0, alpha=0.5, cmap="jet"); ax[3].set_title("CAM no-adapt", fontsize=9)
    ax[4].imshow(base); ax[4].imshow(cam1, alpha=0.5, cmap="jet"); ax[4].set_title("CAM AdaBN", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ------------------------------- metrics -------------------------------------- #
def cam_metrics(cam, leaf, lesion, pct):
    total = cam.sum() + 1e-8
    leaf_mass = float((cam * leaf).sum() / total)
    lesion_mass = float((cam * lesion).sum() / total)
    mask = cam >= np.percentile(cam, pct)
    inter = np.logical_and(mask, lesion).sum()
    union = np.logical_or(mask, lesion).sum()
    lesion_iou = float(inter) / float(union) if union else 0.0
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"leaf_mass": leaf_mass, "lesion_mass": lesion_mass, "lesion_iou": lesion_iou,
            "point_leaf": int(leaf[peak]), "point_lesion": int(lesion[peak])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="images per lesion class")
    ap.add_argument("--pct", type=float, default=80.0)
    ap.add_argument("--overlays", type=int, default=6)
    ap.add_argument("--tune", action="store_true", help="only render overlays (no model)")
    args = ap.parse_args()

    root = config.DATA_RAW / "riceleafs"
    files = []
    for cls in LESION_CLASSES:
        fs = sorted((root / cls).glob("*.jpg"))
        step = max(1, len(fs) // args.n)
        files += [(cls, f) for f in fs[::step][:args.n]]
    print(f"{len(files)} images ({args.n}/class)")

    outdir = config.OUTPUTS / "gradcam_grounding"
    outdir.mkdir(exist_ok=True)

    if args.tune:   # just segmentation overlays (no torch) to visually verify masks
        for i, (cls, f) in enumerate(files[:args.overlays]):
            im = Image.open(f).convert("RGB")
            leaf, lesion = segment(im)
            z = np.zeros((SIZE, SIZE))
            save_overlay(im, leaf, lesion, z, z, outdir / f"tune_{i}_{cls}.png")
        print(f"wrote {args.overlays} tuning overlays to {outdir}")
        return

    import torch
    from models import build_model
    from dg import adabn_adapt
    from data import build_transforms
    from gradcam import gradcam, target_layer
    import train as T
    os.environ["RICE_SOURCE_DOMAIN"] = "kaggle_rice"; os.environ["RICE_TARGET_DOMAIN"] = "riceleafs"
    import importlib; importlib.reload(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = build_transforms(train=False)

    def fresh():
        n = build_model("efficientnet_b0", config.NUM_CLASSES, pretrained=False)
        n.load_state_dict(torch.load(config.OUTPUTS / CKPT, map_location="cpu"))
        return n.to(device)

    net0 = fresh(); net0.eval()
    net1 = fresh()
    _, _, _, ext = T._loaders(64)
    adabn_adapt(net1, ext, device); net1.eval()

    rows0, rows1 = [], []
    for i, (cls, f) in enumerate(files):
        im = Image.open(f).convert("RGB")
        leaf, lesion = segment(im)
        if lesion.sum() < 10:      # skip images where no lesion is detectable
            continue
        t = tf(im).unsqueeze(0).to(device)
        _, cam0 = gradcam(net0, t, target_layer(net0))
        _, cam1 = gradcam(net1, t, target_layer(net1))
        rows0.append(cam_metrics(cam0, leaf, lesion, args.pct))
        rows1.append(cam_metrics(cam1, leaf, lesion, args.pct))
        if i < args.overlays:
            save_overlay(im, leaf, lesion, cam0, cam1, outdir / f"ov_{i}_{cls}.png")

    from scipy import stats
    def agg(key):
        a = np.array([r[key] for r in rows0]); b = np.array([r[key] for r in rows1])
        t, p = stats.ttest_rel(b, a)
        return {"no_adapt": round(float(a.mean()), 4), "adabn": round(float(b.mean()), 4),
                "delta": round(float(b.mean() - a.mean()), 4), "p": round(float(p), 4)}

    res = {"n_images": len(rows0), "cam_percentile": args.pct,
           "leaf_mass": agg("leaf_mass"), "lesion_mass": agg("lesion_mass"),
           "lesion_iou": agg("lesion_iou"),
           "pointing_leaf": agg("point_leaf"), "pointing_lesion": agg("point_lesion")}
    (config.OUTPUTS / "gradcam_grounding.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nSaved gradcam_grounding.json + overlays in {outdir}")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
