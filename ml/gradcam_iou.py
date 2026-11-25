"""Grad-CAM lesion-grounding: turns the qualitative "attends to background, not
lesions" claim into a QUANTITATIVE metric (reviewer concern 3a).

For a small hand-annotated subset of target images (lesion bounding boxes), we
threshold the Grad-CAM heatmap into a mask and measure, BEFORE vs AFTER AdaBN:
  * IoU(CAM-mask, lesion-box-union)   -- localisation overlap
  * pointing-game hit                 -- is the CAM peak inside a lesion box?
A rise in both after AdaBN is direct evidence that fixing the BatchNorm statistics
re-focuses attention onto lesions, not just that Macro-F1 improves.

TWO MODES
  # 1) CPU, run now: make an empty annotation template to hand-label (~50 imgs)
  python gradcam_iou.py --make-template --domain riceleafs --per-class 25

  # 2) GPU, run after labelling: compute IoU before/after AdaBN
  python gradcam_iou.py --ckpt outputs/baseline_efficientnet_b0_kaggle_rice-riceleafs_s42_efficientnet_b0.pt \
                        --model efficientnet_b0 --source kaggle_rice --target riceleafs \
                        --annot annotations/iou_riceleafs.json

Box format in the JSON: normalized [x, y, w, h] in [0,1] (resolution-independent),
one list per image, keyed by "<Class_dir>/<filename>". Healthy images need no boxes
(skipped for IoU; used only as negative controls if boxes are empty).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import config

LESION_CLASSES = ["Rice_Blast", "Brown_Spot"]  # Healthy has no lesion to localise


# --------------------------------------------------------------------------- #
#  Mode 1 — annotation template (CPU only, safe to run now)
# --------------------------------------------------------------------------- #
def make_template(domain: str, per_class: int, out: Path):
    root = config.DATA_RAW / domain
    boxes = {}
    for cls in LESION_CLASSES:
        d = root / cls
        if not d.exists():
            print(f"  (skip {cls}: not in {domain})")
            continue
        imgs = sorted(p.name for p in d.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        # even stride sample across the class for diversity
        step = max(1, len(imgs) // per_class)
        picked = imgs[::step][:per_class]
        for fn in picked:
            boxes[f"{cls}/{fn}"] = []   # <-- annotator fills [[x,y,w,h], ...] normalized
    tmpl = {
        "_instructions": ("Fill each list with normalized lesion boxes [x,y,w,h] in "
                          "[0,1] (x,y = top-left). Multiple boxes per image allowed. "
                          "Leave [] only if truly no visible lesion."),
        "domain": domain,
        "image_root": str((config.DATA_RAW / domain)),
        "boxes_norm": boxes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tmpl, indent=2))
    print(f"Wrote template with {len(boxes)} images to {out}")
    print("Annotate with any tool (e.g. LabelImg / Roboflow), export normalized xywh.")


# --------------------------------------------------------------------------- #
#  Mode 2 — IoU before/after AdaBN (needs torch + GPU; run after labelling)
# --------------------------------------------------------------------------- #
def boxes_to_mask(boxes_norm, size):
    m = np.zeros((size, size), dtype=bool)
    for x, y, w, h in boxes_norm:
        x0, y0 = int(x * size), int(y * size)
        x1, y1 = int((x + w) * size), int((y + h) * size)
        m[max(0, y0):min(size, y1), max(0, x0):min(size, x1)] = True
    return m


def cam_mask(cam, pct=80):
    thr = np.percentile(cam, pct)
    return cam >= thr


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def compute(args):
    import torch
    from PIL import Image
    from data import build_transforms
    from models import build_model
    from dg import adabn_adapt
    from gradcam import gradcam, target_layer
    import train as T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    annot = json.load(open(args.annot))
    size = config.IMG_SIZE
    tf = build_transforms(train=False)

    os.environ["RICE_SOURCE_DOMAIN"] = args.source
    os.environ["RICE_TARGET_DOMAIN"] = args.target
    import importlib; importlib.reload(config)

    def fresh():
        n = build_model(args.model, config.NUM_CLASSES, pretrained=False)
        n.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        return n.to(device)

    # no-adapt and AdaBN-adapted copies
    net0 = fresh(); net0.eval()
    net1 = fresh()
    _, _, _, ext = T._loaders(32)
    adabn_adapt(net1, ext, device); net1.eval()

    root = config.DATA_RAW / args.target
    rows = {"no_adapt": [], "adabn": []}
    hits = {"no_adapt": 0, "adabn": 0}
    n = 0
    for key, bx in annot["boxes_norm"].items():
        if not bx:
            continue
        img = Image.open(root / key).convert("RGB")
        gt = boxes_to_mask(bx, size)
        t = tf(img).unsqueeze(0).to(device)
        for tag, net in (("no_adapt", net0), ("adabn", net1)):
            _, cam = gradcam(net, t, target_layer(net))
            rows[tag].append(iou(cam_mask(cam, args.pct), gt))
            peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
            hits[tag] += int(gt[peak])
        n += 1

    res = {"n_annotated": n, "cam_percentile": args.pct,
           "mean_iou": {k: round(float(np.mean(v)), 4) for k, v in rows.items()},
           "pointing_game_acc": {k: round(hits[k] / n, 4) for k in hits}}
    out = config.OUTPUTS / "gradcam_iou.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--domain", default="riceleafs")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--annot", default="annotations/iou_riceleafs.json")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--source", default="kaggle_rice")
    ap.add_argument("--target", default="riceleafs")
    ap.add_argument("--pct", type=float, default=80.0,
                    help="CAM percentile threshold for the attention mask")
    args = ap.parse_args()

    if args.make_template:
        make_template(args.domain, args.per_class, Path(args.annot))
    else:
        if not args.ckpt:
            raise SystemExit("compute mode needs --ckpt (a trained baseline checkpoint)")
        compute(args)


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0")
    main()
