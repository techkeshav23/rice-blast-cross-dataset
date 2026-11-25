"""Phase 5 — REAL Grad-CAM on a trained checkpoint.

Hooks the last conv block, computes the class-activation map for the predicted
class, and saves a heatmap overlay per sample. This replaces the app's
illustrative overlay once the model is real.

Usage:
  python gradcam.py --tag mixstyle --dg mixstyle
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data import build_transforms
from dg import to_mixstyle
from models import build_model


def load_model(tag: str, model_name: str, dg: str):
    net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
    if dg == "mixstyle":
        net = to_mixstyle(net)
    state = torch.load(config.OUTPUTS / f"{tag}_{model_name}.pt", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


def target_layer(net):
    """Last conv block of the features stack (works for plain + MixStyle nets)."""
    return net.features[-1]


def gradcam(net, tensor, layer):
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["v"] = o.detach()

    def bwd(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    out = net(tensor)
    cls = int(out.argmax(1).item())
    net.zero_grad()
    out[0, cls].backward()

    h1.remove()
    h2.remove()

    a, g = acts["v"], grads["v"]                 # [1,C,h,w]
    weights = g.mean(dim=[2, 3], keepdim=True)    # [1,C,1,1]
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # [1,1,h,w]
    cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE),
                        mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cls, cam


def overlay(pil_img, cam, out_path):
    import matplotlib.cm as cm
    base = np.asarray(pil_img.resize((config.IMG_SIZE, config.IMG_SIZE))).astype(float)
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(float)
    blend = np.clip(0.55 * base + 0.45 * heat, 0, 255).astype(np.uint8)
    Image.fromarray(blend).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mixstyle")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--dg", choices=["none", "mixstyle"], default="mixstyle")
    args = ap.parse_args()

    net = load_model(args.tag, args.model, args.dg)
    layer = target_layer(net)
    tf = build_transforms(train=False)

    out_dir = config.OUTPUTS / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    # one sample per class from the unseen target domain
    for cls in config.CLASSES:
        folder = config.DATA_RAW / config.TARGET_DOMAIN / cls.replace(" ", "_")
        sample = sorted(folder.glob("*.jpg"))[0]
        img = Image.open(sample).convert("RGB")
        tensor = tf(img).unsqueeze(0)
        pred, cam = gradcam(net, tensor, layer)
        out = out_dir / f"gradcam_{cls.replace(' ', '_')}.png"
        overlay(img, cam, out)
        print(f"  {cls:12s} true | predicted={config.CLASSES[pred]:12s} -> {out.name}")

    print(f"Saved Grad-CAM overlays to {out_dir.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
