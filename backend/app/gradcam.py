"""Grad-CAM style heatmap generation.

For the current mock model (``model.torch_model is None``) this produces an
*illustrative* saliency heatmap built from a couple of 2D Gaussian blobs,
colored with a simple numpy jet colormap and blended over the resized input
image. When a real torch model is available, a clearly-marked branch shows
where genuine Grad-CAM (via ``pytorch_grad_cam``, imported lazily) would be
computed instead.

Only numpy and Pillow are required for the current implementation.
"""

from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from app.config import settings
from app.inference import preprocess

# Alpha for blending the heatmap over the source image.
_BLEND_ALPHA = 0.45


def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a normalized [0, 1] grayscale array to a jet-like RGB image.

    This is a lightweight numpy reimplementation of the classic "jet"
    colormap, avoiding a matplotlib dependency.

    Args:
        gray: A 2D float array with values in ``[0, 1]``.

    Returns:
        A ``(H, W, 3)`` uint8 RGB array.
    """
    g = np.clip(gray, 0.0, 1.0)

    def _channel(x: np.ndarray, center: float) -> np.ndarray:
        # Triangular ramp peaking at ``center`` with a width of 0.5.
        return np.clip(1.5 - np.abs(4.0 * x - center), 0.0, 1.0)

    red = _channel(g, 3.0)
    green = _channel(g, 2.0)
    blue = _channel(g, 1.0)

    rgb = np.stack([red, green, blue], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _gaussian_blob(
    size: int, cx: float, cy: float, sigma: float
) -> np.ndarray:
    """Create a 2D Gaussian blob on a ``size x size`` grid.

    Args:
        size: The width/height of the square grid.
        cx: Blob center x-coordinate (pixels).
        cy: Blob center y-coordinate (pixels).
        sigma: Standard deviation of the Gaussian (pixels).

    Returns:
        A ``(size, size)`` float array in ``[0, 1]``.
    """
    ys, xs = np.mgrid[0:size, 0:size]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist_sq / (2.0 * sigma * sigma))


def _illustrative_saliency(img: "Image.Image") -> np.ndarray:
    """Build a deterministic illustrative saliency map from Gaussian blobs.

    The blob positions are derived from a hash of the image so the heatmap
    is stable for a given input (matching the deterministic mock model).

    Args:
        img: The (already resizable) input PIL image.

    Returns:
        A normalized ``(IMAGE_SIZE, IMAGE_SIZE)`` float array in ``[0, 1]``.
    """
    size = settings.IMAGE_SIZE
    digest = hashlib.sha256(preprocess(img).tobytes()).digest()

    # Derive two blob centers and radii from hash bytes.
    b = list(digest)
    cx1 = (b[0] / 255.0) * size
    cy1 = (b[1] / 255.0) * size
    s1 = size * (0.12 + (b[2] / 255.0) * 0.10)

    cx2 = (b[3] / 255.0) * size
    cy2 = (b[4] / 255.0) * size
    s2 = size * (0.10 + (b[5] / 255.0) * 0.10)

    blob = _gaussian_blob(size, cx1, cy1, s1)
    blob += 0.7 * _gaussian_blob(size, cx2, cy2, s2)

    peak = float(blob.max())
    if peak > 0.0:
        blob = blob / peak
    return blob


def generate_gradcam(img: "Image.Image", model: object) -> bytes:
    """Generate a Grad-CAM style heatmap PNG for the given image.

    Args:
        img: The input PIL image.
        model: The active model instance. If ``model.torch_model`` is not
            ``None`` a real Grad-CAM would be computed (see the marked
            branch below); otherwise an illustrative heatmap is produced.

    Returns:
        PNG-encoded bytes of the blended heatmap overlay.
    """
    base = preprocess(img)
    base_arr = np.asarray(base).astype(np.float32)

    torch_model = getattr(model, "torch_model", None)

    if torch_model is not None:
        # -------------------------------------------------------------------
        # TODO: REAL GRAD-CAM BRANCH (only when a torch model is loaded).
        # Lazily import here so torch/pytorch_grad_cam are never required for
        # the mock path::
        #
        #     import torch
        #     from pytorch_grad_cam import GradCAM
        #     from pytorch_grad_cam.utils.image import show_cam_on_image
        #
        #     target_layers = [torch_model.features[-1]]
        #     cam = GradCAM(model=torch_model, target_layers=target_layers)
        #     input_tensor = _to_normalized_tensor(base)  # match training
        #     grayscale_cam = cam(input_tensor=input_tensor)[0]
        #     overlay = show_cam_on_image(
        #         base_arr / 255.0, grayscale_cam, use_rgb=True
        #     )
        #     return _encode_png(overlay)
        #
        # For now, fall through to the illustrative heatmap even if a torch
        # model is somehow present, since real Grad-CAM is not wired in yet.
        # -------------------------------------------------------------------
        pass

    # Illustrative (mock) heatmap path.
    saliency = _illustrative_saliency(img)
    heat_rgb = _jet_colormap(saliency).astype(np.float32)

    blended = (1.0 - _BLEND_ALPHA) * base_arr + _BLEND_ALPHA * heat_rgb
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    out = Image.fromarray(blended, mode="RGB")
    buffer = BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()
