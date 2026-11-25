"""Model inference layer.

Defines a small model abstraction with two concrete implementations:

* :class:`MockModel` -- a deterministic, dependency-light mock used for
  demos and development. The same image always yields the same result.
* :class:`TorchModel` -- a stub for the eventual MobileNetV3 + MixStyle
  model. It lazily imports torch inside its methods so importing this
  module never requires torch to be installed.

The :func:`get_model` factory returns a singleton, preferring the real
Torch model only when both the weights file and torch are available, and
falling back to the mock otherwise. It never raises if torch is absent.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Optional

from PIL import Image

from app.config import settings


def preprocess(img: "Image.Image") -> "Image.Image":
    """Preprocess a PIL image for inference.

    Converts the image to RGB and resizes it to a square of
    ``settings.IMAGE_SIZE`` pixels. This mirrors the spatial preprocessing
    that a trained model would expect (normalization is applied later, at
    tensor-conversion time, by the concrete model).

    Args:
        img: The input PIL image.

    Returns:
        A resized RGB :class:`PIL.Image.Image`.
    """
    size = settings.IMAGE_SIZE
    rgb = img.convert("RGB")
    return rgb.resize((size, size), Image.Resampling.BILINEAR)


class BaseModel:
    """Abstract base class for classification models.

    Concrete subclasses must implement :meth:`predict` and expose the
    attributes ``version`` (str), ``loaded`` (bool), and ``torch_model``
    (the underlying torch model or ``None``).
    """

    version: str = settings.VERSION
    loaded: bool = False
    torch_model: Optional[object] = None
    # Human-readable description reported by GET /model-info.
    backbone: str = "none"
    method: str = "none"
    trained_on: str = "none"

    @property
    def classes(self) -> List[str]:
        """The ordered list of class labels."""
        return list(settings.CLASSES)

    def predict(self, img: "Image.Image") -> Dict[str, object]:
        """Run a prediction on the given image.

        Args:
            img: The input PIL image.

        Returns:
            A dict with keys ``label``, ``confidence``, ``scores``, and
            ``low_confidence``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


def _softmax(values: List[float]) -> List[float]:
    """Compute a numerically stable softmax over a list of floats."""
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    if total == 0.0:
        n = len(values)
        return [1.0 / n] * n
    return [e / total for e in exps]


class MockModel(BaseModel):
    """Deterministic mock classifier.

    The prediction is derived from a SHA-256 hash of the image's pixel
    bytes, so the same image always produces the same distribution. This
    makes demos reproducible without any trained weights.
    """

    backbone = "none (mock model, no trained weights loaded)"
    method = "deterministic mock: scores derived from a hash of the image"
    trained_on = "none"

    def __init__(self) -> None:
        self.version = "mock"
        self.loaded = True
        self.torch_model = None

    def _seed_from_image(self, img: "Image.Image") -> int:
        """Derive a deterministic integer seed from image pixels."""
        prepared = preprocess(img)
        # Use raw pixel bytes so visually identical images seed identically.
        digest = hashlib.sha256(prepared.tobytes()).hexdigest()
        return int(digest[:16], 16)

    def predict(self, img: "Image.Image") -> Dict[str, object]:
        """Produce a deterministic softmax distribution over the classes.

        Args:
            img: The input PIL image.

        Returns:
            A dict with keys ``label``, ``confidence``, ``scores``, and
            ``low_confidence``.
        """
        classes = self.classes
        seed = self._seed_from_image(img)
        rng = random.Random(seed)

        # Draw raw logits and bias one class upward so the argmax is
        # reasonably confident but still varies per image.
        logits = [rng.uniform(-1.0, 1.0) for _ in classes]
        boosted = rng.randrange(len(classes))
        logits[boosted] += rng.uniform(1.0, 3.0)

        probs = _softmax(logits)
        scores = {cls: round(float(p), 6) for cls, p in zip(classes, probs)}

        top_index = max(range(len(classes)), key=lambda i: probs[i])
        label = classes[top_index]
        confidence = round(float(probs[top_index]), 6)
        low_confidence = confidence < settings.LOW_CONF_THRESHOLD

        return {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "low_confidence": low_confidence,
        }


class TorchModel(BaseModel):
    """Stub for the real MobileNetV3 + MixStyle model.

    This class lazily imports torch/torchvision *inside* its methods so that
    merely importing this module (or constructing the object) never requires
    torch. It is not yet functional; :meth:`predict` raises
    ``NotImplementedError`` with a TODO describing the intended
    implementation.
    """

    def __init__(self) -> None:
        self.version = "torch-stub"
        self.loaded = False
        self.torch_model = None

    def _load(self) -> None:
        """Lazily construct and load the torch model.

        TODO: Implement the real loading path::

            import torch
            from torchvision.models import mobilenet_v3_large

            model = mobilenet_v3_large(weights=None)
            # Inject MixStyle layers into the feature extractor here, then
            # swap the classifier head for ``len(settings.CLASSES)`` outputs.
            state = torch.load(settings.MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.torch_model = model
            self.loaded = True
        """
        raise NotImplementedError(
            "TorchModel loading is not implemented yet. See the docstring "
            "for the intended MobileNetV3 + MixStyle loading procedure."
        )

    def predict(self, img: "Image.Image") -> Dict[str, object]:
        """Run inference with the torch model.

        TODO: Implement the real inference path::

            import torch
            from torchvision import transforms

            tfm = transforms.Compose([
                transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            tensor = tfm(preprocess(img)).unsqueeze(0)
            with torch.no_grad():
                logits = self.torch_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            # Map probs back onto settings.CLASSES and build the response.

        Args:
            img: The input PIL image.

        Raises:
            NotImplementedError: Always, until the real model is wired in.
        """
        raise NotImplementedError(
            "TorchModel.predict is not implemented yet. Preprocess must match "
            "training exactly (Resize -> ToTensor -> ImageNet Normalize)."
        )


class OnnxModel(BaseModel):
    """Real classifier backed by the exported ONNX model (MobileNetV3 + MixStyle).

    Uses onnxruntime + numpy only (no torch needed on the server). Preprocessing
    mirrors training exactly: Resize -> RGB float [0,1] -> ImageNet normalize -> CHW.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    backbone = "MobileNetV3-Small"
    method = "MixStyle (early single-seed checkpoint); ONNX Runtime CPU inference"
    trained_on = "Paddy Doctor (Rice Blast, Brown Spot, Healthy)"

    def __init__(self) -> None:
        import numpy as np  # noqa: F401 (ensure available early)
        import onnxruntime as ort

        self._np = __import__("numpy")
        self._session = ort.InferenceSession(
            str(settings.MODELS_DIR / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input = self._session.get_inputs()[0].name
        self.version = "onnx-mobilenetv3-small-v1"
        self.loaded = True
        self.torch_model = None

    def _to_input(self, img: "Image.Image"):
        np = self._np
        arr = np.asarray(preprocess(img), dtype=np.float32) / 255.0
        arr = (arr - np.array(self._MEAN, dtype=np.float32)) / np.array(
            self._STD, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1,C,H,W
        return arr.astype(np.float32)

    def predict(self, img: "Image.Image") -> Dict[str, object]:
        np = self._np
        logits = self._session.run(None, {self._input: self._to_input(img)})[0][0]
        probs = _softmax([float(v) for v in logits])
        classes = self.classes
        scores = {c: round(float(p), 6) for c, p in zip(classes, probs)}
        top = int(np.argmax(probs))
        confidence = round(float(probs[top]), 6)
        return {
            "label": classes[top],
            "confidence": confidence,
            "scores": scores,
            "low_confidence": confidence < settings.LOW_CONF_THRESHOLD,
        }


# Module-level singleton cache.
_model_instance: Optional[BaseModel] = None


def _onnx_ready() -> bool:
    """True if an ONNX model file and onnxruntime are both available."""
    try:
        import importlib.util

        has_rt = importlib.util.find_spec("onnxruntime") is not None
        return has_rt and (settings.MODELS_DIR / "model.onnx").is_file()
    except Exception:
        return False


def _torch_available() -> bool:
    """Return True if torch can be imported, without hard-depending on it."""
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def get_model() -> BaseModel:
    """Return the singleton model instance.

    Prefers :class:`TorchModel` only when the weights file at
    ``settings.MODEL_PATH`` exists AND torch is importable; otherwise
    returns a :class:`MockModel`. This function never raises due to a
    missing torch installation.

    Returns:
        The shared :class:`BaseModel` instance.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    # Prefer the real exported ONNX model when present (torch-free inference).
    if _onnx_ready():
        try:
            _model_instance = OnnxModel()
            return _model_instance
        except Exception:
            pass

    use_torch = False
    try:
        use_torch = settings.MODEL_PATH.is_file() and _torch_available()
    except OSError:
        use_torch = False

    if use_torch:
        try:
            model: BaseModel = TorchModel()
            model._load()  # type: ignore[attr-defined]
            _model_instance = model
            return _model_instance
        except Exception:
            # Any failure loading the real model falls back to the mock so the
            # service always starts.
            pass

    _model_instance = MockModel()
    return _model_instance
