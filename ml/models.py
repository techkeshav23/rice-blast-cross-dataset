"""Backbone factory. Baseline = MobileNetV3-Small (EfficientNet-B0 optional).

The MixStyle domain-generalization variant (Phase 3) will build on the same
backbones by inserting MixStyle after early feature blocks.
"""
import torch.nn as nn
from torchvision import models


def build_model(name: str = "mobilenet_v3_small",
                num_classes: int = 3,
                pretrained: bool = True) -> nn.Module:
    """Return a backbone with the classifier head resized to `num_classes`.

    Falls back to random init if pretrained weights can't be downloaded
    (e.g. offline), so training still runs.
    """
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        try:
            net = models.mobilenet_v3_small(weights=weights)
        except Exception:
            net = models.mobilenet_v3_small(weights=None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        return net

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            net = models.efficientnet_b0(weights=weights)
        except Exception:
            net = models.efficientnet_b0(weights=None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        return net

    raise ValueError(f"Unknown model: {name}")
