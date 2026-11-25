"""Rice Blast Detector backend package.

A FastAPI backend serving rice leaf disease classification, Grad-CAM
visualizations, and feedback collection. The current model is a
deterministic mock; a real MobileNetV3 + MixStyle model can be dropped in
later without changing the public API.
"""

from app.config import settings

__all__ = ["settings"]
__version__ = settings.VERSION
