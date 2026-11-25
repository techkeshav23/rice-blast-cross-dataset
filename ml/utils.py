"""Small shared helpers."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (and Torch if present) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional in Phase 1
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
