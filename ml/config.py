"""Central configuration for the Rice Blast ML pipeline."""
import os
from pathlib import Path

# Reproducibility
SEED = 42

# Task
CLASSES = ["Rice Blast", "Brown Spot", "Healthy"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

# Split ratios (applied at FILE level, BEFORE augmentation)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"            # datasets live here (per domain / per class)
SPLITS_DIR = ROOT / "data" / "splits"       # split manifests
OUTPUTS = ROOT / "outputs"                   # metrics, checkpoints, plots

# Domain names. Default to the synthetic pilot; override with environment
# variables for the REAL run, e.g. RICE_SOURCE_DOMAIN=paddy_doctor
# RICE_TARGET_DOMAIN=philippines. The rest of the pipeline is unchanged.
SOURCE_DOMAIN = os.environ.get("RICE_SOURCE_DOMAIN", "synthetic_source")
TARGET_DOMAIN = os.environ.get("RICE_TARGET_DOMAIN", "synthetic_target")

for _d in (DATA_RAW, SPLITS_DIR, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)
