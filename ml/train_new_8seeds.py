"""Raise the 4 NEW-pairing cells from 4 -> 8 seeds (baseline + MixStyle), epochs 20,
batch 64 (matches the existing new-cell recipe). Resumes via run_seeds.py (skips
seeds already trained). Guards app/backend metrics.json from run_seeds' push
side-effect (which fires for MNet + paddy_doctor source).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config

ROOT = config.ROOT
CELLS = [
    ("paddy_doctor", "riceleafs",   "mobilenet_v3_small"),
    ("paddy_doctor", "riceleafs",   "efficientnet_b0"),
    ("kaggle_rice",  "philippines", "mobilenet_v3_small"),
    ("kaggle_rice",  "philippines", "efficientnet_b0"),
]
SEEDS = ["42", "1", "2", "3", "4", "5", "6", "7"]

# --- backup metrics run_seeds may clobber ---
guards = [
    (config.OUTPUTS / "metrics.json", config.OUTPUTS / "metrics.json.bak_8seed"),
    (ROOT.parent / "backend" / "models" / "metrics.json",
     ROOT.parent / "backend" / "models" / "metrics.json.bak_8seed"),
]
for src, bak in guards:
    if src.exists():
        shutil.copy2(src, bak)

# num_workers=0: on Windows each DataLoader worker re-imports CUDA torch in a fresh
# process, which can fail DLL init (WinError 1114) under memory pressure. Single-
# process loading is safe; images are RAM-cached (RICE_CACHE=1) so speed is fine.
env = {**os.environ, "RICE_NUM_WORKERS": "0", "RICE_CACHE": "1", "PYTHONUNBUFFERED": "1"}
try:
    for s_src, s_tgt, model in CELLS:
        env["RICE_SOURCE_DOMAIN"] = s_src
        env["RICE_TARGET_DOMAIN"] = s_tgt
        print(f"\n########## {model} | {s_src}->{s_tgt} | seeds {' '.join(SEEDS)} ##########",
              flush=True)
        subprocess.run(
            [sys.executable, "run_seeds.py", "--methods", "none", "mixstyle",
             "--seeds", *SEEDS, "--epochs", "20", "--batch", "64", "--model", model],
            check=True, env=env, cwd=str(ROOT),
        )
finally:
    # restore the app/backend metrics (keep the official canopy MNet numbers)
    for src, bak in guards:
        if bak.exists():
            shutil.copy2(bak, src)
    print("\nAll 4 new cells at 8 seeds (where training succeeded). "
          "Restored app/backend metrics.json.", flush=True)
