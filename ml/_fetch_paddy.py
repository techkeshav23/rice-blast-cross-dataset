"""Download + arrange ONLY the Paddy Doctor competition into the 'paddy_doctor'
source domain (reuses helpers from fetch_real_data). Run after accepting the
competition rules."""
import zipfile
from pathlib import Path

import config
from fetch_real_data import _kaggle_api, arrange

HERE = Path(__file__).resolve().parent
api = _kaggle_api(HERE)

MIRROR = "imbikramsaha/paddy-doctor"  # public dataset mirror (no competition rules)
d = config.ROOT / "data" / "downloads" / "paddy"
d.mkdir(parents=True, exist_ok=True)
print(f"Downloading Paddy Doctor mirror ({MIRROR}) ...")
api.dataset_download_files(MIRROR, path=str(d), unzip=True)
arrange(d, "paddy_doctor", 500)
print("Paddy Doctor ready at data/raw/paddy_doctor")
