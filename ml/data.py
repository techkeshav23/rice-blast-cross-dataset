"""Torch Dataset built from the split manifests. Augmentation is applied ONLY
to the training split (per the leakage-control protocol).

Images are decoded + resized ONCE and cached in RAM, so across many epochs the
slow JPEG-decode step is not repeated (the GPU is otherwise starved by disk I/O).
This makes training GPU-bound and much faster on a capable GPU. Disable with
RICE_CACHE=0.
"""
import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class SplitDataset(Dataset):
    """Loads records of a given split from a domain manifest JSON."""

    def __init__(self, manifest_path, split: str, train: bool = False):
        data = json.loads(manifest_path.read_text())
        self.records = [r for r in data["records"] if r["split"] == split]
        # Optional pairing-specific leakage exclusion: when RICE_EXTERNAL_EXCLUDE
        # points to a file of basenames, drop those from the external_test split
        # (used to de-duplicate the canopy target; off by default, see cross_leakage.py).
        excl_file = os.environ.get("RICE_EXTERNAL_EXCLUDE", "")
        if excl_file and split == "external_test":
            excl = {ln.strip() for ln in open(excl_file) if ln.strip()}
            self.records = [r for r in self.records
                            if os.path.basename(r["path"]) not in excl]
        if not self.records:
            raise ValueError(f"No records for split='{split}' in {manifest_path}")
        self.class_to_idx = {c: i for i, c in enumerate(config.CLASSES)}
        self.tf = build_transforms(train)
        self._cache_on = os.environ.get("RICE_CACHE", "1") != "0"
        self._cache = {}  # idx -> resized RGB PIL image (decoded once)

    def __len__(self):
        return len(self.records)

    def _image(self, idx):
        if self._cache_on:
            img = self._cache.get(idx)
            if img is None:
                img = (Image.open(config.ROOT / self.records[idx]["path"])
                       .convert("RGB")
                       .resize((config.IMG_SIZE, config.IMG_SIZE)))
                self._cache[idx] = img
            return img
        return Image.open(config.ROOT / self.records[idx]["path"]).convert("RGB")

    def __getitem__(self, idx):
        return self.tf(self._image(idx)), self.class_to_idx[self.records[idx]["class"]]
