import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .download_datasets import ensure_dataset_ready
from .utils import resize_image, setup_logging


logger = setup_logging(level="INFO")


class GTSRBDataset(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 224, transform: Optional[transforms.Compose] = None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform or self._default_transform()
        self.samples = self._load_samples()

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        if not os.path.exists(self.root):
            logger.warning("GTSRB dataset not found at %s", self.root)
            return samples
        for class_dir in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_dir)
            if not os.path.isdir(class_path):
                continue
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(class_path, file_name), int(class_dir)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.transform(image) if isinstance(self.transform, transforms.Compose) else image
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


class KITTIDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.samples = self._load_samples()

    def _load_samples(self) -> List[str]:
        if not os.path.exists(self.root):
            return []
        files = []
        for root, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    files.append(os.path.join(root, filename))
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.samples[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"image": image, "image_path": image_path}


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 4))
    dataset_root = config["dataset"].get("gtsrb_root", "datasets/gtsrb")
    ensure_dataset_ready(dataset_root, dataset_name="gtsrb")

    train_dataset = GTSRBDataset(dataset_root, split="train", image_size=image_size)
    val_dataset = GTSRBDataset(dataset_root, split="val", image_size=image_size)
    test_dataset = GTSRBDataset(dataset_root, split="test", image_size=image_size)

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples were found in the GTSRB dataset directory. Download the data first or point the config to a valid location.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
