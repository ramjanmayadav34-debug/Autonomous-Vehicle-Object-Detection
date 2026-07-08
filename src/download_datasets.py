import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests

from .utils import ensure_dir, setup_logging


logger = setup_logging(level="INFO")


def _download_and_extract(url: str, destination: str) -> None:
    ensure_dir(os.path.dirname(destination))
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    archive_path = destination
    with open(archive_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(os.path.dirname(archive_path))

    os.remove(archive_path)


def ensure_dataset_ready(root: str, dataset_name: str = "gtsrb") -> None:
    root_path = Path(root)
    if dataset_name.lower() == "gtsrb":
        if root_path.exists() and any(root_path.iterdir()):
            logger.info("GTSRB dataset already present at %s", root)
            return
        try:
            ensure_dir(str(root_path))
            archive_path = root_path.parent / "gtsrb.zip"
            _download_and_extract(
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152e081b5f0e8f4f8f87d/GTSRB_Final_Training_Images.zip",
                str(archive_path),
            )
            logger.info("Downloaded and extracted GTSRB dataset to %s", root)
        except Exception as exc:
            logger.warning("Automatic GTSRB download failed: %s. Please download manually and place files under %s", exc, root)
    elif dataset_name.lower() == "kitti":
        if root_path.exists() and any(root_path.iterdir()):
            logger.info("KITTI dataset already present at %s", root)
            return
        logger.warning("Automatic KITTI download is not configured in this starter project. Please download manually and place files under %s", root)
