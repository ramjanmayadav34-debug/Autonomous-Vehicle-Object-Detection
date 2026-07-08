import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("autonomous_vehicle_detection")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def save_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)
