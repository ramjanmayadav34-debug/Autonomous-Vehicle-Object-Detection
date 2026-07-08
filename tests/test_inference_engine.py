import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference_engine import InferenceEngine
from src.utils import load_config


def test_engine_initialization():
    config = load_config("configs/config.yaml")
    engine = InferenceEngine(config)
    assert engine is not None


def test_dummy_prediction():
    config = load_config("configs/config.yaml")
    engine = InferenceEngine(config)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = engine.predict_image(image, confidence=0.1, iou=0.3)
    assert "detections" in result
    assert "stats" in result
