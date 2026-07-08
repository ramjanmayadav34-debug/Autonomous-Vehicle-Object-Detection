import os
from typing import Any, Dict, Optional

import torch
from torch import nn
from torchvision import models
from ultralytics import YOLO

from .utils import ensure_dir, setup_logging


logger = setup_logging(level="INFO")


class TrafficSignClassifier(nn.Module):
    def __init__(self, num_classes: int = 43, backbone: str = "mobilenet_v2") -> None:
        super().__init__()
        if backbone == "mobilenet_v2":
            try:
                self.backbone = models.mobilenet_v2(weights="DEFAULT")
            except Exception:
                logger.warning("Falling back to MobileNetV2 without pretrained weights.")
                self.backbone = models.mobilenet_v2(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_trained_classifier(checkpoint_path: str, num_classes: int = 43, device: Optional[torch.device] = None) -> TrafficSignClassifier:
    model = TrafficSignClassifier(num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
    if device is not None:
        model.to(device)
    return model


def build_pedestrian_detector(model_path: Optional[str] = None) -> YOLO:
    if model_path and os.path.exists(model_path):
        return YOLO(model_path)
    return YOLO("yolov8n.pt")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str, best_val: float) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }, path)


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
