import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(losses: List[float], val_losses: List[float], save_path: str = "outputs/training_curves.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(confusion: np.ndarray, save_path: str = "outputs/confusion_matrix.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_gradcam(model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    return np.zeros_like(image.detach().cpu().numpy().transpose(1, 2, 0))
