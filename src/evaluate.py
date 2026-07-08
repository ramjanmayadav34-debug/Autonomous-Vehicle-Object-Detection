import argparse
import os
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .data_loader import build_dataloaders
from .model import TrafficSignClassifier, load_trained_classifier
from .utils import load_config, setup_logging


logger = setup_logging(level="INFO")


def evaluate_model(config: Dict[str, Any]) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(config)
    checkpoint_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "best_model.pt")
    model = load_trained_classifier(checkpoint_path, num_classes=43, device=device)
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds.tolist())
            targets.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the autonomous vehicle detection pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
