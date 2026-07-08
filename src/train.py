import argparse
import os
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data_loader import build_dataloaders
from .model import TrafficSignClassifier, save_checkpoint
from .utils import load_config, set_seed, setup_logging


logger = setup_logging(level="INFO")


def train_model(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, _ = build_dataloaders(config)
    model = TrafficSignClassifier(num_classes=43, backbone=config["model"].get("traffic_sign_model", "mobilenet_v2"))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=float(config.get("weight_decay", 0.0001)))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = GradScaler(enabled=bool(config.get("mixed_precision", True)) and torch.cuda.is_available())
    writer = SummaryWriter(log_dir=config.get("log_dir", "outputs/runs"))

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(int(config.get("epochs", 5))):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "best_model.pt")
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= int(config.get("patience", 3)):
                logger.info("Early stopping triggered")
                break

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the autonomous vehicle detection pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
