import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from .model import TrafficSignClassifier, build_pedestrian_detector, load_trained_classifier
from .utils import load_config, read_image, setup_logging


logger = setup_logging(level="INFO")


def _get_output_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_traffic_sign_inference(image: np.ndarray, config: Dict[str, Any]) -> tuple[np.ndarray, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "best_model.pt")
    model = load_trained_classifier(checkpoint_path, num_classes=43, device=device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())

    label = f"traffic_sign:{predicted_class} ({confidence:.2f})"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image, label


def run_inference_on_image(source: Union[str, Any], config: Dict[str, Any], model_choice: str = "Both") -> str:
    if hasattr(source, "read"):
        image_bytes = source.read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(image_bytes)
            temp_path = handle.name
        image = cv2.imread(temp_path)
        os.remove(temp_path)
    else:
        image = read_image(source) if isinstance(source, str) else np.array(source)

    if image is None:
        raise ValueError("Unable to process input image")

    annotated = image.copy()
    if model_choice in {"Traffic Sign", "Both"}:
        annotated, _ = _run_traffic_sign_inference(annotated, config)
    if model_choice in {"Pedestrian", "Both"}:
        try:
            detector = build_pedestrian_detector()
            cv2.putText(annotated, "pedestrian: model-ready", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as exc:
            logger.warning("Pedestrian detector unavailable: %s", exc)

    output_path = _get_output_dir() / "inference_result.jpg"
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        raise RuntimeError(f"Failed to save inference image to {output_path}")
    logger.info("Saved inference output to %s", output_path)
    return str(output_path)


def run_inference_on_video(source: Union[str, Any], config: Dict[str, Any], model_choice: str = "Both") -> str:
    output_path = _get_output_dir() / "inference_video.mp4"
    logger.info("Video inference placeholder for %s", source)
    return str(output_path)


def run_inference_on_webcam(config: Dict[str, Any], model_choice: str = "Both") -> None:
    logger.info("Webcam inference placeholder")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on image/video/webcam")
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.export_onnx:
        logger.info("ONNX export is not implemented in this starter skeleton")
    else:
        if args.source:
            run_inference_on_image(args.source, config)
        else:
            run_inference_on_webcam(config)


if __name__ == "__main__":
    main()
