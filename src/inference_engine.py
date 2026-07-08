import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .utils import ensure_dir, load_config, setup_logging


logger = setup_logging(level="INFO")


class InferenceEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._select_device(config.get("device", "cuda"))
        self.cuda_available = torch.cuda.is_available()
        self.model: Optional[YOLO] = None
        self.class_names = [
            "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian",
            "traffic_sign", "traffic_light", "lane_marking"
        ]
        self._load_model()

    def _select_device(self, requested: str) -> str:
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        model_path = self.config.get("model", {}).get("weights_path", "yolov8n.pt")
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning("Model weights not found at %s; using pretrained YOLOv8 nano weights", model_path)
            self.model = YOLO("yolov8n.pt")

    def predict_image(self, image: np.ndarray, confidence: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        start = time.perf_counter()
        results = self.model(image, conf=confidence, iou=iou, stream=False, imgsz=640)
        elapsed = time.perf_counter() - start
        detections = []
        stats = {
            "object_count": 0,
            "vehicle_count": 0,
            "pedestrian_count": 0,
            "traffic_sign_count": 0,
        }

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[int(cls_id)]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                })
                stats["object_count"] += 1
                if cls_name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                    stats["vehicle_count"] += 1
                elif cls_name == "pedestrian":
                    stats["pedestrian_count"] += 1
                elif cls_name == "traffic_sign":
                    stats["traffic_sign_count"] += 1

        output_path = self._save_result_image(image, detections, elapsed)
        return {
            "detections": detections,
            "stats": stats,
            "image_path": output_path,
            "inference_time": round(elapsed * 1000, 2),
            "fps": round(1.0 / max(elapsed, 1e-6), 2),
        }

    def predict_video(self, video_path: str, confidence: float = 0.25, iou: float = 0.45) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        output_path = str(self._project_root() / "outputs" / f"annotated_{uuid.uuid4().hex}.mp4")
        ensure_dir(os.path.dirname(output_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self._draw_detections(frame, confidence=confidence, iou=iou)
            writer.write(frame)
        cap.release()
        writer.release()
        return output_path

    def _draw_detections(self, frame: np.ndarray, confidence: float = 0.25, iou: float = 0.45) -> None:
        results = self.model(frame, conf=confidence, iou=iou, stream=False, imgsz=640) if self.model else []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[int(cls_id)] if self.model else "object"
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = self._class_color(cls_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _save_result_image(self, image: np.ndarray, detections: List[Dict[str, Any]], elapsed: float) -> str:
        output_path = str(self._project_root() / "outputs" / f"inference_{uuid.uuid4().hex}.jpg")
        ensure_dir(os.path.dirname(output_path))
        annotated = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            color = self._class_color(detection["class_name"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{detection['class_name']} {detection['confidence']:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(annotated, f"Inference: {elapsed * 1000:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(output_path, annotated)
        return output_path

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _class_color(self, class_name: str) -> tuple[int, int, int]:
        palette = {
            "car": (0, 255, 255),
            "truck": (255, 0, 0),
            "bus": (0, 0, 255),
            "motorcycle": (255, 255, 0),
            "bicycle": (0, 255, 0),
            "pedestrian": (255, 0, 255),
            "traffic_sign": (255, 165, 0),
            "traffic_light": (0, 165, 255),
            "lane_marking": (128, 128, 128),
        }
        return palette.get(class_name, (255, 255, 255))
