from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import io

from src.inference_engine import InferenceEngine
from src.utils import load_config

app = FastAPI(title="Autonomous Vehicle Perception API", version="1.0.0")
config = load_config("configs/config.yaml")
engine = InferenceEngine(config)


class PredictionRequest(BaseModel):
    model: str = "yolov8n"
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cuda"


@app.get("/model/info")
def model_info():
    return {
        "status": "ok",
        "available_models": ["yolov8n", "yolov8s", "best", "last"],
        "device": engine.device,
        "classes": engine.class_names,
    }


@app.get("/system/status")
def system_status():
    return {
        "status": "ok",
        "cuda_available": engine.cuda_available,
        "device": engine.device,
        "model_loaded": engine.model is not None,
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), confidence: float = 0.25, iou: float = 0.45):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = engine.predict_image(image, confidence=confidence, iou=iou)
    return {
        "status": "ok",
        "detections": results["detections"],
        "stats": results["stats"],
        "image_path": results["image_path"],
    }


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...), confidence: float = 0.25, iou: float = 0.45):
    contents = await file.read()
    temp_path = f"outputs/{uuid.uuid4().hex}_{file.filename}"
    os.makedirs("outputs", exist_ok=True)
    with open(temp_path, "wb") as fh:
        fh.write(contents)
    output_path = engine.predict_video(temp_path, confidence=confidence, iou=iou)
    return {"status": "ok", "output_path": output_path}


@app.get("/docs")
def docs_redirect():
    return {"message": "Open /docs for Swagger UI"}
