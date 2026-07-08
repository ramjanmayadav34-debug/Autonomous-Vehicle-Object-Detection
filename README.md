# Autonomous Vehicle Object Detection

This project provides a production-oriented Python pipeline for detecting traffic signs and pedestrians from images, videos, and webcam streams.

## Features
- Traffic sign classification with MobileNetV2
- Pedestrian detection with YOLOv8
- Training, evaluation, inference, and Streamlit UI
- Dataset download helpers for GTSRB and KITTI
- ONNX export and explainability utilities

## Project Structure
```text
autonomous_vehicle_detection/
├── datasets/
├── models/
├── checkpoints/
├── outputs/
├── notebooks/
├── configs/
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── detect.py
│   ├── webcam.py
│   ├── utils.py
│   ├── model.py
│   └── visualization.py
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation
- GTSRB: automatic download is attempted when internet access is available.
- KITTI: the script can download the object detection subset when available.

## Training
```bash
python src/train.py --config configs/config.yaml
```

## Evaluation
```bash
python src/evaluate.py --config configs/config.yaml
```

## Inference
```bash
python src/detect.py --source path/to/image.jpg --config configs/config.yaml
```

## Webcam
```bash
python src/webcam.py --config configs/config.yaml
```

## Streamlit App
```bash
streamlit run app.py
```

## Export to ONNX
```bash
python src/detect.py --export_onnx --config configs/config.yaml
```
