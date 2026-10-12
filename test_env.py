import torch
from ultralytics import YOLO
import pennylane as qml

print("Torch version:", torch.__version__)
print("PennyLane version:", qml.version())

# Test YOLO load
model = YOLO("yolov8n.pt")
print("YOLO Loaded")

# Test prediction
results = model.predict(source="train/images/img000008_jpg.rf.2c95cb6fb03fec5dc092c669adab86ea.jpg", imgsz=640)
print("Prediction Done")
print("Classes detected:", results[0].boxes.cls)
