from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=15,
    imgsz=640,
    batch=2,
    workers=0,
    device="cpu"
)

model = YOLO("runs/detect/train2/weights/best.pt")

results = model.predict(
    source="test/images",
    conf=0.25,
    save=True
)
