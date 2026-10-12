import os
from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model
    model = YOLO("yolov8n.pt") 

    print("🚀 Starting Rapid YOLO Training for SPARK Dataset...")
    results = model.train(
        data="spark.yaml",
        epochs=10,         # Need at least this much to get reasonable mAP
        imgsz=320,         # Half resolution computes 4x faster on CPU
        batch=8,           # Smaller batch for CPU
        patience=10,       
        device="cpu",      # Fallback to CPU!         
        workers=2,
        project="runs/spark",
        name="opt_train",
        optimizer="auto",
        fraction=0.03,     # Train on just 3% of the data to finish extremely fast natively!
    )
    
    print("\n✅ Training Complete. Models are in 'runs/spark/opt_train'!")

if __name__ == "__main__":
    main()
