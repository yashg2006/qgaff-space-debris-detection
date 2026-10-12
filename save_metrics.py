from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/train3/weights/best.pt")
metrics = model.val(data="data_cross.yaml", conf=0.25)

p = metrics.box.mean_results()[0]  
r = metrics.box.mean_results()[1]  
mAP50 = metrics.box.mean_results()[2]
mAP50_95 = metrics.box.mean_results()[3]

f1_per_class = metrics.box.f1
mean_f1 = np.mean(f1_per_class) if f1_per_class is not None and len(f1_per_class) > 0 else (2 * p * r) / (p + r + 1e-16)

with open("latest_metrics.txt", "w") as f:
    f.write(f"Precision (rp): {p:.4f}\n")
    f.write(f"Recall (r): {r:.4f}\n")
    f.write(f"F1 Score: {mean_f1:.4f}\n")
    f.write(f"mAP@0.5: {mAP50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {mAP50_95:.4f}\n")
