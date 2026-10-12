from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

print("Starting cross-domain validation...")
metrics = model.val(data="data_cross.yaml", save_json=True, conf=0.25)

print("\n--- Cross-Domain Testing Results ---")

# Accessing mean F1, Precision, Recall across classes
# metrics.box.f1 is an array of F1 scores per class. We take mean.
p = metrics.box.mean_results()[0]  # mean precision
r = metrics.box.mean_results()[1]  # mean recall
mAP50 = metrics.box.mean_results()[2]
mAP50_95 = metrics.box.mean_results()[3]

# There is a shortcut for F1 if it's available, otherwise we compute from P and R 
import numpy as np
f1_per_class = metrics.box.f1
mean_f1 = np.mean(f1_per_class) if f1_per_class is not None and len(f1_per_class) > 0 else (2 * p * r) / (p + r + 1e-16)

print(f"Precision (rp): {p:.4f}")
print(f"Recall (r): {r:.4f}")
print(f"F1 Score: {mean_f1:.4f}")
print(f"mAP@0.5: {mAP50:.4f}")
print(f"mAP@0.5:0.95: {mAP50_95:.4f}")

# Confirmation on normalized confusion matrix
print("Normalized Confusion Matrix successfully generated and saved to runs/detect/val* directory!")
