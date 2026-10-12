import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MODEL_PATH = "runs/detect/train3/weights/best.pt"
IMG_DIR = os.path.join("Roboflow Dataset", "valid", "images")
LABELS_DIR = os.path.join("Roboflow Dataset", "valid", "labels")
OUT_DIR = "roboflow_quantum_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[*] Extracting features from: {IMG_DIR}...")
model = YOLO(MODEL_PATH)
model.model.eval()

features_list = []
labels_list = []

def hook_fn(module, input, output):
    features_list.append(output.detach().cpu().numpy())

# Register forward hook on layer 8 (C2f block, 256 channels)
hook = model.model.model[8].register_forward_hook(hook_fn)

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"[*] Found {len(img_files)} images in validation set.")

for i, img_name in enumerate(img_files, 1):
    img_path = os.path.join(IMG_DIR, img_name)
    results = model.predict(source=img_path, verbose=False)
    
    # Ground truth label
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABELS_DIR, txt_name)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 0 and lines[0].strip():
                cls_id = int(lines[0].split()[0])
                labels_list.append(cls_id)
            else:
                labels_list.append(-1)
    else:
        labels_list.append(-1)

hook.remove()

# Global Average Pooling: (N, C, H, W) -> (N, C)
features = np.vstack([f.mean(axis=(2, 3)) for f in features_list])
labels = np.array(labels_list)

# Filter out -1
mask = labels != -1
features = features[mask]
labels = labels[mask]

np.save(os.path.join(OUT_DIR, "features.npy"), features)
np.save(os.path.join(OUT_DIR, "labels.npy"), labels)

print(f"[OK] Extracted {len(features)} valid samples from Roboflow Dataset.")
print(f"     Features shape: {features.shape}, Labels classes: {np.unique(labels).tolist()}")
