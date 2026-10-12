from ultralytics import YOLO
import torch
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/train3/weights/best.pt"  # Pre-trained on SPARK/Zenodo
IMG_FOLDER  = "train/images"                      # Roboflow images
LABELS_FOLDER = "train/labels"                    # Roboflow labels (GT)
OUT_DIR     = "roboflow_quantum_fusion"
HOOK_LAYER  = 8          # Same as SPARK/Zenodo features
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
model.model.eval()

features_list = []
labels_list   = []

def hook_fn(module, input, output):
    features_list.append(output.detach().cpu().numpy())

hook = model.model.model[HOOK_LAYER].register_forward_hook(hook_fn)

img_files = sorted([
    f for f in os.listdir(IMG_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"[roboflow_features] Processing {min(len(img_files), 500)} images (Limit: 500) …")

for i, img_name in enumerate(img_files[:500], 1):
    img_path = os.path.join(IMG_FOLDER, img_name)
    
    # Run prediction just to trigger the hook and get the backbone features
    results = model.predict(source=img_path, verbose=False)
    
    # Get ground truth label from corresponding txt file
    label_path = os.path.join(LABELS_FOLDER, os.path.splitext(img_name)[0] + ".txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                # Take class ID of the first object
                cls_gt = int(lines[0].split()[0])
                labels_list.append(cls_gt)
            else:
                # No objects in label file
                labels_list.append(-1)
                # We still need to pop from features_list if we don't want it
                # But hook_fn appends every time model.predict is called.
                # So we keep count carefully.
    else:
        labels_list.append(-1)
        
    if i % 100 == 0:
        print(f"  [{i}/{len(img_files)}] done")

hook.remove()

# Global average pool: (N, C, H, W) → (N, C)
# Note: features_list has N items, each (1, C, H, W)
features = np.vstack([f.mean(axis=(2, 3)) for f in features_list])
labels   = np.array(labels_list)

# Filter out -1 labels (no objects)
mask = (labels != -1)
features = features[mask]
labels   = labels[mask]

np.save(os.path.join(OUT_DIR, "features.npy"),  features)
np.save(os.path.join(OUT_DIR, "labels.npy"),    labels)

print(f"\n[roboflow_features] features shape : {features.shape}")
print(f"[roboflow_features] labels   shape : {labels.shape}")
print(f"[roboflow_features] Saved to {OUT_DIR}/")
