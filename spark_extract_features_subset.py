from ultralytics import YOLO
import torch
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/train3/weights/best.pt"
IMG_FOLDER  = "spark_yolo/images/val"
OUT_DIR     = "spark_quantum_fusion"
HOOK_LAYER  = 8          # C2f block
SUBSET_LIMIT = 1000
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

# Limit to 1000 images
img_files = img_files[:SUBSET_LIMIT]

print(f"[spark_features] Processing {len(img_files)} images (Limit: {SUBSET_LIMIT}) …")
for i, img_name in enumerate(img_files, 1):
    path = os.path.join(IMG_FOLDER, img_name)
    results = model.predict(source=path, verbose=False)
    
    # We'll use predicted class for now to stay consistent with previous extraction logic
    # but the Goal is to fix the mismatch.
    cls = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    labels_list.append(int(cls[0]) if len(cls) > 0 else -1)
    
    if i % 100 == 0:
        print(f"  [{i}/{len(img_files)}] done")

hook.remove()

# Global average pool: (N, C, H, W) → (N, C)
features = np.vstack([f.mean(axis=(2, 3)) for f in features_list])
labels   = np.array(labels_list)

# Filter out -1
mask = labels != -1
features = features[mask]
labels = labels[mask]

np.save(os.path.join(OUT_DIR, "features.npy"),  features)
np.save(os.path.join(OUT_DIR, "labels.npy"),    labels)

print(f"\n[spark_features] features shape : {features.shape}")
print(f"[spark_features] labels   shape : {labels.shape}")
print(f"[spark_features] Saved to {OUT_DIR}/")
