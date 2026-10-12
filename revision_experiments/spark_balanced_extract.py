"""
Task B, step 1: class-balanced SPARK subset with GROUND-TRUTH labels.

Samples 70 val images per class (11 classes, 770 images) from
spark-2022-stream-1/labels/val.csv with seed 42, and extracts layer-8 GAP
features with the Roboflow-trained detector (runs/detect/train3), using the same
hook as spark_extract_features_subset.py. Unlike that script, labels come from the
CSV (not from the detector's predictions) and no image is dropped.

Run from project root:
    python revision_experiments/spark_balanced_extract.py
Output (revision_experiments/spark_balanced/):
    features.npy (770, 256) float32, labels.npy (770,) SPARK class ids,
    subset.csv (filename, class, label) in the same row order
"""
import os

import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train3/weights/best.pt"
CSV = "spark-2022-stream-1/labels/val.csv"
IMG_DIR = "spark-2022-stream-1/val/val"
OUT_DIR = "revision_experiments/spark_balanced"
HOOK_LAYER = 8
PER_CLASS = 70
SEED = 42

SPARK_IDS = {"debris": 0, "lisa_pathfinder": 1, "soho": 2, "xmm_newton": 3, "cheops": 4,
             "smart_1": 5, "proba_2": 6, "proba_3_csc": 7, "proba_3_ocs": 8,
             "earth_observation_sat_1": 9, "double_star": 10}

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV)
sub = (df.groupby("class", group_keys=False)
         .apply(lambda g: g.sample(n=PER_CLASS, random_state=SEED))
         .sort_values("filename").reset_index(drop=True))
sub["label"] = sub["class"].map(SPARK_IDS)
sub["filename"] = sub["filename"].str.replace(".png", ".jpg")
assert len(sub) == PER_CLASS * len(SPARK_IDS) and sub["label"].value_counts().eq(PER_CLASS).all()

model = YOLO(MODEL_PATH)
model.model.eval()
feats = []
hook = model.model.model[HOOK_LAYER].register_forward_hook(
    lambda m, i, o: feats.append(o.detach().cpu().numpy().mean(axis=(2, 3))))
for k, name in enumerate(sub["filename"], 1):
    model.predict(source=os.path.join(IMG_DIR, name), verbose=False)
    if k % 100 == 0:
        print(f"  [{k}/{len(sub)}]")
hook.remove()

features = np.vstack(feats).astype(np.float32)
assert features.shape[0] == len(sub)
np.save(os.path.join(OUT_DIR, "features.npy"), features)
np.save(os.path.join(OUT_DIR, "labels.npy"), sub["label"].to_numpy())
sub[["filename", "class", "label"]].to_csv(os.path.join(OUT_DIR, "subset.csv"), index=False)
print("features", features.shape, "labels", sub["label"].value_counts().sort_index().to_dict())
