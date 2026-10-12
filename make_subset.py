import os
import shutil
import random

BASE_DIR = "spark-2022-stream-1"  # run from the project root
SRC_IMG_DIR = os.path.join(BASE_DIR, "val", "val")
SRC_LABEL_DIR = os.path.join(BASE_DIR, "labels", "val")

DST_IMG_DIR = os.path.join(BASE_DIR, "val_subset", "images")
DST_LABEL_DIR = os.path.join(BASE_DIR, "val_subset", "labels")

if os.path.exists(DST_IMG_DIR): shutil.rmtree(DST_IMG_DIR)
if os.path.exists(DST_LABEL_DIR): shutil.rmtree(DST_LABEL_DIR)
os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_LABEL_DIR, exist_ok=True)

all_images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith('.jpg')]
random.seed(42)  # For reproducibility
subset = random.sample(all_images, min(300, len(all_images)))

for img in subset:
    shutil.copy2(os.path.join(SRC_IMG_DIR, img), os.path.join(DST_IMG_DIR, img))
    label_file = img.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(SRC_LABEL_DIR, label_file)):
        shutil.copy2(os.path.join(SRC_LABEL_DIR, label_file), os.path.join(DST_LABEL_DIR, label_file))

print(f"Created subset of {len(subset)} images in {DST_IMG_DIR}")
