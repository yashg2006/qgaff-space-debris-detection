"""
Convert SPARK-2022 stream-1 CSV labels to YOLO format.

The CSV "bbox" column is [rmin, cmin, rmax, cmax] in pixels, i.e.
[ymin, xmin, ymax, xmax] (rows first), NOT [x, y, width, height] and NOT
[xmin, ymin, xmax, ymax]. Example "[375, 273, 610, 496]" -> x 273..496, y 375..610.
This was checked by drawing both readings on 30 random val images.

Corrected labels are written to a NEW folder (spark_yolo/labels_rowcol/<split>/)
so the old labels are kept for comparison. Each label file is written once
("w" mode), so re-running the script does not duplicate lines.

Run from anywhere:
    python spark_yolo/csv_to_yolo.py
"""
import os
import ast
from collections import defaultdict

import pandas as pd
from PIL import Image

# ================== PATHS ==================
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["val", "train"]
BASE_DIR = os.path.join(HERE, "..", "spark-2022-stream-1")
OUTPUT_DIR = os.path.join(HERE, "labels_rowcol")
# ===========================================

# -------- CLASS NAME → ID MAPPING --------
CLASS_MAP = {
    "debris": 0,
    "lisa_pathfinder": 1,
    "soho": 2,
    "xmm_newton": 3,
    "cheops": 4,
    "smart_1": 5,
    "proba_2": 6,
    "proba_3_csc": 7,
    "proba_3_ocs": 8,
    "earth_observation_sat_1": 9,
    "double_star": 10
}
# -----------------------------------------

for split in SPLITS:
    CSV_PATH = os.path.join(BASE_DIR, "labels", f"{split}.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, split, split)
    OUTPUT_LABEL_DIR = os.path.join(OUTPUT_DIR, split)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Skipping {split} split.")
        continue

    print(f"Processing {split} split...")
    df = pd.read_csv(CSV_PATH)
    lines = defaultdict(list)
    stats = defaultdict(int)

    for _, row in df.iterrows():
        # The CSV has .png, but actual images are .jpg
        actual_image_name = row["filename"].replace(".png", ".jpg")
        class_name = row["class"]

        if class_name not in CLASS_MAP:
            stats["skipped_unknown_class"] += 1
            continue

        img_path = os.path.join(IMAGE_DIR, actual_image_name)
        if not os.path.exists(img_path):
            stats["skipped_missing_image"] += 1
            continue
        with Image.open(img_path) as im:   # reads the header only
            w, h = im.size

        # SPARK stores rows first: [rmin, cmin, rmax, cmax]
        rmin, cmin, rmax, cmax = ast.literal_eval(row["bbox"])
        x1, y1, x2, y2 = cmin, rmin, cmax, rmax

        if x2 <= x1 or y2 <= y1:
            stats["skipped_empty_box"] += 1
            continue

        # Clip to the image, and count how many boxes needed it
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats["clipped"] += 1
        if cx2 <= cx1 or cy2 <= cy1:
            stats["skipped_outside_image"] += 1
            continue

        # YOLO normalization from pixel corners
        x_center = (cx1 + cx2) / 2 / w
        y_center = (cy1 + cy2) / 2 / h
        width = (cx2 - cx1) / w
        height = (cy2 - cy1) / h

        lines[actual_image_name].append(
            f"{CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        stats["converted"] += 1

    for image_name, label_lines in lines.items():
        label_file = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(image_name)[0] + ".txt")
        with open(label_file, "w") as f:
            f.writelines(label_lines)

    print(f"Split '{split}': {dict(stats)} -> {len(lines)} label files in {OUTPUT_LABEL_DIR}")

print("Label conversion to YOLO format finished!")
