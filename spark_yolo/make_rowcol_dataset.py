"""
Make the corrected SPARK labels (spark_yolo/labels_rowcol/) usable by Ultralytics
without copying images.

Ultralytics finds a label by replacing "/images/" with "/labels/" in the image
path. The SPARK images live in spark-2022-stream-1/<split>/<split>/, next to the
OLD (wrong) .txt labels, so pointing a yaml at that folder would load the old labels.

This script:
  1. creates directory junctions OUTSIDE OneDrive (so OneDrive does not sync a
     second copy of the images):
        LINK_ROOT/images/<split> -> spark-2022-stream-1/<split>/<split>
        LINK_ROOT/labels/<split> -> spark_yolo/labels_rowcol/<split>
  2. writes spark_yolo/rowcol_<split>.txt image lists through those junctions.
     spark.yaml points at these lists. (Ultralytics resolve()s directory paths in
     the yaml, which would follow the junction back to the old labels, but it does
     not resolve the paths listed inside a .txt file.)

Run from anywhere:
    python spark_yolo/make_rowcol_dataset.py
"""
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
LINK_ROOT = os.path.join(os.path.expanduser("~"), "spark_rowcol_links")
SPLITS = ["train", "val"]

for split in SPLITS:
    targets = {
        "images": os.path.join(PROJECT, "spark-2022-stream-1", split, split),
        "labels": os.path.join(HERE, "labels_rowcol", split),
    }
    for kind, target in targets.items():
        link = os.path.join(LINK_ROOT, kind, split)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.exists(link):
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True,
                           stdout=subprocess.DEVNULL)
        print(f"{link} -> {target}")

    # Image list from the CSV (every CSV row has an image and a label; see csv_to_yolo.py)
    names = pd.read_csv(os.path.join(PROJECT, "spark-2022-stream-1", "labels", f"{split}.csv"))["filename"]
    lines = [os.path.join(LINK_ROOT, "images", split, n.replace(".png", ".jpg")) for n in sorted(names)]
    out = os.path.join(HERE, f"rowcol_{split}.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} images")
