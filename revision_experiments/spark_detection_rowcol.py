"""
Task A: SPARK cross-domain detection with CORRECT boxes.

Runs the Roboflow-trained detector (runs/detect/train3, never trained on SPARK)
on the same 300-image val_subset that produced latest_metrics.txt, three ways:

  old_labels_conf0.25 : val_subset/labels (fix_labels.py, x/y swapped), conf=0.25
                        -> should reproduce latest_metrics.txt
  rowcol_conf0.25     : spark_yolo/labels_rowcol (correct), conf=0.25 (same as old)
  rowcol_conf0.001    : spark_yolo/labels_rowcol (correct), Ultralytics default
                        val conf (the standard setting for mAP)

labels_rowcol uses SPARK class ids (debris=0 ...); train3 uses Roboflow ids
(alphabetical, debris=1). Labels are remapped by class NAME to the model ids.

Images are reached through a junction and an image-list .txt outside OneDrive so
Ultralytics reads the labels written here (see spark_yolo/make_rowcol_dataset.py).

Run from project root:
    python revision_experiments/spark_detection_rowcol.py
Output: revision_experiments/results/spark_detection_rowcol.json
"""
import os
import json
import glob
import shutil
import subprocess
import tempfile

import yaml
from ultralytics import YOLO

MODEL = "runs/detect/train3/weights/best.pt"
SUBSET_IMG = os.path.abspath("spark-2022-stream-1/val_subset/images")
OLD_LBL = os.path.abspath("spark-2022-stream-1/val_subset/labels")
ROWCOL_LBL = os.path.abspath("spark_yolo/labels_rowcol/val")
LINK_ROOT = os.path.join(os.path.expanduser("~"), "spark_rowcol_links", "val_subset")
OUT = "revision_experiments/results/spark_detection_rowcol.json"

SPARK_IDS = {"debris": 0, "lisa_pathfinder": 1, "soho": 2, "xmm_newton": 3, "cheops": 4,
             "smart_1": 5, "proba_2": 6, "proba_3_csc": 7, "proba_3_ocs": 8,
             "earth_observation_sat_1": 9, "double_star": 10}
SPARK_NAME = {v: k for k, v in SPARK_IDS.items()}

model = YOLO(MODEL)
# model.names has the Roboflow typo 'double_start'
MODEL_ID = {("double_star" if n == "double_start" else n): i for i, n in model.names.items()}
assert set(MODEL_ID) == set(SPARK_IDS), (MODEL_ID, SPARK_IDS)

images = sorted(glob.glob(os.path.join(SUBSET_IMG, "*.jpg")))
assert len(images) == 300, len(images)


def build(name, write_label):
    """Make LINK_ROOT/<name>/{images/val (junction), labels/val} and an image list."""
    root = os.path.join(LINK_ROOT, name)
    shutil.rmtree(os.path.join(root, "labels"), ignore_errors=True)
    os.makedirs(os.path.join(root, "labels", "val"))
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    link = os.path.join(root, "images", "val")
    if not os.path.exists(link):
        subprocess.run(["cmd", "/c", "mklink", "/J", link, SUBSET_IMG], check=True,
                       stdout=subprocess.DEVNULL)
    for im in images:
        base = os.path.splitext(os.path.basename(im))[0]
        write_label(base, os.path.join(root, "labels", "val", base + ".txt"))
    lst = os.path.join(root, "val.txt")
    with open(lst, "w") as f:
        f.write("\n".join(os.path.join(link, os.path.basename(im)) for im in images) + "\n")
    data = os.path.join(root, "data.yaml")
    with open(data, "w") as f:
        yaml.safe_dump({"path": root, "train": lst, "val": lst,
                        "names": {i: n for i, n in model.names.items()}}, f)
    return data


def old_label(base, dst):
    shutil.copy(os.path.join(OLD_LBL, base + ".txt"), dst)   # already in model ids


def rowcol_label(base, dst):
    with open(os.path.join(ROWCOL_LBL, base + ".txt")) as f, open(dst, "w") as g:
        for line in f:
            c, *box = line.split()
            g.write(" ".join([str(MODEL_ID[SPARK_NAME[int(c)]])] + box) + "\n")


def run(data, conf):
    with tempfile.TemporaryDirectory() as tmp:
        m = model.val(data=data, conf=conf, imgsz=640, batch=16, device="cpu",
                      plots=False, project=tmp, name="val", verbose=False)
    b = m.box
    per_class = {}
    for j, c in enumerate(b.ap_class_index):
        per_class[model.names[int(c)]] = {
            "P": round(float(b.p[j]), 4), "R": round(float(b.r[j]), 4),
            "mAP50": round(float(b.ap50[j]), 4), "mAP50_95": round(float(b.ap[j]), 4),
            "n_images": int(m.nt_per_image[int(c)]) if hasattr(m, "nt_per_image") else None}
    p, r, m50, m5095 = b.mean_results()
    return {"conf": conf, "P": round(float(p), 4), "R": round(float(r), 4),
            "mAP50": round(float(m50), 4), "mAP50_95": round(float(m5095), 4),
            "per_class": per_class}


old_yaml = build("old_labels", old_label)
new_yaml = build("rowcol", rowcol_label)
results = {
    "model": MODEL, "n_images": len(images), "imgsz": 640,
    "old_labels_conf0.25": run(old_yaml, 0.25),
    "rowcol_conf0.25": run(new_yaml, 0.25),
    "rowcol_conf0.001": run(new_yaml, 0.001),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
for k in ["old_labels_conf0.25", "rowcol_conf0.25", "rowcol_conf0.001"]:
    r = results[k]
    print(f"{k:22s} P={r['P']:.4f} R={r['R']:.4f} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f}")
print("Saved", OUT)
