"""
verify_spark_labels.py
========================
Draws YOLO-format SPARK-2022 boxes on random SPARK images so they can be
checked by eye (Editor Comment 5), and reports boxes that are empty,
clipped at the image border, or outside the image.

Green  = labels in --labels (default: corrected spark_yolo/labels_xyxy/val)
Red    = labels in --compare (optional, e.g. the old labels)

Overlays and a JSON report are written to --out. This script does not decide
whether a box is correct; open the overlays and look at them.

Run from project root:
    python verify_spark_labels.py
    python verify_spark_labels.py --compare spark-2022-stream-1/val/val
"""

import os
import sys
import json
import random
import argparse
import glob

import cv2

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BORDER_TOL = 1.0 / 1024   # a box edge within ~1 px of the border counts as clipped
ROUND_EPS = 1e-5          # labels are written with 6 decimals, so edges at 1.0 can read as 1.0000005


def read_boxes(lbl_path):
    """Return list of (cls, xc, yc, w, h) from a YOLO label file (None if missing)."""
    if lbl_path is None or not os.path.exists(lbl_path):
        return None
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def check_box(xc, yc, bw, bh):
    """Return the list of problems for one normalized box."""
    problems = []
    x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
    if bw <= 0 or bh <= 0:
        problems.append("empty")
    if x2 <= 0 or y2 <= 0 or x1 >= 1 or y1 >= 1:
        problems.append("fully_outside")
    elif x1 < -ROUND_EPS or y1 < -ROUND_EPS or x2 > 1 + ROUND_EPS or y2 > 1 + ROUND_EPS:
        problems.append("partly_outside")
    if (abs(x1) < BORDER_TOL or abs(y1) < BORDER_TOL or
            abs(x2 - 1) < BORDER_TOL or abs(y2 - 1) < BORDER_TOL):
        problems.append("clipped_at_border")
    return problems


def draw(img, boxes, color, tag):
    h, w = img.shape[:2]
    for cls_id, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{tag} cls{cls_id}", (max(x1, 0), max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="spark-2022-stream-1/val/val")
    ap.add_argument("--labels", default="spark_yolo/labels_xyxy/val")
    ap.add_argument("--compare", default=None, help="optional second label dir, drawn in red")
    ap.add_argument("--out", default="spark_label_verify/labels_xyxy_val")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if not all_imgs:
        sys.exit(f"[!] No images found in {args.images}")
    random.seed(args.seed)
    sample = random.sample(all_imgs, min(args.n, len(all_imgs)))
    print(f"Images: {args.images}\nLabels: {args.labels}\nCompare: {args.compare}\n"
          f"Sample: {len(sample)} images (seed {args.seed}) -> {args.out}\n")

    report = []
    counts = {"no_label_file": 0, "no_boxes": 0, "empty": 0, "fully_outside": 0,
              "partly_outside": 0, "clipped_at_border": 0, "ok": 0}
    for img_path in sample:
        name = os.path.basename(img_path)
        base = os.path.splitext(name)[0]
        boxes = read_boxes(os.path.join(args.labels, base + ".txt"))
        entry = {"image": name, "boxes": boxes, "problems": []}
        img = cv2.imread(img_path)

        if boxes is None:
            entry["problems"].append("no_label_file")
        elif not boxes:
            entry["problems"].append("no_boxes")
        else:
            for _, xc, yc, bw, bh in boxes:
                entry["problems"] += check_box(xc, yc, bw, bh)
            draw(img, boxes, (0, 220, 0), "new")

        if args.compare:
            old = read_boxes(os.path.join(args.compare, base + ".txt"))
            entry["compare_boxes"] = old
            if old:
                draw(img, old, (0, 0, 255), "old")

        for p in set(entry["problems"]) or {"ok"}:
            counts[p] += 1
        report.append(entry)
        cv2.imwrite(os.path.join(args.out, name), img)
        print(f"  {name}: {boxes}  problems={entry['problems'] or 'none'}")

    with open(os.path.join(args.out, "label_check_report.json"), "w") as f:
        json.dump({"args": vars(args), "counts": counts, "images": report}, f, indent=2)

    print(f"\nImages with each problem: {counts}")
    print(f"Overlays + label_check_report.json in {os.path.abspath(args.out)}")
    print("Open the overlays and check the green boxes against the objects by eye.")


if __name__ == "__main__":
    main()
