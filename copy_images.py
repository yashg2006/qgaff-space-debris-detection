import os
import shutil

# -------- PATHS --------
SRC_DIR = "spark-2022-stream-1/val/val"
DST_DIR = "spark_yolo/images/val"
# ----------------------

os.makedirs(DST_DIR, exist_ok=True)

count = 0

for img_name in os.listdir(SRC_DIR):
    if img_name.lower().endswith(".jpg"):
        src_path = os.path.join(SRC_DIR, img_name)
        dst_path = os.path.join(DST_DIR, img_name)

        # Skip if already copied
        if os.path.exists(dst_path):
            continue

        shutil.copy2(src_path, dst_path)
        count += 1

        # Print progress every 500 images
        if count % 500 == 0:
            print(f"✅ Copied {count} images...")

print(f"🎉 DONE! Total images copied: {count}")
