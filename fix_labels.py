import os
import csv
import ast
import glob

# Correct class mapping extracted exactly from the trained model's config
CLASS_MAP = {
    'cheops': 0,
    'debris': 1,
    'double_star': 2,
    'double_start': 2, # Catch typo in model
    'earth_observation_sat_1': 3,
    'lisa_pathfinder': 4,
    'proba_2': 5,
    'proba_3_csc': 6,
    'proba_3_ocs': 7,
    'smart_1': 8,
    'soho': 9,
    'xmm_newton': 10
}

IMG_SIZE = 1024

def convert_csv_to_yolo(csv_file, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            filename = row["filename"]
            cls = row["class"]
            bbox = row["bbox"]
            
            # Skip empty entries
            if not cls or not bbox:
                continue
                
            # Skip unknown classes
            if cls not in CLASS_MAP:
                continue
                
            class_id = CLASS_MAP[cls]
            xmin, ymin, xmax, ymax = ast.literal_eval(bbox)
            
            x_center = ((xmin + xmax) / 2) / IMG_SIZE
            y_center = ((ymin + ymax) / 2) / IMG_SIZE
            width = (xmax - xmin) / IMG_SIZE
            height = (ymax - ymin) / IMG_SIZE
            
            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            txt_name = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_label_dir, txt_name)
            
            with open(txt_path, "a", encoding="utf-8") as out:
                out.write(label_line)

if __name__ == "__main__":
    import shutil
    # Ensure previous misplaced labels are cleaned up or just outputting to specific directories
    BASE_DIR = "spark-2022-stream-1"  # run from the project root
    
    val_csv = os.path.join(BASE_DIR, "labels", "val.csv")
    train_csv = os.path.join(BASE_DIR, "labels", "train.csv")
    
    val_out = os.path.join(BASE_DIR, "labels", "val")
    train_out = os.path.join(BASE_DIR, "labels", "train")
    if os.path.exists(val_out): shutil.rmtree(val_out)
    if os.path.exists(train_out): shutil.rmtree(train_out)
    
    print("Converting validation labels...")
    convert_csv_to_yolo(val_csv, val_out)
    
    print("Converting training labels...")
    convert_csv_to_yolo(train_csv, os.path.join(BASE_DIR, "labels", "train"))
    
    print("Finished converting labels successfully!")
