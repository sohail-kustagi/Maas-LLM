import os
import glob
from collections import defaultdict

MASTER_DIR = os.path.join("datasets", "master_vision")
CLASSES = {0: "fire", 1: "flood_water", 2: "person", 3: "rescue_tag"}

def analyze_dataset():
    if not os.path.exists(MASTER_DIR):
        print(f"Error: {MASTER_DIR} not found.")
        return

    print("=== Dataset Quality Analysis ===")
    
    total_images = 0
    total_labels = 0
    class_counts = defaultdict(int)
    empty_images = 0
    bbox_areas = []

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(MASTER_DIR, split, "images")
        lbl_dir = os.path.join(MASTER_DIR, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        images = glob.glob(os.path.join(img_dir, "*.*"))
        total_images += len(images)
        
        for img_path in images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")
            
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                total_labels += 1
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            class_counts[cls_id] += 1
                            
                            w = float(parts[3])
                            h = float(parts[4])
                            bbox_areas.append(w * h)
            else:
                empty_images += 1

    print(f"\nTotal Images: {total_images}")
    print(f"Annotated Images (with objects): {total_labels}")
    print(f"Background Images (no objects): {empty_images}")
    
    print("\n--- Class Distribution ---")
    for cls_id, name in CLASSES.items():
        print(f"[{cls_id}] {name.ljust(15)}: {class_counts.get(cls_id, 0)} bounding boxes")
        
    if bbox_areas:
        avg_area = sum(bbox_areas) / len(bbox_areas)
        print(f"\n--- Bounding Box Quality ---")
        print(f"Average Object Size: {avg_area * 100:.2f}% of the image frame")
        if avg_area < 0.01:
            print("WARNING: Objects are extremely small (less than 1% of the frame). Consider increasing YOLO input resolution (imgsz).")
        else:
            print("INFO: Object sizes are healthy and suitable for standard 640x640 YOLO training.")
    else:
        print("\nWARNING: No bounding boxes found in the entire dataset!")

if __name__ == "__main__":
    analyze_dataset()
