import os
import shutil
import yaml
import glob
from roboflow import Roboflow
import subprocess
import cv2
import json

# Unified Class Mapping
# We want our final model to output these 4 specific classes:
UNIFIED_CLASSES = {
    0: "infrastructure",
    1: "person",
    2: "vehicle",
    3: "watercraft"
}

# Global class mapping from dataset label strings to our unified class IDs
CLASS_NAME_TO_UNIFIED_ID = {
    # Infrastructure (Class 0)
    "flooded": 0,
    "partially-flooded": 0,
    "not-flooded": 0,
    "house": 0,
    "houses": 0,
    "building": 0,
    "infrastructure": 0,
    "dock": 0,
    
    # Person (Class 1)
    "person": 1,
    "people": 1,
    "pedestrian": 1,
    "swimmer": 1,
    "survivor": 1,
    "human": 1,
    
    # Vehicle (Class 2)
    "car": 2,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "motor": 2,
    "motorcycle": 2,
    "bicycle": 2,
    "tricycle": 2,
    "awning-tricycle": 2,
    "vehicle": 2,
    
    # Watercraft (Class 3)
    "boat": 3,
    "jetski": 3,
    "life_saving_appliances": 3,
    "buoy": 3,
    "watercraft": 3,
    "vessel": 3,
    "ship": 3,
    "rescue_tag": 3
}

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "ZugxIA3oPeoHBSmz7dB3")
MASTER_DIR = os.path.join(os.getcwd(), "datasets", "master_vision")

def setup_directories():
    if os.path.exists(MASTER_DIR):
        shutil.rmtree(MASTER_DIR)
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(MASTER_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MASTER_DIR, split, "labels"), exist_ok=True)

def fast_copy(src, dst):
    """Attempt hard link first for maximum speed and zero extra disk usage, fallback to copy."""
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def download_and_merge_roboflow(rf, workspace_name, project_name, version_num, default_unified_id=None):
    """Downloads a Roboflow dataset and smartly remaps all its labels to unified classes."""
    print(f"\n[Dataset] Downloading {workspace_name}/{project_name} (v{version_num})...")
    workspace = rf.workspace(workspace_name)
    proj = workspace.project(project_name)
    
    # Download dataset
    dataset = proj.version(version_num).download("yolov8")
    dataset_path = dataset.location
    
    # Inspect data.yaml to find the mapping of old class IDs in this downloaded dataset
    yaml_path = os.path.join(dataset_path, "data.yaml")
    old_to_new = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        names = d.get("names", {})
        if isinstance(names, list):
            for old_id, name in enumerate(names):
                name_lower = str(name).strip().lower()
                if name_lower in CLASS_NAME_TO_UNIFIED_ID:
                    old_to_new[old_id] = CLASS_NAME_TO_UNIFIED_ID[name_lower]
                elif default_unified_id is not None:
                    old_to_new[old_id] = default_unified_id
        elif isinstance(names, dict):
            for old_id, name in names.items():
                name_lower = str(name).strip().lower()
                if name_lower in CLASS_NAME_TO_UNIFIED_ID:
                    old_to_new[int(old_id)] = CLASS_NAME_TO_UNIFIED_ID[name_lower]
                elif default_unified_id is not None:
                    old_to_new[int(old_id)] = default_unified_id
    
    print(f"[Dataset] Merging {project_name} into Master Dataset with mapping: {old_to_new}...")
    
    img_count = 0
    box_count = 0
    
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        for img_file in os.listdir(img_dir):
            base_name = os.path.splitext(img_file)[0]
            lbl_file = base_name + ".txt"
            
            src_lbl = os.path.join(lbl_dir, lbl_file)
            valid_lines = []
            
            if os.path.exists(src_lbl):
                with open(src_lbl, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_id = int(parts[0])
                        new_id = old_to_new.get(old_id, default_unified_id)
                        if new_id is not None:
                            valid_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                            box_count += 1

            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(MASTER_DIR, split, "images", f"{project_name}_{img_file}")
            fast_copy(src_img, dst_img)
            
            dst_lbl = os.path.join(MASTER_DIR, split, "labels", f"{project_name}_{lbl_file}")
            with open(dst_lbl, "w") as f:
                f.writelines(valid_lines)
            
            img_count += 1
            
    print(f"[Dataset] Successfully added {img_count} images and {box_count} bounding boxes from {project_name}.")

def pull_youtube_and_auto_annotate():
    print("\n[Dataset] Downloading a relevant YouTube video for extra data...")
    cmd = [
        "yt-dlp",
        "ytsearch1:drone flood rescue people",
        "--match-filter", "duration < 180",
        "-f", "b[ext=mp4]",
        "-o", "youtube_raw.mp4"
    ]
    subprocess.run(cmd, check=False)
    
    if not os.path.exists("youtube_raw.mp4") and os.path.exists("tests/disaster_sample.mp4"):
        print("[Dataset] Failed to download YouTube video, using existing disaster_sample.mp4...")
        shutil.copy("tests/disaster_sample.mp4", "youtube_raw.mp4")

    if not os.path.exists("youtube_raw.mp4"):
        print("[Dataset] No video available for auto-annotation, skipping.")
        return

    print("[Dataset] Auto-annotating video frames using YOLOv10 COCO base model...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov10n.pt")
    except Exception as e:
        print(f"[Dataset] Skipping auto-annotation due to model error: {e}")
        return
    
    cap = cv2.VideoCapture("youtube_raw.mp4")
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            results = model(frame, verbose=False)
            labels = []
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # COCO mappings: 0=person, 2,3,5,7=vehicles, 8=boat
                if conf > 0.2:
                    if cls_id == 0:  # person -> 1 (person)
                        unified_id = 1 
                    elif cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck -> 2 (vehicle)
                        unified_id = 2
                    elif cls_id == 8:  # boat -> 3 (watercraft)
                        unified_id = 3
                    else:
                        continue
                        
                    xywh = box.xywhn[0]
                    labels.append(f"{unified_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
            
            if labels:
                img_name = f"youtube_frame_{saved_count}.jpg"
                lbl_name = f"youtube_frame_{saved_count}.txt"
                
                cv2.imwrite(os.path.join(MASTER_DIR, "train", "images", img_name), frame)
                with open(os.path.join(MASTER_DIR, "train", "labels", lbl_name), "w") as f:
                    f.write("\n".join(labels))
                
                saved_count += 1
                
        frame_count += 1
        
    cap.release()
    print(f"[Dataset] Auto-annotated and added {saved_count} frames from video stream.")

def build_data_yaml():
    yaml_content = {
        "path": MASTER_DIR,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {k: v for k, v in UNIFIED_CLASSES.items()}
    }
    with open(os.path.join(MASTER_DIR, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print("\n[Dataset] Generated data.yaml.")

if __name__ == "__main__":
    setup_directories()
    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace_name = "sohails-workspace-pvlqk"
    
    # 1. Houses in Flood -> Class 0 (infrastructure)
    try:
        download_and_merge_roboflow(rf, workspace_name, "houses_in_flood-62zqj", 1, default_unified_id=0)
    except Exception as e:
        print(f"Warning: Failed to download houses_in_flood: {e}")
        
    # 2. SeaDrone -> Classes 1 (person/swimmer) & 3 (watercraft/boat/buoy)
    try:
        download_and_merge_roboflow(rf, workspace_name, "seadrone-opbc9-slqmz", 1)
    except Exception as e:
        print(f"Warning: Failed to download seadrone: {e}")
        
    # 3. Aerial Person Detection -> Classes 1 (person/pedestrian) & 2 (vehicle/car/truck/van)
    try:
        download_and_merge_roboflow(rf, workspace_name, "aerial-person-detection-hudj7", 1)
    except Exception as e:
        print(f"Warning: Failed to download aerial person detection: {e}")
        
    # Optional: Video enrichment
    pull_youtube_and_auto_annotate()
    
    build_data_yaml()
    
    print("\n[Dataset] Zipping master dataset...")
    shutil.make_archive("master_dataset", "zip", "datasets/master_vision")
    print("✅ Master Dataset successfully generated at master_dataset.zip!")
