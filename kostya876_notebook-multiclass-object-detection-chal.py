!pip install ultralytics --upgrade -q
from ultralytics import YOLO


data_yaml='''

train:  /kaggle/input/d/kostya876/multi-class-object-detection-challenge/merged_dataset_all/train/images
val:  /kaggle/input/d/kostya876/multi-class-object-detection-challenge/merged_dataset_all/val/images
test:  /kaggle/input/multi-class-object-detection-challenge/testImages/images
nc: 2
names: ['cheerios', 'soup']
'''
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


from ultralytics import YOLO
import yaml, os, shutil
import torch
import random
import numpy as np
from pathlib import Path
import os

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

model = YOLO("yolo11x.pt")

model.train(
    data="yolo_params.yaml",
    epochs=20,                
    batch=12,                   
    imgsz=640,
    patience=50,               
    optimizer='SGD',
    momentum=0.937,          
    lr0=0.001,                
    weight_decay=0.0005,       
    cos_lr=True,               
    save_period=5,             
    workers=8,
    # Augmentations
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    shear=0.01,
    agnostic_nms=True,
    project="Duality-YOLO-Local",
    name="run-1",
)


model.train(
    data="data.yaml",
    exist_ok=False,
    epochs=100,
    batch=16,
    imgsz=640,
    patience=100,               
    optimizer='SGD',
    momentum=0.937,          
    lr0=0.0025,
    lrf=0.0001,
    weight_decay=0.0001,
    dropout=0.3,
    dfl=0.75,
    cos_lr=True,               
    save_period=5,             
    workers=8,
    freeze=3,
    mosaic=1.0,           
    close_mosaic=20,      
    mixup=0.3,            
    copy_paste=0.2,       
    degrees=10,           
    perspective=0.002,    
    scale=0.5,            
    translate=0.2,        
    shear=0.02,           
    hsv_h=0.02,          
    hsv_s=0.3,
    hsv_v=0.3,
    flipud=0.1,          
    fliplr=0.1,          
    agnostic_nms=True,
    project="Duality-YOLO-Local",
    name="run-1",
)


import uuid
import shutil
from pathlib import Path

def merge_nested(sources: list[Path], dst: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (dst / split / kind).mkdir(parents=True, exist_ok=True)

    for src_root in sources:
        for split in ("train", "val"):
            img_dir = src_root / split / "images"
            lbl_dir = src_root / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                new_base = uuid.uuid4().hex
                new_img  = dst / split / "images" / f"{new_base}{img_path.suffix.lower()}"
                new_lbl  = dst / split / "labels" / f"{new_base}.txt"

                shutil.copy2(img_path, new_img)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, new_lbl)
                else:
                    print(f"[WARN] no label for {img_path}")

if __name__ == "__main__":
    srcs = [
       
    Path(r""),

    Path(r""),
   
    Path(r"")

    ]     #Path to FalconCloud scenarios

    dst = Path(r"")     #Path for new dataset

    merge_nested(srcs, dst)
    print("Merged into", dst.resolve())


from ultralytics import YOLO
from pathlib import Path
import glob
from collections import defaultdict
from datetime import datetime
from PIL import Image

# Path to the model
best_path = Path(r"")
model = YOLO(best_path)

# Path to images
test_imgs = glob.glob(r"")

class_names = ['cheerios', 'soup']
num_classes = len(class_names)

# Thresholds
thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
above_thresholds = [0.8, 0.9, 0.95, 0.97]

# Counters for thresholds
below_counts_global = {thr: 0 for thr in thresholds}
above_counts_global = {thr: 0 for thr in above_thresholds}

below_counts_by_class = {cls: {thr: 0 for thr in thresholds} for cls in range(num_classes)}
above_counts_by_class = {cls: {thr: 0 for thr in above_thresholds} for cls in range(num_classes)}

class_distribution = defaultdict(int)
per_image_summary = {}

# Image counters
empty_images = 0
corrupt_images = 0

# Additional counters for objects and images
images_all_above_95 = 0
images_all_above_97 = 0
objects_above_95 = 0
objects_above_97 = 0

# Counters of objects > thresholds per class
objects_above_95_by_class = {cls: 0 for cls in range(num_classes)}
objects_above_97_by_class = {cls: 0 for cls in range(num_classes)}

# Dictionary to store images with objects conf < 0.6
below_threshold_images = []

# Image processing
for img in test_imgs:
    try:
        results = model.predict([img], imgsz=640, conf=0.25, iou=0.45, verbose=False)
    except Exception:
        corrupt_images += 1
        continue

    boxes = results[0].boxes

    if len(boxes) == 0:
        empty_images += 1
        per_image_summary[Path(img).name] = {}
        continue

    image_class_dist = defaultdict(int)
    all_conf = []

    for box in boxes:
        conf = float(box.conf.item())
        cls = int(box.cls.item())
        all_conf.append(conf)

        for thr in thresholds:
            if conf < thr:
                below_counts_global[thr] += 1
                below_counts_by_class[cls][thr] += 1
                if thr == 0.6:
                    below_threshold_images.append((img, box))  # save for display

        for thr in above_thresholds:
            if conf > thr:
                above_counts_global[thr] += 1
                above_counts_by_class[cls][thr] += 1

        # Class distribution
        class_distribution[cls] += 1
        image_class_dist[cls] += 1

        # Counters of objects >0.95/0.97
        if conf > 0.95:
            objects_above_95 += 1
            objects_above_95_by_class[cls] += 1
        if conf > 0.97:
            objects_above_97 += 1
            objects_above_97_by_class[cls] += 1

    # Counters of images where all objects >0.95/0.97
    if all(conf > 0.95 for conf in all_conf):
        images_all_above_95 += 1
    if all(conf > 0.97 for conf in all_conf):
        images_all_above_97 += 1

    per_image_summary[Path(img).name] = dict(image_class_dist)

# Results file with date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = best_path.parent / f"results_{timestamp}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Counts by thresholds (below)\n")
    for thr, count in below_counts_global.items():
        f.write(f"All classes - Confidence < {thr}: {count}\n")
    for cls in range(num_classes):
        f.write(f"\nClass {class_names[cls]}:\n")
        for thr, count in below_counts_by_class[cls].items():
            f.write(f"  Confidence < {thr}: {count}\n")

    f.write("\nCounts by thresholds (above)\n")
    for thr, count in above_counts_global.items():
        f.write(f"All classes - Confidence > {thr}: {count}\n")
    for cls in range(num_classes):
        f.write(f"\nClass {class_names[cls]}:\n")
        for thr, count in above_counts_by_class[cls].items():
            f.write(f"  Confidence > {thr}: {count}\n")

    f.write("\nOverall class distribution\n")
    for cls, count in class_distribution.items():
        f.write(f"{class_names[cls]}: {count}\n")

    f.write(f"\nNumber of images with no detections: {empty_images}\n")
    f.write(f"Number of corrupted images: {corrupt_images}\n")

    f.write("\nImages and objects above thresholds 0.95 / 0.97\n")
    f.write(f"Images with all objects > 0.95: {images_all_above_95}\n")
    f.write(f"Images with all objects > 0.97: {images_all_above_97}\n")
    f.write(f"Number of objects > 0.95: {objects_above_95}\n")
    f.write(f"Number of objects > 0.97: {objects_above_97}\n")

    f.write("\n=== Number of objects > thresholds per class ===\n")
    for cls in range(num_classes):
        f.write(f"{class_names[cls]} - >0.95: {objects_above_95_by_class[cls]}, >0.97: {objects_above_97_by_class[cls]}\n")

print(f"Results saved to file: {output_file}")


#if you want to display images with object conf less than certain number run this code
'''
from IPython.display import display
from PIL import Image

# Show images with objects conf < 0.6 via display
for img_path, box in below_threshold_images:
    result_copy = model.predict([img_path], imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]
    result_copy.boxes = [box]
    im = result_copy.plot()
    im = Image.fromarray(im)
    display(im)  
'''


from ultralytics import YOLO
from pathlib import Path
import shutil
import random

#Path
orig_train_dir = Path(r"")
orig_val_dir = Path(r"")
save_dir = Path(r"")

train_save_dir = save_dir / "train"
val_save_dir = save_dir / "val"

train_save_dir.mkdir(parents=True, exist_ok=True)
val_save_dir.mkdir(parents=True, exist_ok=True)

best_path = Path(r"")
model = YOLO(best_path)

for split in ["images", "labels"]:
    orig_split_dir = orig_val_dir / split
    save_split_dir = val_save_dir / split
    save_split_dir.mkdir(exist_ok=True, parents=True)
    for f in orig_split_dir.glob("*"):
        shutil.copy(f, save_split_dir / f.name)

train_images = list((orig_train_dir / "images").glob("*"))
selected_for_val = []  

for img_path in train_images:
    label_path = orig_train_dir / "labels" / f"{img_path.stem}.txt"
    keep_image = False
    passed_097 = False

    if not label_path.exists() or label_path.stat().st_size == 0:
        keep_image = True
    else:
        try:
            results = model.predict([str(img_path)], imgsz=640, conf=0.25, iou=0.45, verbose=False)
        except Exception:
            continue

        boxes = results[0].boxes
        if len(boxes) == 0:
            keep_image = True
        else:
            all_conf = [float(box.conf) for box in boxes]
            if all(conf > 0.65 for conf in all_conf):
                keep_image = True
                passed_0xx = True

    if keep_image:
        (train_save_dir / "images").mkdir(exist_ok=True, parents=True)
        shutil.copy(img_path, train_save_dir / "images" / img_path.name)

        if label_path.exists():
            (train_save_dir / "labels").mkdir(exist_ok=True, parents=True)
            shutil.copy(label_path, train_save_dir / "labels" / label_path.name)

        if passed_0xx:
            selected_for_val.append((img_path, label_path if label_path.exists() else None))

#to increase val run this
'''
if len(selected_for_val) > 200:
    chosen = random.sample(selected_for_val, 200)
else:
    chosen = selected_for_val

for img_path, label_path in chosen:
    (val_save_dir / "images").mkdir(exist_ok=True, parents=True)
    shutil.copy(img_path, val_save_dir / "images" / img_path.name)

    if label_path:
        (val_save_dir / "labels").mkdir(exist_ok=True, parents=True)
        shutil.copy(label_path, val_save_dir / "labels" / label_path.name)

    train_img_copy = train_save_dir / "images" / img_path.name
    if train_img_copy.exists():
        train_img_copy.unlink()

    if label_path:
        train_label_copy = train_save_dir / "labels" / label_path.name
        if train_label_copy.exists():
            train_label_copy.unlink()
'''


from ultralytics import YOLO
from pathlib import Path
import glob
from PIL import Image

best_path = Path(r"")   #Path to your model

model = YOLO(best_path)

test_imgs = glob.glob('testImages/images/*')[:20]   #First 20 photos

for img in test_imgs:
    results = model.predict(img, imgsz=640, conf=0.25, iou=0.45)
    im = Image.fromarray(results[0].plot()[:, :, ::-1])
    display(im)  


model=YOLO("")   #Path to your model

test_images_path = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
output_dir = "/kaggle/working/predictions/labels"

conf=0.001

def predict(test_images_path, output_dir , model, conf):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.training = False
    for img_path in Path(test_images_path).glob("*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
    
        results = model.predict(img_path, conf=conf, augment=True, iou=0.4, max_det=600, verbose=False)  
        
        output_txt = Path(output_dir) / f"{img_path.stem}.txt"
    
        with open(output_txt, "w") as f:
            for result in results:
                img_height, img_width = result.orig_shape
                for box in result.boxes.data:
                    x1, y1, x2, y2, confidence, cls_id = box.tolist()
    
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
    
                    f.write(f"{cls_id} {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"[notice] ✅ Predictions saved: {output_dir}")
predict(test_images_path, output_dir , model, conf)


import pandas as pd
import csv
# Convert predictions to CSV
def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submission.csv", 
    test_images_folder: str = "/kaggle/input/multi-class-object-detection-challenge/testImages/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    preds_path = Path(preds_folder)
    test_images_path = Path(test_images_folder)

    test_images = {p.stem for p in test_images_path.glob("*") if p.suffix.lower() in allowed_extensions}

    predictions = []
    predicted_images = set()

    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        predicted_images.add(image_id)

        with open(txt_file, "r") as f:
            valid_lines = [line.strip() for line in f if len(line.strip().split()) == 6]

        pred_str = " ".join(valid_lines) if valid_lines else "no boxes"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})

    missing_images = test_images - predicted_images
    for image_id in missing_images:
        predictions.append({"image_id": image_id, "prediction_string": "no boxes"})

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(submission_df.shape)
    print(submission_df.head(10))
    print(f"[notice] ✅ Submission saved to {output_csv}")

predictions_to_csv()

