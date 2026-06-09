%%capture
!pip install ultralytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import csv
import os


!wget -O ObjectDetectionDataset.zip "https://storage.googleapis.com/duality-public-share/Datasets/ObjectDetectionDataset.zip"



import zipfile

with zipfile.ZipFile("ObjectDetectionDataset.zip", "r") as zip_ref:
    zip_ref.extractall("./")  # Extract to the current directory

print("Dataset extracted successfully! âœ…")


data_yaml = """
path: /kaggle/working/ObjectDetectionDataset

train: train/images
val: val/images

nc: 1
names: ['cheerios']
"""

# DosyayÄ± kaydet
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


from ultralytics import YOLO

# Load YOLOv8s model
model = YOLO("yolov8s.pt")

# Optimized training configuration
model.train(
    data="data.yaml",
    epochs=100,  # Increased for better convergence
    imgsz=960,
    batch=8,  # Smaller batch size for better gradient estimation
    lr0=0.0001,  # Lower initial learning rate
    optimizer="AdamW",
    weight_decay=0.0005,  # Regularization to prevent overfitting
    cos_lr=True,  # Early stopping patience
    augment=True,
    # Augmentation adjustments
    scale=0.5,  # More aggressive scaling for partial visibility
    translate=0.1,  # Add translation augmentation
    fliplr=0.4, flipud=0.3,  # Reduced flip probability
    mosaic=0.8,  # Slightly reduced mosaic
    mixup=0.2,  # Reduced mixup
    copy_paste=0.1,  # Add copy-paste augmentation
    erasing=0.3,  # Random erasing for occlusion simulation
    # Detection parameters
    conf=0.5,  # Lower confidence threshold
    iou=0.5,  # Higher IoU threshold for stricter matching
    overlap_mask=True,
    single_cls=True,  # Explicit single-class mode
    # Regularization
    dropout=0.2,  # Add dropout for regularization
    # Optimization
    nbs=64,  # Nominal batch size
    # Data handling
    cache="disk",  # More stable caching
    workers=8,  # Optimal for most systems
    # Model saving
    save=True,
    save_period=25,  # More frequent checkpoints
    deterministic=True,  # Reproducibility
)


from ultralytics import YOLO
import os
from pathlib import Path
import torch

model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

test_images_path = "/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images"
output_dir = "/kaggle/working/predictions/labels"

os.makedirs(output_dir, exist_ok=True)

for img_path in Path(test_images_path).glob("*"):
    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue

    results = model.predict(img_path, conf=0.05)  # Set min conf here

    output_txt = Path(output_dir) / f"{img_path.stem}.txt"

    with open(output_txt, "w") as f:
        for result in results:
            img_height, img_width = result.orig_shape

            boxes = result.boxes.data

            if len(boxes) == 0:
                continue  # No predictions

            # Filter boxes with conf >= 0.05
            filtered_boxes = boxes[boxes[:, 4] >= 0.05]

            if len(filtered_boxes) == 0:
                continue  # No box passes threshold

            # Get the box with the highest confidence
            best_box = filtered_boxes[filtered_boxes[:, 4].argmax()]

            x1, y1, x2, y2, confidence, cls_id = best_box.tolist()

            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Save in YOLO format: class_id confidence x_center y_center width height
            f.write(f"0 {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"[notice] âœ… Tahminler kaydedildi: {output_dir}")


def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submissionv.csv", 
    test_images_folder: str = "/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images",
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

    print(f"[notice] âœ… Submission saved to {output_csv}")


predictions_to_csv()





