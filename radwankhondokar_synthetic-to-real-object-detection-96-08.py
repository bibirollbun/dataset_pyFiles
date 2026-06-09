rm -rf /kaggle/working/*


!pip install ultralytics comet_ml


import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from pathlib import Path
import os
import yaml
from PIL import Image
from collections import Counter


# with open("yolo_params.yaml", "w") as f:
#     f.write("""
# # Dataset paths
# train: /kaggle/input/falcon-object-detection/ObjectDetectionDataset/train/images  # Path to training images
# val: /kaggle/input/falcon-object-detection/ObjectDetectionDataset/val/images      # Path to validation images
# test: /kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test    # Path to test images

# # Class information
# nc: 1                     # Number of classes
# names: ['cheerios']       # Class names
# """)


!wget -O ObjectDetectionDataset.zip "https://storage.googleapis.com/duality-public-share/Datasets/ObjectDetectionDataset.zip"


import zipfile

with zipfile.ZipFile("ObjectDetectionDataset.zip", "r") as zip_ref:
    zip_ref.extractall("./")  # Extract to the current directory

print("Dataset extracted successfully! ✅")


data_yaml = """
path: /kaggle/working/ObjectDetectionDataset

train: train/images
val: val/images

nc: 1
names: ['cheerios']
"""

# Dosyayı kaydet
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


model = YOLO("yolov8x.pt")


import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


model.train(
    data="data.yaml",
    epochs=100,  # Increased for better convergence
    imgsz=640,
    patience=20,
    batch=8,  # Smaller batch size for better gradient estimation
    lr0=0.0001,  # Lower initial learning rate
    optimizer="SGD",
    cos_lr=True,  # Early stopping patience
    augment=True,
    # # Augmentation adjustments
    scale=1.0,  # More aggressive scaling for partial visibility
    # translate=0.1,  # Add translation augmentation
    # fliplr=0.4, flipud=0.3,  # Reduced flip probability
    mosaic=0.2,  # Slightly reduced mosaic
    mixup=0.2,  # Reduced mixup
    copy_paste=0.1,  # Add copy-paste augmentation
    erasing=0.3,  # Random erasing for occlusion simulation
    flipud=0.25,
    # Detection parameters
    conf=0.5,  # Lower confidence threshold
    iou=0.5,  # Higher IoU threshold for stricter matching
    overlap_mask=True,
    single_cls=True,  # Explicit single-class mode
    momentum=0.975,
    weight_decay=0.0001,
    # Regularization
    dropout=0.4,  # Add dropout for regularization
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


metrics_path = "/kaggle/working/runs/detect/train/results.csv"
metrics = pd.read_csv(metrics_path)

metrics[['metrics/mAP50(B)', 'metrics/mAP50-95(B)']].plot(figsize=(10, 6))
plt.title("mAP Value")
plt.xlabel("Epoch")
plt.ylabel("mAP Value")
plt.grid()
plt.show()


metrics[['train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss']].plot(figsize=(10, 6))
plt.title("Loss Value")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()


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

print(f"[notice] ✅ Tahminler kaydedildi: {output_dir}")


import os

file_path = "/kaggle/working/submission.csv"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"[notice] ✅ File removed: {file_path}")
else:
    print(f"[warn] ⚠️ File not found: {file_path}")


def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submission.csv", 
    test_images_folder: str = "/kaggle/working/data_augmented/test/images",
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

    print(f"[notice] ✅ Submission saved to {output_csv}")


predictions_to_csv()







