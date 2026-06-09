! pip install ultralytics
! pip install -U albumentations


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from pathlib import Path


# data.yaml content
data_yaml = """
path: /kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data

train: train/images
val: val/images
test: test/images

nc: 1
names: ['cheerios']
"""

# write the content into a file
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


# using yolo v8 large
model = YOLO("yolov8s.pt")


# training configurations
model.train(
    data="data.yaml",
    seed=42,
    epochs=1, # For saving the notebooks             
    batch=64,
    imgsz=640,
    patience=20,
    optimizer='Adam',
    lr0=0.0003,
    weight_decay=0.0001,
    cos_lr=True,
    # # Augmentations config
    hsv_h=0.02, # control the hue
    hsv_s=0.7, # control the saturation
    hsv_v=0.4, # control the brightness
    flipud=0.5,
    fliplr=0.5,
    translate=0.35,
    scale=0.5,
    erasing=0.35,# add erasing augmentations
    shear=0.02, # up the shear
    # Miscellaneous
    single_cls=True,
)

# mAP 0.995 mAP95 0.994


metrics_path = "/kaggle/working/runs/detect/train/results.csv"
metrics = pd.read_csv(metrics_path)

# Visualize the evaluation metric at each training step
metrics[['metrics/mAP50(B)', 'metrics/mAP50-95(B)']].plot(figsize=(10, 6))
plt.title("mAP Value")
plt.xlabel("Epoch")
plt.ylabel("mAP Value")
plt.grid()
plt.show()

# Visualize the training and validation loss at each training step
metrics[['train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss']].plot(figsize=(10, 6))
plt.title("Loss Value")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()


model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

test_images_path = "/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images"
output_dir = "/kaggle/working/predictions/labels"

os.makedirs(output_dir, exist_ok=True)

for img_path in Path(test_images_path).glob("*"):
    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue

    results = model.predict(img_path, conf=0.05)  

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

                f.write(f"0 {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"✅ Predictions saved to: {output_dir}")


def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submissioni.csv", 
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

    print(f"✅ Submission saved to {output_csv}")


predictions_to_csv()

