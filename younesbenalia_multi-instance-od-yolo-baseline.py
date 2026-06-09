!pip install ultralytics


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import csv
import os
import random
import torch
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


model = YOLO("yolo11m.pt")
data_yaml = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml"

model.train(
    data=data_yaml,
    epochs=50,                
    batch=4,                   
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
    close_mosaic=15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    shear=0.01
)


model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

test_images_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
output_dir = "/kaggle/working/predictions/labels"

conf=0.0001

def predict(test_images_path, output_dir , model, conf):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.training = False
    for img_path in Path(test_images_path).glob("*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
    
        results = model.predict(img_path, conf=conf)  
        
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
    
    print(f"[notice] ✅ Predictions saved: {output_dir}")
predict(test_images_path, output_dir , model, conf)


# Convert predictions to CSV
def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submissionv.csv", 
    test_images_folder: str = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images",
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
    print(submission_df.head(10))
    print(f"[notice] ✅ Submission saved to {output_csv}")

predictions_to_csv()




