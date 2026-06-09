!pip install ultralytics
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import csv
import os


data_yaml = """
path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2

train: train/images
val: val/images
test: testImages/images

nc: 1
names: ['object']
"""

# Dosyayı kaydet
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


from ultralytics import YOLO

model = YOLO("yolo11x.yaml")  
model = YOLO("yolo11x.pt") 
model = YOLO("yolo11x.yaml").load("yolo11x.pt")

IMG_SIZE = 640
# Train the model
results = model.train(data="data.yaml", 
                      epochs=100,
                      imgsz=IMG_SIZE, 
                      patience=20,
                      cos_lr=True,
                      dropout=0.4, 
                      mosaic=0.2, 
                      lr0=0.0001, 
                      optimizer="SGD", 
                      momentum=0.975,
                      weight_decay=0.0001,
                      single_cls=True, 
                      plots=True,
                      cache=True,
                      flipud=0.25,
                      scale=1.0
                     )


from ultralytics import YOLO
import os
from pathlib import Path

model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

test_images_path = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images"
output_dir = "/kaggle/working/predictions/labels"

os.makedirs(output_dir, exist_ok=True)

for img_path in Path(test_images_path).glob("*"):
    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue

    results = model.predict(img_path, conf=0.05, verbose=False)  # Predict with confidence threshold

    output_txt = Path(output_dir) / f"{img_path.stem}.txt"

    with open(output_txt, "w") as f:
        for result in results:
            img_height, img_width = result.orig_shape
            boxes = result.boxes.data

            if boxes is None or len(boxes) == 0:
                continue  # No predictions

            # Filter boxes with conf >= 0.05
            filtered_boxes = boxes[boxes[:, 4] >= 0.05]

            if len(filtered_boxes) == 0:
                continue  # No box passes threshold

            for box in filtered_boxes:
                x1, y1, x2, y2, confidence, cls_id = box.tolist()

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Save all detections above threshold in YOLO format
                f.write(f"0 {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"[notice] ✅ All detections saved to: {output_dir}")



def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submissionv1.csv", 
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

    print(f"[notice] ✅ Submission saved to {output_csv}")


predictions_to_csv()




