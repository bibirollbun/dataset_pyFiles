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

train:
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/clutter/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/couch_far_10/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/far_10_half_clutter/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/film_grain_10_half_clutter/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/large_plant_10/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/no_clutter_10/train/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/table_close_10/train/images
val: 
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/clutter/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/couch_far_10/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/far_10_half_clutter/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/film_grain_10_half_clutter/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/large_plant_10/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/no_clutter_10/val/images
  - /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/table_close_10/val/images
  
test: /kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images

nc: 1
names: ['soup can']
"""

# DosyayÄ± kaydet
with open('data.yaml', 'w') as file:
    file.write(data_yaml)


from ultralytics import YOLO

model = YOLO("yolo11s.yaml").load("yolo11s.pt")


results = model.train(data="data.yaml", 
                      epochs=300,
                      batch=16,
                      optimizer='SGD',
                      momentum=0.937,
                      lr0=0.0005,
                      lrf=0.0005,
                      weight_decay=0.0001,
                      dropout=0.4,
                      hsv_h=0.0,
                      degrees=45,
                      mosaic=0.4,
                      erasing=0.3,
                      cos_lr=True)


from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2  # Added for BGR-RGB conversion

# 1. Load the trained YOLO model
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

# 2. Set test image directory and parameters
test_data_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
conf_threshold = 0.8
start_idx = 0
end_idx = 1
display_images = True

# 3. Create output directory
os.makedirs("output_predictions", exist_ok=True)

# 4. Get list of test images
image_files = sorted([
    f for f in os.listdir(test_data_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
if end_idx is None or end_idx > len(image_files):
    end_idx = len(image_files)

# 5. Run predictions and save/display results
for idx in range(start_idx, end_idx):
    image_path = os.path.join(test_data_path, image_files[idx])
    
    # Predict with YOLO
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Get the plotted image (BGR format)
    plotted_image_bgr = results[0].plot()
    
    # Convert BGR â†’ RGB
    plotted_image_rgb = cv2.cvtColor(plotted_image_bgr, cv2.COLOR_BGR2RGB)  # Fix colors
    
    # Save prediction
    pil_image = Image.fromarray(plotted_image_rgb)
    output_path = os.path.join("output_predictions", f"pred_{image_files[idx]}")
    pil_image.save(output_path)
    print(f"âœ… Saved: {output_path}")
    
    # Display with correct colors
    if display_images:
        plt.figure(figsize=(8, 6))
        plt.imshow(pil_image)  # Now shows correct RGB colors
        plt.axis('off')
        plt.title(f"Prediction: {image_files[idx]}", fontsize=12)
        plt.tight_layout()
        plt.show()


from ultralytics import YOLO
import os
from pathlib import Path

# Load your trained YOLO model
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

# Define test images and output directory
test_images_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
output_dir = "/kaggle/working/predictions/labels"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Confidence threshold
conf_threshold = 0.1

# Process all images in the test folder
for img_path in Path(test_images_path).glob("*"):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue  # Skip non-image files

    # Run prediction
    results = model.predict(img_path, conf=conf_threshold, verbose=False)

    # Output label file (YOLO format)
    output_txt = Path(output_dir) / f"{img_path.stem}.txt"

    with open(output_txt, "w") as f:
        for result in results:
            img_height, img_width = result.orig_shape
            boxes = result.boxes.data  # (N, 6) -> [x1, y1, x2, y2, conf, class]

            if boxes is None or len(boxes) == 0:
                continue  # No predictions

            # Sort boxes by confidence (descending) and take top 2
            sorted_boxes = sorted(boxes.tolist(), key=lambda x: x[4], reverse=True)[:2]

            for box in sorted_boxes:
                x1, y1, x2, y2, conf, cls_id = box

                # Skip boxes below threshold (defensive check)
                if conf < conf_threshold:
                    continue

                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Save to file: class_id confidence x_center y_center width height
                f.write(f"{int(cls_id)} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"[âœ…] All predictions saved in: {output_dir}")


import pandas as pd
import csv

def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submission.csv", 
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

    print(f"[notice] âœ… Submission saved to {output_csv}")
    

predictions_to_csv()

