!pip install ultralytics > /dev/null


import os
import cv2
import csv
import random
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO


# Load pre-trained YOLO models
model1_path = '/kaggle/input/2-top-models/pytorch/default/1/habijabii.pt'
model2_path = '/kaggle/input/2-top-models/pytorch/default/1/nadiatriki.pt'

model1 = YOLO(model1_path, verbose=False)
model2 = YOLO(model2_path, verbose=False)

# Load test images
test_images_dir = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]


# Helper function to convert prediction results into a submission string
def format_boxes(results, class_offset=0):
    boxes = results.boxes
    width, height = results.orig_shape[1], results.orig_shape[0]

    if boxes is None or len(boxes) == 0:
        return ""

    parts = []
    for box in boxes:
        cls = int(box.cls.cpu().numpy()) + class_offset
        conf = float(box.conf.cpu().numpy())
        x_center_abs, y_center_abs, w_abs, h_abs = box.xywh[0].cpu().numpy()

        x_center = x_center_abs / width
        y_center = y_center_abs / height
        w = w_abs / width
        h = h_abs / height

        parts.append(f"{cls} {conf:.6f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return " ".join(parts)

output_rows = []

# Inference on 2 models
for img_name in image_files:
    img_path = os.path.join(test_images_dir, img_name)

    results1 = model1.predict(img_path, conf=1e-6, device=0, verbose=False)[0]
    results2 = model2.predict(img_path, conf=1e-6, device=0, verbose=False)[0]

    pred_str1 = format_boxes(results1, class_offset=1)
    pred_str2 = format_boxes(results2, class_offset=0)

    combined_pred_str = (pred_str1 + " " + pred_str2).strip()
    if combined_pred_str == "":
        combined_pred_str = "no boxes"

    image_id = os.path.splitext(img_name)[0]

    output_rows.append({
        "image_id": image_id,
        "prediction_string": combined_pred_str
    })


# Save predictions to CSV for submission
csv_path = "submission.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "prediction_string"])
    writer.writeheader()
    writer.writerows(output_rows)


random_images = random.sample(image_files, 10)

def draw_boxes(image, boxes, color=(0, 255, 0), label_prefix=''):
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())

        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        label = f"{label_prefix}{cls} {conf:.2f}"
        cv2.putText(image, label, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

plt.figure(figsize=(20, 40))

for i, img_name in enumerate(random_images):
    img_path = os.path.join(test_images_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results1 = model1(img_path)[0]
    results2 = model2(img_path)[0]

    img_with_boxes = draw_boxes(img.copy(), results1.boxes, color=(0, 255, 0), label_prefix='M1:')
    img_with_boxes = draw_boxes(img_with_boxes, results2.boxes, color=(255, 0, 0), label_prefix='M2:')

    plt.subplot(5, 2, i+1)
    plt.imshow(img_with_boxes)
    plt.title(img_name)
    plt.axis('off')

plt.tight_layout()
plt.show()




