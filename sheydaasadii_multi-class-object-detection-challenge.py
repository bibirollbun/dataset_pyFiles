# Install the ultralytics library for YOLO models
!pip install ultralytics > /dev/null

import os
import cv2
import csv
import random
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Define the file paths for the two object detection models
model_path_a = '/kaggle/input/2-top-models/pytorch/default/1/habijabii.pt'
model_path_b = '/kaggle/input/2-top-models/pytorch/default/1/nadiatriki.pt'

# Load the models using the YOLO class
detector_a = YOLO(model_path_a, verbose=False)
detector_b = YOLO(model_path_b, verbose=False)

# Set the directory for test images
test_image_folder = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
test_image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg', '.png'))]


def serialize_predictions(yolo_results, class_id_offset=0):
    """
    Converts YOLO detection results into a formatted string for submission.

    Args:
        yolo_results: The prediction results from a single image.
        class_id_offset: An integer to add to the class ID, used to handle different
                         class mappings between models.

    Returns:
        A string of space-separated predictions, or an empty string if no boxes are found.
    """
    detected_boxes = yolo_results.boxes
    img_width, img_height = yolo_results.orig_shape[1], yolo_results.orig_shape[0]

    if detected_boxes is None or len(detected_boxes) == 0:
        return ""

    prediction_strings = []
    for box_info in detected_boxes:
        class_id = int(box_info.cls.cpu().numpy()) + class_id_offset
        confidence = float(box_info.conf.cpu().numpy())
        x_center, y_center, box_width, box_height = box_info.xywh[0].cpu().numpy()

        # Normalize coordinates and dimensions to be between 0 and 1
        x_norm = x_center / img_width
        y_norm = y_center / img_height
        w_norm = box_width / img_width
        h_norm = box_height / img_height

        prediction_strings.append(f"{class_id} {confidence:.6f} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return " ".join(prediction_strings)


submission_data = []

for image_filename in test_image_files:
    image_full_path = os.path.join(test_image_folder, image_filename)

    # Run inference on both models for the same image
    results_a = detector_a.predict(image_full_path, conf=1e-6, device=0, verbose=False)[0]
    results_b = detector_b.predict(image_full_path, conf=1e-6, device=0, verbose=False)[0]

    # Format predictions from each model
    predictions_model_a = serialize_predictions(results_a, class_id_offset=1)
    predictions_model_b = serialize_predictions(results_b, class_id_offset=0)

    # Combine the prediction strings from both models (ensembling)
    final_prediction_string = (predictions_model_a + " " + predictions_model_b).strip()

    # Handle the case where no detections were made
    if final_prediction_string == "":
        final_prediction_string = "no boxes"

    image_identifier = os.path.splitext(image_filename)[0]

    # Append the combined prediction to our list
    submission_data.append({
        "image_id": image_identifier,
        "prediction_string": final_prediction_string
    })


output_csv_path = "submission.csv"
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "prediction_string"])
    writer.writeheader()
    writer.writerows(submission_data)

