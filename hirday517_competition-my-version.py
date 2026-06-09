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
model1 = YOLO(model_path_a, verbose=True)
model2 = YOLO(model_path_b, verbose=False)

model1.to("cuda")
model2.to("cuda")
# Set the directory for test images
test_image_folder = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
test_image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg', '.png'))]


# # Core YOLOv8 library
# pip install ultralytics

# # Plotting
# pip install matplotlib

# # Image handling
# pip install pillow

# # OpenCV for image processing
# pip install opencv-python

# # NumPy for array ops
# pip install numpy

# # Pandas for dataframes
# pip install pandas

# # CSV is built into Python â†’ no install needed

# # Pathlib is built into Python â†’ no install needed

# # Ensemble methods for object detection
# pip install ensemble-boxes
!pip install ultralytics ensemble-boxes
!pip install opencv-python



from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

# 1. Load both trained YOLO models
# model1 = YOLO("model1")
# model2 = YOLO("model2")

# 2. Set test image directory and parameters
test_data_path = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
conf_threshold = 0.05
start_idx = 0
end_idx = 100
display_images = True #display images or not

# WBF parameters
iou_thr = 0.5  # IoU threshold for WBF
skip_box_thr = 0.0001  # Skip boxes with confidence lower than this
conf_type = 'avg'  # How to calculate confidence in weighted boxes

# 3. Create output directories
os.makedirs("output_predictions", exist_ok=True)
os.makedirs("predictions/labels", exist_ok=True)

# 4. Get list of test images
image_files = sorted([
    f for f in os.listdir(test_data_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if end_idx is None or end_idx > len(image_files):
    end_idx = len(image_files)

def get_predictions_for_wbf(results, img_width, img_height):
    """Get predictions from YOLO results in format suitable for WBF"""
    if len(results[0].boxes) == 0:
        return [], [], []
    
    # Get boxes, confidences, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Convert to normalized coordinates for WBF (x1, y1, x2, y2 in range [0, 1])
    normalized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        norm_box = [x1/img_width, y1/img_height, x2/img_width, y2/img_height]
        normalized_boxes.append(norm_box)
    
    return normalized_boxes, confidences.tolist(), class_ids.tolist()

def wbf_predictions_to_display_format(boxes, scores, labels, class_names, img_width, img_height):
    """Convert WBF output back to display format"""
    predictions = []
    
    for i in range(len(boxes)):
        # Convert normalized coordinates back to pixel coordinates
        x1, y1, x2, y2 = boxes[i]
        pixel_box = [x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height]
        
        predictions.append({
            'box': np.array(pixel_box),
            'confidence': scores[i],
            'class_id': int(labels[i]),
            'class_name': class_names[int(labels[i])]
        })
    
    return predictions

def convert_to_yolo_format(predictions, img_width, img_height):
    """Convert predictions to YOLO format"""
    yolo_lines = []
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['box']
        conf = pred['confidence']
        cls_id = pred['class_id']
        
        # Convert to YOLO format
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_lines.append(f"{int(cls_id)} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines

# Store all predictions for submission
all_predictions = []

# 5. Run predictions and apply WBF ensemble
for idx in range(start_idx, end_idx):
    image_path = os.path.join(test_data_path, image_files[idx])
    
    # print(f"\nğŸ“¸ Processing: {image_files[idx]}")
    
    # Get predictions from both models
    results1 = model1.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)
    results2 = model2.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)
    
    # Get image dimensions
    img_height, img_width = results1[0].orig_shape
    
    # Get class names (assuming both models have same classes)
    class_names = results1[0].names
    
    # Get predictions in WBF format
    boxes1, scores1, labels1 = get_predictions_for_wbf(results1, img_width, img_height)
    boxes2, scores2, labels2 = get_predictions_for_wbf(results2, img_width, img_height)
    
    print(f"Model 1 predictions: {len(boxes1)}")
    print(f"Model 2 predictions: {len(boxes2)}")
    
    # Prepare data for WBF
    boxes_list = [boxes1, boxes2]
    scores_list = [scores1, scores2]
    labels_list = [labels1, labels2]
    weights = [1, 1]  # Equal weights for both models
    
    # Apply Weighted Boxes Fusion
    if len(boxes1) > 0 or len(boxes2) > 0:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, 
            scores_list, 
            labels_list, 
            weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr,
            conf_type=conf_type
        )
        
        # Convert back to display format
        wbf_predictions = wbf_predictions_to_display_format(
            fused_boxes, fused_scores, fused_labels, class_names, img_width, img_height
        )
    else:
        wbf_predictions = []
    
    print(f"WBF fused predictions: {len(wbf_predictions)}")
    
    # Convert to YOLO format and save
    base_name = os.path.splitext(image_files[idx])[0]
    output_txt = os.path.join("predictions/labels", f"{base_name}.txt")
    
    yolo_lines = convert_to_yolo_format(wbf_predictions, img_width, img_height)
    
    with open(output_txt, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")
    
    # Store for submission CSV
    pred_string = " ".join(yolo_lines) if yolo_lines else "no boxes"
    all_predictions.append({
        "image_id": base_name,
        "prediction_string": pred_string
    })
    
    if display_images:
        # Load original image
        combined_image = cv2.imread(image_path)
        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        
        # Get individual model predictions for visualization
        pred1 = wbf_predictions_to_display_format(boxes1, scores1, labels1, class_names, img_width, img_height)
        pred2 = wbf_predictions_to_display_format(boxes2, scores2, labels2, class_names, img_width, img_height)
        
        # Draw Model 1 predictions in GREEN (thin lines)
        for pred in pred1:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green thin
            label = f"M1: {pred['class_name']} {pred['confidence']:.2f}"
            cv2.putText(combined_image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw Model 2 predictions in BLUE (thin lines)
        for pred in pred2:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Blue thin
            label = f"M2: {pred['class_name']} {pred['confidence']:.2f}"
            cv2.putText(combined_image_rgb, label, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw WBF fused predictions in RED (thick lines)
        for pred in wbf_predictions:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red thick
            label = f"WBF: {pred['class_name']} {pred['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(combined_image_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (255, 0, 0), -1)
            cv2.putText(combined_image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display combined image

    
        # plt.figure(figsize=(15, 10))
        # plt.imshow(combined_image_rgb)
        # plt.axis('off')
        
        # Create title with counts

    
        # title = f"WBF Ensemble: {image_files[idx]}\n"
        # title += f"ğŸŸ¢ Model1: {len(pred1)} | ğŸ”µ Model2: {len(pred2)} | ğŸ”´ WBF Fused: {len(wbf_predictions)}"
        # title += f"\nWBF Parameters: IoU={iou_thr}, conf_type={conf_type}"
        # plt.title(title, fontsize=14, pad=20)
        # plt.tight_layout()
        # plt.show()
    
    # Save combined result
    combined_save_image = cv2.imread(image_path)
    combined_save_image_rgb = cv2.cvtColor(combined_save_image, cv2.COLOR_BGR2RGB)
    
    # Draw WBF predictions on save image
    for pred in wbf_predictions:
        x1, y1, x2, y2 = pred['box'].astype(int)
        cv2.rectangle(combined_save_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red
        label = f"WBF: {pred['class_name']} {pred['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(combined_save_image_rgb, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (255, 0, 0), -1)
        cv2.putText(combined_save_image_rgb, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save combined image
    combined_pil = Image.fromarray(combined_save_image_rgb)
    combined_path = os.path.join("output_predictions", f"wbf_ensemble_{base_name}.jpg")
    combined_pil.save(combined_path)
    # print(f"âœ… Saved WBF ensemble: {combined_path}")

print(f"\n[âœ…] All predictions saved in: predictions/labels")

# Create submission CSV
def create_submission_csv(
    predictions_list,
    output_csv: str = "submission.csv",
    test_images_folder: str = None
):
    """Create submission CSV from predictions list"""
    
    # Convert predictions list to DataFrame
    submission_df = pd.DataFrame(predictions_list)
    
    # If test images folder is provided, check for missing images
    if test_images_folder:
        test_images_path = Path(test_images_folder)
        allowed_extensions = (".jpg", ".png", ".jpeg")
        test_images = {p.stem for p in test_images_path.glob("*") if p.suffix.lower() in allowed_extensions}
        
        predicted_images = set(submission_df['image_id'].tolist())
        missing_images = test_images - predicted_images
        
        # Add missing images with "no boxes"
        for image_id in missing_images:
            submission_df = pd.concat([
                submission_df,
                pd.DataFrame([{"image_id": image_id, "prediction_string": "no boxes"}])
            ], ignore_index=True)
    
    # Sort by image_id for consistency
    submission_df = submission_df.sort_values('image_id').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] âœ… Submission saved to {output_csv}")
    print(f"[notice] ğŸ“Š Total submissions: {len(submission_df)}")
    
    return submission_df

# Create submission file
submission_df = create_submission_csv(
    all_predictions,
    output_csv="submission.csv",
    test_images_folder=test_data_path
)

print("\nğŸ�‰ WBF ensemble submission completed!")
print(f"ğŸ“� Label files: predictions/labels/")
print(f"ğŸ“„ Submission CSV: wbf_ensemble_submission.csv")
print(f"ğŸ–¼ï¸� Visualization images: output_predictions/")
print(f"âš™ï¸� WBF Parameters used: IoU threshold={iou_thr}, conf_type={conf_type}")


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


import os

# Prepare submission list
submission_data = []

# âœ… Proper path to image folder
test_image_folder = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'

# âœ… Get list of all image files (jpg/png only)
test_image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg', '.png'))]

# âœ… Loop through each image
for image_filename in test_image_files:
    image_full_path = os.path.join(test_image_folder, image_filename)
    print(f"Processing: {image_full_path}")  # optional debug print

    # Run inference on both models
    results_a = model1.predict(image_full_path, conf=1e-6, device=0, verbose=False)[0]
    results_b = model2.predict(image_full_path, conf=1e-6, device=0, verbose=False)[0]

    # Format predictions from each model
    predictions_model_a = serialize_predictions(results_a, class_id_offset=1)
    predictions_model_b = serialize_predictions(results_b, class_id_offset=0)

    # Combine predictions
    final_prediction_string = (predictions_model_a + " " + predictions_model_b).strip()

    # If no detection at all
    if final_prediction_string == "":
        final_prediction_string = "no boxes"

    # Image ID without .jpg
    image_identifier = os.path.splitext(image_filename)[0]

    # Add to final list
    submission_data.append({
        "image_id": image_identifier,
        "prediction_string": final_prediction_string
    })



output_csv_path = "submission.csv"
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "prediction_string"])
    writer.writeheader()
    writer.writerows(submission_data)





import matplotlib.pyplot as plt
import cv2

for image_filename in test_image_files[:5]:  # visualize first 5 images only
    image_path = os.path.join(test_image_folder, image_filename)

    results = detector_b.predict(image_path, conf=0.25)[0]
    img = results.plot()  # Draw bounding boxes

    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Predictions for {image_filename}")
    plt.axis('off')
    plt.show()


!yolo task=detect mode=val model=your_model.pt data=data.yaml



yaml_code = """
path: /kaggle/input/ai-gen
train: /kaggle/input/ai-gen/output (1)/Output/2025-07-29-17-59-27/train
val: /kaggle/input/ai-gen/output (1)/Output/2025-07-29-17-59-27/val
nc: 2
names:
  0: 0
  1: 1
"""

with open("data.yaml", "w") as f:
    f.write(yaml_code)



# from ultralytics import YOLO

# # Load base model (YOLOv8n = nano, v8s = small)
# model = YOLO("yolov8s.pt")

# # Start training
# model.train(
#     data="data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     device=0  # 0 for GPU, -1 for CPU
# )
from ultralytics import YOLO

# Load your existing trained model
# model = YOLO("/kaggle/input/my-prebuilt-model/best.pt")
model=model1
# detector_a = YOLO(model_path_a, verbose=True)
# detector_b = YOLO(model_path_b, verbose=False)

# Fine-tune on new or bigger dataset
model.train(
    data="data.yaml",   # your dataset config
    epochs=20,          # fewer if you're just tweaking
    imgsz=640,
    batch=16,
    device=0
)


model=model2
# detector_a = YOLO(model_path_a, verbose=True)
# detector_b = YOLO(model_path_b, verbose=False)

# Fine-tune on new or bigger dataset
model.train(
    data="data.yaml",   # your dataset config
    epochs=20,          # fewer if you're just tweaking
    imgsz=640,
    batch=16,
    device=0
)





yaml_code = """
path: /kaggle/input/ai-gen
train: /kaggle/input/ai-gen/output (1)/Output/2025-07-29-17-59-27/train
val: /kaggle/input/ai-gen/output (1)/Output/2025-07-29-17-59-27/val
nc: 2
names:
  0: 0
  1: 1
"""

with open("falcon_data.yaml", "w") as f:
    f.write(yaml_code)



model.train(
    data="falcon_data.yaml",  # yaml for AI-generated dataset
    epochs=10,                # short fine-tuning
    imgsz=640,
    batch=16,
    device=0,                 # GPU
    lr0=1e-4,                  # smaller learning rate for fine-tuning
    pretrained=True,          # keep pretrained backbone
    degrees=15,                # augmentation
    translate=0.2,
    scale=0.5,
    shear=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)



from IPython.display import Image

Image(filename='runs/detect/train/results.png')






# try conf=0.15
!yolo val model=runs/detect/train/weights/best.pt data=data.yaml conf=0.15
# try conf=0.25
!yolo val model=runs/detect/train/weights/best.pt data=data.yaml conf=0.25






from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
for conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    print("=== conf =", conf, "===")
    metrics = model.val(data="data.yaml", conf=conf)  # prints mAP, precision, recall
    # metrics contains values you can inspect programmatically






model.val()



pip install ensemble-boxes



pip install torchvision


