from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import csv
from pathlib import Path

# 1. Load both trained YOLO models
model1 = YOLO("model1")
model2 = YOLO("model2")

# 2. Set test image directory and parameters
test_data_path = "testimages/path"
conf_threshold = 0.05
start_idx = 0
end_idx = 100
display_images = True #display images or not

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

def get_all_predictions(results):
    """Get all predictions from YOLO results (no limit)"""
    if len(results[0].boxes) == 0:
        return []
    
    # Get boxes, confidences, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names
    
    predictions = []
    for idx in range(len(boxes)):
        predictions.append({
            'box': boxes[idx],
            'confidence': confidences[idx],
            'class_id': class_ids[idx],
            'class_name': class_names[class_ids[idx]]
        })
    
    return predictions

def find_common_predictions(pred1, pred2, iou_threshold=0.5):
    """Find common predictions between two models based on IoU and class"""
    common_pairs = []
    used_pred2_indices = set()
    
    for i, p1 in enumerate(pred1):
        for j, p2 in enumerate(pred2):
            if j in used_pred2_indices:
                continue
                
            # Check if same class
            if p1['class_id'] == p2['class_id']:
                # Calculate IoU
                iou = calculate_iou(p1['box'], p2['box'])
                if iou > iou_threshold:
                    common_pairs.append((i, j, p1, p2))
                    used_pred2_indices.add(j)
                    break
    
    return common_pairs

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

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

# 5. Run predictions and compare results
for idx in range(start_idx, end_idx):
    image_path = os.path.join(test_data_path, image_files[idx])
    
    print(f"\nğŸ“¸ Processing: {image_files[idx]}")
    
    # Get predictions from both models
    results1 = model1.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)
    results2 = model2.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)
    
    # Get image dimensions
    img_height, img_width = results1[0].orig_shape
    
    # Get ALL predictions from each model (no limit)
    pred1 = get_all_predictions(results1)
    pred2 = get_all_predictions(results2)
    
    print(f"Model 1 predictions: {len(pred1)}")
    print(f"Model 2 predictions: {len(pred2)}")
    
    # Find common predictions
    common_pairs = find_common_predictions(pred1, pred2)
    
    # Get common predictions (use higher confidence from both models)
    common_predictions = []
    for pair in common_pairs:
        pred1_common = pair[2]
        pred2_common = pair[3]
        
        # Use prediction with higher confidence
        if pred1_common['confidence'] >= pred2_common['confidence']:
            common_predictions.append(pred1_common)
        else:
            common_predictions.append(pred2_common)
    
    # Get unique predictions (not in common)
    used_pred1_indices = {pair[0] for pair in common_pairs}
    used_pred2_indices = {pair[1] for pair in common_pairs}
    
    unique_pred1 = [pred1[i] for i in range(len(pred1)) if i not in used_pred1_indices]
    unique_pred2 = [pred2[i] for i in range(len(pred2)) if i not in used_pred2_indices]
    
    # Union ensemble: combine ALL predictions (common + unique from both models)
    union_predictions = common_predictions + unique_pred1 + unique_pred2
    
    # Sort by confidence for better visualization
    union_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Common predictions: {len(common_predictions)}")
    print(f"Unique to Model 1: {len(unique_pred1)}")
    print(f"Unique to Model 2: {len(unique_pred2)}")
    print(f"Union ensemble predictions: {len(union_predictions)}")
    
    # Convert to YOLO format and save
    base_name = os.path.splitext(image_files[idx])[0]
    output_txt = os.path.join("predictions/labels", f"{base_name}.txt")
    
    yolo_lines = convert_to_yolo_format(union_predictions, img_width, img_height)
    
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
        
        # Draw common predictions in RED
        for pred in common_predictions:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red for common
            label = f"COMMON: {pred['class_name']} {pred['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(combined_image_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (255, 0, 0), -1)
            cv2.putText(combined_image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw unique predictions from Model 1 in GREEN
        for pred in unique_pred1:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for Model 1 unique
            label = f"M1: {pred['class_name']} {pred['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(combined_image_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(combined_image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw unique predictions from Model 2 in BLUE
        for pred in unique_pred2:
            x1, y1, x2, y2 = pred['box'].astype(int)
            cv2.rectangle(combined_image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue for Model 2 unique
            label = f"M2: {pred['class_name']} {pred['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(combined_image_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(combined_image_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display single combined image
        plt.figure(figsize=(15, 10))
        plt.imshow(combined_image_rgb)
        plt.axis('off')
        
        # Create title with counts
        title = f"Union Ensemble: {image_files[idx]}\n"
        title += f"ğŸ”´ Common: {len(common_predictions)} | ğŸŸ¢ Model1 Only: {len(unique_pred1)} | ğŸ”µ Model2 Only: {len(unique_pred2)}"
        title += f"\nğŸ“„ Total Union Predictions: {len(union_predictions)}"
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    
    # Save single combined result
    combined_save_image = cv2.imread(image_path)
    combined_save_image_rgb = cv2.cvtColor(combined_save_image, cv2.COLOR_BGR2RGB)
    
    # Draw all predictions on save image
    for pred in common_predictions:
        x1, y1, x2, y2 = pred['box'].astype(int)
        cv2.rectangle(combined_save_image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red for common
        label = f"COMMON: {pred['class_name']} {pred['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(combined_save_image_rgb, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (255, 0, 0), -1)
        cv2.putText(combined_save_image_rgb, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for pred in unique_pred1:
        x1, y1, x2, y2 = pred['box'].astype(int)
        cv2.rectangle(combined_save_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for Model 1
        label = f"M1: {pred['class_name']} {pred['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(combined_save_image_rgb, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(combined_save_image_rgb, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for pred in unique_pred2:
        x1, y1, x2, y2 = pred['box'].astype(int)
        cv2.rectangle(combined_save_image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue for Model 2
        label = f"M2: {pred['class_name']} {pred['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(combined_save_image_rgb, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 0, 255), -1)
        cv2.putText(combined_save_image_rgb, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save combined image
    combined_pil = Image.fromarray(combined_save_image_rgb)
    combined_path = os.path.join("output_predictions", f"union_ensemble_{base_name}.jpg")
    combined_pil.save(combined_path)
    print(f"âœ… Saved union ensemble: {combined_path}")

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
    output_csv="union_ensemble_submission.csv",
    test_images_folder=test_data_path
)

print("\nğŸ�‰ Union ensemble submission completed!")
print(f"ğŸ“� Label files: predictions/labels/")
print(f"ğŸ“„ Submission CSV: union_ensemble_submission.csv")
print(f"ğŸ–¼ï¸� Visualization images: output_predictions/")


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
model1 = YOLO("model1")
model2 = YOLO("model2")

# 2. Set test image directory and parameters
test_data_path = "testimages/path"
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
    
    print(f"\nğŸ“¸ Processing: {image_files[idx]}")
    
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
        plt.figure(figsize=(15, 10))
        plt.imshow(combined_image_rgb)
        plt.axis('off')
        
        # Create title with counts
        title = f"WBF Ensemble: {image_files[idx]}\n"
        title += f"ğŸŸ¢ Model1: {len(pred1)} | ğŸ”µ Model2: {len(pred2)} | ğŸ”´ WBF Fused: {len(wbf_predictions)}"
        title += f"\nWBF Parameters: IoU={iou_thr}, conf_type={conf_type}"
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    
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
    print(f"âœ… Saved WBF ensemble: {combined_path}")

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
    output_csv="wbf_ensemble_submission.csv",
    test_images_folder=test_data_path
)

print("\nğŸ�‰ WBF ensemble submission completed!")
print(f"ğŸ“� Label files: predictions/labels/")
print(f"ğŸ“„ Submission CSV: wbf_ensemble_submission.csv")
print(f"ğŸ–¼ï¸� Visualization images: output_predictions/")
print(f"âš™ï¸� WBF Parameters used: IoU threshold={iou_thr}, conf_type={conf_type}")


import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

# 1. Define your paths
model_path = "/kaggle/input/igyh/pytorch/default/1/last (5).pt"
image_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/IMG_9617.jpg"
output_path = "/kaggle/working/result.jpg"

# 2. Load model with reduced computation
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.35,  # Slightly higher to reduce false positives
    device="cpu",  # Force CPU usage
    load_at_init=True
)

# 3. Optimized SAHI prediction with fewer slices
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=480,  # Larger slices = fewer total slices
    slice_width=480,
    overlap_height_ratio=0.15,  # Reduced overlap
    overlap_width_ratio=0.15,
    perform_standard_pred=True,  # First try standard prediction
    postprocess_type="NMS",
    postprocess_match_threshold=0.4  # Slightly more aggressive NMS
)

# 4. Check if we got enough detections (50-60)
if len(result.object_prediction_list) < 50:
    # Fallback to slightly more slices if needed
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=400,
        slice_width=400,
        overlap_height_ratio=0.2,
        perform_standard_pred=False
    )

# 5. Filter to get top 50-60 most confident detections
final_predictions = sorted(
    result.object_prediction_list,
    key=lambda x: x.score.value,
    reverse=True
)[:60]  # Take top 60

# 6. Visualize results
visualization_result = visualize_object_predictions(
    cv2.imread(image_path),
    final_predictions,
    output_dir="/kaggle/working/",
    file_name="optimized_result",
    export_format="jpg"
)

print(f"Total detections: {len(final_predictions)}")
print(f"Visualization saved to: {output_path}")


# Install the required packages
!pip install albumentations ultralytics
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

