!pip install ultralytics


from ultralytics import YOLO
import torch

# Load models trained on different folds
models = [
#   YOLO("/kaggle/input/soup-can-yolov8n-0-1-2/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8n-0-1-2/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8n-0-1-2/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8n-3-4/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8n-3-4/yolo_kfold_results/fold_4/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8m-0/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8m-1/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8m-2/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8m-3-4/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8m-3-4/yolo_kfold_results/fold_4/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolo11m-0-1/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolo11m-0-1/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8scv/best_0.pt"),
#   YOLO("/kaggle/input/yolov8scv/best_1.pt"),
#   YOLO("/kaggle/input/yolov8scv/best_2.pt"),
#   YOLO("/kaggle/input/yolov8scv/best_3.pt"),
#   YOLO("/kaggle/input/yolov8scv/best_4.pt")
#   YOLO("/kaggle/input/yolov8s-home-16-960/yolo_kfold_results/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-home-16-960/yolo_kfold_results/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-home-16-960/yolo_kfold_results/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-home-16-960/yolo_kfold_results/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-home-16-960/yolo_kfold_results/yolo_kfold_results/fold_4/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-16-960-v3/yolo_kfold_results/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-16-960-v3/yolo_kfold_results/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-16-960-v3/yolo_kfold_results/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-16-960-v3/yolo_kfold_results/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-16-960-v3/yolo_kfold_results/yolo_kfold_results/fold_4/weights/best.pt"),
    YOLO("/kaggle/input/yolov8s-16-960-ckp/weights_0/best_custom.pt"),
    YOLO("/kaggle/input/yolov8s-16-960-ckp/weights_1/best_custom.pt"),
    YOLO("/kaggle/input/yolov8s-16-960-ckp/weights_2/best_custom.pt"),
    YOLO("/kaggle/input/yolov8s-16-960-ckp/weights_3/best_custom.pt"),
    YOLO("/kaggle/input/yolov8s-16-960-ckp/weights_4/best_custom.pt"),
#   YOLO("/kaggle/input/can-yolov12n-16-960-0/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/can-yolov12n-16-960-0/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8s-16-960-0-1/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8s-16-960-0-1/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8s-16-960-2-3/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8s-16-960-2-3/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/soup-can-yolov8s-16-960-4/yolo_kfold_results/fold_4/weights/best.pt")
#   YOLO("/kaggle/input/yolov8s-8-1280/yolo_kfold_results/yolo_kfold_results/fold_0/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-8-1280/yolo_kfold_results/yolo_kfold_results/fold_1/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-8-1280/yolo_kfold_results/yolo_kfold_results/fold_2/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-8-1280/yolo_kfold_results/yolo_kfold_results/fold_3/weights/best.pt"),
#   YOLO("/kaggle/input/yolov8s-8-1280/yolo_kfold_results/yolo_kfold_results/fold_4/weights/best.pt")
]


from pathlib import Path
import pandas as pd
import csv
import sys

def predictions_to_csv(
    preds_folder: str = "./predictions/labels", 
    output_csv: str = "submission.csv", 
    test_images_folder: str = "./TestImages/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    """
    Convert YOLO prediction files to Kaggle submission CSV format
    with strict validation.
    """
    # Validate inno boxputs
    preds_path = Path(preds_folder)
    if not preds_path.exists():
        print(f"ERROR: Prediction folder '{preds_folder}' does not exist")
        sys.exit(1)

    # Get test image IDs (without extensions)
    test_images_path = Path(test_images_folder)
    if not test_images_path.exists():
        print(f"ERROR: Test images folder '{test_images_folder}' not found")
        sys.exit(1)
        
    test_images = {
        p.stem: True 
        for p in test_images_path.glob("*") 
        if p.suffix.lower() in allowed_extensions
    }
    print(f"Found {len(test_images)} test images")

    # Collect predictions with validation
    predictions = []
    error_count = 0
    
    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        
        # Validate image_id
        if image_id not in test_images:
            print(f"Skipping non-test image prediction: {txt_file.name}")
            continue
            
        with open(txt_file, "r") as f:
            valid_lines = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                            
                parts = line.split()
                if len(parts) != 6:
                    print(f"Invalid prediction in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue
                            
                try:
                    [float(x) for x in parts]
                    valid_lines.append(line)
                except ValueError:
                    print(f"Non-numeric values in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue

        flat_pred = " ".join(" ".join(line.split()) for line in valid_lines)
        pred_str = flat_pred if flat_pred else "no box"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})


    # Create submission dataframe
    submission_df = pd.DataFrame({"image_id": list(test_images.keys())})
    
    if predictions:
        preds_df = pd.DataFrame(predictions)
        final_df = submission_df.merge(preds_df, on="image_id", how="left").fillna("no boxes")
    else:
        final_df = submission_df
        final_df["prediction_string"] = "no boxes"

    # Save with CSV quoting rules
    final_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"\n Success! Submission saved to {output_csv}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Validation errors: {error_count}")


!mkdir predictions
!mkdir predictions/images
!mkdir predictions/labels


from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
from tqdm import tqdm

def predict_and_save(model, image_path, output_path, output_path_txt):
    img = cv2.imread(str(image_path))
    # Perform prediction
    results = model.predict(img,conf=0.5,augment=True,verbose=False)

    result = results[0]
    # Draw boxes on the image
    # img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    # cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open('predictions/labels/'+output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box.conf[0])  # confidence is a tensor with 1 value
            f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def predict_and_save_90(model, image_path, output_path, output_path_txt):
    img = cv2.imread(str(image_path))
    rimg = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # Perform prediction
    results = model.predict(rimg,conf=0.5,augment=True,verbose=False)

    result = results[0]
    # Draw boxes on the image
    # img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    # cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open('predictions/labels/'+output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_rot, y_rot, height, width = box.xywhn[0].tolist()
            x_center = y_rot
            y_center = 1 - x_rot
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box.conf[0])  # confidence is a tensor with 1 value
            f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def predict_and_save_180(model, image_path, output_path, output_path_txt):
    img = cv2.imread(str(image_path))
    rimg = cv2.rotate(img, cv2.ROTATE_180)
    # Perform prediction
    results = model.predict(rimg,conf=0.5,augment=True,verbose=False)

    result = results[0]
    # Draw boxes on the image
    # img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    # cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open('predictions/labels/'+output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_rot, y_rot, width, height = box.xywhn[0].tolist()
            x_center = 1 - x_rot
            y_center = 1 - y_rot
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box.conf[0])  # confidence is a tensor with 1 value
            f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def predict_and_save_270(model, image_path, output_path, output_path_txt):
    img = cv2.imread(str(image_path))
    rimg = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Perform prediction
    results = model.predict(rimg,conf=0.5,augment=True,verbose=False)

    result = results[0]
    # Draw boxes on the image
    # img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    # cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open('predictions/labels/'+output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_rot, y_rot, height, width = box.xywhn[0].tolist()
            x_center = 1 - y_rot
            y_center = x_rot
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box.conf[0])  # confidence is a tensor with 1 value
            f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == '__main__':

    for fold in range(len(models)):
        for img_path in tqdm(os.listdir("/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images")):
            predict_and_save(
                models[fold],
                "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/"+img_path,
                img_path,
                img_path.split('.')[0]+'.txt'
            )

        predictions_to_csv(
            preds_folder='predictions/labels',
            output_csv=f'submission_{fold}_0.csv',
            test_images_folder="/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
        )

        for img_path in tqdm(os.listdir("/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images")):
            predict_and_save_90(
                models[fold],
                "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/"+img_path,
                img_path,
                img_path.split('.')[0]+'.txt'
            )

        predictions_to_csv(
            preds_folder='predictions/labels',
            output_csv=f'submission_{fold}_1.csv',
            test_images_folder="/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
        )

        for img_path in tqdm(os.listdir("/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images")):
            predict_and_save_180(
                models[fold],
                "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/"+img_path,
                img_path,
                img_path.split('.')[0]+'.txt'
            )

        predictions_to_csv(
            preds_folder='predictions/labels',
            output_csv=f'submission_{fold}_2.csv',
            test_images_folder="/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
        )

        for img_path in tqdm(os.listdir("/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images")):
            predict_and_save_270(
                models[fold],
                "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/"+img_path,
                img_path,
                img_path.split('.')[0]+'.txt'
            )

        predictions_to_csv(
            preds_folder='predictions/labels',
            output_csv=f'submission_{fold}_3.csv',
            test_images_folder="/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
        )



import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def apply_nms(boxes, iou_threshold=0.5):
    boxes.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence
    selected = []
    while boxes:
        current = boxes.pop(0)
        selected.append(current)
        boxes = [box for box in boxes 
                 if compute_iou(current[:4], box[:4]) <= iou_threshold]
    return selected

# Read all fold CSVs
fold_files = []
for i in range(len(models)):
    for j in range(4):
        fold_files.append(f'submission_{i}_{j}.csv')
        
all_image_ids = set()
aggregated_boxes = defaultdict(list)

for file in fold_files:
    df = pd.read_csv(file)
    all_image_ids.update(df['image_id'])
    for _, row in df.iterrows():
        if row['prediction_string'] == "no box":
            continue
        tokens = row['prediction_string'].split()
        num_boxes = len(tokens) // 6
        for i in range(num_boxes):
            idx = i * 6
            class_id = int(tokens[idx])
            conf = float(tokens[idx+1])
            cx, cy, w, h = map(float, tokens[idx+2:idx+6])
            aggregated_boxes[row['image_id']].append([class_id, conf, cx, cy, w, h])

# Process each image
results = []
iou_threshold = 0.5

for image_id in tqdm(all_image_ids):
    if image_id not in aggregated_boxes:
        results.append({'image_id': image_id, 'prediction_string': "no box"})
        continue
        
    boxes = aggregated_boxes[image_id]
    converted_boxes = []
    for box in boxes:
        class_id, conf, cx, cy, w, h = box
        x1 = max(0, min(1, cx - w/2))
        y1 = max(0, min(1, cy - h/2))
        x2 = max(0, min(1, cx + w/2))
        y2 = max(0, min(1, cy + h/2))
        converted_boxes.append([x1, y1, x2, y2, conf, class_id])
    
    # Apply NMS per class
    classes = set(box[5] for box in converted_boxes)
    selected_boxes = []
    for cls in classes:
        class_boxes = [box for box in converted_boxes if box[5] == cls]
        selected_boxes.extend(apply_nms(class_boxes, iou_threshold))
    
    # Convert back to YOLO format
    output_tokens = []
    for box in selected_boxes:
        x1, y1, x2, y2, conf, class_id = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        output_tokens.extend([
            str(class_id), 
            f"{conf:.6f}", 
            f"{cx:.6f}", f"{cy:.6f}", 
            f"{w:.6f}", f"{h:.6f}"
        ])
    
    pred_string = " ".join(output_tokens) if output_tokens else "no box"
    results.append({'image_id': image_id, 'prediction_string': pred_string})

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv('submission.csv', index=False)

