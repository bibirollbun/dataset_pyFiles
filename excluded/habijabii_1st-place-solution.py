import pandas
import yaml

data = {
    'train': [
        '/kaggle/input/soup-can/CAN_Dataset/cameraDistance/train',
        '/kaggle/input/soup-can/CAN_Dataset/baseData/train',
        '/kaggle/input/soup-can/CAN_Dataset/coolLighting/train',
        '/kaggle/input/soup-can/CAN_Dataset/furniture/train',
        '/kaggle/input/soup-can/CAN_Dataset/plants/train',
        '/kaggle/input/soup-can/CAN_Dataset/genericObjects/train',
        '/kaggle/input/soup-can/CAN_Dataset/misclassifications2/train',
        '/kaggle/input/soup-can/CAN_Dataset/misclassified-objects/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputallcouch/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputallfridge/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputallplant/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputalltable/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputalltv/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv5/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/plants/train',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge1/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge2/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge3/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge4/Output/1/train',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge5/Output/1/train',
        
    ],
    'val': [
        '/kaggle/input/soup-can/CAN_Dataset/cameraDistance/val',
        '/kaggle/input/soup-can/CAN_Dataset/baseData/val',
        '/kaggle/input/soup-can/CAN_Dataset/coolLighting/val',
        '/kaggle/input/soup-can/CAN_Dataset/furniture/val',
        '/kaggle/input/soup-can/CAN_Dataset/plants/val',
        '/kaggle/input/soup-can/CAN_Dataset/genericObjects/val',
        '/kaggle/input/soup-can/CAN_Dataset/misclassifications2/val',
        '/kaggle/input/soup-can/CAN_Dataset/misclassified-objects/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputallcouch/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputallfridge/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputallplant/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputalltable/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputalltv/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcarpet5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputcouch5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputfridge5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputplant5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtable5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/outputtv5/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/plants/val',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge1/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge2/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge3/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge4/Output/1/val',
        '/kaggle/input/soup-can/CAN_Dataset/topfridge5/Output/1/val',
       
    ],
    'test': '/kaggle/input/kan-dataset/Kan/test_dataset',
    'nc': 1,
    'names': ['Soup']
}

with open('yolo_params.yaml', 'w') as file:
    yaml.dump(data, file)



!pip install ultralytics > /dev/null 


!pip install ensemble-boxes  > /dev/null 


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
data_yaml = "/kaggle/working/yolo_params.yaml"

model.train(
    data=data_yaml,
    epochs=10,                
    batch=4,                   
    imgsz=1056,
    patience=150,               
    optimizer='SGD',
    momentum=0.937,          
    lr0=0.001,                
    weight_decay=0.0005,       
    cos_lr=True,               
    save_period=1,             
    workers=8,
    # Augmentations
    close_mosaic=20,
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
    
        results = model.predict(img_path, conf=conf, augment=True, iou=0.4, max_det=600, verbose=False)  
        
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
    
    print(f"[notice] âœ… Predictions saved: {output_dir}")
predict(test_images_path, output_dir , model, conf)


# Convert predictions to CSV
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
    print(submission_df.head(10))
    print(f"[notice] âœ… Submission saved to {output_csv}")

predictions_to_csv()


def filter_invalid_boxes(boxes, scores, labels):
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if abs(b[2] - b[0]) > 1e-6 and abs(b[3] - b[1]) > 1e-6:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels
    
def run_inference(models, image_sizes, test_images_path):
    image_paths = [p for p in Path(test_images_path).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    predictions = {}

    for model_idx, model in enumerate(models):
        model.eval()
        predictions[model_idx] = {}
        for size in image_sizes:
            predictions[model_idx][size] = {}
            pred = []
            for img_path in image_paths:
                image_id = img_path.stem
                image = Image.open(img_path)
                img_width, img_height = image.size

                results = model.predict(source=str(img_path), conf=conf,iou=0.4, max_det=600, augment=True, imgsz=size, verbose=False)
                boxes, scores, labels = [], [], []

                for result in results:
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist()
                    labels = result.boxes.cls.cpu().numpy().tolist()

                    norm_boxes = [
                        [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]
                        for x1, y1, x2, y2 in boxes
                    ]
                    norm_boxes, scores, labels = filter_invalid_boxes(norm_boxes, scores, labels)

                predictions[model_idx][size][image_id] = {
                    "boxes": norm_boxes,
                    "scores": scores,
                    "labels": labels
                }
                
                if boxes:
                    prediction_string = " ".join(
                        f"{int(lbl)} {score:.6f} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {(b[2]-b[0]):.6f} {(b[3]-b[1]):.6f}"
                        for b, score, lbl in zip(norm_boxes, scores, labels)
                    )
                else:
                    prediction_string = "no boxes"

                pred.append({
                    "image_id": image_id,
                    "prediction_string": prediction_string
                })

            # Save CSV per model and size
            df = pd.DataFrame(pred)
            csv_path = f"submission_{model_idx}_{size}.csv"
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"[saved] {csv_path}")
            print(df.head(10))

    return predictions

def apply_wbf_and_save_final_submission(predictions, image_ids, output_path="submission_wbf.csv"):
    wbf_results = []

    for image_id in image_ids:
        all_boxes, all_scores, all_labels = [], [], []

        for model_preds in predictions.values():
            for size_preds in model_preds.values():
                if image_id not in size_preds:
                    continue
                pred = size_preds[image_id]
                if not pred["boxes"]:
                    continue
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])

        if not all_boxes:
            pred_str = "no boxes"
        else:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )

            pred_str = " ".join(
                f"{int(lbl)} {score:.6f} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {(b[2]-b[0]):.6f} {(b[3]-b[1]):.6f}"
                for b, score, lbl in zip(fused_boxes, fused_scores, fused_labels)
            )

        wbf_results.append({
            "image_id": image_id,
            "prediction_string": pred_str
        })

    wbf_df = pd.DataFrame(wbf_results)
    wbf_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] âœ… WBF submission saved to {output_path}")
    print(wbf_df.head(10))




import os
from pathlib import Path
import pandas as pd
import csv
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

model_paths = [
    "/kaggle/working/runs/detect/train/weights/best.pt",
    # "/kaggle/working/runs/detect/train/weights/last.pt",
]

test_images_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"
output_dir = "/kaggle/working/predictions/labels"
conf = 0.0001
iou_thr = 0.5
skip_box_thr = 0.01
image_sizes = [1056, 1440, 1920, 2560, 3200]

models = [YOLO(path) for path in model_paths]
predictions = run_inference(models, image_sizes, test_images_path)

image_ids = list(next(iter(next(iter(predictions.values())).values())).keys())

apply_wbf_and_save_final_submission(predictions, image_ids)







