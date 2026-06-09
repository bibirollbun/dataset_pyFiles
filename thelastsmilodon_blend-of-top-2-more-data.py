!pip install ultralytics
!pip install ensemble-boxes


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


import yaml
base_path = "/kaggle/input/multi-class-object-detection-challenge/Dataset"

# Build YAML dictionary
data_yaml = {
    "train": [f"{base_path}/train/images",'/kaggle/input/sample-synthetic-data-generated/home/ubuntu/Output/2025-07-15-06-26-28/train/images',
             '/kaggle/input/sample-synthetic-data-generated/home/ubuntu/Output/2025-07-15-06-26-28/val/images'],
    "val":   f"{base_path}/val/images" ,
    "test":  f"{base_path}/TestImages",
    "nc":    2,
    "names": ["cheerios","Soup"]
}

# Save to data.yaml
with open("data.yaml", "w") as f:
    yaml.safe_dump(data_yaml, f, default_flow_style=False)

print("Created data.yaml with the following content:")
print(yaml.safe_dump(data_yaml, default_flow_style=False))



TRAIN = False

if TRAIN : 
    TRAIN = YOLO("yolo11x.pt")
    data_yaml = '/kaggle/working/data.yaml'
    
    model.train(
        data=data_yaml,
        epochs=60,                
        batch=16,                   
        imgsz=512,
        patience=100,               
        optimizer='SGD',        
        lr0=0.001,
        lrf = 0.0001,
        dropout = 0.5,
        weight_decay=0.0001,       
        cos_lr=True,               
        save_period=10,             
        workers=4,
        # Augmentations
        close_mosaic=15,
        hsv_h=0.025,
        hsv_s=0.75,
        hsv_v=0.45,
        flipud=0.05,
        fliplr=0.5,
        translate=0.05,
        scale=0.5,
        shear=0.0,
        warmup_epochs= 5,
        warmup_momentum= 1,
        
        exist_ok = True,
        project = 'runs2/train',
        plots = True, 
        augment = True,
        conf = 0.15,
        iou = 0.45,
        
        # agnostic_nms=True,
    )


if TRAIN: 
    # Plot Validation Losses & Confusion Matrix
    
    # %% Cell: plot_val_and_confusion_matrix
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    
    # 1. Point to your training experiment directory
    exp_dir = Path("/kaggle/working/runs2/train/train")  # adjust if your run folder is different
    results_csv = exp_dir / "results.csv"
    
    # 2. Load the epoch-by-epoch metrics
    results = pd.read_csv(results_csv)
    
    # 3. Plot validation losses over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(results['epoch'], results['val/box_loss'], label='val box loss')
    plt.plot(results['epoch'], results['val/cls_loss'], label='val cls loss')
    plt.plot(results['epoch'], results['val/dfl_loss'], label='val dfl loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('YOLO Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 4. Locate and display the confusion matrix image
    #    (typically saved when you run `model.val()` in Ultralytics)
    cm_paths = ['/kaggle/working/runs2/train/train/results.png',
                '/kaggle/working/runs2/train/train/confusion_matrix.png']
    print(cm_paths)
    for p in cm_paths :
        cm_img = Image.open(p)
        plt.figure(figsize=(25, 10))
        plt.imshow(cm_img)
        plt.axis('off')
        # plt.title('Validation Confusion Matrix')
        plt.show()



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

                results = model.predict(source=str(img_path), conf=conf,iou=iou_thr, max_det=100, augment=True, imgsz=size, verbose=False)
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

def apply_wbf_and_save_final_submission(predictions, image_ids, output_path="submission.csv"):
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
    '/kaggle/input/3lc-yolo-baseline-submission/Duality-3LC-Kaggle/run-1/weights/best.pt',
    '/kaggle/input/yolo-baseline-updated-dataset-more-data/runs2/train/train/weights/best.pt',
]

test_images_path = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
output_dir = "/kaggle/working/predictions/labels"

conf = 0.04
iou_thr = 0.4
skip_box_thr = 0.01
image_sizes = [640,1024,864]


models = [YOLO(path) for path in model_paths]
predictions = run_inference(models, image_sizes, test_images_path)

image_ids = list(next(iter(next(iter(predictions.values())).values())).keys())

apply_wbf_and_save_final_submission(predictions, image_ids)

