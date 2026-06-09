!pip install ultralytics
!pip install ensemble-boxes


from pathlib import Path
import shutil

# Original val directories
val_root        = Path("/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/val")
val_images_dir  = val_root / "images"
val_labels_dir  = val_root / "labels"

# Destination for real images only
real_val_root   = Path("/kaggle/working/val_real")
real_images_dir = real_val_root / "images"
real_labels_dir = real_val_root / "labels"

# Make sure destination dirs exist
real_images_dir.mkdir(parents=True, exist_ok=True)
real_labels_dir.mkdir(parents=True, exist_ok=True)

# Copy only files starting with IMG and with jpg/png/jpeg extensions
for img_path in val_images_dir.iterdir():
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"] and img_path.name.startswith("IMG"):
        shutil.copy(img_path, real_images_dir / img_path.name)
        # Corresponding label
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, real_labels_dir / label_path.name)


PATHS = [
#   Falcon
    '/kaggle/input/falcon-soup-cans/outputallcarpet/Output/2025-06-11-23-00-26',
    '/kaggle/input/falcon-soup-cans/outputallcouch/Output/2025-06-08-20-02-03',
    '/kaggle/input/falcon-soup-cans/outputallfridge/Output/2025-06-11-22-20-13',
    '/kaggle/input/falcon-soup-cans/outputallplant/Output/2025-06-11-22-36-54',
    '/kaggle/input/falcon-soup-cans/outputalltable/Output/2025-06-08-13-07-03',
    '/kaggle/input/falcon-soup-cans/outputalltv/Output/2025-06-11-23-12-59',
#   More Falcon
    '/kaggle/input/falcon-soup-cans/outputcarpet2/Output/2025-06-14-14-56-25',
    '/kaggle/input/falcon-soup-cans/outputcarpet3/Output/2025-06-14-15-16-52',
    '/kaggle/input/falcon-soup-cans/outputcarpet4/Output/2025-06-14-15-38-01',
    '/kaggle/input/falcon-soup-cans/outputcarpet5/Output/2025-06-14-15-51-52',
    '/kaggle/input/falcon-soup-cans/outputcouch2/Output/2025-06-14-18-16-50',
    '/kaggle/input/falcon-soup-cans/outputcouch3/Output/2025-06-14-18-36-09',
    '/kaggle/input/falcon-soup-cans/outputcouch4/Output/2025-06-14-19-02-23',
    '/kaggle/input/falcon-soup-cans/outputcouch5/Output/2025-06-14-19-38-30',
    '/kaggle/input/falcon-soup-cans/outputfridge2/Output/2025-06-14-20-29-36',
    '/kaggle/input/falcon-soup-cans/outputfridge3/Output/2025-06-14-21-11-53',
    '/kaggle/input/falcon-soup-cans/outputfridge4/Output/2025-06-14-21-41-28',
    '/kaggle/input/falcon-soup-cans/outputfridge5/Output/2025-06-14-22-07-39',
    '/kaggle/input/falcon-soup-cans/outputplant2/Output/2025-06-15-18-40-41',
    '/kaggle/input/falcon-soup-cans/outputplant3/Output/2025-06-15-18-55-16',
    '/kaggle/input/falcon-soup-cans/outputplant4/Output/2025-06-15-19-10-39',
    '/kaggle/input/falcon-soup-cans/outputplant5/Output/2025-06-15-19-37-50',
    '/kaggle/input/falcon-soup-cans/outputtable2/Output/2025-06-14-12-36-22',
    '/kaggle/input/falcon-soup-cans/outputtable3/Output/2025-06-14-13-28-58',
    '/kaggle/input/falcon-soup-cans/outputtable4/Output/2025-06-14-13-46-12',
    '/kaggle/input/falcon-soup-cans/outputtable5/Output/2025-06-14-14-23-33',
    '/kaggle/input/falcon-soup-cans/outputtv2/Output/2025-06-15-19-57-01',
    '/kaggle/input/falcon-soup-cans/outputtv3/Output/2025-06-15-20-13-50',
    '/kaggle/input/falcon-soup-cans/outputtv4/Output/2025-06-15-20-44-05',
    '/kaggle/input/falcon-soup-cans/outputtv5/Output/2025-06-15-20-59-03',
    '/kaggle/input/falcon-soup-cans/topfridge1/Output/2025-06-17-18-29-33',
    '/kaggle/input/falcon-soup-cans/topfridge2/Output/2025-06-17-20-41-52',
    '/kaggle/input/falcon-soup-cans/topfridge3/Output/2025-06-17-21-02-37',
    '/kaggle/input/falcon-soup-cans/topfridge4/Output/2025-06-17-21-24-25',
    '/kaggle/input/falcon-soup-cans/topfridge5/Output/2025-06-17-21-57-17'
]
PATHS = [x + "/train/images" for x in PATHS] 
print(PATHS)


import shutil
from pathlib import Path

# List of image directories to copy
extra = [
    '/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset/train/images',
    '/kaggle/input/synthetic-soup1/output2/home/ubuntu/Output/2025-07-07-06-18-34/train/images',
    '/kaggle/input/synthetic-soup1/output3/home/ubuntu/Output/2025-07-07-06-21-39/train/images',
    '/kaggle/input/synthetic-soup1/output4/home/ubuntu/Output/2025-07-07-06-38-42/train/images',
    '/kaggle/input/synthetic-soup1/output5/home/ubuntu/Output/2025-07-07-10-27-36/train/images',
    '/kaggle/input/synthetic-soup1/output6/home/ubuntu/Output/2025-07-07-11-01-43/train/images',
    '/kaggle/input/synthetic-soup1/output7/home/ubuntu/Output/2025-07-07-11-44-04/train/images',
    '/kaggle/input/synthetic-soup1/output8/home/ubuntu/Output/2025-07-07-12-06-23/train/images',
    '/kaggle/input/synthetic-soup1/output10/home/ubuntu/Output/2025-07-07-18-27-05/train/images',
    '/kaggle/input/synthetic-soup1/output11/home/ubuntu/Output/2025-07-07-18-36-42/train/images',
    '/kaggle/input/synthetic-soup1/output12/home/ubuntu/Output/2025-07-07-18-51-48/train/images',
]
vals = [
    '/kaggle/input/synthetic-soup1/output/Output/2025-07-04-04-57-25/val/images',
    '/kaggle/input/synthetic-soup1/output3/home/ubuntu/Output/2025-07-07-06-21-39/val/images',
    '/kaggle/input/synthetic-soup1/output4/home/ubuntu/Output/2025-07-07-06-38-42/val/images',
    '/kaggle/input/synthetic-soup1/output9/home/ubuntu/Output/2025-07-07-12-31-34/train/images',
    '/kaggle/input/synthetic-soup1/output5/home/ubuntu/Output/2025-07-07-10-27-36/val/images',
    '/kaggle/input/synthetic-soup1/output6/home/ubuntu/Output/2025-07-07-11-01-43/val/images',
    '/kaggle/input/synthetic-soup1/output7/home/ubuntu/Output/2025-07-07-11-44-04/val/images',
    '/kaggle/input/synthetic-soup1/output8/home/ubuntu/Output/2025-07-07-12-06-23/val/images',
    '/kaggle/input/synthetic-soup1/output10/home/ubuntu/Output/2025-07-07-18-27-05/val/images',
    '/kaggle/input/synthetic-soup1/output11/home/ubuntu/Output/2025-07-07-18-36-42/val/images',
    '/kaggle/input/synthetic-soup1/output12/home/ubuntu/Output/2025-07-07-18-51-48/val/images',
]

# Base paths
base_input   = Path("/kaggle/input/falcon-soup-cans")
base_input   = Path("/kaggle/input/synthetic-soup1")
working_base = Path("/kaggle/working/synthetic_soup_extra")
PATHS = vals + extra
# 1) Copy each train/val folder into a writable working dir
for img_dir in PATHS:
    src_root = Path(img_dir).parent  # e.g. .../train or .../val
    rel      = src_root.relative_to(base_input)
    dst_root = working_base / rel
    if not dst_root.exists():
        dst_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_root, dst_root)

# 2) Remap old-class “0” → new-class “1” in all labels under the working copy
for labels_dir in working_base.rglob("labels"):
    for txt_file in labels_dir.glob("*.txt"):
        lines     = txt_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts: 
                continue 
            if parts[0] == "0":    # old “soup” class
                parts[0] = "1"     # new “soup” class
            new_lines.append(" ".join(parts))
        # write back with trailing newline
        txt_file.write_text("\n".join(new_lines) + "\n")


import yaml
from pathlib import Path

# Paths
base_path       = "/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset"
synthetic_base  = "/kaggle/input/extra-synthetic-data"
synthetic_base2 = '/kaggle/input/falcon-multiclass-cheerios-soupv2/falcon-multiclass-cheerios-soupV2'
synthetic_base3 = '/kaggle/working/synthetic_soup_extra'
# Gather synthetic train/val image dirs
synthetic_train_dirs = [str(p) for p in Path(synthetic_base).rglob("train/images")]
synthetic_val_dirs   = [str(p) for p in Path(synthetic_base).rglob("val/images")]
# Gather synthetic train/val image dirs
synthetic_train_dirs2 = [str(p) for p in Path(synthetic_base2).rglob("train/images")]
synthetic_val_dirs2   = [str(p) for p in Path(synthetic_base2).rglob("val/images")]
# Gather synthetic train/val image dirs
synthetic_train_dirs3 = [str(p) for p in Path(synthetic_base3).rglob("train/images")]
synthetic_val_dirs3   = [str(p) for p in Path(synthetic_base3).rglob("val/images")]

# Build YAML dict
data_yaml = {
    "train": (
        [
            f"{base_path}/train/images",
        '/kaggle/input/sample-synthetic-data-generated/home/ubuntu/Output/2025-07-15-06-26-28/train/images',
        '/kaggle/input/sample-synthetic-data-generated/home/ubuntu/Output/2025-07-15-06-26-28/val/images',
        ]
        + synthetic_train_dirs
        + synthetic_val_dirs
        # + synthetic_train_dirs2
        # + synthetic_val_dirs2
        # + synthetic_train_dirs3
        # + synthetic_val_dirs3
    ),
    "val":   '/kaggle/working/val_real/images',
    "test": f"{base_path}/TestImages",
    "nc":   2,
    "names": ["cheerios", "Soup"]
}

# Save to data.yaml
with open("data.yaml", "w") as f:
    yaml.safe_dump(data_yaml, f, default_flow_style=False)

print("Created data.yaml with the following content:")
print(yaml.safe_dump(data_yaml, default_flow_style=False))


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


# # model = YOLO("yolo11l.pt")
# # data_yaml = '/kaggle/working/data.yaml'

# # model.train(
# #     data=data_yaml,
# #     epochs=60,                
# #     batch=4,                   
# #     imgsz=1024,
# #     patience=500,               
# #     optimizer='SGD',        
# #     lr0=0.001,
# #     lrf = 0.0001,
# #     weight_decay=0.0001,       
# #     cos_lr=True,               
# #     save_period=10,             
# #     workers=4,
# #     # Augmentations
# #     close_mosaic=20,
# #     hsv_h=0.015,
# #     hsv_s=0.75,
# #     hsv_v=0.45,
# #     flipud=0.01,
# #     fliplr=0.5,
# #     translate=0.05,
# #     scale=0.75,
# #     shear=0.05,
# #     mixup = 0.35,
# #     cutmix = 0.1,
# #     warmup_epochs= 5,
# #     warmup_momentum= 1,
    
# #     exist_ok = True,
# #     project = 'runs1/train',
# #     plots = True, 
# #     augment = True,
# #     conf = 0.05,
# #     iou = 0.4,
# #     # multi_scale = True,
# #     # agnostic_nms=True,
# # )
model = YOLO("yolo11x.pt")
data_yaml = '/kaggle/working/data.yaml'

model.train(
    data=data_yaml,
    epochs=15,                
    batch=4,                   
    imgsz=672,
    patience=500,               
    optimizer='SGD',        
    lr0=0.001,
    lrf = 0.001,
    weight_decay=0.0003,       
    cos_lr=True,               
    save_period=10,             
    workers=4,
    # Augmentations
    close_mosaic=5,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0,
    fliplr=0.5,
    translate=0.01,
    scale=0.25,
    # shear=0.05,
    mixup = 0.05,
    cutmix = 0.1,
    warmup_epochs= 3,
    warmup_momentum= 1,
    
    exist_ok = True,
    project = 'runs1/train',
    plots = True, 
    augment = True,
    conf = 0.001,
    iou = 0.3,
    multi_scale = True,
    freeze = 4
    # agnostic_nms=True,
)


# model.val(iou = 0.3,conf = 0.08,augment= True)


# ## 8. Plot Validation Losses & Confusion Matrix

# # %% Cell: plot_val_and_confusion_matrix
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image

# # 1. Point to your training experiment directory
# exp_dir = Path("/kaggle/working/runs1/train/train")  # adjust if your run folder is different
# results_csv = exp_dir / "results.csv"

# # 2. Load the epoch-by-epoch metrics
# results = pd.read_csv(results_csv)

# # 3. Plot validation losses over epochs
# plt.figure(figsize=(10, 5))
# plt.plot(results['epoch'], results['val/box_loss'], label='val box loss')
# plt.plot(results['epoch'], results['val/cls_loss'], label='val cls loss')
# plt.plot(results['epoch'], results['val/dfl_loss'], label='val dfl loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# # plt.title('YOLO Validation Loasses')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 4. Locate and display the confusion matrix image
# #    (typically saved when you run `model.val()` in Ultralytics)
# cm_paths = ['/kaggle/working/runs1/train/train/results.png',
#             '/kaggle/working/runs1/train/train/confusion_matrix.png']
# print(cm_paths)
# for p in cm_paths :
#     cm_img = Image.open(p)
#     plt.figure(figsize=(25, 10))
#     plt.imshow(cm_img)
#     plt.axis('off')
#     # plt.title('Validation Confusion Matrix')
#     plt.show()


model = YOLO("yolo12x.pt")
# data_yaml = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml"
data_yaml = '/kaggle/working/data.yaml'

model.train(
    data=data_yaml,
    epochs=20,           
    batch=16,                   
    imgsz=512,
    patience=500,               
    optimizer='SGD', 
    lr0=0.001,
    lrf = 0.001,  
    weight_decay=0.0001,       
    cos_lr=True,               
    save_period=10,             
    workers=8,
    # Augmentations
    close_mosaic=5,
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.25,
    flipud=0.0,
    fliplr=0.5,
    translate=0.01,
    scale=0.75,
    shear=2,
    perspective = 0.001,
    mixup = 0.15,
    cutmix = 0.25,
    warmup_epochs= 5,
    warmup_momentum= 1,
    exist_ok = True,
    project = 'runs2/train',
    plots = True, 
    augment = True,
    conf = 0.001,
    iou = 0.3,
    freeze = 6,
    # multi_scale = True
    
)


model = YOLO("yolov8x.pt")
# data_yaml = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml"
data_yaml = '/kaggle/working/data.yaml'

model.train(
    data=data_yaml,
    epochs=30,                
    batch=32,                   
    imgsz=640,
    patience=500,               
    optimizer='SGD',         
    lr0=0.001,
    lrf = 0.0001,
    weight_decay=0.0002,       
    cos_lr=True,               
    save_period=10,             
    workers=8,
    # Augmentations
    close_mosaic=10,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0,
    fliplr=0.5,
    translate=0.05,
    erasing = 0.5,
    scale=0.75,
    # shear=0.2,
    mixup = 0.05,
    cutmix = 0.10,
    warmup_epochs= 3,
    warmup_momentum= 1,
    
    # exist_ok = True,
    project = 'runs3/train',
    plots = True, 
    augment = True,
    conf = 0.001,
    iou = 0.35,
    freeze = 8,
    # multi_scale = True,
    
)


# model.val(iou=0.3,conf =0.01,augment = True)


from pathlib import Path
from PIL import Image
import pandas as pd
import csv
from ensemble_boxes import weighted_boxes_fusion

def filter_invalid_boxes(boxes, scores, labels, conf_thr):
    # keep only non-degenerate boxes; ignore confidence here by passing conf_thr < 0
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if abs(b[2] - b[0]) <= 1e-6 or abs(b[3] - b[1]) <= 1e-6:
            continue
        if s < conf_thr:  # will be disabled by using conf_thr=-1.0
            continue
        filtered_boxes.append(b)
        filtered_scores.append(s)
        filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels

def run_inference(models,
                  image_sizes,
                  test_images_path,
                  conf=0.15,          # user-threshold to apply AFTER WBF
                  iou_thr=0.45,
                  max_det=50,
                  pre_conf=1e-3):     # very low pre-threshold so we don't drop boxes before WBF
    """
    Runs YOLO inference at multiple scales, stores raw preds (no confidence filtering),
    saves per-model/size CSVs, and returns a nested dict for WBF.
    """
    image_paths = [p for p in Path(test_images_path).glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    predictions = {}

    for model_idx, model in enumerate(models):
        model.eval()
        predictions[model_idx] = {}

        for size in image_sizes:
            predictions[model_idx][size] = {}
            rows = []

            for img_path in image_paths:
                image_id = img_path.stem
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                # run prediction with a very low conf to keep candidates
                results = model.predict(
                    source=str(img_path),
                    conf=pre_conf,
                    iou=iou_thr,
                    max_det=max_det,
                    augment=True,
                    imgsz=size,
                    verbose=False
                )
                res = results[0]

                raw_boxes  = res.boxes.xyxy.cpu().numpy().tolist()
                raw_scores = res.boxes.conf.cpu().numpy().tolist()
                raw_labels = res.boxes.cls.cpu().numpy().tolist()

                # normalize and keep geometry-valid boxes only (no conf filtering here)
                rel_boxes = [[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in raw_boxes]
                boxes, scores, labels = filter_invalid_boxes(rel_boxes, raw_scores, raw_labels, conf_thr=-1.0)

                predictions[model_idx][size][image_id] = {
                    "boxes":  boxes,
                    "scores": scores,
                    "labels": labels
                }

                # per-model CSV (raw, unfiltered by confidence)
                if boxes:
                    pred_str = " ".join(
                        f"{int(lbl)} {score:.6f} "
                        f"{(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} "
                        f"{(b[2]-b[0]):.6f} {(b[3]-b[1]):.6f}"
                        for b, score, lbl in zip(boxes, scores, labels)
                    )
                else:
                    pred_str = "no boxes"

                rows.append({"image_id": image_id, "prediction_string": pred_str})

            df = pd.DataFrame(rows)
            csv_path = f"submission_{model_idx}_{size}.csv"
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"[saved] {csv_path}")
            print(df.head(5))

    return predictions

def apply_wbf_and_save_final_submission(predictions,
                                        image_ids,
                                        output_path="submission.csv",
                                        iou_thr=0.4,
                                        skip_box_thr=0.0,    # do NOT filter before fusion
                                        conf_post=0.15,      # filter AFTER WBF using this threshold
                                        conf_type="avg"):    # or "max" if you prefer
    """
    Applies WBF across all models & sizes, then filters fused boxes by conf_post,
    and saves a single merged submission CSV.
    """
    final_rows = []

    for image_id in image_ids:
        all_boxes, all_scores, all_labels = [], [], []

        for model_preds in predictions.values():
            for size_preds in model_preds.values():
                pred = size_preds.get(image_id)
                if not pred or not pred["boxes"]:
                    continue
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])

        if not all_boxes:
            pred_str = "no boxes"
        else:
            fb, fs, fl = weighted_boxes_fusion(
                all_boxes,
                all_scores,
                all_labels,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
                conf_type=conf_type
            )

            keep = [i for i, s in enumerate(fs) if s >= conf_post]
            if not keep:
                pred_str = "no boxes"
            else:
                pred_str = " ".join(
                    f"{int(fl[i])} {fs[i]:.6f} "
                    f"{(fb[i][0]+fb[i][2])/2:.6f} {(fb[i][1]+fb[i][3])/2:.6f} "
                    f"{(fb[i][2]-fb[i][0]):.6f} {(fb[i][3]-fb[i][1]):.6f}"
                    for i in keep
                )

        final_rows.append({"image_id": image_id, "prediction_string": pred_str})

    pd.DataFrame(final_rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] ✅ WBF submission saved to {output_path}")


import os
from pathlib import Path
import pandas as pd
import csv
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

model_paths = [
    # '/kaggle/working/runs2/train/train/weights/last.pt',
    # '/kaggle/working/runs1/train/train/weights/last.pt',
    # '/kaggle/working/runs3/train/train/weights/last.pt',
    # '/kaggle/working/runs2/train/train/weights/best.pt',
    '/kaggle/working/runs1/train/train/weights/best.pt',
    '/kaggle/working/runs3/train/train/weights/best.pt',
    '/kaggle/working/runs2/train/train/weights/best.pt',
]

test_images_path = "/kaggle/input/multi-class-object-detection-challenge/testImages/images"
output_dir = "/kaggle/working/predictions/labels"

conf = 0.0001
iou_thr = 0.3
skip_box_thr = conf
image_sizes = [640,800,864,1024,1216,512,1344,2048]
# image_sizes = [512,640,1024,864,1216,2048]
# image_sizes = [992,1024,1056]


models = [YOLO(path) for path in model_paths]
predictions = run_inference(models, image_sizes, test_images_path,conf=conf,iou_thr=iou_thr)

image_ids = list(next(iter(next(iter(predictions.values())).values())).keys())

apply_wbf_and_save_final_submission(predictions, image_ids,iou_thr=0.4,skip_box_thr=0.001,conf_post=0.14)


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.patches import Rectangle

def plot_submission_predictions(
    submission_csv: str,
    test_images_path: str,
    K: int = 5,
    min_conf: float = 0.0
):
    """
    Plot the first K images from submission.csv with their predicted boxes,
    but only show boxes with confidence >= min_conf.
    
    submission_csv:   path to your final submission.csv
    test_images_path: folder containing the test images
    K:                number of images to visualize
    min_conf:         minimum confidence threshold for drawing a box
    """
    df = pd.read_csv(submission_csv)
    img_folder = Path(test_images_path)
    
    for _, row in df.head(K).iterrows():
        image_id = row["image_id"]
        pred_str = row["prediction_string"]
        
        # locate the image file
        matches = list(img_folder.glob(f"{image_id}.*"))
        if not matches:
            print(f"⚠️  Could not find file for {image_id}")
            continue
        img = Image.open(matches[0])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis("off")
        
        if pred_str.lower() != "no boxes":
            toks = pred_str.split()
            for i in range(0, len(toks), 6):
                lbl   = int(toks[i])
                score = float(toks[i+1])
                if score < min_conf:
                    continue
                x_c   = float(toks[i+2])
                y_c   = float(toks[i+3])
                w     = float(toks[i+4])
                h     = float(toks[i+5])
                
                # convert normalized center w,h to absolute top-left corner + size
                x1    = (x_c - w/2) * img.width
                y1    = (y_c - h/2) * img.height
                abs_w = w * img.width
                abs_h = h * img.height
                
                rect = Rectangle((x1, y1), abs_w, abs_h,
                                 fill=False, edgecolor="red", lw=2)
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 3,
                    f"{lbl}:{score:.2f}",
                    color="yellow", fontsize=10,
                    backgroundcolor="black", alpha=0.7
                )
        
        plt.show()
plot_submission_predictions(
    submission_csv="/kaggle/working/submission.csv",
    test_images_path= "/kaggle/input/multi-class-object-detection-challenge/testImages/images",
    K=280,
    min_conf=0.005   # only show boxes with confidence ≥ min_conf
)


