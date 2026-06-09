# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install ultralytics
!pip install ensemble_boxes


"""# %% ì…€ 1: ê¸°ë³¸ ì…‹ì—…
import os
import itertools
import pandas as pd
from ultralytics import YOLO

# ê²°ê³¼ë¥¼ ì €ì�¥í•  ë£¨íŠ¸ ë””ë ‰í† ë¦¬
ROOT = "runs/search"
os.makedirs(ROOT, exist_ok=True)

# ë² ì�´ìŠ¤ ëª¨ë�¸ ë˜�ëŠ” config
MODEL = "yolov8n.pt"  # ë˜�ëŠ” pretrained weights ê²½ë¡œ
PARAM_FILE = "/kaggle/input/strodc/yolo_params.yaml"

# %% ì…€ 2: íƒ�ìƒ‰í•  augmentation ì„¤ì • ë¦¬ìŠ¤íŠ¸
aug_list = [
    {"mosaic": 0.0, "mixup": 0.0},        # augmentation ì—†ì�Œ
    {"mosaic": 0.3, "mixup": 0.1},        # ê¸°ë³¸ ëª©í‘œ
    {"mosaic": 0.5, "mixup": 0.2},
    {"mosaic": 0.0, "mixup": 0.2},
]

# %% ì…€ 3: íƒ�ìƒ‰í•  í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° ë¦¬ìŠ¤íŠ¸
hyp_list = [
    {"optimizer": "Adam",  "lr0": 0.001, "weight_decay": 0.0001},
    {"optimizer": "Adam",  "lr0": 0.0005, "weight_decay": 0.0001},
    {"optimizer": "SGD",   "lr0": 0.001, "weight_decay": 0.0001, "momentum": 0.937},
    {"optimizer": "SGD",   "lr0": 0.0005, "weight_decay": 0.0001, "momentum": 0.937},
]

# %% ì…€ 4: ì‹¤í—˜ ë°˜ë³µ ë°� ê²°ê³¼ ìˆ˜ì§‘
results = []

for aug, hyp in itertools.product(aug_list, hyp_list):
    # ì‹¤í—˜ ì�´ë¦„ ìƒ�ì„±
    aug_name = f"m{aug['mosaic']}_x{aug['mixup']}"
    hyp_name = f"{hyp['optimizer']}_lr{hyp['lr0']}"
    exp_name = f"{aug_name}__{hyp_name}"
    save_dir = os.path.join(ROOT, exp_name)
    
    # ëª¨ë�¸ ì´ˆê¸°í™”
    model = YOLO(MODEL)
    
    # train ì‹¤í–‰
    model.train(
        data=PARAM_FILE,
        epochs=30,
        patience=10,
        project=ROOT,
        name=exp_name,
        # augmentation
        mosaic=aug["mosaic"],
        mixup=aug["mixup"],
        # hyperparams
        optimizer=hyp["optimizer"],
        lr0=hyp["lr0"],
        weight_decay=hyp["weight_decay"],
        **({ "momentum": hyp["momentum"] } if "momentum" in hyp else {}),
        cos_lr=True,
        device="0,1"
    )
    
    # ê²°ê³¼ CSV ë¡œë“œ
    csv_path = os.path.join(save_dir, "results.csv")
    df = pd.read_csv(csv_path)
    # ë§ˆì§€ë§‰ epochì�˜ mAP50-95 ì¶”ì¶œ (ì»¬ëŸ¼ëª…ì�´ 'metrics/mAP_0.5:0.95'ì�¼ ê²½ìš°)
    # ì „ì²´ epoch ì¤‘ ìµœê³  mAP@50-95 ì¶”ì¶œ
    best_map095 = df["metrics/mAP50-95(B)"].max()
    # ìµœê³ ê°’ì�„ ê¸°ë¡�í•œ epoch ë²ˆí˜¸ (1-based)
    best_epoch = df["metrics/mAP50-95(B)"].idxmax() + 1
    
    
    # ê²°ê³¼ ì €ì�¥
    results.append({
        "exp": exp_name,
        "mosaic": aug["mosaic"],
        "mixup": aug["mixup"],
        "optimizer": hyp["optimizer"],
        "lr0": hyp["lr0"],
        "weight_decay": hyp["weight_decay"],
        "momentum": hyp.get("momentum", None),
        "best_epoch": best_epoch,
        "best_mAP@50-95": best_map095
    })

# %% ì…€ 5: ê²°ê³¼ ì •ë¦¬ ë°� ìµœì � ì¡°í•© ì¶œë ¥
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("best_mAP@50-95", ascending=False).reset_index(drop=True)

print("===== Top 5 Configurations =====")
print(results_df.head(5))

# í•„ìš”ì‹œ CSVë¡œ ì €ì�¥
results_df.to_csv(os.path.join(ROOT, "hyp_search_summary.csv"), index=False)
"""


"""
# ğŸ“� í•™ìŠµ: YOLOv8 ëª¨ë�¸ í•™ìŠµ
from ultralytics import YOLO
import os

# ê²½ë¡œ ì„¤ì •
WORK_DIR = os.getcwd()
MODEL_NAME = "yolov8s.pt"  # ì‹œì�‘ ê°€ì¤‘ì¹˜
PARAM_FILE = "/kaggle/input/strodc/yolo_params.yaml"  # train/val path, nc ë“± ì„¤ì •ë�œ íŒŒì�¼

# ëª¨ë�¸ ë¶ˆëŸ¬ì˜¤ê¸°
model = YOLO(MODEL_NAME)

# í•™ìŠµ
model.train(
    data=PARAM_FILE,
    epochs=30,
    patience=10,
    batch=16,
    device='cpu',       # â†� ë¦¬ìŠ¤íŠ¸ë¡œ ì „ë‹¬
    single_cls=True,
    degrees=10,
    mosaic=0,
    mixup=0.2,
    optimizer="Adam",
    lr0=0.001,               # ì´ˆê¸° í•™ìŠµë¥ 
    weight_decay=0.0001,      # ì •ê·œí™”ìš© weight decay
    cos_lr=True,
    amp=False,
    val=True,
    exist_ok=True,
)
"""


"""import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path
import cv2

# ëª¨ë�¸ ë¡œë“œ
model1 = YOLO('best1.pt')
model2 = YOLO('best2.pt')

# í…ŒìŠ¤íŠ¸ ì�´ë¯¸ì§€ í�´ë�”
image_dir = Path("/kaggle/input/your-dataset/test/images")
image_paths = sorted(image_dir.glob("*.jpg"))  # or *.png

# ê²°ê³¼ ì €ì�¥
results = []

def get_normalized_boxes(result):
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    h, w = result.orig_shape
    boxes_norm = [[x[0]/w, x[1]/h, x[2]/w, x[3]/h] for x in boxes]
    return boxes_norm, scores.tolist(), labels.tolist()

for img_path in image_paths:
    img = cv2.imread(str(img_path))

    # ëª¨ë�¸ë³„ ì˜ˆì¸¡
    res1 = model1.predict(img, conf=0.3, verbose=False)[0]
    res2 = model2.predict(img, conf=0.3, verbose=False)[0]

    # ë°•ìŠ¤ ë³€í™˜
    boxes1, scores1, labels1 = get_normalized_boxes(res1)
    boxes2, scores2, labels2 = get_normalized_boxes(res2)

    # ì•™ìƒ�ë¸”
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        [boxes1, boxes2], [scores1, scores2], [labels1, labels2],
        iou_thr=0.5, skip_box_thr=0.3
    )

    prediction_parts = []
    for box, score in zip(boxes_fused, scores_fused):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        prediction_parts.append(f"0 {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    prediction_string = " ".join(prediction_parts)
    image_id = img_path.stem
    results.append({"image_id": image_id, "prediction_string": prediction_string})

# CSVë¡œ ì €ì�¥
df = pd.DataFrame(results)
df.to_csv("submission.csv", index=False)
"""



# ğŸ”� ì˜ˆì¸¡: í…ŒìŠ¤íŠ¸ ì�´ë¯¸ì§€ì—� ëŒ€í•´ ë°”ìš´ë”© ë°•ìŠ¤ ì˜ˆì¸¡ + .txt ì €ì�¥
from ultralytics import YOLO
from pathlib import Path
import yaml
import os
from ensemble_boxes import weighted_boxes_fusion

# test ê²½ë¡œ ì�½ê¸°
images_dir = Path("/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images")

# ìµœì‹  í•™ìŠµ ëª¨ë�¸ ë¶ˆëŸ¬ì˜¤ê¸°
model1_path = "/kaggle/input/s2rodcyolo/pytorch/default/5/mosaic0.3mixup0.1SGDcos_lr0.00050.0005.pt"
model2_path = "/kaggle/input/s2rodcyolo/pytorch/default/5/mosaic0.3mixup0.1Adam0.00010.0001.pt"
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# ê²°ê³¼ ì €ì�¥í•  ë””ë ‰í† ë¦¬
labels_dir = Path("/kaggle/working/runs/detect/train/predictions/labels")
labels_dir.mkdir(parents=True, exist_ok=True)
def convert_result(result):
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    # Normalize box (0~1) for WBF
    h, w = result.orig_shape
    boxes_norm = [[x[0]/w, x[1]/h, x[2]/w, x[3]/h] for x in boxes]
    return boxes_norm, scores.tolist(), labels.tolist()

# ì˜ˆì¸¡ ë°� YOLO formatìœ¼ë¡œ ì €ì�¥
for img_path in images_dir.glob("*"):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    result1 = model1.predict(img_path, conf=0.015)[0]
    result2 = model2.predict(img_path, conf=0.015)[0]
    
    boxes1, scores1, labels1 = convert_result(result1)
    boxes2, scores2, labels2 = convert_result(result2)
    
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
    [boxes1, boxes2], [scores1, scores2], [labels1, labels2],
    iou_thr=0.5, skip_box_thr=0.3)

    label_path = labels_dir / (img_path.stem + ".txt")
    with open(label_path, "w") as f:
        for box, score in zip(boxes_fused, scores_fused):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            f.write(f"0 {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        """for box in result1.boxes:
            cls = int(box.cls)
            conf = box.conf.item()
            x, y, bw, bh = box.xywh[0]            # ëª¨ë‘� Tensor
            x_c = x.item()  / w
            y_c = y.item()  / h
            bw_norm = bw.item() / w
            bh_norm = bh.item() / h

            f.write(f"{cls} {conf:.4f} {x_c:.6f} {y_c:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")"""



# ğŸ§¾ CSV ë³€í™˜: YOLO .txt â†’ Kaggle ì œì¶œìš© CSV
import pandas as pd
import csv

def predictions_to_csv(preds_folder="/kaggle/working/runs/detect/train/predictions/labels", 
                       output_csv="submission.csv", 
                       test_images_folder="/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images",
                       allowed_extensions=(".jpg", ".jpeg", ".png")):

    preds_path = Path(preds_folder)
    test_path = Path(test_images_folder)
    assert preds_path.exists(), f"{preds_folder} not found"
    assert test_path.exists(), f"{test_images_folder} not found"

    test_ids = {p.stem for p in test_path.glob("*") if p.suffix.lower() in allowed_extensions}
    rows = []

    for txt_file in preds_path.glob("*.txt"):
        img_id = txt_file.stem
        if img_id not in test_ids:
            continue
        with open(txt_file) as f:
            lines = [line.strip() for line in f if line.strip()]
            valid = [line for line in lines if len(line.split()) == 6]
        pred_str = " ".join(valid) if valid else "no box"
        rows.append({"image_id": img_id, "prediction_string": pred_str})

    df_sub = pd.DataFrame({"image_id": list(test_ids)})
    df_pred = pd.DataFrame(rows)
    df_merged = df_sub.merge(df_pred, on="image_id", how="left").fillna("no boxes")
    df_merged.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Saved submission to {output_csv}")

# ì‹¤í–‰
predictions_to_csv()



submission = pd.read_csv("/kaggle/working/submission.csv")
submission




