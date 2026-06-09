# Install Ultralytics
!pip install -q "ultralytics==8.2.103"
!pip uninstall -y ray ray[tune]
!pip install ensemble_boxes


# Core libs
import os
import random
from pathlib import Path
import shutil
import ast

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Disable Weights & Biases everywhere
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from ensemble_boxes import weighted_boxes_fusion

from ultralytics import YOLO, settings

print("Ultralytics version:", YOLO.__module__.split('.')[0])

settings.update({"wandb": False})


SRC_ROOT = Path("/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset")
WORK_ROOT = Path("/kaggle/working/CottonWeedDetection")  # editable copy

if not WORK_ROOT.exists():
    print("Creating working dataset...")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        src_split = SRC_ROOT / split
        dst_split = WORK_ROOT / split
        if src_split.exists():
            print(f"Copying {split} ...")
            shutil.copytree(src_split, dst_split)
        else:
            print(f"WARNING: {src_split} not found")
else:
    print("Working copy already exists:", WORK_ROOT)

print("SRC_ROOT:", SRC_ROOT)
print("WORK_ROOT:", WORK_ROOT)



DATA_ROOT = Path("/kaggle/working/CottonWeedDetection")

CSV_BY_SPLIT = {
    "train": Path("/kaggle/input/3cl-cotton-weed-det-final-labels/train_labels.csv"),
    "val":   Path("/kaggle/input/3cl-cotton-weed-det-final-labels/val_labels.csv"),
}

for split, csv_path in CSV_BY_SPLIT.items():
    img_dir = DATA_ROOT / split / "images"
    labels_dir = DATA_ROOT / split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {split.upper()} ===")
    print("CSV        :", csv_path)
    print("Images dir :", img_dir)
    print("Labels dir :", labels_dir)

    df = pd.read_csv(csv_path)

    required_cols = {"image", "width", "height", "bbs", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required cols: {missing}")

    # Wipe old labels so we only keep 3LC-corrected ones
    for txt in labels_dir.glob("*.txt"):
        txt.unlink()

    num_files = 0
    num_boxes = 0

    # One image per row
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Normalize to just filename so it matches dataset layout
        file_name = Path(row["image"]).name
        img_path = img_dir / file_name
        if not img_path.exists():
            # Label for an image that is not in this split – skip
            continue

        bbs_raw = row["bbs"]
        if pd.isna(bbs_raw):
            continue

        try:
            bbs_dict = ast.literal_eval(bbs_raw)
        except Exception as e:
            print(f"  [WARN] Could not parse bbs for {file_name}: {e}")
            continue

        bb_list = bbs_dict.get("bb_list", [])
        if not bb_list:
            continue

        lines = []
        for bb in bb_list:
            # x0,y0 are centre; x1,y1 are width,height (already normalized 0–1)
            cls_id = int(bb["label"])
            xc = float(bb["x0"])
            yc = float(bb["y0"])
            w  = float(bb["x1"])
            h  = float(bb["y1"])

            # Clamp to [0, 1] just in case
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w  = min(max(w,  0.0), 1.0)
            h  = min(max(h,  0.0), 1.0)

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            num_boxes += 1

        if lines:
            label_path = labels_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            num_files += 1

    print(f"[{split}] wrote {num_files} label files, total boxes = {num_boxes}")



import os
import random
import numpy as np

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass 

set_seed(42) 


ROOT = Path("/kaggle/working/CottonWeedDetection")
print("Dataset root:", ROOT)

yaml_path = Path("/kaggle/working/cotton_weed.yaml")
# for final model we also included validation set in training
yaml_text = f"""
path: {ROOT}          

train:
  - train/images      
  - val/images        

val: val/images      
test: test/images     

nc: 3
names:
  0: carpetweed
  1: morningglory
  2: palmer_amaranth
"""

yaml_path.write_text(yaml_text.strip() + "\n")
print("==== cotton_weed.yaml ====\n")
print(yaml_path.read_text())


def count_images(p: Path):
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

for split in ["train", "val", "test"]:
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    print(f"\n== {split.upper()} ==")
    print(" images:", count_images(img_dir), "in", img_dir)
    print(" labels:", len(list(lbl_dir.glob("*.txt"))), "in", lbl_dir)


# import os
# os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_MODE"] = "disabled"

# from ultralytics import settings, YOLO
# settings.update({"wandb": False})

# DATASET = "cotton_weed.yaml"

# search_space = {
#     "lr0":            (1e-5, 1e-1),
#     "lrf":            (0.01, 1.0),
#     "momentum":       (0.6, 0.98),
#     "weight_decay":   (0.0, 0.001),
#     "warmup_epochs":   (0.0, 5.0),
#     "warmup_momentum": (0.6, 0.98),
#     "warmup_bias_lr":  (0.01, 0.5),
#     "box":      (1.0, 12.0),
#     "cls":      (0.1, 3.0),
#     "dfl":      (0.5, 3.0),
#     "dropout":  (0.0, 0.2),
#     "hsv_h": (0.0, 0.1),
#     "hsv_s": (0.0, 0.9),
#     "hsv_v": (0.0, 0.9),
#     "degrees":     (0.0, 15.0),
#     "translate":   (0.0, 0.3),
#     "scale":       (0.0, 0.9),
#     "shear":       (0.0, 10.0),
#     "perspective": (0.0, 0.001),
#     "flipud":  (0.0, 0.2),
#     "fliplr":  (0.0, 0.7),
#     "bgr":     (0.0, 0.5),
#     "mosaic":  (0.0, 1.0),
#     "mixup":   (0.0, 0.7),
# }

# from pathlib import Path

# def run_tune_and_get_yaml(model_path, optimizer_name, epochs=10, iterations=12, batch=32):
#     model = YOLO(model_path)
#     print(f"\n=== Tuning {optimizer_name}: {iterations} iters x {epochs} epochs, batch={batch} ===\n")
#     model.tune(
#         data=DATASET,
#         epochs=epochs,
#         iterations=iterations,
#         optimizer=optimizer_name,
#         batch=batch,
#         imgsz=640,
#         space=search_space,
#         plots=False,
#         save=False,
#         val=True,
#         project="cotton_weed",
#         name=f"tune_{optimizer_name.lower()}",
#     )

#     # Pick the latest best_hyperparameters.yaml
#     candidates = list(Path("runs").rglob("best_hyperparameters.yaml"))
#     assert candidates, "No best_hyperparameters.yaml found; check runs/"
#     best_yaml = max(candidates, key=lambda p: p.stat().st_mtime)
#     print("Best hyperparameters yaml:", best_yaml)
#     return str(best_yaml)

# # Example: tune AdamW
# best_yaml_adam = run_tune_and_get_yaml("yolov8n.pt", optimizer_name="AdamW",
#                                        epochs=40, iterations=10, batch=64)
# print("AdamW best hyperparameters saved at:", best_yaml_adam)


# # # MODEL A
# !yolo task=detect mode=train \
#     model=yolov8n.pt \
#     data=/kaggle/working/cotton_weed.yaml \
#     epochs=200 \
#     patience=50 \
#     imgsz=640 \
#     batch=64 \
#     workers=4 \
#     optimizer=AdamW \
#     lr0=0.005 \
#     lrf=0.011 \
#     momentum=0.723 \
#     weight_decay=0.0004 \
#     warmup_epochs=3 \
#     warmup_momentum=0.66 \
#     warmup_bias_lr=0.096 \
#     box=6.345 \
#     cls=0.652 \
#     dfl=1.512 \
#     dropout=0.0 \
#     mosaic=0.721 \
#     close_mosaic=20 \
#     hsv_h=0.015 \
#     hsv_s=0.8 \
#     hsv_v=0.3 \
#     degrees=0.0 \
#     translate=0.094 \
#     scale=0.421 \
#     shear=0.0 \
#     perspective=0.0 \
#     flipud=0.0 \
#     fliplr=0.466 \
#     bgr=0.0 \
#     mixup=0.0 \
#     seed=42


# # # MODEL B ************************************
# !yolo task=detect mode=train \
#   model=yolov8n.pt \
#   data=/kaggle/working/cotton_weed.yaml \
#   epochs=200 patience=50 \
#   imgsz=640 batch=64 workers=4 \
#   optimizer=SGD \
#   lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
#   warmup_epochs=3 warmup_momentum=0.8 warmup_bias_lr=0.1 \
#   box=7.5 cls=0.5 dfl=1.5 \
#   dropout=0.0 \
#   mosaic=1.0 close_mosaic=10 \
#   hsv_h=0.02 hsv_s=0.8 hsv_v=0.45 \
#   degrees=10.0 translate=0.10 scale=0.50 shear=2.0 perspective=0.0 \
#   flipud=0.0 fliplr=0.5 bgr=0.0 mixup=0.0 \
#   seed=0

# # # MODEL C ***********************************************************************************************************
# !yolo task=detect mode=train \
#   model=yolov8n.pt \
#   data=/kaggle/working/cotton_weed.yaml \
#   epochs=150 patience=40 \
#   imgsz=640 batch=64 workers=4 \
#   optimizer=AdamW \
#   lr0=0.003 lrf=0.01 momentum=0.8 weight_decay=0.0006 \
#   warmup_epochs=2 warmup_momentum=0.7 warmup_bias_lr=0.08 \
#   box=5.5 cls=1.0 dfl=1.4 \
#   dropout=0.0 \
#   mosaic=0.3 close_mosaic=5 \
#   hsv_h=0.01 hsv_s=0.6 hsv_v=0.25 \
#   degrees=0.0 translate=0.05 scale=0.25 shear=0.0 perspective=0.0 \
#   flipud=0.0 fliplr=0.3 bgr=0.0 mixup=0.0 \
#   seed=123

# # MODEL D
# !yolo task=detect mode=train \
#   model=yolov8n.pt \
#   data=/kaggle/working/cotton_weed.yaml \
#   device=0,1 \
#   epochs=600 patience=80 \
#   imgsz=640 batch=128 workers=4 \
#   optimizer=AdamW \
#   lr0=0.004 lrf=0.01 momentum=0.80 weight_decay=0.0006 \
#   warmup_epochs=5 warmup_momentum=0.70 warmup_bias_lr=0.10 \
#   box=5.8 cls=1.2 dfl=1.5 \
#   label_smoothing=0.05 \
#   cos_lr=True \
#   dropout=0.0 \
#   mosaic=0.9 close_mosaic=30 \
#   hsv_h=0.03 hsv_s=0.90 hsv_v=0.50 \
#   degrees=10.0 translate=0.15 scale=0.60 shear=4.0 perspective=0.0005 \
#   flipud=0.10 fliplr=0.60 bgr=0.0 \
#   mixup=0.20 copy_paste=0.30 \
#   seed=7


# MODEL E
# !yolo task=detect mode=train \
#   model=yolov8n.pt \
#   data=/kaggle/working/cotton_weed.yaml \
#   device=0,1 \
#   epochs=900 patience=120 \
#   imgsz=640 batch=128 workers=4 \
#   optimizer=SGD \
#   lr0=0.009 lrf=0.01 momentum=0.94 weight_decay=0.0005 \
#   warmup_epochs=3 warmup_momentum=0.85 warmup_bias_lr=0.10 \
#   box=7.8 cls=0.45 dfl=1.6 \
#   label_smoothing=0.02 \
#   cos_lr=True \
#   dropout=0.0 \
#   mosaic=0.7 close_mosaic=40 \
#   hsv_h=0.02 hsv_s=0.80 hsv_v=0.40 \
#   degrees=5.0 translate=0.12 scale=0.70 shear=3.0 perspective=0.0005 \
#   flipud=0.05 fliplr=0.50 bgr=0.0 \
#   mixup=0.10 copy_paste=0.20 \
#   seed=2025


MODEL_PATHS = [
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_A.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_B.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_C.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_D1.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_D2.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_D3.pt",
    r"/kaggle/input/3cl-cotton-weed-det-yolov8n-models/pytorch/default/1/yolov8n_model_E.pt",
]

# Test dataset root (images/ + labels/)
TEST_ROOT = r"/kaggle/working/CottonWeedDetection/test"
TEST_IMAGES_DIR = os.path.join(TEST_ROOT, "images")

# YOLO inference settings for raw predictions
INFER_IMGSZ = 640
INFER_CONF = 0.01  # very low, keep almost everything
INFER_IOU = 0.9     # NMS IoU (just to remove exact dups per model)

# WBF settings
WBF_IOU_THR = 0.45
WBF_SKIP_BOX_THR = 0.01
WBF_WEIGHTS = [1, 1, 0.75, 0.5, 0.5, 1, 1] 

# Number of classes
NUM_CLASSES = 3

OUT_SUBMISSION_CSV = "submission.csv"





# =========================
# 1. LOAD MODELS
# =========================

def load_models(model_paths):
    """Load YOLO models from given paths."""
    models = []
    for p in model_paths:
        print(f"Loading model: {p}")
        models.append(YOLO(p))
    return models


# =========================
# 2. RAW PREDICTIONS (ALL MODELS, LOW CONF/IOU)
# =========================

def get_raw_predictions(models, images_dir, imgsz=640, conf=0.001, iou=0.7):
    """
    Run all models on all images in images_dir with low conf/IoU and
    return a DataFrame of raw predictions (normalized xyxy).
    Columns: image_id, model_idx, class_id, score, x1,y1,x2,y2
    """
    image_paths = sorted(
        list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    )
    rows = []

    for img_path in tqdm(image_paths, desc="Running models on test images"):
        image_id = img_path.stem
        for m_idx, model in enumerate(models):
            results = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )[0]

            if results.boxes is None or results.boxes.shape[0] == 0:
                continue

            # Normalized xyxy coordinates in [0, 1]
            xyxyn = results.boxes.xyxyn.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for b, s, c in zip(xyxyn, scores, classes):
                rows.append(
                    {
                        "image_id": image_id,
                        "model_idx": m_idx,
                        "class_id": int(c),
                        "score": float(s),
                        "x1": float(b[0]),
                        "y1": float(b[1]),
                        "x2": float(b[2]),
                        "y2": float(b[3]),
                    }
                )

    df = pd.DataFrame(rows)
    return df


# =========================
# 3. APPLY WBF ON DATAFRAME
# =========================

def apply_wbf_to_dataframe(df_preds, num_models, weights=None,
                           iou_thr=0.55, skip_box_thr=0.01):
    """
    Apply Weighted Boxes Fusion per image over predictions from multiple models.

    df_preds: DataFrame with columns [image_id, model_idx, class_id, score, x1,y1,x2,y2]
    Returns fused_df: DataFrame with columns [image_id, class_id, score, x1,y1,x2,y2]
    """
    fused_rows = []

    for image_id, df_img in tqdm(df_preds.groupby("image_id"), desc="Applying WBF"):
        boxes_list = []
        scores_list = []
        labels_list = []

        # Prepare list entries for each model 0..num_models-1
        for m_idx in range(num_models):
            df_m = df_img[df_img["model_idx"] == m_idx]
            if df_m.empty:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue

            boxes = df_m[["x1", "y1", "x2", "y2"]].values.tolist()
            scores = df_m["score"].values.tolist()
            labels = df_m["class_id"].values.tolist()

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if all(len(b) == 0 for b in boxes_list):
            # No boxes from any model
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='box_and_model_avg'
        )

        for b, s, c in zip(fused_boxes, fused_scores, fused_labels):
            fused_rows.append(
                {
                    "image_id": image_id,
                    "class_id": int(c),
                    "score": float(s),
                    "x1": float(b[0]),
                    "y1": float(b[1]),
                    "x2": float(b[2]),
                    "y2": float(b[3]),
                }
            )

    fused_df = pd.DataFrame(fused_rows)
    return fused_df


# =========================
# 5. BUILD SUBMISSION CSV
# =========================

def build_submission_csv(df_preds, images_dir, out_csv_path):
    """
    Convert final predictions (normalized xyxy) into Kaggle submission format:
    image_id,prediction_string
    """
    image_paths = sorted(
        list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    )
    img_ids = [p.stem for p in image_paths]

    grouped = {k: v for k, v in df_preds.groupby("image_id")} if not df_preds.empty else {}

    rows = []
    for img_id in img_ids:
        if img_id not in grouped:
            rows.append({"image_id": img_id, "prediction_string": "no box"})
            continue

        df_img = grouped[img_id]
        parts = []
        for row in df_img.itertuples(index=False):
            x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            xc = x1 + w / 2.0
            yc = y1 + h / 2.0

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            parts.extend([
                str(int(row.class_id)),
                f"{row.score:.6f}",
                f"{xc:.6f}",
                f"{yc:.6f}",
                f"{w:.6f}",
                f"{h:.6f}",
            ])

        pred_str = " ".join(parts) if parts else "no box"
        rows.append({"image_id": img_id, "prediction_string": pred_str})

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(str(out_csv_path), index=False)
    print(f"Saved submission CSV to: {out_csv_path}")
    return sub_df


# 1) load models
models = load_models(MODEL_PATHS)

# 2) raw predictions from all models on test set
raw_df = get_raw_predictions(
    models,
    TEST_IMAGES_DIR,
    imgsz=INFER_IMGSZ,
    conf=INFER_CONF,
    iou=INFER_IOU,
)


# 3) WBF fusion
fused_df = apply_wbf_to_dataframe(
    raw_df,
    num_models=len(models),
    weights=WBF_WEIGHTS,
    iou_thr=WBF_IOU_THR,
    skip_box_thr=WBF_SKIP_BOX_THR,
)


# 5) build submission CSV (for Kaggle)
_ = build_submission_csv(fused_df, TEST_IMAGES_DIR, OUT_SUBMISSION_CSV)

