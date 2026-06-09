# Single-cell Kaggle notebook for the 3LC Cotton Weed Detection Challenge (UPDATED)
# - Entire workflow in one runnable cell.
# - Comments explain research ideas, pipeline steps, and rationale (the word "you" is not used in comments).
# - No external 3LC dependency.
# - Pipeline: environment, dataset discovery, copy writable label files, label sanitation,
#   baseline YOLOv8n training, pseudo-label mining (self-training), retrain, inference, submission CSV.
# - Fixed ultralytics argument compatibility (replaced deprecated 'hsv' argument with hsv_h/hsv_s/hsv_v).
#
# Notes:
# - This cell can be long-running and GPU-intensive. Reduce epochs if kernel runtime is constrained.
# - The script copies label .txt files to /kaggle/working to avoid read-only filesystem errors.

import os
import sys
import math
import glob
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# --- 1) Ensure ultralytics YOLOv8 availability ---
try:
    import ultralytics
except Exception:
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.*"])
    import ultralytics

from ultralytics import YOLO

# --- 2) Dataset root discovery (explicit path for Kaggle competition) ---
dataset_root = Path("/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset")
if not dataset_root.exists():
    # fallback: scan /kaggle/input
    for p in Path("/kaggle/input").iterdir():
        if "cotton" in p.name.lower() and "weed" in p.name.lower():
            dataset_root = p
            break
if not dataset_root.exists():
    raise FileNotFoundError("Dataset root not found under /kaggle/input. Place dataset there or adjust dataset_root path.")

# --- 3) Image and label directories (images read from input; labels will be copied to writable area) ---
train_images_dir = dataset_root / "train" / "images"
orig_train_labels_dir = dataset_root / "train" / "labels"
val_images_dir = dataset_root / "val" / "images"
orig_val_labels_dir = dataset_root / "val" / "labels"
test_images_dir = dataset_root / "test" / "images"

# Validate presence and warn if something missing
for p in (train_images_dir, orig_train_labels_dir, val_images_dir, orig_val_labels_dir, test_images_dir):
    if not p.exists():
        print(f"Warning: expected path not found: {p}")

# --- 4) Create writable copies of label files under /kaggle/working (Kaggle /kaggle/input is read-only) ---
wrk = Path("/kaggle/working/cotton_dataset")
wrk_train_labels = wrk / "train" / "labels"
wrk_val_labels = wrk / "val" / "labels"
os.makedirs(wrk_train_labels, exist_ok=True)
os.makedirs(wrk_val_labels, exist_ok=True)

def copy_label_files(src_dir: Path, dst_dir: Path):
    txts = sorted(glob.glob(str(src_dir / "*.txt"))) if src_dir.exists() else []
    for t in txts:
        dst = dst_dir / Path(t).name
        if not dst.exists():
            shutil.copy2(t, str(dst))

copy_label_files(orig_train_labels_dir, wrk_train_labels)
copy_label_files(orig_val_labels_dir, wrk_val_labels)

# Update label dir variables to writable copies
train_labels_dir = wrk_train_labels
val_labels_dir = wrk_val_labels

print("Dataset directories:")
print(" train_images_dir ->", train_images_dir)
print(" train_labels_dir ->", train_labels_dir)
print(" val_images_dir   ->", val_images_dir)
print(" val_labels_dir   ->", val_labels_dir)
print(" test_images_dir  ->", test_images_dir)

# --- 5) Utility: read/write YOLO-format label files ---
def read_yolo_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            vals = list(map(float, parts[1:5]))
            boxes.append((cls, *vals))
    return boxes

def write_yolo_label(path, boxes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for b in boxes:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(b[0]), b[1], b[2], b[3], b[4]))

# --- 6) Geometry helpers (normalized coords -> absolute and back) ---
def yolo_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    x_c = x1 + w/2
    y_c = y1 + h/2
    if img_w == 0 or img_h == 0:
        return [0.5, 0.5, 0.0, 0.0]
    return [x_c/img_w, y_c/img_h, w/img_w, h/img_h]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

# --- 7) Label sanitation: clip coords, remove tiny boxes, de-duplicate by IoU ---
MIN_REL_AREA = 0.0008  # 0.08% of image area threshold for filtering tiny boxes

def sanitize_labels(images_dir, labels_dir, min_rel_area=MIN_REL_AREA, dry_run=False):
    print("Sanitizing labels in:", labels_dir)
    modified = 0
    total_boxes = 0
    label_paths = sorted(glob.glob(str(labels_dir / "*.txt")))
    for lab in label_paths:
        img_name = Path(lab).stem
        img_path = None
        for ext in ("jpg","jpeg","png","JPG","PNG"):
            cand = images_dir / (img_name + "." + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            # If corresponding image not present, skip sanitation for this label file
            continue
        img = Image.open(img_path)
        w,h = img.size
        boxes = read_yolo_label(lab)
        total_boxes += len(boxes)
        new_boxes = []
        for b in boxes:
            cls, x_c, y_c, bw, bh = b
            x_c = min(0.999999, max(0.0, x_c))
            y_c = min(0.999999, max(0.0, y_c))
            bw = min(0.999999, max(0.0, bw))
            bh = min(0.999999, max(0.0, bh))
            area_rel = bw * bh
            if area_rel < min_rel_area:
                continue
            if bw <= 0 or bh <= 0:
                continue
            new_boxes.append((cls, x_c, y_c, bw, bh))
        # de-duplicate by IoU among boxes of same class (keep larger area)
        kept = []
        for cand in sorted(new_boxes, key=lambda x: x[3]*x[4], reverse=True):
            cls, x_c, y_c, bw, bh = cand
            cand_xy = yolo_to_xyxy((x_c,y_c,bw,bh), w, h)
            overlap = False
            for k in kept:
                if k[0] != cls:
                    continue
                k_xy = yolo_to_xyxy((k[1],k[2],k[3],k[4]), w, h)
                if iou_xyxy(cand_xy, k_xy) > 0.85:
                    overlap = True
                    break
            if not overlap:
                kept.append(cand)
        if dry_run:
            if len(kept) != len(boxes):
                modified += 1
        else:
            write_yolo_label(lab, kept)
            if len(kept) != len(boxes):
                modified += 1
    print(f"Sanitation complete. Modified files: {modified}. Total input boxes: {total_boxes}")
    return modified

# Run sanitation on the writable copies of labels
sanitize_labels(train_images_dir, train_labels_dir, dry_run=False)
sanitize_labels(val_images_dir, val_labels_dir, dry_run=False)

# --- 8) Prepare YOLO data.yaml for ultralytics (points at image folders; labels are auto-discovered by filename) ---
data_yaml_path = Path("cotton_yolo_data.yaml")
data_yaml = {
    "names": {0: "carpetweed", 1: "morningglory", 2: "palmeramaranth"},
    "nc": 3,
    "train": str(train_images_dir.resolve()),
    "val": str(val_images_dir.resolve()),
    "test": str(test_images_dir.resolve())
}
import yaml
with open(data_yaml_path, "w") as f:
    yaml.safe_dump(data_yaml, f)
print("Wrote data yaml to", data_yaml_path)

# --- 9) Training hyperparameters (adaptive) ---
import torch

# device selection (keep as-is)
device = 0 if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.is_available()

# Training image size and batch tuned for Tesla P100 16GB (adjust if OOM)
imgsz = 896             # larger than 640 -> better small-object handling
bs = 8 if gpu else 4    # batch size (per-GPU). reduce if OOM.

# Baseline / retrain epochs increased
epochs_baseline = 80

# Slightly lower minimum relative box area to keep small weeds
MIN_REL_AREA = 0.0005  # 0.05% of image area

# Pseudo-label mining thresholds (more conservative, higher confidence)
CONF_THR = 0.70
IOU_MATCH = 0.50

# Final inference confidence threshold (lowered a bit to keep recall, adjust if precision suffers)
FINAL_CONF_THR = 0.25

# Baseline training args (ultralytics train(**train_args))
train_args = dict(
    data=str(data_yaml_path),
    epochs=epochs_baseline,
    imgsz=imgsz,
    batch=bs,
    device=device,
    name="yolov8n_baseline",
    exist_ok=True,
    workers=8,           # more workers if I/O allows
    mosaic=1,            # keep mosaic for small dataset
    mixup=0.20,          # useful regularization
    lr0=0.002,           # start LR (optimizer auto may adapt; this is a good starting point for imgsz~896)
    weight_decay=1e-4,   # slightly lower wd
    degrees=15.0,
    translate=0.20,
    scale=0.25,
    shear=2.5,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.45,
    cache=True,          # cache images in RAM for faster epochs (warning: memory use increases)
    patience=50          # reduce patience so training can adapt early
)

# Retrain (self-train) hyperparameters
retrain_epochs = 60
retrain_name = "yolov8n_selftrain"
retrain_args = dict(
    data=str(data_yaml_path),
    epochs=retrain_epochs,
    imgsz=imgsz,
    batch=bs,
    device=device,
    name=retrain_name,
    exist_ok=True,
    workers=8,
    lr0=0.0015,          # slightly lower initial LR for fine-tuning
    weight_decay=1e-4,
    mosaic=1,
    mixup=0.15,
    degrees=12.0,
    translate=0.12,
    scale=0.15,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.45,
    cache=True,
    patience=40
)

# --- 10) Baseline training using YOLOv8n pretrained weights ---
print("Initializing YOLOv8n model (pretrained COCO).")
model = YOLO("yolov8n.pt")  # will download weights if needed

print("Starting baseline training (this step may be time-consuming).")
# Wrap training in try/except to provide clearer error messages if configuration issues arise
try:
    results = model.train(**train_args)
except Exception as e:
    print("Error during baseline training. Exception follows:")
    raise

# After training, the model object refers to the trained model; runs saved under runs/detect/<name>
runs_dir = Path("runs") / "detect" / train_args["name"]
best_pt = None
if runs_dir.exists():
    ckpts = sorted(runs_dir.glob("weights/*.pt"), key=os.path.getmtime)
    if ckpts:
        best_pt = str(ckpts[-1])
print("Baseline best checkpoint:", best_pt or "Not found; model object will be used.")


# --- 11) Pseudo-label mining: add high-confidence predictions on training images that do not match GT (IoU < threshold) ---

# Thresholds (tuned)
CONF_THR = 0.70         # higher confidence for pseudo labels => fewer noisy boxes
IOU_MATCH = 0.50        # same matching IoU (if pred IoU < this vs GT, consider adding)
MIN_REL_AREA = 0.0005   # slightly smaller to keep very small weed boxes

trained_model = model  # reference to trained model

train_img_paths = sorted(glob.glob(str(train_images_dir / "*.*")))
train_label_paths = {Path(p).stem: p for p in glob.glob(str(train_labels_dir / "*.txt"))}

# helpers to safely extract scalar and xyxy array
def tensor_to_float(t):
    try:
        return float(t.item())
    except Exception:
        return float(np.array(t).squeeze())

def xyxy_tensor_to_list(box_xyxy):
    # box_xyxy may be a tensor of shape (1,4) or (4,)
    arr = np.array(box_xyxy.cpu())
    arr = np.asarray(arr).reshape(-1)
    if arr.size < 4:
        # fallback to zeros (should not happen)
        return [0.0, 0.0, 0.0, 0.0]
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]

def mine_pseudo_labels(trained_model, img_paths, labels_dir, conf_thr=CONF_THR, iou_match=IOU_MATCH, min_rel_area=MIN_REL_AREA):
    added = 0
    explored = 0
    for img_p in img_paths:
        stem = Path(img_p).stem
        explored += 1

        # run inference on single image; use low-level predict API
        res = trained_model.predict(source=img_p, imgsz=imgsz, conf=conf_thr, iou=0.45, device=device, verbose=False)
        if not res:
            continue
        r = res[0]
        # r.boxes may be an object with fields; if no boxes, skip
        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            continue

        # image size
        img = Image.open(img_p)
        w, h = img.size

        preds = []
        # iterate predicted boxes
        for box in r.boxes:
            # box.conf and box.cls are tensors; extract scalars safely
            conf = tensor_to_float(box.conf)
            # box.cls might be an array-like; coerce to int
            cls = int(tensor_to_float(box.cls))
            # box.xyxy might be tensor shape (1,4) or (4,)
            xyxy_list = xyxy_tensor_to_list(box.xyxy)
            # convert to yolo normalized
            x1, y1, x2, y2 = xyxy_list
            yolo_box = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            rel_area = float(yolo_box[2]) * float(yolo_box[3])
            # filter by confidence and size
            if conf < conf_thr or rel_area < min_rel_area:
                continue
            preds.append((cls, conf, yolo_box))

        if not preds:
            continue

        # load ground-truth boxes for this image (writable labels folder)
        gt_path = os.path.join(labels_dir, stem + ".txt")
        gt_boxes = read_yolo_label(gt_path)
        gt_xy = []
        for g in gt_boxes:
            _, x_c, y_c, bw, bh = g
            gt_xy.append(yolo_to_xyxy((x_c, y_c, bw, bh), w, h))

        to_add = []
        for cls, conf, ybox in preds:
            pred_xy = yolo_to_xyxy((ybox[0], ybox[1], ybox[2], ybox[3]), w, h)
            overlaps = [iou_xyxy(pred_xy, gxy) for gxy in gt_xy] if gt_xy else []
            max_iou = max(overlaps) if overlaps else 0.0
            # only add if it does not match existing GT (likely missed annotation)
            if max_iou < iou_match:
                to_add.append((cls, ybox))

        if to_add:
            # append to writable label file
            label_file = os.path.join(labels_dir, stem + ".txt")
            existing = read_yolo_label(label_file)
            for cls, yb in to_add:
                existing.append((cls, float(yb[0]), float(yb[1]), float(yb[2]), float(yb[3])))
                added += 1
            write_yolo_label(label_file, existing)

    print(f"Pseudo-label mining complete. Images scanned: {explored}. Boxes added: {added}")
    return added

print("Starting pseudo-label mining (high-confidence training set predictions appended to label files).")
added = mine_pseudo_labels(trained_model, train_img_paths, str(train_labels_dir))


# --- 12) Retrain model after pseudo-label augmentation (robust implementation) ---
retrain_epochs = 60
retrain_name = "yolov8n_selftrain"
retrain_args = dict(
    data=str(data_yaml_path),
    epochs=retrain_epochs,
    imgsz=imgsz,
    batch=bs,
    device=device,
    name=retrain_name,
    exist_ok=True,
    workers=8,
    lr0=0.0015,
    weight_decay=1e-4,
    mosaic=1,
    mixup=0.15,
    degrees=12.0,
    translate=0.12,
    scale=0.15,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.45
)

# locate best checkpoint from baseline run
baseline_run_dir = Path("runs") / "detect" / train_args["name"]
best_ckpt = None
if baseline_run_dir.exists():
    # prefer 'weights/best.pt', fallback to 'weights/last.pt'
    cand_best = baseline_run_dir / "weights" / "best.pt"
    cand_last = baseline_run_dir / "weights" / "last.pt"
    if cand_best.exists():
        best_ckpt = str(cand_best)
    elif cand_last.exists():
        best_ckpt = str(cand_last)

# fallback to the original pretrained if no baseline checkpoint found
if best_ckpt is None:
    print("Warning: baseline checkpoint not found; retraining will start from pretrained yolov8n weights.")
    best_ckpt = "yolov8n.pt"

print("Retrain: using checkpoint:", best_ckpt)

# create a fresh YOLO model object loaded from the checkpoint (clears prior overrides)
model_for_retrain = YOLO(best_ckpt)

print("Starting retraining after pseudo-label augmentation.")
try:
    retrain_results = model_for_retrain.train(**retrain_args)
    # update reference so subsequent inference uses the retrained model
    trained_model = model_for_retrain
except Exception as e:
    print("Error during retraining. Exception follows:")
    raise



# --- 13) Inference on test set and build submission CSV (robust xyxy handling for inference too) ---
test_image_paths = sorted(glob.glob(str(test_images_dir / "*.*")))
sub_rows = []
FINAL_CONF_THR = 0.25   # lowered to improve recall (tune if precision drops)

def infer_with_tta(model, img_path, imgsz=imgsz, conf=0.001, iou=0.45, tta=True):
    detections = []

    def parse_result(res_obj, img_w, img_h):
        parsed = []
        if not res_obj or not hasattr(res_obj, "boxes") or len(res_obj.boxes) == 0:
            return parsed
        for box in res_obj.boxes:
            conf = tensor_to_float(box.conf)
            cls = int(tensor_to_float(box.cls))
            xyxy_list = xyxy_tensor_to_list(box.xyxy)
            x1,y1,x2,y2 = xyxy_list
            ybox = xyxy_to_yolo(x1,y1,x2,y2, img_w, img_h)
            parsed.append((cls, conf, ybox))
        return parsed

    # base prediction
    base_res = model.predict(source=img_path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    if base_res:
        # determine image size once
        img = Image.open(img_path)
        w,h = img.size
        detections += parse_result(base_res[0], w, h)

    # TTA: horizontal flip
    if tta:
        img = Image.open(img_path)
        w,h = img.size
        img_flipped = ImageOps.mirror(img)
        tmp_path = "/tmp/tta_tmp.jpg"
        img_flipped.save(tmp_path)
        res_f = model.predict(source=tmp_path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        if res_f:
            # parsed boxes are on flipped image; flip x coords back
            parsed = []
            for box in res_f[0].boxes if hasattr(res_f[0], "boxes") else []:
                c = tensor_to_float(box.conf)
                cl = int(tensor_to_float(box.cls))
                xyxy_list = xyxy_tensor_to_list(box.xyxy)
                x1f,y1f,x2f,y2f = xyxy_list
                # mirror horizontally back to original coords
                mx1 = w - x2f
                mx2 = w - x1f
                ybox = xyxy_to_yolo(mx1, y1f, mx2, y2f, w, h)
                parsed.append((cl, c, ybox))
            detections += parsed

    # Simple per-class NMS (suppress overlapping boxes per class)
    final = []
    detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
    img_w, img_h = Image.open(img_path).size
    for cls, conf, ybox in detections_sorted:
        keep = True
        xy = yolo_to_xyxy((ybox[0], ybox[1], ybox[2], ybox[3]), img_w, img_h)
        for k in final:
            if k[0] != cls:
                continue
            kxy = yolo_to_xyxy((k[2][0], k[2][1], k[2][2], k[2][3]), img_w, img_h)
            if iou_xyxy(xy, kxy) > 0.45:
                keep = False
                break
        if keep:
            final.append((cls, conf, ybox))
    return final

print("Running inference on test images (this will take time).")
for timg in test_image_paths:
    stem = Path(timg).stem
    dets = infer_with_tta(trained_model, timg, imgsz=imgsz, conf=0.001, iou=0.45, tta=True)
    # format prediction string
    parts = []
    for cls, conf, ybox in dets:
        if conf < FINAL_CONF_THR:
            continue
        parts += [str(int(cls)), f"{float(conf):.3f}", f"{ybox[0]:.6f}", f"{ybox[1]:.6f}", f"{ybox[2]:.6f}", f"{ybox[3]:.6f}"]
    pred_str = " ".join(parts) if parts else "no box"
    sub_rows.append({"image_id": stem, "prediction_string": pred_str})

# Build and save submission
submission_df = pd.DataFrame(sub_rows)
all_test_stems = [Path(p).stem for p in test_image_paths]
existing_stems = set(submission_df['image_id'].tolist())
for s in all_test_stems:
    if s not in existing_stems:
        submission_df = submission_df.append({"image_id": s, "prediction_string": "no box"}, ignore_index=True)

submission_df = submission_df[["image_id","prediction_string"]]
submission_csv = "submission.csv"
submission_df.to_csv(submission_csv, index=False)
print("Submission saved to", submission_csv)
print(submission_df.head(10))

