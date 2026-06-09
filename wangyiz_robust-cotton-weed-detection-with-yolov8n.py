# Kaggle kernels may not include ultralytics by default.
# If already installed, this is fast; if not, it installs quietly.

!pip -q install ultralytics



import os
import math
import csv
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO



# Competition dataset may appear as a nested folder in Kaggle input.
# We'll locate dataset.yaml and derive paths from it.

INPUT_ROOT = Path("/kaggle/input")

def find_dataset_root():
    candidates = []
    for p in INPUT_ROOT.rglob("dataset.yaml"):
        candidates.append(p.parent)
    # Prefer the one that has train/val/test folders next to it
    for root in candidates:
        if (root / "train").exists() and (root / "val").exists() and (root / "test").exists():
            return root
    return candidates[0] if candidates else None

DATA_ROOT = find_dataset_root()
print("DATA_ROOT:", DATA_ROOT)

assert DATA_ROOT is not None, "â�Œ Could not find dataset.yaml under /kaggle/input. Please Add Data correctly."
print("Files in DATA_ROOT:", sorted([p.name for p in DATA_ROOT.iterdir()])[:20])



TRAIN_DIR = DATA_ROOT / "train" / "images"
VAL_DIR   = DATA_ROOT / "val" / "images"
TEST_DIR  = DATA_ROOT / "test" / "images"

def count_images(d):
    exts = {".jpg", ".jpeg", ".png"}
    return len([p for p in d.iterdir() if p.suffix.lower() in exts])

print("train images:", count_images(TRAIN_DIR))
print("val images:  ", count_images(VAL_DIR))
print("test images: ", count_images(TEST_DIR))

# The official test should be 170. You repeatedly used "Found test images: 170" as a hard check.
# We'll keep it here as a guardrail.
assert count_images(TEST_DIR) == 170, "â�Œ Test image count is not 170. Dataset may be corrupted or wrong version."
print("âœ… Test image count sanity check passed (170).")



# Minimal label audit:
# - class distribution
# - invalid values check (x,y,w,h in [0,1], w/h>0)

LABEL_DIR = DATA_ROOT / "train" / "labels"

cls_counts = {0:0, 1:0, 2:0}
bad_lines = 0
bad_files = 0

for txt in LABEL_DIR.glob("*.txt"):
    ok_file = True
    with open(txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_lines += 1
                ok_file = False
                continue
            c, x, y, w, h = parts
            try:
                c = int(c); x=float(x); y=float(y); w=float(w); h=float(h)
            except:
                bad_lines += 1
                ok_file = False
                continue
            if c in cls_counts: cls_counts[c] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_lines += 1
                ok_file = False
    if not ok_file:
        bad_files += 1

print("Class counts (train labels):", cls_counts)
print("Bad label lines:", bad_lines)
print("Files with at least one bad line:", bad_files)



MODEL_CANDIDATES = [
    Path("/kaggle/input/cotton-weed-yolov8-best-models/best.pt"),
    # Add more candidates here if you rename your dataset
]

model_path = None
for p in MODEL_CANDIDATES:
    if p.exists():
        model_path = p
        break

if model_path is None:
    print("â�Œ best.pt not found in expected locations.")
    print("ğŸ“Œ /kaggle/input contains:", sorted([x.name for x in INPUT_ROOT.iterdir()])[:50])
    raise FileNotFoundError("Please add your model dataset and ensure best.pt exists.")

print("âœ… Loading model:", model_path)
model = YOLO(str(model_path))
print("Model loaded.")



# Optional: run validation with different conf values.
# Your earlier experiments swept conf around 0.18~0.32 and compared mAP. :contentReference[oaicite:6]{index=6}

CONF_LIST = [0.18, 0.22, 0.25, 0.28, 0.32]
results_summary = []

DATA_YAML = str(DATA_ROOT / "dataset.yaml")

for conf in CONF_LIST:
    print(f"\n===== Val sweep: conf={conf:.2f} =====")
    r = model.val(data=DATA_YAML, imgsz=640, conf=conf, verbose=False)
    # Ultralytics result object contains metrics; we keep it simple and store map50, map
    map50 = float(getattr(r.box, "map50", np.nan))
    map5095 = float(getattr(r.box, "map", np.nan))
    results_summary.append((conf, map50, map5095))

print("\n===== Summary (conf, mAP50, mAP50-95) =====")
for conf, map50, map5095 in results_summary:
    print(f"conf={conf:.2f} -> mAP50={map50:.4f}, mAP50-95={map5095:.4f}")



def xyxy_to_xywhn(xyxy, w, h):
    # xyxy in pixels -> normalized (x_center, y_center, width, height)
    x1, y1, x2, y2 = xyxy
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh

def clip01(v):
    return max(0.0, min(1.0, v))

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_classwise(boxes, scores, classes, iou_thr=0.5):
    # boxes: (N,4) xyxy pixels in original frame
    keep = []
    idxs = np.argsort(-scores)
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        new_rest = []
        for j in rest:
            if classes[j] != classes[i]:
                new_rest.append(j)
                continue
            if iou_xyxy(boxes[i], boxes[j]) < iou_thr:
                new_rest.append(j)
        idxs = np.array(new_rest, dtype=int)
    return keep



def tta_variants(img):
    # Returns list of (variant_name, transformed_img)
    return [
        ("orig", img),
        ("fliplr", cv2.flip(img, 1)),
        ("flipud", cv2.flip(img, 0)),
        ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

def invert_xyxy(xyxy, variant, w, h):
    # Map xyxy (pixels in transformed image) back to original image coordinates
    x1, y1, x2, y2 = xyxy

    if variant == "orig":
        return np.array([x1, y1, x2, y2], dtype=float)

    if variant == "fliplr":
        # x' = w - x
        return np.array([w - x2, y1, w - x1, y2], dtype=float)

    if variant == "flipud":
        return np.array([x1, h - y2, x2, h - y1], dtype=float)

    if variant == "rot90":
        # transformed image dims: (h, w) -> (w, h)
        # rot90 CW: (x, y) in rotated corresponds to (x_orig, y_orig) = (x, y) mapping:
        # Original -> Rot90: (x, y) -> (y, w - x)
        # Inverse: (x_r, y_r) -> (x_o, y_o) = (w - y_r, x_r)
        # For boxes, convert corners:
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=float)
        x_r = pts[:,0]; y_r = pts[:,1]
        x_o = w - y_r
        y_o = x_r
        return np.array([x_o.min(), y_o.min(), x_o.max(), y_o.max()], dtype=float)

    if variant == "rot270":
        # rot270 CCW: Original -> Rot270: (x, y) -> (h - y, x)
        # Inverse: (x_r, y_r) -> (x_o, y_o) = (y_r, h - x_r)
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=float)
        x_r = pts[:,0]; y_r = pts[:,1]
        x_o = y_r
        y_o = h - x_r
        return np.array([x_o.min(), y_o.min(), x_o.max(), y_o.max()], dtype=float)

    raise ValueError("Unknown variant: " + variant)



def predict_tta(img, conf=0.25, iou_thr=0.5):
    h, w = img.shape[:2]
    all_boxes = []
    all_scores = []
    all_cls = []

    for name, im_t in tta_variants(img):
        res = model.predict(im_t, imgsz=640, conf=conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        # boxes.data: (N, 6) => x1,y1,x2,y2,conf,cls
        arr = res.boxes.data.detach().cpu().numpy()
        for x1, y1, x2, y2, sc, cl in arr:
            mapped = invert_xyxy((x1, y1, x2, y2), name, w=w, h=h)
            all_boxes.append(mapped)
            all_scores.append(float(sc))
            all_cls.append(int(cl))

    if len(all_boxes) == 0:
        return np.zeros((0,4), float), np.zeros((0,), float), np.zeros((0,), int)

    all_boxes = np.stack(all_boxes, axis=0)
    all_scores = np.array(all_scores, dtype=float)
    all_cls = np.array(all_cls, dtype=int)

    keep = nms_classwise(all_boxes, all_scores, all_cls, iou_thr=iou_thr)
    return all_boxes[keep], all_scores[keep], all_cls[keep]



def build_prediction_string(boxes_xyxy, scores, classes, w, h):
    if len(boxes_xyxy) == 0:
        return "no box"

    parts = []
    for b, sc, cl in zip(boxes_xyxy, scores, classes):
        xc, yc, bw, bh = xyxy_to_xywhn(b, w=w, h=h)
        # clamp to [0,1] to avoid formatting issues
        xc, yc, bw, bh = map(clip01, [xc, yc, bw, bh])
        parts.extend([
            str(int(cl)),
            f"{float(sc):.4f}",
            f"{xc:.4f}",
            f"{yc:.4f}",
            f"{bw:.4f}",
            f"{bh:.4f}",
        ])
    return " ".join(parts)

OUT_CSV = "submission.csv"
CONF = 0.25
IOU_THR = 0.5

test_images = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
print("Test images:", len(test_images))

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction_string"])

    for p in test_images:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]

        boxes, scores, cls = predict_tta(img, conf=CONF, iou_thr=IOU_THR)
        pred_str = build_prediction_string(boxes, scores, cls, w=w, h=h)

        # image_id in many of your CSVs is without extension; we follow that convention.
        image_id = p.stem
        writer.writerow([image_id, pred_str])

print(f"âœ… Saved: {OUT_CSV}")



# Sanity check: rows should be 171 (header + 170 images)
with open("submission.csv", "r") as f:
    head = [next(f).strip() for _ in range(10)]
print("\n".join(head))

# Count lines
with open("submission.csv", "r") as f:
    n_lines = sum(1 for _ in f)
print("Total lines:", n_lines, "(expected 171)")
assert n_lines == 171, "â�Œ submission.csv line count mismatch."
print("âœ… submission.csv row count correct.")





