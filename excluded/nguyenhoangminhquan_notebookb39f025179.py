# =========================
# ğŸ”§ Step 1: Convert labels to YOLO format (if needed)
# your raw labels are in format: class xmin ymin xmax ymax
# YOLO expects: class x_center y_center width height (normalized)
# =========================
from pathlib import Path
from PIL import Image

labels_dir = Path("/kaggle/input/a0-2025-object-detection/Dataset/Train/labels")
images_dir = Path("/kaggle/input/a0-2025-object-detection/Dataset/Train/images")
out_dir = Path("/kaggle/working/labels_yolo")
out_dir.mkdir(parents=True, exist_ok=True)

for txt_path in labels_dir.glob("*.txt"):
    img_path = images_dir / (txt_path.stem + ".jpg")
    if not img_path.exists():
        continue
    with Image.open(img_path) as im:
        w, h = im.size

    new_lines = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])  # already 0â€“5
        xmin, ymin, xmax, ymax = map(float, parts[1:])
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    (out_dir / txt_path.name).write_text("\n".join(new_lines))
print("âœ… Labels converted to YOLO format in Train/labels_yolo")


# =========================
# ğŸ“„ Step 2: Create dataset YAML
# =========================
dataset_yaml = """
path: .
train: Train/images
val: Train/images   # if you have a val split, change this path
test: Test/images

names:
  0: Apple
  1: Banana
  2: Grapes
  3: Orange
  4: Pineapple
  5: Watermelon
"""

with open("fruits.yaml", "w") as f:
    f.write(dataset_yaml)
print("âœ… fruits.yaml created")


# =========================
# ğŸš€ Step 3: Train YOLOv8
# =========================
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # start with small model; can switch to yolov8m/l/x for higher accuracy
model.train(
    data="fruits.yaml",
    epochs=100,          # max epochs
    patience=15,         # stop if no improvement for 15 epochs
    imgsz=640,
    batch=16,
    workers=2,
    optimizer="AdamW",   # optional: try "SGD" or "AdamW"
    lr0=1e-3,            # initial learning rate
    lrf=1e-4,            # final learning rate at end of schedule
    lr_scheduler="cosine",  # cosine annealing (default), can try "linear" or "onecycle"
    project="runs/train",
    name="fruits_yolo",
    exist_ok=True
)


# =========================
# ğŸ“Š Step 4: Validate
# =========================
metrics = model.val()
print(metrics)  # includes mAP@0.5


# =========================
# ğŸ”� Step 5: Inference on Test set
# =========================
test_dir = Path("/kaggle/input/a0-2025-object-detection/Dataset/Test/images")
results = model.predict(
    source=str(test_dir),
    imgsz=640,
    conf=0.001,   # low to keep all, will filter later
    iou=0.5,
    save=False,
    save_txt=False,
    save_conf=True
)


# =========================
# ğŸ“� Step 6: Generate submission.csv
# Required format:
# ID,bounding_boxes
# bounding_boxes is a JSON string of [{"x_min":..,"y_min":..,"x_max":..,"y_max":..,"class":..,"confidence":..}, ...]
# =========================
import pandas as pd
import json

out_rows = []
for r in results:
    img_name = Path(r.path).stem   # e.g., "00001"
    # change to "img1" if contest requires "imgN"
    img_id = img_name

    boxes = []
    if hasattr(r, "boxes"):
        for box in r.boxes:
            xyxy = box.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            conf = float(box.conf.tolist()[0])
            cls = int(box.cls.tolist()[0])
            boxes.append({
                "x_min": int(xyxy[0]),
                "y_min": int(xyxy[1]),
                "x_max": int(xyxy[2]),
                "y_max": int(xyxy[3]),
                "class": cls,
                "confidence": round(conf, 4)
            })
    out_rows.append({"ID": img_id, "bounding_boxes": json.dumps(boxes)})

df = pd.DataFrame(out_rows)
df.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv created")
df.head()

