# Install necessary packages
!pip install -q ultralytics ensemble-boxes

# Import standard libraries
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# Import PyTorch and YOLO
import torch
from ultralytics import YOLO

# Import ensembling tools
from ensemble_boxes import weighted_boxes_fusion

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("✅ Dependencies loaded and random seeds set.")


# Define base paths
base1 = Path('/kaggle/input/falcon-multiclass-cheerios-soup/falcon-multiclass-cheerios-soup')
base2 = Path('/kaggle/input/falcon-multiclass-cheerios-soupv2/falcon-multiclass-cheerios-soupV2')

# Print data information
print("Original Falcon Dataset:")
for split in ['train', 'val']:
    imgs = list((base1 / split / 'images').glob('*.*'))
    labs = list((base1 / split / 'labels').glob('*.txt'))
    print(f"  {split}: {len(imgs)} images, {len(labs)} labels")

for i in range(1, 7):
    d = base2 / f"Scenario{i}"
    t_imgs = len(list((d / 'train' / 'images').glob('*.*')))
    v_imgs = len(list((d / 'val' / 'images').glob('*.*')))
    print(f"Scenario {i}: train={t_imgs}, val={v_imgs}")


import yaml
from pathlib import Path

# Define mount points
falcon_mount = Path('/kaggle/input/falcon-multiclass-cheerios-soup/falcon-multiclass-cheerios-soup')
v2_mount = Path('/kaggle/input/falcon-multiclass-cheerios-soupv2/falcon-multiclass-cheerios-soupV2')
chal_mount = Path('/kaggle/input/multi-class-object-detection-challenge/Starter_Dataset')

# Auto-discover original Falcon root
train_img_dirs = list(falcon_mount.glob('**/train/images'))
if not train_img_dirs:
    raise FileNotFoundError(f"No nested train/images under {falcon_mount}")
falcon_root = train_img_dirs[0].parents[1]

# Auto-discover V2 root
v2_img_dirs = list(v2_mount.glob('**/Scenario1/train/images'))
if not v2_img_dirs:
    raise FileNotFoundError(f"No Scenario1/train/images under {v2_mount}")
v2_root = v2_img_dirs[0].parents[2]

# Build train list
train_dirs = [
    str(falcon_root / 'train' / 'images'),
    str(falcon_root / 'val' / 'images')
]
for i in range(1, 7):
    train_dirs += [
        str(v2_root / f"Scenario{i}" / 'train' / 'images'),
        str(v2_root / f"Scenario{i}" / 'val' / 'images')
    ]

# Use competition val folder for validation
val_dirs = [
    str(chal_mount / 'val' / 'images')
]

# Test set
test_dir = str(chal_mount / 'testImages' / 'images')

# Assemble and write YAML
data = {
    'train': train_dirs,
    'val': val_dirs,
    'test': test_dir,
    'nc': 2,
    'names': ['cheerios', 'soup']
}

with open('data.yaml', 'w') as f:
    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

print("✅ data.yaml created:")
print(open('data.yaml').read())


from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define roots
orig_root = Path('/kaggle/input/falcon-multiclass-cheerios-soup/falcon-multiclass-cheerios-soup')
v2_root = Path('/kaggle/input/falcon-multiclass-cheerios-soupv2/falcon-multiclass-cheerios-soupV2')

# Build list of (images, labels) for original + each Scenario
train_pairs = [
    (orig_root / 'train' / 'images', orig_root / 'train' / 'labels'),
]
for i in range(1, 7):
    scen = v2_root / f'Scenario{i}'
    train_pairs += [
        (scen / 'train' / 'images', scen / 'train' / 'labels'),
        (scen / 'val' / 'images', scen / 'val' / 'labels'),
    ]

# Gather all (img_path, lbl_path) tuples
all_samples = []
for img_dir, lbl_dir in train_pairs:
    if img_dir.exists() and lbl_dir.exists():
        for img_path in img_dir.glob('*.*'):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            all_samples.append((img_path, lbl_path))

# Sample a few examples
num_to_show = 6
sampled = random.sample(all_samples, k=min(len(all_samples), num_to_show))

classes = ['cheerios', 'soup']

# Plot
for img_path, lbl_path in sampled:
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                cls_id, x_c, y_c, w, h = map(float, line.split())
                img_w, img_h = img.size
                x1 = (x_c - w / 2) * img_w
                y1 = (y_c - h / 2) * img_h
                rect = patches.Rectangle((x1, y1), w * img_w, h * img_h,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, classes[int(cls_id)],
                        color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.5, pad=0.5))
    ax.axis('off')
    plt.show()


from ultralytics import YOLO

# Toggle training on/off
TRAIN = True

if TRAIN:
    model = YOLO("yolov8x.pt")  # or your custom checkpoint

    model.train(
        data="data.yaml",
        epochs=75,
        batch=16,
        imgsz=512,
        optimizer="SGD",
        lr0=0.002,
        lrf=0.0001,
        weight_decay=0.0001,
        dropout=0.3,
        dfl=0.75,
        cos_lr=True,
        patience=100,
        save_period=10,
        project="runs/train",
        exist_ok=True,
        plots=True,
        augment=True,
        mosaic=1.0,
        mixup=0.25,
        cutmix=0.25,
        copy_paste=0.05,
        close_mosaic=10,
        hsv_h=0.05, hsv_s=1.0, hsv_v=0.75,
        flipud=0.1, fliplr=0.6,
        translate=0.1, scale=0.6, shear=0.02,
        warmup_epochs=5, warmup_momentum=1,
        workers=4,
        conf=0.25, iou=0.5
    )


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Point to your training run directory
exp_dir = Path("/kaggle/working/runs/train/train")  # adjust if different
results_csv = exp_dir / "results.csv"

# Load the per-epoch metrics
results = pd.read_csv(results_csv)

# Plot validation losses over epochs
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

# Display the confusion matrix image(s)
cm_paths = [
    exp_dir / "results.png",
    exp_dir / "confusion_matrix.png"
]
for p in cm_paths:
    if p.exists():
        img = Image.open(p)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"No file at {p}")


import csv
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def filter_invalid_boxes(boxes, scores, labels):
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if abs(b[2] - b[0]) > 1e-6 and abs(b[3] - b[1]) > 1e-6:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels

def run_inference(models, image_sizes, test_images_path, conf=0.25, iou_thr=0.5):
    test_dir = Path(test_images_path)
    image_paths = [p for p in test_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    predictions = {}

    for mi, model in enumerate(models):
        model.eval()
        predictions[mi] = {}
        for size in image_sizes:
            predictions[mi][size] = {}
            for img_path in image_paths:
                image_id = img_path.stem
                img = Image.open(img_path)
                w, h = img.size

                # Run predict
                results = model.predict(source=str(img_path),
                                        conf=conf, iou=iou_thr,
                                        imgsz=size, augment=True,
                                        verbose=False)
                # Collect all boxes for this image/size
                boxes, scores, labels = [], [], []
                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes.xyxy.cpu().numpy().tolist()
                    scores = r.boxes.conf.cpu().numpy().tolist()
                    labels = r.boxes.cls.cpu().numpy().tolist()

                # Normalize & filter
                norm_boxes = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in boxes]
                norm_boxes, scores, labels = filter_invalid_boxes(norm_boxes, scores, labels)

                predictions[mi][size][image_id] = {"boxes": norm_boxes,
                                                  "scores": scores,
                                                  "labels": labels}
    return predictions

# Specify your checkpoints and test folder
model_paths = [
    '/kaggle/input/3lc-yolo-baseline-submission/Duality-3LC-Kaggle/run-1/weights/best.pt',
    '/kaggle/working/runs/train/train/weights/best.pt']

test_images_path = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'

# Load models
models = [YOLO(p) for p in model_paths]

# Run batched, multi-scale inference
image_sizes = [640, 800, 864]
predictions = run_inference(models, image_sizes, test_images_path,
                            conf=0.05, iou_thr=0.35)

# Collect all image_ids
first_model = predictions[next(iter(predictions))]
first_scale = first_model[next(iter(first_model))]
image_ids = list(first_scale.keys())

# Fuse & build submission rows
rows = []
for img_id in image_ids:
    all_boxes, all_scores, all_labels = [], [], []
    for model_preds in predictions.values():
        for size_preds in model_preds.values():
            if img_id in size_preds:
                p = size_preds[img_id]
                if p['boxes']:
                    all_boxes.append(p['boxes'])
                    all_scores.append(p['scores'])
                    all_labels.append(p['labels'])
    if all_boxes:
        fb, fs, fl = weighted_boxes_fusion(all_boxes, all_scores, all_labels,
                                           iou_thr=0.35, skip_box_thr=0.01)
        pred_str = " ".join(
            f"{int(l)} {s:.6f} {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {(b[2] - b[0]):.6f} {(b[3] - b[1]):.6f}"
            for b, s, l in zip(fb, fs, fl)
        )
    else:
        pred_str = "no boxes"
    rows.append({'image_id': img_id, 'prediction_string': pred_str})

# Save to CSV
submission_df = pd.DataFrame(rows)
submission_df.to_csv('submission.csv', index=False, quoting=csv.QUOTE_MINIMAL)
print(f"✅ submission.csv created with {len(submission_df)} entries.")

