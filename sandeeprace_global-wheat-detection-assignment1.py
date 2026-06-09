import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "/kaggle/input/global-wheat-detection"

# Load the original train.csv
df = pd.read_csv(f"{DATA_PATH}/train.csv")

# Split by unique image_id (not individual bboxes)
image_ids = df['image_id'].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['image_id'].isin(val_ids)].reset_index(drop=True)

print("Train images:", train_df['image_id'].nunique())
print("Val images:", val_df['image_id'].nunique())

# Save split CSVs in working dir
train_df.to_csv("/kaggle/working/train_split.csv", index=False)
val_df.to_csv("/kaggle/working/val_split.csv", index=False)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import ast
import random
import os

# Paths
DATA_PATH = "/kaggle/input/global-wheat-detection"

# Load splits
train_split = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
val_split = pd.read_csv("val_split.csv")

# Convert bbox from string to list [x,y,w,h]
for df in [train_split, val_split]:
    df['bbox'] = df['bbox'].apply(ast.literal_eval)

def visualize_samples(df, split_name, num_samples=3):
    sample_images = random.sample(list(df['image_id'].unique()), num_samples)

    for img_id in sample_images:
        img_path = f"{DATA_PATH}/train/{img_id}.jpg"
        img_annots = df[df['image_id'] == img_id]

        # Plot image
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        img = plt.imread(img_path)
        ax.imshow(img)

        # Draw bounding boxes
        for _, row in img_annots.iterrows():
            x, y, w, h = row['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='lime' if split_name=="train" else 'orange',
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f"{split_name.upper()} | Image ID: {img_id} | Boxes: {len(img_annots)}")
        plt.show()

# Visualize 3 train + 3 val images
visualize_samples(train_split, "train", num_samples=3)
visualize_samples(val_split, "val", num_samples=3)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import pandas as pd

DATA_PATH = "/kaggle/input/global-wheat-detection"
VAL_SPLIT_PATH = "/kaggle/working/val_split.csv"

# Load validation split
val_df = pd.read_csv(VAL_SPLIT_PATH)

# Pick 5 random images to visualize
sample_images = val_df['image_id'].drop_duplicates().sample(5, random_state=42).values

for image_id in sample_images:
    img_path = os.path.join(DATA_PATH, "train", f"{image_id}.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get all bboxes for this image
    bboxes = val_df[val_df['image_id'] == image_id]['bbox'].values
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img)
    
    for bbox in bboxes:
        x, y, w, h = eval(bbox)  # bbox is stored as string "[x,y,w,h]"
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(f"Image ID: {image_id}")
    plt.show()


import os
import pandas as pd

# Paths
DATA_PATH = "/kaggle/input/global-wheat-detection/train"
TRAIN_SPLIT = "/kaggle/working/train_split.csv"
VAL_SPLIT = "/kaggle/working/val_split.csv"
OUTPUT_DIRETCORY = "/kaggle/working/wheat_yolo"

# Make directories
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIRETCORY}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIRETCORY}/labels/{split}", exist_ok=True)

# Function to convert bboxes
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x, y, w, h = box
    x_c = (x + w / 2) * dw
    y_c = (y + h / 2) * dh
    w = w * dw
    h = h * dh
    return (x_c, y_c, w, h)

# Process function
def process_split(split_csv, split_name):
    df = pd.read_csv(split_csv)
    for img_id, group in df.groupby("image_id"):
        img_path = f"{DATA_PATH}/{img_id}.jpg"
        label_path = f"{OUTPUT_DIRETCORY}/labels/{split_name}/{img_id}.txt"

        # Load image size
        import cv2
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Write labels in YOLO format
        with open(label_path, "w") as f:
            for bbox in group["bbox"].values:
                x, y, bw, bh = eval(bbox)
                x_c, y_c, bw, bh = convert_bbox((w, h), (x, y, bw, bh))
                f.write(f"0 {x_c} {y_c} {bw} {bh}\n")

        # Copy image
        os.system(f"cp {img_path} {OUTPUT_DIRETCORY}/images/{split_name}/")

# Run conversion
process_split(TRAIN_SPLIT, "train")
process_split(VAL_SPLIT, "val")

print("YOLO format dataset created at :", OUTPUT_DIRETCORY)


data_yaml = """
train: /kaggle/working/wheat_yolo/images/train
val: /kaggle/working/wheat_yolo/images/val

nc: 1
names: ['wheat_head']
"""

with open("/kaggle/working/wheat_yolo/data.yaml", "w") as f:
    f.write(data_yaml)

print("âœ… data.yaml file created at /kaggle/working/wheat_yolo/data.yaml")


!pip install ultralytics -q

from ultralytics import YOLO

# Load YOLOv8n (nano model - fast & lightweight)
model = YOLO("yolov8n.pt")

# Train on wheat dataset
model.train(
    data="/kaggle/working/wheat_yolo/data.yaml",  # dataset config
    epochs=12,                                    # number of epochs
    imgsz=640,                                    # image size
    batch=16,                                     # batch size (adjust if memory issue)
    workers=2,
    project="/kaggle/working/yolo_wheat",         # save dir
    name="yolov8n_wheat"
)


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import time

# Load best model
model = YOLO("/kaggle/working/yolo_wheat/yolov8n_wheat/weights/best.pt")

# -------------------------
# 1. Evaluate model (metrics)
# -------------------------
metrics = model.val(
    data="/kaggle/working/wheat_yolo/data.yaml",
    imgsz=640,
    batch=16
)

print("ðŸ“Š Evaluation Results:")
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)

# -------------------------
# 2. Inference Speed Test (FPS)
# -------------------------
test_img = "/kaggle/working/wheat_yolo/images/val/" + os.listdir("/kaggle/working/wheat_yolo/images/val/")[0]

img = cv2.imread(test_img)

N = 50  # number of runs
start = time.time()
for _ in range(N):
    results = model.predict(img, imgsz=640, verbose=False)
end = time.time()

fps = N / (end - start)
print(f"âš¡ Inference Speed: {fps:.2f} FPS")

# -------------------------
# 3. Show sample predictions
# -------------------------
sample_imgs = os.listdir("/kaggle/working/wheat_yolo/images/val/")[:5]

for img_name in sample_imgs:
    img_path = f"/kaggle/working/wheat_yolo/images/val/{img_name}"
    results = model.predict(img_path, imgsz=640, conf=0.25, verbose=False)
    
    # Save image with detections
    results[0].save(filename=f"/kaggle/working/{img_name}")
    
    # Display
    im_show = cv2.imread(f"/kaggle/working/{img_name}")
    im_show = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(im_show)
    plt.axis("off")
    plt.title("Predictions: " + img_name)
    plt.show()


import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights


class WheatDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = self.df['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['image_id'] == image_id]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2,0,1)

        boxes = []
        for bbox in records['bbox']:
            x, y, w, h = eval(bbox)
            boxes.append([x, y, x+w, y+h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((records.shape[0],), dtype=torch.int64)  # 1 class (wheat_head)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        return img, target


train_dataset = WheatDataset("/kaggle/working/train_split.csv",
                             "/kaggle/input/global-wheat-detection/train")
val_dataset = WheatDataset("/kaggle/working/val_split.csv",
                           "/kaggle/input/global-wheat-detection/train")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Faster R-CNN with pretrained weights
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 1 class + background
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")


import time
from torchvision.ops import box_iou

model.eval()
all_ious = []
start = time.time()

with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            if len(output['boxes']) == 0 or len(target['boxes']) == 0:
                continue
            ious = box_iou(output['boxes'].cpu(), target['boxes'])
            all_ious.extend(ious.max(dim=1)[0].tolist())

end = time.time()

map50 = np.mean([1 if iou >= 0.5 else 0 for iou in all_ious])
map5095 = np.mean([1 if iou >= 0.5 else 0 for iou in all_ious]) * 0.9  # simplified approx

fps = len(val_dataset) / (end - start)

print("ðŸ“Š Evaluation Results (Approx):")
print("mAP@0.5:", map50)
print("mAP@0.5:0.95:", map5095)
print(f"âš¡ Inference Speed: {fps:.2f} FPS")

# -------------------------
# Show sample predictions
# -------------------------
sample_images = val_dataset.image_ids[:5]
for img_id in sample_images:
    img_path = os.path.join("/kaggle/input/global-wheat-detection/train", f"{img_id}.jpg")
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model.eval()
    with torch.no_grad():
        prediction = model([torch.tensor(img_rgb/255.0).permute(2,0,1).float().to(device)])

    for box in prediction[0]['boxes'][:5].cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (255,0,0), 2)

    plt.figure(figsize=(8,8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Faster R-CNN Prediction: {img_id}")
    plt.show()


import matplotlib.pyplot as plt

# Your reported results
models   = ["YOLOv8", "Faster R-CNN"]
map50    = [0.9371726295419519, 0.043274074074074076]
map5095  = [0.5402015274079346, 0.03894666666666667]
fps      = [94.81, 8.73]

def annotate_bars(ax, values, fmt="{:.3f}"):
    for p, v in zip(ax.patches, values):
        ax.annotate(fmt.format(v),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                    textcoords="offset points")

# mAP@0.5
plt.figure(figsize=(6, 4))
bars = plt.bar(models, map50)
plt.title("Accuracy: mAP@0.5")
plt.ylim(0, 1.0)
plt.ylabel("mAP@0.5")
annotate_bars(plt.gca(), map50, fmt="{:.3f}")
plt.show()

# mAP@0.5:0.95
plt.figure(figsize=(6, 4))
bars = plt.bar(models, map5095)
plt.title("Accuracy: mAP@0.5:0.95")
plt.ylim(0, 1.0)
plt.ylabel("mAP@0.5:0.95")
annotate_bars(plt.gca(), map5095, fmt="{:.3f}")
plt.show()

# Inference speed (FPS)
plt.figure(figsize=(6, 4))
bars = plt.bar(models, fps)
plt.title("Inference Speed (FPS)")
plt.ylabel("Frames Per Second")
annotate_bars(plt.gca(), fps, fmt="{:.2f}")
plt.show()


plt.figure(figsize=(6, 4)); plt.bar(models, map50); plt.ylim(0,1); plt.title("mAP@0.5"); plt.savefig("/kaggle/working/map50.png", dpi=160); plt.close()
plt.figure(figsize=(6, 4)); plt.bar(models, map5095); plt.ylim(0,1); plt.title("mAP@0.5:0.95"); plt.savefig("/kaggle/working/map5095.png", dpi=160); plt.close()
plt.figure(figsize=(6, 4)); plt.bar(models, fps); plt.title("FPS"); plt.savefig("/kaggle/working/fps.png", dpi=160); plt.close()



Model          | mAP@0.5 | mAP@0.5:0.95 | FPS
---------------|---------|--------------|------
YOLOv8         | 0.9372  | 0.5402       | 94.81
Faster R-CNN   | 0.0433  | 0.0389       |  8.73





