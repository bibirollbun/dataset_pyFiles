import pandas as pd
import os, math, random


DATA_DIR = "/kaggle/input/global-wheat-detection"  
IMG_DIR = os.path.join(DATA_DIR, "train")          
CSV_PATH = os.path.join(DATA_DIR, "train.csv")     

# Load annotations
df = pd.read_csv(CSV_PATH)
df.head()



if 'bbox' in df.columns:
    # Remove brackets and split into columns
    bbox_coords = df['bbox'].str.strip('[]').str.split(',', expand=True).astype(float)
    df[['x', 'y', 'w', 'h']] = bbox_coords



YOLO_IMAGES_TRAIN = "images/train"
YOLO_IMAGES_VAL = "images/val"
YOLO_LABELS_TRAIN = "labels/train"
YOLO_LABELS_VAL = "labels/val"
os.makedirs(YOLO_IMAGES_TRAIN, exist_ok=True)
os.makedirs(YOLO_IMAGES_VAL, exist_ok=True)
os.makedirs(YOLO_LABELS_TRAIN, exist_ok=True)
os.makedirs(YOLO_LABELS_VAL, exist_ok=True)


image_ids = df['image_id'].unique().tolist()
random.shuffle(image_ids)
val_frac = 0.2
val_count = math.floor(len(image_ids) * val_frac)
val_ids = set(image_ids[:val_count])
train_ids = set(image_ids[val_count:])


groups = df.groupby('image_id')



import os
from PIL import Image
import shutil

for img_id in image_ids:
    # Determine split
    img_split = "val" if img_id in val_ids else "train"

    # Image file path (.jpg or .png)
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
    
    # Skip if image doesn't exist
    if not os.path.exists(img_path):
        print(f"Image {img_id} not found in either .jpg or .png format.")
        continue

    # Destination paths
    if img_split == "train":
        dest_img_path = os.path.join(YOLO_IMAGES_TRAIN, f"{img_id}.jpg")
        dest_label_path = os.path.join(YOLO_LABELS_TRAIN, f"{img_id}.txt")
    else:
        dest_img_path = os.path.join(YOLO_IMAGES_VAL, f"{img_id}.jpg")
        dest_label_path = os.path.join(YOLO_LABELS_VAL, f"{img_id}.txt")

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)

    # Copy image
    shutil.copyfile(img_path, dest_img_path)

    # Write YOLO label
    with open(dest_label_path, 'w') as f:
        if img_id in groups.groups:  # has annotations
            for _, row in groups.get_group(img_id).iterrows():
                x, y, w, h = row['x'], row['y'], row['w'], row['h']
                img = Image.open(img_path)
                W, H = img.size
                x_center = (x + w / 2) / W
                y_center = (y + h / 2) / H
                bw = w / W
                bh = h / H
                # Clamp between 0 and 1
                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)
                bw = min(max(bw, 0), 1)
                bh = min(max(bh, 0), 1)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")


!pip install albumentations==1.2.1


import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    # Note: Mosaic is not trivial to include in Compose directly; YOLO does it internally.
    # We could implement a custom mosaic outside the pipeline if needed.
    ToTensorV2()  # Convert image to PyTorch tensor
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


import cv2
import matplotlib.pyplot as plt

# Load a sample image and corresponding bboxes (assuming we have one from our dataset)
sample_img_path = os.path.join(YOLO_IMAGES_TRAIN, os.listdir(YOLO_IMAGES_TRAIN)[0])
sample_label_path = os.path.join(YOLO_LABELS_TRAIN, os.path.splitext(os.listdir(YOLO_IMAGES_TRAIN)[0])[0] + ".txt")
sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)  # convert to RGB for plotting

# Read sample bboxes in YOLO format
sample_bboxes = []
sample_class_labels = []
with open(sample_label_path, 'r') as f:
    for line in f:
        cls, x_center, y_center, w, h = line.strip().split()
        x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
        # Albumentations expects bboxes as [x_min, y_min, x_max, y_max] for pascal_voc or [xc,yc,w,h] for yolo
        # We already have normalized YOLO format and specified format='yolo', so we can use directly:
        sample_bboxes.append([x_center, y_center, w, h])
        sample_class_labels.append(int(cls))

# Apply augmentation
augmented = train_transforms(image=sample_img, bboxes=sample_bboxes, class_labels=sample_class_labels)
aug_img = augmented['image'].cpu().numpy().transpose(1,2,0)  # tensor to numpy image
aug_bboxes = augmented['bboxes']

# Plot original and augmented for comparison
def plot_image_with_bboxes(img, bboxes, title="Image"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    H, W = img.shape[0], img.shape[1]
    for (xc, yc, bw, bh) in bboxes:
        # Convert normalized center format to top-left corner format for drawing
        x_min = int((xc - bw/2) * W); x_max = int((xc + bw/2) * W)
        y_min = int((yc - bh/2) * H); y_max = int((yc + bh/2) * H)
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                              fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

plot_image_with_bboxes(sample_img, sample_bboxes, title="Original")
plot_image_with_bboxes(aug_img, aug_bboxes, title="Augmented")


import cv2
import matplotlib.pyplot as plt

# Load a sample image and corresponding bboxes (assuming we have one from our dataset)
sample_img_path = os.path.join(YOLO_IMAGES_TRAIN, os.listdir(YOLO_IMAGES_TRAIN)[0])
sample_label_path = os.path.join(YOLO_LABELS_TRAIN, os.path.splitext(os.listdir(YOLO_IMAGES_TRAIN)[0])[0] + ".txt")
sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)  # convert to RGB for plotting

# Read sample bboxes in YOLO format
sample_bboxes = []
sample_class_labels = []
with open(sample_label_path, 'r') as f:
    for line in f:
        cls, x_center, y_center, w, h = line.strip().split()
        x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
        # Albumentations expects bboxes as [x_min, y_min, x_max, y_max] for pascal_voc or [xc,yc,w,h] for yolo
        # We already have normalized YOLO format and specified format='yolo', so we can use directly:
        sample_bboxes.append([x_center, y_center, w, h])
        sample_class_labels.append(int(cls))

# Apply augmentation
augmented = train_transforms(image=sample_img, bboxes=sample_bboxes, class_labels=sample_class_labels)
aug_img = augmented['image'].cpu().numpy().transpose(1,2,0)  # tensor to numpy image
aug_bboxes = augmented['bboxes']

# Plot original and augmented for comparison
def plot_image_with_bboxes(img, bboxes, title="Image"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    H, W = img.shape[0], img.shape[1]
    for (xc, yc, bw, bh) in bboxes:
        # Convert normalized center format to top-left corner format for drawing
        x_min = int((xc - bw/2) * W); x_max = int((xc + bw/2) * W)
        y_min = int((yc - bh/2) * H); y_max = int((yc + bh/2) * H)
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                              fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

plot_image_with_bboxes(sample_img, sample_bboxes, title="Original")
plot_image_with_bboxes(aug_img, aug_bboxes, title="Augmented")



path:  ./   # base path for the dataset (we'll assume our images and labels directories are in working dir)
train: images/train  
val: images/val  
names: [ "wheat_head" ]  
nc: 1  


dataset_yaml = """
path: ./
train: images/train
val: images/val
names: ['wheat_head']
nc: 1
"""
with open("wheat.yaml", "w") as f:
    f.write(dataset_yaml)



!pip install -q ultralytics>=8.0.20  # install Ultralytics YOLO

from ultralytics import YOLO

# Load a YOLO model (pretrained on COCO)
model = YOLO('yolov8s.pt')  # using a small model; can also try 'yolov8m.pt' or others



results = model.train(
    data="wheat.yaml",  # dataset config
    epochs=50,
    batch=8,
    imgsz=640,
    project="wheat-yolo-training",  # directory to save runs
    name="yolov8s-globalwheat",
    pretrained=True,   # start from COCO-pretrained weights (True by default when we loaded yolov8s.pt)
    verbose=True
)




best_model = YOLO("wheat-yolo-training/yolov8s-globalwheat/weights/best.pt")
metrics = best_model.val(data="wheat.yaml", split="val") val_images = [os.path.join(YOLO_IMAGES_VAL, img) for img in os.listdir(YOLO_IMAGES_VAL)[:5]]
predictions = best_model.predict(val_images, conf=0.25, imgsz=640)  
for i, result in enumerate(predictions):
    boxes = result.boxes.xyxy.cpu().numpy()  # predicted bounding boxes in [x1,y1,x2,y2] format
    scores = result.boxes.conf.cpu().numpy()  # confidence scores
    img = result.orig_img  # original image array
    # Plot the image with boxes
    plt.figure(figsize=(6,4))
    plt.imshow(img)
    for (x1,y1,x2,y2), score in zip(boxes, scores):
        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='lime', facecolor='none', linewidth=2))
        plt.gca().text(x1, y1-5, f"{score:.2f}", color='lime', fontsize=8)
    plt.title(f"Predictions on val image {i}")
    plt.axis('off')
    plt.show()



print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50:95: {metrics.box.map:.4f}")



import torch
# Get the underlying PyTorch model
model_pt = best_model.model

# Choose a layer to visualize (e.g., the last layer of the backbone or neck)
target_layer = model_pt.model[ -2 ]  # just as an example, choose second to last module in model
activations = {}

# Hook to capture the feature map from target layer
def hook_fn(module, input, output):
    activations["feat"] = output.detach()

hook_handle = target_layer.register_forward_hook(hook_fn)

# Pass an image through the model
img = cv2.imread(val_images[0])
out = best_model.predict(img, verbose=False)  # run prediction (forward pass)
feature_map = activations["feat"]
print(feature_map.shape)  # e.g., (1, C, H, W)

# Remove hook
hook_handle.remove()



import numpy as np
feat = feature_map.cpu().squeeze(0)  
# Mean across channels
heatmap = feat.mean(axis=0).numpy()
# Normalize heatmap
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.4)  
plt.title("Feature activation heatmap")
plt.axis('off')
plt.show()



!pip install pytorch-grad-cam


!pip install pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Suppose we want Grad-CAM for the first detection in the image
# We identify the layer (target_layer as defined) and the target category (which for detection might be the objectness of class 0)
cam = GradCAM(model=model_pt, target_layers=[target_layer], use_cuda=True)
# We need to set up a target for the CAM - this is tricky for detection, but one way is to pick the output neuron corresponding to the highest confidence wheat head detection.
# For simplicity, assume we target class 0 in the network (wheat_head class probability).
targets = [ClassifierOutputTarget(0)]
grayscale_cam = cam(input_tensor=torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).cuda().float()/255.0, targets=targets)
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(img/255.0, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.title("Grad-CAM for wheat head detection")
plt.axis('off')
plt.show()



!pip install ensemble-boxes
from ensemble_boxes import weighted_boxes_fusion

# Suppose preds1 and preds2 are results from original and flipped image
boxes_list = [preds1_boxes_xy, preds2_boxes_xy]  # each list of [ [x1,y1,x2,y2], ... ] normalized 0-1
scores_list = [preds1_scores, preds2_scores]
labels_list = [preds1_labels, preds2_labels]  # all zeros in our case (class 0)
# Apply WBF
refined_boxes, refined_scores, refined_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.001
)

