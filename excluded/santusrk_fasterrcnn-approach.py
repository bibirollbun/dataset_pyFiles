import pandas as pd
import numpy as np
import os
import yaml
from PIL import Image
from pathlib import Path
import torch
import warnings
!python -c "import monai" || pip install -q "monai-weekly[pillow,tqdm]"
warnings.filterwarnings("ignore")

data_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
train_label = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
yolo_dir = '/kaggle/input/phase1-byu'
yolo_train_images = os.path.join(yolo_dir,'images/train')
yolo_train_labels = os.path.join(yolo_dir,'labels/train')
yolo_val_images = os.path.join(yolo_dir,'images/val')
yolo_val_labels = os.path.join(yolo_dir,'labels/val')
yaml_path = os.path.join(yolo_dir,'dataset.yaml')
with open(yaml_path,'r') as file:
    yaml_data = yaml.safe_load(file)

if 'path' in yaml_data:
    yaml_data['path'] = yolo_dir

fixed_yaml_path = "/kaggle/working/fixed_dataset.yaml"
with open(fixed_yaml_path, 'w') as f:
    yaml.dump(yaml_data, f)



from torch.utils.data import DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandFlipd,Resized,
    RandZoomd, RandAffined, Compose, ToTensord
)
from monai.data import DataLoader,Dataset
from PIL import Image , ImageDraw
import matplotlib.pyplot as plt
from monai.transforms import Resize
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform
import numpy as np
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

box_size=64

def get_annotated_data(yolo_train_images):
    data = []
    
    for img_file in os.listdir(yolo_train_images):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
            continue

        img_path = os.path.join(yolo_train_images, img_file)
        img=Image.open(img_path)
        #resize_img = img.resize((640, 640), Image.Resampling.LANCZOS)
        width, height = img.size

        # Extract x and y values directly using split and list comprehension
        parts = img_file.split('_')
        y_centre = int([p[1:] for p in parts if p.startswith('y')][0])
        x_centre = int([p.split('.')[0][1:] for p in parts if p.startswith('x')][0])
       

        half_w, half_h = box_size / 2, box_size / 2

        # Ensure coordinates are within bounds
        x1, x2 = max(0, x_centre - half_w), min(width, x_centre + half_w)
        y1, y2 = max(0, y_centre - half_h), min(height, y_centre + half_h)

        tomo_id = f'tomo_{parts[1]}'
        motors = train_label[train_label['tomo_id']==tomo_id]['Number of motors'].unique()
        if motors[0] > 0:
            label=1
        else:
            label=0

        data.append({
            "image": img_path,
            "box": [[x1, y1, x2, y2]],
            'labels': label
        })

    return data



class ResizeWithBBox(MapTransform):
    def __init__(self, keys: KeysCollection, target_size, image_key="image", bbox_key="box"):
        super().__init__(keys)
        self.resize = Resize(spatial_size=target_size)
        self.target_size = target_size
        self.image_key = image_key
        self.bbox_key = bbox_key

    def __call__(self, data):
        d = dict(data)
        
        # Original image shape
        original_height, original_width = d[self.image_key].shape[1:]
        new_width, new_height = self.target_size

        # Resize the image
        d[self.image_key] = self.resize(d[self.image_key])

        # Resize bounding boxes
        new_bboxes = []
        for bbox in d[self.bbox_key]:  # bbox format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * new_width / original_width)
            x2 = int(x2 * new_width / original_width)
            y1 = int(y1 * new_height / original_height)
            y2 = int(y2 * new_height / original_height)
            new_bboxes.append([x1, y1, x2, y2])

        d[self.bbox_key] = np.array(new_bboxes)

        return d


class Ensure3Channelsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        
    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        if img.shape[0] == 1:  # grayscale image with shape [1, H, W]
            d["image"] = img.repeat(3, 1, 1)
        return d

class FlagellaMotorDataset(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform =transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,Index):
        item= self.data[Index]
        num_boxes = len(item["box"])
        print(num_boxes)
        labels = torch.tensor([item['labels']] * num_boxes, dtype=torch.int64)
        sample ={
            "image":item['image'],
            "box":torch.tensor(item["box"], dtype=torch.float32),
            'labels':labels
            
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
        
def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # [background, motor]
    return model


# Define all transforms
all_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Ensure3Channelsd(keys=["image"]),  # Add this line
    ResizeWithBBox(keys=["image", "box"], target_size=(512, 512)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image"])
])

# Save final model
model_dir = '/kaggle/working/'
os.makedirs(model_dir, exist_ok=True)


train_df = get_annotated_data(yolo_train_images)
val_df =get_annotated_data(yolo_val_images)


train_data = FlagellaMotorDataset(train_df,all_transforms)
val_data = FlagellaMotorDataset(val_df,all_transforms)
train_loader = DataLoader(train_data,batch_size=5,shuffle=True)
val_loader = DataLoader(val_data,batch_size=2,shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(5):
    for batch in train_loader:     
        images = [img.to(device) for img in batch["image"]]
        targets=[]
        for i in range(len(images)):
            targets.append({
                "boxes": batch["box"][i].to(device),
                "labels": batch["labels"][i].to(device)
            })
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")



model_path = os.path.join(model_dir, 'fasterrcnn_final.pth')
torch.save(model.state_dict(), model_path)



from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

# Initialize model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = get_model().to(device)
model.eval()

# Initialize metric
metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)

# Evaluation loop
with torch.no_grad():
    for batch in train_loader:
        # Prepare input images
        images = [img.to(device) for img in batch["image"]]
        
        # Prepare ground-truth targets in required format
        targets = [{
            "boxes": b.to(device),
            "labels": l.to(device)
        } for b, l in zip(batch["box"], batch["labels"])]
        
        # Get model predictions
        outputs = model(images)

        # Convert predictions and targets to CPU for metric
        outputs_cpu = [{
            "boxes": o["boxes"].cpu(),
            "scores": o["scores"].cpu(),
            "labels": o["labels"].cpu()
        } for o in outputs]

        targets_cpu = [{
            "boxes": t["boxes"].cpu(),
            "labels": t["labels"].cpu()
        } for t in targets]

        # Update metric
        metric.update(outputs_cpu, targets_cpu)

# Final metric results
results = metric.compute()
print(f"\nðŸ“Š Evaluation Results:")
print(f"  ðŸ”¹ mAP@0.5: {results['map_50']:.4f}")
print(f"  ðŸ”¹ Overall mAP@[.5:.95]: {results['map']:.4f}")
print(f"  ðŸ”¹ Class-wise AP: {results['map_per_class']}")



import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image_with_boxes(img, boxes,actual_boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
    for box in boxes:
        print(box)
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for box in actual_boxes:
        print(box)
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()

model.eval()
with torch.no_grad():
    for batch in val_loader:
        images = [img.to(device) for img in batch["image"]]
        targets = [boxes.to("cpu") for boxes in batch["box"]] 
        outputs = model(images)
        for img, out , target in zip(images, outputs,targets):
            show_image_with_boxes(img.cpu(), out["boxes"].cpu(),target)
        break


import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 1. Load Image and Define Target -------------------- #
image_path = "image.png"  # change as needed
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512))  # ensure fixed input size

# Example ground-truth box: (x1, y1, x2, y2)
gt_box = torch.tensor([[150.0, 150.0, 250.0, 250.0]], dtype=torch.float32)
gt_label = torch.tensor([1], dtype=torch.int64)

transform = T.ToTensor()
image_tensor = transform(image).to(device)

# -------------------- 2. Define Model -------------------- #
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
model = model.to(device)
model.train()

# -------------------- 3. Training Setup -------------------- #
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Prepare input and target
inputs = [image_tensor]
targets = [{
    "boxes": gt_box.to(device),
    "labels": gt_label.to(device)
}]

# -------------------- 4. Train for 100 epochs -------------------- #
for epoch in range(100):
    loss_dict = model(inputs, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}: Loss = {losses.item():.4f}")

# -------------------- 5. Evaluate -------------------- #
model.eval()
with torch.no_grad():
    output = model([image_tensor])[0]

# -------------------- 6. Visualize -------------------- #
def show_image_with_boxes(img_tensor, gt_boxes, pred_boxes):
    fig, ax = plt.subplots(1)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)

    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.legend(["Green = GT", "Red = Prediction"])
    plt.show()

# Show results
show_image_with_boxes(image_tensor.cpu(), gt_box.cpu(), output["boxes"].cpu())


