import torch
import torchvision
import numpy as np
import pandas as pd
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2


from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim

import cv2


train_csv = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")


train_csv.head()


train_csv.info()


train_csv.groupby("image_id")[["width"]].apply("count").hist(figsize=(10,5), bins=35)
plt.title("Number of BBoxes")


len(train_csv.image_id.unique())


import ast
type(ast.literal_eval(train_csv.iloc[0, 3]))


import ast, os
# map image_id → list of bboxes
bbox_dict = (
    train_csv
    .assign(bbox=lambda df: df.bbox.map(ast.literal_eval))
    .groupby("image_id")
    .bbox
    .apply(list)
    .to_dict()
)
# list of all ids
image_ids = list(bbox_dict.keys())
train_dir  = "/kaggle/input/global-wheat-detection/train"




def visualize(image_id):
    if image_id not in bbox_dict:
        print(f"{image_id} not found")
        return
    path = os.path.join(train_dir, f"{image_id}.jpg")
    img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    for bbox in bbox_dict[image_id]:
        xmin, ymin, w, h = map(int, bbox)
        cv2.rectangle(img, (xmin,ymin), (xmin+w, ymin+h), (255,0,0), 2)
    plt.imshow(img); plt.axis("off")

visualize("00764ad5d")


class CustomDataset(Dataset):
    def __init__(self, imgs_path, annot_dict, transform=None):
        self.imgs_path = imgs_path
        self.annot_dict = annot_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.annot_dict.keys())

    def __getitem__(self, idx):

        image_id = list(self.annot_dict.keys())[idx]
        boxes = np.array(self.annot_dict[image_id], dtype=np.float32)

        # 2. convert to x1,y1,x2,y2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = xmin + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y2 = ymin + height
        
        image_path = os.path.join(self.imgs_path, f"{image_id}.jpg")
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = {
            "labels": torch.ones((boxes.shape[0]), dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float64)
        }
        return img, target

def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)




train_dataset = CustomDataset(train_dir, bbox_dict, transform=None)
train_dataloader = DataLoader(train_dataset,  
                              batch_size=1, 
                              shuffle=True,
                             collate_fn = detection_collate,
                             pin_memory = True if torch.cuda.is_available() else False)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0]
label = train_labels[0]
print(label["labels"].shape)
print(label["boxes"].shape)
plt.xticks([])
plt.yticks([])
plt.imshow(img)
plt.show()


len(train_dataloader)


def evaluate_metrics(model, dataloader, device):
    """
    Runs model on val set, returns:
      - mean IoU (per-image, best-of-preds)
      - mAP@0.5
    """
    model.eval()
    model.to(device)
    
    # meter for COCO‐style AP at IoU=0.5
    metric = MeanAveragePrecision(
        iou_type=('bbox',),
        class_metrics=False
    )
    
    total_iou = 0.0
    img_count = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            # move targets to device; keep images as PIL or [C,H,W] tensors
            imgs = [img.to(device) for img in images]
            targs = [{k: v.to(device) for k,v in t.items()} for t in targets]
            
            # forward
            outputs = model(imgs, targs)
            
            # 1) accumulate IoU
            for gt, pred in zip(targs, outputs):
                gt_boxes   = gt['boxes']            # [N_gt,4]
                pred_boxes = pred['boxes']          # [N_pred,4]
                
                if gt_boxes.numel()==0 or pred_boxes.numel()==0:
                    continue
                ious = box_iou(pred_boxes, gt_boxes)       # [N_pred, N_gt]
                best = ious.max(dim=1).values              # best IoU per pred
                total_iou += best.mean().item()
                img_count += 1
            
            # 2) update mAP metric
            # predictions must be list of dicts with same keys
            preds = [
                {
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu()
                }
                for out in outputs
            ]
            gts = [
                {
                    'boxes': targ['boxes'].cpu(),
                    'labels': targ['labels'].cpu()
                }
                for targ in targs
            ]
            metric.update(preds, gts)
    
    mean_iou = total_iou / img_count if img_count else 0.0
    stats = metric.compute()
    map50_95 = stats['map']
    ap50 = stats['map_50']
    ap75 = stats['map_75']
    
    print(f"Eval → Mean IoU: {mean_iou:.4f}")
    print(f"mAP@[.50:.95]: {map50_95:.4f}")
    print(f"AP@0.50:      {ap50:.4f}")
    print(f"AP@0.75:      {ap75:.4f}")
    model.train()
    return mean_iou, ap50, map50_95

def train_model(model, train_loader, 
                val_loader, num_epochs, optimizer, 
                log_every=20, device="cpu", writer=None):
    model.to(device).train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"----Epoch: {epoch+1}----")
        
        for batch_idx, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [img.to(device) for img in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            
            # forward + loss
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            
            
            # backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % log_every == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss:.4f}")
        avg_loss = epoch_loss / batch_idx
        print(f"Epoch {epoch+1} done | Avg Loss {avg_loss:.4f}")
        if writer:
            writer.add_scalar("Train/AvgLoss", avg_loss, epoch)

        # ─── Validation ───────────────────────────────────
        mean_iou, ap50, map50_95 = evaluate_metrics(model, val_loader, device)
        writer.add_scalar("Val/Mean_IoU",        mean_iou,   epoch)
        writer.add_scalar("Val/AP@0.50",         ap50,       epoch)
        writer.add_scalar("Val/mAP@[.50:.95]",   map50_95,   epoch)
        

    model.to("cpu")
        
            


                


from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import train_test_split
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
transform = weights.transforms()
simple_tf = T.Compose([
    T.ToTensor(),   # PIL → FloatTensor [0,1]
])

num_classes = 2 # wheat + background
in_features = model.roi_heads.box_predictor.cls_score.in_features # original input size
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # change head to match num_classes

train_ind, val_ind = train_test_split(range(len(bbox_dict)), test_size=0.1, random_state=42) 

full_set  = CustomDataset(train_dir, bbox_dict, transform=simple_tf)
train_set = Subset(full_set, train_ind)
val_set   = Subset(full_set, val_ind)

train_dataloader = DataLoader(train_set,  
                              batch_size=8, 
                              shuffle=True,
                             collate_fn = detection_collate,
                             pin_memory = True if torch.cuda.is_available() else False)
val_dataloader = DataLoader(val_set,  
                            batch_size=8, 
                            shuffle=True,
                            collate_fn = detection_collate,
                            pin_memory = True if torch.cuda.is_available() else False)

# Training Start
torch.manual_seed(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
n_epochs = 10
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter("runs/wheat_detector")

train_model(model, train_dataloader, val_dataloader, n_epochs, optimizer, log_every=50, device=device, writer=writer)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_batch(images, outputs, class_names=None, threshold=0.5):
    """
    Visualize a batch of images and their detection outputs.
    
    Args:
      images (List[Tensor]): List of image tensors [C, H, W] in [0,1].
      outputs (List[Dict]): List of model outputs with keys 'boxes', 'labels', 'scores'.
      class_names (Dict[int, str], optional): Mapping label indices to names.
      threshold (float, optional): Confidence threshold to filter boxes.
    """
    batch_size = len(images)
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten() if batch_size > 1 else [axs]
    
    for idx, (img, output) in enumerate(zip(images, outputs)):
        ax = axs[idx]
        # convert tensor to numpy image
        img_np = img.cpu().permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.axis('off')
        # draw boxes above threshold
        boxes = output['boxes'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        labels = output.get('labels')
        if labels is not None:
            labels = labels.cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels if labels is not None else []):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2,
                edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            # label text
            class_text = class_names[label] if class_names and label in class_names else str(label)
            ax.text(x1, y1 - 5, f"{score:.2f}", 
                    color='white', fontsize=8, backgroundcolor='r')
    
    # hide any extra subplots
    for j in range(idx + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_classification.png")
    plt.show()

# Example usage:
model.to(device).eval()
visuals = []
images = next(iter(val_dataloader))[0]
with torch.no_grad():
    for img in images:
        out = model([img.to(device)])[0]      # one image → less mem
        visuals.append({k: v.cpu() for k,v in out.items()})

# now visuals is on CPU, safe to plot
visualize_batch(images, visuals, class_names={1:'wheat'}, threshold=0.8)




class CustomDataset(Dataset):
    def __init__(self, imgs_path, annot_dict, ids, transform):
        self.imgs_path = imgs_path
        self.annot_dict = {k:v for k,v in annot_dict.items() if k in ids}
        self.transform = transform
        self.ids = ids
        
    def __len__(self):
        return len(self.annot_dict.keys())

    def __getitem__(self, idx):
        
        image_id = self.ids[idx]
        boxes = np.array(self.annot_dict[image_id], dtype=np.float32)

        # 2. convert to x1,y1,x2,y2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = xmin + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y2 = ymin + height
        
        img = cv2.imread(f"{self.imgs_path}/{image_id}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = [1] * len(boxes)

        augmented = self.transform(image=img, bboxes=boxes, labels=labels)
        img = augmented["image"].float()
        boxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)
        labels = torch.tensor(augmented["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target

def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)




# Setting up architecture
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_augmented = fasterrcnn_resnet50_fpn_v2(weights=weights)
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop((512,512)),
    A.RandomBrightnessContrast(),
    A.Resize(512, 512),
    ToTensorV2(), # converts to [C,H,W] float tensor
],
    bbox_params=A.BboxParams(  
      format='pascal_voc',  # [x_min, y_min, x_max, y_max]
      label_fields=['labels']
    ))

val_tf = A.Compose([
    A.Resize(512,512),
    ToTensorV2(),
],
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"]
    ))


num_classes = 2 # wheat + background
in_features = model_augmented.roi_heads.box_predictor.cls_score.in_features # original input size
model_augmented.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # change head to match num_classes

train_ids, val_ids = train_test_split(list(bbox_dict.keys()), test_size=0.1, random_state=42) 

train_set = CustomDataset(train_dir, bbox_dict, train_ids, train_tf)
val_set = CustomDataset(train_dir, bbox_dict, val_ids, val_tf)


train_dataloader = DataLoader(train_set,  
                              batch_size=8, 
                              shuffle=True,
                             collate_fn = detection_collate,
                              num_workers=4,
                             pin_memory = True if torch.cuda.is_available() else False)
val_dataloader = DataLoader(val_set,  
                            batch_size=8, 
                            shuffle=True,
                            collate_fn = detection_collate,
                            num_workers=4,
                            pin_memory = True if torch.cuda.is_available() else False)

# Training Start
torch.manual_seed(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
n_epochs = 10
optimizer = optim.SGD(model_augmented.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter("runs/wheat_detector")

train_model(model_augmented, train_dataloader, val_dataloader, n_epochs, optimizer, log_every=50, device=device, writer=writer)




