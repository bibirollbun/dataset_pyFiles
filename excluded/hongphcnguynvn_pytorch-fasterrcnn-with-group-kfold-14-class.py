import pandas as pd
import numpy as np
import cv2
import os
import re
import time

from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt


train_dir = '../input/vinbigdata-512-image-dataset/vinbigdata/train'
test_dir = '../input/vinbigdata-512-image-dataset/vinbigdata/test'
train_df = pd.read_csv('../input/vinbigdata-512-image-dataset/vinbigdata/train.csv')


train_df.head()


train_df = train_df[train_df['class_id'] != 14].reset_index(drop=True)
train_df.head()


train_df['image_path'] = '../input/vinbigdata-512-image-dataset/vinbigdata/train/'+train_df.image_id+'.png'
train_df.head()


gkf  = GroupKFold(n_splits = 5)
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
    train_df.loc[val_idx, 'fold'] = fold
train_df.head()


train_df.groupby('fold')['image_id'].agg(lambda x: x.nunique()).reset_index()


IMG_SIZE = 512
train_df['xmin'] = (train_df['x_min']/train_df['width'])*IMG_SIZE
train_df['ymin'] = (train_df['y_min']/train_df['height'])*IMG_SIZE
train_df['xmax'] = (train_df['x_max']/train_df['width'])*IMG_SIZE
train_df['ymax'] = (train_df['y_max']/train_df['height'])*IMG_SIZE


assert train_df['xmin'].all() <= IMG_SIZE
assert train_df['ymin'].all() <= IMG_SIZE
assert train_df['xmax'].all() <= IMG_SIZE
assert train_df['ymax'].all() <= IMG_SIZE


train_df[train_df['image_id'] == '9a5094b2563a1ef3ff50dc5c7ff71345']


class_dict = dict(set(zip(train_df.class_id, train_df.class_name)))
classes = []
for key in sorted(class_dict.keys()): 
    classes.append(class_dict[key])

classes = ['_'] + classes   # adding background
classes


class VBDDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # all the labels are shifted by 1 to accomodate background
        labels = torch.squeeze(torch.as_tensor((records.class_id.values+1,), dtype=torch.int64))
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.as_tensor(sample['bboxes'])

        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]


dt = VBDDataset(train_df, train_dir)
dt[0]


# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


model.eval()


num_classes = 15  # 14 classes + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# A Class for keeping track of average
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = VBDDataset(train_df, train_dir, get_train_transform())
valid_dataset = VBDDataset(train_df, train_dir, get_valid_transform())


train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
sample = images[2].permute(1,2,0).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)


def get_dataloaders(df, trn_idx, val_idx):
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataset = VBDDataset(train_, train_dir, get_train_transform())
    valid_dataset = VBDDataset(valid_, train_dir, get_valid_transform())


    train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_data_loader, valid_data_loader



def train_model(model, dataloader, device, epochs, optimizer, lr_scheduler, fold):
    
    best_loss = 1e10
    loss_hist = Averager()
    itr = 1
    all_losses = []
    
    model.train()
    
    for epoch in range(epochs):
        loss_hist.reset() 
    
        for images, targets, image_ids in dataloader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)
            all_losses.append(loss_value)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
        
        # saving the model based on training loss for now. - later can be moved to validation
        if loss_hist.value < best_loss:
            best_loss = loss_hist.value
            torch.save(model.state_dict(), f'fasterrcnn_model_{fold}.pt')

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}\n")
        
    return all_losses
        
        
def validate_model(model, dataloader, device):
    print("\n Starting Validation ... ")
    loss_hist = Averager()
    itr = 1

    loss_hist.reset() 

    for images, targets, image_ids in dataloader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

    print(f"\nFinal loss: {loss_hist.value}")


    


def run_fold(fold):
    print(f"Starting fold {fold}")
    start = time.time()
    trn_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    
    
    trainloader, valloader = get_dataloaders(train_df, trn_idx, val_idx)
    
    loss_hist = train_model(model, trainloader, device, epochs, optimizer, lr_scheduler, fold)
    
    # plot training loss
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.title("Training Loss Statistic", size=17)
    plt.xlabel("Iteration", size=15)
    plt.ylabel("Loss Value", size=15)
    plt.show()
    
    validate_model(model, valloader, device)
    
    print(f"Completed Fold {fold} in {round(time.time()-start, 2)} seconds")



model.to(device)

# set params for model
params = [p for p in model.parameters() if p.requires_grad]

# set optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# set lr scheduler
lr_scheduler = None

# set epochs
epochs = 2

# set folds
num_folds = 1


for fold in range(num_folds):
    run_fold(fold)


images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1].permute(1,2,0).cpu().numpy()
clss = targets[1]['labels'].cpu().numpy().astype(np.int32)

model.eval()
cpu_device = torch.device("cpu")

outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box, clas in zip(boxes, clss):
    cv2.putText(sample, f"{classes[clas]}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 1)
    
ax.set_axis_off()
ax.imshow(sample)


import torch
import numpy as np
from tqdm import tqdm

# ==== 1. Tính IoU ====
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2g - x1g) * (y2g - y1g)
    union_area = box_area + gt_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ==== 2. Tính Precision, Recall, AP ====
def calculate_precision_recall_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    pred_boxes.sort(key=lambda x: x[2], reverse=True)
    tp, fp, matched = [], [], set()

    for pred in pred_boxes:
        image_id, cls, score, *pred_box = pred
        match_found = False
        for i, gt in enumerate(gt_boxes):
            if gt[0] == image_id and gt[1] == cls:
                iou = compute_iou(pred_box, gt[2:])
                if iou >= iou_threshold and i not in matched:
                    match_found = True
                    matched.add(i)
                    break
        tp.append(1 if match_found else 0)
        fp.append(0 if match_found else 1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / len(gt_boxes) if gt_boxes else np.zeros_like(tp_cum)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += np.max(p) if p.size else 0
    ap /= 11.0
    return precisions, recalls, ap


# ==== 3. Tính mAP ====
def compute_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    aps = []
    classes = set([b[1] for b in gt_boxes])
    for cls in classes:
        pred_cls = [b for b in pred_boxes if b[1] == cls]
        gt_cls = [b for b in gt_boxes if b[1] == cls]
        _, _, ap = calculate_precision_recall_ap(pred_cls, gt_cls, iou_thresh)
        aps.append(ap)
    return np.mean(aps), aps


# ==== 4. Inference + gom pred/gt từ valid_data_loader ====
def evaluate_model(model, valid_data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")

    pred_boxes = []
    gt_boxes = []

    with torch.no_grad():
        for images, targets, image_ids in tqdm(valid_data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in outputs]

            for i, (target, output) in enumerate(zip(targets, outputs)):
                image_id = image_ids[i]
                gt_cls = target['labels'].cpu().numpy()
                gt_bx = target['boxes'].cpu().numpy()
                for cls, box in zip(gt_cls, gt_bx):
                    gt_boxes.append([image_id, cls, *box])

                pred_cls = output['labels'].cpu().numpy()
                pred_bx = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                for cls, score, box in zip(pred_cls, scores, pred_bx):
                    pred_boxes.append([image_id, cls, score, *box])

    mAP, aps = compute_map(pred_boxes, gt_boxes, iou_thresh=0.5)
    return mAP, aps



mAP, aps = evaluate_model(model, valid_data_loader, device)
print(f"mAP@0.5: {mAP:.4f}")
for idx, ap in enumerate(aps):
    print(f"AP for class {idx}: {ap:.4f}")


import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Calculate full evaluation metrics for one class."""
    pred_boxes.sort(key=lambda x: x[2], reverse=True)
    tp, fp, matched_gt_idx, iou_list = [], [], set(), []

    for pred in pred_boxes:
        image_id, cls, score, *pred_box = pred
        match_found = False
        for i, gt in enumerate(gt_boxes):
            if gt[0] == image_id and gt[1] == cls and i not in matched_gt_idx:
                iou = compute_iou(pred_box, gt[2:])
                if iou >= iou_threshold:
                    match_found = True
                    matched_gt_idx.add(i)
                    iou_list.append(iou)
                    break
        tp.append(1 if match_found else 0)
        fp.append(0 if match_found else 1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    eps = 1e-6

    precision = tp_cum / (tp_cum + fp_cum + eps)
    recall = tp_cum / (len(gt_boxes) + eps)

    # 11-point Interpolated Average Precision
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += np.max(p) if p.size else 0
    ap /= 11.0

    final_precision = precision[-1] if len(precision) else 0
    final_recall = recall[-1] if len(recall) else 0
    f1_score = 2 * final_precision * final_recall / (final_precision + final_recall + eps)
    mean_iou = np.mean(iou_list) if iou_list else 0

    return {
        "AP": ap,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
    }


def evaluate_model(model, valid_data_loader, device, iou_thresh=0.5):
    model.eval()
    cpu_device = torch.device("cpu")
    pred_boxes, gt_boxes = [], []

    with torch.no_grad():
        for images, targets, image_ids in tqdm(valid_data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in outputs]

            for i, (target, output) in enumerate(zip(targets, outputs)):
                image_id = image_ids[i]
                gt_cls = target['labels'].cpu().numpy()
                gt_bx = target['boxes'].cpu().numpy()
                for cls, box in zip(gt_cls, gt_bx):
                    gt_boxes.append([image_id, cls, *box])

                pred_cls = output['labels'].cpu().numpy()
                pred_bx = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                for cls, score, box in zip(pred_cls, scores, pred_bx):
                    pred_boxes.append([image_id, cls, score, *box])

    # === Group predictions and GT by class
    classes = sorted(set([b[1] for b in gt_boxes + pred_boxes]))
    results = {}
    for cls in classes:
        preds = [b for b in pred_boxes if b[1] == cls]
        gts = [b for b in gt_boxes if b[1] == cls]
        metrics = calculate_metrics(preds, gts, iou_threshold=iou_thresh)
        results[cls] = metrics

    # === Compute mean of each metric
    mean_metrics = {}
    for key in ["AP", "precision", "recall", "f1_score", "mean_iou"]:
        mean_metrics[f"mean_{key}"] = np.mean([res[key] for res in results.values()])

    return mean_metrics, results



mean_metrics, per_class_metrics = evaluate_model(model, valid_data_loader, device)

print("=== Mean Metrics (mAP) ===")
for k, v in mean_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n=== Per-Class Metrics ===")
for cls_id, metrics in per_class_metrics.items():
    print(f"Class {cls_id}: ", end="")
    print(", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))



def compute_iou(box1, box2):
    """Tính IoU giữa 2 box"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2g - x1g) * (y2g - y1g)
    union_area = box_area + gt_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_precision_recall_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    pred_boxes: list of [image_id, class_id, confidence, x1, y1, x2, y2]
    gt_boxes:   list of [image_id, class_id, x1, y1, x2, y2]
    """
    pred_boxes.sort(key=lambda x: x[2], reverse=True)
    tp = []
    fp = []
    matched = set()

    for pred in pred_boxes:
        image_id, cls, score, *pred_box = pred
        match_found = False
        for i, gt in enumerate(gt_boxes):
            if gt[0] == image_id and gt[1] == cls:
                iou = compute_iou(pred_box, gt[2:])
                if iou >= iou_threshold and i not in matched:
                    match_found = True
                    matched.add(i)
                    break
        if match_found:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / len(gt_boxes)

    # Tính Average Precision theo cách PASCAL VOC
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += np.max(p) if p.size else 0
    ap /= 11.0
    return precisions, recalls, ap


pred_boxes = []
gt_boxes = []

for i, (image, target) in enumerate(zip(images, targets)):
    image_id = image_ids[i]
    gt_cls = target['labels'].cpu().numpy()
    gt_bx = target['boxes'].cpu().numpy()

    for cls, box in zip(gt_cls, gt_bx):
        gt_boxes.append([image_id, cls, *box])

    out = outputs[i]
    pred_bx = out['boxes'].detach().cpu().numpy()
    pred_cls = out['labels'].detach().cpu().numpy()
    scores = out['scores'].detach().cpu().numpy()
    # pred_cls = out['labels'].cpu().numpy()
    # pred_bx = out['boxes'].cpu().numpy()
    # scores = out['scores'].cpu().numpy()

    for cls, score, box in zip(pred_cls, scores, pred_bx):
        pred_boxes.append([image_id, cls, score, *box])


def compute_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    aps = []
    classes = set([b[1] for b in gt_boxes])
    for cls in classes:
        pred_cls = [b for b in pred_boxes if b[1] == cls]
        gt_cls = [b for b in gt_boxes if b[1] == cls]
        _, _, ap = calculate_precision_recall_ap(pred_cls, gt_cls, iou_thresh)
        aps.append(ap)
    return np.mean(aps)

