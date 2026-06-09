# To import kaggle datasets
'''!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6


# import kaggle json to connect to kaggle user account to download datsets
#files.upload()

# see if kaggle json exists
!ls -lha kaggle.json

# The Kaggle API client expects this file to be in ~/.kaggle,
# so lets move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json

# download our dataset
!kaggle competitions download -c global-wheat-detection --force # Download Global Wheat Detection dataset from Kaggle
!mkdir global-wheat-detection
!unzip global-wheat-detection.zip -d global-wheat-detection

!pip install albumentations==0.4.6'''


!pip install ensemble-boxes
!pip install ffmpeg-python
!pip install ipdb


import pandas as pd
import numpy as np
import cv2
import os
import re
import time
import datetime
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

import ffmpeg

import numba
import cv2
import ast
from glob import glob

from numba import jit
from typing import List, Union, Tuple


torch.cuda.is_available()


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
model_path ='/kaggle/input/global-wheat-detection-pretrained-weights/fasterrcnn_resnet50_fpn2_17July.pth'#'fasterrcnn_resnet50_fpn2.pth' # Path for the best model to be saved
es_patience = 3 #This is required for early stopping, the number of epochs we will wait with no improvement before stopping


train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape
train_df.columns



train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

train_df.head


image_ids = train_df['image_id'].unique() #collecting all unique images
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]


valid_df = train_df[train_df['image_id'].isin(valid_ids)]  
train_df = train_df[train_df['image_id'].isin(train_ids)]



valid_df.shape, train_df.shape



class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id] #Getting all coordinates for the given image

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area,dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
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
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]



dataset_new =  pd.read_csv(f'/kaggle/input/data-bbox-basketball/annotations_layup/instances_default.json')
dataset_new.shape


input_video_1 = '/kaggle/input/all-data/Ludwig/video_layup.mp4'
input_video_2 = '/kaggle/input/all-data/Ludwig/video_long.mp4'
output_directory_1 = '/kaggle/working/video_layup'
output_directory_2 = '/kaggle/working/video_long'

#Create directories if they don't exist
os.makedirs(output_directory_1, exist_ok=True)
os.makedirs(output_directory_2, exist_ok=True)

#Extract frames using ffmpeg-python
stream_1 = ffmpeg.input(input_video_1)
stream_2 = ffmpeg.input(input_video_2)


stream_1.output(f'{output_directory_1}/frame_%06d.jpg', q=15, start_number=0).run()
stream_2.output(f'{output_directory_2}/frame_%06d.jpg', q=15, start_number=0).run()


import json
import pandas as pd
# Pfad zur COCO-JSON-Datei
json_paths = ["/kaggle/input/data-bbox-basketball/annotations_layup/instances_default.json", "/kaggle/input/data-bbox-basketball/annotations_long/instances_default.json"]
annotation_folder_names = ["annotations_layup", "annotations_long"]
rows = []

for annotation_folder_name in annotation_folder_names:
    # Lade JSON-Daten
    json_path = f"/kaggle/input/data-bbox-basketball/{annotation_folder_name}/instances_default.json"
    with open(json_path, "r") as f:
        coco_data = json.load(f)
    
    # Mapping von image_id zu Bildinfos (Breite, Höhe, Dateiname)
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Extrahiere gewünschte Daten
    
    for ann in coco_data['annotations']:
        img = image_info[ann['image_id']]
        bbox = ann['bbox']
        
        row = {
            'image_id': ann['image_id'],
            'label' : ann['category_id'],
            'width': img['width'],
            'height': img['height'],
            'file_name': f"{annotation_folder_name.replace('annotations','video')}/{img['file_name'].strip('png')+'jpg'}", #gab hier noch ein paar kleine Probleme
            'x': bbox[0],
            'y': bbox[1],
            'w': bbox[2],
            'h': bbox[3],
        }
        rows.append(row)

# Erstelle DataFrame
df = pd.DataFrame(rows, columns=['image_id', 'label' , 'width', 'height', 'file_name', 'x', 'y', 'w', 'h'])

# Optional: Ausgabe prüfen
print(df.head())

# Optional: Speichern
df.to_csv("bbox_table.csv", index=False)

#img["file_name"]
#image_info


from sklearn.model_selection import train_test_split

# Alle eindeutigen image_ids extrahieren
image_ids = df['image_id'].unique()

# Zufällig in 80% Training und 20% Validierung splitten
train_ids, valid_ids = train_test_split(
    image_ids,
    test_size=0.2,
    random_state=42,  # für Reproduzierbarkeit – kannst du weglassen, wenn du immer neu mischen willst
    shuffle=True
)

# Jetzt kannst du diese IDs verwenden, um df zu splitten
train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
valid_df = df[df['image_id'].isin(valid_ids)].reset_index(drop=True)

valid_df.shape, train_df.shape



import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BasketballDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]  # Get all records for the given image

        # Get file name and build full path
        file_name = records['file_name'].iloc[0]
        file_path = os.path.join(self.image_dir, file_name)

        # Load image and convert color
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {file_path}")

        # Convert BGR to RGB and cast to float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Resize the image: halved resolution
        new_width = image.shape[1] // 2
        new_height = image.shape[0] // 2
        image = cv2.resize(image, (new_width, new_height))

        # Normalize image
        image /= 255.0

        # Process bounding boxes (if any)
        boxes = records[['x', 'y', 'w', 'h']].values
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # Scale bounding box coordinates by the same factor (0.5) as the image
        boxes = boxes * 0.5

        # Calculate area from bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Read labels from dataframe
        labels = torch.as_tensor(records['label'].values, dtype=torch.int64)

        # Assume all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': labels,
            'image_id': torch.tensor([index]),
            'area': area,
            'iscrowd': iscrowd
        }

        # Apply transforms if provided
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes.tolist(),
                'labels': labels.tolist()
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)



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



# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 4  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


#Function to calculate loss for every epoch
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



dir_train = "/kaggle/working"

train_dataset = BasketballDataset(train_df, dir_train, get_train_transform())
train_dataset[5]


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = BasketballDataset(train_df, dir_train, get_train_transform())
valid_dataset = BasketballDataset(valid_df, dir_train, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
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



#for i in train_data_loader:
#    continue


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area



@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


models = [model]
from ensemble_boxes import *

#device = torch.device('cuda:0')


def make_ensemble_predictions(images):
    images = list(image.to(device) for image in images)    
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result

def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.5, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [prediction[image_index]['labels'].data.cpu().numpy() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
print(device)


model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None
model_path = '/kaggle/working/fasterrcnn_resnet50_fpn2_17July.pth'

num_epochs = 20


from tqdm import tqdm
# Wenn du das hier trainierst packe rechts bei session options als accelerator noch GPU P100 rein, das sollte gehen
loss_hist = Averager()
best_val = None
patience = es_patience
for epoch in range(num_epochs):
    print("start")
    start_time = time.time()
    itr = 1
    loss_hist.reset()
    model.train()
    for images, targets, image_ids in tqdm(train_data_loader):
        
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if k =='labels' else v.float().to(device) for k, v in t.items()} for t in targets]#[{k: v.double().to(device) if k =='boxes' else v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    
    #At every epoch we will also calculate the validation IOU
    validation_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    model.eval()
    for images, targets,imageids in valid_data_loader: #return image, target, image_id
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if k =='labels' else v.float().to(device) for k, v in t.items()} for t in targets]
        #outputs = model(images) 
        
        predictions = make_ensemble_predictions(images)
   
        for i, image in enumerate(images):
            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = boxes.astype(np.int32).clip(min=0, max=1023)
            
            preds = boxes#outputs[i]['boxes'].data.cpu().numpy()
            #scores = outputs[i]['scores'].data.cpu().numpy()
            preds_sorted_idx = np.argsort(scores)[::-1]
            preds_sorted = preds[preds_sorted_idx]
            gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            image_precision = calculate_image_precision(preds_sorted,
                                                    gt_boxes,
                                                    thresholds=iou_thresholds,
                                                    form='coco')

            validation_image_precisions.append(image_precision)
    val_iou = np.mean(validation_image_precisions)
    print(f"Epoch #{epoch+1} loss: {loss_hist.value}","Validation IOU: {0:.4f}".format(val_iou),"Time taken :",str(datetime.timedelta(seconds=time.time() - start_time))[:7])
    if not best_val:
        best_val = val_iou  # So any validation roc_auc we have is the best one for now
        print("Saving model")
        torch.save(model, model_path)  # Saving the model
        #continue
    if val_iou >= best_val:
        print("Saving model as IOU is increased from",best_val,"to",val_iou)
        best_val = val_iou
        patience = es_patience  # Resetting patience since we have new best validation accuracy
        torch.save(model, model_path)  # Saving current best model torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
    else:
        patience -= 1
        if patience == 0:
            print('Early stopping. Best Validation IOU: {:.3f}'.format(best_val))
            break


model = torch.load("/kaggle/input/object-det-final/fasterrcnn_resnet50_fpn2_17July.pth")


#also create the stuff for the test videos:

#Input and output directories
input_video_1 = '/kaggle/input/test-videos/test_ludwig.mp4'
input_video_2 = '/kaggle/input/test-videos/test_valentin.mp4'
input_video_3 = "/kaggle/input/test-winter/test_winter.mp4"

output_directory_1 = '/kaggle/working/test_ludwig'
output_directory_2 = '/kaggle/working/test_valentin'
output_directory_3 = '/kaggle/working/test_winter'

#Create directories if they don't exist
os.makedirs(output_directory_1, exist_ok=True)
os.makedirs(output_directory_2, exist_ok=True)
os.makedirs(output_directory_3, exist_ok=True)

#Extract frames using ffmpeg-python
#stream_1 = ffmpeg.input(input_video_1)
stream_2 = ffmpeg.input(input_video_2)
stream_3 = ffmpeg.input(input_video_3)

#stream_1.output(f'{output_directory_1}/video_layup_frame_%06d.jpg', q=15).run()
stream_2.output(f'{output_directory_2}/video_long_frame_%06d.jpg', q=15).run()
stream_3.output(f'{output_directory_3}/video_winter_frame_%06d.jpg', q=15).run()


DIR_TEST = '/kaggle/working/test_ludwig'
image_dir = '/kaggle/working/test_ludwig'

test_df = pd.DataFrame()
test_df['image_id'] = np.array([path.split('/')[-1][:-4] for path in glob(f'{DIR_TEST}/*.jpg')])
print("jooooooooooooooooooooo")


import torch
import torchvision

# Wähle Gerät
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modell laden und auf Gerät verschieben

model.to(device)  # WICHTIG!!!
#model.eval()


import cv2
import numpy as np
from torch.utils.data import Dataset

class BasketballTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.image_ids.sort()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image_path = f'{self.image_dir}/{image_id}.jpg'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return len(self.image_ids)



def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = BasketballTestDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)


detection_threshold = 0.4
results = []
outputs = []
test_images = []

for images, image_ids in test_data_loader:
    images = list(image.to(device) for image in images)
    
    with torch.no_grad():
        predictions = model(images)

    for i, image in enumerate(images):
        test_images.append(image)
        
        boxes, scores, labels = run_wbf([predictions],skip_box_thr=0.1, image_index=i)

        boxes = boxes.astype(np.int32).clip(min=0, max=1023)
        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = boxes[preds_sorted_idx]

        output = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

        outputs.append(output)
        image_id = image_ids[i]
print("finish")


def apply_nms_to_outputs(outputs, iou_threshold=0.1):
    new_outputs = []
    for output in outputs:
        boxes = torch.tensor(output['boxes'], dtype=torch.float32)
        scores = torch.tensor(output['scores'], dtype=torch.float32)
        labels = torch.tensor(output['labels'], dtype=torch.int64)

        final_boxes = []
        final_scores = []
        final_labels = []

        unique_labels = labels.unique()
        for label in unique_labels:
            inds = (labels == label).nonzero(as_tuple=True)[0]
            label_boxes = boxes[inds]
            label_scores = scores[inds]

            if label_boxes.size(0) == 0:
                continue

            keep = torchvision.ops.nms(label_boxes, label_scores, iou_threshold=iou_threshold)

            final_boxes.append(label_boxes[keep])
            final_scores.append(label_scores[keep])
            final_labels.append(labels[inds][keep])

        # Stack alles zusammen
        if final_boxes:
            final_boxes = torch.cat(final_boxes).cpu().numpy()
            final_scores = torch.cat(final_scores).cpu().numpy()
            final_labels = torch.cat(final_labels).cpu().numpy()
        else:
            final_boxes = np.empty((0, 4))
            final_scores = np.array([])
            final_labels = np.array([])

        new_outputs.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })

    return new_outputs


outputs = keep_best_per_label_from_outputs(outputs)



output_ludwig = outputs



import pickle

with open('/kaggle/working/output_ludwig_3.pkl', 'wb') as f:
   pickle.dump(outputs, f)
#with open('/kaggle/input/auswertung/output_ludwig.pkl', 'rb') as f:
    #test_images = pickle.load(f)




test_images = output_ludwig

for i in range(2):
    print(test_images[i])













import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import albumentations as A

# Nur Resize, kein ToTensor für Visualisierung
def get_visual_transform():
    return A.Compose([
        A.Resize(512, 512)
    ])

def visualize_boxes(test_images, image_folder, start_idx=0, num_images=5):
    colors = {}
    transform = get_visual_transform()  # Nur Resize laden
    
    for img_idx in range(start_idx, start_idx + num_images):
        if img_idx >= len(test_images):
            break
        
        # Bildpfad bauen
        image_path = os.path.join(image_folder, f"video_layup_frame_{img_idx+1:06d}.jpg")
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Bild nicht gefunden: {image_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Bild auf 512x512 transformieren (nur Resize)
        img = transform(image=img)['image']
        
        boxes = test_images[img_idx]['boxes']
        labels = test_images[img_idx]['labels']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        for box, label in zip(boxes, labels):
            # Farbe für jedes Label festlegen
            if label not in colors:
                colors[label] = (random.random(), random.random(), random.random())
            
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color=colors[label], linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 10, 0), f"Label: {int(label)}", color=colors[label], fontsize=10, backgroundcolor='white')
        
        plt.axis('off')
        plt.title(f"Frame {img_idx+1}")
        plt.show()



visualize_boxes(test_images, "/kaggle/working/test_ludwig", start_idx=100, num_images=10)


import cv2
import os
import numpy as np
import random
import albumentations as A
from tqdm import tqdm

# Transformations: Resize auf 512x512
def get_visual_transform():
    return A.Compose([
        A.Resize(512, 512)
    ])

def create_video_from_boxes(test_images, image_folder, output_path, fps=10):
    colors = {}
    transform = get_visual_transform()
    
    # VideoWriter initialisieren
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (512, 512)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img_idx in tqdm(range(len(test_images)), desc="Creating video"):
        # Bildpfad
        image_path = os.path.join(image_folder, f"video_layup_frame_{img_idx+1:06d}.jpg")
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Bild nicht gefunden: {image_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)['image']  # Resize
        
        boxes = test_images[img_idx]['boxes']
        labels = test_images[img_idx]['labels']
        
        for box, label in zip(boxes, labels):
            if label not in colors:
                colors[label] = [int(random.random()*255) for _ in range(3)]
            
            x1, y1, x2, y2 = map(int, box)
            color = colors[label]
            
            # Box zeichnen
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label schreiben
            cv2.putText(img, f"Label: {int(label)}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Bild zurück nach BGR (weil OpenCV erwartet BGR für Videos)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    print(f"Video gespeichert: {output_path}")


create_video_from_boxes(
    test_images,
    image_folder="/kaggle/working/test_ludwig",
    output_path="/kaggle/working/output_video_ludwig_1.mp4",
    fps=6  # z.B. 10 Frames pro Sekunde
)

