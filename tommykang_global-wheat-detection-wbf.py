#!pip download ensemble_boxes
#!pip download ultralytics

!pip install --no-index --find-links /kaggle/input/ultralytics-and-ensemble-boxes /kaggle/input/ultralytics-and-ensemble-boxes/ensemble_boxes-1.0.9-py3-none-any.whl
!pip install --no-index --find-links /kaggle/input/ultralytics-and-ensemble-boxes /kaggle/input/ultralytics-and-ensemble-boxes/ultralytics-8.3.144-py3-none-any.whl


from ensemble_boxes import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
import torch
import cv2
from ultralytics import YOLO
import os
import re
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from glob import glob

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

DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

DIR_WEIGHTS = '/kaggle/input/global-wheat-detection-faster-rcnn/pytorch/default/1'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn_global_wheat_detection.pth'


test_df = pd.read_csv(f"{DIR_INPUT}/sample_submission.csv")
test_df.shape


class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
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
        return self.image_ids.shape[0]


def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)


def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


detection_threshold = 0.5
rcnn_results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        result = {
            'image_id': image_id,
            'boxes': boxes,
            'scores': scores
        }

        rcnn_results.append(result)

rcnn_results[:2]


from ultralytics import YOLO
model = YOLO('/kaggle/input/global-wheat-detection-yolov11n-v2/pytorch/default/1/custom_yolo (1).pt')


test_dir = "/kaggle/input/global-wheat-detection/test"
test_images = sorted(glob(os.path.join(test_dir, "*.jpg")))

yolo_results = []

for image_path in test_images:
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    preds = model.predict(source=image_path, conf=0.5, iou=0.5, save=False, verbose=False)
    boxes = preds[0].boxes
    
    """prediction_string = ""
    for xyxy, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        prediction_string += f\"{score:.4f} {int(x_min)} {int(y_min)} {int(width)} {int(height)} \""""

    yolo_results.append({
        "image_id": image_id,
        'boxes': boxes.xyxy.cpu().numpy().astype(np.int32),
        'scores': boxes.conf.cpu().numpy()
    })

print(yolo_results[:2])


rcnn_results = sorted(rcnn_results, key=lambda x: x['image_id'])
yolo_results = sorted(yolo_results, key=lambda x: x['image_id'])

list(zip(rcnn_results, yolo_results))[:2]


image_size = 1024
results = []
for rcnn_result, yolo_result in list(zip(rcnn_results, yolo_results)):
    boxes = [rcnn_result['boxes'] / (image_size - 1), yolo_result['boxes'] / (image_size - 1)]
    scores = [rcnn_result['scores'], yolo_result['scores']]
    labels = [np.ones(rcnn_result['scores'].shape[0]), np.ones(yolo_result['scores'].shape[0])]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=0.5, skip_box_thr=0.5)
    boxes = boxes * (image_size - 1)
    boxes = boxes.astype(np.int32)
    result = {
        'image_id': rcnn_result['image_id'],
        'rcnn_boxes': rcnn_result['boxes'],
        'yolo_boxes': yolo_result['boxes'],
        'boxes': boxes,
        'scores': scores,
        
    }
    results.append(result)

results[:2]


import matplotlib.pyplot as plt

image = cv2.imread(f'{DIR_TEST}/2fd875eaa.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0

boxes = []
for result in results:
    if result['image_id'] == '2fd875eaa':
        boxes = result['boxes']

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(image,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(image);



print(results[:1])
for i in range(len(results)):
    results[i]['boxes'][:, 2] = results[i]['boxes'][:, 2] - results[i]['boxes'][:, 0]
    results[i]['boxes'][:, 3] = results[i]['boxes'][:, 3] - results[i]['boxes'][:, 1]
    results[i]['PredictionString'] = format_prediction_string(results[i]['boxes'], results[i]['scores'])
print(results[:1])


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()





