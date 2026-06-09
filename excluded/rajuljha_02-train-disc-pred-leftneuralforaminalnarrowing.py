# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install pydicom -q
!pip uninstall torch torchvision -y
!pip install torch==2.1 torchvision==0.16 -q
!pip install -qU pycocotools
!pip install -qU wandb


for name in list(globals()):
    if not name.startswith("_"):  # Avoid deleting built-in and special variables
        del globals()[name]


CONDITION = 'LeftNeuralForaminalNarrowing'


import os
import time
import random
from datetime import datetime
import numpy as np
import collections

from matplotlib import animation, rc
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import tqdm
import sys
import torch


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device


# Directories
# PROJECT_DIR = '/home/jupyter'
DATA_DIR = os.path.join('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification', 'train_images')
SRC_DIR = os.path.join('/kaggle/input/rsna-code/rsna-2024-main', 'src')

CROP_DIR = '/kaggle/working'
os.makedirs(CROP_DIR, exist_ok=True)
MODEL_DIR = os.path.join('/kaggle/working/', 'models', '02_train_disc_detection', CONDITION)
os.makedirs(MODEL_DIR, exist_ok=True)


print(DATA_DIR)
print(SRC_DIR)
print(CROP_DIR)
print(MODEL_DIR)


os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")


import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ----------- Functions for disc detection -----------

def load_model_disc_detection(state_dict=None):
    
    # We use the lastest
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Replace the classifier with a new one, that has Num_classes
    num_classes = 6  # 5 classes (discs) + background

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Import parameters if available
    if state_dict:
        model.load_state_dict(state_dict)
    
    return model


# ----------- Functions for severity classification -----------

def load_model_severity_classification(state_dict=None):
    
    # We use the lastest
    model = torchvision.models.swin_v2_t(weights="DEFAULT")

    # Replace the classifier with a new one, that has Num_classes
    num_classes = 3 

    # Get number of input features for the classifier
    in_features = model.head.in_features

    # Replace the pre-trained head with a new one
    model.head = nn.Linear(in_features, num_classes)
    
    # Import parameters if available
    if state_dict:
        model.load_state_dict(state_dict)
    
    return model


file_disc_detection = """
# # ONLY NEED TO RUN ONCE
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")
import os
import utils
from engine import train_one_epoch, evaluate

# Util functions
with open(os.path.join(SRC_DIR, 'src/data.py')) as file:
    exec(file.read())
with open(os.path.join(SRC_DIR, 'models.py')) as file:
    exec(file.read())
    
    
# LEVEL_LABELS = {
#     "L1/L2": 1,
#     "L2/L3": 2,
#     "L3/L4": 3,
#     "L4/L5": 4,
#     "L5/S1": 5
# }

def model_pipeline(config, model, model_dir, train_df, val_df):

    # make the model, data, and optimization problem
    model, train_loader, val_loader, optimizer, lr_scheduler = make(config, model, train_df, val_df)

    # and use them to train the model
    train_and_validate(model, model_dir, train_loader, val_loader, optimizer, lr_scheduler, config)

    return


def make(config, model, train_df, val_df):
    
    # Make training set
    dataset = RSNAMultipleBBoxesDataset(train_df, w = config['box_w'], h_l1_l4 = config['box_h_l1_l4'], h_l5 = config['box_h_l5'])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=os.cpu_count()
    )
    
    # Make validation set
    dataset_val = RSNAMultipleBBoxesDataset(val_df, w = config['box_w'], h_l1_l4 = config['box_h_l1_l4'], h_l5 = config['box_h_l5'])
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=os.cpu_count()
    )

    # Make model
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=config['lr'],
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    return model, train_loader, val_loader, optimizer, lr_scheduler

def train_and_validate(model, model_dir, train_loader, val_loader, optimizer, lr_scheduler, config):
    
    for epoch in tqdm.tqdm(range(config['num_epochs']), desc="Training Epochs"):
        
        # train for one epoch, printing every 30 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=30)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the validation dataset
        evaluate(model, val_loader, device=device)
        
        # Save model after every epoch
        dirname = f'{model_dir}/epoch_{epoch}'
        os.makedirs(dirname, exist_ok=True,)
        fname = f'{dirname}/model_dict.pt'
        torch.save(model.state_dict(), fname)
"""


SRC_DIR = '/kaggle/input/rsna-code/rsna-2024-main'
exec(file_disc_detection)


CONFIG = dict(
    num_epochs=3,
    batch_size=10,
    lr=0.0001,
    lr_step_size=3,
    lr_gamma=0.1,
    box_w = 70, # width of the bounding boxes
    box_h_l1_l4 = 30, # height of the boxes for levels from L1/L2 to L4/L5
    box_h_l5 = 40 # width of the boxes for level L5/S1
)


train_df = pd.read_csv(os.path.join('/kaggle/input/rsna-metadata/kaggle 4/working', CONDITION, 'train.csv'))
test_df = pd.read_csv(os.path.join('/kaggle/input/rsna-metadata/kaggle 4/working', CONDITION, 'test.csv'))
val_df = pd.read_csv(os.path.join('/kaggle/input/rsna-metadata/kaggle 4/working', CONDITION, 'val.csv'))


train_df.head()


# This is coming because we executed the code of file_disc_detection which in turn executed the src/data.py which contains the implementation of RSNAMultipleBBoxesDataset class.
tmp_ds = RSNAMultipleBBoxesDataset(train_df, w = CONFIG['box_w'], h_l1_l4 = CONFIG['box_h_l1_l4'], h_l5 = CONFIG['box_h_l5'])
tmp_dl = torch.utils.data.DataLoader(
  tmp_ds,
  batch_size=1,
  shuffle=False,
  collate_fn=utils.collate_fn
)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,16))
for i, (img, target) in enumerate(tmp_dl):
    if i == 2: break
    img = img[0]
    target = target[0]
    print(img.shape)
    print(target.keys())
    y = img.squeeze().numpy()
    ax[i].imshow(y, cmap=plt.cm.bone)
    for j, box in enumerate(target['boxes']):
        x0, y0, x1, y1 = box.numpy()
        w = x1 - x0
        h = y1 - y0
        ax[i].add_patch(patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none'))
del tmp_ds, tmp_dl





model = load_model_disc_detection()


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters in the model: {count_trainable_parameters(model)}")


# Create dataset and dataloader
dataset = RSNAMultipleBBoxesDataset(train_df, w = CONFIG['box_w'], h_l1_l4 = CONFIG['box_h_l1_l4'], h_l5 = CONFIG['box_h_l5'])
train_loader = torch.utils.data.DataLoader(
  dataset,
  batch_size=2,
  shuffle=True,
  collate_fn=utils.collate_fn
)

# Get first input from dataloader
images, targets = next(iter(train_loader))
# print(images[0])
print((images))
# print((target))


# Create dataset and dataloader
dataset = RSNAMultipleBBoxesDataset(train_df, w = CONFIG['box_w'], h_l1_l4 = CONFIG['box_h_l1_l4'], h_l5 = CONFIG['box_h_l5'])
train_loader = torch.utils.data.DataLoader(
  dataset,
  batch_size=2,
  shuffle=True,
  collate_fn=utils.collate_fn
)

# Get first input from dataloader
images, targets = next(iter(train_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]

# Inference
model.to('cpu')
model.eval()
with torch.inference_mode():
    predictions = model(images)
    
# Inspect output
print(f"Keys in the prediction: {predictions[0].keys()}")
print(f"Shape of the predicted 'boxes' object: {predictions[0]['boxes'].shape}")  # [100 boxes, 4 dim (x0, y0, x1, y1)]
print(f"Coordinates of the first box: {torch.round(predictions[0]['boxes'][0]).tolist()}") # (x0, y0, x1, y1) of the first box
print(f"Label predictions of the first 5 boxes: {predictions[0]['labels'][:5]}")
print(f"Prediction scores of the first 5 boxes: {predictions[0]['scores'][:5]}") 


image_predicted = predictions[0]
targets_predicted = predictions[1]


targets_predicted


# torch.cuda.empty_cache()
# gc.collect()


model_pipeline(config=CONFIG, model=model, model_dir=MODEL_DIR, train_df=train_df, val_df=val_df)


trained_model = load_model_disc_detection(state_dict=torch.load(os.path.join(f"{MODEL_DIR}/epoch_{CONFIG['num_epochs']-1}/model_dict.pt")))


!zip -r disc_detection_left_neural_foraminal_narrowing.zip /kaggle/working/models/


from IPython.display import FileLink
FileLink(r'disc_detection_left_neural_foraminal_narrowing.zip')


LABELS_DICT = {
    1: "L1_L2",
    2: "L2_L3",
    3: "L3_L4",
    4: "L4_L5",
    5: "L5_S1"
}


def get_best_boxes(pred):
    best_boxes = {}

    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if label.item() not in best_boxes or score > best_boxes[label.item()]['score']:
            best_boxes[label.item()] = {'box': box.tolist(), 'score': score.item()}

    result = {
        'boxes': [entry['box'] for entry in best_boxes.values()],
        'labels': list(best_boxes.keys()),
        'scores': [entry['score'] for entry in best_boxes.values()]
    }

    return result

def plot_prediction(x, pred):
    x = x[0, :]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.imshow(x, cmap="bone")
    pred = get_best_boxes(pred)
    for i in range(len(pred['boxes'])):
        x0, y0, x1, y1 = pred['boxes'][i]
        label = pred['labels'][i]
        score = pred['scores'][i]
        h = y1 - y0
        w = x1 - x0
        ax.add_patch(patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        ax.text(x0+w+10, y0+h/2, f"{LABELS_DICT[label]} ({'{:.2f}'.format(score)})", color='r',fontsize=14)


# Plot predictions for a few validation samples
dataset_val = RSNAMultipleBBoxesDataset(val_df, w = CONFIG['box_w'], h_l1_l4 = CONFIG['box_h_l1_l4'], h_l5 = CONFIG['box_h_l5'])
val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn,
    num_workers=os.cpu_count()
)

for i, (images, targets) in enumerate(val_loader):
    if i == 5: break
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

    trained_model.to(device)
    trained_model.eval()
    with torch.inference_mode():
        predictions = trained_model(images)

    for i in range(len(images)):
        plot_prediction(images[i].cpu(), predictions[i])


def get_true_boxes(target):

    result = {
        'boxes': target['boxes'].tolist(),
        'labels': target['labels'].tolist()
    }

    return result

def crop_bbox(image, bbox):
    x0, y0, x1, y1 = bbox

    cropped_img = torchvision.transforms.functional.crop(
        image,
        top=round(int(y0)),
        left=round(int(x0)),
        height=round(int(y1 - y0)),
        width=round(int(x1 - x0))
    )
    return cropped_img


def plot_crop(image, bboxes):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(4,3))
    plt.subplots_adjust(top=2)

    for i in range(len(bboxes['boxes'])):
        label_i = bboxes['labels'][i] - 1
        label = LABELS_DICT[label_i + 1]
        score = bboxes['scores'][i]
        bbox = bboxes['boxes'][i]

        cropped_img = crop_bbox(image, bbox)
        cropped_img = cropped_img[0, :]
        # print(cropped_img.shape)

        ax[label_i].set_axis_off()
        ax[label_i].imshow(cropped_img, cmap="bone")
        ax[label_i].set_title(f"{label} ({'{:.2f}'.format(score)})")
        

def save_crop(image, bboxes, target):
    series_id = target['series_id']
    study_id = target['study_id']
    instance_number = target['instance_number']

    for i in range(len(bboxes['boxes'])):
        label = LABELS_DICT[bboxes['labels'][i]]

        dirname = f'{CROP_DIR}/{study_id}/{series_id}/{label}'
        os.makedirs(dirname, exist_ok=True)
        filepath = os.path.join(dirname, f'{instance_number}.pt')

        bbox = bboxes['boxes'][i]

        cropped_img = crop_bbox(image, bbox)
        torch.save(cropped_img, filepath)

    return


def crop_and_save_true_boxes(df, limit = None):
    
    dataset = RSNAMultipleBBoxesDataset(df, w = CONFIG['box_w'], h_l1_l4 = CONFIG['box_h_l1_l4'], h_l5 = CONFIG['box_h_l5'], limit=limit)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    for i, (images, targets) in enumerate(tqdm.tqdm(train_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        for i in range(len(images)):
            bboxes = get_true_boxes(targets[i])
            save_crop(images[i].cpu(), bboxes, targets[i])


crop_and_save_true_boxes(df = train_df, limit = None)


crop_and_save_true_boxes(df = val_df, limit = None)


def load_crop(row, plot=True, title=None, output=True):
    file = f"{CROP_DIR}/{row['study_id']}/{row['series_id']}/{LABELS_DICT[row['level_code']]}/{row['instance_number']}.pt"
    crop = torch.load(file).squeeze(0)
    if plot:
        plt.imshow(crop, cmap="bone")
        if title: plt.title(title)
        plt.show()
    if output:
        return crop


x = load_crop(train_df.iloc[0], plot=False, output=True)
trans = T.ToDtype(torch.float, scale=True)
y = trans(x)


train_df[train_df.level_code == 1].groupby('severity').head(n=1)


load_crop(train_df.iloc[0], plot=True, title='Normal/Mild', output=False)
load_crop(train_df.iloc[350], plot=True, title='Moderate', output=False)
load_crop(train_df.iloc[185], plot=True, title='Severe', output=False)


train_df[train_df.level_code == 2].groupby('severity').head(n=1)


load_crop(train_df.iloc[1], plot=True, title='Normal/Mild', output=False)
load_crop(train_df.iloc[51], plot=True, title='Moderate', output=False)
load_crop(train_df.iloc[86], plot=True, title='Severe', output=False)


train_df[train_df.level_code == 5].groupby('severity').head(n=1)


load_crop(train_df.iloc[4], plot=True, title='Normal/Mild', output=False)
load_crop(train_df.iloc[496], plot=True, title='Moderate', output=False)
load_crop(train_df.iloc[1955], plot=True, title='Severe', output=False)




