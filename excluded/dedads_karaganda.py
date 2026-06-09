!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/train_bbox_images.zip
!unzip -q train_bbox_images.zip -d train_bbox_images
!rm train_bbox_images.zip
!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/train_bbox_annotations.json
!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/train_segmentation_images.zip
!unzip -q train_segmentation_images.zip -d train_segmentation_images
!rm train_segmentation_images.zip
!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/train_segmentation_annotations.json

!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/evaluation_bbox_images.zip
!unzip -q evaluation_bbox_images.zip -d evaluation_bbox_images
!rm evaluation_bbox_images.zip
!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/evaluation_segmentation_images.zip
!unzip -q evaluation_segmentation_images.zip -d evaluation_segmentation_images
!rm evaluation_segmentation_images.zip
!wget https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/vacant-lot-detection/dataset/sample_submission.zip
!unzip -q sample_submission.zip -d sample_submission
!rm sample_submission.zip


# Loading all the libararies
!pip install ultralytics -q
!pip install solafune-tools
from solafune_tools.metrics import IOUBasedMetrics, bbox_to_polygon
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from glob import glob
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoImageProcessor
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import match_histograms


from glob import glob
shutil.rmtree('niga17', ignore_errors=True)
os.makedirs('niga17', exist_ok=True)
for path in glob('/kaggle/input/ruchnaya-razmetka-yolo-format2/labels/*'):
    with open(path, 'r') as f:
        infa = f.readlines()
    infa = ['0'+x[1:] for x in infa]
    path = path.replace('/kaggle/input/ruchnaya-razmetka-yolo-format2/labels', 'niga17')
    with open(path, 'w') as f:
        f.write(''.join(infa))


import shutil
# Input paths
image_dir = '/kaggle/input/ruchnaya-razmetka-yolo-format2/images'
label_dir = 'niga17'
output_dir = "/kaggle/working/dataset"


# Create YOLO-compatible directories
for split in ["train", "val"]:
    if os.path.exists(f"{output_dir}/images/{split}"):
        shutil.rmtree(f"{output_dir}/images/{split}")
    if os.path.exists(f"{output_dir}/labels/{split}"):
        shutil.rmtree(f"{output_dir}/labels/{split}")
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Get all .tif images
# train_imgs = [f for f in os.listdir(train_image_dir) if '.tif' in f or '.jpg' in f]
# val_imgs = [f for f in os.listdir(evaluation_image_dir) if '.tif' in f or '.jpg' in f]

# Split into train and val
all_images = [f for f in os.listdir(image_dir) if '.tif' in f or '.jpg' in f]
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=69)
# Function to copy images and labels
ebalsya_nah = []
def copy_files(images, split, image_dir, label_dir):
    suka_flag = False
    for img_file in images:
        base_name = os.path.splitext(img_file)[0]
        if '___' in base_name:
            suka_flag = True
        if suka_flag:
            nomer_suchki = base_name.split('___')[0] + '___'
            base_name = base_name.split('___')[-1]
        
        if base_name not in ['evaluation_222',
 'evaluation_196',
 'evaluation_157',
 'evaluation_115',
 'evaluation_118',
 'evaluation_14',
 'evaluation_240',
 'evaluation_350',
 'evaluation_223',
 'evaluation_334',]:
            img_src = os.path.join(image_dir, img_file)
            img_dst = os.path.join(output_dir, "images", split, img_file)
            shutil.copy(img_src, img_dst)
    
            label_file = f"{base_name}.txt"
            label_src = os.path.join(label_dir,label_file)
            if suka_flag:
                label_dst = os.path.join(output_dir, "labels", split, nomer_suchki+label_file)
            else:
                label_dst = os.path.join(output_dir, "labels", split, label_file)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
            else:
                ebalsya_nah.append(base_name)
                print(f"{img_file[:-4]}")

# Copy train and val sets
copy_files(train_imgs, "train", image_dir, label_dir)
copy_files(val_imgs, "val", image_dir, label_dir)


yaml_content = """
train: /kaggle/working/dataset/images/train
val: /kaggle/working/dataset/images/val
nc: 3
names: ['2redcar', 'artineon-gay', 'bober']
"""

with open('vacant_lot.yaml', 'w') as f:
    f.write(yaml_content)



model2 = YOLO('/kaggle/input/configs-yolo/swin_t-fpn.yaml', task='detect')
model2.train(data='vacant_lot.yaml', task='detect',single_cls=True, verbose=False, batch=32, amp=False, epochs=80, close_mosaic=0, hsv_h=0.1, hsv_s=0.7, hsv_v=0.6, fliplr=0.5, flipud=0.5, imgsz=640, iou=0.5, device=[0, 1, 2, 3])


# Paths
json_path = '/kaggle/input/razmetka-final-final2/train_psevdo_leak_big.json'
images_dir = '/kaggle/input/big-train/big_train'
labels_dir = '/kaggle/working/train_labels'

os.makedirs(labels_dir, exist_ok=True)
shutil.rmtree(labels_dir)
os.makedirs(labels_dir)
# Load JSON
with open(json_path) as f:
    data = json.load(f)

for image_info in data['images']:
    file_name = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    annotations = image_info['annotations']

    label_file = os.path.join(labels_dir, file_name.replace('.tif', '.txt').replace('.jpg', '.txt'))
    with open(label_file, 'w') as f:
        for ann in annotations:
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

# Input paths
train_image_dir = '/kaggle/input/big-train/big_train'
train_label_dir = '/kaggle/working/train_labels'
evaluation_image_dir = '/kaggle/input/big-train/big_train'
evaluation_label_dir = '/kaggle/working/train_labels'
output_dir = "/kaggle/working/dataset"
# Create YOLO-compatible directories
for split in ["train", "val"]:
    if os.path.exists(f"{output_dir}/images/{split}"):
        shutil.rmtree(f"{output_dir}/images/{split}")
    if os.path.exists(f"{output_dir}/labels/{split}"):
        shutil.rmtree(f"{output_dir}/labels/{split}")
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Get all .tif images
train_imgs = [f for f in os.listdir(train_image_dir) if '.tif' in f or '.jpg' in f]
val_imgs = [f for f in os.listdir(evaluation_image_dir) if '.tif' in f or '.jpg' in f]

# Split into train and val
#train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=69)
# Function to copy images and labels
ebalsya_nah = []
def copy_files(images, split, image_dir, label_dir):
    for img_file in images:
        base_name = os.path.splitext(img_file)[0]
        if base_name not in ['sexy_girl']:
            img_src = os.path.join(image_dir, img_file)
            img_dst = os.path.join(output_dir, "images", split, img_file)
    
            label_file = f"{base_name}.txt"
            label_src = os.path.join(label_dir, label_file)
            label_dst = os.path.join(output_dir, "labels", split, label_file)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
                shutil.copy(img_src, img_dst)
            else:
                ebalsya_nah.append(img_file[:-4])

# Copy train and val sets
copy_files(train_imgs, "train", train_image_dir, train_label_dir)
copy_files(val_imgs, "val", evaluation_image_dir, evaluation_label_dir)
yaml_content = """
train: /kaggle/working/dataset/images/train
val: /kaggle/working/dataset/images/val

nc: 1
names: ['vacant_lot']
"""

with open('vacant_lot.yaml', 'w') as f:
    f.write(yaml_content)


#since we use 10 models, you need to run this cell 10 times, with epochs [35, 95] and for each imgsz [576, 608, 640, 736, 800]
imgsz = 640
epochs = 35
model5 = YOLO('/usr/local/lib/python3.11/dist-packages/tests/tmp/runs/detect/train/weights/best.pt', task='detect')
model5.train(data='vacant_lot.yaml', task='detect', verbose=False, cache='disk', batch=24, amp=False, auto_augment=None, epochs=epochs, close_mosaic=30, translate=0.1, hsv_h=0.05, hsv_s=0.1, hsv_v=0.1, erasing=0, scale=0.0, fliplr=0.5, flipud=0.5, imgsz=imgsz, iou=0.5, device=[0, 1, 2, 3])
shutil.copy('/usr/local/lib/python3.11/dist-packages/tests/tmp/runs/detect/train2/weights/last.pt', f'{epochs}-{imgsz}')

