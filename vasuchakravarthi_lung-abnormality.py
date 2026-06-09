import torch
print("CUDA available?", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")



!pip install numpy<2.0 --force-reinstall

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



df = pd.read_csv('../input/vinbigdata-chest-xray-resized-png-1024x1024/train_meta.csv')
print(df.head())




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



annotations = pd.read_csv('/kaggle/input/vinbigdatachestxrayabnormalitiesdetection/train.csv')
train_meta = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024/train_meta.csv')
print(annotations.columns)
print(annotations.head())



annotations = pd.read_csv('/kaggle/input/vinbigdatachestxrayabnormalitiesdetection/train.csv')
print(annotations.columns)
print(annotations.head())



import cv2
import matplotlib.pyplot as plt

def plot_image_with_boxes(img_id, annotations, img_dir):
    img_path = f"{img_dir}/{img_id}.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get all findings (not 'No finding') for the image
    rows = annotations[(annotations['image_id'] == img_id) & (annotations['class_name'] != 'No finding')]
    for idx, row in rows.iterrows():
        x1, y1, x2, y2 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
        label = row['class_name']
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
plot_image_with_boxes('1c32170b4af4ce1a3030eb8167753b06', annotations, '/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024/train')



#List all classes from your DataFrame
print(sorted(annotations['class_name'].unique()))
print('Total classes:', len(annotations['class_name'].unique()))



#Example PyTorch Classes Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
annotations['class_idx'] = label_encoder.fit_transform(annotations['class_name'])
print(label_encoder.classes_)



import os
import numpy as np

# Paths (adjust as necessary)
IMG_DIR = '/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024/train'
YOLO_LABEL_DIR = '/kaggle/working/labels_yolo'  # Output dir for YOLO txt files

os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

# Get image dimensions from the metadata file if needed
meta_df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024/train_meta.csv')
img_dim_dict = dict(zip(meta_df['image_id'], zip(meta_df['dim1'], meta_df['dim0'])))

for img_id, group in annotations.groupby('image_id'):
    label_lines = []
    for _, row in group.iterrows():
        if row['class_name'] == 'No finding':  # Skip if no finding
            continue
        # Get original image width and height
        img_w, img_h = img_dim_dict[img_id]
        
        # YOLO format: <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        # Normalize
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        # YOLO class id (already encoded previously)
        class_idx = int(row['class_idx'])
        label_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        label_lines.append(label_line)
        
    # Write labels only if there are findings
    if label_lines:
        with open(f"{YOLO_LABEL_DIR}/{img_id}.txt", "w") as f:
            for l in label_lines:
                f.write(l + "\n")

print("YOLO label file conversion completed.")
print(f"Sample txt: {YOLO_LABEL_DIR}/{img_id}.txt")



!pip install ultralytics



import shutil
import random
from glob import glob

IMG_SRC = IMG_DIR  # Your VinBigData images folder
TXT_SRC = YOLO_LABEL_DIR  # Your YOLO labels
TRAIN_IMG_DST = '/kaggle/working/dataset/images/train/'
VAL_IMG_DST = '/kaggle/working/dataset/images/val/'
TRAIN_LABEL_DST = '/kaggle/working/dataset/labels/train/'
VAL_LABEL_DST = '/kaggle/working/dataset/labels/val/'

for d in [TRAIN_IMG_DST, VAL_IMG_DST, TRAIN_LABEL_DST, VAL_LABEL_DST]:
    os.makedirs(d, exist_ok=True)

all_imgs = glob(f"{IMG_SRC}/*.png")
random.shuffle(all_imgs)
split_idx = int(0.8 * len(all_imgs))  # 80/20 split
train_imgs, val_imgs = all_imgs[:split_idx], all_imgs[split_idx:]

# Move train images and labels
for img_path in train_imgs:
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    shutil.copy(img_path, TRAIN_IMG_DST)
    label_path = f"{TXT_SRC}/{img_id}.txt"
    if os.path.exists(label_path):
        shutil.copy(label_path, TRAIN_LABEL_DST)
# Move val images and labels
for img_path in val_imgs:
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    shutil.copy(img_path, VAL_IMG_DST)
    label_path = f"{TXT_SRC}/{img_id}.txt"
    if os.path.exists(label_path):
        shutil.copy(label_path, VAL_LABEL_DST)



print("Train images:", len(glob('/kaggle/working/dataset/images/train/*.png')))
print("Train labels:", len(glob('/kaggle/working/dataset/labels/train/*.txt')))
print("Val images:", len(glob('/kaggle/working/dataset/images/val/*.png')))
print("Val labels:", len(glob('/kaggle/working/dataset/labels/val/*.txt')))
print("Sample train images:", glob('/kaggle/working/dataset/images/train/*.png')[:5])
print("Sample val labels:", glob('/kaggle/working/dataset/labels/val/*.txt')[:5])



!pip uninstall -y numpy
!pip install numpy==1.26.4



# Write the YOLO data config
data_yaml = """
train: /kaggle/working/dataset/images/train
val: /kaggle/working/dataset/images/val
nc: 15
names: ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'No finding',
        'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
        'Pneumothorax', 'Pulmonary fibrosis']
"""
with open('/kaggle/working/data.yaml', 'w') as f:
    f.write(data_yaml)

# Model training
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # or yolov8s.pt for small model
# Large images, smaller batch
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=40,
    imgsz=1280,      # Higher resolution
    batch=4,         # Lower batch size to fit in RAM
    device=0,
    save_period=1
)







!pip install ultralytics --upgrade --quiet



from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=40,
    imgsz=1024,
    batch=8,
    device=0
)



import os
print(os.listdir('/kaggle/working/runs/detect/train/'))   # adjust if path is different





