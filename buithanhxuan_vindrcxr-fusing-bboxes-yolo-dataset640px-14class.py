!pip install -q ensemble-boxes


%matplotlib inline

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False})
import cv2
import json
import pandas as pd
import glob
import os.path as osp
from path import Path
import datetime
import numpy as np
from tqdm.auto import tqdm
import random
import shutil
from sklearn.model_selection import train_test_split

from ensemble_boxes import *
import warnings
from collections import Counter


train_annotations = pd.read_csv("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
train_annotations.head(5)


train_annotations = train_annotations[train_annotations.class_id!=14]
train_annotations['image_path'] = train_annotations['image_id'].map(lambda x:os.path.join('/kaggle/input/vinbigdata-chest-xray-original-png/train', str(x)+'.png'))
train_annotations.head(5)


img_size_df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-original-png/train_meta.csv', names=['image_id', 'orig_height', 'orig_width'])

# Merge the original image size information with the training data
train_annotations = train_annotations.merge(img_size_df, on='image_id', how='left')
train_annotations.head(5)


imagepaths = train_annotations['image_path'].unique()
print("Number of Images with abnormalities:",len(imagepaths))
anno_count = train_annotations.shape[0]
print("Number of Annotations with abnormalities:", anno_count)


def plot_img(img, size=(18, 18), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=2, size=10, is_rgb=True, title="", cmap='gray', img_size=None):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    
def draw_bbox(image, box, label, color):   
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output


labels =  [
            "__ignore__",
            "Aortic_enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung_Opacity",
            "Nodule/Mass",
            "Other_lesion",
            "Pleural_effusion",
            "Pleural_thickening",
            "Pneumothorax",
            "Pulmonary_fibrosis"
            ]
viz_labels = labels[1:]


# map label_id to specify color
#label2color = [[random.randint(0,255) for i in range(3)] for class_id in viz_labels]
label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

thickness = 3
imgs = []

for img_id, path in zip(train_annotations['image_id'][:6], train_annotations['image_path'][:6]):

    boxes = train_annotations.loc[train_annotations['image_id'] == img_id,
                                  ['x_min', 'y_min', 'x_max', 'y_max']].values
    img_labels = train_annotations.loc[train_annotations['image_id'] == img_id, ['class_id']].values.squeeze()
    
    img = cv2.imread(path)
    
    for label_id, box in zip(img_labels, boxes):
        color = label2color[label_id]
        img = draw_bbox(img, list(np.int_(box)), viz_labels[label_id], color)
    imgs.append(img)

plot_imgs(imgs, size=9, cmap=None)
plt.show()


iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []

for i, path in tqdm(enumerate(imagepaths[5:8])):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
    print(f"(\'{image_basename}\', \'{path}\')")
    img_annotations = train_annotations[train_annotations.image_id==image_basename]

    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    print("Bboxes before nms:\n", boxes_viz)
    print("Labels before nms:\n", labels_viz)
    
    ## Visualize Original Bboxes
    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []
    
    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(img_annotations['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:       
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
            
    # Perform NMS
    boxes, scores, box_labels = nms(boxes_list, scores_list, labels_list, weights=weights,
                                    iou_thr=iou_thr)
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()

    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    print("Bboxes after nms:\n", boxes)
    print("Labels after nms:\n", box_labels)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
    print()
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"Non-max Suppression", va="top", ha="center", size=25)
plt.savefig('nms.png', bbox_inches='tight')
plt.show()


iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []
sigma = 0.1

for i, path in tqdm(enumerate(imagepaths[5:8])):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
    print(f"(\'{image_basename}\', \'{path}\')")
    img_annotations = train_annotations[train_annotations.image_id==image_basename]
    
    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    print("Bboxes before soft_nms:\n", boxes_viz)
    print("Labels before soft_nms:\n", labels_viz)
    
    ## Visualize Original Bboxes
    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []
    
    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(img_annotations['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:       
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
            
        
    # Perform Soft-NMS
    boxes, scores, box_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights,
                                         iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()
    
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    print("Bboxes after soft_nms:\n", boxes)
    print("Labels after soft_nms:\n", box_labels)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
    print()
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"Soft NMS", va="top", ha="center", size=25)
plt.savefig('snms.png', bbox_inches='tight')
plt.show()


iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []

for i, path in tqdm(enumerate(imagepaths[5:8])):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
    print(f"(\'{image_basename}\', \'{path}\')")
    img_annotations = train_annotations[train_annotations.image_id==image_basename]

    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    print("Bboxes before non_maximum_weighted:\n", boxes_viz)
    print("Labels before non_maximum_weighted:\n", labels_viz)
    
    ## Visualize Original Bboxes
    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []
    
    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(img_annotations['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:       
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
            

    # Perform Non-maximum Weighted
    boxes, scores, box_labels = non_maximum_weighted(boxes_list, scores_list, labels_list,
                                                     weights=weights, iou_thr=iou_thr,skip_box_thr=skip_box_thr)
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()

    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    print("Bboxes after non_maximum_weighted:\n", boxes)
    print("Labels after non_maximum_weighted:\n", box_labels)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
    print()
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"Non-maximum Weighted", va="top", ha="center", size=25)
plt.savefig('nmw.png', bbox_inches='tight')
plt.show()


iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []
sigma = 0.1

for i, path in tqdm(enumerate(imagepaths[5:8])):
    img_array  = cv2.imread(path)
    image_basename = Path(path).stem
    print(f"(\'{image_basename}\', \'{path}\')")
    img_annotations = train_annotations[train_annotations.image_id==image_basename]

    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()
    
    print("Bboxes before WBF:\n", boxes_viz)
    print("Labels before WBF:\n", labels_viz)
    
    ## Visualize Original Bboxes
    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_before)
    
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    
    boxes_single = []
    labels_single = []
    
    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(img_annotations['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:       
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
            

    # Perform WBF
    boxes, scores, box_labels= weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                     iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    
    boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()

    boxes.extend(boxes_single)
    box_labels.extend(labels_single)
    
    print("Bboxes after WBF:\n", boxes)
    print("Labels after WBF:\n", box_labels)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
    print()
        
plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
plt.savefig('wbf.png', bbox_inches='tight')
plt.show()


# First, let's add back the images with class_id 14 (no findings)
all_annotations = pd.read_csv("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
all_annotations['image_path'] = all_annotations['image_id'].map(lambda x:os.path.join('/kaggle/input/vinbigdata-chest-xray-original-png/train', str(x)+'.png'))
all_annotations.head(5)


# # Calculate Class Weights

# import numpy as np

# # Count the number of annotations per class
# class_counts = all_annotations['class_id'].value_counts().to_dict()

# # Total number of samples
# total_samples = sum(class_counts.values())

# # Calculate class weights based on inverse frequency
# class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}

# print(f"Class Weights: {class_weights}")


# Calculate Class Weights

import numpy as np

# Count the number of annotations per class
class_counts = all_annotations['class_id'].value_counts().to_dict()

# Loáº¡i bá»� class_id 14 náº¿u nÃ³ tá»“n táº¡i
if 14 in class_counts:
    del class_counts[14]

# Total number of samples (khÃ´ng bao gá»“m class_id 14)
total_samples = sum(class_counts.values())

# Calculate class weights based on inverse frequency (bá»� qua class_id 14)
class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items() if cls != 14}

print(f"Class Weights: {class_weights}")


# Set random seed for reproducibility
np.random.seed(42)

# # Lá»�c ra cÃ¡c áº£nh khÃ´ng chá»©a class_id 14
# images_without_class14 = all_annotations[all_annotations['class_id'] != 14]['image_id'].unique()

# # Get unique image IDs (Ä‘Ã£ loáº¡i bá»� áº£nh chá»©a class_id 14)
# image_ids = images_without_class14

# Get unique image IDs
image_ids = all_annotations['image_id'].unique()

# Split the data
train_ids, temp_ids = train_test_split(image_ids, test_size=0.25, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.2, random_state=42)

print(f"Train set: {len(train_ids)} images")
print(f"Validation set: {len(val_ids)} images")
print(f"Test set: {len(test_ids)} images")


from pathlib import Path

# Create directories for YOLO dataset
yolo_dir = Path('./vinbigdata-yolo-dataset-with-wbf-640px-14class')
yolo_dir.mkdir(exist_ok=True)

for split in ['train', 'val', 'test']:
    (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)


from typing import List
def ls(path: Path) -> List[Path]:
    return list(path.iterdir())


ls(yolo_dir)


# Function to convert bbox to YOLO format
def convert_to_yolo_format(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2 / img_width
    y_center = (box[1] + box[3]) / 2 / img_height
    width = (box[2] - box[0]) / img_width
    height = (box[3] - box[1]) / img_height
    return [x_center, y_center, width, height]


# Prepare the datasets
viz_images = {
    'train': [],
    'val': [],
    'test': []
}


# Prepare the datasets
for split, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    saved_images_count = 0  # Counter to track saved images per split
    for img_id in tqdm(ids, desc=f"Processing {split} set"):
        img_annotations = all_annotations[all_annotations.image_id == img_id]
        img_path = img_annotations.iloc[0]['image_path']
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Get original image dimensions
        orig_height, orig_width = img.shape[:2]
        
        # Resize the image to 640x640 pixels
        img = cv2.resize(img, (640, 640))
        img_height, img_width = 640, 640  # Set the new width and height after resizing
        
        # Calculate scale factors for width and height
        width_scale = img_width / orig_width
        height_scale = img_height / orig_height
        
        # Save the resized image to the appropriate directory
        output_img_path = yolo_dir / split / 'images' / f"{img_id}.jpg"
        cv2.imwrite(str(output_img_path), img)  # Save the resized image
        
        # Prepare annotations
        with open(yolo_dir / split / 'labels' / f"{img_id}.txt", 'w') as f:
            if 14 in img_annotations['class_id'].values:
                # No findings case
                # f.write("14 0.5 0.5 1 1\n")
                # bá»� qua lá»›p nofinding hay hÃ¬nh khÃ´ng cÃ³ bá»‡nh
                print(f"Skip image(label14): {img_path}")
                continue
            else:
                boxes_list = []
                scores_list = []
                labels_list = []
                
                cls_ids = img_annotations['class_id'].unique().tolist()
                for cid in cls_ids:
                    cls_annotations = img_annotations[img_annotations.class_id == cid]
                    cls_boxes = cls_annotations[['x_min', 'y_min', 'x_max', 'y_max']].values
                    
                    # Scale bounding box coordinates based on original image size before resize
                    cls_boxes[:, [0, 2]] = cls_boxes[:, [0, 2]] * width_scale  # Adjust x_min and x_max
                    cls_boxes[:, [1, 3]] = cls_boxes[:, [1, 3]] * height_scale  # Adjust y_min and y_max
                    
                    # Normalize box coordinates based on resized image dimensions (1280x1280)
                    cls_boxes = cls_boxes / [img_width, img_height, img_width, img_height]
                    cls_boxes = np.clip(cls_boxes, 0, 1)  # Ensure boxes are within bounds
                    
                    boxes_list.append(cls_boxes.tolist())
                    scores_list.append(np.ones(len(cls_boxes)).tolist())
                    labels_list.append([cid] * len(cls_boxes))
                
                # Visualize original bounding boxes before WBF
                if saved_images_count < 5:  # Only save up to 5 images per split
                    img_before = img.copy()
                    
                    # Flatten and ensure all boxes have 4 elements
                    flat_boxes = [box for sublist in boxes_list for box in sublist if len(box) == 4]
                    flat_labels = [label for sublist in labels_list for label in sublist]

                    for box, label in zip(flat_boxes, flat_labels):
                        # Calculate bounding box coordinates based on resized image dimensions
                        x_min, y_min, x_max, y_max = np.array(box) * [img_width, img_height, img_width, img_height]
                        color = label2color[int(label)]
                        img_before = draw_bbox(img_before, [int(x_min), int(y_min), int(x_max), int(y_max)], viz_labels[int(label)], color)
                    
                # Apply WBF (Weighted Boxes Fusion)
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list, 
                    weights=None, iou_thr=0.5, skip_box_thr=0.0001
                )
                
                # Write YOLO format annotations
                for box, label in zip(boxes, labels):
                    # YOLO expects normalized coordinates, so we don't need to scale them back
                    yolo_box = convert_to_yolo_format(box, 1, 1)  # already normalized
                    f.write(f"{int(label)} {' '.join(map(str, yolo_box))}\n")
                
                # Visualize bounding boxes after WBF
                if saved_images_count < 5:  # Only save up to 5 images per split
                    img_after = img.copy()
                    for box, label in zip(boxes, labels):
                        # Calculate bounding box coordinates based on resized image dimensions
                        x_min, y_min, x_max, y_max = np.array(box) * [img_width, img_height, img_width, img_height]
                        color = label2color[int(label)]
                        img_after = draw_bbox(img_after, [int(x_min), int(y_min), int(x_max), int(y_max)], viz_labels[int(label)], color)
                    
                    # Append both images (before and after WBF) to visualization list
                    viz_images[split].append((img_before, img_after))
                    saved_images_count += 1

print("YOLO dataset creation and visualization completed.")


# Visualize and save images before and after WBF
for split, images in viz_images.items():
    for idx, (img_before, img_after) in enumerate(images):
        plt.figure(figsize=(20, 10))

        # Plot image before WBF
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        plt.title(f"{split.capitalize()} Image {idx+1} Before WBF")
        plt.axis('off')

        # Plot image after WBF
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        plt.title(f"{split.capitalize()} Image {idx+1} After WBF")
        plt.axis('off')

        plt.savefig(f'{split}_image_{idx+1}_comparison.png', bbox_inches='tight')
        plt.show()


# Create train.txt, val.txt, and test.txt
for split in ['train', 'val', 'test']:
    with open(yolo_dir / f"{split}.txt", 'w') as f:
        for img_path in (yolo_dir / split / 'images').glob('*.jpg'):
            # Replace "working" in img_path with the specified value
            modified_img_path = str(img_path.absolute()).replace("working", "input/vinbigdata-yolo-dataset-with-wbf-640px-14class")
            f.write(f"{modified_img_path}\n")

print("Dataset split files created.")


data_input_dir = Path("/kaggle/input/vinbigdata-yolo-dataset-with-wbf-640px-14class/vinbigdata-yolo-dataset-with-wbf-640px-14class")

# Add class weights to the YAML configuration
class_weight_list = [class_weights[cls] for cls in sorted(class_weights.keys())]

data_yaml = f"""
train: {(data_input_dir / 'train.txt')}
val: {(data_input_dir / 'val.txt')}
test: {(data_input_dir / 'test.txt')}

nc: {len(viz_labels)}
names: {viz_labels}
class_weights: {class_weight_list}
"""

with open(yolo_dir / 'data.yaml', 'w') as f:
    f.write(data_yaml)

print("data.yaml file with class weights created.")


warnings.filterwarnings("ignore", category=UserWarning)


!find ./vinbigdata-yolo-dataset-with-wbf-640px-14class/train -type f | wc -l


!find ./vinbigdata-yolo-dataset-with-wbf-640px-14class/val -type f | wc -l


!find ./vinbigdata-yolo-dataset-with-wbf-640px-14class/test -type f | wc -l


%%bash
cd ./vinbigdata-yolo-dataset-with-wbf-640px-14class
zip -rq ../vinbigdata-yolo-dataset-with-wbf-640px-14class.zip ./*
echo "Zipping completed successfully."


%%bash
rm -r ./vinbigdata-yolo-dataset-with-wbf-640px-14class
ls -ahl

