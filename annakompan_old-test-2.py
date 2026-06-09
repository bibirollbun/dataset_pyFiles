import os
import time
import random
import collections

import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torchvision import transforms
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2, roi_heads


# #file path in google drive
# #create a shortcut in your Drive.
# drive.mount('/content/drive')
# base_path = "/content/drive/MyDrive/Sartorius_project/"
# data_path = os.path.join(base_path, 'sartorius-cell-instance-segmentation')


# BASE_PATH = '/content/drive/MyDrive/NYCU/Sartorius_project'
# BASE_PATH = "./sartorius-cell-instance-segmentation"
BASE_PATH = "/kaggle/input/sartorius-cell-instance-segmentation"


# # Path to zip file
# zip_path = os.path.join(base_path, 'sartorius-cell-instance-segmentation.zip')
# extract_to = base_path

# # Extract only if not already extracted
# if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
#     print("Extracting data...")
#     os.makedirs(data_path, exist_ok=True)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print("Extraction completed to:", extract_to)
# else:
#     print("Data already extracted at:", data_path)


"""
Function to make results reproducible
"""
def __set__seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) #seed for cpu
    torch.cuda.manual_seed(seed) #seed for gpu
    torch.cuda.manual_seed_all(seed)
    
__set__seeds(2024)


TRAIN_PATH = f"{BASE_PATH}/train"
TEST_PATH = f"{BASE_PATH}/test"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
TEST = False


# CSV - for test read only 5000 rows
df = pd.read_csv(TRAIN_CSV, nrows=5000 if TEST else None)
df.head()


df.head()
cell_type_counts = df['cell_type'].value_counts()
print(cell_type_counts)


# Class histogram by cell type
plt.figure(figsize=(8, 4))
df['cell_type'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Cell Type Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Number of Masks")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


unique_ids = df['id'].unique()
print(f"Total unique images: {len(unique_ids)}")


from sklearn.model_selection import train_test_split

# 90% train, 10% validation
train_ids, val_ids = train_test_split(
    unique_ids,
    test_size=0.1,
    random_state=42,
    shuffle=True
)


train_df = df[df['id'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['id'].isin(val_ids)].reset_index(drop=True)

print(f"Train images: {train_df['id'].nunique()} | Val images: {val_df['id'].nunique()}")


train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

train_image_ids = train_df['id'].unique().tolist()
val_image_ids = val_df['id'].unique().tolist()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

def rle_decode(mask_rle, shape=(520, 704)):
    """Decode RLE (Run-Length Encoding) encoded masks."""
    s = mask_rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def visualize_sample(image_id, df, img_dir):
    img_path = os.path.join(img_dir, f"{image_id}.png")
    image = np.array(Image.open(img_path))

    masks = df[df['id'] == image_id]['annotation'].tolist()
    combined_mask = np.zeros_like(image, dtype=np.uint8)

    for i, rle in enumerate(masks):
        mask = rle_decode(rle)
        combined_mask += mask.astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(combined_mask, alpha=0.5, cmap='jet')
    plt.title('Image + Masks')
    plt.show()

# Example on id 0
visualize_sample(train_image_ids[0], df, TRAIN_PATH)



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

IMG_WIDTH = 704
IMG_HEIGHT = 520

# To normalize
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 2

MOMENTUM = 0.9
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

MASK_THRESHOLD = 0.5

# Normalize to resnet mean and std
NORMALIZE = True 

USE_SCHEDULER = True

NUM_EPOCHS = 50

BOX_DETECTIONS_PER_IMG = 539

# Dictionaries to classify each type of cell
CELL_TYPE_DICT = {"astro": 1, "cort": 2, "shsy5y": 3}
DICT_TO_CELL = {1: "astro", 2: "cort", 3: "shsy5y"}
MASK_THRESHOLD_DICT = {1: 0.55, 2: 0.75, 3:  0.6}
MIN_SCORE_DICT = {1: 0.55, 2: 0.75, 3: 0.5}
NUM_CLASSES = len(CELL_TYPE_DICT)


# def rle_decode_for_train(mask_rle, shape=(520, 704)):
#     if pd.isnull(mask_rle):
#         return np.zeros(shape, dtype=np.uint8)
#     s = mask_rle.strip().split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape)

# def rle_encode(mask):
#     pixels = mask.T.flatten()  # column line
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)

def rle_encoding(x):
    '''
    x : image to be encoded 
    Returns string of encoded image
    '''
    
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def draw_box(box):
    result = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3]) 
    
    for x in range(xmin, xmax):
        if (xmin != 0) and (xmax != IMG_WIDTH):
            result[ymin-1][x] = 1
            result[ymax-1][x] = 1
            
    for y in range(ymin, ymax):
        if (ymin != 0) and (ymax != IMG_HEIGHT):
            result[y][xmax-1] = 1
            result[y][xmin-1] = 1
            
    return result

# Get bbox of given mask
def get_box(a_mask):
    pos = np.where(a_mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])

    return [xmin, ymin, xmax, ymax]

# Mask + img
def combine_masks(masks, mask_threshold):
    maskimg = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    for m, mask in enumerate(masks,1):
        maskimg[mask>mask_threshold] = m
    return maskimg

# Mask + bbox
def combine_masks_boxes(masks, boxes):
    result = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    cur_max = 0
    for i in range(IMG_WIDTH):
        for j in range(IMG_HEIGHT):
            
            result[j][i] = 0
            
            if masks[j][i] != 0:
                result[j][i] = masks[j][i]
                if masks[j][i] > cur_max:
                    cur_max = masks[j][i]

    for i in range(IMG_WIDTH):
        for j in range(IMG_HEIGHT):
            if boxes[j][i] != 0:
                result[j][i] = cur_max                
    return result

"""
Filter masks using MIN_SCORE for mask 
and MAX_THRESHOLD for pixels
"""
def get_filtered_masks(pred):
    use_masks = []   
    for i, mask in enumerate(pred["masks"]):
        scr = pred["scores"][i].cpu().item()
        label = pred["labels"][i].cpu().item()
        if scr > MIN_SCORE_DICT[label]:
            mask = mask.cpu().numpy().squeeze()
            # Keep only highly likely pixels
            binary_mask = mask > MASK_THRESHOLD_DICT[label]
            binary_mask = remove_overlapping_pixels(binary_mask, use_masks)
            use_masks.append(binary_mask)

    return use_masks


"""
Computes the IoU for instance labels and predictions
"""
def compute_iou(labels, y_pred, verbose=0):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    if verbose:
        print("Number of true objects: {}".format(true_objects))
        print("Number of predicted objects: {}".format(pred_objects))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    # exclude background
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union
    
    return iou  

"""
Computes the precision at a given threshold.
"""
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn

"""
Computes the metric for the competition.
"""
def iou_map(truths, preds, verbose=0):
    ious = [compute_iou(truth, pred, verbose) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)

"""
Get average IOU mAP score for a dataset
"""
def get_score(ds, mdl):
    mdl.eval()
    iouscore = 0
    for i in tqdm(range(len(ds))):
        img, targets = ds[i]
        with torch.no_grad():
            result = mdl([img.to(DEVICE)])[0]
            
        masks = combine_masks(targets['masks'], 0.5)
        labels = pd.Series(result['labels'].cpu().numpy()).value_counts()

        mask_threshold = MASK_THRESHOLD_DICT[labels.sort_values().index[-1]]
        pred_masks = combine_masks(get_filtered_masks(result), mask_threshold)
        iouscore += iou_map([masks],[pred_masks])
    return iouscore / len(ds)


#Transform
# import torchvision.transforms as T
# transform = T.Compose([
#     T.Resize((512, 512)),
#     T.ToTensor(),
#     T.Normalize(
#       mean=[0.485, 0.456, 0.406],
#       std=[0.229, 0.224, 0.225])
# ])

"""
Additionally to vertical/horizontal flip (CUDA_OUT):
-> random rotation - cells can appear any orientation
-> gaussian blue - noise
"""

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
# class RandomRotate90:
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             # 90, 180, 270 degrees
#             k = random.choice([1, 2, 3])
#             # Rotate tensor image
#             image = image.rot90(k, [1, 2])
#             # Rotate masks
#             target["masks"] = target["masks"].rot90(k, [1, 2])

#             # Recalculate bboxes
#             h, w = image.shape[-2:]
#             boxes = []
#             for mask in target["masks"]:
#                 boxes.append(get_box(mask.numpy()))
#             target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
#         return image, target


# class GaussianBlur:
#     def __init__(self, prob=0.3, kernel_size=5):
#         self.prob = prob
#         self.kernel_size = kernel_size

#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             image = F.gaussian_blur(image, kernel_size=self.kernel_size)
#         return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
    
def get_transform(train):
    transforms = [ToTensor()]
    if NORMALIZE:
        transforms.append(Normalize())
    
    if train: 
        if NORMALIZE: 
            transforms.append(Normalize())
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))
        # transforms.append(RandomRotate90(0.5))
        # transforms.append(GaussianBlur(0.3))
        
    return Compose(transforms)


class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.height = IMG_HEIGHT
        self.width = IMG_WIDTH
        self.image_info = collections.defaultdict(dict)
        
        temp_df = self.df.groupby(['id', 'cell_type'])['annotation'].agg(lambda x: list(x)).reset_index()
        
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                    'image_id': row['id'],
                    'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                    'annotations': row["annotation"],
                    'cell_type': CELL_TYPE_DICT[row["cell_type"]]
                    }

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        info = self.image_info[idx]
        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []
        
        for i, annotation in enumerate(info['annotations']):
            
            a_mask = rle_decode(annotation, (IMG_HEIGHT, IMG_WIDTH))
            a_mask = Image.fromarray(a_mask)
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            boxes.append(get_box(a_mask))

        labels = [info["cell_type"] for _ in range(n_objects)]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.image_info)


def collate_fn(batch):
    return tuple(zip(*batch))


# DataLoader for training
train_dataset = CellDataset(TRAIN_PATH, df, transforms=get_transform(train=True))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                      num_workers=2, collate_fn=collate_fn)

# DataLoader for validation
val_dataset = CellDataset(TRAIN_PATH, val_df, transforms=get_transform(train=False))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=2, collate_fn=collate_fn)



def show_batch(images, targets, max_images=4):
    plt.figure(figsize=(16, 4 * max_images))

    for i in range(min(max_images, len(images))):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 0.229 + 0.485, 0, 1)  # denomarlization ImageNet

        masks = targets[i]["masks"].cpu().numpy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)

        plt.subplot(max_images, 2, 2 * i + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(max_images, 2, 2 * i + 2)
        plt.imshow(image)
        plt.imshow(combined_mask, alpha=0.5, cmap='viridis')
        plt.title("Image + Masks")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Récupère un batch et affiche
images, targets = next(iter(train_loader))
show_batch(images, targets)


# def sample_images(ncols):
#     '''
#     ncols : number of images to sample
#     Returns plot with 2x(ncols) images from training dataset 
#     '''
#     ns = random.sample(range(100), ncols)
#     fig, axs = plt.subplots(2, ncols, figsize=(20, 6)) 
    
#     for i in range(ncols):
#         img, targets = train_dataset[ns[i]]
#         masks = np.zeros((IMG_HEIGHT, IMG_WIDTH))
#         boxes = np.zeros((IMG_HEIGHT, IMG_WIDTH))
#         axs[0][i].set_title(f"Image {ns[i]}")
#         axs[0][i].imshow(img.numpy().transpose((1,2,0)))
#         axs[0][i].axis("off")
        

#         for mask in targets['masks']:
#             box = get_box(mask)
#             boxes = np.logical_or(boxes, draw_box(box))
            
#         masks = combine_masks(targets['masks'], 0.5)
            
#         axs[1][i].set_title(f"{ns[i]} {DICT_TO_CELL[((targets['labels'])[0]).item()]} mask")
#         detections = combine_masks_boxes(masks, boxes)
#         axs[1][i].imshow(detections)
#         axs[1][i].axis("off")
#     plt.show()
# sample_images(4)


# Load COCO-pretrained weights
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
state_dict = torch.load("//kaggle/input/pretrained/maskrcnn_resnet50_fpn_coco_0.15.1.pth")
model.load_state_dict(state_dict)

# Replace classification head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

# Replace mask head
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 128
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES + 1)

model.to(DEVICE)

for param in model.parameters():
    param.requires_grad = True
    
model.train()


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience} without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0


params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
n_batches, n_batches_val = len(train_loader), len(val_loader)


validation_mask_losses = []
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    time_start = time.time()
    loss_accum = 0.0
    loss_mask_accum = 0.0
    loss_classifier_accum = 0.0
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader), 1):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mask = loss_dict['loss_mask'].item()
        loss_accum += loss.item()
        loss_mask_accum += loss_mask
        loss_classifier_accum += loss_dict['loss_classifier'].item()
        
    # Train losses
    train_loss = loss_accum / n_batches
    train_loss_mask = loss_mask_accum / n_batches
    train_loss_classifier = loss_classifier_accum / n_batches

    if USE_SCHEDULER and epoch >= 5:
        lr_scheduler.step()
    
    # Validation
    # model.eval()
    val_loss_epoch = 0
    val_loss_mask_accum = 0
    val_loss_classifier_accum = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader), 1):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            val_loss_dict = model(images, targets)
            val_batch_loss = sum(loss for loss in val_loss_dict.values())
            val_loss_epoch += val_batch_loss.item()
            val_loss_mask_accum += val_loss_dict['loss_mask'].item()
            val_loss_classifier_accum += val_loss_dict['loss_classifier'].item()

    val_loss = val_loss_epoch / n_batches_val
    val_loss_mask = val_loss_mask_accum / n_batches_val
    val_loss_classifier = val_loss_classifier_accum / n_batches_val
    #time per epoch
    epoch_time = time.time() - time_start
    
    #for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    validation_mask_losses.append(val_loss_mask)
    
    torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
    print(f"[Epoch {epoch} / {NUM_EPOCHS}] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
    print(f"[Epoch {epoch} / {NUM_EPOCHS}] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
    print(f"[Epoch {epoch} / {NUM_EPOCHS}] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
    print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")


#training vall loss curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Curve")
plt.show()


# Plots: the image, The image + the ground truth mask + detection boxes, The image + the predicted mask + predicted boxes
def analyze_sample(model, train_dataset, sample_index):
    img, targets = train_dataset[sample_index]
    fig, axs = plt.subplots(1, 3, figsize=(20, 40), facecolor="#fefefe") 
    
    masks = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    boxes = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    
    axs[0].imshow(img.numpy().transpose((1,2,0)))
    axs[0].set_title("Image")
    axs[0].axis("off")
    
    for mask in targets['masks']:
        box = get_box(mask)
        boxes = np.logical_or(boxes, draw_box(box)) 
        
    masks = combine_masks(targets['masks'], 0.5)
    detections = combine_masks_boxes(masks, boxes)
    axs[1].imshow(detections)
    axs[1].set_title("Ground truth")
    axs[1].axis("off")
    
    model.eval()
    with torch.no_grad():
        preds = model([img.to(DEVICE)])[0]

    axs[2].imshow(img.cpu().numpy().transpose((1,2,0)))
    
    for mask in preds['masks'].cpu().detach():
        box = get_box(mask[[0]])
        boxes = np.logical_or(boxes, draw_box(box))
        
    l = pd.Series(preds['labels'].cpu().numpy()).value_counts()
    lstr = ""
    for i in l.index:
        lstr += f"{l[i]}x{i} "
    mask_threshold = MASK_THRESHOLD_DICT[l.sort_values().index[-1]]
    pred_masks = combine_masks(get_filtered_masks(preds), 0.5)
        
       
    detections = combine_masks_boxes(pred_masks, boxes)
    score = iou_map([masks],[pred_masks])
    axs[2].imshow(detections)
    axs[2].set_title(f"Predictions | IoU score: {score:.2f}")
    axs[2].axis("off")
    plt.show()
analyze_sample(model, train_dataset, random.randint(1, 500))



analyze_sample(model, train_dataset, random.randint(1, 500))


class CellTestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [f[:-4]for f in os.listdir(self.image_dir)]
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)
        return {'image': image, 'image_id': image_id}


ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))


model.eval()

submission = []
for sample in ds_test:
    img = sample['image']
    image_id = sample['image_id']
    with torch.no_grad():
        result = model([img.to(DEVICE)])[0]
    
    previous_masks = []
    for i, mask in enumerate(result["masks"]):
        score = result["scores"][i].cpu().item()
        mask = mask.cpu().numpy()
        # Keep only highly likely pixels
        binary_mask = mask > MASK_THRESHOLD
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
        previous_masks.append(binary_mask)
        rle = rle_encoding(binary_mask)
        submission.append((image_id, rle))
    
    # Add empty prediction if no RLE was generated for this image
    all_images_ids = [image_id for image_id, rle in submission]
    if image_id not in all_images_ids:
        submission.append((image_id, ""))

df_sub = pd.DataFrame(submission, columns=['id', 'predicted'])
df_sub.to_csv("submission.csv", index=False)
df_sub.head()

