# The notebooks is self-contained
# It has very few imports
# No external dependencies (only the model weights)
# No train - inference notebooks
# We only rely on Pytorch
import os
import time
import random
import collections
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Fix randomness

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)


TRAIN_CSV = "../input/severstal-steel-defect-detection/train.csv"
TRAIN_PATH = "../input/severstal-steel-defect-detection/train_images"
TEST_PATH = "../input/severstal-steel-defect-detection/test_images"

WIDTH = 1600
HEIGHT = 256

# Reduced the train dataset to 500 rows
TEST = True

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 20

# No changes tried with the optimizer yet.
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Changes the confidence required for a pixel to be kept for a mask. 
# Only used 0.5 till now.
MASK_THRESHOLD = 0.5

# Normalize to resnet mean and std if True.
NORMALIZE = False 


# Use a StepLR scheduler if True. Not tried yet.
USE_SCHEDULER = False

# Number of epochs
NUM_EPOCHS = 1


BOX_DETECTIONS_PER_IMG = 539


MIN_SCORE = 0.59


# These are slight redefinitions of torch.transformation classes
# The difference is that they handle the target and the mask
# Copied from Abishek, added new ones
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target['masks'] = target['masks'].flip(-2)
            height = image.shape[-2]
            target['boxes'][:, [1, 3]] = height - target['boxes'][:, [3, 1]]
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target['masks'] = target['masks'].flip(-1)
            width = image.shape[-1]
            target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target

class Rotate:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, image, target):
        angle = random.choice(self.angles)
        center = (image.shape[-1] // 2, image.shape[-2] // 2)

        # Rotate image
        image = F.rotate(image, angle, center=center)

        # Rotate masks
        target['masks'] = F.rotate(target['masks'], angle, center=center)

        # Adjust bounding boxes
        radians = -angle * np.pi / 180.0  # Negative for clockwise rotation
        cx, cy = center
        boxes = target['boxes']
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])

            # Translate corners to origin
            corners[:, 0] -= cx
            corners[:, 1] -= cy

            # Rotate corners
            new_corners = np.dot(corners, np.array([
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)]
            ]))

            # Translate corners back
            new_corners[:, 0] += cx
            new_corners[:, 1] += cy

            # Get new bounding box
            x_min = np.min(new_corners[:, 0])
            y_min = np.min(new_corners[:, 1])
            x_max = np.max(new_corners[:, 0])
            y_max = np.max(new_corners[:, 1])

            new_boxes.append([x_min, y_min, x_max, y_max])

        target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)

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
    
    # Data augmentation for train
    # if train: 
    #     transforms.append(HorizontalFlip(0.5))
    #     transforms.append(VerticalFlip(0.5))
    #     transforms.append(Rotate([45, 135, 225, 315]))

    return Compose(transforms)


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


class DefectDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None, resize=False):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH
        
        # self.image_info = collections.defaultdict(dict)
        # temp_df = self.df.groupby('ImageId')['EncodedPixels'].agg(lambda x: list(x)).reset_index()
        # for index, row in temp_df.iterrows():
        #     self.image_info[index] = {
        #             'image_id': row['ImageId'],
        #             'image_path': os.path.join(self.image_dir, row['ImageId']),
        #             'annotations': row["EncodedPixels"]
        #             }
        image_info = collections.defaultdict(lambda: {
            'image_id': None,
            'image_path': None,
            'annotations': ['1 1','1 1','1 1','1 1']
        })
        # å�ˆå¹¶ annotations
        for index, row in self.df.iterrows():
            image_id = row['ImageId']
            class_id = row['ClassId']
            # æŒ‰ image_id å�ˆå¹¶
            image_info[image_id]['image_id'] = image_id
            image_info[image_id]['image_path'] = os.path.join(self.image_dir, row['ImageId'])
            image_info[image_id]['annotations'][class_id-1] = row['EncodedPixels']
        # key è½¬æ�¢æˆ� idx
        self.image_info = {i: value for i, value in enumerate(image_info.values())}
        with open('data.json', 'w') as json_file:
            json.dump(self.image_info, json_file)
    
    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def _oversample_by_class(self):
        """
        æ ¹æ�®ç±»åˆ«åˆ†å¸ƒå¯¹å°‘æ•°ç±»è¿›è¡Œè¿‡é‡‡æ ·ã€‚
        """
        class_counts = Counter(self.dataframe['class_label'])
        max_count = max(class_counts.values())

        oversampled_data = []
        for class_label, count in class_counts.items():
            class_data = self.dataframe[self.dataframe['class_label'] == class_label]
            oversampled_class = class_data.sample(max_count, replace=True)
            
            # å¯¹è¿‡é‡‡æ ·çš„æ•°æ�®è¿›è¡Œéš�æœºå¢�å¼º
            augmented_data = []
            for _, row in oversampled_class.iterrows():
                image_id = row['image_id']
                rle_mask = row['rle_mask']

                # åŠ è½½å›¾åƒ�å’Œè§£ç � mask
                img_path = f"{self.image_dir}/{image_id}.jpg"
                image = Image.open(img_path).convert("RGB")
                mask = rle_decode(rle_mask, self.image_shape)
                mask = Image.fromarray(mask * 255)

                # æ•°æ�®å¢�å¼º
                image, mask = random_augment(image, mask)

                augmented_data.append({
                    'image_id': image_id,
                    'rle_mask': rle_mask,
                    'class_label': class_label,
                    'image': image,
                    'mask': mask
                })

        return pd.DataFrame(augmented_data).reset_index(drop=True)

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []
        
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)
            
            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            
            boxes.append(self.get_box(a_mask))

        # dummy labels
        labels = [1 for _ in range(n_objects)]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
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


df_train = pd.read_csv(TRAIN_CSV, nrows=500 if TEST else None)
ds_train = DefectDataset(TRAIN_PATH, df_train, resize=False, transforms=get_transform(train=True))
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, 
                      num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


# Override pythorch checkpoint with an "offline" version of the file
!mkdir -p /root/.cache/torch/hub/checkpoints/
!cp ../input/cocopre/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth /root/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth


def get_model():
    # This is just a dummy value for the classification head
    NUM_CLASSES = 4
    
    if NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                   image_mean=RESNET_MEAN, 
                                                                   image_std=RESNET_STD)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                  box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    return model


# Get the Mask R-CNN model
# The model does classification, bounding boxes and MASKs for individuals, all at the same time
# We only care about MASKS
model = get_model()
model.to(DEVICE)

# TODO: try removing this for
for param in model.parameters():
    param.requires_grad = True
    
model.train();


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

n_batches = len(dl_train)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Starting epoch {epoch} of {NUM_EPOCHS}")
    
    time_start = time.time()
    loss_accum = 0.0
    loss_mask_accum = 0.0
    
    for batch_idx, (images, targets) in enumerate(dl_train, 1):
    
        # Predict
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        loss_mask = loss_dict['loss_mask'].item()
        loss_accum += loss.item()
        loss_mask_accum += loss_mask
        
        if batch_idx % 50 == 0:
            print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}")
    
    if USE_SCHEDULER:
        lr_scheduler.step()
    
    # Train losses
    train_loss = loss_accum / n_batches
    train_loss_mask = loss_mask_accum / n_batches
    
    
    elapsed = time.time() - time_start
    
    
    torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
    prefix = f"[Epoch {epoch:2d} / {NUM_EPOCHS:2d}]"
    print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
    print(f"{prefix} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")
     


# Plots: the image, The image + the ground truth mask, The image + the predicted mask
def analyze_train_sample(model, ds_train, sample_index):
    
    img, targets = ds_train[sample_index]
    plt.imshow(img.numpy().transpose((1,2,0)))
    plt.title("Image")
    plt.show()
    
    masks = np.zeros((HEIGHT, WIDTH))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
    plt.imshow(img.numpy().transpose((1,2,0)))
    plt.imshow(masks, alpha=0.3)
    plt.title("Ground truth")
    plt.show()
    
    model.eval()
    with torch.no_grad():
        preds = model([img.to(DEVICE)])[0]

    plt.imshow(img.cpu().numpy().transpose((1,2,0)))
    all_preds_masks = np.zeros((HEIGHT, WIDTH))
    for mask in preds['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > MASK_THRESHOLD)
    print(preds)
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title("Predictions")
    plt.show()





# NOTE: It puts the model in eval mode!! Revert for re-training
analyze_train_sample(model, ds_train, 20)


analyze_train_sample(model, ds_train, 100)


analyze_train_sample(model, ds_train, 2)


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

    def __len__(self):
        return len(self.image_ids)


ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))
ds_test[0]


def rle_encoding(x):
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


model.eval();

submission = []
for sample in ds_test:
    img = sample['image']
    image_id = sample['image_id']
    with torch.no_grad():
        result = model([img.to(DEVICE)])[0]
    
    previous_masks = []
    for i, mask in enumerate(result["masks"]):
        
        # Filter-out low-scoring results. Not tried yet.
        score = result["scores"][i].cpu().item()
        if score < MIN_SCORE:
            continue
        
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

