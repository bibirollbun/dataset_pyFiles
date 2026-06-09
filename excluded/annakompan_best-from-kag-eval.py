import torch
torch.cuda.empty_cache()


from google.colab import drive
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
import collections
import torchvision
import time
import matplotlib.patches as patches
import cv2
from torchvision.transforms import functional as F

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from functools import partial


#file path in google drive
#create a shortcut in your Drive.
# from google.colab import drive
# drive.mount('/content/drive')

# !cp -r /content/drive/MyDrive/sartorius-cell-instance-segmentation /content/


data_path = "/kaggle/input/sartorius-cell-instance-segmentation"
# data_path = "/content/sartorius-cell-instance-segmentation"


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



# get the img and mask paths
# TRAIN_CSV = f"{data_dir}/train.csv"
# TRAIN_PATH = f"{data_dir}/train"
# TEST_PATH = f"{data_dir}/test"

test_img_path = f"{data_path}/test"
train_img_path = f"{data_path}/train"
train_df_path = f"{data_path}/train.csv" # annotations (image IDs + RLE masks)


# CSV
df = pd.read_csv(train_df_path)
df.head()


df = pd.read_csv(train_df_path)
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

# 80% train, 20% validation
train_ids, val_ids = train_test_split(
    unique_ids,
    test_size=0.2,
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

WIDTH = 704
HEIGHT = 520

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
visualize_sample(train_image_ids[0], train_df, train_img_path)



#Transform
import torchvision.transforms as T

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
NORMALIZE = False
# transform = T.Compose([
#     T.Resize((512, 512)),
#     T.ToTensor(),
#     T.Normalize(
#       mean=RESNET_MEAN,
#       std=RESNET_STD)
# ])

# for not pretrained model
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# transform = A.Compose(
#   [
#     A.RandomCrop(512,512),
#     A.HorizontalFlip(p=0.5),
#     A.Rotate(limit=30, p=0.5),
#     A.Normalize(mean=RESNET_MEAN, std=RESNET_STD),
#     ToTensorV2()
#   ],
#   bbox_params=A.BboxParams(
#     format='pascal_voc',     # for[xmin,ymin,xmax,ymax]
#     label_fields=['labels']
#   )
# )


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

    # Data augmentation for train
    if train:
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)


cell_type_dict = {
    'shsy5y': 1,
    'astro': 2,
    'cort': 3
}


def rle_decode(mask_rle, shape=(520, 704)):
    if pd.isnull(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None, resize=False):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df

        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
            print("image size used:", self.height, self.width)
        else:
            self.height = HEIGHT
            self.width = WIDTH

        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby(["id", "cell_type"])['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                    'image_id': row['id'],
                    'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                    'annotations': list(row["annotation"]),
                    'cell_type': cell_type_dict[row["cell_type"]]
                    }

    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        ''' Get the image and the target'''

        img_path = self.image_info[idx]["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.should_resize:
            img = cv2.resize(img, (self.width, self.height))

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []
        labels = []
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))

            if self.should_resize:
                a_mask = cv2.resize(a_mask, (self.width, self.height))

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask

            boxes.append(self.get_box(a_mask))

        # labels
        labels = [int(info["cell_type"]) for _ in range(n_objects)]
        #labels = [1 for _ in range(n_objects)]


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



def collate_fn(batch):
    return tuple(zip(*batch))

resize_factor = False


train_dataset = CellDataset(train_img_path, train_df, resize=resize_factor, transforms=get_transform(train=True))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True,
                      num_workers=2, collate_fn=collate_fn)

val_dataset = CellDataset(train_img_path, val_df, resize=resize_factor, transforms=get_transform(train=False))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True,
                    num_workers=2, collate_fn=collate_fn)


# train_dataset = CellDataset(
#     df=train_df,
#     image_dir=train_img_path,
#     transform=transform,
#     augment=True,
#     cls_map=cls_map
# )
# # DataLoader for training
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=2,  # increase if possible
#     collate_fn=collate_fn
# )

# # DataLoader for validation
# val_dataset = CellDataset(
#     df=val_df,
#     image_dir=train_img_path,
#     transform=transform,
#     augment=False,
#     cls_map=cls_map
# )
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=4,
#     shuffle=False,
#     collate_fn=collate_fn
# )


def show_batch(images, targets, max_images=4):
    plt.figure(figsize=(16, 4 * max_images))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(max_images, len(images))):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = std * image + mean  # full channel-wise denorm
        image = np.clip(image, 0, 1)

        masks = targets[i]["masks"].cpu().numpy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for mask in masks:
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
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

# image = image.resize(self.image_size, resample=Image.BILINEAR)  # image is 512x512
show_batch(images, targets)

# Image shape: (520, 704, 3)
# Mask shape: (520, 704)


print(df.columns)


import torch.nn as nn
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

MOMENTUM = 0.9
# early stop patience 10/ epochs ~
# LR = 0.005
# LR = 1e-3 ~17 epoch/7 early stop
WEIGHT_DECAY = 0.0005
MASK_THRESHOLD = 0.5 #0.05
PATIENCE = 3
NUM_CLASSES = 3
WIDTH = 704
HEIGHT = 520
USE_SCHEDULER = True

BATCH_SIZE = 2
# EPOCHS = 20
EPOCHS = 50
# USE_SCHEDULER = False
BOX_DETECTIONS_PER_IMG = 539

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
#                                                                    box_detections_per_img=BOX_DETECTIONS_PER_IMG,
#                                                                    image_mean=RESNET_MEAN,
#                                                                    image_std=RESNET_STD)

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES+1)
# in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# hidden_layer = 256
# model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES+1)

# GroupNorm instead of BatchNorm for stability at small batch
gn = lambda num_channels: nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-5)

backbone = resnet_fpn_backbone(
    'resnet50',
    pretrained=False,
    norm_layer=gn
)


# model = MaskRCNN(backbone=backbone, num_classes=NUM_CLASSES+1)

model = MaskRCNN(
    backbone=backbone,
    num_classes=NUM_CLASSES+1,
    image_mean=list(RESNET_MEAN),
    image_std=list(RESNET_STD)
)

model.to(DEVICE)

# for param in model.parameters():
#     param.requires_grad = True

# model.train()


# scale down the default Kaiming init in RPN, box & mask heads
def init_head(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=1)
        m.weight.data *= 0.1
        if m.bias is not None:
            m.bias.data.zero_()

model.rpn.head.apply(init_head)
model.roi_heads.box_head.apply(init_head)
model.roi_heads.mask_head.apply(init_head)

# freeze all BatchNorm layers
for m in model.backbone.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

# at first train heads only for 5 epochs
for name, p in model.backbone.named_parameters():
    p.requires_grad = False

head_params = [p for p in model.parameters() if p.requires_grad]
opt_stage1 = torch.optim.SGD(
    head_params, lr=1e-4, momentum=0.9, weight_decay=WEIGHT_DECAY
)

# Warm-up scheduler: linearly ramp from 0→1 over the first 500 steps
def warmup_lambda(step):
    return min((step + 1) / 500, 1.0)

warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt_stage1, warmup_lambda)

# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=LR,
#     momentum=MOMENTUM,
#     weight_decay=WEIGHT_DECAY
# )

# # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                           mode='min',
#                                                           factor=0.5,
#                                                           patience=PATIENCE,
#                                                           verbose=True)
n_batches, n_batches_val = len(train_loader), len(val_loader)

validation_mask_losses = []
train_losses = []
val_losses = []

for epoch in range(1, 10+1):
  print(f"Starting epoch {epoch} of 10")
  time_start = time.time()
  epoch_loss = 0.0
  loss_mask_accum = 0.0
  loss_classifier_accum = 0.0
  for images, targets in train_loader:
    images  = [img.to(DEVICE) for img in images]
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    opt_stage1.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt_stage1.step()
    warmup_sched.step()

    loss_mask = loss_dict['loss_mask'].item()
    epoch_loss += loss.item()
    loss_mask_accum += loss_mask
    loss_classifier_accum += loss_dict['loss_classifier'].item()

  train_loss = epoch_loss / n_batches
  train_loss_mask = loss_mask_accum / n_batches
  train_loss_classifier = loss_classifier_accum / n_batches

  val_loss_epoch = 0
  val_loss_mask_accum = 0
  val_loss_classifier_accum = 0

  with torch.no_grad():
      for batch_idx, (images, targets) in enumerate(val_loader, 1):
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

  print(f"[Epoch {epoch} / 10] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / 10] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / 10] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / 10] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
  print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")


for p in model.backbone.parameters():
    p.requires_grad = True

backbone_params, head_params = [], []
for name, p in model.named_parameters():
    (backbone_params if "backbone" in name else head_params).append(p)

opt_stage2 = torch.optim.SGD([
    {"params": head_params, "lr": 1e-3},
    {"params": backbone_params, "lr": 1e-5},
], momentum=0.9, weight_decay=WEIGHT_DECAY)

# Step LR or ReduceLROnPlateau on val loss?
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_stage2, step_size=10, gamma=0.1)


validation_mask_losses = []
train_losses = []
val_losses = []

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 10
early_stop_min_delta = 1e-4

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(11, EPOCHS+1):
  print(f"Starting epoch {epoch} of {EPOCHS}")
  time_start = time.time()
  epoch_loss = 0.0
  loss_mask_accum = 0.0
  loss_classifier_accum = 0.0
  for images, targets in train_loader:
    images  = [img.to(DEVICE) for img in images]
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    opt_stage2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt_stage2.step()
    warmup_sched.step()

    loss_mask = loss_dict['loss_mask'].item()
    epoch_loss += loss.item()
    loss_mask_accum += loss_mask
    loss_classifier_accum += loss_dict['loss_classifier'].item()

  if USE_SCHEDULER:
    lr_scheduler.step()

  train_loss = epoch_loss / n_batches
  train_loss_mask = loss_mask_accum / n_batches
  train_loss_classifier = loss_classifier_accum / n_batches

  val_loss_epoch = 0
  val_loss_mask_accum = 0
  val_loss_classifier_accum = 0

  with torch.no_grad():
      for batch_idx, (images, targets) in enumerate(val_loader, 1):
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

  # Early stopping
  if val_loss < best_val_loss - early_stop_min_delta:
    best_val_loss = val_loss
    epochs_no_improve = 0
    best_model_path = "checkpoints/best_model.pth"
    torch.save(model.state_dict(), best_model_path)
    print(f"New best model saved: {best_model_path} with val loss {val_loss:.4f}")
  else:
    epochs_no_improve += 1
    print(f"No improvement in val loss for {epochs_no_improve} epochs")

  if epochs_no_improve >= early_stop_patience:
    print(f"Early stopping at epoch {epoch}, no improvemenet in {early_stop_patience} epochs.")
    break

  print(f"[Epoch {epoch} / {EPOCHS}] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / {EPOCHS}] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / {EPOCHS}] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
  print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")


# for i, (img, tgt) in enumerate(zip(images, targets)):
#     print(f"\n--- Sample {i} ---")
#     print("Image shape:", img.shape)
#     print("Image stats:",
#           f"min={img.min().item():.3f}",
#           f"max={img.max().item():.3f}",
#           f"mean={img.mean().item():.3f}",
#           f"std={img.std().item():.3f}")


# os.makedirs("checkpoints", exist_ok=True)

# validation_mask_losses = []
# train_losses = []
# val_losses = []

# for epoch in range(1, EPOCHS + 1):
#     print(f"Starting epoch {epoch} of {EPOCHS}")

#     time_start = time.time()
#     epoch_loss = 0.0
#     loss_mask_accum = 0.0
#     loss_classifier_accum = 0.0
#     for batch_idx, (images, targets) in enumerate(train_loader, 1):

#         images = list(image.to(DEVICE) for image in images)
#         # images = [img.to(DEVICE, memory_format=torch.channels_last) for img in images]
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         loss_dict = model(images, targets)
#         loss = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loss_mask = loss_dict['loss_mask'].item()
#         epoch_loss += loss.item()
#         loss_mask_accum += loss_mask
#         loss_classifier_accum += loss_dict['loss_classifier'].item()

#         if batch_idx % 500 == 0:
#             print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}.")

#     if USE_SCHEDULER:
#         lr_scheduler.step()

#     train_loss = epoch_loss / n_batches
#     train_loss_mask = loss_mask_accum / n_batches
#     train_loss_classifier = loss_classifier_accum / n_batches

#     val_loss_epoch = 0
#     val_loss_mask_accum = 0
#     val_loss_classifier_accum = 0

#     with torch.no_grad():
#         for batch_idx, (images, targets) in enumerate(val_loader, 1):
#             images = list(image.to(DEVICE) for image in images)
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#             val_loss_dict = model(images, targets)
#             val_batch_loss = sum(loss for loss in val_loss_dict.values())
#             val_loss_epoch += val_batch_loss.item()
#             val_loss_mask_accum += val_loss_dict['loss_mask'].item()
#             val_loss_classifier_accum += val_loss_dict['loss_classifier'].item()

#     val_loss = val_loss_epoch / n_batches_val
#     val_loss_mask = val_loss_mask_accum / n_batches_val
#     val_loss_classifier = val_loss_classifier_accum / n_batches_val
#     #time per epoch
#     epoch_time = time.time() - time_start

#     #for plotting
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     validation_mask_losses.append(val_loss_mask)

#     checkpoint_path = f"checkpoints/maskrcnn_epoch_{epoch}.pth"
#     torch.save(model.state_dict(), checkpoint_path)

#     print(f"[Epoch {epoch} / {EPOCHS}] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
#     print(f"[Epoch {epoch} / {EPOCHS}] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
#     print(f"[Epoch {epoch} / {EPOCHS}] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
#     print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")
#     print(f"Saved checkpoint: {checkpoint_path}")


#training vall loss curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Curve")
plt.show()


import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def show_batch_with_preds(images, targets, outputs, max_images=4, mask_threshold=0.5):
    plt.figure(figsize=(16, 4 * max_images))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(max_images, len(images))):
        image = images[i].cpu()
        target = targets[i]
        output = outputs[i]

        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip((img_np * std) + mean, 0, 1)

        # Ground Truth
        gt_masks = target["masks"].cpu().numpy()
        gt_combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for mask in gt_masks:
            gt_combined_mask = np.maximum(gt_combined_mask, mask.squeeze().astype(np.uint8))

        ax1 = plt.subplot(max_images, 2, 2 * i + 1)
        ax1.imshow(img_np)
        ax1.imshow(gt_combined_mask, alpha=0.4, cmap='Blues')
        ax1.set_title("Ground Truth")
        ax1.axis('off')

        for box in target["boxes"].cpu():
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)

        # Predictions
        scores = output["scores"].cpu()
        keep = scores > mask_threshold

        pred_masks = output["masks"][keep].cpu().numpy()
        pred_boxes = output["boxes"][keep].cpu().numpy()

        pred_combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for mask in pred_masks:
            pred_combined_mask = np.maximum(pred_combined_mask, (mask.squeeze() > 0.5).astype(np.uint8))

        ax2 = plt.subplot(max_images, 2, 2 * i + 2)
        ax2.imshow(img_np)
        ax2.imshow(pred_combined_mask, alpha=0.4, cmap='Reds')
        ax2.set_title("Prediction")
        ax2.axis('off')

        for box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
    print([output['labels'] for output in outputs])
    print(outputs[0]['scores'])
    plt.tight_layout()
    plt.show()

model.eval()
with torch.no_grad():
    outputs = model([img.to(DEVICE) for img in images])  # make sure images is a list
show_batch_with_preds(images, targets, outputs, max_images=3)



show_batch_with_preds(
    images, targets, outputs,
    max_images=3,
    mask_threshold=0.2  # or even 0.1 if needed
)


class CellTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [f[:-4]for f in os.listdir(self.image_dir)]


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".png")
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)

        return {'image': image, 'image_id': image_id}


    def __len__(self):
        return len(self.image_ids)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
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
    if True:
        transforms.append(Normalize())

    return Compose(transforms)


model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

test_dataset = CellTestDataset(test_img_path, transforms=get_transform(train=False))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


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


print(f"{len(test_dataset)}images found in test dataset")

submission = []
for batch in test_loader:
    images   = batch["image"].to(DEVICE)
    img_ids  = batch["image_id"]

    with torch.no_grad():
        outputs = model(images)

    for out, image_id in zip(outputs, img_ids):
        H, W = out["masks"].shape[-2:]
        occupied = np.zeros((H, W), dtype=bool)

        for mask_tensor, score in zip(out["masks"], out["scores"]):
            if score.item() < 0.3:
                continue

            mask_np = mask_tensor[0].cpu().numpy()
            bin_mask = mask_np > 0.5

            # remove overlaps
            clean_mask = remove_overlapping_pixels(bin_mask, occupied)
            if clean_mask.sum() == 0:
                continue

            # mark those pixels as occupied
            occupied |= clean_mask

            rle = rle_encoding(clean_mask.astype(np.uint8))
            submission.append((image_id, rle))

df = pd.DataFrame(submission, columns=["id", "predicted"])
df.to_csv("submission.csv", index=False)
print(f"{len(df)} mask‐rows in submission file")

