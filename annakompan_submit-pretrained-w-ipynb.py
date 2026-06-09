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
import torch.nn as nn
from torchvision.ops import misc as misc_nn_ops
import cv2
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.models.detection import MaskRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
from functools import partial
from sklearn.model_selection import train_test_split


#file path in google drive
#create a shortcut in your Drive.
# from google.colab import drive
# drive.mount('/content/drive')

# !cp -r /content/drive/MyDrive/sartorius-cell-instance-segmentation /content/


# data_path = "/content/sartorius-cell-instance-segmentation"
data_path = "/kaggle/input/sartorius-cell-instance-segmentation"


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
test_img_path = f"{data_path}/test"
train_img_path = f"{data_path}/train"
train_df_path = f"{data_path}/train.csv"


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
    print(f"Max: {np.max(combined_mask)}, min: {np.min(combined_mask)}")

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

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
NORMALIZE = False
resize_factor = False
cell_type_dict = {"astro": 1, "cort": 2, "shsy5y": 3}
# mask_threshold_dict = {1: 0.55, 2: 0.75, 3:  0.6}
# min_score_dict = {1: 0.55, 2: 0.75, 3: 0.5}
# resize_factor = False
WIDTH = 704
HEIGHT = 520
BATCH_SIZE = 2

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

    # Data augmentation for train ????????????????????
    if train:
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)


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


train_dataset = CellDataset(train_img_path, train_df, resize=resize_factor, transforms=get_transform(train=True))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                      num_workers=2, collate_fn=collate_fn)

val_dataset = CellDataset(train_img_path, val_df, resize=resize_factor, transforms=get_transform(train=False))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                    num_workers=2, collate_fn=collate_fn)
#shuffle false for val


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

show_batch(images, targets)


print(df.columns)


MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WEIGHT_DECAY2 = 1e-4
MASK_THRESHOLD = 0.5
PATIENCE = 3
NUM_CLASSES = 3
WIDTH = 704
HEIGHT = 520
USE_SCHEDULER = False
# USE_SCHEDULER = True
# BATCH_SIZE = 4 (CUDA out of memory)
EPOCHS = 50
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


# for kaggle!!!
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import MaskRCNN

resnet = resnet50(norm_layer=torch.nn.BatchNorm2d, weights=None)

#weights
state_dict = torch.load("/kaggle/input/resnet50-weights/resnet50-0676ba61.pth")
resnet.load_state_dict(state_dict)  # Full load, no strict=False

#fpn
backbone = _resnet_fpn_extractor(resnet, trainable_layers=3)
backbone.out_channels = 256  # important!

model = MaskRCNN(
    backbone=backbone,
    num_classes=NUM_CLASSES + 1  # your foreground + background
)

#trained model before
trained_state = torch.load("/kaggle/input/best-model-pth/best_model.pth", map_location=DEVICE)
model.load_state_dict(trained_state)  # strict=True should now work
model.to(DEVICE)


"""
scale down the default Kaiming init in RPN, box & mask heads
-stabilize early training weights = too large at the beginning of training
"""
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

# train heads 5 epochs
for name, p in model.backbone.named_parameters():
    p.requires_grad = False

head_params = [p for p in model.parameters() if p.requires_grad]
opt_stage1 = torch.optim.SGD(
    head_params, lr=1e-3, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

# Warm-up schedule: step 500 full LR
def warmup_lambda(step):
    return min((step + 1) / 500, 1.0)

warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt_stage1, warmup_lambda)

EPOCH_ = 5

n_batches, n_batches_val = len(train_loader), len(val_loader)

validation_mask_losses = []
train_losses = []
val_losses = []

for epoch in range(1, EPOCH_+1):
  print(f"Starting epoch {epoch} of {EPOCH_}")
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
          #output = model(images)
          #print(output)
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

  print(f"[Epoch {epoch} / {EPOCH_}] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / {EPOCH_}] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
  print(f"[Epoch {epoch} / {EPOCH_}] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
  print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")


#training vall loss curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Curve")
plt.show()


# best_model_path = "best_model.pth"
# torch.save(model.state_dict(), best_model_path)


# model.load_state_dict(torch.load("best_model_pre.pth", map_location=DEVICE))

for p in model.backbone.parameters():
    p.requires_grad = True

backbone_params, head_params = [], []
for name, p in model.named_parameters():
    (backbone_params if "backbone" in name else head_params).append(p)

# opt_stage2 = torch.optim.SGD([
#     {"params": head_params, "lr": 1e-1},
#     {"params": backbone_params, "lr": 1e-4},
# ], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

opt_stage2 = torch.optim.AdamW([
    {"params": head_params, "lr": 1e-4},
    {"params": backbone_params, "lr": 1e-5},
], weight_decay=WEIGHT_DECAY2)

# Step LR or ReduceLROnPlateau on val loss for SGD StepLR
# lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_stage2, step_size=10, gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_stage2, mode='min', factor=0.5, patience=3, threshold=1e-4, verbose=True
)


validation_mask_losses = []
train_losses = []
val_losses = []

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 10
early_stop_min_delta = 1e-4

os.makedirs("checkpoints", exist_ok=True)

EPOCHS = 20

for epoch in range(EPOCH_+1, EPOCHS+1):
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
    # warmup_sched.step()

    loss_mask = loss_dict['loss_mask'].item()
    epoch_loss += loss.item()
    loss_mask_accum += loss_mask
    loss_classifier_accum += loss_dict['loss_classifier'].item()

  # if USE_SCHEDULER:
  #   lr_scheduler.step(val_losses)

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

  if USE_SCHEDULER:
    lr_scheduler.step(val_loss)

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

model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))

def remove_overlapping_pixels(mask, other_masks):

    for other_mask in other_masks:

        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0

    return mask

#mask_threshold_dict = {1: 0.05, 2: 0.05, 3:  0.06}
mask_threshold_dict = {1: 0.55, 2: 0.75, 3:  0.6}
min_score_dict = {1: 0.55, 2: 0.75, 3: 0.5}

def get_filtered_masks_v2(masks, boxes, scores, labels):
    """
    filter masks using MIN_SCORE for mask and MAX_THRESHOLD for pixels
    """
    use_masks = []
    use_boxes = []

    for i, mask in enumerate(masks):
        #print("check mask # ", i)
        label = labels[i]
        #binary_mask = mask > mask_threshold_dict[label]
        binary_mask = remove_overlapping_pixels(mask, use_masks)

        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        #ax.imshow(binary_mask)
        #print(boxes[i])
        #x1, y1, x2, y2 = boxes[i]
        #rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
        #                             linewidth=2, edgecolor='red', facecolor='none')
        #ax.add_patch(rect)
        #plt.show()

        if np.any(binary_mask) and np.sum(np.array(binary_mask, dtype=int)) > 100:
            use_masks.append(binary_mask)
            use_boxes.append(boxes[i])
            #print(f"not all pixels eliminated! {np.sum(np.array(binary_mask, dtype=int))}")
        #else:
            #print("all pixels eliminated")

    return use_masks, use_boxes

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
        mask_threshold = 0.3
        keep = scores > mask_threshold

        pred_masks = output["masks"][keep].cpu().numpy()
        pred_boxes = output["boxes"][keep].cpu().numpy()
        pred_labels = output["labels"][keep].cpu().numpy()
        pred_scores = output["scores"][keep].cpu().numpy()
        print(pred_masks.shape)
        binary_masks = []

        for id , mask in enumerate(pred_masks):
            label = pred_labels[id]
            #print(label)
            mask = mask[0]
            binary_mask = mask > mask_threshold_dict[label]

            binary_masks.append(binary_mask)

            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
            #ax.imshow(binary_mask)
            #print(pred_boxes[id])
            #x1, y1, x2, y2 = pred_boxes[id]
            #rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
            #                         linewidth=2, edgecolor='red', facecolor='none')
            #ax.add_patch(rect)
            #plt.show()

        print("before check ", len(binary_masks))


        binary_masks, pred_boxes = get_filtered_masks_v2(binary_masks, pred_boxes, pred_scores, pred_labels)
        print("after check ", len(pred_masks))
        pred_combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        pred_combined_ADD_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        for id, mask in enumerate(binary_masks):
            #pred_combined_mask += mask
            mask = np.array(mask, dtype=int)
            #x1, y1, x2, y2 = pred_boxes[id]
            #print(mask[int(y1):int(y2), int(x1):int(x2)])
            pred_combined_mask = np.maximum(pred_combined_mask, (mask.squeeze() > 0.5).astype(np.uint8))
            pred_combined_ADD_mask += (mask.squeeze() > 0.5).astype(np.uint8)

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
            ax.imshow(mask)
            print(pred_boxes[id])
            x1, y1, x2, y2 = pred_boxes[id]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')

            ax.add_patch(rect)
            plt.show()
            print(np.sum(mask))

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

        #ax3 = plt.subplot(max_images, 2, 2 * i + 2)
        #ax3.imshow(img_np)
        #ax3.imshow(pred_combined_ADD_mask, alpha=0.4, cmap='Blues')
        #ax3.set_title("Check for overlaps between masks")
        #ax1.axis('off')

    print([output['labels'] for output in outputs])
    print(outputs[0]['scores'])
    print(outputs[0]['masks'])
    plt.tight_layout()
    plt.show()

model.eval()
with torch.no_grad():
    outputs = model([img.to(DEVICE) for img in images])  # make sure images is a list
show_batch_with_preds(images, targets, outputs, max_images=3)




def compute_iou(labels, y_pred, verbose=0):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

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
    intersection = intersection[1:, 1:] # exclude background
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union

    return iou

def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
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

def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
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



cell_type_dict = {"astro": 1, "cort": 2, "shsy5y": 3}
# mask_threshold_dict = {1: 0.55, 2: 0.75, 3:  0.6}
mask_threshold_dict = {1: 0.05, 2: 0.05, 3:  0.06}
min_score_dict = {1: 0.55, 2: 0.75, 3: 0.5}

def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def combine_masks(masks, mask_threshold):
    """
    combine masks into one image
    """
    maskimg = np.zeros((HEIGHT, WIDTH))
    # print(len(masks.shape), masks.shape)
    for m, mask in enumerate(masks,1):
        maskimg[mask>mask_threshold] = m

    return maskimg

def combine_masks_v2(masks, mask_threshold):
    """
    combine masks into one image
    """
    maskimg = np.zeros((HEIGHT, WIDTH))
    print(masks.shape)
    # print(len(masks.shape), masks.shape)
    for m, mask in enumerate(masks,1):
        #print(mask[0].shape)
        maskimg[mask[0]>mask_threshold] = m

    return maskimg


def get_filtered_masks(pred):
    """
    filter masks using MIN_SCORE for mask and MAX_THRESHOLD for pixels
    """
    use_masks = []

    for i, mask in enumerate(pred["masks"]):

        # Filter-out low-scoring results. Not tried yet.
        scr = pred["scores"][i].cpu().item()
        label = pred["labels"][i].cpu().item()
        if scr > min_score_dict[label]:
            mask = mask.cpu().numpy().squeeze()
            # Keep only highly likely pixels
            binary_mask = mask >  mask_threshold_dict[label]
            #binary_mask = remove_overlapping_pixels(binary_mask, use_masks)
            use_masks.append(binary_mask)

    return use_masks

def analyze_train_sample(model, train_dataset, sample_index):

    img, targets = train_dataset[sample_index]
    #print(img.shape)
    l = np.unique(targets["labels"])
    ig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,60), facecolor="#fefefe")
    ax[0].imshow(img.numpy().transpose((1,2,0)))
    ax[0].set_title(f"cell type {l}")
    ax[0].axis("off")

    masks = combine_masks(targets['masks'], 0.5)
    #plt.imshow(img.numpy().transpose((1,2,0)))
    ax[1].imshow(masks)
    ax[1].set_title(f"Ground truth, {len(targets['masks'])} cells")
    ax[1].axis("off")

    model.eval()
    with torch.no_grad():
        preds = model([img.to(DEVICE)])[0]
    print(targets['masks'])
    print(preds['masks'])

    l = pd.Series(preds['labels'].cpu().numpy()).value_counts()
    lstr = ""
    for i in l.index:
        lstr += f"{l[i]}x{i} "
    #print(l, l.sort_values().index[-1])
    #plt.imshow(img.cpu().numpy().transpose((1,2,0)))
    mask_threshold = mask_threshold_dict[l.sort_values().index[-1]]

    print(mask_threshold)
    pred_masks = combine_masks(get_filtered_masks(preds), mask_threshold)
    mask_threshold = 0.5
    pred_masks = combine_masks_v2(preds['masks'].cpu().numpy(), mask_threshold)
    ax[2].imshow(pred_masks)
    ax[2].set_title(f"Predictions, labels: {lstr}")
    ax[2].axis("off")
    plt.show()

    #print(masks.shape, pred_masks.shape)
    score = iou_map([masks],[pred_masks])
    print("Score:", score)


# NOTE: It puts the model in eval mode!! Revert for re-training
analyze_train_sample(model, train_dataset, 20)


#dataset for test
class TestCellDataset(Dataset):
    def __init__(self, image_dir, image_size=(512, 512), transform=None):
        self.image_dir = image_dir
        self.image_ids = [f.replace('.png', '') for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_size = image_size
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, image_id


test_dataset = TestCellDataset(test_img_path)
print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(len(test_loader))


def rle_encode(mask):
    pixels = mask.T.flatten()  # column line
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#Submission generator
def generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device='cuda', threshold=0.5):
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
    model.eval()
    submissions = []

    for images, image_ids in test_loader:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            outputs = model(images)
            #print(outputs)

        for i, output in enumerate(outputs):
            image_id = image_ids[i]

            scores = output["scores"].cpu()
            print(scores)
            mask_threshold = 0.5 #0.2
            keep = scores > mask_threshold
            pred_scores = output["scores"][keep].cpu().numpy()
            #print(keep)
            #masks = output["masks"].squeeze(1).cpu().numpy() if output['masks'].ndim == 4 else []
            masks = output["masks"][keep].cpu().numpy()
            boxes = output["boxes"][keep].cpu().numpy()
            labels = output["labels"][keep].cpu().numpy()


            print(masks.shape)
            print(boxes.shape)
            print(labels.shape)

            binary_masks = []

            for id , mask in enumerate(masks):
                label = labels[id]
                #print(label)
                mask = mask[0]
                binary_mask = mask > mask_threshold_dict[label]

                binary_masks.append(binary_mask)


            print("before check ", len(binary_masks))


            binary_masks, pred_boxes = get_filtered_masks_v2(binary_masks, boxes, pred_scores, labels)
            print("after check ", len(binary_masks))

            if len(binary_masks) == 0:
                submissions.append({"id": image_id, "predicted": ""})
                continue

            for mask in binary_masks:
                bin_mask = mask.astype(np.uint8)
                if bin_mask.sum() == 0:
                    continue
                rle = rle_encode(bin_mask)
                if rle:
                    submissions.append({"id": image_id, "predicted": rle})

    df = pd.DataFrame(submissions)
    df.to_csv(output_csv_path, index=False)
    print(f"Submision file created : {output_csv_path}")
    return df


generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device = DEVICE, threshold=0.5)
# (model, test_loader, output_csv_path='submission.csv', device='cuda', threshold=0.5):


# from skimage.measure import label

# def generate_submission_csv_unet(model, test_loader, output_csv_path='submission.csv', device='cuda', threshold=0.5):
#     model.eval()
#     submissions = []

#     for images, image_ids in test_loader:
#         images = list(img.to(device) for img in images)

#         with torch.no_grad():
#             preds = model(torch.stack(images))  # output shape: [B, 1, H, W]
#             preds = preds.squeeze(1).cpu().numpy()

#         for i in range(len(preds)):
#             image_id = image_ids[i]
#             bin_mask = (preds[i] > threshold).astype(np.uint8)

#             labeled = label(bin_mask)
#             if labeled.max() == 0:
#                 submissions.append({"id": image_id, "predicted": ""})
#                 continue

#             for inst_id in range(1, labeled.max() + 1):
#                 inst_mask = (labeled == inst_id).astype(np.uint8)
#                 rle = rle_encode(inst_mask)
#                 if rle:
#                     submissions.append({"id": image_id, "predicted": rle})

#     df = pd.DataFrame(submissions)
#     df.to_csv(output_csv_path, index=False)
#     print(f"submission file (U-Net) : {output_csv_path}")
#     return df



# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score

# def evaluate_instance_segmentation(preds, gts, threshold=0.5):
#     assert len(preds) == len(gts)
#     ious, dices, precisions, recalls, f1s = [], [], [], [], []
#     empty_pred_count = 0
#     y_true_all, y_pred_all = [], []

#     for pred_mask, gt_mask in zip(preds, gts):
#         pred_bin = (pred_mask > threshold).astype(np.uint8)
#         gt_bin = (gt_mask > 0).astype(np.uint8)

#         if pred_bin.sum() == 0:
#             empty_pred_count += 1

#         y_true_flat = gt_bin.flatten()
#         y_pred_flat = pred_bin.flatten()

#         ious.append(jaccard_score(y_true_flat, y_pred_flat, zero_division=0))
#         dices.append(f1_score(y_true_flat, y_pred_flat, zero_division=0))
#         precisions.append(precision_score(y_true_flat, y_pred_flat, zero_division=0))
#         recalls.append(recall_score(y_true_flat, y_pred_flat, zero_division=0))
#         f1s.append(f1_score(y_true_flat, y_pred_flat, zero_division=0))

#         y_true_all.extend(y_true_flat)
#         y_pred_all.extend(y_pred_flat)

#     cm = confusion_matrix(y_true_all, y_pred_all)

#     return {
#         "mean_IoU": np.mean(ious),
#         "mean_Dice": np.mean(dices),
#         "mean_Precision": np.mean(precisions),
#         "mean_Recall": np.mean(recalls),
#         "mean_F1": np.mean(f1s),
#         "empty_prediction_rate": empty_pred_count / len(preds),
#         "confusion_matrix": cm
#     }

# def plot_confusion_matrix(cm, labels=["background", "cell"]):
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.xticks(ticks=range(len(labels)), labels=labels)
#     plt.yticks(ticks=range(len(labels)), labels=labels)
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.show()

# def test_model_on_batch(model, dataloader, device='cuda', threshold=0.5, max_images=4):
#     model.eval()
#     images, targets = next(iter(dataloader))
#     images = list(img.to(device) for img in images)

#     with torch.no_grad():
#         outputs = model(images)

#     preds = []
#     gts = []
#     for output, target in zip(outputs, targets):
#         # Predicted Masks
#         pred_masks = output['masks'].squeeze(1).cpu().numpy() if output['masks'].ndim == 4 else []
#         combined_pred = np.zeros((512, 512))
#         for mask in pred_masks:
#             combined_pred = np.maximum(combined_pred, mask)
#         preds.append(combined_pred)

#         # Masks ground truth
#         gt_masks = target['masks'].cpu().numpy()
#         combined_gt = np.zeros((512, 512))
#         for mask in gt_masks:
#             combined_gt = np.maximum(combined_gt, mask)
#         gts.append(combined_gt)

#     results = evaluate_instance_segmentation(preds, gts, threshold)

#     for k, v in results.items():
#         if k != "confusion_matrix":
#             print(f"{k}: {v:.4f}")
#     plot_confusion_matrix(results["confusion_matrix"])

#     return results



# test_dataset = TestCellDataset(image_dir='chemin/vers/test/', image_size=(512, 512))
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

"""import model
from torchvision.models.detection import maskrcnn_resnet50_fpn for example"""
#model = maskrcnn_resnet50_fpn(pretrained=True)
#model.to('cuda')


#Submission file choose depend on your model
# generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device='cuda')
# generate_submission_csv_unet(model, test_loader, output_csv_path='submission.csv', device='cuda')

#Visualization
# test_model_on_batch(model, test_loader, device='cuda')


def evaluate_instance_segmentation(preds, gts, threshold=0.5):
    assert len(preds) == len(gts)
    ious, dices, precisions, recalls, f1s = [], [], [], [], []
    empty_pred_count = 0
    y_true_all, y_pred_all = [], []

    for pred_mask, gt_mask in zip(preds, gts):
        pred_bin = (pred_mask > threshold).astype(np.uint8)
        gt_bin = (gt_mask > 0).astype(np.uint8)
        if pred_bin.sum() == 0:
            empty_pred_count += 1
        y_true_flat = gt_bin.flatten()
        y_pred_flat = pred_bin.flatten()
        ious.append(jaccard_score(y_true_flat, y_pred_flat, zero_division=0))
        dices.append(f1_score(y_true_flat, y_pred_flat, zero_division=0))
        precisions.append(precision_score(y_true_flat, y_pred_flat, zero_division=0))
        recalls.append(recall_score(y_true_flat, y_pred_flat, zero_division=0))
        f1s.append(f1_score(y_true_flat, y_pred_flat, zero_division=0))
        y_true_all.extend(y_true_flat)
        y_pred_all.extend(y_pred_flat)

    cm = confusion_matrix(y_true_all, y_pred_all)
    return {
        "mean_IoU": np.mean(ious),
        "mean_Dice": np.mean(dices),
        "mean_Precision": np.mean(precisions),
        "mean_Recall": np.mean(recalls),
        "mean_F1": np.mean(f1s),
        "empty_prediction_rate": empty_pred_count / len(preds),
        "confusion_matrix": cm
    }

# 5. Affichage matrix

def plot_confusion_matrix(cm, labels=["background", "cell"]):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# 6. Test batch visuel (val_loader) pour Mask R-CNN ou UNet

def test_model_on_batch_unet(model, dataloader, device='cuda', threshold=0.5):
    model.eval()
    images, targets = next(iter(dataloader))
    images = list(img.to(device) for img in images)

    with torch.no_grad():
        preds = model(torch.stack(images)).squeeze(1).cpu().numpy()

    preds_bin = [(p > threshold).astype(np.uint8) for p in preds]
    gts = [(t['masks'].sum(0).cpu().numpy() > 0).astype(np.uint8) for t in targets]

    results = evaluate_instance_segmentation(preds_bin, gts, threshold)
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    plot_confusion_matrix(results["confusion_matrix"])
    return results


# import csv

# with open('sample_submission.csv', 'r') as file:
#        csvreader = csv.reader(file)
#        for row in csvreader:
#            print(row)




