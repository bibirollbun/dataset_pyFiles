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
import torchvision
import time
import matplotlib.patches as patches
import cv2

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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
# transform = T.Compose([
#     T.Resize((512, 512)),
#     T.ToTensor(),
#     T.Normalize(
#       mean=[0.485, 0.456, 0.406],
#       std=[0.229, 0.224, 0.225])
# ])

# for not pretrained model

transform = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.2),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])


def rle_decode_for_train(mask_rle, shape=(520, 704)):
    if pd.isnull(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class CellDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, image_size=(512, 512), augment=False):
        self.df = df
        self.image_ids = df['id'].unique()
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(img_path).convert("RGB")

        # Resize
        if self.transform:
            image = self.transform(image)


        records = self.df[self.df['id'] == image_id]
        masks = []
        boxes = []

        for _, row in records.iterrows():
            mask = mask = rle_decode_for_train(row['annotation'])
            mask = Image.fromarray(mask).resize(self.image_size, resample=Image.NEAREST)
            mask = np.array(mask)
            if mask.max() == 0:
                continue
            masks.append(mask)

            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        if len(masks) == 0:
            masks = torch.zeros((0, *self.image_size), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(masks),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        # Simple Data Augmentation (random horizontal flip)
        if self.augment and random.random() > 0.5:
            image = torch.flip(image, dims=[2])  # horizontal flip
            masks = torch.flip(masks, dims=[2])
            boxes[:, [0, 2]] = self.image_size[1] - boxes[:, [2, 0]]  # update x coords
            target["boxes"] = boxes
            target["masks"] = masks

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = CellDataset(
    df=train_df,
    image_dir=train_img_path,
    transform=transform,
    augment=True
)
# DataLoader for training
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,  # increase if possible
    collate_fn=collate_fn
)

# DataLoader for validation
val_dataset = CellDataset(
    df=val_df,
    image_dir=train_img_path,
    transform=transform,
    augment=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)


def show_batch(images, targets, max_images=4):
    plt.figure(figsize=(16, 4 * max_images))

    for i in range(min(max_images, len(images))):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 0.229 + 0.485, 0, 1)  # denomarlization ImageNet

        masks = targets[i]["masks"].cpu().numpy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # for mask in masks:
        #     combined_mask = np.maximum(combined_mask, mask)
        # resize masks to match image size before combining
        for mask in masks:
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, mask)
            # print(f"Image shape: {image.shape}")
            # print(f"Mask shape: {mask.shape}")

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


RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

MOMENTUM = 0.9
# LR = 0.001 #trained
LR = 0.005
WEIGHT_DECAY = 0.0005
MASK_THRESHOLD = 0.5 #0.05
PATIENCE = 3
NUM_CLASSES = 3
WIDTH = 704
HEIGHT = 520
USE_SCHEDULER = False

BATCH_SIZE = 2
# EPOCHS = 20
EPOCHS = 100
USE_SCHEDULER = False
BOX_DETECTIONS_PER_IMG = 539

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
#                                                                    box_detections_per_img=BOX_DETECTIONS_PER_IMG,
#                                                                    image_mean=RESNET_MEAN,
#                                                                    image_std=RESNET_STD)

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES+1)
# in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# hidden_layer = 256
# model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES+1)


# model.to(DEVICE)

# for param in model.parameters():
#     param.requires_grad = True

# model.train()
backbone = resnet_fpn_backbone('resnet50', pretrained=False)

model = MaskRCNN(backbone=backbone, num_classes=NUM_CLASSES+1)

model.to(DEVICE)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode='min',
                                                          factor=0.5,
                                                          patience=PATIENCE,
                                                          verbose=True)
n_batches, n_batches_val = len(train_loader), len(val_loader)


os.makedirs("checkpoints", exist_ok=True)

validation_mask_losses = []
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    print(f"Starting epoch {epoch} of {EPOCHS}")

    time_start = time.time()
    epoch_loss = 0.0
    loss_mask_accum = 0.0
    loss_classifier_accum = 0.0
    for batch_idx, (images, targets) in enumerate(train_loader, 1):

        images = list(image.to(DEVICE) for image in images)
        # images = [img.to(DEVICE, memory_format=torch.channels_last) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mask = loss_dict['loss_mask'].item()
        epoch_loss += loss.item()
        loss_mask_accum += loss_mask
        loss_classifier_accum += loss_dict['loss_classifier'].item()

        if batch_idx % 500 == 0:
            print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}.")

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

    checkpoint_path = f"checkpoints/maskrcnn_epoch_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"[Epoch {epoch:2d} / {EPOCHS:2d}] Train-mask loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
    print(f"[Epoch {epoch:2d} / {EPOCHS:2d}] Val-mask loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
    print(f"[Epoch {epoch:2d} / {EPOCHS:2d}] Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
    print(f"Time for epoch {epoch}: {epoch_time:.2f} seconds")
    print(f"Saved checkpoint: {checkpoint_path}")


#training vall loss curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Curve")
plt.show()


import matplotlib.patches as patches

def show_batch_with_preds(images, targets, outputs, max_images=4, mask_threshold=0.5):
    plt.figure(figsize=(16, 4 * max_images))

    for i in range(min(max_images, len(images))):
        image = images[i].cpu()
        target = targets[i]
        output = outputs[i]

        img_np = image.permute(1, 2, 0).numpy()
        # de-normalize for ImageNet
        img_np = np.clip(img_np * 0.229 + 0.485, 0, 1)

        # GT masks
        gt_masks = target["masks"].cpu().numpy()
        gt_combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for mask in gt_masks:
            gt_combined_mask = np.maximum(gt_combined_mask, mask.squeeze())

        # GT plot
        ax1 = plt.subplot(max_images, 2, 2 * i + 1)
        ax1.imshow(img_np)
        ax1.imshow(gt_combined_mask, alpha=0.4, cmap='Blues')
        ax1.set_title("Ground Truth")
        ax1.axis('off')

        # GT boxes
        for box in target["boxes"].cpu():
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)

        # Prediction masks
        scores = output["scores"].cpu()
        keep = scores > MASK_THRESHOLD

        pred_masks = output["masks"][keep].cpu().numpy()
        pred_boxes = output["boxes"][keep].cpu().numpy()
        pred_labels = output["labels"][keep].cpu().numpy()

        pred_combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for mask in pred_masks:
            pred_combined_mask = np.maximum(pred_combined_mask, mask.squeeze() > 0.5)

        # Plot Prediction
        ax2 = plt.subplot(max_images, 2, 2 * i + 2)
        ax2.imshow(img_np)
        ax2.imshow(pred_combined_mask, alpha=0.4, cmap='Reds')
        ax2.set_title("Prediction")
        ax2.axis('off')

        # Add predicted boxes
        for box, label in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            # ax2.text(x1, y1, f'Class {label}', color='white', fontsize=10,
            #          bbox=dict(facecolor='red', alpha=0.5))

    plt.tight_layout()
    plt.show()


model.eval()
with torch.no_grad():
    outputs = model(images)
show_batch_with_preds(images, targets, outputs, max_images=3)



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
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


def rle_encode(mask):
    pixels = mask.T.flatten()  # column line
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#Submission generator
def generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device='cuda', threshold=0.5):
    model.load_state_dict(torch.load("checkpoints/maskrcnn_epoch_20.pth", map_location=DEVICE))
    model.eval()
    submissions = []

    for images, image_ids in test_loader:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = image_ids[i]
            masks = output['masks'].squeeze(1).cpu().numpy() if output['masks'].ndim == 4 else []

            if len(masks) == 0:
                submissions.append({"id": image_id, "predicted": ""})
                continue

            for mask in masks:
                bin_mask = (mask > threshold).astype(np.uint8)
                if bin_mask.sum() == 0:
                    continue
                rle = rle_encode(bin_mask)
                if rle:
                    submissions.append({"id": image_id, "predicted": rle})

    df = pd.DataFrame(submissions)
    df.to_csv(output_csv_path, index=False)
    print(f"Submision file created : {output_csv_path}")
    return df


generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device = DEVICE, threshold=MASK_THRESHOLD)
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

# """import model
# from torchvision.models.detection import maskrcnn_resnet50_fpn for example"""
# #model = maskrcnn_resnet50_fpn(pretrained=True)
# #model.to('cuda')


# #Submission file choose depend on your model
# generate_submission_csv(model, test_loader, output_csv_path='submission.csv', device='cuda')
# generate_submission_csv_unet(model, test_loader, output_csv_path='submission.csv', device='cuda')

# #Visualization
# test_model_on_batch(model, test_loader, device='cuda')


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

# # 5. Affichage matrix

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

# # 6. Test batch visuel (val_loader) pour Mask R-CNN ou UNet

# def test_model_on_batch_unet(model, dataloader, device='cuda', threshold=0.5):
#     model.eval()
#     images, targets = next(iter(dataloader))
#     images = list(img.to(device) for img in images)

#     with torch.no_grad():
#         preds = model(torch.stack(images)).squeeze(1).cpu().numpy()

#     preds_bin = [(p > threshold).astype(np.uint8) for p in preds]
#     gts = [(t['masks'].sum(0).cpu().numpy() > 0).astype(np.uint8) for t in targets]

#     results = evaluate_instance_segmentation(preds_bin, gts, threshold)
#     for k, v in results.items():
#         if k != "confusion_matrix":
#             print(f"{k}: {v:.4f}")
#     plot_confusion_matrix(results["confusion_matrix"])
#     return results

