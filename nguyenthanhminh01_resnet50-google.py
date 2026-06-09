import pathlib

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import PIL.Image
import albumentations.pytorch
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from typing import List, Tuple



MODEL_FILE = pathlib.Path('../input/google-landmark-2021-validation/model.pth')
TRAIN_LABEL_FILE = pathlib.Path('train.csv')
TRAIN_IMAGE_DIR = pathlib.Path('train')
VALID_LABEL_FILE = pathlib.Path('val.csv')
VALID_IMAGE_DIR = pathlib.Path('val')
TEST_LABEL_FILE = pathlib.Path('test.csv')
TEST_IMAGE_DIR = pathlib.Path('../input/landmark-recognition-2021/test')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_file: pathlib.Path, image_dir: pathlib.Path) -> None:
        super().__init__()
        self.files = [
            image_dir / n[0] / n[1] / n[2] / f'{n}.jpg'
            for n in pd.read_csv(label_file)['id'].values]
        
        self.transformer = albumentations.Compose([
            albumentations.SmallestMaxSize(IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
            albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
            albumentations.Normalize(),
            albumentations.pytorch.ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        path = self.files[index]
        image = PIL.Image.open(self.files[index])
        image = self.transformer(image=np.array(image))['image']

        return path.name[:-4], image


from torchvision.models import resnet50
import torch.nn as nn


import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Config
DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"
TOP_K = 150  # Chá»�n nhiá»�u hÆ¡n Ä‘á»ƒ Ä‘áº£m báº£o sau kiá»ƒm tra váº«n Ä‘á»§ 100 class
TARGET_CLASS_COUNT = 100

# Step 1: Ä�á»�c dá»¯ liá»‡u gá»‘c
df = pd.read_csv('/kaggle/input/landmark-recognition-2021/train.csv')

# Step 2: Lá»�c top K landmark phá»• biáº¿n (chÆ°a kiá»ƒm tra file)
top_landmarks = (
    df['landmark_id']
    .value_counts()
    .head(TOP_K)
    .index
)
df_top = df[df['landmark_id'].isin(top_landmarks)].copy()

# Step 3: Kiá»ƒm tra file áº£nh tá»“n táº¡i
def image_exists(row):
    img_id = row["id"]
    img_path = os.path.join(DATA_DIR, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
    return os.path.exists(img_path)

df_top = df_top[df_top.apply(image_exists, axis=1)].copy()

# Step 4: Lá»�c láº¡i Ä‘Ãºng TARGET_CLASS_COUNT landmark cÃ²n Ä‘á»§ áº£nh
valid_class_counts = df_top['landmark_id'].value_counts()
final_landmarks = valid_class_counts.head(TARGET_CLASS_COUNT).index
filtered_df = df_top[df_top['landmark_id'].isin(final_landmarks)].copy()

actual_class_count = filtered_df['landmark_id'].nunique()
if actual_class_count < TARGET_CLASS_COUNT:
    print(f"Chá»‰ cÃ²n {actual_class_count} class sau khi kiá»ƒm tra áº£nh. Cáº§n tÄƒng TOP_K hoáº·c lá»�c láº¡i.")
else:
    print(f"Ä�Ã£ giá»¯ láº¡i Ä‘Ãºng {actual_class_count} class.")

# Step 5: Chia theo tá»«ng group landmark_id â†’ train/val/test (80/10/10)
train_list = []
val_list = []
test_list = []

for landmark_id, group in filtered_df.groupby('landmark_id'):
    if len(group) < 3:
        continue  # skip náº¿u khÃ´ng Ä‘á»§ tÃ¡ch cáº£ 3 táº­p

    # 80% train, 20% temp
    train_part, temp_part = train_test_split(
        group, test_size=0.2, random_state=42, shuffle=True
    )

    # TÃ¡ch tiáº¿p 10% val, 10% test
    if len(temp_part) >= 2:
        val_part, test_part = train_test_split(
            temp_part, test_size=0.5, random_state=42, shuffle=True
        )
    else:
        val_part = temp_part
        test_part = pd.DataFrame(columns=group.columns)

    train_list.append(train_part)
    val_list.append(val_part)
    test_list.append(test_part)

# Step 6: Gá»™p láº¡i vÃ  ghi file
final_train = pd.concat(train_list).reset_index(drop=True)
final_val = pd.concat(val_list).reset_index(drop=True)
final_test = pd.concat(test_list).reset_index(drop=True)

final_train.to_csv("train.csv", index=False)
final_val.to_csv("val.csv", index=False)
final_test.to_csv("test.csv", index=False)

print(f"Train: {len(final_train)} samples, Val: {len(final_val)}, Test: {len(final_test)}")


import shutil
from tqdm import tqdm

def get_src_path(img_id):
    return os.path.join(DATA_DIR, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")

def copy_images(image_ids, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for img_id in tqdm(image_ids):
        src = get_src_path(img_id)
        dst = os.path.join(dest_dir, f"{img_id}.jpg")
        if not os.path.exists(dst):
            try:
                shutil.copyfile(src, dst)
            except Exception as e:
                print(f"Lá»—i copy {img_id}: {e}")

# Gá»�i:
copy_images(final_train['id'].unique(), "/kaggle/working/train_images")
copy_images(final_val['id'].unique(), "/kaggle/working/val_images")
copy_images(final_test['id'].unique(), "/kaggle/working/test_images")


import pandas as pd

# Load your CSVs
# train_df = pd.read_csv('/kaggle/working/train.csv')
# test_df = pd.read_csv('/kaggle/working/test.csv')
# val_df = pd.read_csv('/kaggle/working/val.csv')
train_df = pd.read_csv('/kaggle/input/landmark/train.csv')
test_df = pd.read_csv('/kaggle/input/landmark/test.csv')
val_df = pd.read_csv('/kaggle/input/landmark/val.csv')
# Build mapping
landmark_id_to_idx = {lid: idx for idx, lid in enumerate(sorted(train_df['landmark_id'].unique()))}
NUM_CLASSES = len(landmark_id_to_idx) 

# Map class_idx
train_df['class_idx'] = train_df['landmark_id'].map(landmark_id_to_idx)
test_df['class_idx'] = test_df['landmark_id'].map(landmark_id_to_idx)
val_df['class_idx'] = val_df['landmark_id'].map(landmark_id_to_idx)

NUM_CLASSES = len(landmark_id_to_idx)


print(NUM_CLASSES)


import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LandmarkDatasetEDA(Dataset):
    def __init__(self, dataframe, data_dir, resize=True, image_size=224):
        """
        Parameters:
        - dataframe: pandas DataFrame, pháº£i cÃ³ cá»™t 'id' vÃ  'class_idx'
        - data_dir: thÆ° má»¥c chá»©a áº£nh (.jpg)
        - resize: náº¿u True, resize vá»� (image_size, image_size)
        - image_size: kÃ­ch thÆ°á»›c Ä‘Ã­ch náº¿u resize
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.resize = resize
        self.image_size = image_size

        if self.resize:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor()
            ])
        else:
            self.transform = None  # sáº½ convert sang numpy array

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_id = row["id"]
        class_idx = int(row["class_idx"])

        folder_path = os.path.join(
            self.data_dir,
            img_id[0],  # First character folder
            img_id[1],  # Second character folder
            img_id[2]   # Third character folder
        )
        
        img_path = os.path.join(folder_path, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Tensor (3, H, W)
        else:
            image = np.array(image)        # Numpy (H, W, 3)

        return image, class_idx




from torch.utils.data import DataLoader
DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"
train_dataset = LandmarkDatasetEDA(train_df,DATA_DIR,resize=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # KhÃ´ng cáº§n collate_fn


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import cv2


import matplotlib.pyplot as plt
import numpy as np
import torch

def show_batch(images, labels, n=8):
    """
    Hiá»ƒn thá»‹ n áº£nh Ä‘áº§u tiÃªn tá»« batch `images` vÃ  `labels`
    Há»— trá»£ áº£nh dáº¡ng tensor (CHW) hoáº·c numpy (HWC)
    """
    plt.figure(figsize=(15, 3))
    for i in range(min(n, len(images))):
        img = images[i]

        # Convert Tensor â†’ numpy
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.shape[0] == 3:  # CHW â†’ HWC
                img = img.transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)

        elif isinstance(img, np.ndarray) and img.ndim == 3:
            pass  # Ä‘Ã£ lÃ  HWC

        else:
            raise ValueError("áº¢nh pháº£i lÃ  torch.Tensor hoáº·c numpy.ndarray 3 chiá»�u")

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"Class {labels[i]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
for images, labels in train_loader:
    show_batch(images, labels, n=8)
    break


def plot_class_distribution_from_df(df, title="Image/class distribution"):
    from collections import Counter
    import matplotlib.pyplot as plt

    class_counts = Counter(df["class_idx"])
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 4))
    plt.hist(counts, bins=min(30, len(counts)))
    plt.title(title)
    plt.xlabel("Number of images")
    plt.ylabel("Number of classes")
    plt.grid()
    plt.show()

    print(f"Number of classes: {len(class_counts)}")
    print(f"Sá»‘ áº£nh min: {min(counts)}, max: {max(counts)}, trung bÃ¬nh: {sum(counts)//len(counts)}")


plot_class_distribution_from_df(train_df)


def plot_brightness_contrast(loader, max_batches=5):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    brightness, contrast = [], []

    with torch.no_grad():  # táº¯t gradient
        for i, (images, _) in enumerate(loader):
            if isinstance(images, torch.Tensor):
                imgs = images.cpu().permute(0, 2, 3, 1).numpy()  # B x H x W x C
            else:
                imgs = np.stack(images, axis=0)  # náº¿u lÃ  list of np.array

            gray = np.mean(imgs, axis=3)  # B x H x W
            brightness.extend(np.mean(gray, axis=(1, 2)))
            contrast.extend(np.std(gray, axis=(1, 2)))

            if i + 1 == max_batches:
                break

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(brightness, bins=30)
    plt.title("Brightness")

    plt.subplot(1, 2, 2)
    plt.hist(contrast, bins=30)
    plt.title("Contrast")
    plt.tight_layout()
    plt.show()


plot_brightness_contrast(train_loader)


import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_rgb_distribution(loader, max_batches=5):
    r_vals, g_vals, b_vals = [], [], []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if isinstance(images, torch.Tensor):
                # B x C x H x W â†’ B x H x W x C
                imgs = images.cpu().permute(0, 2, 3, 1).numpy()
            else:
                imgs = np.stack(images, axis=0)

            # TÃ­nh trung bÃ¬nh tá»«ng kÃªnh RGB
            r_vals.extend(imgs[:, :, :, 0].mean(axis=(1, 2)))
            g_vals.extend(imgs[:, :, :, 1].mean(axis=(1, 2)))
            b_vals.extend(imgs[:, :, :, 2].mean(axis=(1, 2)))

            if i + 1 == max_batches:
                break

    # Váº½ biá»ƒu Ä‘á»“
    plt.hist(r_vals, alpha=0.5, label='R', bins=30)
    plt.hist(g_vals, alpha=0.5, label='G', bins=30)
    plt.hist(b_vals, alpha=0.5, label='B', bins=30)
    plt.legend()
    plt.title("Average RGB")
    plt.xlabel("Mean value (0â€“255)")
    plt.grid()
    plt.show()


plot_rgb_distribution(train_loader)


import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

def plot_blur_distribution(loader, max_batches=5):
    blur_scores = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if isinstance(images, torch.Tensor):
                imgs = images.cpu().permute(0, 2, 3, 1).numpy()  # B x H x W x C
            else:
                imgs = np.stack(images, axis=0)

            for img in imgs:
                # Chuyá»ƒn RGB â†’ Gray (cháº¯c cháº¯n dÃ¹ng uint8)
                gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(score)

            if i + 1 == max_batches:
                break

    # Plot
    plt.hist(blur_scores, bins=30)
    plt.title("Laplacian variance")
    plt.xlabel("Blur Score")
    plt.grid()
    plt.show()

    print(f"Blur trung bÃ¬nh: {np.mean(blur_scores):.2f}")
    print(f"Min: {np.min(blur_scores):.2f}, ğŸ“ˆ Max: {np.max(blur_scores):.2f}")


plot_blur_distribution(train_loader)


import os
from PIL import Image
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_id = row["id"]
        class_idx = int(row["class_idx"])
        
        folder = os.path.join(self.data_dir, img_id[0], img_id[1], img_id[2])
        img_path = os.path.join(folder, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, class_idx


from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMAGE_SIZE = 224
BATCH_SIZE = 32

train_transform = A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ImageCompression(quality_lower=99, quality_upper=100),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.CLAHE(p=0.1),
    A.GaussianBlur(p=0.1),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


print(IMAGE_SIZE)
print(BATCH_SIZE)


DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"
train_dataset = LandmarkDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset = LandmarkDataset(val_df, DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


train_dataset_1 = np.array(train_df)
print(train_dataset_1.shape)


import wandb

wandb.login(key="83b556b8b0cae1769c71ee0296dbf729527a3b1c", relogin=True)


import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def compute_gap(preds, confs, targets):
    """
    Compute simplified GAP@20 assuming top-1 prediction per image.
    """
    df = pd.DataFrame({
        "pred": preds,
        "conf": confs,
        "target": targets
    })

    # Sort globally by confidence
    df = df.sort_values("conf", ascending=False).reset_index(drop=True)

    correct = 0
    total_precision = 0.0

    for i, row in df.iterrows():
        if row["pred"] == row["target"]:
            correct += 1
            total_precision += correct / (i + 1)

    return total_precision / len(df)


print("Sá»‘ class (NUM_CLASSES):", NUM_CLASSES)
print("Min label:", train_df["class_idx"].min())
print("Max label:", train_df["class_idx"].max())
print("Label duy nháº¥t:", sorted(train_df["class_idx"].unique())[:10])


import wandb
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import os
# Khá»Ÿi táº¡o wandb
wandb.init(
    project="landmark-recognition",
    name="resnet50-run",
    config={
        "epochs": 20,
        "model": "resnet50",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader.batch_size,
        "image_size": 224,
    }
)

best_gap = 0.0
EPOCHS = wandb.config.epochs

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"GAP@20: {gap:.4f}")

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model by GAP
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_resnet.pth")
        print(f"Saved best model (GAP={gap:.4f})")

wandb.finish()


img_id = "5551c2a604e9f9b5"
img_path = f"/kaggle/input/landmark-recognition-2021/train/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
os.path.exists(img_path)


sample = test_df.sample(90).iloc[0]
img_id = sample["id"]
true_label = sample["landmark_id"]
class_idx = sample["class_idx"]

folder = os.path.join(DATA_DIR, img_id[0], img_id[1], img_id[2])
img_path = os.path.join(folder, f"{img_id}.jpg")

image = Image.open(img_path).convert("RGB")
image = val_transform(image=np.array(image))["image"].unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(image)
    pred_idx = output.argmax(dim=1).item()

# Reverse map
idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
pred_landmark = idx_to_landmark_id[pred_idx]

print(f"Image ID: {img_id}")
print(f"Ground Truth Landmark ID: {true_label}")
print(f"Predicted Landmark ID: {pred_landmark}")



import matplotlib.pyplot as plt
import PIL.Image

def show_data(img_path, title=None, size=(5, 5)):
    """
    Hiá»ƒn thá»‹ má»™t áº£nh tá»« Ä‘Æ°á»�ng dáº«n img_path.

    Parameters:
        img_path (str): Ä�Æ°á»�ng dáº«n tá»›i áº£nh
        title (str): TiÃªu Ä‘á»� áº£nh (náº¿u cÃ³)
        size (tuple): KÃ­ch thÆ°á»›c figure matplotlib (máº·c Ä‘á»‹nh (5, 5))
    """
    try:
        img = PIL.Image.open(img_path).convert("RGB")
        plt.figure(figsize=size)
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
    except FileNotFoundError:
        print(f"File khÃ´ng tá»“n táº¡i: {img_path}")


img_path = f"/kaggle/input/landmark-recognition-2021/train/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
fig = show_data(img_path)


predict_single(model, test_df=val_df, data_dir=DATA_DIR, idx=5, transform=val_transform)


from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMAGE_SIZE = 260
BATCH_SIZE = 32

train_transform = A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ImageCompression(quality_lower=99, quality_upper=100),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.CLAHE(p=0.1),
    A.GaussianBlur(p=0.1),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


pip install efficientnet_pytorch


import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

NUM_CLASSES = 100  # hoáº·c len(landmark_id_to_idx)

# Táº£i EfficientNet-B2 Ä‘Ã£ pretrained trÃªn ImageNet
model = EfficientNet.from_pretrained('efficientnet-b2')

# Sá»­a classifier
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, NUM_CLASSES)

# Chuyá»ƒn sang GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b2')

# In toÃ n bá»™ cáº¥u trÃºc model
print(model)


for images, labels in tqdm(train_loader, desc="Debug mode"):
    try:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
    except Exception as e:
        print("Lá»—i á»Ÿ batch vá»›i label:", labels)
        raise e


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"



train_dataset = LandmarkDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset   = LandmarkDataset(val_df,   DATA_DIR,   transform=val_transform)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)


import wandb
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# Khá»Ÿi táº¡o wandb (chá»‰ cáº§n gá»�i 1 láº§n)
wandb.init(
    project="landmark-recognition",
    name="efficientnet-b0-run",
    config={
        "epochs": 20,
        "model": "efficientnet_b0",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader.batch_size,
        "image_size": 224,
    }
)

# Khá»Ÿi táº¡o giÃ¡ trá»‹ GAP tá»‘t nháº¥t
best_gap = 0.0

EPOCHS = wandb.config.epochs

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"ğŸ”� Validation Accuracy: {acc:.4f}")
    print(f"ğŸ“ˆ GAP@20: {gap:.4f}")

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model by GAP
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_efficientnet.pth")
        print(f"Saved best model (GAP={gap:.4f})")

wandb.finish()


import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
IMAGE_SIZE = 256
BATCH_SIZE =32
train_transform_2 = A.Compose([
    A.SmallestMaxSize(256),                     # Resize ngáº¯n nháº¥t vá»� 256 (giá»¯ tá»‰ lá»‡)
    A.RandomCrop(224, 224),                     # Cáº¯t ngáº«u nhiÃªn trung tÃ¢m vÃ¹ng quan trá»�ng
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.CLAHE(p=0.1),                             # LÃ m rÃµ chi tiáº¿t (áº£nh má»�)
    A.Sharpen(alpha=(0.1, 0.3), p=0.2),         # LÃ m sáº¯c nÃ©t
    A.GaussianBlur(p=0.1),                      # TÄƒng generalization
    A.Normalize(                                # Chuáº©n hÃ³a theo ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
val_transform_2 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"
train_dataset_2 = LandmarkDataset(train_df, DATA_DIR, transform=train_transform_2)
val_dataset_2 = LandmarkDataset(val_df, DATA_DIR, transform=val_transform_2)

train_loader_2 = DataLoader(train_dataset_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_2 = DataLoader(val_dataset_2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)


import os
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# Khá»Ÿi táº¡o wandb
wandb.init(
    project="landmark-recognition",
    name="resnet50-run_ver2",
    config={
        "epochs": 20,
        "model": "resnet50_ver2",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader_2.batch_size,
        "image_size": 224,
    }
)

best_gap = 0.0
EPOCHS = wandb.config.epochs
checkpoint_path = "checkpoint.pth"

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader_2, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader_2)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader_2, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"GAP@20: {gap:.4f}")

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model + checkpoint
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_resnet_ver_2.pth")
        print(f"Saved best model (GAP={gap:.4f})")

        # Save checkpoint for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_gap': best_gap
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

wandb.finish()


import torch

checkpoint = torch.load("/kaggle/input/model-effiecient/best_model_resnet_ver_2-2.pth", map_location="cpu")
print(f"Checkpoint saved at epoch: {checkpoint['epochs']}")


checkpoint = torch.load("/kaggle/input/model-effiecient/best_model_resnet_ver_2-2.pth", map_location="cpu")

# Náº¿u chá»‰ lÃ  state_dict:
if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
    print("Ä�Ã¢y lÃ  model state_dict, khÃ´ng cÃ³ thÃ´ng tin epoch.")
else:
    print(f"Checkpoint saved at epoch: {checkpoint['epoch']}")


import os
import torch
import wandb
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# Thiáº¿t láº­p device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khá»Ÿi táº¡o wandb (resume run báº±ng name)
wandb.init(
    project="landmark-recognition",
    name="resnet50-run_ver2",  # giá»¯ nguyÃªn Ä‘á»ƒ ná»‘i tiáº¿p log
    id="1wepdhs5",               # ID Ä‘á»ƒ wandb nháº­n diá»‡n Ä‘Ãºng run
    resume="allow",               # tá»± Ä‘á»™ng ná»‘i náº¿u cÃ³ run trÃ¹ng tÃªn
    config={
        "epochs": 20,
        "model": "resnet50",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader_2.batch_size,
        "image_size": 224,
    }
)

# Load mÃ´ hÃ¬nh
checkpoint_path = "/kaggle/working/best_model_resnet_ver_2.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
print("Model weights loaded successfully.")

# CÃ¡c biáº¿n huáº¥n luyá»‡n
EPOCHS = wandb.config.epochs
start_epoch = 13               # â†� Báº¡n Ä‘Ã£ train xong epoch 5
best_gap = 0.0

# Optimizer & loss
optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
criterion = torch.nn.CrossEntropyLoss()

# Epoch loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader_2, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader_2)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader_2, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"GAP@20: {gap:.4f}")

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_resnet_ver_2.pth")
        print(f"Saved best model (GAP={gap:.4f})")

wandb.finish()


IMAGE_SIZE = 260  # Chuáº©n cá»§a EfficientNet-B2

train_transform_2 = A.Compose([
    A.SmallestMaxSize(288),                     # Ä�áº·t lá»›n hÆ¡n IMAGE_SIZE má»™t chÃºt Ä‘á»ƒ cÃ²n crop
    A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.CLAHE(p=0.1),
    A.Sharpen(alpha=(0.1, 0.3), p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_transform_2 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


DATA_DIR = "/kaggle/input/landmark-recognition-2021/train/"
train_dataset_2 = LandmarkDataset(train_df, DATA_DIR, transform=train_transform_2)
val_dataset_2 = LandmarkDataset(val_df, DATA_DIR, transform=val_transform_2)

train_loader_2 = DataLoader(train_dataset_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_2 = DataLoader(val_dataset_2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

NUM_CLASSES = 100  # hoáº·c len(landmark_id_to_idx)

# Táº£i EfficientNet-B2 Ä‘Ã£ pretrained trÃªn ImageNet
model = EfficientNet.from_pretrained('efficientnet-b2')

# Sá»­a classifier
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, NUM_CLASSES)

# Chuyá»ƒn sang GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


import os
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# Khá»Ÿi táº¡o wandb (chá»‰ cáº§n gá»�i 1 láº§n)
wandb.init(
    project="landmark-recognition",
    name="efficientnet-b2-run",
    config={
        "epochs": 20,
        "model": "efficientnet_b2_ver2",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader_2.batch_size,
        "image_size": 260,
    }
)

# Biáº¿n cáº¥u hÃ¬nh vÃ  tráº¡ng thÃ¡i
EPOCHS = wandb.config.epochs
best_gap = 0.0
checkpoint_path = "checkpoint_b2.pth"

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader_2, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader_2)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader_2, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"GAP@20: {gap:.4f}")

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model by GAP + checkpoint
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_efficientnet_ver_2.pth")
        print(f"Saved best model (GAP={gap:.4f})")

        # Save full checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_gap': best_gap
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

wandb.finish()


import os
import torch
import wandb
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# Thiáº¿t láº­p device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wandb.init(
    project="landmark-recognition",
    name="efficientnet-b2-run",  # giá»¯ nguyÃªn Ä‘á»ƒ ná»‘i tiáº¿p log
    id="fm1g0owp",               # ID Ä‘á»ƒ wandb nháº­n diá»‡n Ä‘Ãºng run
    resume="allow",               # tá»± Ä‘á»™ng ná»‘i náº¿u cÃ³ run trÃ¹ng tÃªn
    config={
        "epochs": 20,
        "model": "efficientnet_b2_ver2",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "batch_size": train_loader_2.batch_size,
        "image_size": 260,
    }
)

# Load mÃ´ hÃ¬nh
checkpoint_path = "/kaggle/input/model-effiecient/best_model_efficientnet_ver_2-2.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
print("Model weights loaded successfully.")

# CÃ¡c biáº¿n huáº¥n luyá»‡n
EPOCHS = wandb.config.epochs
start_epoch = 12               # â†� Báº¡n Ä‘Ã£ train xong epoch 5
best_gap = 0.9527

# Optimizer & loss
optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
criterion = torch.nn.CrossEntropyLoss()

# Epoch loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader_2, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader_2)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in tqdm(val_loader_2, desc=f"Epoch {epoch+1} - Validation"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    gap = compute_gap(all_preds, all_confs, all_labels)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"GAP@20: {gap:.4f}")

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_accuracy": acc,
        "val_gap@20": gap
    })

    # Save best model
    if gap > best_gap:
        best_gap = gap
        torch.save(model.state_dict(), "best_model_resnet_ver_2.pth")
        print(f"Saved best model (GAP={gap:.4f})")

wandb.finish()

