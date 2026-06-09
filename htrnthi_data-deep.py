# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        os.path.join(dirname, filename)
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, time, math, json, random, gc
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    TrainingArguments, Trainer, set_seed
)

IMAGE_SIZE = 224
EPOCHS = 10
BATCH_TRAIN = 32
BATCH_EVAL = 64
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

USE_CLASS_WEIGHTS = True


def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); set_seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.inference_mode()
def benchmark_fps(model, device, batch_size=64, image_size=224, steps=50, warmup=10):
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if device.type == "cuda": torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(pixel_values=x).logits
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _ = model(pixel_values=x).logits
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    imgs = steps * batch_size
    return imgs / (t1 - t0)


df = pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')


df_cl = df[["image_id", "label"]].copy()

labels_keep = ["normal", "blast", "brown_spot", "bacterial_leaf_blight"]
df_cl = df_cl[df_cl["label"].isin(labels_keep)]
df_cl["Source"] = "Kaggle"

print(df_cl['label'].value_counts())
df_cl.head()



import os
import pandas as pd

def build_df_from_folder(root_dir, fixed_label="bacterial_leaf_blight"):
    data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            data.append({"path": file_path, "label": fixed_label})
    return pd.DataFrame(data, columns=["path", "label"])
    
root = "/kaggle/input/rice-leaf-diseases-detection/Rice_Leaf_Diease/Rice_Leaf_Diease/train/"

df_blight = build_df_from_folder(os.path.join(root, "sheath_blight"), fixed_label="bacterial_leaf_blight")
df_blast  = build_df_from_folder(os.path.join(root, "leaf_blast"), fixed_label="blast")
df_brown  = build_df_from_folder(os.path.join(root, "brown_spot"), fixed_label="brown_spot")
df_healthy  = build_df_from_folder(os.path.join(root, "healthy"), fixed_label="Healthy")

df_blast  = df_blast.sample(n=400, random_state=42)
df_brown  = df_brown.sample(n=1000, random_state=42)

df_kg = pd.concat([df_blight, df_blast, df_brown, df_healthy], ignore_index=True)
df_kg["Source"] = "Kaggle"

print(df_kg['label'].value_counts())
df_kg.head()



import os
import csv

root_dir = "/kaggle/input/dl-dataset-hf/rice_leaf_disease_dataset/rice_leaf_disease_dataset"
output_csv = "dataset_HF.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")): 
                    writer.writerow([file, label])

print(f"Ä�Ã£ táº¡o file CSV: {output_csv}")


df_hf = pd.read_csv('/kaggle/working/dataset_HF.csv')


labels_keep = ["Healthy", "Blast", "Brownspot", "Bacterialblight"]
df_hf = df_hf[df_hf["label"].isin(labels_keep)]
df_hf["Source"] = "Hugging_face"


print(df_hf['label'].value_counts())
df_hf.head()



import os
import csv

root_dir = "/kaggle/input/dl-data-mendeley/Rice Leaf Diseases Dataset"
output_csv = "dataset_mendeley.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")): 
                    writer.writerow([file, label])

print(f"Ä�Ã£ táº¡o file CSV: {output_csv}")


df_md = pd.read_csv('/kaggle/working/dataset_mendeley.csv')


df_md["Source"] = "Mendeley"

print(df_md['label'].value_counts())
df_md.head()


import os
import csv

root_dir = "/kaggle/input/healthy-mendeley/Healthy Rice Leaf"
output_csv = "dataset_mendeley_healthy.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])

    label = "Healthy"
    for file in os.listdir(root_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")): 
            file_path = os.path.join(root_dir, file)
            writer.writerow([file_path, label])

print(f"Ä�Ã£ táº¡o file CSV: {output_csv}")



df_md2 = pd.read_csv('/kaggle/working/dataset_mendeley_healthy.csv')


df_md2["Source"] = "Mendeley"

print(df_md2['label'].value_counts())
df_md2.head()


import os
import csv

root_dir = "/kaggle/input/blast-roboflow"
output_csv = "dataset_robo.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")): 
                    writer.writerow([file, label])

print(f"Ä�Ã£ táº¡o file CSV: {output_csv}")


df_rb = pd.read_csv('/kaggle/working/dataset_robo.csv')[:-300]


df_rb["Source"] = "Roboflow"

print(df_rb['label'].value_counts())
df_rb.head()


df_cl = df_cl.rename(columns={"image_id": "path"})
df_cl['path'] = '/kaggle/input/paddy-disease-classification/train_images/' + df_cl['label'] + '/' + df_cl['path']
df_cl['path'][1]


df_hf = df_hf.rename(columns={"filename": "path"})
df_hf['path'] = '/kaggle/input/dl-dataset-hf/rice_leaf_disease_dataset/rice_leaf_disease_dataset/' + df_hf['label'] + '/' + df_hf['path']
df_hf = df_hf.reset_index(drop=True)
df_hf['path'][1]


df_md = df_md.rename(columns={"filename": "path"})
df_md['path'] = '/kaggle/input/dl-dataset-gg/data_deeplearning/' + df_md['label'] + '/' + df_md['path']
df_md['path'][1]


df_rb = df_rb.rename(columns={"filename": "path"})
df_rb['path'] = '/kaggle/input/blast-roboflow/Blast/' + df_rb['label'] + '/' + df_rb['path']
df_rb = df_rb.reset_index(drop=True)
df_rb['path'][1]


df = pd.concat([df_cl, df_hf, df_md, df_kg, df_rb, df_md2], ignore_index=True)


label_map = {
    "Browspot": "brown_spot",
    "Bacterialblight": "bacterial_leaf_blight",
    "Healthy": "normal",
    "Blast": "blast",
    "Brownspot": "brown_spot",
    "Bacterialblight": "bacterial_leaf_blight",
    "normal": "normal",
    "blast": "blast",
    "brown_spot": "brown_spot",
    "bacterial_leaf_blight": "bacterial_leaf_blight",
}


df["label"] = df["label"].map(label_map)


classes = ['normal', 'brown_spot', 'blast', 'bacterial_leaf_blight']
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}

df = df[df['label'].isin(classes)]
df['label_id'] = df['label'].map(label2id).astype(int)


print(df['label'].value_counts())
df.head()


import pandas as pd

df_count = df.groupby(["label", "Source"]).size().reset_index(name="Count")

table = df_count.pivot_table(
    index="label", columns="Source", values="Count", aggfunc="sum"
).fillna(0).astype(int)

table["Total"] = table.sum(axis=1)

table = table.reset_index()

table



import matplotlib.pyplot as plt
import pandas as pd

df_count = df.groupby(["label", "Source"]).size().reset_index(name="Count")

pivot_df = df_count.pivot_table(
    index="label", columns="Source", values="Count", aggfunc="sum"
).fillna(0)

order = pivot_df.sum().sort_values(ascending=False).index
pivot_df = pivot_df[order]

ax = pivot_df.plot(
    kind="bar", stacked=True, figsize=(12,6), width=0.6,
    color=plt.cm.Set2.colors
)

plt.title("PhÃ¢n bá»‘ sá»‘ lÆ°á»£ng áº£nh theo nhÃ£n bá»‡nh vÃ  nguá»“n dá»¯ liá»‡u", fontsize=14)
plt.xlabel("Label", fontsize=12)
plt.ylabel("Sá»‘ lÆ°á»£ng", fontsize=12)

plt.legend(title="Nguá»“n", loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=9)

plt.tight_layout()
plt.show()


import cv2
import matplotlib.pyplot as plt
import pandas as pd

df_kaggle = df[df["Source"] == "Kaggle"]

df_sample = df_kaggle.groupby("label").sample(n=3, random_state=42).reset_index(drop=True)

plt.figure(figsize=(12, 16))

for i in range(len(df_sample)):
    img_path = df_sample.loc[i, "path"]
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    plt.subplot(4, 3, i+1) 
    plt.imshow(img_rgb)
    plt.title(f"{df_sample.loc[i, 'label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()



train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print("Classes:", classes)
print("Train size:", len(train_df), "Val size:", len(valid_df))
print("Train distribution:\n", train_df["label"].value_counts())


cols = ['path', 'label_id']
base_train = Dataset.from_pandas(train_df[cols], preserve_index=False)
base_val = Dataset.from_pandas(valid_df[cols], preserve_index=False)


import cv2
import matplotlib.pyplot as plt

img_paths = [
    "/kaggle/input/manual/manual/dom-nau-tren-lua.jpg",
    "/kaggle/input/manual/manual/benh-dao-on-lua-1.jpg",
    "/kaggle/input/manual/manual/benh-dao-on-lua-01.jpg",
    "/kaggle/input/manual/manual/benh-chay-bia-la-tren-cay-lua-va-bien-phap-phong-tru-ra-sao.jpg",
    
    "/kaggle/input/manual/manual/edit_domnau.jpg",
    "/kaggle/input/manual/manual/benh-dao-on-lua-1-edited.jpg",
    "/kaggle/input/manual/manual/benh-dao-on-lua-01_da_edit.jpg",
    "/kaggle/input/manual/manual/edit.jpg"
]

new_size = (224, 224)

plt.figure(figsize=(12, 8))

for i, img_path in enumerate(img_paths):
    img_bgr = cv2.imread(img_path)

    img_resized = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 4, i+1)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"áº¢nh {i+1}")

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch

img_path = "/kaggle/input/paddy-disease-classification/train_images/blast/100004.jpg"
img = Image.open(img_path).convert("RGB")

aug_transforms = {
    "Original": None,
    "RandomResizedCrop": transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomRotation": transforms.RandomRotation(35),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "RandomAffine": transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
    ),
    "GaussianBlur": transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=1.0),
    "RandomGrayscale": transforms.RandomGrayscale(p=1.0),
    "RandomPerspective": transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    "RandomErasing": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.2), ratio=(0.3, 3.3)),
        transforms.ToPILImage()
    ]),
    "CenterCrop": transforms.CenterCrop(224),
}

n = len(aug_transforms)
cols = 4
rows = (n + cols - 1) // cols

plt.figure(figsize=(16, 12))
for i, (name, tfm) in enumerate(aug_transforms.items()):
    if tfm is None:
        aug_img = img
    else:
        aug_img = tfm(img)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(aug_img)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()



from torchvision import transforms
from PIL import Image
import torch
from datasets import DatasetDict

def build_ds_with_processor(processor, image_size=224):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.ToPILImage(),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ])

    def _load(p): 
        return Image.open(p).convert("RGB")

    def train_transform(batch):
        imgs = [_load(p) for p in batch["path"]]
        imgs = [train_tfm(img) for img in imgs]
        proc = processor(images=imgs, return_tensors="pt")
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(batch["label_id"])
        }

    def val_transform(batch):
        imgs = [_load(p) for p in batch["path"]]
        imgs = [val_tfm(img) for img in imgs]
        proc = processor(images=imgs, return_tensors="pt")
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(batch["label_id"])
        }

    train_ds = base_train.with_transform(train_transform)
    val_ds   = base_val.with_transform(val_transform)
    return DatasetDict({"train": train_ds, "validation": val_ds})


