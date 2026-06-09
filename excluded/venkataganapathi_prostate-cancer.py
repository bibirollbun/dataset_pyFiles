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
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q imagecodecs


import os

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
import seaborn as sns
import cv2

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go


# Location of the training images

BASE_PATH = '../input/prostate-cancer-grade-assessment'

# image and mask directories
data_dir = f'{BASE_PATH}/train_images'
mask_dir = f'{BASE_PATH}/train_label_masks'


# Location of training labels
train = pd.read_csv(f'{BASE_PATH}/train.csv').set_index('image_id')
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


display(train.head())
print("Shape of training data :", train.shape)
print("unique data provider :", len(train.data_provider.unique()))
print("unique isup_grade(target) :", len(train.isup_grade.unique()))
print("unique gleason_score :", len(train.gleason_score.unique()))


train.isna().sum()


display(test.head())
print("Shape of training data :", test.shape)
print("unique data provider :", len(test.data_provider.unique()))


import matplotlib.pyplot as plt
import seaborn as sns

def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 3 * size))
    total = float(len(df))
    
    # Fix: pass feature name as x=... and df as data
    sns.countplot(x=feature, data=df, order=df[feature].value_counts().index, palette='Set2', ax=ax)
    
    plt.title(title)
    
    # Loop through bars to add text
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 3,
                '{:1.2f}%'.format(100 * height / total),
                ha="center")
    
    plt.show()



plot_count(df=train, feature='data_provider', title='Data Provider Count and % Plot')


plot_count(df=train, feature='isup_grade', title='isup_grade count and %age plot')


plot_count(df=train, feature='gleason_score', title = 'gleason_score count and %age plot', size=3)


def plot_relative_distribution(df, feature, hue, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', title = 'relative count plot of isup_grade with data_provider', size=2)


plot_relative_distribution(df=train, feature='gleason_score', hue='data_provider', title = 'relative count plot of gleason_score with data_provider', size=3)


plot_relative_distribution(df=train, feature='isup_grade', hue='gleason_score', title = 'relative count plot of isup_grade with gleason_score', size=3)


def display_images(slides): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        image = openslide.OpenSlide(os.path.join(data_dir, f'{slide}.tiff'))
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region((1780,1950), 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 



images = [
    '07a7ef0ba3bb0d6564a73f4f3e1c2293',
    '037504061b9fba71ef6e24c48c6df44d',
    '035b1edd3d1aeeffc77ce5d248a01a53',
    '059cbf902c5e42972587c8d17d49efed',
    '06a0cbd8fd6320ef1aa6f19342af2e68',
    '06eda4a6faca84e84a781fee2d5f47e1',
    '0a4b7a7499ed55c71033cefb0765e93d',
    '0838c82917cd9af681df249264d2769c',
    '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde',
    '05abe25c883d508ecc15b6e857e59f32',
    '05f4e9415af9fdabc19109c980daf5ad',
    '060121a06476ef401d8a21d6567dee6d',
    '068b0e3be4c35ea983f77accf8351cc8',
    '08f055372c7b8a7e1df97c6586542ac8'
]

display_images(images)


def display_masks(slides): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
    plt.show()


display_masks(images)


data_providers = ['karolinska', 'radboud']
train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
masks = os.listdir(mask_dir)
masks_df = pd.Series(masks).to_frame()
masks_df.columns = ['mask_file_name']
masks_df['image_id'] = masks_df.mask_file_name.apply(lambda x: x.split('_')[0])
train_df = pd.merge(train_df, masks_df, on='image_id', how='outer')
del masks_df
print(f"There are {len(train_df[train_df.mask_file_name.isna()])} images without a mask.")

## removing items where image mask is null
train_df = train_df[~train_df.mask_file_name.isna()]


def load_and_resize_image(img_id):
    """
    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
    """
    biopsy = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))
    return cv2.resize(biopsy[-1], (512, 512))

def load_and_resize_mask(img_id):
    """
    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
    """
    biopsy = skimage.io.MultiImage(os.path.join(mask_dir, f'{img_id}_mask.tiff'))
    return cv2.resize(biopsy[-1], (512, 512))[:,:,0]


!pip install imagecodecs


labels = []
for grade in range(train.isup_grade.nunique()):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(22, 22))

    for i, row in enumerate(ax):
        idx = i // 2
        temp = train_df[
            (train_df.isup_grade == grade) & 
            (train_df.data_provider == data_providers[idx])
        ].image_id.head(4).reset_index(drop=True)

        if i % 2 < 1:
            labels.append(f'{data_providers[idx]} (image)')
            for j, col in enumerate(row):
                col.imshow(load_and_resize_image(temp[j]))
                col.set_title(f"ID: {temp[j]}")
        else:
            labels.append(f'{data_providers[idx]} (mask)')
            for j, col in enumerate(row):
                cmap_vals = ['white', 'green', 'red']
                if data_providers[idx] == 'radboud':
                    cmap_vals = ['white', 'lightgrey', 'green', 'orange', 'red', 'darkred']

                col.imshow(load_and_resize_mask(temp[j]),
                           cmap=matplotlib.colors.ListedColormap(cmap_vals),
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=len(cmap_vals)-1, clip=True))
                col.set_title(f"ID: {temp[j]}")

   #Fixed here: remove `size` and keep only `fontsize`
    for row, r in zip(ax[:, 0], labels):
        row.set_ylabel(r, rotation=90, fontsize=14)

    plt.suptitle(f'ISUP Grade {grade}', fontsize=20)
    plt.show()



def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    f, ax = plt.subplots(5,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]
        
        
        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

        alpha_content = PIL.Image.fromarray(alpha_content)
        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        data_provider = train.loc[image_id, 'data_provider']
        isup_grade = train.loc[image_id, 'isup_grade']
        gleason_score = train.loc[image_id, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")


overlay_mask_on_slide(images)


pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b',
    'fd0bb45eba479a7f7d953f41d574bf9f',
    'ff10f937c3d52eff6ad4dd733f2bc3ac',
    'feee2e895355a921f2b75b54debad328',
    'feac91652a1c5accff08217d19116f1c',
    'fb01a0a69517bb47d7f4699b6217f69d',
    'f00ec753b5618cfb30519db0947fe724',
    'e9a4f528b33479412ee019e155e1a197',
    'f062f6c1128e0e9d51a76747d9018849',
    'f39bf22d9a2f313425ee201932bac91a',
]

overlay_mask_on_slide(pen_marked_images)


## refer: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/145182

import random
random.seed(42)


results = np.random.randint(0,6,len(submission))
submission['isup_grade'] = results
submission.to_csv('submission.csv', index=False)


import os
import numpy as np
import pandas as pd
import openslide
import cv2
from tqdm import tqdm


# Constants and paths
DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment'
IMAGE_DIR = os.path.join(DATA_DIR, 'train_images')
PATCH_SAVE_DIR = './tiles'
PATCH_SIZE = 256
STRIDE = 256
TISSUE_THRESHOLD = 0.5  # keep patches with >50% tissue

# Load train metadata
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# Create save directory
os.makedirs(PATCH_SAVE_DIR, exist_ok=True)



def is_tissue(patch, threshold=0.5):
    """Returns True if tissue ratio > threshold in the patch"""
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, tissue_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    tissue_ratio = np.count_nonzero(tissue_mask) / tissue_mask.size
    return tissue_ratio > threshold


tile_info = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row['image_id']
    isup = row['isup_grade']
    gleason = row['gleason_score']
    path = os.path.join(IMAGE_DIR, f'{image_id}.tiff')

    try:
        slide = openslide.OpenSlide(path)
    except:
        print(f"Could not open {image_id}")
        continue

    # Use a lower resolution for faster processing
    slide_level = slide.level_count - 1
    dims = slide.level_dimensions[slide_level]

    slide_img = slide.read_region((0, 0), slide_level, dims).convert("RGB")
    slide_img = np.array(slide_img)

    h, w, _ = slide_img.shape

    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):
            patch = slide_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue

            if is_tissue(patch, threshold=TISSUE_THRESHOLD):
                patch_filename = f'{image_id}_{x}_{y}.png'
                patch_path = os.path.join(PATCH_SAVE_DIR, patch_filename)
                cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

                tile_info.append({
                    'tile_path': patch_path,
                    'image_id': image_id,
                    'x': x,
                    'y': y,
                    'isup_grade': isup,
                    'gleason_score': gleason
                })

    slide.close()



tile_df = pd.DataFrame(tile_info)
tile_df.to_csv('tile_metadata.csv', index=False)
print(f"Extracted {len(tile_df)} tiles and saved metadata to tile_metadata.csv")


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class TileDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['tile_path']
        label = int(row['isup_grade'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Transformations for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset
dataset = TileDataset('tile_metadata.csv', transform=train_transforms)

# Dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


import torchvision
import matplotlib.pyplot as plt

def show_batch(dl):
    images, labels = next(iter(dl))
    grid = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.title([str(label.item()) for label in labels])
    plt.axis('off')
    plt.show()

show_batch(dataloader)


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Load pretrained ResNet18 with correct weights syntax
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Replace final FC layer for 6 ISUP grade classes
model.fc = nn.Linear(model.fc.in_features, 6)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")


train_model(model, dataloader, criterion, optimizer, epochs=5)


from sklearn.model_selection import train_test_split

# Load tile metadata
tile_df = pd.read_csv('tile_metadata.csv')

# Split
train_df, val_df = train_test_split(tile_df, test_size=0.2, stratify=tile_df['isup_grade'], random_state=42)

# Save to separate CSVs
train_df.to_csv('tile_train.csv', index=False)
val_df.to_csv('tile_val.csv', index=False)

print(f"Train: {len(train_df)} tiles, Validation: {len(val_df)} tiles")


class TileDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['tile_path']
        label = int(row['isup_grade'])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Datasets
train_dataset = TileDataset('tile_train.csv', transform=train_transforms)
val_dataset = TileDataset('tile_val.csv', transform=val_transforms)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=5):
    model.train()
    best_val_acc = 0.0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved at epoch {epoch+1}")

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f} | Train Acc = {train_acc:.2f}% | Val Acc = {val_acc:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\nðŸ“Š Accuracy: {acc:.2f}% | Macro F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    model.train()
    return acc


# Loss with label smoothing (if using PyTorch >= 1.10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Optional: Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=5  # Change as needed
)


# Load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Evaluate on validation set
evaluate_model(model, val_loader)


from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set the path to your uploaded test image
img_path = "/kaggle/input/testimg/trailcancer.jpeg"  # âœ… Make sure this path is correct

# Same transforms as validation set
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load trained ResNet18 model
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 6)  # ISUP grades: 0â€“5
model = model.to(device)

# Load best model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Load & preprocess image
image = Image.open(img_path).convert("RGB")
input_tensor = val_transforms(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    pred_class = torch.argmax(outputs, dim=1).item()

#Print and visualize
print(f"Predicted ISUP Grade: {pred_class}")
plt.imshow(image)
plt.title(f"Predicted ISUP Grade: {pred_class}", fontsize=16)
plt.axis('off')
plt.show()





