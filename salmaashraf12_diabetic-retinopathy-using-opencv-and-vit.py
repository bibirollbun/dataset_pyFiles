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


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset (Example: APTOS 2019 from Kaggle)
data_path = "/kaggle/input/aptos2019-blindness-detection/"
train_csv = os.path.join(data_path, "train.csv")
test_csv = os.path.join(data_path, "test.csv")
train_images_dir = os.path.join(data_path, "train_images")
test_images_dir = os.path.join(data_path, "test_images")


# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Display dataset information
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())



# Plot class distribution
sns.countplot(x=train_df['diagnosis'])
plt.title("Distribution of Diabetic Retinopathy Stages")
plt.show()



# Function to visualize images
def show_images(image_paths, labels=None, cols=5):
    rows = len(image_paths) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = axes.flatten()
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        if labels is not None:
            axes[idx].set_title(f"Stage: {labels[idx]}")
        axes[idx].axis("off")
    plt.show()


# Show sample images
sample_images = train_df.sample(10)
image_paths = [os.path.join(train_images_dir, img + ".png") for img in sample_images['id_code']]
labels = sample_images['diagnosis'].tolist()
show_images(image_paths, labels)


# Data preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])


# Load data
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = self.labels.iloc[idx, 1]  # Class label (0 to 4)

        if self.transform:
            image = self.transform(image)
        
        return image, label


train_dataset = DRDataset(csv_file=train_csv, img_dir=train_images_dir, transform=transform)
val_dataset = DRDataset(csv_file=test_csv, img_dir=test_images_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load Vision Transformer model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np


# Plot class distribution using seaborn
sns.countplot(x=train_df['diagnosis'])
plt.title("Distribution of Diabetic Retinopathy Stages")
plt.show()

# Interactive class distribution using plotly
fig = px.histogram(train_df, x='diagnosis', title="Distribution of Diabetic Retinopathy Stages")
fig.show()


# Object Detection - Extract ROI using OpenCV
def extract_roi(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected ROI")
    plt.axis("off")
    plt.show()

# Show ROI extraction on sample image
extract_roi(image_paths[0])


# Function to visualize images using OpenCV
def show_images_opencv(image_paths, labels=None):
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        if labels is not None:
            plt.title(f"Stage: {labels[idx]}")
        plt.axis("off")
        plt.show()

# Show sample images
sample_images = train_df.sample(5)
image_paths = [os.path.join(train_images_dir, img + ".png") for img in sample_images['id_code']]
labels = sample_images['diagnosis'].tolist()
show_images_opencv(image_paths, labels)




