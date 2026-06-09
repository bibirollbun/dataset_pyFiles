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


import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 5
N_FOLDS = 5

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

MODELS_DIR = "/kaggle/input/cassava-resnet-fold-models"

print("Device:", DEVICE)


class CassavaTestDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, row["image_id"]

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


sub_df = pd.read_csv(SAMPLE_SUB)
test_df = sub_df.copy()

test_dataset = CassavaTestDataset(test_df, TEST_DIR, transforms=val_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

len(test_df), test_df.head()


all_preds = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

for fold in range(N_FOLDS):
    print(f"Loading model for fold {fold}...")
    
    model = build_model().to(DEVICE)
    model_path = os.path.join(MODELS_DIR, f"best_model_fold{fold}.pth")
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    fold_preds = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            fold_preds.append(probs.cpu().numpy())

    fold_preds = np.concatenate(fold_preds, axis=0)
    all_preds += fold_preds / N_FOLDS 

pred_labels = all_preds.argmax(axis=1)
sub_df["label"] = pred_labels
sub_df.head()


sub_df.to_csv("submission.csv", index=False)
print("submission.csv saved!")

