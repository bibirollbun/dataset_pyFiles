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
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models




train_csv_path = "/kaggle/input/image-classification-real-or-ai-generated-photo/train.csv"
test_csv_path  = "/kaggle/input/image-classification-real-or-ai-generated-photo/test.csv"

train_img_dir  = "/kaggle/input/image-classification-real-or-ai-generated-photo/train/train"
test_img_dir   = "/kaggle/input/image-classification-real-or-ai-generated-photo/test/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device in use: {device}")




train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print("✅ Paths set successfully")
print("Train dir:", train_img_dir)
print("Test dir:", test_img_dir)
print("Train CSV columns:", train_df.columns.tolist())
print("Total training samples:", len(train_df))




class ImageDataset(Dataset):
    def _init_(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def _len_(self):
        return len(self.df)

    def _getitem_(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["Image"]
        else:
            label = int(row["Label"])
            return image, label




transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])




from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# ✅ Fixed custom dataset class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx, 1]
            return image, label

# ✅ Load datasets and dataloaders
train_dataset = ImageDataset(train_df, img_dir=train_img_dir, transform=transform_train)
test_dataset  = ImageDataset(test_df, img_dir=test_img_dir, transform=transform_test, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"✅ {len(train_dataset)} training samples loaded successfully")
print(f"✅ {len(test_dataset)} testing samples loaded successfully")



# Load resnet18 without internet
model = models.resnet18(weights=None)   # ✅ No internet download
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

print("✅ Training completed successfully!")




import torch
from tqdm import tqdm

model.eval()
predictions = []
image_ids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        # ✅ handle both (image) or (image, name)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, names = batch
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            names = [f"img_{i}" for i in range(len(images))]

        # ✅ add batch dimension if only 1 image (3D tensor)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(device)
        outputs = model(images)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)
        image_ids.extend(names)

print("✅ Prediction completed successfully!")




import pandas as pd

# ✅ Create submission DataFrame
submission = pd.DataFrame({
    "Image": image_ids,
    "Label": predictions
})

# ✅ Save CSV file
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved as /kaggle/working/submission.csv")

# ✅ Display first few rows
submission.head()



submission.to_csv("/kaggle/working/submission.csv", index=False)


