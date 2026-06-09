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


import os, copy, csv, json, random
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns; import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])


from torch.utils.data import DataLoader
train_dataset = datasets.ImageFolder(root='/kaggle/input/street-food-image-classification/train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class_names = train_dataset.classes
class_names


from PIL import Image
import os

test_images = []
test_dir = '/kaggle/input/street-food-image-classification/test_images'

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    test_images.append((img_name, image))


class StreetFoodCNN(nn.Module):
    """
    Simple CNN for multiâ€‘class image classification.
    Works with any input resolution because of AdaptiveAvgPool2d.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (32, H, W)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # (64, H/4, W/4)

            nn.AdaptiveAvgPool2d(1)                      # (64, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 64
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)                  # raw logits
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


num_classes = 10
model = StreetFoodCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(10):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
       

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= total
    train_acc = correct / total



import os
from PIL import Image

test_folder = '/kaggle/input/street-food-image-classification/test_images'
model.eval()
with torch.no_grad():
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0) # add batch dim

        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(1).item()

        print(f"{img_name}: Predicted class index {pred_class} with confidence {probs.max().item():.4f}")


class_names = train_dataset.classes
print("Predicted:", class_names[pred_class])


import os, csv
from PIL import Image
import torch
from torchvision import transforms

class_names  = train_dataset.classes
test_folder  = '/kaggle/input/street-food-image-classification/test_images'

# ðŸ”§  writable path for the CSV
csv_file = '/kaggle/working/streetfood_predictions.csv'   # or simply 'streetfood_predictions.csv'

model.eval()
with torch.no_grad():
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)                            # ðŸ”§ use csv.writer
        writer.writerow(['image_id', 'label'])

        for img_name in os.listdir(test_folder):
            img_path = os.path.join(test_folder, img_name)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            outputs = model(input_tensor)
            pred_class = outputs.argmax(dim=1).item()
            pred_class_name = class_names[pred_class]

            writer.writerow([img_name, pred_class_name])

print(f"âœ… Predictions saved to {csv_file}")





