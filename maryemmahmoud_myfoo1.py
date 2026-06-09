import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_path = '/kaggle/input/rsna-bcd-1024x512-preprocessed'
csv_path = '/kaggle/input/rsna-breast-cancer-detection/train.csv'
images_folder = os.path.join(data_path, 'train_images')


labels_df = pd.read_csv(csv_path)


labels_df['image_path'] = labels_df['patient_id'].astype(str) + '/' + labels_df['image_id'].astype(str) + '.png'
labels_df['image_path'] = labels_df['image_path'].apply(lambda x: os.path.join(images_folder, x))


train_df, temp_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['cancer'])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cancer'])


img_size = (256, 256)

train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

valid_test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])


class BreastCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'cancer']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)



train_dataset = BreastCancerDataset(train_df, transform=train_transform)
valid_dataset = BreastCancerDataset(valid_df, transform=valid_test_transform)
test_dataset = BreastCancerDataset(test_df, transform=valid_test_transform)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)


def show_batch(loader):
    images, labels = next(iter(loader))
    images = images[:8]
    labels = labels[:8]

    plt.figure(figsize=(16, 8))
    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f'Label: {int(labels[i].item())}')
        plt.axis('off')
    plt.show()


weights = EfficientNet_B5_Weights.IMAGENET1K_V1
model = efficientnet_b5(weights=weights)


from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights


model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
model = model.to(device)


model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


epochs = 5
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Saving Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")



print("\nTesting the Best Model...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

correct = 0
total = 0


with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")


