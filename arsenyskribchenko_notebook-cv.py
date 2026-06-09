import os
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DogBreedDataset(Dataset):
    def __init__(self, labels_path, root_dir, transform=None):
        self.labels_df = pd.read_csv(labels_path)
        self.root_dir = root_dir
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.labels_df['breed'] = self.label_encoder.fit_transform(self.labels_df['breed'])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = DogBreedDataset(
    labels_path='/kaggle/input/dog-breed-identification/labels.csv', 
    root_dir='/kaggle/input/dog-breed-identification/train', 
    transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.labels_df['breed'].unique()))  # Установите количество классов

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)  # Обучаем только последний слой

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')


import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()  # Sort to ensure consistent order

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, img_name[:-4]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_image_dir = '/kaggle/input/dog-breed-identification/test'
test_dataset = TestDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels_df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
label_names = sorted(labels_df['breed'].unique())

output_file = 'submission_cv.csv'
with open(output_file, 'w') as f:
    header = 'id,' + ','.join(label_names) + '\n'
    f.write(header)

    for inputs, img_ids in test_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)

        for img_id, probs in zip(img_ids, probabilities):
            probs_str = ','.join(map(str, probs.cpu().numpy()))
            line = f"{img_id},{probs_str}\n"
            f.write(line)

