import os
import pandas as pd
data = []
for i, class_name in enumerate(sorted(os.listdir('/kaggle/input/plant-seedlings-classification/train'))):
    # print(i, class_name)
    for img in os.listdir(f'/kaggle/input/plant-seedlings-classification/train/{class_name}'):
        # print(img)
        img_pa = class_name+ '/' + img
        data.append([img_pa, class_name])

pd.DataFrame(data, columns=['file', 'speicies']).to_csv('train_labels.csv', index=False)


df_train  = pd.read_csv('/kaggle/wogemini-2.0-flashrking/train_labels.csv')


df_train


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_train['speicies'] = le.fit_transform(df_train['speicies'])


num_classes = len(df_train['speicies'].unique())


from torch.utils.data import  Dataset, DataLoader, random_split
import cv2

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


transform  = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda( lambda img: F.adjust_sharpness(img, sharpness_factor=4)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.45, .48, .40], std = [.22, .22, .22])
])

class ImageDataset(Dataset):
    def __init__(self, img_folder, df , transform=None, is_test=False):
        self.df= df
        self.image_folder = img_folder
        self.transform =transform
        self.is_test = is_test
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.df['file'][idx])
    
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img
        else:
            label = int(self.df['speicies'][idx])
            return img, torch.tensor(label)

train_dataset = ImageDataset('/kaggle/input/plant-seedlings-classification/train', df_train, transform, is_test=False)


train_size = int(.8*len(train_dataset))
val_size = len(train_dataset) - train_size

train_set, val_set = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model
model = SimpleCNN(num_classes=num_classes).to('cuda')


from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
model = model.to('cuda')
def run_epoch(loader, model, criterion, optimizer=None):
    model.train() if optimizer else model.eval()

    total_loss, preds, labels_all = 0, [], []

    for x, y in loader:
        x, y = x.to('cuda'), y.to('cuda')
        if optimizer: optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            out = model(x)
            loss = criterion(out, y)

            if optimizer: loss.backward(), optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds += out.argmax(1).cpu().tolist()
        labels_all += y.cpu().tolist()
    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds)*10




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
NUM_EPOCHS = 8
for e in range(NUM_EPOCHS):
    print(f"\nEpoch {e+1}/{NUM_EPOCHS}")
    tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
    vl_loss, vl_acc = run_epoch(val_loader, model, criterion)
    print(f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f}, Acc: {vl_acc:.4f}")

