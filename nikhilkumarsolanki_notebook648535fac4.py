import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


KAGGLE_BASE_DIR = '/kaggle/input/state-farm-distracted-driver-detection'
DATA_DIR = os.path.join(KAGGLE_BASE_DIR, 'imgs')
CSV_PATH = os.path.join(KAGGLE_BASE_DIR, 'driver_imgs_list.csv')
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
RANDOM_SEED = 42 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 

print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")


df = pd.read_csv(CSV_PATH)

unique_drivers = df['subject'].unique()
print(f"Total number of unique drivers: {len(unique_drivers)}")

# Random shuffle of drivers 
np.random.seed(RANDOM_SEED)
np.random.shuffle(unique_drivers)

train_drivers, temp_drivers = train_test_split(unique_drivers, test_size=0.2, random_state=RANDOM_SEED)
val_drivers, test_drivers = train_test_split(temp_drivers, test_size=0.5, random_state=RANDOM_SEED)

print(f"\nTraining drivers ({len(train_drivers)}): {train_drivers}")
print(f"Validation drivers ({len(val_drivers)}): {val_drivers}")
print(f"Test drivers ({len(test_drivers)}): {test_drivers}")

train_df = df[df['subject'].isin(train_drivers)].copy()
val_df = df[df['subject'].isin(val_drivers)].copy()
test_df = df[df['subject'].isin(test_drivers)].copy()

print("\nDataset Split Summary : ")
print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")
print(f"Test set: {len(test_df)} images")
print(f"Total images accounted for: {len(train_df) + len(val_df) + len(test_df)}")

assert len(set(train_df['subject']) & set(val_df['subject'])) == 0
print("\nNo overlap in drivers between sets.")


# transformations for training with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transformations for validation and test set without augmentation
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Transformation pipelines defined.")


class DriverDataset(Dataset):

    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = {classname: i for i, classname in enumerate(sorted(self.df['classname'].unique()))}
        self.idx_to_class = {i: classname for classname, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.df.iloc[idx]
        
        img_path = os.path.join(self.data_dir, 'train', row['classname'], row['img'])
        
        image = Image.open(img_path).convert("RGB")
        
        label = self.class_to_idx[row['classname']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

print("DriverDataset class defined.")


train_dataset = DriverDataset(df=train_df, data_dir=DATA_DIR, transform=train_transforms)
val_dataset = DriverDataset(df=val_df, data_dir=DATA_DIR, transform=val_test_transforms)
test_dataset = DriverDataset(df=test_df, data_dir=DATA_DIR, transform=val_test_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"DataLoaders created successfully.")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in val_loader: {len(val_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")


images, labels = next(iter(train_loader))

print(f"Shape of a batch of images: {images.shape}")
print(f"Shape of a batch of labels: {labels.shape}")

import torchvision

def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

class_names = train_dataset.idx_to_class
label_names = [class_names[l.item()] for l in labels]

img_grid = torchvision.utils.make_grid(images[:8], nrow=4)
plt.figure(figsize=(12, 6))
imshow(img_grid, title=f"Sample Augmented Images\nLabels: {label_names[:8]}")
plt.show()





import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 56 * 56, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SEResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, stride)
        self.se = SEBlock(out_channels)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE block
        out += identity
        out = self.relu(out)
        return out


def get_finetuned_mobilenet(num_classes=10):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

def get_finetuned_efficientnet(num_classes=10):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

def get_finetuned_vit(num_classes=10):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    return model


import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim


EPOCHS_HEAD_ONLY = 5       
EPOCHS_FULL_FINETUNE = 10  
LR_HEAD = 1e-3
LR_FULL = 1e-5             
SAVE_MODELS_PATH = './saved_models/'
LOGS_PATH = './training_logs/'


os.makedirs(SAVE_MODELS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)


finetuned_models = {
    "FineTuned_MobileNetV2": get_finetuned_mobilenet(num_classes=10),
    "FineTuned_EfficientNet": get_finetuned_efficientnet(num_classes=10),
    "FineTuned_ViT": get_finetuned_vit(num_classes=10)
}

print(f"Ready to fine-tune {len(finetuned_models)} models.")


for model_name, model in finetuned_models.items():
    print(f"\n {model_name}")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\nTraining the classifier head ")
    optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    for epoch in range(EPOCHS_HEAD_ONLY):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_head.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_head.step()
        print(f"Head Training Epoch {epoch+1}/{EPOCHS_HEAD_ONLY} complete.")
        

    print("\nUnfreezing and finetuning all layers ")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_full = optim.Adam(model.parameters(), lr=LR_FULL)

    for epoch in range(EPOCHS_FULL_FINETUNE):
        start_time = time.time()
        model.train()
        running_loss, running_corrects = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_full.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_full.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_MODELS_PATH, f'best_{model_name}.pth'))
            
        end_time = time.time()
        print(f"Fine-Tuning Epoch {epoch+1}/{EPOCHS_FULL_FINETUNE} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Time: {end_time-start_time:.2f}s")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(LOGS_PATH, f'{model_name}_history.csv'), index=False)
    print(f"Fine-tuning finished for {model_name}. Best Val Acc: {best_val_acc:.4f}")

