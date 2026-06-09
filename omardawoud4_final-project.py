import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, efficientnet_b0, ResNet50_Weights, EfficientNet_B0_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


from pathlib import Path

DATA_DIR = Path('/kaggle/input/cassava-leaf-disease-classification')
TRAIN_DIR = DATA_DIR / 'train_images'
TEST_DIR = DATA_DIR / 'test_images'
TRAIN_CSV = DATA_DIR / 'train.csv'

train_df = pd.read_csv(TRAIN_CSV)

print(f"Total training images: {len(train_df)}")
train_df.head()



class_names = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

train_df['class_name'] = train_df['label'].map(class_names)

plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='class_name', palette='viridis')
plt.title('Class Distribution in Training Data', fontsize=16, fontweight='bold')
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nClass Distribution:")
print(train_df['class_name'].value_counts())


fig, axes = plt.subplots(5, 5, figsize=(15, 15))
fig.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')

for idx, (label, name) in enumerate(class_names.items()):
    class_samples = train_df[train_df['label'] == label].sample(5)

    for i, (_, row) in enumerate(class_samples.iterrows()):
        img_path = TRAIN_DIR / row['image_id']
        img = Image.open(img_path)
        axes[idx, i].imshow(img)
        axes[idx, i].axis('off')
        if i == 0:
            axes[idx, i].set_title(f'{name}', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(
    train_df,
    test_size=0.3,
    random_state=42,
    stratify=train_df['label']
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.333,  # 0.333 of 30% = 10% of total
    random_state=42,
    stratify=temp_data['label']
)

print(f"Total samples: {len(train_df)}")
print(f"Training samples: {len(train_data)} ({len(train_data)/len(train_df)*100:.1f}%)")
print(f"Validation samples: {len(val_data)} ({len(val_data)/len(train_df)*100:.1f}%)")
print(f"Test samples: {len(test_data)} ({len(test_data)/len(train_df)*100:.1f}%)")

print("\nClass distribution:")
print("Train:", train_data['label'].value_counts().sort_index().tolist())
print("Val:  ", val_data['label'].value_counts().sort_index().tolist())
print("Test: ", test_data['label'].value_counts().sort_index().tolist())


class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.loc[idx, 'image_id'])
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.loc[idx, 'label']

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label



IMG_SIZE = 224

train_transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])



train_dataset = CassavaDataset(train_data, TRAIN_DIR, transform=train_transform)
val_dataset = CassavaDataset(val_data, TRAIN_DIR, transform=val_transform)
test_dataset = CassavaDataset(test_data, TRAIN_DIR, transform=val_transform)

print(f"Train dataset: {len(train_dataset)} images")
print(f"Validation dataset: {len(val_dataset)} images")
print(f"Test dataset: {len(test_dataset)} images")



BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  
    pin_memory=True
)



# Experiment 1: Custom CNN from Scratch
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Experiment 2: ResNet50 with Transfer Learning
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50Model, self).__init__()
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(weights=None)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}',
                         'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}',
                             'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, model_name='model'):
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f'✓ Best model saved with validation accuracy: {best_val_acc:.2f}%')

    return history


model1 = CustomCNN(num_classes=5).to(device)
print(f"Total parameters: {sum(p.numel() for p in model1.parameters()):,}")

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10)

# Train
history1 = train_model(
    model1, train_loader, val_loader, criterion, optimizer1, scheduler1,
    num_epochs=10, device=device, model_name='custom_cnn'
)


model2 = ResNet50Model(num_classes=5, pretrained=True).to(device)
print(f"Total parameters: {sum(p.numel() for p in model2.parameters()):,}")

optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=15)

history2 = train_model(
    model2, train_loader, val_loader, criterion, optimizer2, scheduler2,
    num_epochs=10, device=device, model_name='resnet50'
)


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

#Loss comparison
axes[0].plot(history1['train_loss'], label='Custom CNN - Train', marker='o')
axes[0].plot(history1['val_loss'], label='Custom CNN - Val', marker='o')

axes[0].plot(history2['train_loss'], label='ResNet50 - Train', marker='s')
axes[0].plot(history2['val_loss'], label='ResNet50 - Val', marker='s')

axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

#Accuracy comparison
axes[1].plot(history1['train_acc'], label='Custom CNN - Train', marker='o')
axes[1].plot(history1['val_acc'], label='Custom CNN - Val', marker='o')

axes[1].plot(history2['train_acc'], label='ResNet50 - Train', marker='s')
axes[1].plot(history2['val_acc'], label='ResNet50 - Val', marker='s')

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training and Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Evaluating {model_name}'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    return all_preds, all_labels, acc


#Load best models and evaluate
models_to_eval = [
    ('Custom CNN', model1, 'custom_cnn_best.pth'),
    ('ResNet50', model2, 'resnet50_best.pth')
]

results = {}

for model_name, model, checkpoint_path in models_to_eval:
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # Evaluate on train set
    train_preds, train_labels, train_acc = evaluate_model(
        model, train_loader, device, f'{model_name} (Train)'
    )

    # Evaluate on val set
    val_preds, val_labels, val_acc = evaluate_model(
        model, val_loader, device, f'{model_name} (Val)'
    )

    # Evaluate on test set
    test_preds, test_labels, test_acc = evaluate_model(
        model, test_loader, device, f'{model_name} (Test)'
    )

    results[model_name] = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_preds': test_preds,
        'test_labels': test_labels
    }

    print(f"\n{'='*60}")
    print(f"{model_name} Results:")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")



summary_df = pd.DataFrame([
    {
        'Model': name,
        'Train Accuracy (%)': f"{res['train_acc']:.2f}",
        'Validation Accuracy (%)': f"{res['val_acc']:.2f}",
        'Test Accuracy (%)': f"{res['test_acc']:.2f}"
    }
    for name, res in results.items()
])

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)





