# Required libraries
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Define paths
img_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"
train_csv = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
test_csv = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"


#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)


# Define label from columns
def get_label(row):
    return row[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax()

df_train['label_str'] = df_train.apply(get_label, axis=1)
label_map = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}
df_train['label'] = df_train['label_str'].map(label_map)

# Görüntü isimlerine '.jpg' ekle
df_train['image'] = df_train['image_id'] + '.jpg'
df_test['image'] = df_test['image_id'] + '.jpg'

# Test setinde etiket yoksa (varsayılan 0 veriyoruz)
if 'label' not in df_test.columns:
    df_test['label'] = 0


# Stratified split to keep label distribution consistent
train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train['label'], random_state=42)


# Data augmentation and normalization for training and validation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class PlantDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


train_dataset = PlantDataset(train_df, img_dir, transform=train_transforms)
val_dataset = PlantDataset(val_df, img_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Custom ResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Choose model and define loss and optimizer
use_custom_model = True
if use_custom_model:
    model = CustomResNet(num_classes=4).to(device)
else:
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 4)
    )
    model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")


# Validation evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Test Time Augmentation (TTA) transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
]


class TestPlantDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms  # List of transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Apply all TTA transforms and stack
        augmented_images = [t(image) for t in self.transforms]
        stacked_images = torch.stack(augmented_images, dim=0)

        return stacked_images, img_name


# DataLoader
test_dataset = TestPlantDataset(df_test, img_dir, transforms=tta_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # batch_size=1 is important for TTA


model.eval()
all_probs = []
all_image_names = []

with torch.no_grad():
    for augmented_images, image_names in tqdm(test_loader):
        image_name = image_names[0]  # batch_size=1

        tta_preds = []
        for img_tensor in augmented_images[0]:
            img_tensor = img_tensor.unsqueeze(0).to(device)
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            tta_preds.append(prob.squeeze(0).cpu())

        mean_prob = torch.mean(torch.stack(tta_preds), dim=0)
        all_probs.append(mean_prob)
        all_image_names.append(image_name)


probs_np = torch.stack(all_probs).numpy()
image_ids = [os.path.splitext(name)[0] for name in all_image_names]

submission = pd.DataFrame(probs_np, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
submission.insert(0, 'image_id', image_ids)

submission.to_csv('submission_tta.csv', index=False)

print("Submission file 'submission_tta.csv' successfully created!")



print(submission.dtypes)

