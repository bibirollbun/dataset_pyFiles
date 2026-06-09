#pip install tqdm


# Import libraries
import zipfile
import os
import shutil
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
import time
import os, pandas as pd, zipfile, pathlib
import torch, random, numpy as np
from pathlib import Path





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)




data_dir = Path('/kaggle/input/ifood-2019-fgvc6')
working_dir = Path('/kaggle/working')
TRAIN_CSV = data_dir / 'train_labels.csv'

# Extract and flatten function
def extract_and_flatten(zip_filename, folder_name, internal_folder_name):
    zip_path = data_dir / zip_filename
    extract_path = working_dir  / (folder_name + "_temp")
    final_path = working_dir / folder_name

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    nested = extract_path / internal_folder_name
    final_path.mkdir(exist_ok=True)
    for fname in os.listdir(nested):
        shutil.move(str(nested / fname), str(final_path / fname))

    shutil.rmtree(extract_path)

extract_and_flatten("train_set.zip", "train", "train_set")
extract_and_flatten("val_set.zip", "val", "val_set")
extract_and_flatten("test_set.zip", "test", "test_set")



# Ensure that we loaded the dataset correctly
print("Train images:", len(os.listdir(os.path.join(working_dir, "train"))))
print("Validation images:", len(os.listdir(os.path.join(working_dir, "val"))))
print("Test images:", len(os.listdir(os.path.join(working_dir, "test"))))



#labels_df = pd.read_csv(labels_csv)



# Transforms (updated with dataset-specific mean and std)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6388, 0.5444, 0.4448], std=[0.2229, 0.2414, 0.2638])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6388, 0.5444, 0.4448], std=[0.2229, 0.2414, 0.2638])
])


# Dataset
class FoodDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform, class_to_idx):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['img_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['label']]

        image = self.transform(image)
        return image, label


train_csv = data_dir / 'train_labels.csv'
labels_df = pd.read_csv(train_csv)




# Labels and splits
labels_df = pd.read_csv(train_csv)
classes = sorted(labels_df['label'].unique())
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}


# Stratified split before creating Datasets
labels = labels_df['label'].map(class_to_idx)

train_idx, val_idx = train_test_split(
    np.arange(len(labels_df)), test_size=0.2,
    stratify=labels, random_state=42
)

train_df = labels_df.iloc[train_idx].reset_index(drop=True)
val_df = labels_df.iloc[val_idx].reset_index(drop=True)

# Create Datasets with separate transforms
train_dataset = FoodDataset(working_dir / 'train', train_df, transform=train_transform, class_to_idx=class_to_idx)
val_dataset = FoodDataset(working_dir / 'train', val_df, transform=val_transform, class_to_idx=class_to_idx)



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)




test_csv = data_dir / 'val_labels.csv'
test_labels_df = pd.read_csv(test_csv)
test_labels_df = test_labels_df.reset_index(drop=True)

# Dataset
test_dataset = FoodDataset(working_dir / 'val', test_labels_df, transform=val_transform, class_to_idx=class_to_idx)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)



import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)

class CustomCNN(nn.Module):
    def __init__(self, block, layers, num_classes=251, base_width=32):
        super(CustomCNN, self).__init__()
        self.in_channels = base_width

        self.conv1 = nn.Conv2d(3, base_width, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.act = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def build_custom_cnn(num_classes=251):
    return CustomCNN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=32)




model = build_custom_cnn(num_classes=251)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)




# --- EarlyStopping class ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, save_path='best_model10.pt'):
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            self.counter = 0
            if self.verbose:
                print(f"Validation score improved. Saving model to {self.save_path}")
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation score for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True



lrs = [1e-4, 1e-3]
weight_decays = [0.01, 0.1]


search_space = {
    'lr': [1e-4, 1e-3],
    'weight_decay': [0.01],
}

best_val_acc = 0.0
best_config = None
best_model_state = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 45
patience = 5

for lr in search_space['lr']:
    for wd in search_space['weight_decay']:
        print(f"\n=== Training with lr={lr}, weight_decay={wd} ===")

        # Initialize model + optimizer + scheduler
        model = build_custom_cnn(num_classes=251).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        early_stopper = EarlyStopping(patience=patience, verbose=False)

        # Metric tracking
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epoch_times = []
        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            train_loss, correct, total = 0.0, 0, 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_postfix(loss=loss.item())

            scheduler.step()

            avg_train_loss = train_loss / total
            train_accuracy = 100 * correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / val_total
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Epoch timing
            epoch_duration = time.time() - epoch_start_time
            epoch_times.append(epoch_duration)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}% | "
                  f"Time: {epoch_duration:.2f}s")

            early_stopper(val_accuracy, model)
            if early_stopper.early_stop:
                print("→ Early stopping triggered.")
                break

        total_training_time = time.time() - total_start_time
        print(f"Total training time for config (lr={lr}, wd={wd}): {total_training_time:.2f}s")

        # Save best
        if early_stopper.best_score > best_val_acc:
            best_val_acc = early_stopper.best_score
            best_config = {'lr': lr, 'weight_decay': wd}
            best_model_state = model.state_dict()

        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

print(f"\n Best Val Acc: {best_val_acc:.2f}% with config: {best_config}")
torch.save(best_model_state, "best_model.pt")
print("Saved best model to best_model.pt")





model.load_state_dict(torch.load('best_model6.pt'))
model.eval()
model.to(device)



import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, data_loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



class_names = [f"Class {i}" for i in sorted(labels_df['label'].unique())]



labels_df.head()



evaluate_model(model, test_loader, criterion, device, class_names)





import matplotlib.pyplot as plt

# Plot losses
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()



val_errors = [100 - acc for acc in val_accuracies]

plt.plot(val_errors, label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.title('Validation Error over Epochs')
plt.legend()
plt.show()



total_training_time = time.time() - total_start_time
print(f"\nTotal Training Time: {total_training_time / 60:.2f} minutes")
print(f"Average Epoch Time: {np.mean(epoch_times):.2f} seconds")
print(f"Fastest Epoch: {np.min(epoch_times):.2f} s | Slowest Epoch: {np.max(epoch_times):.2f} s")


import matplotlib.pyplot as plt

plt.plot(range(1, len(epoch_times)+1), epoch_times, marker='o')
plt.title("Epoch Duration Over Time")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.show()


