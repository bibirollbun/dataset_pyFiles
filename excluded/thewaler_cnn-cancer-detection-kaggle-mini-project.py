import numpy as np
import pandas as pd
import os, pathlib
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path('/kaggle/input/histopathologic-cancer-detection')
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
TEST_DIR = os.path.join(DATA_DIR, 'test/')
df = pd.read_csv(DATA_DIR/'train_labels.csv')

# Getting a basic sample of the data
print(df.sample(3))

# Checking for balance
print(df['label'].value_counts(normalize=True))

# Example for label 0
sample_ids_label_0 = df[df.label==0].sample(3).id.values
fig, axes = plt.subplots(1,3)
for ax, img_id in zip(axes.flat, sample_ids_label_0):
    img = plt.imread(TRAIN_DIR+f'{img_id}.tif')
    ax.imshow(img)
plt.suptitle(f'Examples for Label 0', y=0.75)

# Example for label 1
sample_ids_label_1 = df[df.label==1].sample(3).id.values
fig, axes = plt.subplots(1,3)
for ax, img_id in zip(axes.flat, sample_ids_label_1):
    img = plt.imread(TRAIN_DIR+f'{img_id}.tif')
    ax.imshow(img)
plt.suptitle(f'Examples for Label 1', y=0.75)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
import time

df['filename'] = df['id'] + '.tif'
df['label_str'] = df['label'].astype(str)

train_subset_df, validation_subset_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_subset_df)}")
print(f"Validation set size: {len(validation_subset_df)}")

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 96, 96
BATCH_SIZE = 512

class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['id'] + '.tif')
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        if self.is_test:
            return image, self.dataframe.iloc[idx]['id']
        else:
            label = self.dataframe.iloc[idx]['label']
            return image, torch.tensor(label, dtype=torch.float32)

# Include augmentation for training
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Only normalization for validation
val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create Datasets
train_dataset = HistopathologyDataset(train_subset_df, TRAIN_DIR, transform=train_transforms)
validation_dataset = HistopathologyDataset(validation_subset_df, TRAIN_DIR, transform=val_test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print("\nDataLoaders created.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Input: (batch_size, 3, 96, 96)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x) # Output probability
        return x

# Instantiate the model and move to device
model_basic_pt = BasicCNN().to(device)
print("\nBasic Model:")
print(model_basic_pt)


criterion = nn.BCELoss()
history_storage_pt = {}

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=30):
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f"Start of epoch {epoch}")
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        print(f"Epoch {epoch} finished training loop, now calculating stats")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        train_preds_flat = np.array(train_preds).flatten()
        train_targets_flat = np.array(train_targets).flatten()
        
        current_train_auc = roc_auc_score(train_targets_flat, train_preds_flat)
        train_aucs.append(current_train_auc)
        
        train_predicted_classes = (train_preds_flat > 0.5).astype(int)
        current_train_acc = np.mean(train_predicted_classes == train_targets_flat)
        train_accs.append(current_train_acc)

        print(f"Starting eval for epoch {epoch}")
        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(validation_loader.dataset)
        val_losses.append(epoch_val_loss)

        val_preds_flat = np.array(val_preds).flatten()
        val_targets_flat = np.array(val_targets).flatten()
        
        current_val_auc = roc_auc_score(val_targets_flat, val_preds_flat)
        val_aucs.append(current_val_auc)

        val_predicted_classes = (val_preds_flat > 0.5).astype(int)
        current_val_acc = np.mean(val_predicted_classes == val_targets_flat)
        val_accs.append(current_val_acc)

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {epoch_loss:.4f} | Train AUC: {current_train_auc:.4f} | Train Acc: {current_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val AUC: {current_val_auc:.4f} | Val Acc: {current_val_acc:.4f}")

    history = {
        'loss': train_losses, 'val_loss': val_losses,
        'auc': train_aucs, 'val_auc': val_aucs,
        'accuracy': train_accs, 'val_accuracy': val_accs
    }
    print(f"Training complete. Final Validation AUC: {current_val_auc:.4f}")
    return model, history

EPOCHS_PT = 10


print("\n--- Training Basic Model ---")
model_basic_pt = BasicCNN().to(device)
optimizer_basic = optim.Adam(model_basic_pt.parameters(), lr=0.001)


model_basic_pt, history_basic_pt = train_model(
    model_basic_pt, train_loader, validation_loader, criterion, optimizer_basic, num_epochs=EPOCHS_PT
)
history_storage_pt['basic_pt'] = history_basic_pt

# Final evaluation on validation set with the best model
model_basic_pt.eval()
final_val_preds, final_val_targets = [], []
with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        outputs = model_basic_pt(inputs)
        final_val_preds.extend(outputs.cpu().numpy())
        final_val_targets.extend(labels.cpu().numpy())

final_val_auc_basic = roc_auc_score(np.array(final_val_targets).flatten(), np.array(final_val_preds).flatten())
print(f"Basic Model - Final Validation AUC: {final_val_auc_basic:.4f}")


def plot_training_history_pt(history_dict, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.suptitle(f'Training History for {model_name}', fontsize=16)

    # Plot Loss
    axes[0].plot(history_dict.get('loss', []), label='Training Loss')
    axes[0].plot(history_dict.get('val_loss', []), label='Validation Loss')
    axes[0].set_title('Loss vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot AUC
    axes[1].plot(history_dict.get('auc', []), label='Training AUC')
    axes[1].plot(history_dict.get('val_auc', []), label='Validation AUC')
    axes[1].set_title('AUC vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Accuracy
    axes[2].plot(history_dict.get('accuracy', []), label='Training Accuracy')
    axes[2].plot(history_dict.get('val_accuracy', []), label='Validation Accuracy')
    axes[2].set_title('Accuracy vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_training_history_pt(history_storage_pt['basic_pt'], 'Basic Model')


class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(EnhancedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu5(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

model_enhanced_pt = EnhancedCNN().to(device)
print("\nEnhanced Model:")
print(model_enhanced_pt)


optimizer_enhanced = optim.Adam(model_enhanced_pt.parameters(), lr=0.001)

model_enhanced_pt, history_enhanced_pt = train_model(
    model_enhanced_pt, train_loader, validation_loader, criterion, optimizer_enhanced, num_epochs=EPOCHS_PT
)
history_storage_pt['enhanced_pt'] = history_enhanced_pt

model_enhanced_pt.eval()
final_val_preds_enh, final_val_targets_enh = [], []
with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        outputs = model_enhanced_pt(inputs)
        final_val_preds_enh.extend(outputs.cpu().numpy())
        final_val_targets_enh.extend(labels.cpu().numpy())
        
final_val_auc_enhanced = roc_auc_score(np.array(final_val_targets_enh).flatten(), np.array(final_val_preds_enh).flatten())
print(f"Enhanced Model - Final Validation AUC: {final_val_auc_enhanced:.4f}")

plot_training_history_pt(history_storage_pt['enhanced_pt'], 'Enhanced Model')


print("\n--- Generating Predictions for Test Set ---")

model_enhanced_pt.eval()

df = pd.read_csv(DATA_DIR/'sample_submission.csv')

test_df = pd.DataFrame({'id': df['id']})
test_df['label'] = -1


test_dataset_pt = HistopathologyDataset(
    dataframe=test_df,
    image_dir=TEST_DIR,
    transform=val_test_transforms,
    is_test=True
)

test_loader_pt = DataLoader(
    test_dataset_pt,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Predicting on {len(test_dataset_pt)} test images using enhanced model...")
all_predictions_pt = []
all_ids_pt = []

with torch.no_grad():
    for images, ids_batch in test_loader_pt:
        images = images.to(device)
        outputs = model_enhanced_pt(images)
        all_predictions_pt.extend(outputs.cpu().numpy().flatten())
        all_ids_pt.extend(ids_batch)

pred_dict = {img_id: pred for img_id, pred in zip(all_ids_pt, all_predictions_pt)}

final_ordered_predictions = [pred_dict.get(img_id, 0.5) for img_id in df['id']] # Default to 0.5 if an ID was missed

submission_df_pt = pd.DataFrame({
    'id': df['id'],
    'label': final_ordered_predictions
})

submission_path_pt = 'submission.csv'
submission_df_pt.to_csv(submission_path_pt, index=False)
print(f"\nSubmission file created: {submission_path_pt}")
print(submission_df_pt.head())




