import os
import gc
import copy

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.models as models

import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tqdm import tqdm


seed = 42
batch_size = 64
epochs = 4
lr = 5e-5
gamma = 0.7
IMG_SIZE = 224


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used :", device)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(seed)


# Paths to the dataset
folder_path = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv_path = os.path.join(folder_path, 'train.csv')
test_csv_path  = os.path.join(folder_path, 'test.csv')


# Reading the training CSV file
df_train = pd.read_csv(train_csv_path)

# Reading the testing CSV file
df_test = pd.read_csv(os.path.join(folder_path, 'test.csv'))

# Adding the full path to the file_name instead of just "trainORtest_data/xxx.jpg"
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(folder_path, x))
df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(folder_path, x))

all_image_paths = df_train['file_name'].values
all_labels = df_train['label'].values

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.05,        
    random_state=seed,
    shuffle=False
)

print(f"Train Data size: {len(train_paths)}")
print(f"Validation Data size: {len(val_paths)}")


train_transforms = T.Compose([
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

test_transforms = T.Compose([
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class CustomDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img

class TestDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path) 


train_data = CustomDataset(train_paths, train_labels, transform=train_transforms)
val_data   = CustomDataset(val_paths,   val_labels,   transform=val_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(dataset=val_data,   batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")


# Load pre-trained Regnet
model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

block4_section1 = getattr(model.trunk_output.block4, 'block4-1')
block4_section1.requires_grad_(True)

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2) 
)


# Define loss, optimizer, and scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_f1s = []


step = 1000

def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, num_epochs=10, save_path="best_model.pth"):
    model.to(device)

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())  # To store the best weights
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    curr_step = 0
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Compute training metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            curr_step += 1
            if (curr_step + 1) % step == 0:
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_labels = []
                all_preds = []
        
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
        
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
        
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
        
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
        
                        # Store labels and predictions for F1 score calculation
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())
        
                val_loss /= len(val_loader)
                val_f1 = f1_score(all_labels, all_preds, average="weighted")
        
                # Save the best model weights based on validation F1 score
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved with F1 score: {best_f1:.4f}")


                train_loss /= 200
                
                # Log metrics
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

                # Print results after 1k steps
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

                # reset
                train_loss = 0.0
                train_total = 0
                train_correct = 0
                
                # Step the learning rate scheduler
                scheduler.step()

    print("Training complete.")
    print(f"Best F1 Score: {best_f1:.4f}")
    return best_model_wts, history


best_model_wts, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    num_epochs=epochs,
    save_path="/kaggle/working/RegNetCNNCPU.pth"
)


epochs = range(1, len(history["train_loss"]) + 1)
# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Plot validation F1 score
plt.subplot(1, 2, 2)
plt.plot(epochs, history["val_f1"], label="Val F1 Score", marker="o", color="green")
plt.title("Validation F1 Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# Load the test CSV for predictions
test_csv_path = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
test_df = pd.read_csv(test_csv_path)

# Define the base directory where images are stored
test_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"

# Update image paths in the test dataframe
test_df['img_path'] = test_df['id'].apply(lambda x: os.path.join(test_data_dir, x))


test_dataset = TestDataset(df_test['id'].values, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
        outputs = model(data)
        
        # predict
        preds = outputs.argmax(dim=1)  # shape [batch_size]
        
        predictions.extend(preds.cpu().numpy())
        image_names.extend([f"test_data_v2/{name}" for name in names])


submission_df = pd.DataFrame({
    'id': image_names,
    'label': predictions
})


sns.barplot(x=submission_df['label'].value_counts().index, y=submission_df['label'].value_counts().values)

plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')

plt.show()


submission_df['label'].value_counts()


# Save to CSV for submission
submission_csv_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")

