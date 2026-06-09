# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage.io import imread
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score
import time
import copy
from tqdm import tqdm_notebook as tqdm
from torchmetrics.classification import BinaryAUROC
import contextlib
from IPython.display import clear_output 


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


path='/kaggle/input/histopathologic-cancer-detection/train/'
annotation_file='/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
test_path='/kaggle/input/histopathologic-cancer-detection/test/'
firstTrainImage = os.listdir(path)[0].split('.')[0]
print(firstTrainImage)


train_data =pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
sub = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
train_data[train_data['id'] == firstTrainImage]


# Check how balanced are the label
plt.bar(['No Cancer', 'Cancer'], train_data.label.value_counts().values, color=['blue', 'white'], edgecolor='black')
plt.show()
len(train_data)


cancer = np.random.choice(train_data[train_data.label==1].id, size=25, replace=False)
no_cancer = np.random.choice(train_data[train_data.label==0].id, size=25, replace=False)

fig, ax = plt.subplots(5, 10, figsize=(20, 10))
fig.suptitle("Cancer (Red Border) vs No Cancer (Green Border)", fontsize=20)

for i in range(5):
    for j in range(10):
        idx = i * 5 + (j % 5)
        img_id = cancer[idx] if j < 5 else no_cancer[idx]
        image = plt.imread(path + img_id + ".tif")

        ax[i, j].imshow(image)
        ax[i, j].tick_params(labelbottom=False, labelleft=False)
        for spine in ax[i, j].spines.values():
            spine.set_edgecolor('red' if j < 5 else 'green')
            spine.set_linewidth(3)



train_df, val_df = train_test_split(train_data, test_size=0.1, stratify=train_data['label'], random_state=42)

# 2. Plot class distributions
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Training set
train_counts = train_df['label'].value_counts().sort_index()
ax[0].bar(['No Cancer', 'Cancer'], train_counts, color=['blue', 'white'], edgecolor='black')
ax[0].set_title('Training Set')

# Validation set
val_counts = val_df['label'].value_counts().sort_index()
ax[1].bar(['No Cancer', 'Cancer'], val_counts, color=['blue', 'white'], edgecolor='black')
ax[1].set_title('Validation Set')

plt.tight_layout()


class CancerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading cancer image data. To be used with DataLoader()
    
    Each sample consists of an image loaded from disk and its associated label.
    """
    
    def __init__(self, dataframe, image_dir='./', transform=None):
        """
        Args:
            dataframe (pd.DataFrame): A DataFrame with columns [image_id, label].
            image_dir (str): Directory where image files are stored.
            transform (callable, optional): Optional transform to apply to each image.
        """
        self.image_labels = dataframe.values         # Convert to NumPy array for indexing
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """Return total number of samples."""
        return len(self.image_labels)

    def __getitem__(self, index):
        """Load and return a single (image, label) pair."""
        image_id, label = self.image_labels[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.tif")
        
        image = imread(image_path)  # Load image from disk
        
        if self.transform:
            image = self.transform(image)

        return image, label

# funciton to get sample of the data formodel selection and hyperparam tuning
def get_sample_loader(df, image_dir, fraction=0.1, batch_size=32, transform=None):
    df_small, _ = train_test_split(df, test_size=1-fraction, stratify=df['label'], random_state=42)
    ds = CancerDataset(df_small, image_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)


class BasicCNN(nn.Module):
    def __init__(self, base_channels=8, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*4, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm
import torch

def train_one_epoch(model, dataloader, device, criterion, optimizer):
    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device):
    model.eval()
    metric = BinaryAUROC().to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            metric.update(probs, labels)
    return metric.compute().item()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL image (required torchvision transforms)    
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor and scale pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  
                         std=[0.5, 0.5, 0.5])    # Normalize using midpoints. It takes too long to get the real mean and std 
])

# Load metadata
train_data = pd.read_csv("/kaggle/input/histopathologic-cancer-detection/train_labels.csv")
image_dir = "/kaggle/input/histopathologic-cancer-detection/train"

# Create data subset
train_loader = get_sample_loader(train_data, image_dir, fraction=0.1, transform=transform)
val_loader   = get_sample_loader(train_data, image_dir, fraction=0.05, transform=transform)

# List of configs to try
configs = [
    {"channels": 4,  "dropout": 0.3, "lr": 1e-2},
    {"channels": 8, "dropout": 0.25, "lr": 1e-3},
    {"channels": 16,  "dropout": 0.2, "lr": 1e-4},
]

results = []

for cfg in configs:
    print(f"Testing config: {cfg}")
    model = BasicCNN(base_channels=cfg["channels"], dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.BCEWithLogitsLoss()

    train_one_epoch(model, train_loader, device, criterion, optimizer)
    val_auc = evaluate(model, val_loader, device)

    results.append((cfg, val_auc))
    print(f"AUC: {val_auc:.4f}\n")



records = []
for cfg, auc in results:
    row = cfg.copy()
    row["val_auc"] = auc
    records.append(row)

df = pd.DataFrame.from_records(records)
df.index.name = "run_id"


# Bar chart 
plt.figure(figsize=(6, 4))
plt.bar(df.index.astype(str), df["val_auc"])
plt.xlabel("Run ID")
plt.ylabel("Validation AUC")
plt.title("Model Comparison (Rescaled Y-axis)")
plt.ylim(0.80, 0.87)  # Tight Y-axis to amplify visual differences
plt.tight_layout()
plt.show()



best_cfg = max(results, key=lambda x: x[1])
print(f"Best config: {best_cfg[0]}, AUC: {best_cfg[1]:.4f}")


train_dataset = CancerDataset(dataframe=train_df, image_dir=path, transform=transform)
val_dataset = CancerDataset(dataframe=val_df, image_dir=path, transform=transform)
test_dataset = CancerDataset(dataframe=sub, image_dir=test_path, transform=transform)


# Hyperparameter and system setings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size= 256
shuffle=False
num_workers= os.cpu_count()
pin_memory=True
persistent_workers=True
learning_rate = best_cfg[0]['lr']
dropout = best_cfg[0]['dropout']
channels = best_cfg[0]['channels']

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,            
    pin_memory=pin_memory,
    persistent_workers=persistent_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,            
    pin_memory=pin_memory,
    persistent_workers=persistent_workers
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,            
    pin_memory=pin_memory,
    persistent_workers=persistent_workers
)


model = BasicCNN(base_channels=channels, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

print(model)


NUM_EPOCHS = 20
is_cuda = device.type == "cuda"                 
best_val_auc = 0.0
BEST_MODEL_PATH = "best_model.pt"

#  Metric history containers
train_loss_hist = []
val_loss_hist   = []
train_auc_hist  = []
val_auc_hist    = []

# Define AUC metrics and move them to the correct device
train_auc_metric = BinaryAUROC().to(device)
val_auc_metric   = BinaryAUROC().to(device)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_auc_metric.reset()
    running_train_loss = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):

        # Move data to GPU / CPU
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()

        #  forward / backward (mixed precision if CUDA) 
        with torch.cuda.amp.autocast(enabled=is_cuda):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if scaler.is_enabled():       
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                          # CPU fallback
            loss.backward()
            optimizer.step()


        running_train_loss += loss.item()
        train_auc_metric.update(outputs, labels)

    # Epoch-level metrics
    avg_train_loss = running_train_loss / len(train_loader)
    avg_train_auc  = train_auc_metric.compute().item()


    # Validation
    model.eval()
    val_auc_metric.reset()
    running_val_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, total=len(val_loader))):

            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=is_cuda):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            running_val_loss += loss.item()
            val_auc_metric.update(outputs, labels)

    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_auc  = val_auc_metric.compute().item()
    
    # Save best model (based on validation AUC)
    if avg_val_auc > best_val_auc:
        best_val_auc = avg_val_auc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved new best model at epoch {epoch+1} (Val AUC: {avg_val_auc:.4f})")
    
    # Append to history lists
    train_loss_hist.append(avg_train_loss)
    val_loss_hist.append(avg_val_loss)
    train_auc_hist.append(avg_train_auc)
    val_auc_hist.append(avg_val_auc)    
    
    # Epoch summary
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Train AUC : {avg_train_auc :.4f} | Val AUC : {avg_val_auc :.4f}")
    print("-" * 60)

    # Reset resume pointer for the next epoch
    start_batch = -1



plt.figure(figsize=(20, 5))
plt.plot(range(NUM_EPOCHS), train_loss_hist, label="Train Loss")
plt.plot(range(NUM_EPOCHS), val_loss_hist, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Change Over Epochs")
plt.legend()
plt.show()


plt.figure(figsize=(20,5))
plt.plot(range(NUM_EPOCHS),train_auc_hist, label="train")
plt.plot(range(NUM_EPOCHS),val_auc_hist, label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epoch")
plt.legend()


# Load model to device and set to eval mode
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

predictions = []

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
        images = images.to(device, non_blocking=True)

        outputs = model(images)                   # shape: (B, 1)
        probs = torch.sigmoid(outputs)            # convert logits to probabilities
        probs = probs.squeeze(1).cpu().numpy()    # shape: (B,) on CPU

        predictions.extend(probs)                 # add to final list



sub['label'] = predictions
sub.to_csv('submission.csv', index=False)
sub.info()

