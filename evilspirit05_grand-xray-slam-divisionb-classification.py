import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm
from tabulate import tabulate
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import warnings
import os
from PIL import ImageFile
import torch.nn.functional as F

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# Prevent PIL image corruption issues
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Also suppress Python environment warnings
os.environ["PYTHONWARNINGS"] = "ignore"


from sklearn.metrics import roc_auc_score, accuracy_score
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


labels = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_df = pd.read_csv("/kaggle/input/grand-xray-slam-division-b/train2.csv")
image_path="/kaggle/input/grand-xray-slam-division-b/train2"


train_df.head()


# Total subset size
total_samples = 5000

# Ensure each label has at least one sample
subset_list = []

for label in labels:
    label_rows = train_df[train_df[label] == 1]
    if len(label_rows) > 0:
        subset_list.append(label_rows.sample(1, random_state=42))

# Concatenate initial guaranteed samples
subset_df = pd.concat(subset_list)

# Remaining samples to reach 1000
remaining_samples = total_samples - len(subset_df)

# Sample remaining rows randomly from the original df, avoiding duplicates
remaining_df = train_df.drop(subset_df.index)
subset_df = pd.concat([subset_df, remaining_df.sample(remaining_samples, random_state=42)])

# Shuffle final subset
subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Subset shape:", subset_df.shape)
print("Label distribution in subset:\n", subset_df[labels].sum())


subset_df["Cardiomegaly"].value_counts()


subset_df.head()


train_df, val_df = train_test_split(subset_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

columns_to_drop = ['Patient_ID', 'Study', 'Sex', 'Age', 'ViewCategory', 'ViewPosition']
train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
val_df = val_df.drop(columns=columns_to_drop, errors='ignore')


from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, labels, transform=None):
        self.data = df
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['Image_name'])
        try:
            image = Image.open(img_name).convert('L')  # Grayscale
        except:
            image = Image.new('L', (224, 224), color=0)  # fallback blank image
        
        # Select only label columns
        labels = self.data.iloc[idx][self.labels].values.astype('float32')
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels



# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((10, 10)),  # Resize to 30x30
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
val_transform = transforms.Compose([
    transforms.Resize((10, 10)),  # Resize to 30x30
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])



train_dataset = ChestXRayDataset(
    df=train_df,
    image_dir=image_path,
    labels=labels,  # Pass labels
    transform=train_transform
)
val_dataset = ChestXRayDataset(
    df=val_df,
    image_dir=image_path,
    labels=labels,  # Pass labels
    transform=val_transform
)

batch_size=8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


for images, labels in train_loader:
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    break


class SmallCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*5*5, 128)  # because 10x10 -> after pool 5x5
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SmallCNN(num_classes=14).to(device)

summary(model,input_size=(1,10,10))


def compute_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).cpu().numpy()
    labels = labels.cpu().numpy()
    auc_scores = []
    acc_scores = []
    
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:  # Only compute AUC if positive samples exist
            auc = roc_auc_score(labels[:, i], probs[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)
        
        preds = (probs[:, i] > 0.5).astype(int)
        acc = accuracy_score(labels[:, i], preds)
        acc_scores.append(acc)
    
    mean_auc = np.nanmean(auc_scores)
    mean_acc = np.mean(acc_scores)
    return mean_auc, mean_acc


# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

num_epochs = 100
metrics_table = []

# Early stopping parameters
early_stopping_patience = 5
best_val_auc = 0
epochs_no_improve = 0

for epoch in range(num_epochs):
    # ---------- Training ----------
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.append(outputs.detach())
        train_labels.append(labels)
        train_bar.set_postfix({'loss': loss.item()})

    train_loss /= len(train_loader)
    train_preds = torch.cat(train_preds)
    train_labels = torch.cat(train_labels)
    train_auc, train_acc = compute_metrics(train_preds, train_labels)

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.append(outputs)
            val_labels.append(labels)
            val_bar.set_postfix({'loss': loss.item()})

    val_loss /= len(val_loader)
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_auc, val_acc = compute_metrics(val_preds, val_labels)

    # ---------- Scheduler and Early Stopping ----------
    scheduler.step(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"âœ… Model improved and saved (AUC={val_auc:.4f})")
    else:
        epochs_no_improve += 1
        print(f"âš ï¸� No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= early_stopping_patience:
        print(f"â�¹ Early stopping triggered after {epoch+1} epochs (best AUC={best_val_auc:.4f}).")
        break

    # ---------- Logging ----------
    metrics_table.append([epoch + 1, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc])
    print("\nEpoch Summary:")
    print(tabulate(metrics_table[-1:],headers=["Epoch", "Train Loss", "Train AUC", "Train Acc", "Val Loss", "Val AUC", "Val Acc"],
        tablefmt="grid",floatfmt=".4f"))

print("\nğŸ�� Training complete. Best validation AUC:", round(best_val_auc, 4))



class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, labels=None, transform=None):
        self.data = df
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['Image_name'])
        try:
            image = Image.open(img_name).convert('L')
        except:
            image = Image.new('L', (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        # âœ… Handle test mode (no labels)
        if self.labels is not None and all(lbl in self.data.columns for lbl in self.labels):
            labels = self.data.iloc[idx][self.labels].values.astype('float32')
            return image, labels
        else:
            return image, torch.zeros(14, dtype=torch.float32)  # dummy labels



# Test dataset
test_df = pd.read_csv("/kaggle/input/grand-xray-slam-division-b/sample_submission_2.csv")
test_df = test_df.drop(columns=columns_to_drop, errors='ignore')  # Drop non-label columns
test_dataset = ChestXRayDataset(
    df=test_df,
    image_dir='/kaggle/input/grand-xray-slam-division-b/test2/',
    labels=labels,
    transform=val_transform
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


# Generate predictions
model.eval()
predictions = []
image_names = test_df['Image_name'].values
test_bar = tqdm(test_loader, desc="Generating Predictions")
with torch.no_grad():
    for images, _ in test_bar:
        images = images.to(device)
        outputs = torch.sigmoid(model(images))
        predictions.append(outputs.cpu().numpy())
predictions = np.concatenate(predictions)

labels = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]

submission_df = pd.DataFrame(predictions, columns=labels)
submission_df.insert(0, 'Image_name', image_names)
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")




