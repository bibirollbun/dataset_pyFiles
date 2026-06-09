# === 1) Imports & Utilities ===
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Function: load DICOM and convert to PIL image
def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    img = (img * 255).astype(np.uint8)
    # create PIL and convert to 3-channel RGB
    return Image.fromarray(img).convert('RGB')


# in your transforms definitions:

common_norm = transforms.Normalize(mean=[0.485,0.456,0.406],
                                  std =[0.229,0.224,0.225])

axial_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    common_norm
])
sag_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomAffine(15, translate=(0.1,0.1)),
    transforms.ToTensor(),
    common_norm
])




# === 2) Load CSVs & Generate Image Paths ===
# Update train_path to your data directory
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
train_df      = pd.read_csv(os.path.join(train_path, 'train.csv'))
label_df      = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc_df = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))

# ==== Rebuild paths_df correctly ====
def generate_image_paths(desc_df, image_root):
    records = []
    for _, row in desc_df.iterrows():
        study_id           = row['study_id']
        series_id          = row['series_id']
        series_description = row['series_description']  # use the row’s own description
        series_dir = os.path.join(image_root, str(study_id), str(series_id))
        if not os.path.isdir(series_dir):
            continue
        for fname in os.listdir(series_dir):
            # (optionally filter for .dcm)
            records.append({
                'study_id':            study_id,
                'series_id':           series_id,
                'series_description':  series_description,
                'image_path':          os.path.join(series_dir, fname)
            })
    return pd.DataFrame(records)

# Usage: point `train_images_dir` at wherever your DICOM folders live
train_images_dir = os.path.join(train_path, 'train_images')
paths_df = generate_image_paths(train_desc_df, train_images_dir)

print("✔ paths_df:", paths_df.shape)
print(paths_df['series_description'].value_counts())




# ==== Helper: unpivot the original train.csv into (study,condition,level,severity) rows ====
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    # skip the non‐severity columns
    for col, val in row.items():
        if col in ['study_id','series_id','instance_number','x','y','series_description']:
            continue
        parts = col.split('_')
        # reconstruct the condition name (all but last two tokens)
        condition = ' '.join([w.capitalize() for w in parts[:-2]])
        # reconstruct the level as e.g. 'L1/L2'
        level = parts[-2].upper() + '/' + parts[-1].upper()
        data['study_id'].append(row['study_id'])
        data['condition'].append(condition)
        data['level'].append(level)
        data['severity'].append(val)
    return pd.DataFrame(data)



# ==== DEBUGGED MERGE SECTION ====

# 1) Build the flat label table
new_train_df = pd.concat([reshape_row(row) for _, row in train_df.iterrows()],
                         ignore_index=True)
print("➜ new_train_df shape:",      new_train_df.shape)
print(new_train_df.head(3))

# 2) Normalize text keys so merges won’t fail on caps/slashes/spaces
new_train_df['condition'] = (
    new_train_df['condition']
    .str.lower()
    .str.replace(' ', '_')
)
new_train_df['level'] = (
    new_train_df['level']
    .str.lower()
    .str.replace('/', '_')
)
label_df['condition'] = (
    label_df['condition']
    .str.lower()
    .str.replace(' ', '_')
)
label_df['level'] = (
    label_df['level']
    .str.lower()
    .str.replace('/', '_')
)

# 3) Merge1: labels → coordinates
merged1 = pd.merge(
    new_train_df,
    label_df,
    on=['study_id', 'condition', 'level'],
    how='inner'
)
print("➜ after merge1:", merged1.shape)
print(merged1[['study_id','condition','level']].drop_duplicates().head(3))

# 4) Confirm `series_id` alignment
print("➜ merged1 columns:", merged1.columns.tolist())
print("➜ paths_df columns:", paths_df.columns.tolist())
print("➜ dtype(series_id):", merged1['series_id'].dtype,
      paths_df['series_id'].dtype)

# 5) Merge2: bring in series_description & image_path
merged2 = merged1.merge(
    paths_df[['study_id','series_id','series_description','image_path']],
    on=['study_id','series_id'],
    how='inner'
)
print("➜ after merge2:", merged2.shape)
print("➜ unique descriptions:", merged2['series_description'].unique())

# 6) Filter to only the two T2 views you want
merged2 = merged2[merged2['series_description']
                  .isin(['Axial T2', 'Sagittal T2/STIR'])].copy()
print("➜ after filtering modalities:", merged2.shape)

# 7) Normalize severity text, then map to ints
merged2['severity'] = (
    merged2['severity']
    .str.lower()
    .str.replace('/', '_')
)
severity_map = {'normal_mild':0, 'moderate':1, 'severe':2}
merged2['severity'] = merged2['severity'].map(severity_map)

# 8) Drop any rows that failed to map or lost their path
merged2 = merged2.dropna(subset=['image_path','severity']).reset_index(drop=True)
print("➜ final merged2 shape:", merged2.shape)

# 9) Ready for splitting
train_data = merged2.copy()




# ==== 4) Dataset, Transforms & DataLoader (classification-only) ====

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 4a) Build per-study map of the two T2 series IDs
series_map = (
    train_desc_df[
        train_desc_df['series_description']
        .isin(['Axial T2', 'Sagittal T2/STIR'])
    ]
    .drop_duplicates(subset=['study_id','series_description'])
    .pivot(index='study_id',
           columns='series_description',
           values='series_id')
    .reset_index()
)
series_map.columns = ['study_id','ax_series_id','sag_series_id']
print("→ series_map:", series_map.shape)

# 4b) Merge classification labels with that map
clf_map = pd.merge(new_train_df, series_map, on='study_id', how='inner')
print("→ classification merge:", clf_map.shape)

# 4c) Grab one image path per series_id
ax_paths = (
    paths_df[paths_df['series_description']=='Axial T2']
    .drop_duplicates(subset=['series_id'], keep='first')
    [['series_id','image_path']]
    .rename(columns={'series_id':'ax_series_id',
                     'image_path':'axial_t2_path'})
)
sag_paths = (
    paths_df[paths_df['series_description']=='Sagittal T2/STIR']
    .drop_duplicates(subset=['series_id'], keep='first')
    [['series_id','image_path']]
    .rename(columns={'series_id':'sag_series_id',
                     'image_path':'sagittal_t2stir_path'})
)

# 4d) Attach file paths
dataset_df = (
    clf_map
    .merge(ax_paths, on='ax_series_id', how='inner')
    .merge(sag_paths, on='sag_series_id', how='inner')
)
print("→ before cleanup:", dataset_df.shape)

# 4e) Keep only what you need and robustly map severity → 0/1/2
dataset_df = dataset_df[['axial_t2_path','sagittal_t2stir_path','severity']].copy()

# 4e.1) Normalize the text
dataset_df['severity_norm'] = (
    dataset_df['severity']
    .str.lower()
    .str.replace('/', '_')
    .str.strip()
)

# 4e.2) Map to integers
severity_map = {'normal_mild':0, 'moderate':1, 'severe':2}
dataset_df['severity_mapped'] = dataset_df['severity_norm'].map(severity_map)

# 4e.3) Inspect any unmapped rows (optional)
unmapped = dataset_df['severity_mapped'].isna().sum()
print(f"Unmapped severity rows: {unmapped}")

# 4e.4) Drop rows we couldn’t map
dataset_df = dataset_df.dropna(subset=['severity_mapped']).copy()

# 4e.5) Finalize the column
dataset_df['severity'] = dataset_df['severity_mapped'].astype(int)
dataset_df = dataset_df[['axial_t2_path','sagittal_t2stir_path','severity']]

print("→ final dataset_df (post‐drop):", dataset_df.shape)

# 4f) Transforms for each branch
axial_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize([0.485],[0.229])
])
sag_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.RandomAffine(15, translate=(0.1,0.1)),
    transforms.ToTensor(), transforms.Normalize([0.485],[0.229])
])

# 4g) Two-branch Dataset
class SpineT2PairDataset(Dataset):
    def __init__(self, df, axial_transform, sag_transform):
        self.df = df
        self.axial_transform = axial_transform
        self.sag_transform   = sag_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        rec = self.df.iloc[i]
        img_ax  = load_dicom_image(rec['axial_t2_path'])
        img_sag = load_dicom_image(rec['sagittal_t2stir_path'])
        if self.axial_transform: img_ax  = self.axial_transform(img_ax)
        if self.sag_transform:   img_sag = self.sag_transform(img_sag)
        return img_ax, img_sag, int(rec['severity'])

# 4h) Stratified split & sampler
train_df, val_df = train_test_split(
    dataset_df,
    stratify=dataset_df['severity'],
    test_size=0.3,
    random_state=42
)
print("Train/Val:", train_df.shape, "/", val_df.shape)

counts = train_df['severity'].value_counts().sort_index().values
class_weights = 1.0 / counts
sample_weights = train_df['severity'].map(lambda x: class_weights[x]).values
sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 4i) DataLoaders
train_ds = SpineT2PairDataset(train_df, axial_transforms, sag_transforms)
val_ds   = SpineT2PairDataset(val_df,   axial_transforms, sag_transforms)

train_loader = DataLoader(
    train_ds, batch_size=16, sampler=sampler,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False,
    num_workers=4, pin_memory=True
)

print("Batches/epoch:", len(train_loader), "/", len(val_loader))



print("→ dataset_df shape:", dataset_df.shape)



# ==== 5) Model Definition, Loss & Optimizer ====

import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Two-branch VGG16
class TwoBranchVGG16(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        base = models.vgg16(pretrained=pretrained)
        # shared feature extractor
        self.features = base.features
        self.avgpool  = base.avgpool
        feat_dim = base.classifier[0].in_features  # typically 25088

        # fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim*2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_ax, x_sag):
        fa = self.features(x_ax)
        fa = self.avgpool(fa)
        fa = fa.flatten(1)
        fs = self.features(x_sag)
        fs = self.avgpool(fs)
        fs = fs.flatten(1)
        f  = torch.cat([fa, fs], dim=1)
        return self.classifier(f)

# Instantiate model
model = TwoBranchVGG16(num_classes=3).to(device)

# Use class_weights from your sampler setup
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(model)



# ==== 6) Training & Evaluation Functions ====

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def train_epoch(loader):
    model.train()
    total_loss, total_correct, total_samples = 0., 0, 0
    for i, (ax, sag, labels) in enumerate(loader, 1):
        # every 200 batches, show how far along we are
        if i % 200 == 0:
            print(f"  processed {i}/{len(loader)} batches", end="\r", flush=True)

        ax, sag, labels = ax.to(device), sag.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(ax, sag)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss   += loss.item() * labels.size(0)
        preds         = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # print a newline to clear the "\r" on the last batch
    print()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def eval_epoch(loader):
    model.eval()
    total_loss, total_correct, total_samples = 0., 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ax, sag, labels in loader:
            ax, sag, labels = ax.to(device), sag.to(device), labels.to(device)
            outputs = model(ax, sag)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                   labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, accuracy, p, r, f1, cm



import os
import torch

# Where to save checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Try to resume from best checkpoint if it exists
best_val_loss = float('inf')
start_epoch  = 1
best_ckpt    = os.path.join(checkpoint_dir, "best.pt")
if os.path.isfile(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    best_val_loss = ckpt['val_loss']
    start_epoch   = ckpt['epoch'] + 1
    print(f"Resuming from epoch {start_epoch} with val_loss={best_val_loss:.4f}")



# ==== 7) Training Loop ====

num_epochs = 10
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   []
}

for epoch in range(start_epoch, num_epochs+1):
    print(f"\n Starting epoch {epoch}/{num_epochs}", flush=True)
    
    tr_loss, tr_acc = train_epoch(train_loader)
    print()
    
    val_loss, val_acc, p, r, f1, cm = eval_epoch(val_loader)
    scheduler.step(val_loss)

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # ---- save epoch checkpoint ----
    epoch_ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(epoch_ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

    # ---- save best checkpoint ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(epoch_ckpt, best_ckpt)
        print(f"  ✔ New best model saved (val_loss={val_loss:.4f})")

    # ---- print metrics ----
    print(f"  Precision: {p}, Recall: {r}, F1: {f1}")
    print(f"  Confusion Matrix:\n{cm}")




