import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc, 
    confusion_matrix, classification_report
)
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
import itertools


DATA_DIR    = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
CSV_PATH    = os.path.join(DATA_DIR, 'stage_2_train.csv')
DCM_DIR     = os.path.join(DATA_DIR, 'stage_2_train')
PNG_ROOT    = '/kaggle/working/png_data'
OUTPUT_DIR  = '/kaggle/working/output_single_run'
ROC_DIR     = os.path.join(OUTPUT_DIR, 'roc_curves')
CAM_DIR     = os.path.join(OUTPUT_DIR, 'gradcam')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')

for d in [PNG_ROOT, OUTPUT_DIR, ROC_DIR, CAM_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE    = 8
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-6
PATIENCE      = 10
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES   = ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']
BALANCE_N     = 500
VAL_SPLIT_SIZE= 0.2

def display_sample_images_per_class(df, root_dir, classes, num_samples=5):
    n_cols = num_samples
    n_rows = len(classes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle('Sample Images Per Hemorrhage Subtype (Positive Examples)', fontsize=16, y=1.02)
    
    for i, cls in enumerate(classes):
        positive_samples = df[df[cls] == 1]
        samples = positive_samples.head(n_cols)
        image_ids = samples['image'].tolist()

        for j in range(n_cols):
            ax = axes[i, j]
            
            if j < len(image_ids):
                img_id = image_ids[j]
                png_path = os.path.join(root_dir, img_id + '.png')
                
                try:
                    img = Image.open(png_path).convert('RGB')
                    ax.imshow(img)
                    ax.set_title(f"ID: {img_id}", fontsize=10)
                except FileNotFoundError:
                    ax.text(0.5, 0.5, "PNG Not Found", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Missing: {img_id}", fontsize=10)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

        axes[i, 0].set_ylabel(cls.upper(), fontsize=12, rotation=90, labelpad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images_per_class.png'))
    plt.close(fig)
    print(f"Saved sample image grid to {os.path.join(OUTPUT_DIR, 'sample_images_per_class.png')}")

df = pd.read_csv(CSV_PATH)
df[['image','subtype']] = df['ID'].str.rsplit('_', n=1, expand=True)


df


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x="subtype",
    order=df["subtype"].value_counts().index
)

plt.title("Count Plot of Hemorrhage Subtypes")
plt.xlabel("Subtype")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


subtype_counts = df["subtype"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    subtype_counts.values,
    labels=subtype_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Pie Chart of Hemorrhage Subtypes")
plt.axis("equal")
plt.show()


df_ml = df.pivot_table(
    index='image',
    columns='subtype',
    values='Label',
    aggfunc='max',
    fill_value=0
).reset_index()
df_ml['any'] = df_ml[CLASS_NAMES].max(axis=1)

frames = []
for cls in CLASS_NAMES:
    pos = df_ml[df_ml[cls]==1]
    neg = df_ml[df_ml[cls]==0]
    
    pos = pos.sample(BALANCE_N, random_state=42) if len(pos)>BALANCE_N else pos
    neg = neg.sample(BALANCE_N, random_state=42) if len(neg)>BALANCE_N else neg
    
    frames.append(pos)
    frames.append(neg)

balanced_df = pd.concat(frames).drop_duplicates(subset='image').reset_index(drop=True)
print(f"Balanced subset size: {len(balanced_df)} images")

balanced_csv = os.path.join(METRICS_DIR, 'subset_multilabel_balanced.csv')
balanced_df.to_csv(balanced_csv, index=False)

print("Converting balanced subset DICOM to PNG...")
for img_id in tqdm(balanced_df['image'].unique(), desc="DICOM→PNG"):
    dcm_path = os.path.join(DCM_DIR, img_id + '.dcm')
    png_path = os.path.join(PNG_ROOT, img_id + '.png')
    
    if os.path.exists(png_path) or not os.path.exists(dcm_path):
        continue
        
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(png_path)

print("Done! Balanced subset PNGs are in:", PNG_ROOT)

display_sample_images_per_class(balanced_df, PNG_ROOT, CLASS_NAMES, num_samples=5)


class MultiHemoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.copy().reset_index(drop=True) 
        self.root = root_dir
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.iloc[idx]
        
        path = os.path.join(self.root, rec['image'] + '.png')
        img = Image.open(path).convert('RGB')
        if self.tf:
            img = self.tf(img)
            
        label_arr = rec[CLASS_NAMES].astype(np.float32).to_numpy()
        labels = torch.from_numpy(label_arr)
        
        return img, labels

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_df, val_df = train_test_split(
    balanced_df, 
    test_size=VAL_SPLIT_SIZE, 
    random_state=42, 
    stratify=balanced_df['any']
)

train_loader = DataLoader(
    MultiHemoDataset(train_df, PNG_ROOT, train_transform),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(
    MultiHemoDataset(val_df,   PNG_ROOT, val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training images: {len(train_df)}, Validation images: {len(val_df)}")

model     = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc  = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model     = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)

best_val_loss    = np.inf
no_improve       = 0
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []
best_model_path  = os.path.join(OUTPUT_DIR, 'best_model.pth')

print(f"\n=== Starting Training (Max {NUM_EPOCHS} Epochs) ===")
for epoch in range(1, NUM_EPOCHS+1):
    
    model.train()
    running_loss = running_corr = running_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * imgs.size(0)
        preds         = (torch.sigmoid(outputs) >= 0.5).long()
        running_corr  += (preds == labels).all(dim=1).sum().item()
        running_total += imgs.size(0)

    epoch_train_loss = running_loss / running_total
    epoch_train_acc  = running_corr  / running_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    model.eval()
    val_running_loss = val_running_corr = val_running_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            val_running_loss  += loss.item() * imgs.size(0)
            preds              = (torch.sigmoid(outputs) >= 0.5).long()
            val_running_corr  += (preds == labels).all(dim=1).sum().item()
            val_running_total += imgs.size(0)

    epoch_val_loss = val_running_loss / val_running_total
    epoch_val_acc  = val_running_corr  / val_running_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch}/{NUM_EPOCHS} — "
          f"Train loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f} | "
          f" Val loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        no_improve    = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE and NUM_EPOCHS > PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print("\n=== Final Evaluation and Metrics ===")
model.load_state_dict(torch.load(best_model_path))
model.eval()

val_probs_list, val_targets_list = [], []
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Final Evaluation"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        val_probs_list.extend(torch.sigmoid(outputs).cpu().numpy())
        val_targets_list.extend(labels.cpu().numpy())
        
val_probs   = np.array(val_probs_list)
val_targets = np.array(val_targets_list)
val_preds   = (val_probs >= 0.5).astype(int)

print("\n--- Multi-Label Classification Report (Threshold=0.5) ---")
report = classification_report(
    val_targets, val_preds, 
    target_names=CLASS_NAMES, 
    zero_division=0, 
    output_dict=True
)
report_df = pd.DataFrame(report).transpose().round(4)
print(report_df.to_markdown())
report_df.to_csv(os.path.join(METRICS_DIR, 'classification_report.csv'))

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'cm_{title.replace(" ", "_")}.png'))
    plt.close()

for i, cls in enumerate(CLASS_NAMES):
    cm = confusion_matrix(val_targets[:, i], val_preds[:, i])
    plot_confusion_matrix(
        cm, 
        classes=['Negative', 'Positive'], 
        title=f'Confusion Matrix: {cls}'
    )
    
print(f"\nSaved {len(CLASS_NAMES)} Confusion Matrix plots to: {METRICS_DIR}")

plt.figure(figsize=(8,6))
metrics_list = []
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(val_targets[:, i], val_probs[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    metrics_list.append({
        'class': cls,
        'AUC': roc_auc,
        'Precision': report_df.loc[cls, 'precision'],
        'Recall': report_df.loc[cls, 'recall'],
        'F1-Score': report_df.loc[cls, 'f1-score']
    })

plt.plot([0,1],[0,1],'--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig(os.path.join(ROC_DIR, 'roc_all_classes.png'))
plt.close()

pd.DataFrame(metrics_list).to_csv(os.path.join(METRICS_DIR, 'final_metrics_summary.csv'), index=False)
print(f"Saved ROC curve plot to: {ROC_DIR}")

print(f"\nAll results (metrics, CM, ROC) saved to: {OUTPUT_DIR}")


class MultiHemoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.copy().reset_index(drop=True) 
        self.root = root_dir
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.iloc[idx]
        
        path = os.path.join(self.root, rec['image'] + '.png')
        img = Image.open(path).convert('RGB')
        if self.tf:
            img = self.tf(img)
            
        label_arr = rec[CLASS_NAMES].astype(np.float32).to_numpy()
        labels = torch.from_numpy(label_arr)
        
        return img, labels

train_transform = transforms.Compose([
    transforms.Resize(340), 
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(299), 
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_df, val_df = train_test_split(
    balanced_df, 
    test_size=VAL_SPLIT_SIZE, 
    random_state=42, 
    stratify=balanced_df['any']
)

train_loader = DataLoader(
    MultiHemoDataset(train_df, PNG_ROOT, train_transform),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(
    MultiHemoDataset(val_df,   PNG_ROOT, val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training images: {len(train_df)}, Validation images: {len(val_df)}")

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=True)

model.AuxLogits = None
model.aux_logits = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)

best_val_loss    = np.inf
no_improve       = 0
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []
best_model_path  = os.path.join(OUTPUT_DIR, 'best_model.pth')

print(f"\n=== Starting Training (Max {NUM_EPOCHS} Epochs) ===")
for epoch in range(1, NUM_EPOCHS+1):
    
    model.train()
    running_loss = running_corr = running_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(imgs) 
        
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * imgs.size(0)
        preds         = (torch.sigmoid(outputs) >= 0.5).long()
        running_corr  += (preds == labels).all(dim=1).sum().item()
        running_total += imgs.size(0)

    epoch_train_loss = running_loss / running_total
    epoch_train_acc  = running_corr  / running_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    model.eval()
    val_running_loss = val_running_corr = val_running_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            val_running_loss  += criterion(outputs, labels).item() * imgs.size(0)
            preds              = (torch.sigmoid(outputs) >= 0.5).long()
            val_running_corr  += (preds == labels).all(dim=1).sum().item()
            val_running_total += imgs.size(0)

    epoch_val_loss = val_running_loss / val_running_total
    epoch_val_acc  = val_running_corr  / val_running_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch}/{NUM_EPOCHS} — "
          f"Train loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f} | "
          f" Val loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        no_improve    = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE and NUM_EPOCHS > PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print("\n=== Final Evaluation and Metrics ===")
model.load_state_dict(torch.load(best_model_path))
model.eval()

val_probs_list, val_targets_list = [], []
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Final Evaluation"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        val_probs_list.extend(torch.sigmoid(outputs).cpu().numpy())
        val_targets_list.extend(labels.cpu().numpy())
        
val_probs   = np.array(val_probs_list)
val_targets = np.array(val_targets_list)
val_preds   = (val_probs >= 0.5).astype(int)

print("\n--- Multi-Label Classification Report (Threshold=0.5) ---")
report = classification_report(
    val_targets, val_preds, 
    target_names=CLASS_NAMES, 
    zero_division=0, 
    output_dict=True
)
report_df = pd.DataFrame(report).transpose().round(4)
print(report_df.to_markdown())
report_df.to_csv(os.path.join(METRICS_DIR, 'classification_report.csv'))

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'cm_{title.replace(" ", "_")}.png'))
    plt.close()

for i, cls in enumerate(CLASS_NAMES):
    cm = confusion_matrix(val_targets[:, i], val_preds[:, i])
    plot_confusion_matrix(
        cm, 
        classes=['Negative', 'Positive'], 
        title=f'Confusion Matrix: {cls}'
    )
    
print(f"\nSaved {len(CLASS_NAMES)} Confusion Matrix plots to: {METRICS_DIR}")

plt.figure(figsize=(8,6))
metrics_list = []
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(val_targets[:, i], val_probs[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    metrics_list.append({
        'class': cls,
        'AUC': roc_auc,
        'Precision': report_df.loc[cls, 'precision'],
        'Recall': report_df.loc[cls, 'recall'],
        'F1-Score': report_df.loc[cls, 'f1-score']
    })

plt.plot([0,1],[0,1],'--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig(os.path.join(ROC_DIR, 'roc_all_classes.png'))
plt.close()

pd.DataFrame(metrics_list).to_csv(os.path.join(METRICS_DIR, 'final_metrics_summary.csv'), index=False)
print(f"Saved ROC curve plot to: {ROC_DIR}")

print(f"\nAll results (metrics, CM, ROC) saved to: {OUTPUT_DIR}")


class MultiHemoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.copy().reset_index(drop=True) 
        self.root = root_dir
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.iloc[idx]
        
        path = os.path.join(self.root, rec['image'] + '.png')
        img = Image.open(path).convert('RGB')
        if self.tf:
            img = self.tf(img)
            
        label_arr = rec[CLASS_NAMES].astype(np.float32).to_numpy()
        labels = torch.from_numpy(label_arr)
        
        return img, labels

train_transform = transforms.Compose([
    transforms.Resize(340), 
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(299), 
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_df, val_df = train_test_split(
    balanced_df, 
    test_size=VAL_SPLIT_SIZE, 
    random_state=42, 
    stratify=balanced_df['any']
)

class_counts = train_df[CLASS_NAMES].sum()
total_samples = len(train_df)

weights = (total_samples - class_counts) / (class_counts + 1e-6)
weights = weights / weights.mean() 
class_weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

train_loader = DataLoader(
    MultiHemoDataset(train_df, PNG_ROOT, train_transform),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(
    MultiHemoDataset(val_df,   PNG_ROOT, val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training images: {len(train_df)}, Validation images: {len(val_df)}")

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=True)

model.AuxLogits = None
model.aux_logits = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)

best_val_loss    = np.inf
no_improve       = 0
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []
best_model_path  = os.path.join(OUTPUT_DIR, 'best_model.pth')

print(f"\n=== Starting Training (Max {NUM_EPOCHS} Epochs) ===")
for epoch in range(1, NUM_EPOCHS+1):
    
    model.train()
    running_loss = running_corr = running_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(imgs) 
        
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * imgs.size(0)
       
        preds         = (torch.sigmoid(outputs) >= 0.5).long()
        running_corr  += (preds == labels).all(dim=1).sum().item()
        running_total += imgs.size(0)

    epoch_train_loss = running_loss / running_total
    epoch_train_acc  = running_corr  / running_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    model.eval()
    val_running_loss = val_running_corr = val_running_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            unweighted_criterion = nn.BCEWithLogitsLoss() 
            val_running_loss  += unweighted_criterion(outputs, labels).item() * imgs.size(0)
            preds              = (torch.sigmoid(outputs) >= 0.5).long()
            val_running_corr  += (preds == labels).all(dim=1).sum().item()
            val_running_total += imgs.size(0)

    epoch_val_loss = val_running_loss / val_running_total
    epoch_val_acc  = val_running_corr  / val_running_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch}/{NUM_EPOCHS} — "
          f"Train loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f} | "
          f" Val loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        no_improve    = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE and NUM_EPOCHS > PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print("\n=== Final Evaluation and Metrics ===")

model.load_state_dict(torch.load(best_model_path))
model.eval()

val_probs_list, val_targets_list = [], []
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Final Evaluation"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        val_probs_list.extend(torch.sigmoid(outputs).cpu().numpy())
        val_targets_list.extend(labels.cpu().numpy())
        
val_probs   = np.array(val_probs_list)
val_targets = np.array(val_targets_list)
val_preds   = (val_probs >= 0.5).astype(int)

print("\n--- Multi-Label Classification Report (Threshold=0.5) ---")

report = classification_report(
    val_targets, val_preds, 
    target_names=CLASS_NAMES, 
    zero_division=0, 
    output_dict=True
)
report_df = pd.DataFrame(report).transpose().round(4)
print(report_df.to_markdown())
report_df.to_csv(os.path.join(METRICS_DIR, 'classification_report.csv'))

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'cm_{title.replace(" ", "_")}.png'))
    plt.close()

for i, cls in enumerate(CLASS_NAMES):
    cm = confusion_matrix(val_targets[:, i], val_preds[:, i])
    plot_confusion_matrix(
        cm, 
        classes=['Negative', 'Positive'], 
        title=f'Confusion Matrix: {cls}'
    )
    
print(f"\nSaved {len(CLASS_NAMES)} Confusion Matrix plots to: {METRICS_DIR}")

plt.figure(figsize=(8,6))
metrics_list = []
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(val_targets[:, i], val_probs[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    metrics_list.append({
        'class': cls,
        'AUC': roc_auc,
        'Precision': report_df.loc[cls, 'precision'],
        'Recall': report_df.loc[cls, 'recall'],
        'F1-Score': report_df.loc[cls, 'f1-score']
    })

plt.plot([0,1],[0,1],'--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig(os.path.join(ROC_DIR, 'roc_all_classes.png'))
plt.close()

pd.DataFrame(metrics_list).to_csv(os.path.join(METRICS_DIR, 'final_metrics_summary.csv'), index=False)
print(f"Saved ROC curve plot to: {ROC_DIR}")

print(f"\nAll results (metrics, CM, ROC) saved to: {OUTPUT_DIR}")

