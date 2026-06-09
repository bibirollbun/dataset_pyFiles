import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
# Note: Using train_test_split twice to achieve a three-way split (Train/Val/Test)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import timm 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Configuration

CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 32, 
    "epochs": 15,            
    "lr": 1e-4,              
    "num_classes": 3,        
    "model_name": "swin_tiny_patch4_window7_224", 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "image_dir": "/kaggle/input/preprocessed-images-224"
}

# Label Mapping: 0 -> 0 (Healthy); 1,2 -> 1 (Mild/Mod); 3,4 -> 2 (Sev/Prolif)
LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# Dataset
class APTOSFastDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        
        image = cv2.imread(img_path)
        if image is None: 
            # Return a black image if loading fails
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform: 
            image = self.transform(image)
            
        new_label = LABEL_MAP[row['diagnosis']]
        return image, torch.tensor(new_label, dtype=torch.long)

# Data Augmentation Transforms

train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Use val_tf for the test set as well
test_tf = val_tf

# ====================================================
# Data Preparation: Three-way Split (Train/Validation/Test) & Sampler Calculation
# ====================================================
df = pd.read_csv(CONFIG['csv_path'])
df['new_label'] = df['diagnosis'].map(LABEL_MAP)

# 1. First split: Separate 20% for the Test Set
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, # 20% for final independent testing
    random_state=CONFIG['seed'], 
    stratify=df['new_label']
)

# 2. Second split: Separate Validation Set from the remaining 80%
# We use 0.25 of the remaining data (0.25 * 0.8 = 0.2 total) to ensure Validation is also 20% of total
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.25, 
    random_state=CONFIG['seed'], 
    stratify=train_val_df['new_label']
)

print(f"Dataset Split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
print("-" * 65)

# --- Calculate WeightedRandomSampler weights (Train set only) ---
targets = train_df['new_label'].values
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = torch.FloatTensor([class_weights[t] for t in targets])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

print(f"âœ… Sampler Ready. Class Counts in Train: {class_counts}")

# Loaders
train_loader = DataLoader(
    APTOSFastDataset(train_df, CONFIG['image_dir'], train_tf), 
    batch_size=CONFIG['batch_size'], 
    sampler=sampler, 
    shuffle=False,    
    num_workers=2
)

val_loader = DataLoader(
    APTOSFastDataset(val_df, CONFIG['image_dir'], val_tf), 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=2
)

# New Test Loader
test_loader = DataLoader(
    APTOSFastDataset(test_df, CONFIG['image_dir'], test_tf), 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=2
)

# Model Training

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Using Swin Transformer (Tiny)
model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=3).to(CONFIG['device'])
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

best_sick_f1 = -1.0 
print(f"\nğŸš€ Starting Training with WeightedSampler & Augmentation...")

for epoch in range(CONFIG['epochs']):
    # --------------------------- Train ---------------------------
    model.train()
    train_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)"):
        imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --------------------------- Validation ---------------------------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(CONFIG['device'])
            outputs = model(imgs)
            _, p = torch.max(outputs, 1)
            val_preds.extend(p.cpu().numpy())
            val_labels.extend(lbls.numpy())

    # Metrics
    metrics = precision_recall_fscore_support(val_labels, val_preds, labels=[0, 1, 2], zero_division=0)
    precision, recall, f1_scores = metrics[0], metrics[1], metrics[2]
    avg_sick_f1 = (f1_scores[1] + f1_scores[2]) / 2.0
    overall_acc = np.mean(np.array(val_preds) == np.array(val_labels))

    # Log
    print(f"\nâœ… Epoch {epoch+1} Summary:")
    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Overall Acc: {overall_acc:.4f}")
    print("-" * 65)
    print(f"{'Class':<20} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 65)
    print(f"{'0 (Healthy)':<20} | {recall[0]:.4f}      | {precision[0]:.4f}         | {f1_scores[0]:.4f}")
    print(f"{'1 (Mild/Mod)':<20} | {recall[1]:.4f}      | {precision[1]:.4f}         | {f1_scores[1]:.4f}")
    print(f"{'2 (Sev/Prolif)':<20} | {recall[2]:.4f}      | {precision[2]:.4f}         | {f1_scores[2]:.4f}")
    print("-" * 65)
    print(f"ğŸ�¯ Val Avg Sick F1: {avg_sick_f1:.4f} (Best: {best_sick_f1:.4f})")

    # Save
    if avg_sick_f1 > best_sick_f1:
        best_sick_f1 = avg_sick_f1
        # Save model to the new save_dir path
        torch.save(model.state_dict(), os.path.join(save_dir, "best_sick_f1_model.pth"))
        print("ğŸ�† Best Model Updated!")

    scheduler.step(avg_sick_f1)
    print("\n")

# Final Evaluation on Independent Test Set
print("\n" + "="*70)
print("ğŸ�† FINAL EVALUATION ON INDEPENDENT TEST SET")
print("="*70)

# 1. Load best model
model.load_state_dict(torch.load(os.path.join(save_dir, "best_sick_f1_model.pth")))
model.eval()

# 2. Inference on Test Set
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Final Test"): # Use test_loader
        imgs = imgs.to(CONFIG['device'])
        outputs = model(imgs)
        _, p = torch.max(outputs, 1)
        test_preds.extend(p.cpu().numpy())
        test_labels.extend(lbls.numpy())

# 3. Calculate Final Metrics
test_metrics = precision_recall_fscore_support(test_labels, test_preds, labels=[0, 1, 2], zero_division=0)
test_precision, test_recall, test_f1_scores = test_metrics[0], test_metrics[1], test_metrics[2]
test_avg_sick_f1 = (test_f1_scores[1] + test_f1_scores[2]) / 2.0
test_overall_acc = np.mean(np.array(test_preds) == np.array(test_labels))

# 4. Print Report
print("\n" + "#"*65)
print(f"âœ¨ Test Overall Acc: {test_overall_acc:.4f}")
print("-" * 65)
print(f"{'Class':<20} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 65)
print(f"{'0 (Healthy)':<20} | {test_recall[0]:.4f}      | {test_precision[0]:.4f}         | {test_f1_scores[0]:.4f}")
print(f"{'1 (Mild/Mod)':<20} | {test_recall[1]:.4f}      | {test_precision[1]:.4f}         | {test_f1_scores[1]:.4f}")
print(f"{'2 (Sev/Prolif)':<20} | {test_recall[2]:.4f}      | {test_precision[2]:.4f}         | {test_f1_scores[2]:.4f}")
print("-" * 65)
print(f"ğŸ�‰ Final Test Avg Sick F1: {test_avg_sick_f1:.4f} (Val Best F1: {best_sick_f1:.4f})")
print("#"*65)

# 5. Plot Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
plt.title(f'Confusion Matrix on **Independent Test Set**\nTest Avg Sick F1: {test_avg_sick_f1:.4f}')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.show()
print("ğŸ�� Training and Final Evaluation Complete.")

