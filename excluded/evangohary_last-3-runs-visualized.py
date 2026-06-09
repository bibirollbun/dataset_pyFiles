import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
import os
import json


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# used 0.5 for the old ones, and 0.98 for your final tuned version
MODELS_CONFIG = [
    ("First (20 Rotations)", "/kaggle/input/presentedtoclasscheckpoints/balanced_do_not_touch_model_rotating_safe_20rotations_rich.pth", 0.5),
    ("Middle (40 Rotations)", "/kaggle/input/presentedtoclasscheckpoints/balanced_do_not_touch_model_rotating_safe_40rotations_On_20_rotations.pth", 0.5),
    ("Final (60 Rotations)", "/kaggle/input/presentedtoclasscheckpoints/another_60_runs.pth", 0.98) 
]


class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(row["do_not_touch"]), dtype=torch.long)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


DATA_DIR = "/kaggle/input/herbarium-2022-fgvc9"
print("Loading Metadata...")
with open(os.path.join(DATA_DIR, "train_metadata.json"), "r") as f:
    train_meta = json.load(f)

ann_df = pd.DataFrame(train_meta["annotations"])
img_df = pd.DataFrame(train_meta["images"])
cat_df = pd.DataFrame(train_meta["categories"])
merged = ann_df.merge(img_df, on="image_id", how="left").merge(cat_df, on="category_id", how="left")
merged["image_path"] = merged["file_name"].apply(lambda fn: os.path.join(DATA_DIR, "train_images", fn))


toxic_genus_list = ["Toxicodendron", "Euphorbia", "Urtica", "Cicuta", "Conium", "Heracleum"]
merged["do_not_touch"] = merged["genus"].isin(toxic_genus_list).astype(int)


from sklearn.model_selection import train_test_split
_, test_df = train_test_split(merged, test_size=0.05, random_state=42, stratify=merged["do_not_touch"])

print(f"Test Set Size: {len(test_df)}")
test_dataset = PlantDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


results = {} 
for name, path, thresh in MODELS_CONFIG:
    print(f"\nProcessing: {name}...")
    
    # Init Model
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))
    model = model.to(device)
    
    # Load Weights
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        if "model_state_dict" in ckpt: model.load_state_dict(ckpt["model_state_dict"])
        else: model.load_state_dict(ckpt)
    else:
        print(f"!! WARNING: File not found {path}")
        continue
        
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            
    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    y_pred = (y_scores >= thresh).astype(int)
    
    results[name] = {
        "y_true": y_true,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "threshold": thresh
    }


plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: ROC Curve Overlay
plt.figure(figsize=(10, 6))
for name, data in results.items():
    fpr, tpr, _ = roc_curve(data["y_true"], data["y_scores"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Improvement in ROC Curve over Training Stages')
plt.legend(loc="lower right")
plt.show()

# Plot 2: Precision-Recall Curve Overlay 
plt.figure(figsize=(10, 6))
for name, data in results.items():
    precision, recall, _ = precision_recall_curve(data["y_true"], data["y_scores"])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.4f})', linewidth=2)

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Confidence)')
plt.title('Precision-Recall Curve: The Real Improvement')
plt.legend(loc="lower left")
plt.show()

# plot 3: Confusion Matrices Side-by-Side 
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, (name, data) in zip(axes, results.items()):
    cm = confusion_matrix(data["y_true"], data["y_pred"])
    
    # Custom annotations with "Safe" and "Toxic" labels
    group_names = ['True Safe','False Toxic','False Safe','True Toxic']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 14})
    ax.set_title(f"{name}\nThreshold: {data['threshold']}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Safe', 'Toxic'])
    ax.set_yticklabels(['Safe', 'Toxic'])

plt.tight_layout()
plt.show()

