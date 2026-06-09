# ========================================
# âš™ï¸�  Section 0 â€” Configs
# ----------------------------------------
import os, random, gc, json, zipfile, math, warnings, hashlib, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import albumentations as A
import albumentations.pytorch
import tifffile, cv2
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, roc_curve, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class CFG:
    COMP          = "mayo-clinic-strip-ai"
    TILE_SIZE     = 224          # è¯»å…¥å��å†�ç¼©æ”¾
    BATCH_SIZE    = 16
    EPOCHS        = 4             # DEBUG æ¨¡å¼�ä¸‹å…ˆè·‘ 1 ä¸ª epoch
    LR            = 1e-4
    IMG_MEAN      = (0.485,0.456,0.406)
    IMG_STD       = (0.229,0.224,0.225)
    DEBUG         = False          # ğŸš€ æ”¹æˆ� False è®­ç»ƒå…¨é‡�
    DEBUG_FRAC    = 1         # å�ªç”¨ 5 % æ ·æœ¬
    NUM_WORKERS   = 1
    SEED          = 42
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CFG()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(cfg.SEED)

# ----------------------------------------
# Section 1 â€” Initialization
# ----------------------------------------
data_dir = Path("/kaggle/input/mayo-clinic-strip-ai")
train_csv = pd.read_csv(data_dir / "train.csv")
test_csv  = pd.read_csv(data_dir / "test.csv")

# ----------------------------------------
# Section 2 â€” Configuration
# ----------------------------------------
if cfg.DEBUG:
    train_csv = (train_csv.groupby("label", group_keys=False)    # â†� æ”¹è¿™é‡Œ
                           .apply(lambda x: x.sample(frac=cfg.DEBUG_FRAC,
                                                     random_state=cfg.SEED))
                           .reset_index(drop=True))

# === Section 3 â€” patient-level split  ===
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
train_idx, val_idx = next(sgkf.split(train_csv,
                                    y=train_csv["label"],
                                    groups=train_csv["patient_id"]))

train_df = train_csv.iloc[train_idx].reset_index(drop=True)
val_df   = train_csv.iloc[val_idx].reset_index(drop=True)

label_map = {lbl: i for i, lbl in enumerate(sorted(train_csv["label"].unique()))}
train_df["label_id"] = train_df["label"].map(label_map)
val_df  ["label_id"] = val_df["label"].map(label_map)
num_classes = len(label_map)


# ----------------------------------------
# Section 4 â€” Dataset & Loading Mechanism
# ----------------------------------------
tfm_train = A.Compose([
    # A.RandomResizedCrop(size=(cfg.TILE_SIZE, cfg.TILE_SIZE),  # âœ… ç”¨ size
    #                     scale=(0.8, 1.0)),
    A.Resize(height=cfg.TILE_SIZE, width=cfg.TILE_SIZE),
    A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
    A.Normalize(cfg.IMG_MEAN, cfg.IMG_STD), albumentations.pytorch.ToTensorV2(),
])

tfm_val = A.Compose([
    A.Resize(height=cfg.TILE_SIZE, width=cfg.TILE_SIZE),      # âœ… ç”¨ height/width
    A.Normalize(cfg.IMG_MEAN, cfg.IMG_STD), albumentations.pytorch.ToTensorV2(),
])

class TileDataset(Dataset):
    def __init__(self, df, transforms=None, split="train"):
        self.df = df
        self.tfm = transforms
        self.split = split            # "train" / "test"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # â‡¢ è·¯å¾„
        folder = "train" if self.split == "train" else "test"
        img_path = data_dir / folder / f"{row.image_id}.tif"
        img = tifffile.imread(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.tfm:
            img = self.tfm(image=img)["image"]

        # â‡¢ æ ‡ç­¾ï¼šæµ‹è¯•é›†æ²¡æœ‰ label_idï¼Œç»™ä¸ªå“‘å…ƒ -1
        label = row.label_id if "label_id" in row else -1
        return img.float(), torch.tensor(label).long()

train_dl = DataLoader(
    TileDataset(train_df, tfm_train, split="train"),
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

val_dl = DataLoader(
    TileDataset(val_df, tfm_val, split="train"),
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

test_ds = TileDataset(test_csv, tfm_val, split="test")
test_dl = DataLoader(
    test_ds,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)
sample_id = train_df.image_id.iloc[0]
print((data_dir / "train" / f"{sample_id}.tif").exists())  # True

# ----------------------------------------
# Section 5 Model Creation & Initialization
# ---------------------------------------

# Adapted from https://github.com/rishikksh20/CrossViT-pytorch/blob/master/module.py

class AttentionMILPooling(nn.Module):
    
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention_fc1 = nn.Linear(in_dim, hidden_dim)
        self.attention_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = self.attention_fc2(torch.tanh(self.attention_fc1(x))) 
        attn_weights = F.softmax(attn_weights, dim=1) 
        weighted_sum = torch.sum(attn_weights * x, dim=1)
        return weighted_sum, attn_weights

class ViT_MIL_Model(nn.Module):
    
    def __init__(self, model_name, num_classes, attn_hidden_dim=64, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.reset_classifier(0)  # remove default classifier head
        self.mil_pool = AttentionMILPooling(self.vit.num_features, attn_hidden_dim)
        self.classifier = nn.Linear(self.vit.num_features, num_classes)

    def forward(self, x):
        # Extract patch embeddings + pooling
        patch_embeddings = self.vit.forward_features(x)  
        pooled, attn_weights = self.mil_pool(patch_embeddings)  
        
        # Classifier on the pooled features
        logits = self.classifier(pooled)  
        return logits, attn_weights

# chose the 224 base for simplicity
model = ViT_MIL_Model("vit_base_patch16_224", num_classes=num_classes).to(cfg.DEVICE)

# model.vit.patch_embed.img_size = (1024, 1024)
# model.vit.pos_embed = torch.nn.Parameter(
#     F.interpolate(
#         model.vit.pos_embed.reshape(1, 65, 768).transpose(1, 2).reshape(1, 768, 8, 8),
#         size=(64, 64),
#         mode='bicubic',
#         align_corners=False
#     ).reshape(1, 768, -1).transpose(1, 2)
# )

model = model.to(cfg.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=cfg.EPOCHS)

# ----------------------------------------
# Section 6 â€” Training
# ----------------------------------------
def train_one_epoch(dl):
    model.train()
    total = 0
    correct = 0
    for x, y in dl:
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        optimizer.zero_grad()
        output = model(x)
        out = output[0] if isinstance(output, tuple) else output
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        preds = out.argmax(1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return correct / total

@torch.no_grad()
def validate(dl):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_prob = []
    y_pred = []
    
    for x, y in dl:
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        output = model(x)
        out = output[0] if isinstance(output, tuple) else output
        prob = out.softmax(1)[:, 1]
        preds = out.argmax(1)
        y_true.append(y.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        total += y.size(0)
        correct += (preds == y).sum().item()

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.concatenate(y_pred)

    # calcualte metrics
    auc = roc_auc_score(y_true, y_prob)
    acc = correct / total
    average_precision = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    return auc, acc, average_precision, fpr, tpr, precision, recall
    
best_auc = 0.0
for epoch in range(cfg.EPOCHS):
    acc  = train_one_epoch(train_dl)
    val_auc, val_acc, average_precision, fpr, tpr, precision, recall  = validate(val_dl)
    scheduler.step()
    print(f"Epoch {epoch+1}/{cfg.EPOCHS}  train-acc={acc:.4f} val-acc={val_acc:.4f} val-AUC={val_auc:.4f} avg_precision={average_precision:.4f}")
    if val_auc > best_auc:
        # save the better model
        # print("saving")
        best_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")
        
# ----------------------------------------
# Section 7 â€” Test
# ----------------------------------------
test_ds = TileDataset(test_csv, tfm_val)
test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                     shuffle=False, num_workers=cfg.NUM_WORKERS,
                     pin_memory=True)

# --- ROC Curve based on best model --- 

plt.figure()
plt.plot(fpr, tpr, label=f'AUROC = {val_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Save to file
plt.savefig("ROC.png")

# --- Precision-Recall (PR) Curve based on best model ---

plt.figure()
plt.plot(recall, precision, label=f'AUPRC = {average_precision:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Save to file
plt.savefig("PR.png")

model.load_state_dict(torch.load("best_model.pth")); model.eval()

probs=[]
for x,_ in test_dl:
    x = x.to(cfg.DEVICE)
    logits, _ = model(x)
    prob = logits.softmax(1)[:, label_map["CE"]]
    probs.append(prob.detach().cpu().numpy())

sub = pd.DataFrame({
    "image_id": test_csv.image_id,
    "CE_prob":  np.concatenate(probs)
})
sub.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!\n", sub.head())

