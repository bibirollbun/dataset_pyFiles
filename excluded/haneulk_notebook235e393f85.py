import matplotlib.pyplot as plt
import os, random, gc, json, zipfile, math, warnings, hashlib, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
import albumentations as A
import albumentations.pytorch
import tifffile, cv2
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, roc_curve, precision_score, recall_score
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier

# ------------------------------
# 1. Setup
# ------------------------------
    
class CFG:
    COMP          = "mayo-clinic-strip-ai"
    TILE_SIZE     = 1024
    BATCH_SIZE    = 16
    LR            = 1e-4
    IMG_MEAN      = (0.485,0.456,0.406)
    IMG_STD       = (0.229,0.224,0.225)
    FRAC          = 0.5      # Percentage of dataset to run
    NUM_WORKERS   = 0
    SEED          = 42
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_DEPTH     = 4         # XGBoost max depth
    NUM_EST       = 50        # XGBoost number of estimators
    LEARNING_RATE = 0.05      # XGBoost eta
    REG_L1        = 1         # XGBoost L1 regularization
    REG_L2        = 1         # XGBoost L2 regularization
    RESNET        = 'resnet50'    # resnet50 or resnet18
cfg = CFG()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(cfg.SEED)

data_dir = Path("/kaggle/input/mayo-clinic-strip-ai")
train_csv = pd.read_csv(data_dir / "train.csv")
test_csv  = pd.read_csv(data_dir / "test.csv")

train_csv = (train_csv.groupby("label", group_keys=False)
                       .apply(lambda x: x.sample(frac=cfg.FRAC,
                                                 random_state=cfg.SEED))
                       .reset_index(drop=True))

# ------------------------------
# 2. Image Preprocessing
# ------------------------------

transform = A.Compose([
    A.Resize(height=cfg.TILE_SIZE, width=cfg.TILE_SIZE),
    A.Normalize(cfg.IMG_MEAN, cfg.IMG_STD), albumentations.pytorch.ToTensorV2(),
])


class TileDataset(Dataset):
    def __init__(self, df, transforms=None, split="train"):
        self.df = df
        self.tfm = transforms
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        folder = "train" if self.split == "train" else "test"
        img_path = data_dir / folder / f"{row.image_id}.tif"
        img = tifffile.imread(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.tfm:
            img = self.tfm(image=img)["image"]

        label = row.label_id if "label_id" in row else -1
        return img.float(), torch.tensor(label).long()

# ------------------------------
# 3. Split Before Feature Extraction
# ------------------------------

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)

# ------------------------------
# 4. Feature Extraction Function
# ------------------------------

for i, (train_idx, test_idx) in enumerate(sgkf.split(train_csv, y=train_csv["label"], groups=train_csv["patient_id"])):
    if i == 0:  # second split (0-based indexing)
        break

train_df = train_csv.iloc[train_idx].reset_index(drop=True)
test_df   = train_csv.iloc[test_idx].reset_index(drop=True)

label_map = {lbl: i for i, lbl in enumerate(sorted(train_csv["label"].unique()))}
train_df["label_id"] = train_df["label"].map(label_map)
test_df  ["label_id"] = test_df["label"].map(label_map)
num_classes = len(label_map)

model = timm.create_model(cfg.RESNET, pretrained=True, num_classes=num_classes)
model = model.to(cfg.DEVICE)
criterion = nn.CrossEntropyLoss()

def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            out = model(x);  loss = criterion(out,y)
            features.append(out.cpu().numpy())
            labels.extend(y.numpy())
    return np.concatenate(features), np.array(labels)


train_dl = DataLoader(
    TileDataset(train_df, transform, split="train"),
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

test_dl = DataLoader(
    TileDataset(test_df, transform, split="train"),
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True,
)

sample_id = train_df.image_id.iloc[0]
print((data_dir / "train" / f"{sample_id}.tif").exists())

X_train, y_train = extract_features(train_dl)
X_test, y_test = extract_features(test_dl)


# ------------------------------
# 5. Train XGBoost Classifier
# ------------------------------

xgb = XGBClassifier(n_estimators=cfg.NUM_EST,
                    max_depth=cfg.MAX_DEPTH,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    learning_rate=cfg.LEARNING_RATE,
                    reg_alpha=cfg.REG_L1,       # L1 regularization
                    reg_lambda=cfg.REG_L2)      # L2 regularization)
xgb.fit(X_train, y_train)


# ------------------------------
# 6. Evaluation
# ------------------------------

y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)
y_prob_train = xgb.predict_proba(X_train)[:,label_map["CE"]]
y_prob_test = xgb.predict_proba(X_test)[:,label_map["CE"]]
auc_test = roc_auc_score(y_test, y_prob_test)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
prc_test = average_precision_score(y_test, y_prob_test)
prec = precision_score(y_test, y_pred_test)
rec = recall_score(y_test, y_pred_test)

print(f"max_depth={cfg.MAX_DEPTH} n_estimators={cfg.NUM_EST} learning_rate={cfg.LEARNING_RATE} reg_alpha={cfg.REG_L1} reg_lambda={cfg.REG_L2}")
print(f"train-acc={acc_train:.4f} val-acc={acc_test:.4f} val-AUC={auc_test:.4f} val-PRC={prc_test:.4f} precision={prec:.4f} recall={rec:.4f}")


# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob_test)

plt.figure()
plt.plot(fpr, tpr, label=f'AUROC = {auc_test:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Save to file
plt.savefig("ROC.png")

# --- Precision-Recall (PR) Curve ---
precision, recall, _ = precision_recall_curve(y_test, y_prob_test)


plt.figure()
plt.plot(recall, precision, label=f'AUPRC = {prc_test:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Save to file
plt.savefig("PR.png")

torch.save(model.state_dict(), "cnn.pth")
xgb.save_model("xgb.json")
xgb.save_model("model.xgb")


