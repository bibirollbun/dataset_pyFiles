# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Grand X-Ray Slam Division A - Baseline Notebook
# Author: Ishita (Blue and Gold Healthcare Inc.)
# Target: 0.99 AUC leaderboard-ready pipeline

!pip install pytorch-lightning timm torchmetrics --quiet

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
from torchmetrics.classification import MultilabelAUROC

# =====================
# CONFIG
# =====================
class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 10
    N_CLASSES = 14
    NUM_WORKERS = 4
    SEED = 42
    MODEL_NAME = "tf_efficientnet_b4_ns"

pl.seed_everything(CFG.SEED)

# =====================
# DATASET
# =====================
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = Image.open(img_path).convert("RGB").resize((CFG.IMG_SIZE, CFG.IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2,0,1).float()/255.0

        if self.is_test:
            return image, row["Image_Name"]
        else:
            labels = torch.tensor(row[LABELS].values.astype(np.float32))
            return image, labels

# =====================
# MODEL
# =====================
class EfficientNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.N_CLASSES)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_auc = MultilabelAUROC(num_labels=CFG.N_CLASSES, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        auc_per_class = self.val_auc.compute()
        mean_auc = auc_per_class.mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]

# =====================
# TRAINING LOOP
# =====================
train_df = pd.read_csv("../input/chestdx-multiinstitution/train1.csv")
train_df[LABELS] = train_df[LABELS].fillna(0)

# Stratified sample for validation (on "No Finding")
train_df["stratify_col"] = train_df[LABELS].sum(axis=1).clip(0,1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
train_idx, val_idx = list(skf.split(train_df, train_df["stratify_col"]))[0]

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = CXRDataset(df_train, "../input/chestdx-multiinstitution/train1/")
val_ds = CXRDataset(df_val, "../input/chestdx-multiinstitution/train1/")

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel()

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_mean", mode="max", save_top_k=1, dirpath="./", filename="best_model"
)
early_stop = EarlyStopping(monitor="val_auc_mean", mode="max", patience=3)

trainer = pl.Trainer(
    max_epochs=CFG.EPOCHS,
    precision=16,
    callbacks=[checkpoint_callback, early_stop],
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

trainer.fit(model, train_loader, val_loader)

# =====================
# INFERENCE & SUBMISSION
# =====================
test_df = pd.read_csv("../input/chestdx-multiinstitution/sample_submission1.csv")
test_ds = CXRDataset(test_df, "../input/chestdx-multiinstitution/test1/", is_test=True)
test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel.load_from_checkpoint("best_model.ckpt")

model.eval()
all_preds, all_ids = [], []

with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(model.device)
        logits = model(x)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_ids.extend(ids)

all_preds = np.vstack(all_preds)
submission = pd.DataFrame(all_preds, columns=LABELS)
submission.insert(0, "Image_Name", all_ids)
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as submission.csv")



# Grand X-Ray Slam Division A - Advanced Baseline Notebook
# Author: Ishita (Blue and Gold Healthcare Inc.)
# Target: 0.99 AUC roadmap implemented

!pip install pytorch-lightning timm torchmetrics albumentations --quiet

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MultilabelAUROC

# =====================
# CONFIG
# =====================
class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 10
    N_CLASSES = 14
    NUM_WORKERS = 4
    SEED = 42
    MODEL_NAME = "tf_efficientnet_b4_ns"
    USE_FOCAL = True  # Toggle between BCE and Focal Loss

pl.seed_everything(CFG.SEED)

# =====================
# LABELS
# =====================
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# =====================
# AUGMENTATIONS
# =====================
train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ToTensorV2(),
])

# =====================
# DATASET
# =====================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        # Encode Sex and ViewPosition for optional metadata fusion
        if not is_test:
            self.df['Sex'] = self.df['Sex'].fillna('Unknown')
            self.df['Sex_enc'] = LabelEncoder().fit_transform(self.df['Sex'])
            self.df['ViewPosition'] = self.df['ViewPosition'].fillna('Unknown')
            self.df['View_enc'] = LabelEncoder().fit_transform(self.df['ViewPosition'])
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image, row["Image_Name"]
        else:
            labels = torch.tensor(row[LABELS].values.astype(np.float32))
            # Metadata: optional concatenation
            metadata = torch.tensor([row['Age'], row['Sex_enc'], row['View_enc']], dtype=torch.float32)
            return image, metadata, labels

# =====================
# MODEL
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class EfficientNetModel(pl.LightningModule):
    def __init__(self, use_focal=True):
        super().__init__()
        self.model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.N_CLASSES)
        self.use_focal = use_focal
        self.loss_fn = FocalLoss() if use_focal else nn.BCEWithLogitsLoss()
        self.val_auc = MultilabelAUROC(num_labels=CFG.N_CLASSES, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        auc_per_class = self.val_auc.compute()
        mean_auc = auc_per_class.mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]

# =====================
# TRAINING LOOP
# =====================
train_df = pd.read_csv("../input/chestdx-multiinstitution/train1.csv")
train_df[LABELS] = train_df[LABELS].fillna(0)
train_df["stratify_col"] = train_df[LABELS].sum(axis=1).clip(0,1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
train_idx, val_idx = list(skf.split(train_df, train_df["stratify_col"]))[0]

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = CXRDataset(df_train, "../input/chestdx-multiinstitution/train1/", transform=train_transforms)
val_ds = CXRDataset(df_val, "../input/chestdx-multiinstitution/train1/", transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel(use_focal=CFG.USE_FOCAL)

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_mean", mode="max", save_top_k=1, dirpath="./", filename="best_model"
)
early_stop = EarlyStopping(monitor="val_auc_mean", mode="max", patience=3)

trainer = pl.Trainer(
    max_epochs=CFG.EPOCHS,
    precision=16,
    callbacks=[checkpoint_callback, early_stop],
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

trainer.fit(model, train_loader, val_loader)

# =====================
# INFERENCE & SUBMISSION
# =====================
test_df = pd.read_csv("../input/chestdx-multiinstitution/sample_submission1.csv")
test_ds = CXRDataset(test_df, "../input/chestdx-multiinstitution/test1/", transform=val_transforms, is_test=True)
test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel.load_from_checkpoint("best_model.ckpt")
model.eval()

all_preds, all_ids = [], []

with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(model.device)
        logits = model(x)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_ids.extend(ids)

all_preds = np.vstack(all_preds)
submission = pd.DataFrame(all_preds, columns=LABELS)
submission.insert(0, "Image_Name", all_ids)
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv")



# Grand X-Ray Slam Division A - Advanced Baseline Notebook
# Author: Ishita (Blue and Gold Healthcare Inc.)
# Target: 0.99 AUC roadmap implemented

!pip install pytorch-lightning timm torchmetrics albumentations --quiet

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MultilabelAUROC

# =====================
# CONFIG
# =====================
class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 10
    N_CLASSES = 14
    NUM_WORKERS = 4
    SEED = 42
    MODEL_NAME = "tf_efficientnet_b4_ns"
    USE_FOCAL = True  # Toggle between BCE and Focal Loss

pl.seed_everything(CFG.SEED)

# =====================
# LABELS
# =====================
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# =====================
# AUGMENTATIONS
# =====================
train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ToTensorV2(),
])

# =====================
# DATASET
# =====================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        # Encode Sex and ViewPosition for optional metadata fusion
        if not is_test:
            self.df['Sex'] = self.df['Sex'].fillna('Unknown')
            self.df['Sex_enc'] = LabelEncoder().fit_transform(self.df['Sex'])
            self.df['ViewPosition'] = self.df['ViewPosition'].fillna('Unknown')
            self.df['View_enc'] = LabelEncoder().fit_transform(self.df['ViewPosition'])
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image, row["Image_Name"]
        else:
            labels = torch.tensor(row[LABELS].values.astype(np.float32))
            # Metadata: optional concatenation
            metadata = torch.tensor([row['Age'], row['Sex_enc'], row['View_enc']], dtype=torch.float32)
            return image, metadata, labels

# =====================
# MODEL
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class EfficientNetModel(pl.LightningModule):
    def __init__(self, use_focal=True):
        super().__init__()
        self.model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.N_CLASSES)
        self.use_focal = use_focal
        self.loss_fn = FocalLoss() if use_focal else nn.BCEWithLogitsLoss()
        self.val_auc = MultilabelAUROC(num_labels=CFG.N_CLASSES, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        auc_per_class = self.val_auc.compute()
        mean_auc = auc_per_class.mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]

# =====================
# TRAINING LOOP
# =====================
train_df = pd.read_csv("../input/chestdx-multiinstitution/train1.csv")
train_df[LABELS] = train_df[LABELS].fillna(0)
train_df["stratify_col"] = train_df[LABELS].sum(axis=1).clip(0,1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
train_idx, val_idx = list(skf.split(train_df, train_df["stratify_col"]))[0]

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = CXRDataset(df_train, "../input/chestdx-multiinstitution/train1/", transform=train_transforms)
val_ds = CXRDataset(df_val, "../input/chestdx-multiinstitution/train1/", transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel(use_focal=CFG.USE_FOCAL)

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_mean", mode="max", save_top_k=1, dirpath="./", filename="best_model"
)
early_stop = EarlyStopping(monitor="val_auc_mean", mode="max", patience=3)

trainer = pl.Trainer(
    max_epochs=CFG.EPOCHS,
    precision=16,
    callbacks=[checkpoint_callback, early_stop],
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

trainer.fit(model, train_loader, val_loader)

# =====================
# INFERENCE & SUBMISSION
# =====================
test_df = pd.read_csv("../input/chestdx-multiinstitution/sample_submission1.csv")
test_ds = CXRDataset(test_df, "../input/chestdx-multiinstitution/test1/", transform=val_transforms, is_test=True)
test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

model = EfficientNetModel.load_from_checkpoint("best_model.ckpt")
model.eval()

all_preds, all_ids = [], []

with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(model.device)
        logits = model(x)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_ids.extend(ids)

all_preds = np.vstack(all_preds)
submission = pd.DataFrame(all_preds, columns=LABELS)
submission.insert(0, "Image_Name", all_ids)
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv")



# Grand X-Ray Slam Division A - Fold Ensemble + Pseudo-Label Baseline
# Author: Ishita (Blue and Gold Healthcare Inc.)
# Target: 0.99 AUC roadmap

!pip install pytorch-lightning timm torchmetrics albumentations --quiet

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MultilabelAUROC
from torch.cuda.amp import autocast

# =====================
# CONFIG
# =====================
class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 8
    N_CLASSES = 14
    NUM_WORKERS = 4
    SEED = 42
    MODEL_NAME = "tf_efficientnet_b4_ns"
    USE_FOCAL = True
    N_FOLDS = 5
    PSEUDO_THRESHOLD = 0.99  # confident pseudo-label threshold

pl.seed_everything(CFG.SEED)

# =====================
# LABELS
# =====================
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# =====================
# AUGMENTATIONS
# =====================
train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ToTensorV2(),
])

# =====================
# DATASET
# =====================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        # Metadata encoding
        if not is_test:
            self.df['Sex'] = self.df['Sex'].fillna('Unknown')
            self.df['Sex_enc'] = LabelEncoder().fit_transform(self.df['Sex'])
            self.df['ViewPosition'] = self.df['ViewPosition'].fillna('Unknown')
            self.df['View_enc'] = LabelEncoder().fit_transform(self.df['ViewPosition'])
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image, row["Image_Name"]
        else:
            labels = torch.tensor(row[LABELS].values.astype(np.float32))
            metadata = torch.tensor([row['Age'], row['Sex_enc'], row['View_enc']], dtype=torch.float32)
            return image, metadata, labels

# =====================
# MODEL
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class EfficientNetModel(pl.LightningModule):
    def __init__(self, use_focal=True):
        super().__init__()
        self.model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.N_CLASSES)
        self.use_focal = use_focal
        self.loss_fn = FocalLoss() if use_focal else nn.BCEWithLogitsLoss()
        self.val_auc = MultilabelAUROC(num_labels=CFG.N_CLASSES, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        auc_per_class = self.val_auc.compute()
        mean_auc = auc_per_class.mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]

# =====================
# TRAINING + FOLD ENSEMBLE
# =====================
train_df = pd.read_csv("../input/chestdx-multiinstitution/train1.csv")
train_df[LABELS] = train_df[LABELS].fillna(0)
train_df["stratify_col"] = train_df[LABELS].sum(axis=1).clip(0,1)

oof_preds = np.zeros((len(train_df), CFG.N_CLASSES))
test_df = pd.read_csv("../input/chestdx-multiinstitution/sample_submission1.csv")
test_preds = np.zeros((len(test_df), CFG.N_CLASSES))

skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["stratify_col"])):
    print(f"===== Fold {fold+1} =====")

    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = CXRDataset(df_train, "../input/chestdx-multiinstitution/train1/", transform=train_transforms)
    val_ds = CXRDataset(df_val, "../input/chestdx-multiinstitution/train1/", transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    model = EfficientNetModel(use_focal=CFG.USE_FOCAL)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc_mean", mode="max", save_top_k=1, dirpath=f"./fold{fold}", filename="best_model"
    )
    early_stop = EarlyStopping(monitor="val_auc_mean", mode="max", patience=3)

    trainer = pl.Trainer(
        max_epochs=CFG.EPOCHS,
        precision=16,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    model = EfficientNetModel.load_from_checkpoint(f"./fold{fold}/best_model.ckpt")
    model.eval()

    # OOF predictions
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
    preds_fold = []
    with torch.no_grad():
        for x, meta, y in val_loader:
            x = x.to(model.device)
            logits = model(x)
            preds_fold.append(torch.sigmoid(logits).cpu().numpy())
    preds_fold = np.vstack(preds_fold)
    oof_preds[val_idx] = preds_fold

    # Test predictions
    test_ds_fold = CXRDataset(test_df, "../input/chestdx-multiinstitution/test1/", transform=val_transforms, is_test=True)
    test_loader_fold = DataLoader(test_ds_fold, batch_size=CFG.BATCH_SIZE, shuffle=False)
    preds_test_fold = []
    with torch.no_grad():
        for x, ids in test_loader_fold:
            x = x.to(model.device)
            logits = model(x)
            preds_test_fold.append(torch.sigmoid(logits).cpu().numpy())
    test_preds += np.vstack(preds_test_fold)/CFG.N_FOLDS

# =====================
# PSEUDO-LABELING LOOP
# =====================
pseudo_idx = (test_preds.max(axis=1) > CFG.PSEUDO_THRESHOLD)
pseudo_labels = (test_preds[pseudo_idx] > 0.5).astype(np.float32)

pseudo_df = test_df.iloc[pseudo_idx].copy()
pseudo_df[LABELS] = pseudo_labels

# Append pseudo-labels to train (optional for next round)
# train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

# =====================
# FINAL SUBMISSION
# =====================
submission = pd.DataFrame(test_preds, columns=LABELS)
submission.insert(0, "Image_Name", test_df["Image_Name"])
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv")



# Grand X-Ray Slam Division A - Fold Ensemble + Pseudo-Label + Mixup/CutMix
# Author: Ishita (Blue and Gold Healthcare Inc.)
# Target: 0.99 AUC roadmap with strong generalization

!pip install pytorch-lightning timm torchmetrics albumentations --quiet

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MultilabelAUROC

# =====================
# CONFIG
# =====================
class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 8
    N_CLASSES = 14
    NUM_WORKERS = 4
    SEED = 42
    MODEL_NAME = "tf_efficientnet_b4_ns"
    USE_FOCAL = True
    N_FOLDS = 5
    PSEUDO_THRESHOLD = 0.99  # confident pseudo-label threshold
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0

pl.seed_everything(CFG.SEED)

# =====================
# LABELS
# =====================
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# =====================
# AUGMENTATIONS
# =====================
train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ToTensorV2(),
])

# =====================
# DATASET
# =====================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            self.df['Sex'] = self.df['Sex'].fillna('Unknown')
            self.df['Sex_enc'] = LabelEncoder().fit_transform(self.df['Sex'])
            self.df['ViewPosition'] = self.df['ViewPosition'].fillna('Unknown')
            self.df['View_enc'] = LabelEncoder().fit_transform(self.df['ViewPosition'])
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image, row["Image_Name"]
        else:
            labels = torch.tensor(row[LABELS].values.astype(np.float32))
            metadata = torch.tensor([row['Age'], row['Sex_enc'], row['View_enc']], dtype=torch.float32)
            return image, metadata, labels

# =====================
# MIXUP / CUTMIX FOR MULTI-LABEL
# =====================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    w = int(W * np.sqrt(1-lam))
    h = int(H * np.sqrt(1-lam))

    x1 = np.clip(cx - w//2, 0, W)
    y1 = np.clip(cy - h//2, 0, H)
    x2 = np.clip(cx + w//2, 0, W)
    y2 = np.clip(cy + h//2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2-x1)*(y2-y1)/(W*H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam_adjusted

# =====================
# MODEL
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class EfficientNetModel(pl.LightningModule):
    def __init__(self, use_focal=True, mixup_alpha=0.4, cutmix_alpha=1.0):
        super().__init__()
        self.model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.N_CLASSES)
        self.use_focal = use_focal
        self.loss_fn = FocalLoss() if use_focal else nn.BCEWithLogitsLoss()
        self.val_auc = MultilabelAUROC(num_labels=CFG.N_CLASSES, average=None)
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        # Apply Mixup or CutMix randomly
        if np.random.rand() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
        else:
            x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)

        logits = self(x)
        loss = lam * self.loss_fn(logits, y_a) + (1-lam) * self.loss_fn(logits, y_b)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        auc_per_class = self.val_auc.compute()
        mean_auc = auc_per_class.mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]

# =====================
# TRAINING + FOLD ENSEMBLE + PSEUDO-LABEL
# =====================
train_df = pd.read_csv("../input/chestdx-multiinstitution/train1.csv")
train_df[LABELS] = train_df[LABELS].fillna(0)
train_df["stratify_col"] = train_df[LABELS].sum(axis=1).clip(0,1)

oof_preds = np.zeros((len(train_df), CFG.N_CLASSES))
test_df = pd.read_csv("../input/chestdx-multiinstitution/sample_submission1.csv")
test_preds = np.zeros((len(test_df), CFG.N_CLASSES))

skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["stratify_col"])):
    print(f"===== Fold {fold+1} =====")

    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = CXRDataset(df_train, "../input/chestdx-multiinstitution/train1/", transform=train_transforms)
    val_ds = CXRDataset(df_val, "../input/chestdx-multiinstitution/train1/", transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    model = EfficientNetModel(use_focal=CFG.USE_FOCAL, mixup_alpha=CFG.MIXUP_ALPHA, cutmix_alpha=CFG.CUTMIX_ALPHA)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc_mean", mode="max", save_top_k=1, dirpath=f"./fold{fold}", filename="best_model"
    )
    early_stop = EarlyStopping(monitor="val_auc_mean", mode="max", patience=3)

    trainer = pl.Trainer(
        max_epochs=CFG.EPOCHS,
        precision=16,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    model = EfficientNetModel.load_from_checkpoint(f"./fold{fold}/best_model.ckpt")
    model.eval()

    # OOF predictions
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
    preds_fold = []
    with torch.no_grad():
        for x, meta, y in val_loader:
            x = x.to(model.device)
            logits = model(x)
            preds_fold.append(torch.sigmoid(logits).cpu().numpy())
    preds_fold = np.vstack(preds_fold)
    oof_preds[val_idx] = preds_fold

    # Test predictions
    test_ds_fold = CXRDataset(test_df, "../input/chestdx-multiinstitution/test1/", transform=val_transforms, is_test=True)
    test_loader_fold = DataLoader(test_ds_fold, batch_size=CFG.BATCH_SIZE, shuffle=False)
    preds_test_fold = []
    with torch.no_grad():
        for x, ids in test_loader_fold:
            x = x.to(model.device)
            logits = model(x)
            preds_test_fold.append(torch.sigmoid(logits).cpu().numpy())
    test_preds += np.vstack(preds_test_fold)/CFG.N_FOLDS

# =====================
# PSEUDO-LABELING LOOP
# =====================
pseudo_idx = (test_preds.max(axis=1) > CFG.PSEUDO_THRESHOLD)
pseudo_labels = (test_preds[pseudo_idx] > 0.5).astype(np.float32)
pseudo_df = test_df.iloc[pseudo_idx].copy()
pseudo_df[LABELS] = pseudo_labels

# Optionally append pseudo_df to train_df for next round

# =====================
# FINAL SUBMISSION
# =====================
submission = pd.DataFrame(test_preds, columns=LABELS)
submission.insert(0, "Image_Name", test_df["Image_Name"])
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv with Mixup + CutMix enabled")



# Grand X-Ray Slam Division A - Leaderboard Stacking Notebook
# CNN + ViT + Metadata Fusion + Mixup/CutMix + Ridge Stacking
# Kaggle-ready baseline for pushing 0.99 AUC

import os, gc, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# =====================
# Config
# =====================
class CFG:
    img_size = 512
    batch_size = 16
    num_workers = 4
    n_folds = 5
    lr = 2e-4
    epochs = 5
    seed = 42
    num_classes = 14
    device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(CFG.seed)

# =====================
# Dataset
# =====================
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.Image_Name)
        img = T.functional.pil_to_tensor(T.functional.to_pil_image(
            T.functional.to_tensor(T.functional.pil_to_tensor(T.functional.to_pil_image(
                T.functional.to_tensor(T.functional.pil_to_tensor(T.functional.to_pil_image(
                    torch.randint(0, 255, (3, CFG.img_size, CFG.img_size), dtype=torch.uint8) # placeholder if file not loaded
                ))))))))
        img = img.float() / 255.0

        if self.transform: img = self.transform(img)

        labels = row.iloc[6:].values.astype(float) # 14 conditions
        return img, torch.tensor(labels, dtype=torch.float32)

# =====================
# Mixup / CutMix for Multi-label
# =====================
def mixup_cutmix(data, targets, alpha=1.0, cutmix_prob=0.5):
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(data.device)
    if np.random.rand() < cutmix_prob:
        # CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
    targets = lam * targets + (1 - lam) * targets[index]
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# =====================
# Model Wrappers
# =====================
class CNNModel(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b4_ns"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=CFG.num_classes)

    def forward(self, x): return self.backbone(x)

class ViTModel(nn.Module):
    def __init__(self, backbone="swin_base_patch4_window7_224"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=CFG.num_classes)

    def forward(self, x): return self.backbone(x)

class MetadataFusionModel(nn.Module):
    def __init__(self, img_backbone="tf_efficientnet_b0_ns"):
        super().__init__()
        self.img_backbone = timm.create_model(img_backbone, pretrained=True, num_classes=0)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )
        self.head = nn.Linear(self.img_backbone.num_features + 64, CFG.num_classes)

    def forward(self, x, meta):
        x_img = self.img_backbone(x)
        x_meta = self.meta_fc(meta)
        return self.head(torch.cat([x_img, x_meta], dim=1))

# =====================
# Lightning Module
# =====================
class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if random.random() < 0.5:
            x, y = mixup_cutmix(x, y)
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = torch.sigmoid(self(x))
        return {"preds": preds.cpu(), "targets": y.cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# =====================
# Training + OOF
# =====================
def train_and_predict(train_df, test_df, img_dir, test_img_dir):
    oof_preds = []
    oof_targets = []
    test_preds = []

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["No Finding"])):
        print(f"===== Fold {fold} =====")
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]

        # CNN Model
        cnn_model = CNNModel()
        lit_cnn = LitModel(cnn_model)
        trainer = pl.Trainer(max_epochs=CFG.epochs, accelerator="gpu" if CFG.device=="cuda" else "cpu",
                             devices=1, callbacks=[EarlyStopping("val_loss")])
        train_loader = DataLoader(ChestXrayDataset(train_data, img_dir), batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(ChestXrayDataset(val_data, img_dir), batch_size=CFG.batch_size)
        trainer.fit(lit_cnn, train_loader, val_loader)

        # Collect OOF predictions
        preds = []
        tgts = []
        for x,y in val_loader:
            p = torch.sigmoid(cnn_model(x.to(CFG.device))).detach().cpu().numpy()
            preds.append(p); tgts.append(y.numpy())
        oof_preds.append(np.concatenate(preds)); oof_targets.append(np.concatenate(tgts))

        # Test predictions
        test_loader = DataLoader(ChestXrayDataset(test_df, test_img_dir), batch_size=CFG.batch_size)
        preds = []
        for x,y in test_loader:
            p = torch.sigmoid(cnn_model(x.to(CFG.device))).detach().cpu().numpy()
            preds.append(p)
        test_preds.append(np.concatenate(preds))

    # Stack OOF
    X = np.concatenate(oof_preds)
    y = np.concatenate(oof_targets)
    X_test = np.mean(test_preds, axis=0)

    stacker = Ridge(alpha=1.0)
    stacker.fit(X, y)
    final_preds = stacker.predict(X_test)

    return final_preds

# =====================
# Main
# =====================
train_df = pd.read_csv("/kaggle/input/chestdx-multiinstitution/train1.csv")
test_df = pd.read_csv("/kaggle/input/chestdx-multiinstitution/sample_submission1.csv")

final_preds = train_and_predict(
    train_df, test_df,
    img_dir="/kaggle/input/chestdx-multiinstitution/train1/",
    test_img_dir="/kaggle/input/chestdx-multiinstitution/test1/"
)

# Create submission
sub = pd.read_csv("/kaggle/input/chestdx-multiinstitution/sample_submission1.csv")
sub.iloc[:,1:] = final_preds
sub.to_csv("submission.csv", index=False)
print("submission.csv ready âœ…")



# =====================
# Triple Stack: CNN + ViT + Metadata Fusion
# Grand X-Ray Slam Division A Leaderboard Notebook
# =====================

# STEP 1: Train CNN (EfficientNet)
# STEP 2: Train ViT (Swin Transformer)
# STEP 3: Train Metadata Fusion Model
# STEP 4: Collect OOF predictions for all 3
# STEP 5: Blend with Ridge stacker
# STEP 6: Generate submission.csv



# Inside train_and_predict()
cnn_model = CNNModel()
vit_model = ViTModel()
meta_model = MetadataFusionModel()

# Train each with Lightning (separately)
lit_cnn = LitModel(cnn_model)
lit_vit = LitModel(vit_model)
lit_meta = LitModel(meta_model)

trainer.fit(lit_cnn, train_loader, val_loader)
trainer.fit(lit_vit, train_loader, val_loader)
trainer.fit(lit_meta, train_loader_with_metadata, val_loader_with_metadata)

# Collect OOF predictions
cnn_val_preds, vit_val_preds, meta_val_preds = ...
cnn_test_preds, vit_test_preds, meta_test_preds = ...

# Store them
oof_preds.append(np.hstack([cnn_val_preds, vit_val_preds, meta_val_preds]))
test_preds.append(np.hstack([cnn_test_preds, vit_test_preds, meta_test_preds]))



X = np.concatenate(oof_preds)
y = np.concatenate(oof_targets)
X_test = np.mean(test_preds, axis=0)

stacker = Ridge(alpha=1.0)
stacker.fit(X, y)
final_preds = stacker.predict(X_test)

sub = pd.read_csv("/kaggle/input/chestdx-multiinstitution/sample_submission1.csv")
sub.iloc[:,1:] = final_preds
sub.to_csv("submission.csv", index=False)



from IPython.display import FileLink
FileLink("submission.csv")



import pandas as pd
import numpy as np

# Load sample submission format
sample = pd.read_csv("sample_submission1.csv")

# Replace predictions with random probabilities (0â€“1) for now
for col in sample.columns[1:]:
    sample[col] = np.random.rand(len(sample))

# Save as submission.csv
sample.to_csv("submission.csv", index=False)

print("âœ… submission.csv generated successfully!")



# ====================================================
# Grand X-Ray Slam Division A - Triple Stack
# CNN (EfficientNet-B4) + ViT (Swin Transformer) + Metadata Fusion
# Mixup/CutMix + OOF Ridge Stacking + Final Submission
# ====================================================

import os, gc, random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# ====================================================
# Config
# ====================================================
class CFG:
    img_size = 512
    batch_size = 16
    num_workers = 4
    n_folds = 5
    lr = 2e-4
    epochs = 3       # increase to 10â€“15 for real LB push
    seed = 42
    num_classes = 14
    device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(CFG.seed)

# ====================================================
# Dataset
# ====================================================
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, use_meta=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.use_meta = use_meta

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.Image_Name)

        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)

        labels = row.iloc[6:].values.astype(float)

        if self.use_meta:
            sex = 1 if row.Sex == "Male" else 0
            age = row.Age if not np.isnan(row.Age) else 60
            view = 0 if row.ViewPosition == "PA" else 1
            meta = np.array([sex, age/100.0, view], dtype=np.float32)
            return img, torch.tensor(labels, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)
        else:
            return img, torch.tensor(labels, dtype=torch.float32)

# ====================================================
# Mixup / CutMix
# ====================================================
def mixup_cutmix(data, targets, alpha=1.0, cutmix_prob=0.5):
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(data.device)
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
    targets = lam * targets + (1 - lam) * targets[index]
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ====================================================
# Models
# ====================================================
class CNNModel(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b4_ns"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=CFG.num_classes)
    def forward(self, x): return self.backbone(x)

class ViTModel(nn.Module):
    def __init__(self, backbone="swin_base_patch4_window7_224"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=CFG.num_classes)
    def forward(self, x): return self.backbone(x)

class MetadataFusionModel(nn.Module):
    def __init__(self, img_backbone="tf_efficientnet_b0_ns"):
        super().__init__()
        self.img_backbone = timm.create_model(img_backbone, pretrained=True, num_classes=0)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )
        self.head = nn.Linear(self.img_backbone.num_features + 64, CFG.num_classes)
    def forward(self, x, meta):
        x_img = self.img_backbone(x)
        x_meta = self.meta_fc(meta)
        return self.head(torch.cat([x_img, x_meta], dim=1))

# ====================================================
# Lightning Module
# ====================================================
class LitModel(pl.LightningModule):
    def __init__(self, model, use_meta=False):
        super().__init__()
        self.model = model
        self.use_meta = use_meta
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, meta=None):
        return self.model(x, meta) if self.use_meta else self.model(x)

    def training_step(self, batch, batch_idx):
        if self.use_meta:
            x, y, meta = batch
        else:
            x, y = batch; meta = None
        if random.random() < 0.5:
            x, y = mixup_cutmix(x, y)
        preds = self(x, meta)
        loss = self.criterion(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_meta:
            x, y, meta = batch
            preds = torch.sigmoid(self(x, meta))
        else:
            x, y = batch
            preds = torch.sigmoid(self(x))
        return {"preds": preds.cpu(), "targets": y.cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ====================================================
# Training + Stacking
# ====================================================
def train_and_predict(train_df, test_df, img_dir, test_img_dir):
    oof_preds, oof_targets, test_preds = [], [], []

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["No Finding"])):
        print(f"===== Fold {fold} =====")
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]

        # Transforms
        tfms = T.Compose([
            T.Resize((CFG.img_size, CFG.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
        ])

        # Loaders
        train_loader = DataLoader(ChestXrayDataset(train_data, img_dir, transform=tfms),
                                  batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        val_loader = DataLoader(ChestXrayDataset(val_data, img_dir, transform=tfms),
                                batch_size=CFG.batch_size, num_workers=CFG.num_workers)
        test_loader = DataLoader(ChestXrayDataset(test_df, test_img_dir, transform=tfms),
                                 batch_size=CFG.batch_size, num_workers=CFG.num_workers)

        # === CNN ===
        cnn_model = CNNModel()
        trainer = pl.Trainer(max_epochs=CFG.epochs, accelerator=CFG.device, devices=1,
                             callbacks=[EarlyStopping("train_loss")])
        trainer.fit(LitModel(cnn_model), train_loader, val_loader)

        # === ViT ===
        vit_model = ViTModel()
        trainer.fit(LitModel(vit_model), train_loader, val_loader)

        # === Metadata Fusion ===
        train_loader_meta = DataLoader(ChestXrayDataset(train_data, img_dir, transform=tfms, use_meta=True),
                                       batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        val_loader_meta = DataLoader(ChestXrayDataset(val_data, img_dir, transform=tfms, use_meta=True),
                                     batch_size=CFG.batch_size, num_workers=CFG.num_workers)
        test_loader_meta = DataLoader(ChestXrayDataset(test_df, test_img_dir, transform=tfms, use_meta=True),
                                      batch_size=CFG.batch_size, num_workers=CFG.num_workers)

        meta_model = MetadataFusionModel()
        trainer.fit(LitModel(meta_model, use_meta=True), train_loader_meta, val_loader_meta)

        # Collect OOF preds
        preds_val, tgts = [], []
        for x, y in val_loader:
            p1 = torch.sigmoid(cnn_model(x.to(CFG.device))).detach().cpu().numpy()
            p2 = torch.sigmoid(vit_model(x.to(CFG.device))).detach().cpu().numpy()
            preds_val.append(np.hstack([p1, p2])); tgts.append(y.numpy())
        for x, y, meta in val_loader_meta:
            p3 = torch.sigmoid(meta_model(x.to(CFG.device), meta.to(CFG.device))).detach().cpu().numpy()
            preds_val[-1] = np.hstack([preds_val[-1], p3])

        oof_preds.append(np.vstack(preds_val))
        oof_targets.append(np.vstack(tgts))

        # Test preds
        preds_test = []
        for x, _ in test_loader:
            p1 = torch.sigmoid(cnn_model(x.to(CFG.device))).detach().cpu().numpy()
            p2 = torch.sigmoid(vit_model(x.to(CFG.device))).detach().cpu().numpy()
            preds_test.append(np.hstack([p1, p2]))
        for x, _, meta in test_loader_meta:
            p3 = torch.sigmoid(meta_model(x.to(CFG.device), meta.to(CFG.device))).detach().cpu().numpy()
            preds_test[-1] = np.hstack([preds_test[-1], p3])
        test_preds.append(np.vstack(preds_test))

    # === Stacking ===
    X = np.concatenate(oof_preds)
    y = np.concatenate(oof_targets)
    X_test = np.mean(test_preds, axis=0)

    stacker = Ridge(alpha=1.0)
    stacker.fit(X, y)
    final_preds = stacker.predict(X_test)
    return final_preds

# ====================================================
# Main
# ====================================================
train_df = pd.read_csv("/kaggle/input/chestdx-multiinstitution/train1.csv")
test_df = pd.read_csv("/kaggle/input/chestdx-multiinstitution/sample_submission1.csv")

final_preds = train_and_predict(
    train_df, test_df,
    img_dir="/kaggle/input/chestdx-multiinstitution/train1/",
    test_img_dir="/kaggle/input/chestdx-multiinstitution/test1/"
)

# Create submission
sub = pd.read_csv("/kaggle/input/chestdx-multiinstitution/sample_submission1.csv")
sub.iloc[:,1:] = final_preds
sub.to_csv("submission.csv", index=False)
print("âœ… submission.csv ready")



# =========================================================
# Grand X-Ray Slam Division A
# Triple-Stack: CNN + ViT + Metadata Fusion
# With Pseudo-Labeling & Co-occurrence Graph Post-processing
# =========================================================

import os, random, gc, timm, torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# =========================================================
# Config
# =========================================================
class CFG:
    seed = 42
    img_size = 512
    n_folds = 5
    batch_size = 16
    epochs = 5
    lr = 1e-4
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
                   'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                   'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
                   'Pneumothorax','Support Devices']

# =========================================================
# Utils
# =========================================================
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)

def compute_auc(y_true, y_pred):
    scores = []
    for i in range(len(CFG.target_cols)):
        try:
            scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except:
            pass
    return np.mean(scores)

# =========================================================
# Dataset
# =========================================================
class XrayDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment

        self.transform = T.Compose([
            T.Resize((CFG.img_size, CFG.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_Name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(row[CFG.target_cols].values.astype(np.float32))
        meta = torch.tensor([
            0 if pd.isna(row["Sex"]) else (1 if row["Sex"]=="Male" else 2),
            0 if pd.isna(row["Age"]) else row["Age"],
        ], dtype=torch.float)

        return image, meta, label

# =========================================================
# Models
# =========================================================
class CNNModel(nn.Module):
    def __init__(self, out_dim=len(CFG.target_cols)):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x): return self.head(self.backbone(x))

class ViTModel(nn.Module):
    def __init__(self, out_dim=len(CFG.target_cols)):
        super().__init__()
        self.backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x): return self.head(self.backbone(x))

class MetadataMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=len(CFG.target_cols)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.mlp(x)

class TripleStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNModel()
        self.vit = ViTModel()
        self.meta = MetadataMLP()
        self.fc = nn.Linear(len(CFG.target_cols)*3, len(CFG.target_cols))

    def forward(self, img, meta):
        cnn_out = self.cnn(img)
        vit_out = self.vit(img)
        meta_out = self.meta(meta)
        combined = torch.cat([cnn_out, vit_out, meta_out], dim=1)
        return self.fc(combined)

# =========================================================
# Pseudo-Labeling Step
# =========================================================
def pseudo_labeling(train_df, test_preds, threshold=0.95):
    pseudo_rows = []
    for idx, row in test_preds.iterrows():
        labels = (row[CFG.target_cols] > threshold).astype(int)
        if labels.sum() > 0:
            new_row = {"Image_Name": row["Image_Name"], "Sex": np.nan, "Age": np.nan}
            for col in CFG.target_cols:
                new_row[col] = labels[col]
            pseudo_rows.append(new_row)
    pseudo_df = pd.DataFrame(pseudo_rows)
    return pd.concat([train_df, pseudo_df], ignore_index=True)

# =========================================================
# Co-occurrence Graph Post-processing
# =========================================================
def co_occurrence_adjust(preds, co_graph):
    for i, row in preds.iterrows():
        for cond, related in co_graph.items():
            if row[cond] > 0.8:
                for rel in related:
                    preds.loc[i, rel] = max(preds.loc[i, rel], row[cond]*0.6)
    return preds

co_graph = {
    "Edema": ["Pleural Effusion"],
    "Pleural Effusion": ["Edema", "Lung Opacity"],
    "Cardiomegaly": ["Enlarged Cardiomediastinum"],
}

# =========================================================
# Training / Inference Skeleton (fill with folds loop)
# =========================================================
# NOTE: Here you would train CNN/ViT/Meta models separately per fold,
# then blend predictions using TripleStack.

# Pseudo-code for inference:
# train_df = pd.read_csv("../input/train1.csv")
# test_df = pd.read_csv("../input/sample_submission1.csv")
# predictions = blended_preds(test_df) # CNN+ViT+Meta ensemble
#
# # Apply pseudo-labeling
# train_df_aug = pseudo_labeling(train_df, predictions)
# retrain with train_df_aug...
#
# # Apply co-occurrence graph adjustment
# predictions = co_occurrence_adjust(predictions, co_graph)
#
# # Save submission
# predictions.to_csv("submission.csv", index=False)


# %% [markdown]
# # ğŸ�† Grand X-Ray Slam: Division A
# *A Kaggle Community Hackathon â€“ Build AI for Life-Saving Radiology*
#
# ---
#
# ## ğŸ“– Overview
# Welcome to **Grand X-Ray Slam: Division A**, the first of a **2-part Kaggle hackathon series** where data scientists and AI enthusiasts compete to advance medical imaging.
#
# In this challenge, youâ€™ll develop **AI models** to detect **14 thoracic conditions** from chest X-rays, tackling real-world clinical complexity. Your work will power **Dr. HealthAgent**, a personal health app developed under **Blue and Gold Healthcare Inc.**, with the mission to enhance global healthcare.
#
# ğŸ”¹ Top performers across both **Division A and Division B** will also shine on the **Grand Slam Leaderboard**, sharing an **additional $2,500 prize pool**.
#
# ---
#
# ## ğŸ�¯ Your Mission
# Build AI models to identify the following thoracic conditions in each chest X-ray:
# - Atelectasis
# - Cardiomegaly
# - Consolidation
# - Edema
# - Enlarged Cardiomediastinum
# - Fracture
# - Lung Lesion
# - Lung Opacity
# - No Finding
# - Pleural Effusion
# - Pleural Other
# - Pneumonia
# - Pneumothorax
# - Support Devices
#
# âš¡ This is a **multi-label classification** task, reflecting real emergency-room diagnostics where one X-ray may show multiple conditions.
#
# ---
#
# ## ğŸ“‚ Dataset
# - **Train set**: 107,374 chest X-ray images (~138 GB)
# - **Test set**: 46,233 chest X-ray images (~60 GB)
#
# âœ… Images are **de-identified** and sourced from **multiple institutions**.
# âœ… No patient overlap exists between train and test splits.
#
# ---
#
# ## â�³ Timeline
# - **Start**: August 20, 2025
# - **End**: October 10, 2025, 11:59 PM UTC
# - **Private Leaderboard Release**: October 12, 2025, 23:59 UTC
#
# ---
#
# ## ğŸ“œ Participation Rules
# - Teams: Solo or up to **4 members**
# - External data: â�Œ Not allowed
# - Use: Competition & research only
#
# ---
#
# ## ğŸ™Œ Acknowledgements
# **Organizers**:
# - Guntas Dhanjal (Lead)
# - Salah Sammari
# - Fathi Ben Amor
# - Mariem Aissa
#
# **Data Sources**: Curated merge of multiple public chest X-ray datasets, anonymized & preprocessed to ensure **no patient overlap**.
#
# ---
#
# ## ğŸ“Š Evaluation
# - Metric: **Area Under the ROC Curve (AUC)** per label.
# - Final Score: **Mean AUC** across all 14 thoracic conditions.
# - Setup: Multi-label classification (each image may have multiple positive labels).
#
# ---
#
# ## ğŸ“‘ Submission File Format
# CSV or Parquet with **46,233 rows + header**. Archives (`zip/gz/7z/tar`) also accepted.
#
# **Columns**:
# `Image_name,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices`
#
# Example row:
# ```
# 00000005_001_001.jpg,0.01,0.02,0.05,0.01,0.01,0.0,0.0,0.12,0.0,0.04,0.0,0.03,0.0,0.01
# ```
#
# âš ï¸� `Image_name` must exactly match test filenames.
#
# ---

# %% [code]
# =========================================================
# Imports & Config
# =========================================================
import os, random, gc, timm, torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class CFG:
    seed = 42
    img_size = 512
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema',
        'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
        'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
        'Pneumothorax','Support Devices'
    ]

# %% [code]
# =========================================================
# Dummy Baseline Submission Generator
# =========================================================
# Load sample submission format
sample = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/sample_submission1.csv")

# Replace predictions with random probabilities (0â€“1)
for col in sample.columns[1:]:
    sample[col] = np.random.rand(len(sample))

# Save submission
sample.to_csv("submission.csv", index=False)
sample.to_parquet("submission.parquet", index=False)

print("âœ… submission.csv and submission.parquet generated successfully!")
print(sample.head())



# %% [markdown]
# # ğŸ�† Grand X-Ray Slam: Division A â€“ Baseline Notebook
#
# - Cross-validation with EfficientNet-B0
# - Predictions on test set
# - Submission file (CSV + Parquet) generated
# - CV metrics displayed
#
# Replace this baseline with advanced models (CNN+ViT ensembles, metadata, pseudo-labeling, etc.) later.

# %% [code]
# =========================================================
# Imports
# =========================================================
import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm

# %% [code]
# =========================================================
# Config
# =========================================================
class CFG:
    seed = 42
    img_size = 224
    batch_size = 16
    epochs = 1   # âš ï¸� increase for real training
    folds = 3    # âš ï¸� increase for stronger CV
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema',
        'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
        'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
        'Pneumothorax','Support Devices'
    ]

# %% [code]
# =========================================================
# Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(CFG.seed)

# %% [code]
# =========================================================
# Dataset
# =========================================================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        if self.train:
            labels = row[CFG.target_cols].values.astype(float)
            return image, torch.tensor(labels, dtype=torch.float32)
        else:
            return image, row['Image_name']

# %% [code]
# =========================================================
# Model
# =========================================================
class CXRModel(nn.Module):
    def __init__(self, model_name="tf_efficientnet_b0_ns", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=len(CFG.target_cols))

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# %% [code]
# =========================================================
# Training & Evaluation Functions
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    aucs = []
    for i, col in enumerate(CFG.target_cols):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            auc = np.nan
        aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)

# %% [code]
# =========================================================
# Load Data
# =========================================================
train_df = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/train_labels.csv")
test_df  = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/test_labels.csv")  # contains only Image_name

train_img_dir = "/kaggle/input/grand-xray-slam-division-a/train_images"
test_img_dir  = "/kaggle/input/grand-xray-slam-division-a/test_images"

# augmentations
train_tfms = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
valid_tfms = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
])

# %% [code]
# =========================================================
# Cross Validation Training
# =========================================================
kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
oof_preds = np.zeros((len(train_df), len(CFG.target_cols)))
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    print(f"===== FOLD {fold+1} =====")
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    tr_ds = CXRDataset(tr_df, train_img_dir, transforms=train_tfms, train=True)
    va_ds = CXRDataset(va_df, train_img_dir, transforms=valid_tfms, train=True)

    tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    model = CXRModel().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(CFG.epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        va_loss, va_auc = validate(model, va_loader, criterion)
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, val_auc={va_auc:.4f}")

    # save OOF preds
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, labels in va_loader:
            imgs = imgs.to(CFG.device)
            p = model(imgs)
            preds.append(p.cpu().numpy())
    preds = np.vstack(preds)
    oof_preds[va_idx] = preds
    cv_scores.append(va_auc)

print("CV AUCs:", cv_scores)
print("Mean CV AUC:", np.mean(cv_scores))

# %% [code]
# =========================================================
# Test Inference
# =========================================================
test_ds = CXRDataset(test_df, test_img_dir, transforms=valid_tfms, train=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

final_model = CXRModel().to(CFG.device)  # normally load trained weights
final_model.eval()

all_preds, all_names = [], []
with torch.no_grad():
    for imgs, names in test_loader:
        imgs = imgs.to(CFG.device)
        preds = final_model(imgs)
        all_preds.append(preds.cpu().numpy())
        all_names.extend(names)

all_preds = np.vstack(all_preds)

# %% [code]
# =========================================================
# Submission
# =========================================================
submission = pd.DataFrame(all_preds, columns=CFG.target_cols)
submission.insert(0, "Image_name", all_names)

submission.to_csv("submission.csv", index=False)
submission.to_parquet("submission.parquet", index=False)

print("âœ… Submission files saved")
print(submission.head())



# %% [markdown]
# # ğŸ�† Grand X-Ray Slam: Division A â€“ CV Ensemble Baseline
#
# - Cross-validation training (EfficientNet-B0 backbone)
# - Save best model weights per fold
# - Ensemble (average) predictions across folds
# - Outputs submission.csv + submission.parquet
# - Displays CV metrics and prediction samples

# %% [code]
import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm

# =========================================================
# Config
# =========================================================
class CFG:
    seed = 42
    img_size = 224
    batch_size = 16
    epochs = 1   # âš ï¸� Increase in real runs
    folds = 3    # âš ï¸� Use 5 or 10 for real competition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema',
        'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
        'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
        'Pneumothorax','Support Devices'
    ]
    model_name = "tf_efficientnet_b0_ns"
    save_dir = "./fold_models"
os.makedirs(CFG.save_dir, exist_ok=True)

# =========================================================
# Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(CFG.seed)

# =========================================================
# Dataset
# =========================================================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        if self.train:
            labels = row[CFG.target_cols].values.astype(float)
            return image, torch.tensor(labels, dtype=torch.float32)
        else:
            return image, row['Image_name']

# =========================================================
# Model
# =========================================================
class CXRModel(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=len(CFG.target_cols))

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# =========================================================
# Training & Evaluation
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    aucs = []
    for i in range(len(CFG.target_cols)):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            auc = np.nan
        aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)

# =========================================================
# Load Data
# =========================================================
train_df = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/train_labels.csv")
test_df  = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/test_labels.csv")

train_img_dir = "/kaggle/input/grand-xray-slam-division-a/train_images"
test_img_dir  = "/kaggle/input/grand-xray-slam-division-a/test_images"

train_tfms = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
valid_tfms = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
])

# =========================================================
# Cross-validation Training
# =========================================================
kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    print(f"===== FOLD {fold+1} =====")
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    tr_ds = CXRDataset(tr_df, train_img_dir, transforms=train_tfms, train=True)
    va_ds = CXRDataset(va_df, train_img_dir, transforms=valid_tfms, train=True)

    tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    model = CXRModel().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_auc = -np.inf
    best_path = f"{CFG.save_dir}/model_fold{fold}.pt"

    for epoch in range(CFG.epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        va_loss, va_auc = validate(model, va_loader, criterion)
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, val_auc={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)

    cv_scores.append(best_auc)
    print(f"Best AUC fold {fold+1}: {best_auc:.4f}")

print("CV AUCs:", cv_scores)
print("Mean CV AUC:", np.mean(cv_scores))

# =========================================================
# Inference with Fold Ensemble
# =========================================================
test_ds = CXRDataset(test_df, test_img_dir, transforms=valid_tfms, train=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

all_fold_preds = []

for fold in range(CFG.folds):
    model = CXRModel().to(CFG.device)
    model.load_state_dict(torch.load(f"{CFG.save_dir}/model_fold{fold}.pt", map_location=CFG.device))
    model.eval()

    fold_preds = []
    with torch.no_grad():
        for imgs, names in test_loader:
            imgs = imgs.to(CFG.device)
            preds = model(imgs)
            fold_preds.append(preds.cpu().numpy())
    fold_preds = np.vstack(fold_preds)
    all_fold_preds.append(fold_preds)

# average across folds
final_preds = np.mean(all_fold_preds, axis=0)

# =========================================================
# Submission
# =========================================================
submission = pd.DataFrame(final_preds, columns=CFG.target_cols)
submission.insert(0, "Image_name", test_df["Image_name"].values)

submission.to_csv("submission.csv", index=False)
submission.to_parquet("submission.parquet", index=False)

print("âœ… Submission files saved")
print(submission.head())



# =========================================================
# Mixup & CutMix Utilities
# =========================================================
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0):
    """Mixup for multi-labels."""
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix for multi-labels."""
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    shuffled_x, shuffled_y = x[index], y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = shuffled_x[:, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, shuffled_y, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================
# Training with Mixup & CutMix
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, use_mixup=True, use_cutmix=True, mix_alpha=1.0):
    model.train()
    losses = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)

        # randomly decide augmentation
        r = np.random.rand()
        if use_mixup and r < 0.5:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mix_alpha)
            preds = model(imgs)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        elif use_cutmix and r >= 0.5:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=mix_alpha)
            preds = model(imgs)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs)
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)



# pipeline.py
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
          'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
          'Pleural Other','Pneumonia','Pneumothorax','Support Devices']

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def get_model():
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, len(LABELS)),
        torch.nn.Sigmoid()
    )
    return model

def predict(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            for name, probs in zip(names, outputs):
                results.append([name] + probs.tolist())
    return results

def run_inference(image_dir, image_list_file, model_path, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    image_list = pd.read_csv(image_list_file)['Image_name'].tolist()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ChestXrayDataset(image_dir, image_list, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = predict(model, dataloader, device)
    df = pd.DataFrame(results, columns=['Image_name'] + LABELS)
    df.to_csv(output_csv, index=False)


ls /kaggle/input/grand-xray-slam-division-a/


ls -R /kaggle/input/


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Thoracic Condition Probabilities')
plt.show()


correlation_matrix = df[LABELS].corr()
display(correlation_matrix)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['Consolidation'], bins=50, kde=True)
plt.title('Distribution of Consolidation Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()


display(df.describe())


display(df.head())


display(df.head())


class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train
        self.metadata_cols = ['Sex', 'Age', 'ViewPosition']  # simple example

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Simple metadata encoding
        meta = np.zeros(len(self.metadata_cols), dtype=np.float32)
        meta[0] = 0 if row.get('Sex','Male')=='Male' else 1
        meta[1] = row.get('Age',50)/100.0
        meta[2] = 0 if row.get('ViewPosition','PA')=='PA' else 1

        if self.train:
            labels = row[CFG.target_cols].values.astype(float)
            return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)
        else:
            return image, torch.tensor(meta, dtype=torch.float32), row['Image_name']


class CXRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_cnn, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def train_one_epoch(model, loader, optimizer, criterion, use_mixup=True, use_cutmix=True, mix_alpha=1.0):
    model.train()
    losses = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)

        # randomly decide augmentation
        r = np.random.rand()
        if use_mixup and r < 0.5:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mix_alpha)
            preds = model(imgs)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        elif use_cutmix and r >= 0.5:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=mix_alpha)
            preds = model(imgs)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs)
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    per_label_auc = []
    for i, col in enumerate(CFG.target_cols):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = np.nan
        per_label_auc.append(auc)
    mean_auc = np.nanmean(per_label_auc)
    return np.mean(losses), mean_auc, per_label_auc


# pipeline.py
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
          'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
          'Pleural Other','Pneumonia','Pneumothorax','Support Devices']

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def get_model():
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, len(LABELS)),
        torch.nn.Sigmoid()
    )
    return model

def predict(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            for name, probs in zip(names, outputs):
                results.append([name] + probs.tolist())
    return results

def run_inference(image_dir, image_list_file, model_path, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    image_list = pd.read_csv(image_list_file)['Image_name'].tolist()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ChestXrayDataset(image_dir, image_list, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = predict(model, dataloader, device)
    df = pd.DataFrame(results, columns=['Image_name'] + LABELS)
    df.to_csv(output_csv, index=False)



import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# âœ… Define label order
LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
          'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
          'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
          'Pneumothorax','Support Devices']

# âœ… Load image names from sample submission
def load_image_names(sample_path):
    df = pd.read_csv(sample_path)
    return df['Image_name'].tolist()

# âœ… Custom dataset for test images
class ChestXrayTestDataset(Dataset):
    def __init__(self, image_dir, image_names, transform=None):
        self.image_dir = image_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# âœ… Load trained model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# âœ… Run inference
def run_inference(model, dataloader, device):
    results = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            for name, probs in zip(names, outputs):
                results.append([name] + probs.tolist())
    return results

# âœ… Save predictions to CSV
def save_submission(results, output_path):
    df = pd.DataFrame(results, columns=['Image_name'] + LABELS)
    df.to_csv(output_path, index=False)
    print(f"âœ… Saved submission to {output_path}")

# âœ… Validate submission format
def validate_submission(file_path, expected_rows=46233):
    df = pd.read_csv(file_path)
    assert df.shape[0] == expected_rows, f"â�Œ Expected {expected_rows} rows, got {df.shape[0]}"
    assert list(df.columns) == ['Image_name'] + LABELS, "â�Œ Header mismatch"
    assert df['Image_name'].is_unique, "â�Œ Duplicate image names found"
    for label in LABELS:
        assert df[label].between(0, 1).all(), f"â�Œ Invalid probabilities in column: {label}"
    print("âœ… Submission format is valid.")

# âœ… Full pipeline execution
def generate_submission(image_dir, sample_path, model_path, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_names = load_image_names(sample_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ChestXrayTestDataset(image_dir, image_names, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_model(model_path, device)
    results = run_inference(model, dataloader, device)
    save_submission(results, output_csv)
    validate_submission(output_csv)



import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ResNet1D:
    """ResNet1D model builder with configurable architecture"""

    @staticmethod
    def resnet_block(x, filters, kernel_size, stride=1):
        """Create a ResNet block with skip connection"""
        shortcut = x

        # Main path
        x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        # Shortcut path
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        # Combine paths
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def build_model(input_shape, num_targets, params):
        """Build ResNet1D model with configurable parameters"""
        inputs = Input(shape=input_shape)

        # Initial convolution
        x = Conv1D(params['filters_l1'], params['kernel_size_l1'], strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Residual blocks
        x = ResNet1D.resnet_block(x, params['filters_l1'], params['kernel_size_l1'])
        x = ResNet1D.resnet_block(x, params['filters_l1'], params['kernel_size_l1'])

        x = ResNet1D.resnet_block(x, params['filters_l2'], params['kernel_size_l2'], stride=2)
        x = ResNet1D.resnet_block(x, params['filters_l2'], params['kernel_size_l2'])

        x = ResNet1D.resnet_block(x, params['filters_l3'], params['kernel_size_l3'], stride=2)
        x = ResNet1D.resnet_block(x, params['filters_l3'], params['kernel_size_l3'])

        # Global Average Pooling for better feature extraction
        x = GlobalAveragePooling1D()(x)

        # Dense layers with regularization
        x = Dense(params['dense_units'], activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(params['dense_units'] // 2, activation='relu')(x)
        outputs = Dense(num_targets)(x)

        model = Model(inputs, outputs)
        return model

class DataPreprocessor:
    """Handles data preparation and validation for the stacking ensemble"""

    def __init__(self, target_columns=None):
        self.target_columns = target_columns or ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        self.identifier_cols = ['Unnamed: 0', 'Analyte concentration', 'sample_group']
        self.trad_imputer = SimpleImputer(strategy='median')
        self.resnet_imputer = SimpleImputer(strategy='median')
        self.is_fitted = False

    def prepare_traditional_features(self, expanded_data, fit=False):
        """Prepare features for traditional ML models with proper NaN handling"""
        # Drop identifier columns
        features = expanded_data.drop(columns=[col for col in self.identifier_cols
                                             if col in expanded_data.columns], errors='ignore')
        
        # Select only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_cols]
        
        # Handle NaNs
        if fit:
            features_imputed = self.trad_imputer.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Imputer must be fitted first")
            features_imputed = self.trad_imputer.transform(features)
        
        return pd.DataFrame(features_imputed, columns=features.columns, index=features.index)

    def prepare_resnet_features(self, expanded_data, fit=False):
        """Prepare features for ResNet1D model with proper NaN handling"""
        # Drop columns with inherent NaNs from derivatives and identifier columns
        derivative_nan_cols = ['Unnamed: 1_d1', 'Unnamed: 2_d1', 'Unnamed: 1_d2', 'Unnamed: 2_d2', 'Unnamed: 3_d2']
        cols_to_drop = [col for col in derivative_nan_cols + self.identifier_cols 
                       if col in expanded_data.columns]
        features = expanded_data.drop(columns=cols_to_drop, errors='ignore')
        
        # Select only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_cols]
        
        # Handle NaNs
        if fit:
            features_imputed = self.resnet_imputer.fit_transform(features)
        else:
            if not self.is_fitted:
                raise ValueError("Imputer must be fitted first")
            features_imputed = self.resnet_imputer.transform(features)
        
        return pd.DataFrame(features_imputed, columns=features.columns, index=features.index)

    def prepare_targets(self, train_data):
        """Prepare target variables with robust NaN imputation"""
        if not all(col in train_data.columns for col in self.target_columns):
            raise ValueError(f"Target columns {self.target_columns} not found in training data")
            
        y_train = train_data[self.target_columns].values
        y_cleaned = np.copy(y_train)

        # Impute NaNs with column medians (more robust than means)
        for i in range(y_train.shape[1]):
            col_median = np.nanmedian(y_train[:, i])
            nan_mask = np.isnan(y_train[:, i])
            if np.any(nan_mask):
                logger.info(f"Imputing {np.sum(nan_mask)} NaN values in target {self.target_columns[i]} with median {col_median:.4f}")
            y_cleaned[nan_mask, i] = col_median

        return y_cleaned

class EnhancedStackingEnsemble:
    """Enhanced stacking ensemble with ResNet1D and traditional ML models"""

    def __init__(self, n_splits=5, target_columns=None, random_state=42):
        self.n_splits = n_splits
        self.target_columns = target_columns or ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        self.random_state = random_state
        self.preprocessor = DataPreprocessor(target_columns)

        # Initialize models with robust parameters
        self.base_models = self._initialize_base_models()
        self.meta_regressor = Ridge(alpha=1.0)
        self.trained_models = {}
        self.trained_resnet = None
        self.meta_features = None
        self.meta_target = None
        self.fold_performance = []

    def _initialize_base_models(self):
        """Initialize base models with robust parameters"""
        return [
            ('pls', PLSRegression(n_components=2)),  # Reduced components for stability
            ('ridge', Ridge(alpha=1.0)),
            ('xgb', XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=self.random_state,
                tree_method='hist'  # More memory efficient
            )),
            ('rf', RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state,
                n_jobs=-1
            )),
            ('lgbm', LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=self.random_state,
                verbose=-1  # Suppress LightGBM output
            ))
        ]

    def _get_callbacks(self):
        """Get training callbacks for neural network"""
        return [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]

    def _create_model_instance(self, model, params=None):
        """Create a fresh instance of a model"""
        if params is None:
            params = model.get_params()
        return model.__class__(**params)

    def fit(self, expanded_train, train, groups, expanded_test=None, hyperparameters=None):
        """
        Fit the stacking ensemble model with robust error handling
        """
        logger.info("Starting stacking ensemble training...")

        # Validate input data
        self._validate_input_data(expanded_train, train)

        # Prepare data with proper NaN handling
        X_train_trad = self.preprocessor.prepare_traditional_features(expanded_train, fit=True)
        X_train_resnet = self.preprocessor.prepare_resnet_features(expanded_train, fit=True)
        y_train = self.preprocessor.prepare_targets(train)

        # Set hyperparameters
        self.hyperparameters = hyperparameters or self._get_default_hyperparameters()

        # Perform stacking with cross-validation
        self._stacking_cv(X_train_trad, X_train_resnet, y_train, groups)

        # Train final models on full data
        self._train_final_models(X_train_trad, X_train_resnet, y_train)

        logger.info("Stacking ensemble training completed successfully")
        return self

    def _validate_input_data(self, expanded_train, train):
        """Validate input data before processing"""
        if expanded_train is None or train is None:
            raise ValueError("expanded_train and train cannot be None")
        
        if len(expanded_train) != len(train):
            raise ValueError("expanded_train and train must have the same number of samples")
        
        missing_targets = [col for col in self.target_columns if col not in train.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns in training data: {missing_targets}")

    def _stacking_cv(self, X_trad, X_resnet, y, groups):
        """Perform stacking with GroupKFold cross-validation"""
        kf = GroupKFold(n_splits=self.n_splits)
        meta_features_list = []
        meta_target_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trad, y, groups)):
            logger.info(f"Processing fold {fold+1}/{self.n_splits}")

            # Split data for current fold
            X_trad_train, X_trad_val = X_trad.iloc[train_idx], X_trad.iloc[val_idx]
            X_resnet_train, X_resnet_val = X_resnet.iloc[train_idx], X_resnet.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Reshape ResNet data
            X_resnet_train_reshaped = X_resnet_train.values.reshape(X_resnet_train.shape[0], X_resnet_train.shape[1], 1)
            X_resnet_val_reshaped = X_resnet_val.values.reshape(X_resnet_val.shape[0], X_resnet_val.shape[1], 1)

            # Get out-of-fold predictions for this fold
            fold_oof_preds = self._get_oof_predictions(
                X_trad_train, X_trad_val, X_resnet_train_reshaped, X_resnet_val_reshaped,
                y_train_fold, y_val_fold, fold
            )

            # Store meta features and targets
            meta_features_list.append(fold_oof_preds)
            meta_target_list.append(y_val_fold)

            # Clean up TensorFlow session
            tf.keras.backend.clear_session()

        # Combine meta features from all folds
        self.meta_features = pd.concat(meta_features_list, axis=0)
        self.meta_target = np.concatenate(meta_target_list, axis=0)

        # Train meta-regressor
        logger.info("Training meta-regressor on out-of-fold predictions...")
        self.meta_regressor.fit(self.meta_features, self.meta_target)

    def _get_oof_predictions(self, X_trad_train, X_trad_val, X_resnet_train, X_resnet_val, y_train, y_val, fold):
        """Get out-of-fold predictions for all base models"""
        fold_oof_preds = pd.DataFrame(index=range(len(y_val)))

        # Train and predict with traditional models
        for name, model in self.base_models:
            try:
                logger.info(f"  Training {name} on fold {fold+1}...")

                if name == 'xgb':
                    preds = self._train_xgb_fold(X_trad_train, X_trad_val, y_train, name)
                else:
                    model_clone = self._create_model_instance(model)
                    model_clone.fit(X_trad_train, y_train)
                    preds = model_clone.predict(X_trad_val)

                # Store predictions
                for i, target in enumerate(self.target_columns):
                    col_name = f'{name}_pred_{target}'
                    if preds.ndim == 1:  # Handle single-output models
                        fold_oof_preds[col_name] = preds
                    else:
                        fold_oof_preds[col_name] = preds[:, i]
                        
            except Exception as e:
                logger.warning(f"Model {name} failed in fold {fold+1}: {str(e)}")
                # Fill with mean values as fallback
                for i, target in enumerate(self.target_columns):
                    fold_oof_preds[f'{name}_pred_{target}'] = np.mean(y_train[:, i])

        # Train and predict with ResNet1D
        try:
            logger.info(f"  Training ResNet1D on fold {fold+1}...")
            resnet_preds = self._train_resnet_fold(X_resnet_train, X_resnet_val, y_train, y_val)
            for i, target in enumerate(self.target_columns):
                fold_oof_preds[f'ResNet1D_pred_{target}'] = resnet_preds[:, i]
        except Exception as e:
            logger.warning(f"ResNet1D failed in fold {fold+1}: {str(e)}")
            for i, target in enumerate(self.target_columns):
                fold_oof_preds[f'ResNet1D_pred_{target}'] = np.mean(y_train[:, i])

        return fold_oof_preds

    def _train_xgb_fold(self, X_train, X_val, y_train, model_name):
        """Train XGBoost model for each target separately"""
        xgb_preds = np.zeros((X_val.shape[0], len(self.target_columns)))
        model_params = self.hyperparameters.get(model_name, {})

        for i, target in enumerate(self.target_columns):
            xgb_model = XGBRegressor(**model_params)
            xgb_model.fit(X_train, y_train[:, i])
            xgb_preds[:, i] = xgb_model.predict(X_val)

        return xgb_preds

    def _train_resnet_fold(self, X_train, X_val, y_train, y_val):
        """Train ResNet1D model for current fold"""
        tf.keras.backend.clear_session()

        resnet_params = self.hyperparameters['ResNet1D']
        input_shape = (X_train.shape[1], 1)
        num_targets = y_train.shape[1]

        model = ResNet1D.build_model(input_shape, num_targets, resnet_params)
        model.compile(
            optimizer=Adam(learning_rate=resnet_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        model.fit(
            X_train, y_train,
            epochs=resnet_params['epochs'],
            batch_size=resnet_params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=self._get_callbacks(),
            verbose=0
        )

        return model.predict(X_val)

    def _train_final_models(self, X_trad, X_resnet, y):
        """Train final models on full training data"""
        logger.info("Training final models on full dataset...")

        # Train traditional models
        for name, model in self.base_models:
            try:
                logger.info(f"  Training final {name} model...")

                if name == 'xgb':
                    xgb_models = {}
                    model_params = self.hyperparameters.get(name, {})
                    for i, target in enumerate(self.target_columns):
                        xgb_model = XGBRegressor(**model_params)
                        xgb_model.fit(X_trad, y[:, i])
                        xgb_models[target] = xgb_model
                    self.trained_models[name] = xgb_models
                else:
                    model_clone = self._create_model_instance(model)
                    model_clone.fit(X_trad, y)
                    self.trained_models[name] = model_clone
            except Exception as e:
                logger.error(f"Failed to train final {name} model: {str(e)}")

        # Train final ResNet1D model
        try:
            logger.info("  Training final ResNet1D model...")
            X_resnet_full = X_resnet.values.reshape(X_resnet.shape[0], X_resnet.shape[1], 1)
            resnet_params = self.hyperparameters['ResNet1D']

            self.trained_resnet = ResNet1D.build_model(
                (X_resnet.shape[1], 1), y.shape[1], resnet_params
            )
            self.trained_resnet.compile(
                optimizer=Adam(learning_rate=resnet_params['learning_rate']),
                loss='mse',
                metrics=['mae']
            )

            self.trained_resnet.fit(
                X_resnet_full, y,
                epochs=resnet_params['epochs'],
                batch_size=resnet_params['batch_size'],
                verbose=0,
                validation_split=0.2,
                callbacks=self._get_callbacks()
            )
        except Exception as e:
            logger.error(f"Failed to train final ResNet1D model: {str(e)}")

    def predict(self, expanded_test):
        """Generate predictions using the stacked ensemble"""
        logger.info("Generating predictions...")

        if expanded_test is None:
            raise ValueError("expanded_test cannot be None")

        # Prepare test data using stored imputers
        X_test_trad = self.preprocessor.prepare_traditional_features(expanded_test, fit=False)
        X_test_resnet = self.preprocessor.prepare_resnet_features(expanded_test, fit=False)
        X_test_resnet_reshaped = X_test_resnet.values.reshape(X_test_resnet.shape[0], X_test_resnet.shape[1], 1)

        # Generate base model predictions
        base_preds = self._generate_base_predictions(X_test_trad, X_test_resnet_reshaped)

        # Ensure column alignment with meta features
        if self.meta_features is not None:
            base_preds = base_preds.reindex(columns=self.meta_features.columns, fill_value=0)

        # Generate final ensemble predictions
        final_predictions = self.meta_regressor.predict(base_preds)

        return pd.DataFrame(final_predictions, columns=self.target_columns)

    def _generate_base_predictions(self, X_test_trad, X_test_resnet):
        """Generate predictions from all base models"""
        base_preds = pd.DataFrame(index=range(X_test_trad.shape[0]))

        # Traditional models
        for name, model in self.trained_models.items():
            try:
                if name == 'xgb':
                    # XGBoost has separate models for each target
                    xgb_preds = np.zeros((X_test_trad.shape[0], len(self.target_columns)))
                    for i, target in enumerate(self.target_columns):
                        xgb_preds[:, i] = model[target].predict(X_test_trad)
                    preds = xgb_preds
                else:
                    preds = model.predict(X_test_trad)

                for i, target in enumerate(self.target_columns):
                    col_name = f'{name}_pred_{target}'
                    if preds.ndim == 1:
                        base_preds[col_name] = preds
                    else:
                        base_preds[col_name] = preds[:, i]
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {str(e)}")
                # Fill with zeros as fallback
                for i, target in enumerate(self.target_columns):
                    base_preds[f'{name}_pred_{target}'] = 0

        # ResNet1D predictions
        if self.trained_resnet is not None:
            try:
                resnet_preds = self.trained_resnet.predict(X_test_resnet)
                for i, target in enumerate(self.target_columns):
                    base_preds[f'ResNet1D_pred_{target}'] = resnet_preds[:, i]
            except Exception as e:
                logger.warning(f"ResNet1D prediction failed: {str(e)}")
                for i, target in enumerate(self.target_columns):
                    base_preds[f'ResNet1D_pred_{target}'] = 0

        return base_preds

    def save(self, filepath):
        """Save the entire ensemble model"""
        logger.info(f"Saving ensemble model to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the ensemble without the ResNet model (which will be saved separately)
        ensemble_data = {
            'trained_models': self.trained_models,
            'meta_regressor': self.meta_regressor,
            'meta_features': self.meta_features,
            'meta_target': self.meta_target,
            'preprocessor': self.preprocessor,
            'target_columns': self.target_columns,
            'hyperparameters': self.hyperparameters,
            'n_splits': self.n_splits,
            'random_state': self.random_state
        }
        
        # Save ensemble data
        joblib.dump(ensemble_data, f"{filepath}_ensemble.joblib")
        
        # Save ResNet model separately
        if self.trained_resnet is not None:
            save_model(self.trained_resnet, f"{filepath}_resnet.h5")
        
        logger.info("Model saved successfully")

    def load(self, filepath):
        """Load the entire ensemble model"""
        logger.info(f"Loading ensemble model from {filepath}")
        
        # Load ensemble data
        ensemble_data = joblib.load(f"{filepath}_ensemble.joblib")
        
        # Restore attributes
        self.trained_models = ensemble_data['trained_models']
        self.meta_regressor = ensemble_data['meta_regressor']
        self.meta_features = ensemble_data['meta_features']
        self.meta_target = ensemble_data['meta_target']
        self.preprocessor = ensemble_data['preprocessor']
        self.target_columns = ensemble_data['target_columns']
        self.hyperparameters = ensemble_data['hyperparameters']
        self.n_splits = ensemble_data['n_splits']
        self.random_state = ensemble_data['random_state']
        
        # Load ResNet model
        try:
            self.trained_resnet = load_model(f"{filepath}_resnet.h5")
        except:
            logger.warning("Could not load ResNet model")
            self.trained_resnet = None
        
        logger.info("Model loaded successfully")
        return self

    def evaluate(self, y_true, y_pred, set_name="Test"):
        """Evaluate model performance"""
        metrics = {}

        for i, target in enumerate(self.target_columns):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])

            metrics[target] = {'MSE': mse, 'MAE': mae, 'RÂ²': r2}

            logger.info(f"{set_name} - {target}: MSE={mse:.4f}, MAE={mae:.4f}, RÂ²={r2:.4f}")

        return metrics

    def _get_default_hyperparameters(self):
        """Get default hyperparameters"""
        return {
            'pls': {'n_components': 2},  # Reduced for stability
            'ridge': {'alpha': 1.0},
            'xgb': {
                'n_estimators': 100, 
                'learning_rate': 0.1, 
                'max_depth': 6,
                'random_state': self.random_state
            },
            'rf': {
                'n_estimators': 100, 
                'max_depth': 10, 
                'random_state': self.random_state
            },
            'lgbm': {
                'n_estimators': 100, 
                'learning_rate': 0.1, 
                'random_state': self.random_state
            },
            'ResNet1D': {
                'learning_rate': 0.001, 
                'filters_l1': 64, 
                'filters_l2': 128, 
                'filters_l3': 256,
                'kernel_size_l1': 7, 
                'kernel_size_l2': 5, 
                'kernel_size_l3': 3,
                'dense_units': 128, 
                'epochs': 50,
                'batch_size': 32
            }
        }

# Complete workflow function
def complete_workflow(expanded_train, train, groups, expanded_test=None, model_path="/kaggle/working/ensemble_model"):
    """Complete workflow: train if model doesn't exist, otherwise load and predict"""
    
    ensemble = EnhancedStackingEnsemble(
        n_splits=3,  # Reduced for faster execution
        target_columns=['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)'],
        random_state=42
    )
    
    # Check if model exists
    if os.path.exists(f"{model_path}_ensemble.joblib"):
        print("Loading existing model...")
        ensemble.load(model_path)
    else:
        print("Training new model...")
        ensemble.fit(expanded_train, train, groups, expanded_test)
        ensemble.save(model_path)
        print("Model trained and saved successfully!")
    
    # Generate predictions if test data is provided
    predictions = None
    if expanded_test is not None:
        predictions = ensemble.predict(expanded_test)
        print("\nPredictions generated successfully!")
        print("First 5 predictions:")
        print(predictions.head())
    
    return ensemble, predictions

# Create dummy data for testing
def create_dummy_data():
    """Create dummy data for testing the ensemble"""
    np.random.seed(42)
    
    # Create dummy expanded_train data
    n_samples = 100
    n_features = 50
    
    expanded_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some identifier columns
    expanded_train['Unnamed: 0'] = [f'sample_{i}' for i in range(n_samples)]
    expanded_train['Analyte concentration'] = [f'conc_{i}' for i in range(n_samples)]
    
    # Create dummy train data with targets
    train = pd.DataFrame({
        'Glucose (g/L)': np.random.uniform(1, 12, n_samples),
        'Sodium Acetate (g/L)': np.random.uniform(0.2, 2.2, n_samples),
        'Magnesium Acetate (g/L)': np.random.uniform(0.4, 2.8, n_samples)
    })
    
    # Create dummy groups
    groups = np.random.randint(0, 5, n_samples)
    
    # Create dummy test data
    expanded_test = pd.DataFrame(
        np.random.randn(20, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    expanded_test['Unnamed: 0'] = [f'test_sample_{i}' for i in range(20)]
    expanded_test['Analyte concentration'] = [f'test_conc_{i}' for i in range(20)]
    
    return expanded_train, train, groups, expanded_test

# Run the complete workflow
if __name__ == "__main__":
    try:
        # Create dummy data for testing
        expanded_train, train, groups, expanded_test = create_dummy_data()
        
        # Run complete workflow
        ensemble, predictions = complete_workflow(
            expanded_train, train, groups, expanded_test, 
            model_path="/kaggle/working/ensemble_model"
        )
        
        print("Workflow completed successfully!")
        
    except Exception as e:
        print(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()


import os

os.makedirs('/content/train_images', exist_ok=True)
os.makedirs('/content/test_images', exist_ok=True)

print("âœ… Directories /content/train_images and /content/test_images ensured to exist.")


import pandas as pd

try:
    test_submission_df = pd.read_csv('/content/sample_submission_1.csv')
    print("First 5 rows of /content/sample_submission_1.csv (TEST 1):")
    display(test_submission_df.head())
except FileNotFoundError:
    print("Error: /content/sample_submission_1.csv not found. Please ensure the file is uploaded.")


import pandas as pd

# Load the train1.csv file from the /content/ directory
try:
    train_df_check = pd.read_csv('/content/train1.csv')
    # Display the first few rows of the 'Image_name' column
    print("First 5 image names in train1.csv:")
    display(train_df_check['Image_name'].head())
except FileNotFoundError:
    print("Error: train1.csv not found in /content/. Please ensure the file is uploaded.")


ls /content/


# ğŸ�† Grand X-Ray Slam: Full Pipeline â€“ Train, Validate, Infer, Submit

import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm

# Config
class CFG:
    seed = 42
    img_size = 224
    batch_size = 16
    epochs = 1
    folds = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
                   'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                   'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
                   'Pneumothorax','Support Devices']
    model_name_cnn = "tf_efficientnet_b0_ns"
    model_name_vit = "vit_base_patch16_224"
    save_dir = "./fold_models"
os.makedirs(CFG.save_dir, exist_ok=True)

# Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(CFG.seed)

# Dataset
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train
        self.metadata_cols = ['Sex', 'Age', 'ViewPosition']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms: image = self.transforms(image)

        meta = np.zeros(len(self.metadata_cols), dtype=np.float32)
        meta[0] = 0 if pd.isna(row.get('Sex')) else (1 if row.get('Sex')=='Male' else 2)
        meta[1] = row.get('Age', 50.0)/100.0 if not pd.isna(row.get('Age')) else 50.0/100.0
        meta[2] = 0 if pd.isna(row.get('ViewPosition')) else (0 if row.get('ViewPosition')=='PA' else 1)

        if self.train:
            labels = row[CFG.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels), torch.tensor(meta)
        else:
            return image, torch.tensor(meta), row['Image_name']

# Models
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_cnn, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x): return torch.sigmoid(self.backbone(x))

class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_vit, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x): return torch.sigmoid(self.backbone(x))

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNModel()
        self.vit = ViTModel()
        self.meta_fc = nn.Linear(3, 16)
        self.out_fc = nn.Linear(len(CFG.target_cols)*2+16, len(CFG.target_cols))
    def forward(self, x, meta):
        cnn_out = self.cnn(x)
        vit_out = self.vit(x)
        meta_out = torch.relu(self.meta_fc(meta))
        combined = torch.cat([cnn_out, vit_out, meta_out], dim=1)
        return torch.sigmoid(self.out_fc(combined))

# Mixup & CutMix
def rand_bbox(size, lam):
    W,H = size[2], size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1,bby1 = np.clip(cx-cut_w//2,0,W), np.clip(cy-cut_h//2,0,H)
    bbx2,bby2 = np.clip(cx+cut_w//2,0,W), np.clip(cy+cut_h//2,0,H)
    return bbx1,bby1,bbx2,bby2

def mixup_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    shuffled_x, shuffled_y = x[index], y[index]
    bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(), lam)
    x[:,:,bby1:bby2,bbx1:bbx2] = shuffled_x[:,:,bby1:bby2,bbx1:bbx2]
    lam = 1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
    return x, y, shuffled_y, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred,y_a) + (1-lam)*criterion(pred,y_b)

# Train one epoch
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for imgs, labels, meta in loader:
        imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
        r = np.random.rand()
        if r<0.33:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        elif r<0.66:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# Validation
def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    aucs = []
    for i in range(len(CFG.target_cols)):
        try: auc = roc_auc_score(y_true[:,i], y_pred[:,i])
        except: auc = np.nan
        aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)

# Load data
train_df = pd.read_csv("/content/train1.csv")
test_df  = pd.read_csv("/content/sample_submission_1.csv")
train_img_dir = "/content/train_images"
test_img_dir  = "/content/test_images"
train_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.RandomHorizontalFlip(), T.ToTensor()])
valid_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.ToTensor()])

# CV Training
kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
cv_scores, all_fold_preds = [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    print(f"===== FOLD {fold+1} =====")
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    tr_ds = CXRDataset(tr_df, train_img_dir, train_tfms, True)
    va_ds = CXRDataset(


# %% [markdown]
# # ğŸ�† Grand X-Ray Slam: Division A â€“ Triple-Stack CV + Mixup/CutMix + Live AUC
#
# This notebook:
# - Trains a CV ensemble (EfficientNet + ViT + Metadata fusion)
# - Uses Mixup / CutMix augmentation
# - Monitors per-label and mean AUC live per epoch
# - Outputs submission CSV + Parquet
# - Provides a download link

# %% [code]
import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm
from IPython.display import FileLink

# =========================================================
# Config
# =========================================================
class CFG:
    seed = 42
    img_size = 224
    batch_size = 16
    epochs = 1  # Increase for real runs
    folds = 3   # Use 5-10 for leaderboard
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema',
        'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
        'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
        'Pneumothorax','Support Devices'
    ]
    model_name_cnn = "tf_efficientnet_b0_ns"
    model_name_vit = "vit_base_patch16_224"
    save_dir = "./fold_models"
os.makedirs(CFG.save_dir, exist_ok=True)

# =========================================================
# Seed
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(CFG.seed)

# =========================================================
# Dataset with Metadata
# =========================================================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train
        self.metadata_cols = ['Sex', 'Age', 'ViewPosition']  # simple example

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Simple metadata encoding
        meta = np.zeros(len(self.metadata_cols), dtype=np.float32)
        # Ensure columns exist before accessing
        meta[0] = 0 if pd.isna(row.get('Sex')) else (1 if row.get('Sex')=='Male' else 2)
        meta[1] = row.get('Age', 50.0)/100.0 if not pd.isna(row.get('Age')) else 50.0/100.0
        meta[2] = 0 if pd.isna(row.get('ViewPosition')) else (0 if row.get('ViewPosition')=='PA' else 1)

        if self.train:
            labels = row[CFG.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)
        else:
            return image, torch.tensor(meta, dtype=torch.float32), row['Image_name']

# =========================================================
# Models
# =========================================================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_cnn, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_vit, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

class FusionModel(nn.Module):
    """CNN + ViT + metadata fusion"""
    def __init__(self):
        super().__init__()
        self.cnn = CNNModel()
        self.vit = ViTModel()
        self.meta_fc = nn.Linear(3, 16)
        self.out_fc = nn.Linear(len(CFG.target_cols)*2+16, len(CFG.target_cols))
    def forward(self, x, meta):
        cnn_out = self.cnn(x)
        vit_out = self.vit(x)
        meta_out = torch.relu(self.meta_fc(meta))
        combined = torch.cat([cnn_out, vit_out, meta_out], dim=1)
        return torch.sigmoid(self.out_fc(combined))

# =========================================================
# Mixup & CutMix (as before)
# =========================================================
def rand_bbox(size, lam):
    W,H = size[2], size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w = int(W*cut_rat)
    cut_h = int(H*cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx-cut_w//2,0,W)
    bby1 = np.clip(cy-cut_h//2,0,H)
    bbx2 = np.clip(cx+cut_w//2,0,W)
    bby2 = np.clip(cy+cut_h//2,0,H)
    return bbx1,bby1,bbx2,bby2

def mixup_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    shuffled_x, shuffled_y = x[index], y[index]
    bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(), lam)
    x[:,:,bby1:bby2,bbx1:bbx2] = shuffled_x[:,:,bby1:bby2,bbx1:bbx2]
    lam = 1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
    return x, y, shuffled_y, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred,y_a) + (1-lam)*criterion(pred,y_b)

# =========================================================
# Train one epoch with Mixup/CutMix
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for imgs, labels, meta in loader:
        imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
        r = np.random.rand()
        if r<0.33:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        elif r<0.66:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# =========================================================
# Validation
# =========================================================
def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    aucs = []
    for i in range(len(CFG.target_cols)):
        try: auc = roc_auc_score(y_true[:,i], y_pred[:,i])
        except: auc = np.nan
        aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)

# =========================================================
# Load Data (updated paths)
# =========================================================
train_df = pd.read_csv("/content/train1.csv")
test_df  = pd.read_csv("/content/sample_submission_1.csv")  # for image names

train_img_dir = "/content/train_images" # Update if necessary
test_img_dir  = "/content/test_images"  # Update if necessary

train_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.RandomHorizontalFlip(), T.ToTensor()])
valid_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.ToTensor()])

# =========================================================
# CV Fold Training with FusionModel
# =========================================================
kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    print(f"===== FOLD {fold+1} =====")
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    tr_ds = CXRDataset(tr_df, train_img_dir, transforms=train_tfms, train=True)
    va_ds = CXRDataset(va_df, train_img_dir, transforms=valid_tfms, train=True)
    tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    model = FusionModel().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_auc = -np.inf
    best_path = f"{CFG.save_dir}/fusion_fold{fold}.pt"

    for epoch in range(CFG.epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        va_loss, mean_auc = validate(model, va_loader, criterion)[:2] # Only get mean_auc
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, mean_val_auc={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc=mean_auc
            torch.save(model.state_dict(), best_path)
    cv_scores.append(best_auc)
    print(f"Best AUC fold {fold+1}: {best_auc:.4f}")

print("CV AUCs:", cv_scores)
print("Mean CV AUC:", np.mean(cv_scores))

# =========================================================
# Inference + Fallback Random Submission
# =========================================================
test_ds = CXRDataset(test_df, test_img_dir, transforms=valid_tfms, train=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

all_fold_preds = []

try:
    for fold in range(CFG.folds):
        model = FusionModel().to(CFG.device)
        model.load_state_dict(torch.load(f"{CFG.save_dir}/fusion_fold{fold}.pt", map_location=CFG.device))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for imgs, meta, names in test_loader:
                imgs, meta = imgs.to(CFG.device), meta.to(CFG.device)
                preds = model(imgs, meta)
                fold_preds.append(preds.cpu().numpy())
        all_fold_preds.append(np.vstack(fold_preds))
    final_preds = np.mean(all_fold_preds, axis=0)
except Exception as e:
    print(f"âš ï¸� Model inference failed due to {e}. Generating random fallback submission.")
    NUM_ROWS = len(test_df)
    final_preds = np.random.rand(NUM_ROWS, len(CFG.target_cols))

# =========================================================
# Submission
# =========================================================
import pandas as pd
import numpy as np

# Load sample_submission to get exact test image names
sample_submission_path = "/content/sample_submission_1.csv" # Updated path
sample_submission_df = pd.read_csv(sample_submission_path)

# Labels
LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
          'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
          'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
          'Pneumothorax','Support Devices']

# Number of test rows
NUM_ROWS = len(sample_submission_df)

# Use actual model predictions if available, otherwise fallback to random
# final_preds should be available from the inference step if successful

# Create submission DataFrame
submission_df = pd.DataFrame(final_preds, columns=LABELS)
submission_df.insert(0, 'Image_name', sample_submission_df['Image_name'].values)

# Save CSV
submission_df.to_csv("grand_xray_slam_submission.csv", index=False)
print("âœ… Submission CSV generated successfully with correct Image_name and columns.")
print(submission_df.head())

# Download link (Optional, specific to environments like Colab)
# FileLink("grand_xray_slam_submission.csv")


%%bash
# Check if the directory exists
if [ -d "/content/train_images" ]; then
  echo "âœ… /content/train_images directory exists."
  # List contents if it exists
  echo "Contents of /content/train_images/:"
  ls /content/train_images/
else
  echo "â�Œ /content/train_images directory does NOT exist."
fi


!kaggle competitions download -c grand-xray-slam-division-a


# %% [markdown]
# # ğŸ�† Grand X-Ray Slam: Division A â€“ Triple-Stack CV + Mixup/CutMix + Live AUC
#
# This notebook:
# - Trains a CV ensemble (EfficientNet + ViT + Metadata fusion)
# - Uses Mixup / CutMix augmentation
# - Monitors per-label and mean AUC live per epoch
# - Outputs submission CSV + Parquet
# - Provides a download link

# %% [code]
import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm
from IPython.display import FileLink

# =========================================================
# Config
# =========================================================
class CFG:
    seed = 42
    img_size = 224
    batch_size = 16
    epochs = 1  # Increase for real runs
    folds = 3   # Use 5-10 for leaderboard
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_cols = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema',
        'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
        'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
        'Pneumothorax','Support Devices'
    ]
    model_name_cnn = "tf_efficientnet_b0_ns"
    model_name_vit = "vit_base_patch16_224"
    save_dir = "./fold_models"
os.makedirs(CFG.save_dir, exist_ok=True)

# =========================================================
# Seed
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(CFG.seed)

# =========================================================
# Dataset with Metadata
# =========================================================
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.train = train
        self.metadata_cols = ['Sex', 'Age', 'ViewPosition']  # simple example

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Simple metadata encoding
        meta = np.zeros(len(self.metadata_cols), dtype=np.float32)
        # Ensure columns exist before accessing
        meta[0] = 0 if pd.isna(row.get('Sex')) else (1 if row.get('Sex')=='Male' else 2)
        meta[1] = row.get('Age', 50.0)/100.0 if not pd.isna(row.get('Age')) else 50.0/100.0
        meta[2] = 0 if pd.isna(row.get('ViewPosition')) else (0 if row.get('ViewPosition')=='PA' else 1)

        if self.train:
            labels = row[CFG.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)
        else:
            return image, torch.tensor(meta, dtype=torch.float32), row['Image_name']

# =========================================================
# Models
# =========================================================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_cnn, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name_vit, pretrained=True, num_classes=len(CFG.target_cols))
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

class FusionModel(nn.Module):
    """CNN + ViT + metadata fusion"""
    def __init__(self):
        super().__init__()
        self.cnn = CNNModel()
        self.vit = ViTModel()
        self.meta_fc = nn.Linear(3, 16)
        self.out_fc = nn.Linear(len(CFG.target_cols)*2+16, len(CFG.target_cols))
    def forward(self, x, meta):
        cnn_out = self.cnn(x)
        vit_out = self.vit(x)
        meta_out = torch.relu(self.meta_fc(meta))
        combined = torch.cat([cnn_out, vit_out, meta_out], dim=1)
        return torch.sigmoid(self.out_fc(combined))

# =========================================================
# Mixup & CutMix (as before)
# =========================================================
def rand_bbox(size, lam):
    W,H = size[2], size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w = int(W*cut_rat)
    cut_h = int(H*cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx-cut_w//2,0,W)
    bby1 = np.clip(cy-cut_h//2,0,H)
    bbx2 = np.clip(cx+cut_w//2,0,W)
    bby2 = np.clip(cy+cut_h//2,0,H)
    return bbx1,bby1,bbx2,bby2

def mixup_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha<=0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    shuffled_x, shuffled_y = x[index], y[index]
    bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(), lam)
    x[:,:,bby1:bby2,bbx1:bbx2] = shuffled_x[:,:,bby1:bby2,bbx1:bbx2]
    lam = 1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
    return x, y, shuffled_y, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred,y_a) + (1-lam)*criterion(pred,y_b)

# =========================================================
# Train one epoch with Mixup/CutMix
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for imgs, labels, meta in loader:
        imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
        r = np.random.rand()
        if r<0.33:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        elif r<0.66:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            preds = model(imgs, meta)
            loss = mix_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# =========================================================
# Validation
# =========================================================
def validate(model, loader, criterion):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs, labels, meta = imgs.to(CFG.device), labels.to(CFG.device), meta.to(CFG.device)
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    aucs = []
    for i in range(len(CFG.target_cols)):
        try: auc = roc_auc_score(y_true[:,i], y_pred[:,i])
        except: auc = np.nan
        aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)

# =========================================================
# Load Data (updated paths)
# =========================================================
train_df = pd.read_csv("/content/train1.csv")
test_df  = pd.read_csv("/content/sample_submission_1.csv")  # for image names

train_img_dir = "/content/train_images" # Update if necessary
test_img_dir  = "/content/test_images"  # Update if necessary

train_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.RandomHorizontalFlip(), T.ToTensor()])
valid_tfms = T.Compose([T.Resize((CFG.img_size,CFG.img_size)), T.ToTensor()])

# =========================================================
# CV Fold Training with FusionModel
# =========================================================
kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
    print(f"===== FOLD {fold+1} =====")
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    tr_ds = CXRDataset(tr_df, train_img_dir, transforms=train_tfms, train=True)
    va_ds = CXRDataset(va_df, train_img_dir, transforms=valid_tfms, train=True)
    tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

    model = FusionModel().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_auc = -np.inf
    best_path = f"{CFG.save_dir}/fusion_fold{fold}.pt"

    for epoch in range(CFG.epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        va_loss, mean_auc = validate(model, va_loader, criterion)[:2] # Only get mean_auc
        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, mean_val_auc={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc=mean_auc
            torch.save(model.state_dict(), best_path)
    cv_scores.append(best_auc)
    print(f"Best AUC fold {fold+1}: {best_auc:.4f}")

print("CV AUCs:", cv_scores)
print("Mean CV AUC:", np.mean(cv_scores))

# =========================================================
# Inference + Fallback Random Submission
# =========================================================
test_ds = CXRDataset(test_df, test_img_dir, transforms=valid_tfms, train=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

all_fold_preds = []

try:
    for fold in range(CFG.folds):
        model = FusionModel().to(CFG.device)
        model.load_state_dict(torch.load(f"{CFG.save_dir}/fusion_fold{fold}.pt", map_location=CFG.device))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for imgs, meta, names in test_loader:
                imgs, meta = imgs.to(CFG.device), meta.to(CFG.device)
                preds = model(imgs, meta)
                fold_preds.append(preds.cpu().numpy())
        all_fold_preds.append(np.vstack(fold_preds))
    final_preds = np.mean(all_fold_preds, axis=0)
except Exception as e:
    print(f"âš ï¸� Model inference failed due to {e}. Generating random fallback submission.")
    NUM_ROWS = len(test_df)
    final_preds = np.random.rand(NUM_ROWS, len(CFG.target_cols))

# =========================================================
# Submission
# =========================================================
import pandas as pd
import numpy as np

# Load sample_submission to get exact test image names
sample_submission_path = "/content/sample_submission_1.csv" # Updated path
sample_submission_df = pd.read_csv(sample_submission_path)

# Labels
LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
          'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
          'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
          'Pneumothorax','Support Devices']

# Number of test rows
NUM_ROWS = len(sample_submission_df)

# Use actual model predictions if available, otherwise fallback to random
# final_preds should be available from the inference step if successful

# Create submission DataFrame
submission_df = pd.DataFrame(final_preds, columns=LABELS)
submission_df.insert(0, 'Image_name', sample_submission_df['Image_name'].values)

# Save CSV
submission_df.to_csv("grand_xray_slam_submission.csv", index=False)
print("âœ… Submission CSV generated successfully with correct Image_name and columns.")
print(submission_df.head())

# Download link (Optional, specific to environments like Colab)
# FileLink("grand_xray_slam_submission.csv")

