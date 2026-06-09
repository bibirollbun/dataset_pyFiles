!pip install opencv-python
!pip install tqdm
!pip install pandas
!pip install scikit-image
!pip install scikit-learn
!pip install seaborn



! pip install -qU "python-gdcm" pydicom pylibjpeg "opencv-python-headless"



# ===== Minimal imports for local PNG dataset =====

import os
import re
import gc
import cv2
import random
import math
from glob import glob
from tqdm import tqdm
from pprint import pprint
from time import time
import datetime as dtime
from datetime import datetime
import itertools
import warnings
import pandas as pd
import numpy as np

from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, normalize


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 16})

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom colors and print helper (optional)
class clr:
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
my_colors = ["#517664", "#73AA90", "#94DDBC", "#DAB06C", 
             "#DF928E", "#C97973", "#B25F57"]
CMAP1 = ListedColormap(my_colors)

print(clr.S+"Notebook Color Schemes:"+clr.E)
sns.palplot(sns.color_palette(my_colors))
plt.show()




# === General Functions ===

def set_seed(seed = 1234):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of a seaborn barplot.
    axs: the ax of the plot
    h_v: whether the barplot is vertical/horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        

# === W&B functions removed ===

# Replacing save_dataset_artifact with a simple save message
def save_dataset_artifact(run_name, artifact_name, path, data_type="dataset"):
    print(f"[INFO] Dataset artifact '{artifact_name}' would be saved from {path} (wandb removed).")
    # You can implement local saving/versioning here if needed
    

# Skipping wandb plots functions since wandb is removed
def create_wandb_plot(*args, **kwargs):
    print("[INFO] Wandb plot functions removed. Plotting skipped.")

def create_wandb_hist(*args, **kwargs):
    print("[INFO] Wandb histogram function removed. Plotting skipped.")



import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Load the original CSV
train = pd.read_csv("/kaggle/input/mammography-breast-cancer-detection/train.csv")

base_path = "/kaggle/input/mammography-breast-cancer-detection/train/"

all_paths = []
for idx, row in tqdm(train.iterrows(), total=len(train)):
    label_folder = str(row['cancer'])  # 0 or 1
    img_file = f"{row['patient_id']}_{row['image_id']}.png"
    full_path = os.path.join(base_path, label_folder, img_file)
    all_paths.append(full_path)

train["path"] = all_paths




print(clr.S+"Number of TOTAL images:"+clr.E,
      len(glob("/kaggle/input/rsna-breast-cancer-detection/train_images/*/*")))
print(clr.S+"Records gathered in Site 1:"+clr.E, train["site_id"].value_counts().values[0], "\n"+
      clr.S+"Records gathered in Site 2:"+clr.E, train["site_id"].value_counts().values[1])
print("-------------------------------------------------")
print(clr.S+"Total unique patients:"+clr.E, train["patient_id"].nunique())
print("-------------------------------------------------")
print(clr.S+"Total unique images:"+clr.E, train["image_id"].nunique())
print("-------------------------------------------------")
print(clr.S+"Statistics: Images per Patient"+clr.E)
print(train.groupby("patient_id")["image_id"].count().reset_index().describe()["image_id"])
print("-------------------------------------------------")
print(clr.S+"Image records count per laterality (R):"+clr.E, train["laterality"].value_counts().values[0], "\n"+
      clr.S+"Image records count per laterality (L):"+clr.E, train["laterality"].value_counts().values[1])
print("-------------------------------------------------")
print(clr.S+"Image records count per View:"+clr.E)
print(train["view"].value_counts())


# Keep only columns in test + target variable
train = train[["patient_id", "image_id", "laterality", "view", "age", "implant", "path", "cancer"]]

# Encode categorical variables
le_laterality = LabelEncoder()
le_view = LabelEncoder()

train['laterality'] = le_laterality.fit_transform(train['laterality'])
train['view'] = le_view.fit_transform(train['view'])

train.head()


print(clr.S+"Number of missing values in Age:"+clr.E, train["age"].isna().sum())
train['age'] = train['age'].fillna(58)


def save_dataset_artifact(run_name, artifact_name, path, data_type):
    # wandb removed: just print info about saving
    print(f"[INFO] Dataset artifact '{artifact_name}' saved locally at: {path} (wandb logging removed).")

# Save new dataset locally
train.to_csv("train_path.csv", index=False)

# Save "artifact" locally (just a print now)
save_dataset_artifact(run_name="save_train_prep", 
                      artifact_name="train_prep",
                      path="train_path.csv",
                      data_type="dataset")



!pip install -q efficientnet_pytorch
!pip install -q transformers
!pip install -q albumentations albumentations.pytorch



    # PyTorch
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import FloatTensor, LongTensor
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim import AdamW
    
    
    # Data Augmentation for Image Preprocessing

    from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, ShiftScaleRotate, ToGray)
    from albumentations.pytorch.transforms import ToTensorV2

    
    
    from efficientnet_pytorch import EfficientNet
    from torchvision.models import resnet34, resnet50
    
    # SKlearn
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Seed
set_seed()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', DEVICE)

# Read in Data
train = pd.read_csv("/kaggle/input/helper/train_path.csv")



# # Shuffle full dataset first
# full_df = train.sample(frac=1, random_state=13).reset_index(drop=True)

# # Split into train (10k) and test (1k)
# # train = full_df.iloc[0:3000].reset_index(drop=True)

# # Filter positives and negatives
# positive_df = train[train['cancer'] == 1].sample(n=200, random_state=13)
# negatives = train[train['cancer'] == 0].iloc[5000:]
# negative_df = negatives.sample(n=2800, random_state=13)

# # Combine and shuffle
# train = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=13).reset_index(drop=True)




# test = full_df.iloc[5000:6000].reset_index(drop=True)

# # Check class balance
# print("Train cancer distribution:\n", train["cancer"].value_counts())
# print("Test cancer distribution:\n", test["cancer"].value_counts())





# Assuming `train` is your original full dataset

# Select training set
negatives_train = train[train['cancer'] == 0].head(2000)
positives_train = train[train['cancer'] == 1].head(500)
train_set = pd.concat([negatives_train, positives_train]).reset_index(drop=True)

# Select test set from original dataset (not the reduced train_set)
test_set = train.iloc[-1000:].reset_index(drop=True)

print("Train cancer distribution:\n", train_set["cancer"].value_counts())
print(f"Train set size: {len(train_set)}")

print("Test cancer distribution:\n", test_set["cancer"].value_counts())
print(f"Test set size: {len(test_set)}")




# ----- GLOBAL PARAMS -----
vertical_flip = 0.5
horizontal_flip = 0.5

csv_columns = ['laterality', 'view', 'age', 'implant']
no_columns = len(csv_columns)
output_size = 1
# -------------------------


from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise, Resize
)
from albumentations.pytorch import ToTensorV2

class RSNADataset(Dataset):
    def __init__(self, dataframe, vertical_flip, horizontal_flip, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

        self.positive_transform = Compose([
            RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0)),
            HorizontalFlip(p=self.horizontal_flip),
            VerticalFlip(p=self.vertical_flip),
            ShiftScaleRotate(rotate_limit=45, scale_limit=0.3, shift_limit=0.2, p=0.7),
            RandomBrightnessContrast(p=0.7),
            HueSaturationValue(p=0.5),
            Blur(blur_limit=3, p=0.3),
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ToTensorV2()
        ])

        self.general_transform = Compose([
            RandomResizedCrop(height=224, width=224),
            HorizontalFlip(p=self.horizontal_flip),
            VerticalFlip(p=self.vertical_flip),
            ToTensorV2()
        ])

        self.test_transform = Compose([
            Resize(224, 224),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        # Load PNG image (PIL)
        image = Image.open(row['path']).convert('RGB')  # ensures 3-channel
        
        # Convert to numpy array for albumentations
        image_np = np.array(image)
        
        if self.is_train:
            transform = self.positive_transform if row['cancer'] == 1 else self.general_transform
        else:
            transform = self.test_transform

        transformed = transform(image=image_np)
        image_tensor = transformed['image'].float()  # ensure float32 tensor

        meta = torch.tensor(row[csv_columns].values.astype(np.float32))
        target = torch.tensor(row['cancer'], dtype=torch.float32) if 'cancer' in row else None

        if target is not None:
            return {"image": image_tensor, "meta": meta, "target": target}
        else:
            return {"image": image_tensor, "meta": meta}



def data_to_device(data):
    image = data["image"].to(DEVICE)
    metadata = data["meta"].to(DEVICE)
    target = data.get("target", None)
    
    if target is not None:
        target = target.to(DEVICE)
    
    return image, metadata, target



# Sample data
sample_df = train.head(6)

# Instantiate Dataset object
dataset = RSNADataset(sample_df, vertical_flip, horizontal_flip,
                      is_train=True)
# The Dataloader
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

# Output of the Dataloader
for k, data in enumerate(dataloader):
    image, meta, targets = data_to_device(data)
    print(clr.S + f"Batch: {k}" + clr.E, "\n" +
          clr.S + "Image:" + clr.E, image.shape, "\n" +
          clr.S + "Meta:" + clr.E, meta, "\n" +
          clr.S + "Targets:" + clr.E, targets, "\n" +
          "="*50)


class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        
        # Define Feature part (IMAGE)
        self.features = resnet50(pretrained=True) # 1000 neurons out
        # (metadata)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 500),
                                 nn.LayerNorm(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Linear(1000 + 500, output_size)
        
        
    def forward(self, image, meta, prints=False):
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
            
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        out = self.classification(image_meta_data)
        if prints: print('Out shape:', out.shape)
        
        return out


# Load Model
model_example = ResNet50Network(output_size=output_size, no_columns=no_columns).to(DEVICE)

# Outputs
out = model_example(image, meta, prints=True)

# Criterion example
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] to shape=[3, 1]
loss = criterion_example(out, targets.unsqueeze(1).float()) 
print("="*50)
print(clr.S+'Loss:'+clr.E, loss.item())


class EffNetNetwork(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        
        # Define Feature part (IMAGE)
        self.features = EfficientNet.from_pretrained('efficientnet-b2')
        
        # (CSV)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 
                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Sequential(nn.Linear(1408 + 250, self.output_size))
        
        
    def forward(self, image, meta, prints=False):   
        
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.features.extract_features(image)
        image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
            
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        out = self.classification(image_meta_data)
        if prints: print('Out shape:', out.shape)
        
        return out











# Load Model
model_example2 = EffNetNetwork(output_size=output_size, no_columns=no_columns).to(DEVICE)

# Outputs
out = model_example2(image, meta, prints=True)

# Criterion example
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] to shape=[3, 1]
loss = criterion_example(out, targets.unsqueeze(1).float()) 
print("="*50)
print(clr.S+'Loss:'+clr.E, loss.item())


def add_in_file(text, f):
    
    with open(f'logs_{VERSION}.txt', 'a+') as f:
        print(text, file=f)


def mixup_data(x, meta, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_meta = lam * meta + (1 - lam) * meta[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_meta, y_a, y_b, lam



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss



from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from transformers import get_cosine_schedule_with_warmup
import gc
import datetime as dtime
from time import time
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch


def plot_confusion_matrix(y_true, y_pred, epoch, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Epoch {epoch}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def reset_weights(m):
    # Recursively reset weights of a model (common layers)
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            reset_weights(layer)


def train_folds(model, train_original):
    f = open(f"logs_{VERSION}.txt", "w+")
    os.makedirs("saved_models", exist_ok=True)

    group_fold = StratifiedGroupKFold(n_splits=FOLDS)
    k_folds = group_fold.split(train_original, train_original['cancer'], groups=train_original['patient_id'])

    for i, (train_index, valid_index) in enumerate(k_folds):
        print(clr.S + f"---------- Fold: {i+1} ----------" + clr.E)
        add_in_file(f"---------- Fold: {i+1} ----------", f)

        # Reset weights to fresh state at start of fold
        reset_weights(model)

        best_roc = None
        patience_f = PATIENCE

        train_data = train_original.iloc[train_index].reset_index(drop=True)
        valid_data = train_original.iloc[valid_index].reset_index(drop=True)

        # Debug: print data distribution
        print(f"Fold {i+1}: Train size = {len(train_data)}, Positives = {train_data['cancer'].sum()}, "
              f"Negatives = {len(train_data) - train_data['cancer'].sum()}")
        print(f"Fold {i+1}: Valid size = {len(valid_data)}, Positives = {valid_data['cancer'].sum()}, "
              f"Negatives = {len(valid_data) - valid_data['cancer'].sum()}")

        train_ds = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
        valid_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE1, shuffle=True, num_workers=WORKERS)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE2, shuffle=False, num_workers=WORKERS)

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(0.1 * total_steps)  # 10% warm-up

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        criterion = FocalLoss(alpha=0.25, gamma=2)

        for epoch in range(EPOCHS):
            start_time = time()
            correct = 0
            train_losses = 0

            model.train()
            for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                image, meta, targets = data_to_device(data)

                # Sanity check on targets
                assert not torch.isnan(targets).any(), "NaN detected in targets!"
                assert not torch.isinf(targets).any(), "Inf detected in targets!"

                optimizer.zero_grad()
                mixed_x, mixed_meta, y_a, y_b, lam = mixup_data(image, meta, targets.unsqueeze(1).float())
                out = model(mixed_x, mixed_meta)

                # Sanity check on outputs
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print("Warning: NaN or Inf detected in model output!")
                    continue  # Skip problematic batch

                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

                with torch.autograd.detect_anomaly():
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_losses += loss.item()
                train_preds = torch.round(torch.sigmoid(out))
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

                if k % 10 == 0:
                    print(f"Batch {k}: GPU memory allocated = {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

            train_acc = correct / len(train_index)

            model.eval()
            valid_preds = torch.zeros(size=(len(valid_index), 1), device=DEVICE, dtype=torch.float32)

            tta_steps = 5
            with torch.no_grad():
                for k, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, meta, targets = data_to_device(data)
                    batch_size = image.size(0)
                    batch_preds = torch.zeros((batch_size, 1), device=DEVICE)

                    for t in range(tta_steps):
                        tta_images = []
                        for b in range(batch_size):
                            idx = k * batch_size + b
                            if idx >= len(valid_data):
                                continue
                            sample = RSNADataset(valid_data.iloc[[idx]], vertical_flip, horizontal_flip, is_train=True)[0]
                            tta_images.append(sample["image"].unsqueeze(0))

                        if len(tta_images) == 0:
                            continue  # Skip empty TTA batch

                        tta_batch = torch.cat(tta_images).to(DEVICE)
                        meta_repeated = meta.repeat(tta_steps, 1)[:tta_batch.size(0)]
                        out = model(tta_batch, meta_repeated)
                        batch_preds += torch.sigmoid(out)

                    avg_preds = batch_preds / tta_steps
                    valid_preds[k * batch_size: k * batch_size + batch_size] = avg_preds

            y_true = valid_data['cancer'].values
            y_score = valid_preds.cpu().numpy()

            # Sanitize predictions to remove NaNs/Infs
            y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
            val_preds_bin = np.round(y_score)

            # Check for NaNs/Infs before plotting confusion matrix
            if np.isnan(y_true).any() or np.isnan(val_preds_bin).any() or np.isinf(val_preds_bin).any():
                print("Warning: NaN or Inf detected in true or predicted labels, skipping confusion matrix plot.")
            else:
                plot_confusion_matrix(y_true, val_preds_bin, epoch + 1, i + 1)

            valid_acc = accuracy_score(y_true, val_preds_bin)
            valid_roc = 0.0 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
            val_precision = precision_score(y_true, val_preds_bin)
            val_recall = recall_score(y_true, val_preds_bin)
            val_f1 = f1_score(y_true, val_preds_bin)
            try:
                val_loss = log_loss(y_true, y_score)
            except Exception:
                val_loss = float('nan')

            print(f"loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f} - precision: {val_precision:.3f} - recall: {val_recall:.3f} - f1score: {val_f1:.3f}")

            duration = str(dtime.timedelta(seconds=time() - start_time))[:7]
            final_logs = '{} | Epoch: {}/{} | Loss: {:.4} | Acc_tr: {:.3} | Acc_vd: {:.3} | ROC: {:.3}'.format(
                duration, epoch + 1, EPOCHS, train_losses, train_acc, valid_acc, valid_roc)
            add_in_file(final_logs, f)
            print(final_logs)

            if not best_roc or valid_roc > best_roc:
                best_roc = valid_roc
                patience_f = PATIENCE
                model_name = f"BEST_Fold{i + 1}_Epoch{epoch + 1}_ROC{valid_roc:.3f}.pth"
                torch.save(model.state_dict(), os.path.join("saved_models", model_name))
                with open("saved_models/best_model_name.txt", "w") as f_name:
                    f_name.write(model_name)
                print(f"✅ Best model saved: {model_name}")
            else:
                patience_f -= 1
                if patience_f == 0:
                    print(f"⛔ Early stopping triggered — Best ROC: {best_roc:.4f}")
                    break

        del train_ds, valid_ds, train_loader, valid_loader, image, targets
        gc.collect()
    f.close()



from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from time import time
import os
import gc
import numpy as np
from tqdm import tqdm

def plot_confusion_matrix(y_true, y_pred, epoch, fold):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"⚠️ Skipping confusion matrix for Epoch {epoch}, Fold {fold} due to empty inputs.")
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Epoch {epoch}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def train_folds(model, train_original):
    log_file = open(f"logs_{VERSION}.txt", "w+")
    os.makedirs("saved_models", exist_ok=True)
    group_fold = StratifiedGroupKFold(n_splits=FOLDS)
    k_folds = group_fold.split(train_original,
                               train_original['cancer'],
                               groups=train_original['patient_id'])

    for fold_idx, (train_index, valid_index) in enumerate(k_folds, start=1):
        print(f"---------- Fold: {fold_idx} ----------")
        print(f"---------- Fold: {fold_idx} ----------", file=log_file)

        best_roc = None
        patience_f = PATIENCE

        train_data = train_original.iloc[train_index].reset_index(drop=True)
        valid_data = train_original.iloc[valid_index].reset_index(drop=True)

        train_ds = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
        valid_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE1, shuffle=True, num_workers=WORKERS)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE2, shuffle=False, num_workers=WORKERS)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        criterion = FocalLoss(alpha=0.25, gamma=2)

        tta_steps = 5  # ✅ Defined here now

        for epoch in range(EPOCHS):
            start_time = time()
            correct = 0
            train_losses = 0

            model.train()
            for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                image, meta, targets = data_to_device(data)
                optimizer.zero_grad()
                mixed_x, mixed_meta, y_a, y_b, lam = mixup_data(image, meta, targets.unsqueeze(1).float())
                out = model(mixed_x, mixed_meta)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses += loss.item()
                train_preds = torch.round(torch.sigmoid(out))
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

            train_acc = correct / len(train_index)

            # Validation with TTA
            model.eval()
            valid_preds = torch.zeros(size=(len(valid_data), 1), device=DEVICE, dtype=torch.float32)
            tta_failed = True

            with torch.no_grad():
                for k, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, meta, targets = data_to_device(data)
                    batch_size = image.size(0)
                    batch_preds = torch.zeros((batch_size, 1), device=DEVICE)

                    for t in range(tta_steps):
                        tta_images = []
                        for i in range(batch_size):
                            idx = k * BATCH_SIZE2 + i
                            if idx >= len(valid_data):
                                continue
                            try:
                                sample = RSNADataset(valid_data.iloc[[idx]],
                                                     vertical_flip, horizontal_flip,
                                                     is_train=True)[0]
                                tta_images.append(sample["image"].unsqueeze(0))
                            except Exception as e:
                                print(f"[TTA ERROR] Skipping idx {idx}: {e}")
                                continue
                        if len(tta_images) == 0:
                            continue
                        tta_batch = torch.cat(tta_images).to(DEVICE)
                        out = model(tta_batch, meta[:len(tta_batch)])
                        batch_preds[:len(tta_batch)] += torch.sigmoid(out)

                    if batch_preds.isnan().any():
                        print(f"[WARN] TTA failed for batch {k}, skipping")
                        continue

                    avg_preds = batch_preds / tta_steps
                    start_idx = k * batch_size
                    end_idx = start_idx + avg_preds.size(0)
                    valid_preds[start_idx:end_idx] = avg_preds
                    tta_failed = False

            # Fallback if TTA failed completely
            if tta_failed:
                print("[FALLBACK] TTA failed — using direct inference instead.")
                with torch.no_grad():
                    for k, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                        image, meta, targets = data_to_device(data)
                        out = model(image, meta)
                        valid_preds[k * image.size(0): k * image.size(0) + image.size(0)] = torch.sigmoid(out)

            # Compute metrics
            y_true = valid_data['cancer'].values
            y_score = valid_preds[:len(valid_data)].cpu().numpy()
            val_preds_bin = torch.round(valid_preds[:len(valid_data)].cpu()).numpy()

            print(f"[INFO] Fold {fold_idx}, Epoch {epoch+1} — Validation sample count: {len(y_true)}")

            val_preds_bin_1d = val_preds_bin.squeeze()
            y_score_1d = y_score.squeeze()
            valid_mask = (~np.isnan(y_true)) & (~np.isnan(val_preds_bin_1d)) & (~np.isnan(y_score_1d))

            y_true = y_true[valid_mask]
            val_preds_bin = val_preds_bin_1d[valid_mask]
            y_score = y_score_1d[valid_mask]

            if len(y_true) < 20:
                print(f"⚠️ Too few valid samples in Fold {fold_idx}, Epoch {epoch+1} — skipping confusion matrix.")
            else:
                plot_confusion_matrix(y_true, val_preds_bin, epoch+1, fold_idx)

            valid_acc = accuracy_score(y_true, val_preds_bin)
            valid_roc = 0.0 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
            val_precision = precision_score(y_true, val_preds_bin, zero_division=0)
            val_recall = recall_score(y_true, val_preds_bin, zero_division=0)
            val_f1 = f1_score(y_true, val_preds_bin, zero_division=0)
            try:
                val_loss = log_loss(y_true, y_score)
            except:
                val_loss = float('nan')

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f} - "
                  f"precision: {val_precision:.3f} - recall: {val_recall:.3f} - f1score: {val_f1:.3f}", flush=True)

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f}", file=log_file)

            if not best_roc or valid_roc > best_roc:
                best_roc = valid_roc
                patience_f = PATIENCE
                model_name = f"BEST_Fold{fold_idx}_Epoch{epoch+1}_ROC{valid_roc:.3f}.pth"
                torch.save(model.state_dict(), os.path.join("saved_models", model_name))
                with open("saved_models/best_model_name.txt", "w") as f:
                    f.write(model_name)
                print(f"✅ Best model saved: {model_name}")
            else:
                patience_f -= 1
                if patience_f == 0:
                    print(f"⛔ Early stopping triggered — Best ROC: {best_roc:.4f}")
                    break

        del train_ds, valid_ds, train_loader, valid_loader, image, targets
        gc.collect()

    log_file.close()



FOLDS = 3
EPOCHS = 12
PATIENCE = 12
WORKERS = 8
# iLR = 0.0005
# WD = 0.0
LR_PATIENCE = 1            # 1 model not improving until lr is decreasing
LR_FACTOR = 0.4            # by how much the lr is decreasing

LR = 2e-4       # Lower LR works better with AdamW
WD = 1e-2       # Use non-zero weight decay for regularization



# LR = 0.001
# LR_PATIENCE = 2
# LR_FACTOR = 0.5




BATCH_SIZE1 = 32           # for train
BATCH_SIZE2 = 16           # for valid

VERSION = 'v1'
MODEL = 'resnet50'

model1 = ResNet50Network(output_size=output_size, no_columns=no_columns).to(DEVICE)

# with open("saved_models/best_model_name.txt", "r") as f:
#     best_model_name = f.read().strip()

# model_path = os.path.join("saved_models", best_model_name)
# model1.load_state_dict(torch.load(model_path, map_location=DEVICE))
# print(f"✅ Loaded model: {model_path}")


# ------------------

# Run the cell below to train
#Ran it locally on all data, see the results below
train_folds(model=model1, train_original=train_set)

# Print the logs during training
# f = open('/kaggle/input/rsna-breast-cancer-helper-data/logs_v1.txt', "r")
# contents = f.read()
# print(contents)


FOLDS = 3
EPOCHS = 3
PATIENCE = 3
WORKERS = 8
LR = 0.0005
WD = 0.0
LR_PATIENCE = 1            # 1 model not improving until lr is decreasing
LR_FACTOR = 0.4            # by how much the lr is decreasing

BATCH_SIZE1 = 32           # for train
BATCH_SIZE2 = 16           # for valid

VERSION = 'v2'
MODEL = 'effnet'

model2 = EffNetNetwork(output_size=output_size, no_columns=no_columns).to(DEVICE)

# ------------------

# Run the cell below to train
# Ran it locally on all data, see the results below
train_folds(model=model2, train_original=train_set)

# Print the logs during training
# f = open('/kaggle/input/rsna-breast-cancer-helper-data/logs_v2.txt', "r")
# contents = f.read()
# print(contents)


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os
import gc
from time import time
import datetime as dtime


def plot_confusion_matrix(y_true, y_pred, epoch, fold):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"⚠️ Skipping confusion matrix for Epoch {epoch}, Fold {fold} due to empty inputs.")
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Epoch {epoch}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def train_folds(model, train_original):
    log_file = open(f"logs_{VERSION}.txt", "w+")
    os.makedirs("saved_models", exist_ok=True)

    group_fold = StratifiedGroupKFold(n_splits=FOLDS)
    k_folds = group_fold.split(train_original,
                               train_original['cancer'],
                               groups=train_original['patient_id'])

    for fold_idx, (train_index, valid_index) in enumerate(k_folds, start=1):
        print(f"---------- Fold: {fold_idx} ----------")
        print(f"---------- Fold: {fold_idx} ----------", file=log_file)

        best_roc = None
        patience_f = PATIENCE

        train_data = train_original.iloc[train_index].reset_index(drop=True)
        valid_data = train_original.iloc[valid_index].reset_index(drop=True)

        train_ds = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
        valid_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=False)

        # Pre-create TTA dataset (with augmentation) for validation fold
        valid_tta_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE1, shuffle=True, num_workers=WORKERS)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE2, shuffle=False, num_workers=WORKERS)

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(0.1 * total_steps)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        criterion = FocalLoss(alpha=0.25, gamma=2)

        tta_steps = 5  # Number of TTA augmentations

        for epoch in range(EPOCHS):
            start_time = time()
            model.train()
            correct = 0
            train_losses = 0

            for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                image, meta, targets = data_to_device(data)
                optimizer.zero_grad()
                mixed_x, mixed_meta, y_a, y_b, lam = mixup_data(image, meta, targets.unsqueeze(1).float())
                out = model(mixed_x, mixed_meta)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_losses += loss.item()
                train_preds = torch.round(torch.sigmoid(out))
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

            train_acc = correct / len(train_index)

            model.eval()
            valid_preds = torch.zeros(size=(len(valid_data), 1), device=DEVICE, dtype=torch.float32)

            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, meta, targets = data_to_device(data)
                    batch_size = image.size(0)
                    batch_preds = torch.zeros((batch_size, 1), device=DEVICE)

                    for t in range(tta_steps):
                        tta_images = []
                        # Collect augmented images for TTA
                        for i in range(batch_size):
                            idx = batch_idx * BATCH_SIZE2 + i
                            # Fix: Make sure idx is within valid_tta_ds range
                            if idx >= len(valid_tta_ds):
                                continue
                            sample = valid_tta_ds[idx]
                            tta_images.append(sample["image"].unsqueeze(0))

                        if not tta_images:
                            continue

                        tta_batch = torch.cat(tta_images).to(DEVICE)
                        meta_repeated = meta[:len(tta_batch)]  # Align meta size
                        out = model(tta_batch, meta_repeated)
                        batch_preds[:len(tta_batch)] += torch.sigmoid(out)

                    avg_preds = batch_preds / tta_steps
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + avg_preds.size(0)
                    valid_preds[start_idx:end_idx] = avg_preds

            y_true = valid_data['cancer'].values
            y_score = valid_preds[:len(valid_data)].cpu().numpy()
            val_preds_bin = torch.round(valid_preds[:len(valid_data)].cpu()).numpy()

            val_preds_bin_1d = val_preds_bin.squeeze()
            y_score_1d = y_score.squeeze()
            mask = (~np.isnan(y_true)) & (~np.isnan(val_preds_bin_1d)) & (~np.isnan(y_score_1d))

            y_true = y_true[mask]
            val_preds_bin = val_preds_bin_1d[mask]
            y_score = y_score_1d[mask]

            if len(y_true) >= 20:
                plot_confusion_matrix(y_true, val_preds_bin, epoch + 1, fold_idx)

            valid_acc = accuracy_score(y_true, val_preds_bin)
            valid_roc = 0.0 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
            val_precision = precision_score(y_true, val_preds_bin, zero_division=0)
            val_recall = recall_score(y_true, val_preds_bin, zero_division=0)
            val_f1 = f1_score(y_true, val_preds_bin, zero_division=0)

            try:
                val_loss = log_loss(y_true, y_score)
            except Exception:
                val_loss = float('nan')

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f} - "
                  f"precision: {val_precision:.3f} - recall: {val_recall:.3f} - f1score: {val_f1:.3f}")

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f}", file=log_file)

            if best_roc is None or valid_roc > best_roc:
                best_roc = valid_roc
                patience_f = PATIENCE
                model_name = f"BEST_Fold{fold_idx}_Epoch{epoch+1}_ROC{valid_roc:.3f}.pth"
                torch.save(model.state_dict(), os.path.join("saved_models", model_name))
                with open("saved_models/best_model_name.txt", "w") as f:
                    f.write(model_name)
                print(f"✅ Best model saved: {model_name}")
            else:
                patience_f -= 1
                if patience_f == 0:
                    print(f"⛔ Early stopping triggered — Best ROC: {best_roc:.4f}")
                    break

        del train_ds, valid_ds, train_loader, valid_loader, image, targets
        gc.collect()

    log_file.close()



2nd


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from transformers import get_cosine_schedule_with_warmup
import gc
import datetime as dtime
from time import time
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch


def plot_confusion_matrix(y_true, y_pred, epoch, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Epoch {epoch}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def reset_weights(m):
    # Recursively reset weights of a model (common layers)
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            reset_weights(layer)


def train_folds(model, train_original):
    f = open(f"logs_{VERSION}.txt", "w+")
    os.makedirs("saved_models", exist_ok=True)

    group_fold = StratifiedGroupKFold(n_splits=FOLDS)
    k_folds = group_fold.split(train_original, train_original['cancer'], groups=train_original['patient_id'])

    for i, (train_index, valid_index) in enumerate(k_folds):
        print(clr.S + f"---------- Fold: {i+1} ----------" + clr.E)
        add_in_file(f"---------- Fold: {i+1} ----------", f)

        reset_weights(model)

        best_roc = None
        patience_f = PATIENCE

        train_data = train_original.iloc[train_index].reset_index(drop=True)
        valid_data = train_original.iloc[valid_index].reset_index(drop=True)

        print(f"Fold {i+1}: Train size = {len(train_data)}, Positives = {train_data['cancer'].sum()}, "
              f"Negatives = {len(train_data) - train_data['cancer'].sum()}")
        print(f"Fold {i+1}: Valid size = {len(valid_data)}, Positives = {valid_data['cancer'].sum()}, "
              f"Negatives = {len(valid_data) - valid_data['cancer'].sum()}")

        train_ds = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
        valid_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=False)  # <--- No augmentation on validation

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE1, shuffle=True, num_workers=WORKERS)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE2, shuffle=False, num_workers=WORKERS)

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(0.1 * total_steps)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        criterion = FocalLoss(alpha=0.25, gamma=2)

        tta_steps = 5

        for epoch in range(EPOCHS):
            start_time = time()
            model.train()
            correct = 0
            train_losses = 0

            for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                image, meta, targets = data_to_device(data)

                optimizer.zero_grad()
                mixed_x, mixed_meta, y_a, y_b, lam = mixup_data(image, meta, targets.unsqueeze(1).float())
                out = model(mixed_x, mixed_meta)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_losses += loss.item()
                train_preds = torch.round(torch.sigmoid(out))
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

                if k % 10 == 0:
                    print(f"Batch {k}: GPU memory allocated = {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

            train_acc = correct / len(train_index)

            model.eval()
            valid_preds = torch.zeros(size=(len(valid_data), 1), device=DEVICE, dtype=torch.float32)

            with torch.no_grad():
                for k, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, meta, targets = data_to_device(data)
                    batch_size = image.size(0)
                    batch_preds = torch.zeros((batch_size, 1), device=DEVICE)

                    # TTA: For each sample in batch, generate tta_steps augmented versions and average predictions
                    for t in range(tta_steps):
                        tta_images = []
                        for b in range(batch_size):
                            idx = k * BATCH_SIZE2 + b
                            if idx >= len(valid_data):
                                continue
                            # Manually augment image for TTA
                            sample = RSNADataset(valid_data.iloc[[idx]], vertical_flip, horizontal_flip, is_train=True)[0]
                            tta_images.append(sample["image"].unsqueeze(0))

                        if not tta_images:
                            continue

                        tta_batch = torch.cat(tta_images).to(DEVICE)
                        out = model(tta_batch, meta[:len(tta_batch)])
                        batch_preds[:len(tta_batch)] += torch.sigmoid(out)

                    avg_preds = batch_preds / tta_steps
                    start_idx = k * batch_size
                    valid_preds[start_idx:start_idx + avg_preds.size(0)] = avg_preds

            y_true = valid_data['cancer'].values
            y_score = valid_preds[:len(valid_data)].cpu().numpy()
            val_preds_bin = np.round(y_score)

            mask = (~np.isnan(y_true)) & (~np.isnan(val_preds_bin.squeeze())) & (~np.isnan(y_score.squeeze()))
            y_true = y_true[mask]
            val_preds_bin = val_preds_bin.squeeze()[mask]
            y_score = y_score.squeeze()[mask]

            if len(y_true) >= 20:
                plot_confusion_matrix(y_true, val_preds_bin, epoch + 1, i + 1)

            valid_acc = accuracy_score(y_true, val_preds_bin)
            valid_roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
            val_precision = precision_score(y_true, val_preds_bin, zero_division=0)
            val_recall = recall_score(y_true, val_preds_bin, zero_division=0)
            val_f1 = f1_score(y_true, val_preds_bin, zero_division=0)

            try:
                val_loss = log_loss(y_true, y_score)
            except Exception:
                val_loss = float('nan')

            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f} - "
                  f"precision: {val_precision:.3f} - recall: {val_recall:.3f} - f1score: {val_f1:.3f}")

            duration = str(dtime.timedelta(seconds=time() - start_time))[:7]
            final_logs = '{} | Epoch: {}/{} | Loss: {:.4} | Acc_tr: {:.3} | Acc_vd: {:.3} | ROC: {:.3}'.format(
                duration, epoch + 1, EPOCHS, train_losses, train_acc, valid_acc, valid_roc)
            add_in_file(final_logs, f)
            print(final_logs)

            if best_roc is None or valid_roc > best_roc:
                best_roc = valid_roc
                patience_f = PATIENCE
                model_name = f"BEST_Fold{i + 1}_Epoch{epoch + 1}_ROC{valid_roc:.3f}.pth"
                torch.save(model.state_dict(), os.path.join("saved_models", model_name))
                with open("saved_models/best_model_name.txt", "w") as f_name:
                    f_name.write(model_name)
                print(f"✅ Best model saved: {model_name}")
            else:
                patience_f -= 1
                if patience_f == 0:
                    print(f"⛔ Early stopping triggered — Best ROC: {best_roc:.4f}")
                    break

        del train_ds, valid_ds, train_loader, valid_loader, image, targets
        gc.collect()
    f.close()



3rd


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from transformers import get_cosine_schedule_with_warmup
import gc
import datetime as dtime
from time import time
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

def plot_confusion_matrix(y_true, y_pred, epoch, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Epoch {epoch}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            reset_weights(layer)

def train_folds(model, train_original):
    f = open(f"logs_{VERSION}.txt", "w+")
    os.makedirs("saved_models", exist_ok=True)

    group_fold = StratifiedGroupKFold(n_splits=FOLDS)
    k_folds = group_fold.split(train_original, train_original['cancer'], groups=train_original['patient_id'])

    for i, (train_index, valid_index) in enumerate(k_folds):
        print(clr.S + f"---------- Fold: {i+1} ----------" + clr.E)
        add_in_file(f"---------- Fold: {i+1} ----------", f)

        reset_weights(model)

        best_roc = None
        patience_f = PATIENCE

        train_data = train_original.iloc[train_index].reset_index(drop=True)
        valid_data = train_original.iloc[valid_index].reset_index(drop=True)

        print(f"Fold {i+1}: Train size = {len(train_data)}, Positives = {train_data['cancer'].sum()}, "
              f"Negatives = {len(train_data) - train_data['cancer'].sum()}")
        print(f"Fold {i+1}: Valid size = {len(valid_data)}, Positives = {valid_data['cancer'].sum()}, "
              f"Negatives = {len(valid_data) - valid_data['cancer'].sum()}")

        train_ds = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
        valid_ds = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=True)  # for TTA validation, use is_train=True

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE1, shuffle=True, num_workers=WORKERS)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE2, shuffle=False, num_workers=WORKERS)

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(0.1 * total_steps)  # 10% warm-up

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        criterion = FocalLoss(alpha=0.25, gamma=2)

        tta_steps = 5

        for epoch in range(EPOCHS):
            start_time = time()
            correct = 0
            train_losses = 0

            model.train()
            for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                image, meta, targets = data_to_device(data)

                assert not torch.isnan(targets).any(), "NaN detected in targets!"
                assert not torch.isinf(targets).any(), "Inf detected in targets!"

                optimizer.zero_grad()
                mixed_x, mixed_meta, y_a, y_b, lam = mixup_data(image, meta, targets.unsqueeze(1).float())
                out = model(mixed_x, mixed_meta)

                if torch.isnan(out).any() or torch.isinf(out).any():
                    print("Warning: NaN or Inf detected in model output!")
                    continue

                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

                with torch.autograd.detect_anomaly():
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_losses += loss.item()
                train_preds = torch.round(torch.sigmoid(out))
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

                if k % 10 == 0:
                    print(f"Batch {k}: GPU memory allocated = {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

            train_acc = correct / len(train_index)

            model.eval()
            valid_preds = torch.zeros(size=(len(valid_index), 1), device=DEVICE, dtype=torch.float32)

            with torch.no_grad():
                for k, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    image, meta, targets = data_to_device(data)
                    batch_size = image.size(0)
                    batch_preds = torch.zeros((batch_size, 1), device=DEVICE)

                    for t in range(tta_steps):
                        tta_images = []
                        for b in range(batch_size):
                            idx = k * batch_size + b
                            if idx >= len(valid_data):
                                continue
                            # Use valid_ds created above instead of recreating dataset sample by sample
                            sample = valid_ds[idx]
                            tta_images.append(sample["image"].unsqueeze(0))

                        if not tta_images:
                            print(f"[WARN] Empty TTA batch at batch {k}, skipping TTA for this batch.")
                            continue

                        tta_batch = torch.cat(tta_images).to(DEVICE)
                        meta_repeated = meta[:len(tta_batch)]
                        out = model(tta_batch, meta_repeated)
                        batch_preds[:len(tta_batch)] += torch.sigmoid(out)

                    avg_preds = batch_preds / tta_steps
                    valid_preds[k * batch_size: k * batch_size + batch_size] = avg_preds

            y_true = valid_data['cancer'].values
            y_score = valid_preds.cpu().numpy()

            y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
            val_preds_bin = np.round(y_score)

            if np.isnan(y_true).any() or np.isnan(val_preds_bin).any() or np.isinf(val_preds_bin).any():
                print("Warning: NaN or Inf detected in true or predicted labels, skipping confusion matrix plot.")
            else:
                plot_confusion_matrix(y_true, val_preds_bin, epoch + 1, i + 1)

            valid_acc = accuracy_score(y_true, val_preds_bin)
            valid_roc = 0.0 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
            val_precision = precision_score(y_true, val_preds_bin)
            val_recall = recall_score(y_true, val_preds_bin)
            val_f1 = f1_score(y_true, val_preds_bin)
            try:
                val_loss = log_loss(y_true, y_score)
            except Exception:
                val_loss = float('nan')

            print(f"loss: {train_losses:.4f} - accuracy: {train_acc:.3f} - val_loss: {val_loss:.4f} - val_accuracy: {valid_acc:.3f} - val_auc: {valid_roc:.3f} - precision: {val_precision:.3f} - recall: {val_recall:.3f} - f1score: {val_f1:.3f}")

            duration = str(dtime.timedelta(seconds=time() - start_time))[:7]
            final_logs = '{} | Epoch: {}/{} | Loss: {:.4} | Acc_tr: {:.3} | Acc_vd: {:.3} | ROC: {:.3}'.format(
                duration, epoch + 1, EPOCHS, train_losses, train_acc, valid_acc, valid_roc)
            add_in_file(final_logs, f)
            print(final_logs)

            if not best_roc or valid_roc > best_roc:
                best_roc = valid_roc
                patience_f = PATIENCE
                model_name = f"BEST_Fold{i + 1}_Epoch{epoch + 1}_ROC{valid_roc:.3f}.pth"
                torch.save(model.state_dict(), os.path.join("saved_models", model_name))
                with open("saved_models/best_model_name.txt", "w") as f_name:
                    f_name.write(model_name)
                print(f"✅ Best model saved: {model_name}")
            else:
                patience_f -= 1
                if patience_f == 0:
                    print(f"⛔ Early stopping triggered — Best ROC: {best_roc:.4f}")
                    break

        del train_ds, valid_ds, train_loader, valid_loader, image, targets
        gc.collect()
    f.close()









from fastai.vision.all import *
from fastai.data.all import *
from sklearn.model_selection import StratifiedShuffleSplit
import gc


seed = 42
save_path = '/kaggle/working'
train_size = 0.8
batch_size = 32
image_resize = 256
lr_unfreeze = slice(1e-7, 3e-6)
n_epochs = 15


#https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369267  
def pfbeta_torch(preds, labels, beta=1):
    softmax = torch.nn.Softmax(dim = -1)
    preds = softmax(preds)
    preds = preds[:, 1]
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


base_path = Path('/kaggle/input')
base_images_path = base_path/'rsna-mammography-breast-cancer-detection-png'/'png_images'
base_data_path = base_path/'rsna-breast-cancer-detection'
df = pd.read_csv(base_data_path/'train.csv')
print(df.shape)
print(f"Total Number of patient {len(df['patient_id'].unique())}")
df.head()


only_cc_view_data = df[df['view'] == 'CC'].copy()
print(only_cc_view_data.shape)
print(f'''Number of patient: {len(only_cc_view_data['patient_id'].unique())} 
and patient with more than 2 scans {(only_cc_view_data['patient_id'].value_counts() > 2).sum()}''')
only_cc_view_data.head()


only_cc_view_data['path'] = only_cc_view_data.apply(lambda x: base_images_path/str(x['patient_id'])/(str(x['image_id'])+'.png'), axis = 1)
final_subset = only_cc_view_data.drop_duplicates(subset = ['patient_id', 'laterality'], keep = 'last').copy()
print(final_subset.shape)
print(f'''Number of patient: {len(final_subset['patient_id'].unique())} 
and patient with more than 2 scans {(final_subset['patient_id'].value_counts() > 2).sum()}''')
final_subset.head()


final_subset['is_valid'] = False
strata = StratifiedShuffleSplit(n_splits=2, train_size = train_size, random_state=seed)
for (train_idx, valid_idx) in strata.split(final_subset.index, final_subset['cancer']):
    final_subset.iloc[train_idx, -1] = False
    final_subset.iloc[valid_idx, -1] = True

print(final_subset['is_valid'].value_counts(normalize = True))
final_subset.head()


number_class_0 = (final_subset['cancer'] == 0).sum()
number_class_1 = (final_subset['cancer'] == 1).sum()
weight_class_0 = 1
weight_class_1 = number_class_0//number_class_1
print(f"Cross Entropy weight for class 1: {weight_class_0}, and for class 0: {weight_class_1}")
weights = torch.tensor([weight_class_0, weight_class_1], dtype = torch.float32)
print(weights)


datablock = DataBlock(blocks = (ImageBlock(), CategoryBlock),
                     splitter = ColSplitter(),
                     get_x = ColReader(-2),
                     get_y = ColReader(6),
                     item_tfms = Resize(image_resize, ResizeMethod.Pad, pad_mode = 'zeros'),)


dataloaders = datablock.dataloaders(final_subset, bs = 2*batch_size)
dataloaders.show_batch()


learn = vision_learner(dataloaders, resnet50, loss_func = CrossEntropyLossFlat(weight = weights),
                       metrics = [accuracy, pfbeta_torch]).to_fp16()
learn.fine_tune(3, cbs=[SaveModelCallback(monitor = 'pfbeta_torch', fname = 'resnet50')])
learn.recorder.plot_loss()


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(dataloaders.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7))


interp.plot_top_losses(9, figsize=(15,10))


del learn
torch.cuda.empty_cache()
gc.collect()


learn = vision_learner(dataloaders, resnet50, loss_func = CrossEntropyLossFlat(weight = weights),
                       metrics = [accuracy, pfbeta_torch]).to_fp16()
learn.load('/kaggle/working/models/resnet50')


learn.unfreeze()
learn.lr_find()


learn.fit_one_cycle(n_epochs, lr_unfreeze, wd = 0.1,
                    cbs=[SaveModelCallback(monitor = 'pfbeta_torch', fname = 'resnet50_unfreeze'), 
                        EarlyStoppingCallback(monitor='pfbeta_torch', patience = 4)]) 
learn.recorder.plot_loss()


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(dataloaders.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7))


interp.plot_top_losses(9, figsize=(15,10))

