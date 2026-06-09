!pip install --upgrade scikit-learn
!pip install --upgrade imbalanced-learn


import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
img_dir = os.path.join(data_dir, 'train_images')
processed_dir = '/kaggle/working/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Step 1: Visualize Initial Data
def visualize_initial_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, num_samples_per_class, figsize=(10, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            img = Image.open(img_path)
            axs[class_label, i].imshow(img)
            axs[class_label, i].set_title(f"Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_initial_data()






train_csv = pd.read_csv(train_csv_path)
print(train_csv.shape[0])
print(train_csv['diagnosis'].value_counts())


sizes = train_csv['diagnosis'].value_counts().values
labels = ['No Dr','Moderate DR','Mild Dr','Proliferative DR', 'Sever DR']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%');


import cv2

def smart_resize(img, size=(1024,1024)):
    h, w = img.shape[:2]
    if h > size[0] and w > size[1]:
        interp = cv2.INTER_AREA
    elif h < size[0] and w < size[1]:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation = interp)


def crop_circular(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = gray > 30
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis = 0)
    y1, x1 = coords.max(axis = 0)
    cropped = img[y0:y1, x0:x1]
    return cropped


def ben_graham_preprocess(img, sigmaX = 10):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    return img


img_path = os.path.join(img_dir, f"{train_csv.iloc[0,0]}.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.shape


img_resized = smart_resize(img)
img_resized.shape


img_cropped = crop_circular(img_resized)
img_cropped.shape


img_braham = ben_graham_preprocess(img_cropped)
img_braham.shape


plt.imshow(img_braham);


plt.imshow(img);


def preprocess_image(img_path, target_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Order: Crop -> Resize -> Ben Graham (per Kaggle load_ben_color)
    img = smart_resize(img,(1024, 1024))
    img = crop_circular(img)
    img = ben_graham_preprocess(img, sigmaX=10)
    img = smart_resize(img,(target_size, target_size))
    return img


import pandas as pd
# Step 3: Visualize Processed Data
def visualize_processed_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, 2 * num_samples_per_class, figsize=(20, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            original = cv2.imread(img_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            processed = preprocess_image(img_path)
            
            axs[class_label, 2*i].imshow(original)
            axs[class_label, 2*i].set_title(f"Original Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i].axis('off')
            
            axs[class_label, 2*i + 1].imshow(processed)
            axs[class_label, 2*i + 1].set_title(f"Processed Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i + 1].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_processed_data()


# Step 4: Save Processed Images
def save_processed_images():
    train_csv = pd.read_csv(train_csv_path)
    for _, row in train_csv.iterrows():
        img_path = os.path.join(img_dir, f"{row['id_code']}.png")
        processed_img = preprocess_image(img_path)
        save_path = os.path.join(processed_dir, f"{row['id_code']}.png")
        cv2.imwrite(save_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved {len(train_csv)} images to {processed_dir}")

# Call
save_processed_images()


import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
processed_dir = '/kaggle/working/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(train_csv_path)

# --- Step 1: Train+Val vs Test (80:20) ---
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['diagnosis'],
    random_state=42
)

# --- Step 2: Train vs Val (80:20 of train_val -> 64:16 overall) ---
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,  # 20% of 80% = 16% of total
    stratify=train_val_df['diagnosis'],
    random_state=42
)

# --- Optional: Print class distribution to verify ---
print("Train class distribution:\n", train_df['diagnosis'].value_counts(normalize=True))
print("Val class distribution:\n", val_df['diagnosis'].value_counts(normalize=True))
print("Test class distribution:\n", test_df['diagnosis'].value_counts(normalize=True))

# --- Save CSVs ---
train_df.to_csv(os.path.join(processed_dir, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(processed_dir, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'test_split.csv'), index=False)



import seaborn as sns
import matplotlib.pyplot as plt

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train distribution
sns.countplot(x='diagnosis', data=train_df, ax=axes[0], palette='viridis')
axes[0].set_title('Train Class Distribution')
axes[0].set_xlabel('Diagnosis')
axes[0].set_ylabel('Count')

# Validation distribution
sns.countplot(x='diagnosis', data=val_df, ax=axes[1], palette='magma')
axes[1].set_title('Validation Class Distribution')
axes[1].set_xlabel('Diagnosis')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()



class APTOSDataset(Dataset):
    def __init__(self, df, processed_dir, transform=None):
        self.data = df
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(self.processed_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        
        if self.transform:
            if type(self.transform) == list:
                if label.item() in [1,3, 4]:
                    image = self.transform[1](image = image)['image']
                else:
                    image = self.transform[0](image = image)['image']
            else:
                image = self.transform(image = image)['image']
        
        return image, label






import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Base (normal) transformation ===
base_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, border_mode=0, p=0.6),  # small safe rotation
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, border_mode=0, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.MedianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# === Strong (for minority classes) transformation ===
strong_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=40, border_mode=0, p=0.8),  # stronger rotation
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.07, rotate_limit=0, border_mode=0, p=0.5),
    A.OpticalDistortion(distort_limit=0.03, shift_limit=0.02, border_mode=0, p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.MedianBlur(blur_limit=5, p=0.3),
    A.GaussNoise(var_limit=(10.0, 25.0), p=0.4),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_transforms = [base_transform, strong_transform]

val_test_transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])



# Step 4: Visualize Augmented Images
def visualize_augmented_data(df, processed_dir, transform, num_samples_per_class=2):
    dataset = APTOSDataset(df, processed_dir, transform=transform)
    fig, axs = plt.subplots(5, num_samples_per_class, figsize=(15, 25))
    for class_label in range(5):
        class_samples = df[df['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_name = f"{row['id_code']}.png"
            img_path = os.path.join(processed_dir, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Apply transforms
            augmented = transform[0](image = img)
            img_tensor = augmented["image"]
            # Convert back to PIL for display (denormalize)
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_array = np.clip(img_array, 0, 1)
            axs[class_label, i].imshow(img_array)
            axs[class_label, i].set_title(f"Augmented Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
            axs[class_label, i].imshow(img)
            axs[class_label, i].set_title(f"Not Augmented Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_augmented_data(train_df, processed_dir, train_transforms)


train_dataset = APTOSDataset(train_df, processed_dir, transform=train_transforms)
val_dataset = APTOSDataset(val_df, processed_dir, transform=val_test_transforms)
test_dataset = APTOSDataset(test_df, processed_dir, transform=val_test_transforms)




num_workers = min(6, os.cpu_count() or multiprocessing.cpu_count())
print(f"ğŸ§  Using {num_workers} workers")


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True,)


# Compute class weights for sampling and loss
def get_class_weights(df):
    class_counts = df['diagnosis'].value_counts().sort_index()
    total_samples = len(df)
    num_classes = 5
    weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights

class_weights = get_class_weights(train_df)
loss_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import f1_score
import random


import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha = None, gamma = 2.0, reduction ='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, reduction = 'none')
        pt = logpt.exp()
        if self.alpha is not None:
            at = self.alpha[targets].to(inputs.device)
            logpt = at * logpt
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reducton == 'sum':
            return loss.sum()
        return loss
alpha = torch.tensor(loss_weights/ loss_weights.sum(), dtype=torch.float32)
criterion = FocalLoss(alpha=alpha, gamma = 1.0)


# # === Dropout Model ===
# class DenseNetWithDropout(nn.Module):
#     def __init__(self, num_classes=5, dropout_rate=0.4):
#         super(DenseNetWithDropout, self).__init__()
#         base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
#         self.features = base.features
#         num_features = base.classifier.in_features
#         self.dropout = nn.Dropout(dropout_rate)
#         self.classifier = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = nn.functional.relu(x, inplace=True)
#         x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.classifier(x)
#         return x

# # === MixUp Function ===
# def mixup_data(x, y, alpha=0.4):
#     """Apply MixUp with probability proportional to minority class."""
#     if alpha <= 0:
#         return x, y, 1.0

#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


# def mixup_criterion(criterion, preds, y_a, y_b, lam):
#     return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# # === Device ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BEST_MODEL_PATH = "best_f1_model.pth"

# # === Initialize ===
# model = DenseNetWithDropout(num_classes=5, dropout_rate=0.4).to(device)
# criterion = FocalLoss(alpha=alpha, gamma=2.0)


# # === Helper Training Function ===
# def train_phase(model, train_loader, val_loader, lr, unfrozen_layers, num_epochs, best_f1):
#     print(f"\nğŸš€ Starting Phase with LR={lr} | Unfrozen layers: {unfrozen_layers}")

#     # Freeze all first
#     for name, param in model.named_parameters():
#         param.requires_grad = False

#     # Unfreeze specified parts
#     for layer_name in unfrozen_layers:
#         for name, param in model.named_parameters():
#             if layer_name in name:
#                 param.requires_grad = True

#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)

#     for epoch in tqdm(range(num_epochs), desc=f"Phase ({unfrozen_layers})"):
#         # --- TRAIN ---
#         model.train()
#         running_loss, preds, labels = 0.0, [], []

#         for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
#             imgs, lbls = imgs.to(device), lbls.to(device)
#             optimizer.zero_grad()

#             # === Apply MixUp augmentation ===
#             # More probability for minority classes
#             mix_prob = 0.6 if torch.rand(1).item() < 0.6 else 0.2
#             if random.random() < mix_prob:
#                 imgs, y_a, y_b, lam = mixup_data(imgs, lbls, alpha=0.4)
#                 outputs = model(imgs)
#                 loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
#             else:
#                 outputs = model(imgs)
#                 loss = criterion(outputs, lbls)

#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
#             optimizer.step()

#             running_loss += loss.item()
#             preds.extend(outputs.argmax(1).detach().cpu().numpy())
#             labels.extend(lbls.cpu().numpy())

#         train_loss = running_loss / len(train_loader)
#         train_f1 = f1_score(labels, preds, average='macro')

#         # --- VALIDATE ---
#         model.eval()
#         val_loss, val_preds, val_labels = 0.0, [], []
#         with torch.no_grad():
#             for imgs, lbls in val_loader:
#                 imgs, lbls = imgs.to(device), lbls.to(device)
#                 outputs = model(imgs)
#                 loss = criterion(outputs, lbls)
#                 val_loss += loss.item()
#                 val_preds.extend(outputs.argmax(1).cpu().numpy())
#                 val_labels.extend(lbls.cpu().numpy())

#         val_loss /= len(val_loader)
#         val_f1 = f1_score(val_labels, val_preds, average='macro')

#         print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Loss={val_loss:.4f}")

#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             torch.save(model.state_dict(), BEST_MODEL_PATH)
#             print(f"âœ… Model saved! New best F1: {best_f1:.4f}")

#         scheduler.step(val_f1)

#     return best_f1





# # === PHASES ===
# best_f1 = 0.0

# # Phase 1: Train classifier only
# best_f1 = train_phase(model, train_loader, val_loader, lr=1e-4, unfrozen_layers=["classifier"], num_epochs=10, best_f1=best_f1)

# # Phase 2: Unfreeze denseblock4 + classifier
# best_f1 = train_phase(model, train_loader, val_loader, lr=5e-5, unfrozen_layers=["denseblock4", "classifier"], num_epochs=15, best_f1=best_f1)

# # Phase 3: Unfreeze denseblock3â€“4 + classifier
# best_f1 = train_phase(model, train_loader, val_loader, lr=2e-5, unfrozen_layers=["denseblock3", "denseblock4", "classifier"], num_epochs=20, best_f1=best_f1)

# # Phase 4: Fine-tune entire model
# best_f1 = train_phase(model, train_loader, val_loader, lr=1e-5, unfrozen_layers=["features", "classifier"], num_epochs=20, best_f1=best_f1)

# print(f"\nğŸ�� Training Complete! Best Macro F1: {best_f1:.4f}")


# === ResNet with Dropout ===
class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super(ResNetWithDropout, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = base.fc.in_features
        base.fc = nn.Identity()  # remove the original classifier
        self.backbone = base
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)


# === MixUp ===
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.notebook import tqdm
import random
import gc
from sklearn.metrics import f1_score

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_f1_model.pth"

# === Initialize Model ===
model = ResNetWithDropout(num_classes=5, dropout_rate=0.4).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2.0)


# === Helper Training Function (Memory Safe) ===
def train_phase(model, train_loader, val_loader, lr, unfrozen_layers, num_epochs, best_f1):
    print(f"\nğŸš€ Starting Phase with LR={lr} | Unfrozen layers: {unfrozen_layers}")

    # --- Freeze / Unfreeze layers ---
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer_name in unfrozen_layers:
        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    try:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)
    except TypeError:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)

    # === Training Loop ===
    for epoch in tqdm(range(num_epochs), desc=f"Phase {unfrozen_layers}"):
        model.train()
        running_loss, preds, labels = 0.0, [], []

        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # --- MixUp augmentation ---
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, lbls, alpha=0.4)
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            preds.extend(outputs.detach().argmax(1).cpu().numpy())
            labels.extend(lbls.detach().cpu().numpy())

            # --- Clear per-batch cache ---
            del imgs, lbls, outputs, loss
            torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(labels, preds, average='macro')

        # === VALIDATION ===
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

                # Free up GPU memory each batch
                del imgs, lbls, outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Loss={val_loss:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"âœ… Model saved! New best F1: {best_f1:.4f}")

        scheduler.step(val_f1)

        # --- Cleanup after each epoch ---
        gc.collect()
        torch.cuda.empty_cache()

    return best_f1



# === PHASES ===
best_f1 = 0.0

# Phase 1: Train classifier only (FC)
best_f1 = train_phase(model, train_loader, val_loader, lr=1e-4,
                      unfrozen_layers=["fc"], num_epochs=10, best_f1=best_f1)

# Phase 2: Unfreeze layer4 + fc
best_f1 = train_phase(model, train_loader, val_loader, lr=5e-5,
                      unfrozen_layers=["layer4", "fc"], num_epochs=15, best_f1=best_f1)

# Phase 3: Unfreeze layer3â€“4 + fc
best_f1 = train_phase(model, train_loader, val_loader, lr=2e-5,
                      unfrozen_layers=["layer3", "layer4", "fc"], num_epochs=20, best_f1=best_f1)

# Phase 4: Fine-tune entire model
best_f1 = train_phase(model, train_loader, val_loader, lr=1e-5,
                      unfrozen_layers=["layer1", "layer2", "layer3", "layer4", "fc"], num_epochs=20, best_f1=best_f1)

print(f"\nğŸ�� Training Complete! Best Macro F1: {best_f1:.4f}")


feature_dir = '/kaggle/working/features'


import torch
import numpy as np

# Extract features
model.eval()
def extract_features(loader, model, save_path):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images).cpu().numpy().reshape(len(images), -1)
            features.append(feats)
            labels.append(lbls.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    np.savez(save_path, features=features, labels=labels)
    return features, labels

# Extract and save
train_features, train_labels = extract_features(train_loader, model, os.path.join(feature_dir, 'train_features.npz'))
val_features, val_labels = extract_features(val_loader, model, os.path.join(feature_dir, 'val_features.npz'))
test_features, test_labels = extract_features(test_loader, model, os.path.join(feature_dir, 'test_features.npz'))
print(f"Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")


import os
import torch
import numpy as np

def extract_features(loader, model, device, save_path):
    """
    Extracts features from a model using the given DataLoader.
    Saves them as a .npz file with 'features' and 'labels' arrays.
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            lbls = lbls.cpu().numpy()

            # Forward pass
            outputs = model(images)

            # Flatten features safely
            feats = outputs.detach().cpu().numpy()
            feats = feats.reshape(feats.shape[0], -1)

            all_features.append(feats)
            all_labels.append(lbls)

    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, features=all_features, labels=all_labels)
    print(f"âœ… Saved features to: {save_path}")
    print(f"   â†’ Features shape: {all_features.shape} | Labels shape: {all_labels.shape}")

    return all_features, all_labels


# === Example usage ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)

train_features, train_labels = extract_features(train_loader, model, device, os.path.join(feature_dir, 'train_features.npz'))
val_features, val_labels = extract_features(val_loader, model, device, os.path.join(feature_dir, 'val_features.npz'))
test_features, test_labels = extract_features(test_loader, model, device, os.path.join(feature_dir, 'test_features.npz'))

print(f"\nğŸ“Š Train features: {train_features.shape} | Val features: {val_features.shape} | Test features: {test_features.shape}")



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from scipy.stats import randint, loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
import joblib

# === Visualization Settings ===
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)
colors = cycle(plt.cm.tab10.colors)

# === Paths ===
feature_dir = 'features'
output_dir = "tuned_classifiers"
roc_dir = os.path.join("images", "tunedModel")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(roc_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# === Load Data ===
try:
    train_data = np.load(os.path.join(feature_dir, 'train_features.npz'))
    val_data = np.load(os.path.join(feature_dir, 'val_features.npz'))
    test_data = np.load(os.path.join(feature_dir, 'test_features.npz'))
except FileNotFoundError:
    print(f"â�Œ Error: Feature files not found in {feature_dir}")
    exit()

train_features, train_labels = train_data['features'], train_data['labels']
val_features, val_labels = val_data['features'], val_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']
print(f"Training samples: {len(train_features)}, Validation: {len(val_features)}, Test: {len(test_features)}")




# Step 1: Expanded classifiers + param distributions
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import randint, loguniform

# sklearn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV

# Try to import gradient boosting libs if available (optional)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# --- Base classifiers (start with your original set, then extended) ---
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'SVM': SVC(probability=True, random_state=42),                # kernel SVM
    'CalibratedSVM': CalibratedClassifierCV(SVC(probability=True, random_state=42), cv=3),  # better calibrated probs
    'LinearSVC': LinearSVC(random_state=42, max_iter=2000),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'Ridge': RidgeClassifier(random_state=42),
    'SGD': SGDClassifier(max_iter=1000, random_state=42),
    'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'GaussianNB': GaussianNB(),
}

# Add gradient boosting models if available
if XGBClassifier is not None:
    classifiers['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, n_jobs=-1, random_state=42)
else:
    print("XGBoost not available â€” skipping XGBoost.")

# if LGBMClassifier is not None:
#     classifiers['LightGBM'] = LGBMClassifier(
#     objective='binary',
#     boosting_type='gbdt',
#     learning_rate=0.05,
#     num_leaves=31,
#     n_estimators=200,
#     max_depth=-1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     n_jobs=min(6, os.cpu_count()),
#     random_state=42
# )
# else:
#     print("LightGBM not available â€” skipping LightGBM.")

if CatBoostClassifier is not None:
    classifiers['CatBoost'] = CatBoostClassifier(verbose=0, task_type="CPU", random_state=42)
else:
    print("CatBoost not available â€” skipping CatBoost.")


# === Hyperparameter distributions (for RandomizedSearchCV inside pipeline) ===
# All hyperparams are prefixed with 'model__' since we use Pipeline([..., ('model', clf)])
param_distributions = {
    'DecisionTree': {
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    },
    'RandomForest': {
        'model__n_estimators': randint(100, 400),
        'model__max_depth': [10, 20, 40, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    },
    'ExtraTrees': {
        'model__n_estimators': randint(100, 400),
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    },
    'SVM': {
        'model__C': loguniform(0.1, 10),
        'model__kernel': ['rbf', 'linear'],
        'model__gamma': ['scale', 'auto']
    },
    'CalibratedSVM': {
        'model__estimator__C': loguniform(0.1, 10),
        'model__estimator__kernel': ['rbf', 'linear'],
        'model__estimator__gamma': ['scale', 'auto']
    },
    'LinearSVC': {
        'model__C': loguniform(0.01, 10),
        'model__loss': ['squared_hinge'],
        'model__tol': [1e-4, 1e-3, 1e-2]
    },
    'KNN': {
        'model__n_neighbors': randint(3, 12),
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },
    'LogisticRegression': {
        'model__C': loguniform(0.01, 10),
        'model__solver': ['lbfgs', 'liblinear'],
        'model__penalty': ['l2']
    },
    'MLP': {
        'model__hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
        'model__alpha': loguniform(1e-6, 1e-2),
        'model__learning_rate': ['constant', 'adaptive']
    },
    'Ridge': {
        'model__alpha': loguniform(0.01, 100)
    },
    'SGD': {
        'model__loss': ['log_loss', 'modified_huber'],
        'model__penalty': ['l2', 'elasticnet'],
        'model__alpha': loguniform(1e-6, 1e-2)
    },
    'PassiveAggressive': {
        'model__C': loguniform(0.01, 10)
    },
    'GaussianNB': {
        # very few hyperparams but we can tune var_smoothing
        'model__var_smoothing': loguniform(1e-12, 1e-6)
    },
}

# Add gradient boosting hyperparams if present
if 'XGBoost' in classifiers:
    param_distributions['XGBoost'] = {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': randint(3, 8),
        'model__learning_rate': loguniform(1e-3, 0.3),
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    }

if 'LightGBM' in classifiers:
    param_distributions['LightGBM'] = {
        'model__n_estimators': randint(100, 500),
        'model__num_leaves': randint(20, 128),
        'model__learning_rate': loguniform(1e-3, 0.2),
        'model__subsample': [0.6, 0.8, 1.0]
    }

if 'CatBoost' in classifiers:
    param_distributions['CatBoost'] = {
        'model__iterations': randint(200, 800),
        'model__depth': randint(4, 8),
        'model__learning_rate': loguniform(1e-3, 0.2),
        'model__l2_leaf_reg': loguniform(1e-3, 10)
    }

print("Defined classifiers:", list(classifiers.keys()))
print(len(classifiers))



# ==============================================
# ğŸ”¹ Base ML Models Setup (Paper-Consistent)
# ==============================================
import os
import warnings
from scipy.stats import randint, loguniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# 1ï¸�âƒ£ Define all base classifiers from the paper
# ==================================================
classifiers = {
    'SVM': SVC(probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'GaussianNB': GaussianNB(),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'Ridge': RidgeClassifier(random_state=42),
    'SGD': SGDClassifier(max_iter=1000, random_state=42),
    'NearestCentroid': NearestCentroid(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

print("âœ… Defined base classifiers:")
for name in classifiers:
    print(f" - {name}")

# ==================================================
# 2ï¸�âƒ£ Define hyperparameter distributions
# ==================================================
param_distributions = {
    'SVM': {
        'model__C': loguniform(0.1, 10),
        'model__kernel': ['rbf', 'linear'],
        'model__gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'model__C': loguniform(0.01, 10),
        'model__solver': ['lbfgs', 'liblinear'],
        'model__penalty': ['l2']
    },
    'RandomForest': {
        'model__n_estimators': randint(100, 400),
        'model__max_depth': [10, 20, 40, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    },
    'KNN': {
        'model__n_neighbors': randint(3, 12),
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },
    'GaussianNB': {
        'model__var_smoothing': loguniform(1e-12, 1e-6)
    },
    'ExtraTrees': {
        'model__n_estimators': randint(100, 400),
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    },
    'MLP': {
        'model__hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
        'model__alpha': loguniform(1e-6, 1e-2),
        'model__learning_rate': ['constant', 'adaptive']
    },
    'PassiveAggressive': {
        'model__C': loguniform(0.01, 10)
    },
    'Ridge': {
        'model__alpha': loguniform(0.01, 100)
    },
    'SGD': {
        'model__loss': ['log_loss', 'modified_huber'],
        'model__penalty': ['l2', 'elasticnet'],
        'model__alpha': loguniform(1e-6, 1e-2)
    },
    'NearestCentroid': {
        'model__metric': ['euclidean', 'manhattan']
    },
    'DecisionTree': {
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5)
    }
}

print(f"\nâœ… Total classifiers: {len(classifiers)}")




# === Helper Functions ===

def plot_roc_curve(model_name, model, X, y, save_dir):
    """Plot ROC Curve for multiclass classification."""
    if not hasattr(model, "predict_proba"):
        print(f"âš ï¸� {model_name} does not support predict_proba, skipping ROC.")
        return

    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    y_score = model.predict_proba(X)

    plt.figure(figsize=(7, 6))
    for i, color in zip(range(len(classes)), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {classes[i]} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc.png"))
    plt.close()

def build_pipeline(model):
    """Build a pipeline with SMOTEENN, StandardScaler, PCA, and model."""
    return Pipeline([
        ('smoteenn', SMOTEENN(random_state=42, n_jobs=-1)),
        ('scaler', 'passthrough'),  # scaler handled inside PCA standardization
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('model', model)
    ])

# === Cross-Validation Setup ===
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Train & Tune Models ===




# === Results & Storage Setup ===
results = []
best_models = {}
val_meta_features = {}   # store validation probabilities for stacking
test_meta_features = {}  # store test probabilities for stacking

for name, clf in classifiers.items():
    print(f"\nğŸ”¹ Tuning {name}...")

    # === Build pipeline and parameter grid ===
    pipeline = build_pipeline(clf)
    params = param_distributions.get(name, {})

    # === Randomized Search ===
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=20,
        cv=cv_strategy,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # === Fit model ===
    search.fit(train_features, train_labels)
    best_model = search.best_estimator_
    best_models[name] = best_model

    print(f"âœ… Best Params for {name}: {search.best_params_}")

    # === Evaluate ===
    val_pred = best_model.predict(val_features)
    test_pred = best_model.predict(test_features)

    val_f1 = f1_score(val_labels, val_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)

    results.append((name, val_f1, test_f1))
    print(f"Validation F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    # === Predict probabilities for stacking ===
    if hasattr(best_model, "predict_proba"):
        val_proba = best_model.predict_proba(val_features)
        test_proba = best_model.predict_proba(test_features)
    else:
        # Fallback: one-hot encode predictions if no predict_proba available
        classes = np.unique(train_labels)
        val_preds = best_model.predict(val_features)
        test_preds = best_model.predict(test_features)

        val_proba = np.zeros((len(val_preds), len(classes)))
        test_proba = np.zeros((len(test_preds), len(classes)))

        for i, c in enumerate(classes):
            val_proba[val_preds == c, i] = 1
            test_proba[test_preds == c, i] = 1

    # === Save model and prediction probabilities ===
    joblib.dump(best_model, os.path.join(output_dir, f"{name}_tuned.pkl"))
    np.save(os.path.join(output_dir, f"{name}_val_proba.npy"), val_proba)
    np.save(os.path.join(output_dir, f"{name}_test_proba.npy"), test_proba)

    # Store for in-memory stacking combination
    val_meta_features[name] = val_proba
    test_meta_features[name] = test_proba

    # === Plot ROC curve (optional but recommended) ===
    plot_roc_curve(name, best_model, test_features, test_labels, roc_dir)

# === Summary Visualization ===
results = sorted(results, key=lambda x: x[2], reverse=True)
names, val_scores, test_scores = zip(*results)

plt.figure(figsize=(10, 5))
x = np.arange(len(names))
plt.bar(x - 0.15, val_scores, width=0.3, label='Validation F1')
plt.bar(x + 0.15, test_scores, width=0.3, label='Test F1')
plt.xticks(x, names, rotation=30, ha='right')
plt.ylabel("F1 Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(roc_dir, "model_comparison.png"))
plt.show()

print("\nğŸ�� Training complete! Models and probabilities saved in:", output_dir)

# === Optional: Combine probabilities for stacking (save meta features) ===
val_stack = np.hstack([val_meta_features[n] for n in names])
test_stack = np.hstack([test_meta_features[n] for n in names])

np.savez(os.path.join(output_dir, "stack_features.npz"),
         X_val=val_stack, X_test=test_stack,
         y_val=val_labels, y_test=test_labels)

print("âœ… Saved stacked features for meta model training.")



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import numpy as np
import os

# === Directories ===
stack_dir = "/kaggle/working/stacking"
os.makedirs(stack_dir, exist_ok=True)

# === Load Base Models ===
base_models = {}
for name in classifiers.keys():
    model_path = os.path.join(output_dir, f"{name}_tuned.pkl")
    if os.path.exists(model_path):
        base_models[name] = joblib.load(model_path)
print(f"Loaded {len(base_models)} base models: {list(base_models.keys())}")

# === Create Stacked Features ===
def get_model_preds(models, X):
    preds = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
        else:
            # If model doesn't have predict_proba, use label encoding
            y_pred = model.predict(X)
            p = np.eye(len(np.unique(y_pred)))[y_pred]
        preds.append(p)
    return np.hstack(preds)

X_train_stack = get_model_preds(base_models, train_features)
X_val_stack = get_model_preds(base_models, val_features)
X_test_stack = get_model_preds(base_models, test_features)

y_train_stack = train_labels
y_val_stack = val_labels
y_test_stack = test_labels

np.savez(os.path.join(stack_dir, "stack_features.npz"),
         train=X_train_stack, val=X_val_stack, test=X_test_stack,
         y_train=y_train_stack, y_val=y_val_stack, y_test=y_test_stack)

print(f"Stacked features shapes: train={X_train_stack.shape}, val={X_val_stack.shape}, test={X_test_stack.shape}")







# === Define Meta MLP ===
class MetaMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === Prepare Data ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train_stack, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_stack, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_stack, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_stack, dtype=torch.long).to(device)
y_val_t = torch.tensor(y_val_stack, dtype=torch.long).to(device)
y_test_t = torch.tensor(y_test_stack, dtype=torch.long).to(device)



# === Initialize Meta Model ===
input_dim = X_train_stack.shape[1]
num_classes = len(np.unique(y_train_stack))
meta_model = MetaMLP(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_model.parameters(), lr=1e-4)
best_val_f1 = 0

# === Train Meta Model ===
EPOCHS = 50
for epoch in range(EPOCHS):
    meta_model.train()
    optimizer.zero_grad()
    outputs = meta_model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    meta_model.eval()
    with torch.no_grad():
        val_outputs = meta_model(X_val_t)
        val_preds = val_outputs.argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val_stack, val_preds, average='weighted', zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(meta_model.state_dict(), os.path.join(stack_dir, "best_meta_mlp.pth"))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}")

print(f"âœ… Training complete! Best Val F1: {best_val_f1:.4f}")


# === Evaluate on Test Set ===
meta_model.load_state_dict(torch.load(os.path.join(stack_dir, "best_meta_mlp.pth")))
meta_model.eval()
with torch.no_grad():
    test_outputs = meta_model(X_test_t)
    test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
    test_preds = np.argmax(test_probs, axis=1)

test_f1 = f1_score(y_test_stack, test_preds, average='weighted', zero_division=0)
print(f"ğŸ�� Meta MLP Test F1: {test_f1:.4f}")

# === Optional: ROC Curve ===
from sklearn.metrics import roc_curve, auc, label_binarize
import matplotlib.pyplot as plt

y_bin = label_binarize(y_test_stack, classes=list(range(num_classes)))
plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], test_probs[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("Meta MLP ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(stack_dir, "meta_mlp_roc.png"))
plt.close()
print("ğŸ“Š ROC curve saved.")



import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

# --- Paths ---
output_dir = "tuned_classifiers"  # where base models were saved
stack_dir = "stacking"
os.makedirs(stack_dir, exist_ok=True)

# --- Load train/val/test features & labels (if not already in scope) ---
# train_features, train_labels = ...
# val_features, val_labels = ...
# test_features, test_labels = ...

# --- Determine number of classes globally ---
classes = np.unique(train_labels)
num_classes = len(classes)
print("Num classes:", num_classes, "Classes:", classes)

# --- Load base models that were saved earlier ---
base_models = {}
for fname in os.listdir(output_dir):
    if fname.endswith("_tuned.pkl"):
        name = fname.replace("_tuned.pkl", "")
        path = os.path.join(output_dir, fname)
        base_models[name] = joblib.load(path)
print("Loaded models:", list(base_models.keys()))

# --- Robust function to get fixed-size preds per model ---
def model_probs_fixed(model, X, num_classes):
    """
    Return array shape (n_samples, num_classes).
    If model has predict_proba -> use it.
    Else: use predict() and convert to one-hot of fixed size num_classes.
    """
    n = X.shape[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # Some sklearn multiclass may return shape (n, k) where k==num_classes, OK.
        # If it's not num_classes (weird), we expand/trim safely:
        if probs.shape[1] != num_classes:
            # Create fixed array and try to map by label order if model has classes_
            fixed = np.zeros((n, num_classes), dtype=float)
            if hasattr(model, "classes_"):
                # model.classes_ gives the class labels ordering of columns
                for idx, cls in enumerate(model.classes_):
                    if cls in classes:
                        pos = np.where(classes == cls)[0][0]
                        fixed[:, pos] = probs[:, idx]
                return fixed
            else:
                # Fallback: if different size and no classes_ attribute, pad/truncate
                if probs.shape[1] < num_classes:
                    fixed[:, :probs.shape[1]] = probs
                    return fixed
                else:
                    return probs[:, :num_classes]
        return probs
    else:
        # No predict_proba: use predict and one-hot into num_classes columns
        preds = model.predict(X)
        onehot = np.zeros((n, num_classes), dtype=float)
        # Map class label -> index in global classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            idx = class_to_idx.get(p)
            if idx is None:
                # should not happen, but guard:
                continue
            onehot[i, idx] = 1.0
        return onehot

# --- Build stacked features (concatenate probs/onehots for each base model) ---
def get_stacked_features(models_dict, X):
    parts = []
    for name, mdl in models_dict.items():
        probs = model_probs_fixed(mdl, X, num_classes)
        # sanity check shape
        if probs.shape[1] != num_classes:
            raise ValueError(f"Model {name} produced {probs.shape[1]} cols, expected {num_classes}")
        parts.append(probs)
    return np.hstack(parts)

X_train_stack = get_stacked_features(base_models, train_features)
X_val_stack   = get_stacked_features(base_models, val_features)
X_test_stack  = get_stacked_features(base_models, test_features)

print("Stack shapes =>",
      "train:", X_train_stack.shape,
      "val:", X_val_stack.shape,
      "test:", X_test_stack.shape)

# --- Save stacked features (optional) ---
np.savez(os.path.join(stack_dir, "stack_features_fixed.npz"),
         X_train=X_train_stack, X_val=X_val_stack, X_test=X_test_stack,
         y_train=train_labels, y_val=val_labels, y_test=test_labels)

# --- Meta MLP (PyTorch) ---
class MetaMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train_stack.shape[1]  # this must equal len(base_models) * num_classes
print("Meta input_dim:", input_dim)
meta_model = MetaMLP(input_dim, num_classes).to(device)

# --- Prepare tensors ---
X_train_t = torch.tensor(X_train_stack, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val_stack,   dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test_stack,  dtype=torch.float32).to(device)

y_train_t = torch.tensor(train_labels, dtype=torch.long).to(device)
y_val_t   = torch.tensor(val_labels,   dtype=torch.long).to(device)
y_test_t  = torch.tensor(test_labels,  dtype=torch.long).to(device)

# --- Training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_model.parameters(), lr=1e-4)
best_val_f1 = 0.0
best_path = os.path.join(stack_dir, "best_meta_mlp.pth")
EPOCHS = 50
batch_size = 64

# Simple mini-batch training loop (optional batching)
dataset_train = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader_stack = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

for epoch in range(EPOCHS):
    meta_model.train()
    running_loss = 0.0
    for xb, yb in train_loader_stack:
        optimizer.zero_grad()
        out = meta_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(meta_model.parameters(), 2.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(dataset_train)

    # Validation
    meta_model.eval()
    with torch.no_grad():
        v_out = meta_model(X_val_t)
        v_preds = v_out.argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val_t.cpu().numpy(), v_preds, average='weighted', zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(meta_model.state_dict(), best_path)

    if (epoch+1) % 10 == 0 or epoch==0:
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_f1: {val_f1:.4f}")

print("Best val F1:", best_val_f1)

# Load best and evaluate on test
meta_model.load_state_dict(torch.load(best_path, map_location=device))
meta_model.eval()
with torch.no_grad():
    t_out = meta_model(X_test_t)
    t_probs = torch.softmax(t_out, dim=1).cpu().numpy()
    t_preds = t_probs.argmax(1)
test_f1 = f1_score(y_test_t.cpu().numpy(), t_preds, average='weighted', zero_division=0)
print("Meta MLP Test F1:", test_f1)





