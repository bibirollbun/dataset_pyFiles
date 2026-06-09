import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm



# --- 1. Configuration and Setup ---
DATA_PATH = '/kaggle/input/siim-isic-melanoma-classification/'
IMAGE_PATH = os.path.join(DATA_PATH, 'jpeg/train')
CSV_PATH = os.path.join(DATA_PATH, 'train.csv')

# New directory for our augmented dataset
AUG_DATA_PATH = '/kaggle/working/original_augmented/'
AUG_IMAGE_PATH = os.path.join(AUG_DATA_PATH, 'jpeg')
AUG_CSV_PATH = os.path.join(AUG_DATA_PATH, 'augmented_train.csv')

# Create the directories, removing any old ones first
if os.path.exists(AUG_DATA_PATH):
    shutil.rmtree(AUG_DATA_PATH)
os.makedirs(AUG_IMAGE_PATH, exist_ok=True)


# --- 2. Data Loading and Balancing ---
print("Loading and creating a balanced dataset for augmentation...")
df = pd.read_csv(CSV_PATH)
#df.dropna(reset_index=True,inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df


malignant_df = df[df['target'] == 1].copy()
benign_df = df[df['target'] == 0].copy()

# Create initial balanced set
n_malignant = len(malignant_df)
n_sites = benign_df['anatom_site_general_challenge'].nunique()
n_per_site = n_malignant // n_sites
remainder = n_malignant % n_sites

benign_list = []
for site, group in benign_df.groupby('anatom_site_general_challenge'):
    sample_n = n_per_site + 1 if remainder > 0 else n_per_site
    benign_list.append(group.sample(n=sample_n, random_state=42))
    remainder -= 1
sampled_benign_df = pd.concat(benign_list)

balanced_df = pd.concat([malignant_df, sampled_benign_df], ignore_index=True)


# --- 3. Augmentation Process ---
print(f"Augmenting {len(balanced_df)} images to triple the dataset size...")
augmented_data = []
hflip = transforms.RandomHorizontalFlip(p=1.0)
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
    img_name = row['image_name']
    original_img_path = os.path.join(IMAGE_PATH, f"{img_name}.jpg")
    image = Image.open(original_img_path)

    # 1. Original Image, 2. Flipped Image, 3. Color Jittered Image
    images_to_save = [(image, "orig"), (hflip(image), "flipped"), (color_jitter(image), "jittered")]
    for img, suffix in images_to_save:
        new_name = f"{img_name}_{suffix}.jpg"
        img.save(os.path.join(AUG_IMAGE_PATH, new_name))
        new_row = row.copy()
        new_row['image_name'] = new_name
        augmented_data.append(new_row)

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv(AUG_CSV_PATH, index=False)
print(f"Augmentation complete. New dataset size: {len(augmented_df)} images.")


# --- 4. Show File Structure and CSV Preview ---
print("\n--- Generated File Structure ---")
print(f"Data saved in: {AUG_DATA_PATH}")
print("Directory contents:")
for dirname, _, filenames in os.walk(AUG_DATA_PATH):
    for filename in filenames[:5]: # Show first 5 files as an example
        print(os.path.join(dirname, filename))
    if len(filenames) > 5:
        print("...")

print("\n--- Preview of augmented_train.csv ---")
preview_df = pd.read_csv(AUG_CSV_PATH)
print(preview_df.head())
print("------------------------------------")



# --- 1. Configuration and Setup ---
# This script assumes the augmented data has already been created.
# Paths now point to the augmented dataset directory.
AUG_DATA_PATH = '/kaggle/working/original_augmented/'
AUG_IMAGE_PATH = os.path.join(AUG_DATA_PATH, 'jpeg')
AUG_CSV_PATH = os.path.join(AUG_DATA_PATH, 'augmented_train.csv')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model & Training parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FULL = 50
LR_HEAD = 1e-3
LR_FULL = 1e-5
PATIENCE = 5
MODEL_SAVE_PATH = 'best_efficientnet_model.pth'

# --- 2. Load Augmented Data and Split (80/10/10) ---
print("Loading augmented dataset...")
augmented_df = pd.read_csv(AUG_CSV_PATH)

# Preprocessing
augmented_df['sex'] = pd.to_numeric(augmented_df['sex'].map({'male': 0, 'female': 1}))
numerical_features = ['age_approx', 'sex']
categorical_features = ['anatom_site_general_challenge']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# First split: 80% train, 20% temp (for validation and test)
train_df, temp_df = train_test_split(augmented_df, test_size=0.2, random_state=42, stratify=augmented_df['target'])
# Second split: split the 20% temp into 10% validation and 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])

preprocessor.fit(train_df[numerical_features + categorical_features])

print("\n--- Data Split Summary ---")
print(f"Training set size:   {len(train_df)} samples")
print(f"Validation set size: {len(val_df)} samples")
print(f"Test set size:       {len(test_df)} samples")
print("--------------------------")

# --- 3. Dataset and DataLoader Classes ---
class MelanomaDataset(Dataset):
    def __init__(self, df, tabular_preprocessor, image_dir, transform=None):
        self.df = df
        self.tabular_preprocessor = tabular_preprocessor
        self.image_dir = image_dir
        self.transform = transform
        self.tabular_data = self.tabular_preprocessor.transform(self.df[numerical_features + categorical_features])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float)
        label = torch.tensor(row['target'], dtype=torch.float)
        return image, tabular, label

# --- 4. EfficientNet-B0 Multimodal Model ---
class MultimodalEfficientNet(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalEfficientNet, self).__init__()
        self.image_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_image_features = self.image_branch.classifier[1].in_features
        self.image_branch.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_image_features, 128)
        )
        self.tabular_branch = nn.Sequential(nn.Linear(num_tabular_features, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3))
        self.fusion = nn.Linear(128 + 32, 64)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
    def forward(self, image, tabular):
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        combined_features = torch.cat([image_features, tabular_features], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        return output

# --- 5. Training and Evaluation Functions ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='best_model.pth'):
        self.patience, self.verbose, self.path = patience, verbose, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, max_epochs, patience):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=MODEL_SAVE_PATH)
    for epoch in range(max_epochs):
        model.train()
        train_loss, train_corrects = 0.0, 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
        for images, tabular, labels in train_loop:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_corrects += torch.sum(preds == labels.data)
            train_loop.set_postfix(loss=loss.item())
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images, tabular, labels_dev = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images, tabular)
                loss = criterion(outputs, labels_dev)
                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_corrects += torch.sum(preds == labels_dev.data)
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_train_acc = train_corrects.double() / len(train_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
        history['train_loss'].append(epoch_train_loss); history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item()); history['val_acc'].append(epoch_val_acc.item())
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}")
        scheduler.step(epoch_val_loss)
        early_stopper(epoch_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
    print("\nFinished Training.")
    return history

# --- 6. Main Execution ---
# DataLoaders with more advanced augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])
val_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = MelanomaDataset(train_df, preprocessor, AUG_IMAGE_PATH, transform=train_transform)
val_dataset = MelanomaDataset(val_df, preprocessor, AUG_IMAGE_PATH, transform=val_transform)
test_dataset = MelanomaDataset(test_df, preprocessor, AUG_IMAGE_PATH, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Instantiate Model
num_tab_features = preprocessor.transform(train_df.head(1)[numerical_features + categorical_features]).shape[1]
model = MultimodalEfficientNet(num_tab_features).to(DEVICE)

# --- Two-Phase Training ---
# Phase 1: Train the head
for param in model.image_branch.parameters(): param.requires_grad = False
for param in model.image_branch.classifier.parameters(): param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)
print("--- Phase 1: Training Classifier Head ---")
history_head = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, max_epochs=EPOCHS_HEAD, patience=PATIENCE)

# Phase 2: Fine-tune the full model
for param in model.parameters(): param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=LR_FULL)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)
print("\n--- Phase 2: Fine-Tuning Full Model ---")
history_full = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, max_epochs=EPOCHS_FULL, patience=PATIENCE)

# --- 7. Final Evaluation on the HOLD-OUT TEST SET ---
print("\n--- Loading best model for final TEST evaluation ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

model.eval()
test_preds, test_labels, test_probs = [], [], []
with torch.no_grad():
    for images, tabular, labels in tqdm(test_loader, desc="Testing"):
        images, tabular = images.to(DEVICE), tabular.to(DEVICE)
        outputs = model(images, tabular)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        test_preds.extend(preds)
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probs)

# --- Final Reports for the TEST SET ---
test_accuracy = accuracy_score(test_labels, test_preds)
auc_score = roc_auc_score(test_labels, test_probs)
f1 = f1_score(test_labels, test_preds)
print(f"\n--- TEST SET RESULTS ---")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC:      {auc_score:.4f}")
print(f"Test F1-Score:   {f1:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(test_labels, test_preds, target_names=['Benign', 'Malignant']))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix (Test Set)'); plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.show()




