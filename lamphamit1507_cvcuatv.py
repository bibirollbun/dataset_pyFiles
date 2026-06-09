# Cai dat cac thu vien can thiet
# numpy<2.0 de tranh loi tuong thich giua cac thu vien cu va moi
!pip install "numpy<2.0" "pillow==10.4.0" torchvision grad-cam timm scikit-image scikit-learn seaborn matplotlib pandas --no-cache-dir
!pip install huggingface_hub joblib --quiet

import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from skimage.feature import hog
import timm

# Thiet lap thiet bi (GPU hoac CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device dang su dung: {device}")

# Cau hinh Seed de dong bo ket qua (reproducibility)
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# Cau hinh duong dan den Dataset tren Kaggle
DATA_DIR = '../input/cassava-leaf-disease-classification'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_images')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

# Doc file CSV va tao duong dan day du cho anh
df = pd.read_csv(TRAIN_CSV)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

# Dinh nghia ten cac lop benh theo tai lieu cuoc thi
class_names = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}
NUM_CLASSES = 5

print(f"Tong so anh: {len(df)}")
print("Phan bo du lieu ban dau:")
print(df['label'].value_counts())

# Tinh toan Class Weights de xu ly mat can bang du lieu (Rat quan trong voi bo San)
# Cong thuc: N_total / (N_classes * N_samples_per_class)
class_counts = df['label'].value_counts().sort_index().values
class_weights = len(df) / (NUM_CLASSES * class_counts)
# Chuyen weights sang tensor va dua vao GPU
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class Weights da tinh toan: {class_weights}")

# Chia Tap du lieu: 70% Train - 15% Val - 15% Test
# Su dung stratify de giu nguyen ti le cac benh trong tung tap
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Kich thuoc tap: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")


class CassavaDataset(Dataset):
    def __init__(self, df, root_dir="./train_images", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame chá»©a thÃ´ng tin áº£nh (image_id, label)
            root_dir (string): Ä�Æ°á»�ng dáº«n thÆ° má»¥c chá»©a áº£nh.
            transform (callable, optional): HÃ m xá»­ lÃ½ áº£nh (augmentation).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Láº¥y tÃªn áº£nh vÃ  nhÃ£n tá»« dataframe
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        # Táº¡o Ä‘Æ°á»�ng dáº«n Ä‘áº§y Ä‘á»§ tá»›i áº£nh
        # LÆ°u Ã½: Náº¿u trÃªn Kaggle thÃ¬ Ä‘Æ°á»�ng dáº«n thÆ°á»�ng lÃ : "../input/cassava-leaf-disease-classification/train_images"
        # Báº¡n hÃ£y kiá»ƒm tra láº¡i biáº¿n TRAIN_DIR hoáº·c Ä‘Æ°á»�ng dáº«n root_dir nhÃ©
        img_path = os.path.join(self.root_dir, image_id)
        
        # Ä�á»�c áº£nh báº±ng PIL (Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch vá»›i transforms cá»§a PyTorch)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback náº¿u Ä‘Æ°á»�ng dáº«n sai, thá»­ Ä‘Æ°á»�ng dáº«n máº·c Ä‘á»‹nh Kaggle
            img_path = os.path.join("../input/cassava-leaf-disease-classification/train_images", image_id)
            image = Image.open(img_path).convert("RGB")

        # Ã�p dá»¥ng Augmentation
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
# --- Sá»¬A Láº I DATA LOADER (Cell 3) ---
IMG_SIZE = 224

# Cáº¬P NHáº¬T: Sá»­ dá»¥ng AutoAugment cho chiáº¿n lÆ°á»£c tÄƒng cÆ°á»�ng dá»¯ liá»‡u máº¡nh máº½
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), # <--- THAY Ä�á»”I Lá»šN
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25) # GiÃºp mÃ´ hÃ¬nh há»�c tá»‘t hÆ¡n khi bá»‹ che khuáº¥t
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset & DataLoader (Giá»¯ nguyÃªn pháº§n gá»�i Dataset cá»§a báº¡n)
train_ds = CassavaDataset(train_df, transform=train_transforms)
val_ds = CassavaDataset(val_df, transform=val_test_transforms)
test_ds = CassavaDataset(test_df, transform=val_test_transforms)

# LÆ°u Ã½: num_workers=2 Ä‘á»ƒ tá»‘i Æ°u tá»‘c Ä‘á»™ load trÃªn Kaggle/Colab
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)


import time
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.cuda.amp import autocast, GradScaler

# Giá»¯ nguyÃªn cáº¥u hÃ¬nh Mixup
mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
    mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES)

def train_model(model, optimizer, criterion, scheduler=None, mixup_fn=None, epochs=10, patience=3, model_name="model", log_interval=30):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # --- KHá»�I Táº O SCALER CHO AMP (TÄ‚NG Tá»�C) ---
    scaler = GradScaler() 

    print(f"--- Bat dau train {model_name} (Mixup: {'BAT' if mixup_fn else 'TAT'}) [AMP Enabled] ---\n")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 1. Ã�p dá»¥ng Mixup
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)
            
            optimizer.zero_grad()
            
            # --- CHáº Y Báº°NG FLOAT16 (AMP) ---
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scale loss Ä‘á»ƒ trÃ¡nh bá»‹ underflow khi dÃ¹ng float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # -------------------------------

            running_loss += loss.item() * images.size(0)
            
            # =================================================================
            # 2. Sá»¬A Lá»–I TÃ�NH ACCURACY KHI CÃ“ MIXUP
            # =================================================================
            _, predicted = torch.max(outputs, 1)
            
            # Náº¿u labels lÃ  dáº¡ng Mixup (Shape: [Batch_size, Num_classes]) -> Soft labels
            if labels.ndim > 1:
                # Láº¥y index cá»§a class cÃ³ trá»�ng sá»‘ lá»›n nháº¥t trong soft label Ä‘á»ƒ so sÃ¡nh
                _, targets = torch.max(labels, 1)
            else:
                # Náº¿u labels lÃ  dáº¡ng thÆ°á»�ng (Shape: [Batch_size]) -> Hard labels (Integer)
                targets = labels

            total += labels.size(0)
            correct += (predicted == targets).sum().item()
            current_batch_acc = 100 * (predicted == targets).sum().item() / labels.size(0)
            # =================================================================

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                print(f"   [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} | Acc: {current_batch_acc:.1f}%")

        epoch_loss = running_loss / len(train_loader.dataset)
        # TÃ­nh láº¡i epoch_acc tá»•ng (khÃ´ng bá»‹ 0 ná»¯a)
        epoch_acc = 100 * correct / total 

        # --- VALIDATION (Validation khÃ´ng bao giá»� dÃ¹ng Mixup nÃªn giá»¯ nguyÃªn) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = val_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"==> END EPOCH {epoch+1}: Train Loss={epoch_loss:.4f} Acc={epoch_acc:.2f}% | Val Loss={epoch_val_loss:.4f} Acc={epoch_val_acc:.2f}%")
        print("-" * 50)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict()
            patience_counter = 0
            print("   (Save Model - Best Val Loss)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"STOP! Early stopping.")
                break
    
    time_elapsed = time.time() - start_time
    if best_weights:
        model.load_state_dict(best_weights)
    return model, history, time_elapsed


print("=== TRAINING SVM (BASELINE) ===")
# Luu y: SVM voi 21k anh va HOG se ton rat nhieu RAM.
# Chung ta chi lay mot tap con (subset) 3000 anh de train SVM lam muc so sanh

def extract_hog_features(dataset, limit=5000):
    features = []
    labels = []
    print(f"Dang trich xuat HOG cho {limit} anh...")

    # Chon ngau nhien index
    indices = np.random.choice(len(dataset), min(len(dataset), limit), replace=False)

    for i in indices:
        img, label = dataset[i]
        # Chuyen Tensor ve Numpy va Denormalize so bo de lay anh Gray
        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min()) # Normalize ve 0-1
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Tinh HOG descriptors
        fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        features.append(fd)
        labels.append(label)

    return np.array(features), np.array(labels)

# Trich xuat dac trung
X_train_hog, y_train_hog = extract_hog_features(train_ds, limit=5000)
X_test_hog, y_test_hog = extract_hog_features(test_ds, limit=2000)

# Train SVM
start_svm = time.time()
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_hog, y_train_hog)
svm_time = time.time() - start_svm
print(f"SVM Training Time: {svm_time:.2f}s")

# Danh gia SVM
y_pred_svm = svm_model.predict(X_test_hog)
print("Classification Report SVM:")
print(classification_report(y_test_hog, y_pred_svm, target_names=[class_names[i] for i in range(NUM_CLASSES)]))


print("=== TRAINING RESNET50 (OPTIMIZED) ===")
# 1. Setup Model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Dropout(p=0.5), # ThÃªm Dropout chá»‘ng Overfitting
    nn.Linear(num_ftrs, NUM_CLASSES)
)

# 2. Setup Optimizer & Scheduler
criterion_res = nn.CrossEntropyLoss(weight=class_weights)
optimizer_res = optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler_res = optim.lr_scheduler.ReduceLROnPlateau(optimizer_res, mode='min', factor=0.1, patience=2)

# 3. Train (Epochs Ä‘áº·t cao, Ä‘á»ƒ Early Stopping tá»± ngáº¯t)
resnet, resnet_hist, resnet_time = train_model(
    resnet, optimizer_res, criterion_res, scheduler=scheduler_res,
    mixup_fn=None, 
    epochs=50, patience=6, model_name="ResNet50_Opt",
    log_interval=100 
)


print("=== TRAINING ViT (OPTIMIZED + MIXUP) ===")
# 1. Setup Model (ThÃªm drop_path_rate)
model_vit = timm.create_model(
    'vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES,
    drop_rate=0.1, drop_path_rate=0.1 # Ká»¹ thuáº­t chá»‘ng overfit riÃªng cho ViT
)

# 2. Setup Optimizer & Loss cho Mixup
# Quan trá»�ng: DÃ¹ng SoftTargetCrossEntropy vÃ¬ nhÃ£n Ä‘Ã£ bá»‹ Mixup trá»™n
criterion_vit = SoftTargetCrossEntropy() 
optimizer_vit = optim.AdamW(model_vit.parameters(), lr=5e-5, weight_decay=0.05)
scheduler_vit = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vit, mode='min', factor=0.1, patience=3)

# 3. Train (Báº­t Mixup)
model_vit, vit_hist, vit_time = train_model(
    model_vit, optimizer_vit, criterion_vit, scheduler=scheduler_vit,
    mixup_fn=mixup_fn, 
    epochs=50, patience=7, model_name="ViT_Base_Mixup",
    log_interval=100
)


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# 1. Ä�á»ŠNH NGHÄ¨A HÃ€M get_predictions (Bá»‹ thiáº¿u trÆ°á»›c Ä‘Ã³)
def get_predictions(model, loader):
    model.eval()
    all_targets = []
    all_preds = []
    
    # Ä�áº£m báº£o model Ä‘ang á»Ÿ Ä‘Ãºng device
    model = model.to(device)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds)

# ---------------------------------------------------------
print("=== SO SANH KET QUA 3 MO HINH ===")

# 2. Láº¥y dá»± Ä‘oÃ¡n trÃªn táº­p Test Ä‘áº§y Ä‘á»§
print("Dang lay du doan tu ResNet...")
y_true_resnet, y_pred_resnet = get_predictions(resnet, test_loader)

print("Dang lay du doan tu ViT...")
y_true_vit, y_pred_vit = get_predictions(model_vit, test_loader)

# 3. TÃ­nh metrics
def compute_metrics(y_true, y_pred, time_taken, name):
    acc = accuracy_score(y_true, y_pred)
    return {"Model": name, "Accuracy": acc*100, "Time (s)": time_taken}

# LÆ¯U Ã�: Ä�áº£m báº£o biáº¿n svm_time, y_test_hog, y_pred_svm Ä‘Ã£ cÃ³ tá»« cÃ¡c cell trÆ°á»›c
# Náº¿u chÆ°a cháº¡y SVM, hÃ£y comment dÃ²ng SVM láº¡i Ä‘á»ƒ trÃ¡nh lá»—i tiáº¿p theo
metrics = []

# Kiá»ƒm tra xem SVM Ä‘Ã£ cháº¡y chÆ°a
try:
    metrics.append({"Model": "SVM (HOG)", "Accuracy": accuracy_score(y_test_hog, y_pred_svm)*100, "Time (s)": svm_time})
except NameError:
    print("Warning: Khong tim thay ket qua SVM, bo qua SVM trong bang so sanh.")

metrics.append(compute_metrics(y_true_resnet, y_pred_resnet, resnet_time, "ResNet50"))
metrics.append(compute_metrics(y_true_vit, y_pred_vit, vit_time, "ViT"))

df_metrics = pd.DataFrame(metrics)
print("\nBAO CAO TONG HOP:")
print(df_metrics)

# 4. Váº½ biá»ƒu Ä‘á»“ so sÃ¡nh
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="Accuracy", data=df_metrics, palette="viridis")
plt.title("So sanh Accuracy giua cac mo hinh")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.show()

# 5. Váº½ Confusion Matrix
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    # Láº¥y class names tá»« dataset gá»‘c (náº¿u cÃ³) hoáº·c dÃ¹ng sá»‘
    labels = [str(i) for i in range(NUM_CLASSES)]
    try:
        # Thá»­ láº¥y label map náº¿u cÃ³
        labels = [class_names[i] for i in range(NUM_CLASSES)]
    except:
        pass
        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title(title)
    plt.show()

print("Ve Confusion Matrix...")
plot_cm(y_true_resnet, y_pred_resnet, "Confusion Matrix - ResNet50")
plot_cm(y_true_vit, y_pred_vit, "Confusion Matrix - ViT")


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# --- 1. Váº¼ BIá»‚U Ä�á»’ LOSS & ACCURACY ---
def plot_training_history(history, model_name):
    if not history:
        print(f"ChÆ°a cÃ³ dá»¯ liá»‡u lá»‹ch sá»­ huáº¥n luyá»‡n cho {model_name}")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Biá»ƒu Ä‘á»“ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Biá»ƒu Ä‘á»“ Accuracy (Náº¿u cÃ³)
    # LÆ°u Ã½: ViT dÃ¹ng Mixup nÃªn train_acc cÃ³ thá»ƒ khÃ´ng chÃ­nh xÃ¡c hoáº·c báº±ng 0
    if any(history['val_acc']):
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b--o', label='Train Acc', alpha=0.6)
        plt.plot(epochs, history['val_acc'], 'g-o', label='Val Acc')
        plt.title(f'{model_name} - Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Váº½ biá»ƒu Ä‘á»“ cho ResNet50
if 'resnet_hist' in globals():
    plot_training_history(resnet_hist, "ResNet50")

# Váº½ biá»ƒu Ä‘á»“ cho ViT
if 'vit_hist' in globals():
    plot_training_history(vit_hist, "ViT (Vision Transformer)")

# --- 2. CLASSIFICATION REPORT CHI TIáº¾T ---
# Láº¥y tÃªn cÃ¡c lá»›p bá»‡nh
target_names = [class_names[i] for i in range(NUM_CLASSES)]

print("\n" + "="*40)
print("ğŸ“Š CHI TIáº¾T HIá»†U SUáº¤T Tá»ªNG Lá»šP (CLASSIFICATION REPORT)")
print("="*40)

# BÃ¡o cÃ¡o cho SVM
if 'y_test_hog' in globals() and 'y_pred_svm' in globals():
    print(f"\nModel: SVM (HOG)")
    print("-" * 20)
    print(classification_report(y_test_hog, y_pred_svm, target_names=target_names))

# BÃ¡o cÃ¡o cho ResNet50
if 'y_true_resnet' in globals() and 'y_pred_resnet' in globals():
    print(f"\nModel: ResNet50")
    print("-" * 20)
    print(classification_report(y_true_resnet, y_pred_resnet, target_names=target_names))

# BÃ¡o cÃ¡o cho ViT
if 'y_true_vit' in globals() and 'y_pred_vit' in globals():
    print(f"\nModel: Vision Transformer (ViT)")
    print("-" * 20)
    print(classification_report(y_true_vit, y_pred_vit, target_names=target_names))


# ==============================================================================
# CELL CODE: PHÃ‚N TÃ�CH Lá»–I CHI TIáº¾T & Ä�O Tá»�C Ä�á»˜ (INFERENCE TIME)
# (ThÃªm cell nÃ y vÃ o cuá»‘i Notebook)
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

# 1. HÃ€M HIá»‚N THá»Š CÃ�C áº¢NH Ä�OÃ�N SAI (VISUAL ERROR ANALYSIS)
def visualize_errors(model, loader, device, class_names, model_name="Model", num_images=10):
    """
    TÃ¬m vÃ  hiá»ƒn thá»‹ cÃ¡c áº£nh mÃ  model dá»± Ä‘oÃ¡n sai so vá»›i nhÃ£n thá»±c táº¿.
    """
    model.to(device)
    model.eval()
    images_so_far = 0
    
    # Thiáº¿t láº­p kÃ­ch thÆ°á»›c khung hÃ¬nh (tá»± Ä‘á»™ng tÃ­nh sá»‘ hÃ ng/cá»™t)
    cols = 5
    rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)
    plt.figure(figsize=(15, 3.5 * rows))
    
    print(f"\nğŸ”� [{model_name}] Ä�ang quÃ©t test set Ä‘á»ƒ tÃ¬m cÃ¡c máº«u Ä‘oÃ¡n sai...")
    
    # Chuáº©n ImageNet Ä‘á»ƒ Un-normalize (tráº£ láº¡i mÃ u gá»‘c cho áº£nh)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # TÃ¬m cÃ¡c index sai
            misclassified_idxs = (preds != labels).nonzero(as_tuple=False)

            for idx in misclassified_idxs:
                idx = idx.item() # Láº¥y index cá»¥ thá»ƒ
                if images_so_far >= num_images:
                    break

                images_so_far += 1
                
                ax = plt.subplot(rows, cols, images_so_far)
                ax.axis('off')
                
                # Chuyá»ƒn Tensor sang Numpy & Un-normalize
                img = inputs[idx].cpu().numpy().transpose((1, 2, 0))
                img = std * img + mean
                img = np.clip(img, 0, 1)

                ax.imshow(img)
                
                # Láº¥y tÃªn nhÃ£n (xá»­ lÃ½ cáº£ dict vÃ  list)
                if isinstance(class_names, dict):
                    true_label = class_names[labels[idx].item()]
                    pred_label = class_names[preds[idx].item()]
                else: # Náº¿u lÃ  list
                    true_label = class_names[labels[idx]]
                    pred_label = class_names[preds[idx]]
                
                # TiÃªu Ä‘á»� mÃ u Ä‘á»�: NhÃ£n Ä�Ãºng vs NhÃ£n Ä�oÃ¡n
                ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=10, fontweight='bold')

            if images_so_far >= num_images:
                plt.suptitle(f"Error Analysis: {model_name} (Top {num_images} Errors)", fontsize=16, y=1.02)
                plt.tight_layout()
                plt.show()
                return

    print(f"   -> Tuyá»‡t vá»�i! Model Ä‘oÃ¡n Ä‘Ãºng gáº§n háº¿t hoáº·c khÃ´ng tÃ¬m Ä‘á»§ {num_images} lá»—i.")


# 2. HÃ€M Ä�O Tá»�C Ä�á»˜ Xá»¬ LÃ� (INFERENCE TIME / FPS)
def measure_speed(model, device, model_name="Model", input_shape=(1, 3, 224, 224), n_runs=100):
    """
    Ä�o tá»‘c Ä‘á»™ xá»­ lÃ½ trung bÃ¬nh vÃ  sá»‘ khung hÃ¬nh trÃªn giÃ¢y (FPS).
    """
    model.to(device)
    model.eval()
    
    # Táº¡o dá»¯ liá»‡u giáº£ láº­p (Dummy input)
    dummy_input = torch.randn(input_shape).to(device)

    # A. Warmup (LÃ m nÃ³ng GPU - quan trá»�ng Ä‘á»ƒ sá»‘ liá»‡u chÃ­nh xÃ¡c)
    print(f"\nâ�±ï¸� [{model_name}] Ä�ang Warmup GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # B. Ä�o thá»�i gian thá»±c táº¿
    print(f"   -> Ä�ang test tá»‘c Ä‘á»™ trÃªn {n_runs} máº«u...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / n_runs) * 1000 # Mili-giÃ¢y
    fps = n_runs / total_time # Khung hÃ¬nh/giÃ¢y
    
    print("-" * 50)
    print(f"ğŸš€ Káº¾T QUáº¢ HIá»†U NÄ‚NG: {model_name}")
    print(f"   - Tá»•ng thá»�i gian ({n_runs} áº£nh): {total_time:.4f}s")
    print(f"   - Ä�á»™ trá»… (Latency): {avg_time_ms:.2f} ms/áº£nh")
    print(f"   - Tá»‘c Ä‘á»™ (FPS): {fps:.2f} frames/sec")
    print("-" * 50)


# ==============================================================================
# 3. CHáº Y THá»°C Táº¾ (EXECUTION)
# ==============================================================================
# Cáº¥u hÃ¬nh thiáº¿t bá»‹
if 'device' not in globals(): device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Giáº£ Ä‘á»‹nh class_names náº¿u chÆ°a cÃ³ (PhÃ²ng há»� biáº¿n bá»‹ máº¥t)
if 'class_names' not in globals(): 
    # Thay Ä‘á»•i danh sÃ¡ch nÃ y khá»›p vá»›i dataset Cassava cá»§a báº¡n
    class_names = {0: 'CBB', 1: 'CBSD', 2: 'CGM', 3: 'CMD', 4: 'Healthy'}

print("ğŸ�¯ Báº®T Ä�áº¦U PHÃ‚N TÃ�CH VÃ€ KIá»‚M TRA HIá»†U NÄ‚NG...\n")

# --- A. Xá»¬ LÃ� RESNET ---
if 'resnet' in globals() and 'test_loader' in globals():
    visualize_errors(resnet, test_loader, device, class_names, model_name="ResNet50", num_images=10)
    measure_speed(resnet, device, model_name="ResNet50")
else:
    print("âš ï¸� Bá»� qua ResNet (KhÃ´ng tÃ¬m tháº¥y biáº¿n 'resnet' hoáº·c 'test_loader')")

# --- B. Xá»¬ LÃ� VIT (VISION TRANSFORMER) ---
if 'model_vit' in globals() and 'test_loader' in globals():
    visualize_errors(model_vit, test_loader, device, class_names, model_name="ViT Base", num_images=10)
    measure_speed(model_vit, device, model_name="ViT Base")
else:
    print("âš ï¸� Bá»� qua ViT (KhÃ´ng tÃ¬m tháº¥y biáº¿n 'model_vit' hoáº·c 'test_loader')")



from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

print("=== VISUALIZATION HEATMAP (GRAD-CAM) ===")

def visualize_gradcam(model, dataset, num_images=4):
    model.eval()
    # Chon layer cuoi cung cua ResNet (layer4)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Chon ngau nhien anh tu tap test
    indices = np.random.choice(len(dataset), num_images, replace=False)

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Tao Heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=None) # None = lay class du doan cao nhat
        grayscale_cam = grayscale_cam[0, :]

        # Denormalize anh de hien thi
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Hien thi anh
        ax_orig = axes[i, 0] if num_images > 1 else axes[0]
        ax_cam = axes[i, 1] if num_images > 1 else axes[1]

        # Lay du doan
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)

        true_name = class_names[label.item()]
        pred_name = class_names[pred_idx.item()]
        color = 'green' if label == pred_idx else 'red'

        ax_orig.imshow(rgb_img)
        ax_orig.set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
        ax_orig.axis('off')

        ax_cam.imshow(visualization)
        ax_cam.set_title(f"Grad-CAM (Conf: {conf.item()*100:.1f}%)")
        ax_cam.axis('off')

    plt.tight_layout()
    plt.show()

# Ve Heatmap cho ResNet50
visualize_gradcam(resnet, test_ds, num_images=5)


# Ham nay dung de demo: Nhan duong dan file anh -> Tra ve ket qua va Heatmap
def predict_single_image(image_path, model):
    model.eval()
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
    # Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    
    # Visualize
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Pred: {class_names[pred_idx.item()]}\nConf: {conf.item()*100:.2f}%")
    plt.axis('off')
    plt.show()

# Lay thu 1 anh tu tap Test de demo
sample_row = test_df.iloc[0]
print(f"Demo voi file: {sample_row['image_path']}")
predict_single_image(sample_row['image_path'], resnet)


# --- 1. CÃ€I Ä�áº¶T & IMPORT ---
import os
import torch
import joblib
!pip install huggingface_hub --quiet
from huggingface_hub import login, HfApi, create_repo


HF_TOKEN = "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" 
HF_USERNAME = "lamphamit"             
REPO_NAME = "cassava-leaf-disease-models"

FULL_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

print("=== BÆ¯á»šC 1: LÆ¯U MODEL XUá»�NG FILE ===")

# LÆ°u ResNet
try:
    if 'resnet' in globals():
        torch.save(resnet.state_dict(), "resnet50_best.pth")
        print("âœ… Ä�Ã£ lÆ°u: resnet50_best.pth")
except: pass

# LÆ°u ViT
try:
    if 'model_vit' in globals():
        torch.save(model_vit.state_dict(), "vit_base_best.pth")
        print("âœ… Ä�Ã£ lÆ°u: vit_base_best.pth")
except: pass

# LÆ°u SVM
try:
    if 'svm_model' in globals():
        joblib.dump(svm_model, "svm_hog_model.joblib")
        print("âœ… Ä�Ã£ lÆ°u: svm_hog_model.joblib")
    elif 'clf' in globals():
        joblib.dump(clf, "svm_hog_model.joblib")
        print("âœ… Ä�Ã£ lÆ°u: svm_hog_model.joblib")
except: pass

# ==============================================================================
# 4. UPLOAD LÃŠN HUGGING FACE
# ==============================================================================
print("\n=== BÆ¯á»šC 2: Káº¾T Ná»�I & UPLOAD ===")
try:
    login(token=HF_TOKEN)
    api = HfApi()
    create_repo(FULL_REPO_ID, repo_type="model", exist_ok=True)
    print(f"âœ… Ä�Ã£ káº¿t ná»‘i Repo: {FULL_REPO_ID}")
except Exception as e:
    print(f"â�Œ Lá»—i káº¿t ná»‘i (Check láº¡i Token): {e}")

# HÃ m upload an toÃ n
def upload_file_safe(local_name, remote_name):
    if os.path.exists(local_name):
        print(f"â�³ Ä�ang upload {local_name}...")
        try:
            api.upload_file(
                path_or_fileobj=local_name,
                path_in_repo=remote_name,
                repo_id=FULL_REPO_ID,
                repo_type="model"
            )
            print("   -> Xong!")
        except Exception as e:
            print(f"   -> Lá»—i: {e}")

# Upload 3 model
upload_file_safe("resnet50_best.pth", "resnet50_best.pth")
upload_file_safe("vit_base_best.pth", "vit_base_best.pth")
upload_file_safe("svm_hog_model.joblib", "svm_hog_model.joblib")

# ==============================================================================
# 5. Táº O README (CÃ�CH Má»šI: KHÃ”NG DÃ™NG NGOáº¶C KÃ‰P DÃ€I Ä�á»‚ TRÃ�NH Lá»–I)
# ==============================================================================
print("\n=== BÆ¯á»šC 3: Táº O README ===")

with open("README.md", "w") as f:
    f.write("---\n")
    f.write("tags:\n- image-classification\n- cassava\n")
    f.write("---\n")
    f.write(f"# Models for Cassava Leaf Disease\n\n")
    f.write("This repo contains models trained on Kaggle dataset:\n")
    f.write("1. **ResNet50** (.pth)\n")
    f.write("2. **ViT Base** (.pth)\n")
    f.write("3. **SVM** (.joblib)\n")

upload_file_safe("README.md", "README.md")

print(f"\nğŸ�‰ HOÃ€N Táº¤T! Link: https://huggingface.co/{FULL_REPO_ID}")




