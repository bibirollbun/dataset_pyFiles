import cv2, time, random, timm, json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

data_path = Path('/kaggle/input/cassava-leaf-disease-classification/')
train_df = pd.read_csv(data_path/'train.csv')

# Load label map
with open(data_path/'label_num_to_disease_map.json') as f:
    label_map = json.load(f)

# Convert label numbers to strings before mapping
train_df['label_name'] = train_df['label'].apply(lambda x: label_map[str(x)])
train_df['filepath'] = train_df['image_id'].apply(lambda x: str(data_path/'train_images'/x))
num_classes = train_df['label'].nunique()

train_df.head()


plt.figure(figsize=(8,5))
sns.countplot(
    x='label_name', 
    data=train_df, 
    order=train_df['label_name'].value_counts().index,
    palette='viridis'
)

plt.title('Distribution of Cassava Leaf Disease Classes', fontsize=14)
plt.xlabel('Disease Type', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.show 


def show_samples(df, n=10):
    sample = df.sample(n)
    plt.figure(figsize=(12,8))
    for i,(fp,lab) in enumerate(zip(sample['filepath'], sample['label'])):
        plt.subplot(2, n//2, i+1); plt.imshow(plt.imread(fp)); plt.axis('off')
        plt.title(f"{lab}: {label_map[str(lab)]}")
    plt.tight_layout(); plt.show()

show_samples(train_df, n=8)


# --- KONFIGURASI UTAMA (CFG) ---
class CFG:
    seed = 42
    models_to_compare = ['resnet50', 'tf_efficientnet_b0_ns', 'vit_base_patch16_224']
    img_size = 224      # Ukuran gambar input
    num_classes = 5     # Jumlah kategori penyakit (0-4)
    batch_size = 32     # Sesuaikan dengan VRAM
    epochs = 25          # Jumlah putaran training
    patience = 3
    lr = 1e-4           # Learning Rate
    weight_decay = 1e-6
    num_workers = 0 #4     # Sesuai core CPU laptop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_size = 5     # Sama dengan num_classes
    label_smoothing = 0.1
    
# Kunci Randomness agar hasil bisa diulang (Reproducible)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.seed)
print(f"Device yang digunakan: {CFG.device}")


# --- AUGMENTASI DATA ---
def get_transforms(data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# --- DATASET CLASS ---
class CassavaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df['path'].values 
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        if image is None:
            image = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx]).long()
        return image, label

# --- FUNGSI TRAIN & VALID ---
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = GradScaler()
    dataset_size = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

    if scheduler is not None:
        scheduler.step()
        
    return running_loss / dataset_size

def valid_one_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    valid_labels = []
    
    criterion = nn.CrossEntropyLoss() 
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            preds.append(outputs.softmax(1).detach().cpu().numpy())
            valid_labels.append(labels.detach().cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    pred_classes = preds.argmax(1)
    acc = accuracy_score(valid_labels, pred_classes)
    f1 = f1_score(valid_labels, pred_classes, average='macro')
    
    return running_loss / dataset_size, acc, f1


# --- FUNGSI VISUALISASI ---
def plot_history(all_histories):
    plt.figure(figsize=(18, 5))
    
    # Plot Accuracy
    plt.subplot(1, 3, 1)
    for model_name, history in all_histories.items():
        plt.plot(history['val_acc'], label=f'{model_name}', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot F1 Score (NEW)
    plt.subplot(1, 3, 2)
    for model_name, history in all_histories.items():
        plt.plot(history['val_f1'], label=f'{model_name}', marker='x', linestyle='--')
    plt.title('Validation F1 Score (Macro)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 3, 3)
    for model_name, history in all_histories.items():
        plt.plot(history['val_loss'], label=f'{model_name}')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): Berapa epoch harus menunggu setelah last time score improved.
            verbose (bool): Jika True, cetak pesan setiap kali validasi membaik.
            delta (float): Minimum perubahan agar dianggap sebagai perbaikan.
            path (str): Nama file untuk menyimpan model terbaik.
            trace_func (function): Fungsi untuk print output (default: print).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        # val_metric disini adalah F1 Score
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            print(f'Metric improved ({self.best_score:.4f} --> {val_metric:.4f}). Saving model ...')
        torch.save(model.state_dict(), self.path)

# --- MODEL WRAPPER ---
class UniversalModel(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Logika dinamis untuk mengganti classifier head
        if 'resnet' in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif 'efficientnet' in model_name:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        elif 'vit' in model_name:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
            
    def forward(self, x):
        return self.model(x)

# --- TRAINING ENGINE ---
def train_model(model_name, train_loader, valid_loader):
    print(f"\nâš¡ START TRAINING (Metric: F1 Macro): {model_name} âš¡")
    model = UniversalModel(model_name, num_classes=CFG.num_classes).to(CFG.device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs, T_mult=1, eta_min=1e-6, last_epoch=-1)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # Early Stopping sekarang memantau F1 Score
    early_stopping = EarlyStopping(patience=CFG.patience, verbose=True, path=f"{model_name}_best.pth")
    
    for epoch in range(CFG.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, CFG.device, epoch)
        val_loss, val_acc, val_f1 = valid_one_epoch(model, valid_loader, CFG.device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        early_stopping(val_f1, model)
        
        if early_stopping.early_stop:
            print("ğŸ›‘ Early stopping triggered!")
            break
            
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    # Kembalikan Best F1 Score
    return early_stopping.best_score, history, model

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== 1. Mempersiapkan Data ===")
    base_image_dir = data_path/'train_images/' 
    train_df = pd.read_csv(data_path/'train.csv')
    
    # Buat kolom 'path' (Full absolute path ke gambar)
    train_df['path'] = train_df['image_id'].apply(lambda x: os.path.join(base_image_dir, x))
    
    # Cek apakah path valid (Debugging)
    if not os.path.exists(train_df['path'][0]):
        print(f"â�Œ PERINGATAN: Gambar tidak ditemukan di {train_df['path'][0]}")
        print("Cek apakah folder 'dataset' sudah benar hasil ekstraknya.")
    else:
        print(f"âœ… Path Valid. Contoh: {train_df['path'][0]}")

    base_image_dir = data_path/'train_images/' 
    train_df['path'] = train_df['image_id'].apply(lambda x: os.path.join(base_image_dir, x))

    # 3. Split Data (Stratified K-Fold)
    folds = train_df.copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(skf.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    
    fold_to_train = 0
    trn_idx = folds[folds['fold'] != fold_to_train].index
    val_idx = folds[folds['fold'] == fold_to_train].index
    train_data = folds.loc[trn_idx].reset_index(drop=True)
    valid_data = folds.loc[val_idx].reset_index(drop=True)
    
    train_dataset = CassavaDataset(train_data, transform=get_transforms('train'))
    valid_dataset = CassavaDataset(valid_data, transform=get_transforms('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True)
    
    # --- LOOP KOMPARASI MODEL ---
    all_histories = {}
    best_models = {}
    results = []
    
    print(f"ğŸš€ Memulai Komparasi {len(CFG.models_to_compare)} Model pada Device: {CFG.device}")
    
    for model_name in CFG.models_to_compare:
        acc, hist, trained_model = train_model(model_name, train_loader, valid_loader)
        all_histories[model_name] = hist
        best_models[model_name] = trained_model
        results.append({
            'Model Architecture': model_name,
            'Best Validation Accuracy': acc,
            'Image Size': CFG.img_size,
            'Epochs': CFG.epochs
        })
        torch.cuda.empty_cache() # Bersihkan VRAM


def unnormalize(tensor):
    """Mengembalikan gambar tensor yang ternormalisasi ke warna aslinya"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Pindah ke CPU, ubah dimensi (C, H, W) -> (H, W, C)
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    
    # Denormalisasi
    img = std * img + mean
    return np.clip(img, 0, 1)

# --- 2. Definisi Fungsi Visualisasi ---
def visualize_predictions(model, loader, device, label_map, num_images=10):
    """Menampilkan grid gambar prediksi vs label asli"""
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 8))
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(images.size(0)):
                if images_shown >= num_images: break
                
                ax = plt.subplot(2, 5, images_shown + 1)
                
                # Panggil fungsi unnormalize yang sudah didefinisikan di atas
                img = unnormalize(images[j])
                plt.imshow(img)
                
                true_lb = label_map[str(labels[j].item())]
                pred_lb = label_map[str(preds[j].item())]
                
                # Warna teks: Hijau (Benar), Merah (Salah)
                color = 'green' if labels[j] == preds[j] else 'red'
                
                ax.set_title(f"True: {true_lb}\nPred: {pred_lb}", color=color, fontsize=9, fontweight='bold')
                plt.axis('off')
                images_shown += 1
            
            if images_shown >= num_images: break
            
    plt.tight_layout()
    plt.show()

# --- TAMPILKAN HASIL AKHIR ---
print("\n\n" + "="*40)
print("       HASIL AKHIR KOMPARASI")
print("="*40)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Best Validation Accuracy', ascending=False).reset_index(drop=True)

print(results_df)

# --- OUTPUT VISUALISASI ---
print("\n" + "="*40)
print("ğŸ“Š HASIL AKHIR & VISUALISASI")
print("="*40)

# 1. Tabel Skor
df_res = pd.DataFrame(results).sort_values(by='Best Validation Accuracy', ascending=False)
print(df_res)

# 2. Grafik Chart
plot_history(all_histories)

# 3. Contoh Gambar Prediksi (dari Model Terbaik)
best_model_name = df_res.iloc[0]['Model Architecture']
print(f"ğŸ–¼ï¸� Menampilkan Prediksi Model Terbaik ({best_model_name})...")
visualize_predictions(best_models[best_model_name], valid_loader, CFG.device, label_map)




