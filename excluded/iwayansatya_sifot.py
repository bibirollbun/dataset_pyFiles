import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import os


# =============================================================================
# 1. KONFIGURASI AWAL
# =============================================================================
# --- Sesuaikan path ini dengan struktur folder Anda ---
TRAIN_DIR = '/kaggle/input/srifoton-25-machine-learning-competition/train/train'
VALIDATION_DIR = '/kaggle/input/srifoton-25-machine-learning-competition/val/val'
TEST_DIR = '/kaggle/input/srifoton-25-machine-learning-competition/test/test'

# --- Konfigurasi Model dan Training ---
# ResNet umumnya menggunakan ukuran 224x224
IMG_SIZE = 840
BATCH_SIZE = 64      # Naikkan batch size untuk GPU Kaggle
LEARNING_RATE_HEAD = 1e-3  # Learning rate untuk melatih kepala baru
LEARNING_RATE_FINETUNE = 1e-5 # Learning rate SANGAT KECIL untuk fine-tuning
EPOCHS_HEAD = 5      # Jumlah epoch untuk melatih kepala saja
EPOCHS_FINETUNE = 25 # Jumlah epoch untuk fine-tuning seluruh model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {DEVICE}")


# =============================================================================
# DEFINISI KELAS DATASET (DIPERLUKAN UNTUK TEST SET)
# =============================================================================

class XRayDataset(Dataset):
    """Dataset kustom untuk membaca gambar dan label dari file CSV."""
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        if not self.is_test:
            # Fungsi ini hanya akan relevan jika Anda menggunakannya untuk train/val
            self.labels = self.df['label'].unique()
            self.label_to_int = {label: i for i, label in enumerate(self.labels)}
            self.int_to_label = {i: label for label, i in self.label_to_int.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Ambil nama file dari kolom pertama di CSV
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            # Untuk data test, kembalikan gambar dan nama filenya
            return image, self.df.iloc[idx, 0]
        else:
            # Untuk data train/val, kembalikan gambar dan labelnya
            label_str = self.df.iloc[idx, 1]
            label = self.label_to_int[label_str]
            return image, torch.tensor(label, dtype=torch.long) # Pastikan dtype adalah long


# =============================================================================
# 2. AUGMENTASI DATA & DATALOADER
# =============================================================================

# --- Transformasi Gambar dengan Augmentasi yang Lebih Kuat ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Membuat Objek Dataset dan DataLoader ---
train_dataset = ImageFolder(root=TRAIN_DIR, transform=data_transforms['train'])
val_dataset = ImageFolder(root=VALIDATION_DIR, transform=data_transforms['val'])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
print(f"Kelas yang ditemukan: {class_names}")


# =============================================================================
# 3. MEMBANGUN PRETRAINED MODEL (RESNET34 DENGAN KEPALA LEBIH KUAT)
# =============================================================================
model = models.resnet34(weights='IMAGENET1K_V1')

# Ganti 'kepala' klasifikasi terakhir dengan versi yang lebih baik
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    # Batch Normalization menstabilkan input ke lapisan berikutnya
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)

print("Arsitektur model diperbarui dengan BatchNorm1d.")


# =============================================================================
# 4. STRATEGI TRAINING DUA TAHAP
# =============================================================================

# --- TAHAP 1: Latih Kepala Saja ---
print("\n--- TAHAP 1: Melatih Kepala Klasifikasi Saja ---")

# Bekukan semua lapisan kecuali kepala (fc)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# =============================================================================
# 4. PERSIAPAN OPTIMIZER & SCHEDULER YANG LEBIH BAIK
# =============================================================================

# Gunakan optimizer AdamW yang sering kali sedikit lebih baik dari Adam biasa
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=1e-2)

# Definisikan Cosine Annealing Scheduler
# T_max adalah jumlah total epoch, di mana LR akan mencapai nilai minimumnya.
# Kita set T_max sama dengan total epoch training Anda.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD + EPOCHS_FINETUNE, eta_min=1e-6)

criterion = nn.CrossEntropyLoss()

# --- PENTING: Panggil scheduler di dalam training loop ---
# Di dalam training loop Anda, setelah `optimizer.step()`, tambahkan:
# scheduler.step()

# --- TAHAP 1: Latih Kepala Saja (PERBAIKAN) ---
# ... (kode optimizer dan scheduler tidak berubah) ...

for epoch in range(EPOCHS_HEAD): # Atau EPOCHS_FINETUNE
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_HEAD}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # =========================================================
        # KEMBALIKAN BARIS-BARIS YANG HILANG INI
        # =========================================================
        # Lakukan forward pass untuk mendapatkan output dari model
        outputs = model(images)
        # Hitung loss dengan membandingkan output dan label asli
        loss = criterion(outputs, labels)
        # Lakukan backward pass untuk menghitung gradien
        loss.backward()
        # =========================================================

        # Baris Anda selanjutnya sekarang akan berjalan dengan benar
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    # Pindahkan scheduler.step() ke sini, di luar loop batch
    scheduler.step() 
    
    epoch_loss = running_loss / len(train_dataset)
    # Anda bisa tambahkan print LR untuk memonitor
    print(f"Training Loss Tahap 1: {epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

# --- TAHAP 2: Fine-Tuning Seluruh Model ---
print("\n--- TAHAP 2: Fine-Tuning Seluruh Model ---")

# "Cairkan" semua lapisan
for param in model.parameters():
    param.requires_grad = True

# Optimizer untuk SEMUA parameter model dengan learning rate SANGAT KECIL
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=1e-2)
# scheduler untuk menurunkan learning rate jika performa stagnan
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

for epoch in range(EPOCHS_FINETUNE):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_FINETUNE}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    
    # Validasi di setiap akhir epoch fine-tuning untuk scheduler
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_dataset)
    scheduler.step(val_loss)
    
    print(f"Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

print("Training Selesai!")


# =============================================================================
# 5. MENGEVALUASI MODEL
# =============================================================================

print("\nMemulai evaluasi model pada data validasi...")
model.eval() # Set model ke mode evaluasi
all_preds = []
all_labels = []

# Tidak perlu menghitung gradien saat evaluasi
with torch.no_grad(): 
    for images, labels in tqdm(val_loader, desc="Validasi"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        
        # --- PERBAIKAN DI SINI ---
        # Ambil kelas dengan skor tertinggi sebagai prediksi
        # `outputs` memiliki shape [batch_size, 5]
        # `preds` akan memiliki shape [batch_size]
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Ubah ke format numpy array
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Menampilkan Laporan Klasifikasi ---
# Sekarang panjang all_labels dan all_preds akan sama
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- Menampilkan Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Label Sebenarnya')
plt.xlabel('Label Prediksi')
plt.show()


# =============================================================================
# 6. MELAKUKAN PREDIKSI PADA DATA UJI (VERSI PERBAIKAN)
# =============================================================================

print("\nMemulai prediksi pada data test...")
# Bagian ini untuk membuat daftar file tes dan memuatnya, ini sudah benar.
test_files = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'filename': test_files})
test_csv_path = 'test_files.csv'
test_df.to_csv(test_csv_path, index=False)

# Menggunakan class XRayDataset untuk memuat data test yang tidak berlabel
# Pastikan definisi 'class XRayDataset' sudah dijalankan sebelumnya.
test_dataset = XRayDataset(csv_file=test_csv_path, img_dir=TEST_DIR, transform=data_transforms['val'], is_test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Gunakan BATCH_SIZE agar lebih cepat

model.eval()
test_predictions = []
test_filenames = []

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Prediksi Test Set"):
        images = images.to(DEVICE)
        outputs = model(images)
        
        # --- PERBAIKAN 1: Gunakan argmax untuk prediksi multi-kelas ---
        # Cari indeks dari skor tertinggi dari 5 output model
        preds = torch.argmax(outputs, dim=1)
        
        # Kumpulkan hasil prediksi dan nama filenya
        test_predictions.extend(preds.cpu().numpy())
        test_filenames.extend(filenames)

# --- PERBAIKAN 2: Buat mapping dari indeks ke nama kelas dari ImageFolder ---
# `train_dataset` (dari ImageFolder) punya atribut `class_to_idx`
# Contoh: {'NORMAL': 0, 'COVID': 1, ...}
# Kita perlu membaliknya menjadi {0: 'NORMAL', 1: 'COVID', ...}
int_to_label = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}

# Konversi prediksi (integer) kembali ke nama kelas (string) menggunakan mapping yang benar
predicted_labels_str = [int_to_label[p] for p in test_predictions]

# Gabungkan nama file dan label prediksi ke dalam DataFrame
hasil_prediksi_df = pd.DataFrame({
    'ID': test_filenames,
    'Predicted': predicted_labels_str
})

print("\n\n--- Hasil Prediksi pada Data Test ---")
print(hasil_prediksi_df.head(10))

# Menyimpan hasil prediksi ke file CSV
output_csv_path = 'hasil_prediksi_pytorch.csv'
hasil_prediksi_df.to_csv(output_csv_path, index=False)
print(f"\nHasil prediksi telah disimpan di: {output_csv_path}")

# Hapus file CSV dummy yang tadi kita buat
os.remove(test_csv_path)


# import pandas as pd

# # Baca file CSV
# df = pd.read_csv("/kaggle/working/hasil_prediksi_pytorch.csv")

# # Mapping label
# mapping = {
#     "Bacterial Pneumonia": 0,
#     "Corona Virus Disease": 1,
#     "Normal": 2,
#     "Tuberculosis": 3,
#     "Viral Pneumonia": 4
# }

# # Terapkan mapping
# df["Label Prediksi"] = df["Label Prediksi"].map(mapping)

# df = df.sort_values(by="Label Prediksi").reset_index(drop=True)

# # Simpan ke file baru
# df.to_csv("sub7_wagurisan.csv", index=False)
# print("File berhasil disimpan: hasil_prediksi_encoded.csv")


