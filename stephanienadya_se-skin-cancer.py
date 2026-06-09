import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


df_full = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv')


print(df_full['target'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df_full)
plt.title('Distribusi Kasus Jinak (0) vs. Ganas (1)')
plt.xticks([0, 1], ['Jinak (0)', 'Ganas (1)'])
plt.show()


try:
    print("\n--- TAHAP 1: Menyeimbangkan & Membersihkan Data ---")
    
    # 1a. Undersampling
    df_ganas = df_full[df_full['target'] == 1].copy()
    df_jinak = df_full[df_full['target'] == 0].copy()
    df_jinak_undersampled = df_jinak.sample(n=len(df_ganas), random_state=42)
    df_balanced = pd.concat([df_ganas, df_jinak_undersampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1b. Filtering (Krusial untuk menghindari error)
    IMAGE_DIR = '/kaggle/input/isic-2024-challenge/train-image/image/' # <-- Path ini sudah benar
    
    print(f"\nMemeriksa isi dari folder: {IMAGE_DIR}...")
    existing_files = set(os.listdir(IMAGE_DIR))
    df_balanced['expected_filename'] = df_balanced['isic_id'].astype(str) + '.jpg'
    df_filtered = df_balanced[df_balanced['expected_filename'].isin(existing_files)].copy()
    
    if len(df_filtered) == 0:
        raise ValueError("FATAL: Setelah filtering, tidak ada gambar yang tersisa. Terjadi ketidakcocokan nama file.")
    
    print(f"Data setelah filtering (siap pakai): {len(df_filtered)} baris.")

    
    # ===================================================================
    # BAGIAN 2: PERSIAPAN DATALOADER (Gunakan df_filtered & num_workers=0)
    # ===================================================================
    print("\n--- TAHAP 2: Menyiapkan Pipa Data (DataLoader) ---")
    IMG_SIZE = 336
    transformations = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    class ISICDataset(Dataset):
        def __init__(self, df, transforms=None):
            self.df = df
            self.image_ids = df['isic_id'].values
            self.labels = df['target'].values
            self.transforms = transforms
        def __len__(self):
            return len(self.df)
        def __getitem__(self, index):
            image_id = self.image_ids[index]
            label = self.labels[index]
            # =======================================================
            # >>> PERBAIKAN ADA DI BARIS INI <<<
            # =======================================================
            image_path = f'/kaggle/input/isic-2024-challenge/train-image/image/{image_id}.jpg'
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            if self.transforms:
                image = self.transforms(image=image)['image']
            return image, torch.tensor(label, dtype=torch.float)

    clean_dataset = ISICDataset(df_filtered, transforms=transformations)
    data_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"DataLoader siap dengan {len(clean_dataset)} gambar.")

    # ===================================================================
    # BAGIAN 3: MEMUAT MODEL PRE-TRAINED
    # ===================================================================
    print("\n--- TAHAP 3: Memuat Model Pre-trained ---")
    class ISICModel(nn.Module):
        def __init__(self, model_name, num_classes=3, pretrained=False):
            super(ISICModel, self).__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained)
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
            self.softmax = nn.Softmax(dim=1)
        def forward(self, images):
            return self.softmax(self.model(images))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
    PATH_TO_WEIGHTS = '/kaggle/input/pretrain/pytorch/default/1/ema_small_pretrained'

    model = ISICModel(MODEL_NAME, num_classes=3, pretrained=False)
    weights = torch.load(PATH_TO_WEIGHTS, map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    print(f"Model berhasil dimuat dan berjalan di device: {device}")


    # ===================================================================
    # BAGIAN 4: MENJALANKAN PREDIKSI
    # ===================================================================
    print("\n--- TAHAP 4: Menjalankan Prediksi ---")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Memprediksi"):
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print("Prediksi selesai!")


    # ===================================================================
    # BAGIAN 5: EVALUASI HASIL
    # ===================================================================
    print("\n--- TAHAP 5: Mengevaluasi Hasil ---")
    auc_scores = [roc_auc_score(all_labels, all_preds[:, i]) for i in range(all_preds.shape[1])]
    ganas_class_index = np.argmax(auc_scores)
    print(f"AUC per kelas: {auc_scores}")
    print(f"Indeks kelas 'Ganas' yang terdeteksi: {ganas_class_index}")
    
    ganas_probabilities = all_preds[:, ganas_class_index]
    binary_preds = (ganas_probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, ganas_probabilities)

    print(f"\nAccuracy on Balanced Set: {accuracy * 100:.2f}%")
    print(f"AUC on Balanced Set: {auc:.4f}")
    
    print("\nLaporan Klasifikasi:")
    print(classification_report(all_labels, binary_preds, target_names=['Jinak (0)', 'Ganas (1)']))
    
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi Jinak', 'Prediksi Ganas'], 
                yticklabels=['Aktual Jinak', 'Aktual Ganas'])
    plt.title('Confusion Matrix pada Data Seimbang')
    plt.ylabel('Label Aktual')
    plt.xlabel('Label Prediksi')
    plt.show()

except Exception as e:
    print(f"\n\n❌ TERJADI ERROR PADA PROSES UTAMA:")
    import traceback
    traceback.print_exc()


transformations_train = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),      # 50% kemungkinan dibalik horizontal
    A.VerticalFlip(p=0.5),        # 50% kemungkinan dibalik vertikal
    A.Rotate(limit=45, p=0.5),      # Diputar acak hingga 45 derajat
    A.RandomBrightnessContrast(p=0.4), # Kecerahan & kontras diubah acak
    A.HueSaturationValue(p=0.4),   # Mengubah warna sedikit
    A.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758], max_pixel_value=255.0),
    ToTensorV2(),
])


train_dataset = ISICDataset(df_filtered, transforms=transformations_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
print(f"DataLoader untuk training siap dengan {len(train_dataset)} gambar.")


improved_model = ISICModel(MODEL_NAME, num_classes=3, pretrained=False)
weights = torch.load(PATH_TO_WEIGHTS, map_location=device)
improved_model.load_state_dict(weights)
improved_model.to(device)


import torch.optim as optim

learning_rate = 1e-4
optimizer = optim.Adam(improved_model.parameters(), lr=learning_rate)
# CrossEntropyLoss cocok untuk klasifikasi multi-kelas (karena model kita punya 3 output)
loss_function = nn.CrossEntropyLoss()


EPOCHS = 20

improved_model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Reset gradien dari langkah sebelumnya
        optimizer.zero_grad()
        
        # 2. Lakukan prediksi
        outputs = improved_model(images)
        
        # 3. Hitung seberapa salah prediksinya (loss)
        # Kita ubah tipe data label menjadi long agar cocok dengan loss function
        loss = loss_function(outputs, labels.long())
        
        # 4. Hitung mundur untuk memperbaiki kesalahan (backpropagation)
        loss.backward()
        
        # 5. Update bobot model untuk menjadi lebih baik
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})


improved_model.eval() 

transformations_eval = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758], max_pixel_value=255.0),
    ToTensorV2(),
])
eval_dataset = ISICDataset(df_filtered, transforms=transformations_eval)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)


all_preds_final = []
all_labels_final = []

# Tidak perlu menghitung gradien, jadi prosesnya cepat
with torch.no_grad():
    for images, labels in tqdm(eval_loader, desc="Menguji Model"):
        images = images.to(device)
        preds = improved_model(images) # Menggunakan model yang sudah ada di memori
        all_preds_final.append(preds.cpu().numpy())
        all_labels_final.append(labels.cpu().numpy())

all_preds_final = np.concatenate(all_preds_final)
all_labels_final = np.concatenate(all_labels_final)


auc_scores = [roc_auc_score(all_labels_final, all_preds_final[:, i]) for i in range(all_preds_final.shape[1])]
ganas_class_index = np.argmax(auc_scores)
print(f"AUC per kelas: {auc_scores}")
print(f"Indeks kelas 'Ganas' yang terdeteksi: {ganas_class_index}")

ganas_probabilities = all_preds_final[:, ganas_class_index]
binary_preds = (ganas_probabilities >= 0.5).astype(int)

accuracy = accuracy_score(all_labels_final, binary_preds)
auc = roc_auc_score(all_labels_final, ganas_probabilities)

print(f"\nAkurasi Final: {accuracy * 100:.2f}%")
print(f"AUC Final: {auc:.4f}")


print("\nLaporan Klasifikasi Final:")
print(classification_report(all_labels_final, binary_preds, target_names=['Jinak (0)', 'Ganas (1)']))

cm = confusion_matrix(all_labels_final, binary_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Prediksi Jinak', 'Prediksi Ganas'], 
            yticklabels=['Aktual Jinak', 'Aktual Ganas'])
plt.title('Confusion Matrix Final')
plt.ylabel('Label Aktual')
plt.xlabel('Label Prediksi')
plt.show()


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['target'])


transformations_train = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.4), A.HueSaturationValue(p=0.4),
    A.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
    ToTensorV2(),
])
transformations_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
    ToTensorV2(),
])


train_dataset = ISICDataset(train_df, transforms=transformations_train)
val_dataset = ISICDataset(val_df, transforms=transformations_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


improved_model2 = ISICModel(MODEL_NAME, num_classes=3, pretrained=False)
weights = torch.load(PATH_TO_WEIGHTS, map_location=device)
improved_model2.load_state_dict(weights)
improved_model2.to(device)


loss_function = nn.CrossEntropyLoss()


for param in improved_model2.model.parameters():
    param.requires_grad = False
for param in improved_model2.model.head.parameters():
    param.requires_grad = True


optimizer_head = optim.Adam(improved_model.model.head.parameters(), lr=1e-3)
EPOCHS_HEAD = 20

for epoch in range(EPOCHS_HEAD):
    improved_model.train()
    for images, labels in tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{EPOCHS_HEAD}"):
        images, labels = images.to(device), labels.to(device).long()
        optimizer_head.zero_grad()
        outputs = improved_model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer_head.step()


for param in improved_model2.model.parameters():
    param.requires_grad = True

optimizer_full = optim.Adam(improved_model2.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=15, eta_min=1e-6)
EPOCHS_FULL = 15

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}


for epoch in range(EPOCHS_FULL):
    # Fase Training
    improved_model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_FULL} [Training]")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).long()
        optimizer_full.zero_grad()
        outputs = improved_model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer_full.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    
    # Fase Validasi
    improved_model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = improved_model(images)
            loss = loss_function(outputs, labels)
            total_val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ganas_class_index = np.argmax([roc_auc_score(all_labels, all_preds[:, i]) for i in range(3)])
    preds_binary = (all_preds[:, ganas_class_index] >= 0.5).astype(int)
    val_acc = accuracy_score(all_labels, preds_binary)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS_FULL} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    scheduler.step()


epochs_range = range(EPOCHS_FULL)

plt.figure(figsize=(14, 5))

# Plot Training & Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Training Loss')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


for param in improved_model2.model.parameters():
    param.requires_grad = True

optimizer_full = optim.Adam(improved_model2.parameters(), lr=2e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=15, eta_min=1e-6)
EPOCHS_FULL = 15

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}


for epoch in range(EPOCHS_FULL):
    # Fase Training
    improved_model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_FULL} [Training]")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).long()
        optimizer_full.zero_grad()
        outputs = improved_model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer_full.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    
    # Fase Validasi
    improved_model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = improved_model(images)
            loss = loss_function(outputs, labels)
            total_val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ganas_class_index = np.argmax([roc_auc_score(all_labels, all_preds[:, i]) for i in range(3)])
    preds_binary = (all_preds[:, ganas_class_index] >= 0.5).astype(int)
    val_acc = accuracy_score(all_labels, preds_binary)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS_FULL} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    scheduler.step()




