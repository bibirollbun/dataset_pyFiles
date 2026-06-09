import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


import zipfile
import os


# Zip dosyalarını aç
train_zip_path = '/kaggle/input/facial-keypoints-detection/training.zip'
test_zip_path = '/kaggle/input/facial-keypoints-detection/test.zip'

# Training zip'i aç
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# Test zip'i aç
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')



train_df = pd.read_csv('/kaggle/working/training.csv')
test_df = pd.read_csv('/kaggle/working/test.csv')
lookup_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")
print(f"Lookup tablosu boyutu: {lookup_df.shape}")


train_df['Image'] = train_df['Image'].apply(lambda x: np.fromstring(x, sep=' '))
test_df['Image'] = test_df['Image'].apply(lambda x: np.fromstring(x, sep=' '))


print(f"\nEksik değer sayısı:\n{train_df.isnull().sum()}")


train_df_clean = train_df.dropna()
print(f"Temizlenmiş eğitim verisi boyutu: {train_df_clean.shape}")


# X ve y verilerini hazırla
X_train = np.vstack(train_df_clean['Image'].values).reshape(-1, 96, 96, 1)
y_train = train_df_clean.drop('Image', axis=1).values

X_test = np.vstack(test_df['Image'].values).reshape(-1, 96, 96, 1)


# Normalize et (0-255 -> 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Koordinatları normalize et (-1, 1 aralığına)
y_train = (y_train - 48) / 48


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Yatay çevirme (flip) ile veri arttır
X_train_flipped = np.flip(X_tr, axis=2).copy()
y_train_flipped = y_tr.copy()


# Koordinatları yatay çevirme için ayarla (x koordinatlarını tersine çevir)
for i in range(0, 30, 2):
    y_train_flipped[:, i] = -y_train_flipped[:, i]


# Gözler ve kaşlar için sağ-sol değiştir
swap_indices = [
    (0, 2), (1, 3),  # sol-sağ göz merkezi
    (4, 6), (5, 7),  # sol-sağ göz iç köşe
    (8, 10), (9, 11),  # sol-sağ göz dış köşe
    (12, 14), (13, 15),  # sol-sağ kaş iç uç
    (16, 18), (17, 19),  # sol-sağ kaş dış uç
    (22, 24), (23, 25)  # sol-sağ ağız köşe
]



for idx1, idx2 in swap_indices:
    temp = y_train_flipped[:, idx1].copy()
    y_train_flipped[:, idx1] = y_train_flipped[:, idx2]
    y_train_flipped[:, idx2] = temp

# Arttırılmış veriyi birleştir
X_tr_aug = np.concatenate([X_tr, X_train_flipped], axis=0)
y_tr_aug = np.concatenate([y_tr, y_train_flipped], axis=0)

print(f"Arttırılmış eğitim verisi boyutu: {X_tr_aug.shape}")


# ==================== PYTORCH DATASET ====================
class FacialKeypointsDataset(Dataset):
    def __init__(self, images, keypoints=None):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.keypoints = torch.FloatTensor(keypoints) if keypoints is not None else None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.keypoints is not None:
            return self.images[idx], self.keypoints[idx]
        return self.images[idx]

# Dataset ve DataLoader oluştur
train_dataset = FacialKeypointsDataset(X_tr_aug, y_tr_aug)
val_dataset = FacialKeypointsDataset(X_val, y_val)
test_dataset = FacialKeypointsDataset(X_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== MODEL TANIMLAMA ====================
class FacialKeypointsCNN(nn.Module):
    def __init__(self):
        super(FacialKeypointsCNN, self).__init__()
        
        # Convolutional katmanlar
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_fc = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 30)
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Conv block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Conv block 5
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(-1, 512 * 3 * 3)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        return x



# Model, loss ve optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nKullanılan cihaz: {device}")


model = FacialKeypointsCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters())}")


num_epochs = 100
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15

train_losses = []
val_losses = []


for epoch in range(num_epochs):
    # Eğitim
    model.train()
    train_loss = 0.0
    
    for images, keypoints in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        images = images.to(device)
        keypoints = keypoints.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, keypoints in val_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Model kaydetme
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
        print(f"Model kaydedildi! Val Loss: {val_loss:.6f}")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\nErken durma! {early_stop_patience} epoch'ta iyileşme yok.")
        break



# ==================== LOSS GRAFİĞİ ====================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eğitim ve Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/training_loss.png')
plt.show()


# ==================== EN İYİ MODELİ YÜKLE ====================
print("\nEn iyi model yükleniyor...")
model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
model.eval()



# ==================== TAHMİN ====================
print("\nTest verisi üzerinde tahmin yapılıyor...")

predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Tahmin"):
        images = images.to(device)
        outputs = model(images)
        predictions.append(outputs.cpu().numpy())

predictions = np.vstack(predictions)

# Denormalize et (koordinatları orijinal ölçeğe çevir)
predictions = predictions * 48 + 48

print(f"Tahmin şekli: {predictions.shape}")



# IdLookupTable kullanarak submission oluştur
submission = []
feature_names = train_df_clean.columns[:-1].tolist()

for idx, row in lookup_df.iterrows():
    image_id = row['ImageId'] - 1  # 0-indexed
    feature_name = row['FeatureName']
    row_id = row['RowId']
    
    # Feature index'ini bul
    feature_idx = feature_names.index(feature_name)
    
    # İlgili tahmini al
    predicted_value = predictions[image_id, feature_idx]
    
    submission.append([row_id, predicted_value])

submission_df = pd.DataFrame(submission, columns=['RowId', 'Location'])
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("\nSubmission dosyası 'submission.csv' olarak kaydedildi!")
print(f"Submission boyutu: {submission_df.shape}")
print("\nİlk 10 tahmin:")
print(submission_df.head(10))




