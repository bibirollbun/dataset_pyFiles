import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")


data_dir = '/kaggle/input/histopathologic-cancer-detection'
train_image_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test')


train_df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
print(f"Eğitim veri seti boyutu: {len(train_df)}")
train_df.head()


plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_df)
plt.title('Kanser Etiketi Dağılımı')
plt.xlabel('Etiket (0: Kansersiz, 1: Kanserli)')
plt.ylabel('Sayı')
plt.show()

print(train_df['label'].value_counts(normalize=True))


fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, idx in enumerate(train_df[train_df['label'] == 1].sample(5).index):
    img_id = train_df.loc[idx, 'id']
    img_path = os.path.join(train_image_dir, f"{img_id}.tif") # .tif uzantısına dikkat
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"Label: {train_df.loc[idx, 'label']}")
    axes[i].axis('off')
plt.suptitle('Kanserli Örnek Görüntüler')
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, idx in enumerate(train_df[train_df['label'] == 0].sample(5).index):
    img_id = train_df.loc[idx, 'id']
    img_path = os.path.join(train_image_dir, f"{img_id}.tif")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"Label: {train_df.loc[idx, 'label']}")
    axes[i].axis('off')
plt.suptitle('Kansersiz Örnek Görüntüler')
plt.show()


class CancerDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Görüntü ID'sini ve etiketini al
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['id'] + '.tif')
        label = self.df.iloc[idx]['label']

        # Görüntüyü yükle
        image = Image.open(img_name).convert('RGB') # .tif dosyaları için bazen 'RGB'ye çevirmek gerekebilir

        # Dönüşümleri uygula (eğer varsa)
        if self.transform:
            image = self.transform(image)

        # Etiketi PyTorch tensörüne çevir (float32 çünkü BCEWithLogitsLoss float bekler)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


data_dir = '/kaggle/input/histopathologic-cancer-detection'
train_labels_path = os.path.join(data_dir, 'train_labels.csv')
train_image_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test')

train_df = pd.read_csv(train_labels_path)

IMAGE_SIZE = 96


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), # Rastgele kırpma ve boyutlandırma
    transforms.RandomHorizontalFlip(), # Rastgele yatay çevirme
    transforms.RandomVerticalFlip(),   # Rastgele dikey çevirme
    transforms.RandomRotation(90),     # Rastgele 90 derece döndürme
    transforms.ToTensor(),             # Görüntüyü PyTorch tensörüne çevir (0-1 aralığına otomatik ölçekler)
    transforms.Normalize(NORM_MEAN, NORM_STD) # Normalizasyon
])

# Doğrulama (validation) ve Test verisi için dönüşümler (Sadece boyutlandırma, Tensör'e çevirme, normalizasyon)
# Veri artırma test/doğrulama setlerinde uygulanmaz
val_test_transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE), # Görüntünün ortasından kırpma
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])



train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])

print(f"\nEğitim veri seti boyutu: {len(train_df)} görüntü")
print(f"Doğrulama veri seti boyutu: {len(val_df)} görüntü")

# Dataset nesnelerini oluştur
train_dataset = CancerDataset(df=train_df, img_dir=train_image_dir, transform=train_transform)
val_dataset = CancerDataset(df=val_df, img_dir=train_image_dir, transform=val_test_transform)

# DataLoader nesnelerini oluştur
BATCH_SIZE = 32 # Batch büyüklüğünü ayarlayabilirsiniz

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # num_workers CPU çekirdeği kullanımı
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\nEğitim DataLoader: {len(train_loader)} batch, her batch {BATCH_SIZE} görüntü")
print(f"Doğrulama DataLoader: {len(val_loader)} batch, her batch {BATCH_SIZE} görüntü")

# Bir örnek batch alıp kontrol edelim
for images, labels in train_loader:
    print(f"\nİlk eğitim batch'i: Resimler boyutu {images.shape}, Etiketler boyutu {labels.shape}")
    break # Sadece ilk batch'i kontrol edip çık



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        
        self.fc1 = nn.Linear(in_features=64*24*24, out_features=600)
        self.dropout = nn.Dropout(p=0.3)
       
        self.fc2 = nn.Linear(in_features=600, out_features=1)
        

    def forward(self, x):
        # İlk blok
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.max_pool2d(x, kernel_size=2) # nn.MaxPool2d yerine F.max_pool2d kullanabiliriz

        # İkinci blok
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.max_pool2d(x, kernel_size=2) # nn.MaxPool2d yerine F.max_pool2d kullanabiliriz

        # Düzleştirme
        x = x.view(x.size(0), -1) # Flatten

        # Tam bağlı katmanlar
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Son çıktı (logit)
        x = self.fc2(x)
        return x


model = CNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Metrikleri kaydetmek için listeler
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_aucs = [] # AUC skoru da önemli bir metrik

best_val_auc = 0.0 # En iyi modeli kaydetmek için



num_epochs = 5 

print("\n--- Model Eğitimi Başlıyor ---")
for epoch in range(num_epochs):
   
    model.train() 
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images, labels = images.to(device), labels.to(device).unsqueeze(1) # Etiketlerin boyutunu (N,1) yap

        
        optimizer.zero_grad()

        
        outputs = model(images)

       
        loss = criterion(outputs, labels)

        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0) 

       
        probabilities = torch.sigmoid(outputs)
        all_train_preds.extend(probabilities.cpu().detach().numpy())
        all_train_labels.extend(labels.cpu().detach().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    train_preds_binary = (np.array(all_train_preds) >= 0.5).astype(int)
    epoch_train_accuracy = accuracy_score(all_train_labels, train_preds_binary)
    epoch_train_auc = roc_auc_score(all_train_labels, np.array(all_train_preds))


    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

  
    model.eval() 
    val_running_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad(): # Gradyan hesaplamayı kapat, bellekten tasarruf et
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)
            all_val_preds.extend(probabilities.cpu().detach().numpy())
            all_val_labels.extend(labels.cpu().detach().numpy())

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_preds_binary = (np.array(all_val_preds) >= 0.5).astype(int)
    epoch_val_accuracy = accuracy_score(all_val_labels, val_preds_binary)
    epoch_val_auc = roc_auc_score(all_val_labels, np.array(all_val_preds))

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)
    val_aucs.append(epoch_val_auc)


    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Train AUC: {epoch_train_auc:.4f}")
    print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_accuracy:.4f}, Val AUC:   {epoch_val_auc:.4f}")

    
    if epoch_val_auc > best_val_auc:
        best_val_auc = epoch_val_auc
        torch.save(model.state_dict(), 'best_cnn_model.pth')
        print(f"  En iyi model kaydedildi! Val AUC: {best_val_auc:.4f}")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Eğitim Kaybı')
plt.plot(range(1, num_epochs + 1), val_losses, label='Doğrulama Kaybı')
plt.title('Kayıp (Loss) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Eğitim Doğruluğu')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Doğrulama Doğruluğu')
plt.plot(range(1, num_epochs + 1), val_aucs, label='Doğrulama AUC', linestyle='--')
plt.title('Doğruluk (Accuracy) ve AUC Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




