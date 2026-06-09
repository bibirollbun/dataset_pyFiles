import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import pydicom
import torch.nn as nn
from torchvision.models import ResNet18_Weights

class SpineDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_id = self.df.iloc[idx]['study_id']
        study_dir = os.path.join(self.img_dir, str(study_id))
        
        # Dizin yoksa veya DICOM dosyası yoksa None döndür
        if not os.path.isdir(study_dir):
            return None
        
        # DICOM dosyasını arama
        dicom_file_found = False
        dicom_file_path = None

        for root, dirs, files in os.walk(study_dir):
            dicom_files = [f for f in files if f.lower().endswith('.dcm')]
            if dicom_files:
                dicom_file_found = True
                dicom_file_path = os.path.join(root, dicom_files[0])
                break
        
        if not dicom_file_found:
            return None
        
        # DICOM dosyasını yükle
        dicom_data = pydicom.dcmread(dicom_file_path)
        image = dicom_data.pixel_array

        # Gri tonlamalı resmi RGB'ye dönüştür
        image = Image.fromarray(image).convert('RGB')
        
        # Etiketleri al
        labels = self.df.iloc[idx, 1:].values
        labels = labels.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)


# CSV Dosyasını Yükleme
train_labels = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
train_labels['study_id'] = train_labels['study_id'].astype(str)

# Etiketleri binary sınıflara dönüştürme
label_columns = train_labels.columns[1:]  # İlk sütun haricindeki tüm sütunlar etiket
train_labels[label_columns] = train_labels[label_columns].apply(lambda col: col.map(lambda v: 1 if v == 'Severe' else 0))

# DICOM Veri Seti
img_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Geçerli satırlardan yeni bir dataframe oluştur
valid_rows = []
for idx in range(len(train_labels)):
    study_id = train_labels.iloc[idx]['study_id']
    study_dir = os.path.join(img_dir, str(study_id))
    
    # Dizin yoksa ya da DICOM dosyası yoksa bu satırı atla
    dicom_file_found = False
    for root, dirs, files in os.walk(study_dir):
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]
        if dicom_files:
            dicom_file_found = True
            break
    
    if dicom_file_found:
        valid_rows.append(idx)

# Geçerli veri kümesi
valid_train_labels = train_labels.iloc[valid_rows]

# Eğer geçerli veri yoksa, eğitime devam etme
if len(valid_train_labels) == 0:
    print("Geçerli veri bulunamadı. Eğitim yapılamaz.")
else:
    # Dataset ve DataLoader
    dataset = SpineDataset(dataframe=valid_train_labels, img_dir=img_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: list(filter(None, x)))  # Boş batch'leri atla

    # Model Tanımlaması
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Güncellenmiş model yükleme
    model.fc = nn.Linear(model.fc.in_features, len(label_columns))  # Son katmanı etiket sayısına göre ayarlayın

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Eğitim döngüsü
    model.train()
    all_epoch_models = {}  # Her epok için model ağırlıklarını saklamak için bir sözlük

    for epoch in range(10):
        train_loss = 0
        for batch in train_loader:
            if not batch:
                continue  # Eğer batch boşsa, bir sonraki batch'e geç

            images, labels = zip(*batch)  # Batch'i ayır
            images = torch.stack(images).to(device)
            labels = torch.stack(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}')

        # Her epokta modelin ağırlıklarını kaydetme
        all_epoch_models[epoch] = model.state_dict()

    # Sonunda tüm epokları kaydetme
    model_save_path = '/kaggle/working/all_epoch_models.pth'  # Kaydedilecek dosyanın yolu
    torch.save(all_epoch_models, model_save_path)  # Modelin her epoktaki ağırlıklarını kaydediyoruz
    print(f"Tüm epokların modelleri kaydedildi: {model_save_path}")





