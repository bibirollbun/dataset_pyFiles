import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import torch.nn as nn

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

# Etiketleri sayısal verilere dönüştürme için bir fonksiyon
def label_to_numeric(label):
    if label == 'Severe':
        return 1
    elif label == 'Moderate':
        return 2
    elif label == 'Mild':
        return 3
    else:
        return 0  # 'Normal' gibi bir değer

# Test Verisini Yükleme
test_labels = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv')
test_labels['study_id'] = test_labels['study_id'].astype(str)

# Etiketleri sayıya dönüştürme
test_labels.iloc[:, 1:] = test_labels.iloc[:, 1:].applymap(label_to_numeric)

# DICOM Veri Seti
img_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset ve DataLoader
dataset = SpineDataset(dataframe=test_labels, img_dir=img_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: list(filter(None, x)))  # Boş batch'leri atla

# Modeli Yükleyin
model_path = '/kaggle/input/2222222/all_epoch_models-2.pth'
model = models.resnet18(weights=None)  # Pretrained kullanmıyoruz
model.fc = nn.Linear(model.fc.in_features, len(test_labels.columns[1:]))  # Test verisi için etiket sayısını ayarlayın

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Modeli yükleyin (CPU'ya yükleme)
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)  # weights_only=True kullanıldı
    model.load_state_dict(checkpoint, strict=False)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")

model.eval()

# Tahminler için boş bir liste oluşturun
predictions = []

# Test verisinde tahmin yapma
with torch.no_grad():
    for batch in test_loader:
        if not batch:
            continue  # Eğer batch boşsa, bir sonraki batch'e geç

        images, _ = zip(*batch)  # Etiketlere gerek yok
        images = torch.stack(images).to(device)

        outputs = model(images)
        predictions.append(outputs.cpu().numpy())

# Tahminleri birleştirme
predictions = np.concatenate(predictions, axis=0)

# Sonuçları DataFrame'e dönüştürme
submission = pd.DataFrame(predictions, columns=test_labels.columns[1:])
submission['study_id'] = test_labels['study_id']

# Sonuçları 'Label' ve 'Count' başlıklarıyla uzun formata dönüştürme
submission_long = pd.melt(submission, id_vars=['study_id'], var_name='Label', value_name='Count')

# Sonuçları CSV'ye kaydetme
submission_file = '/kaggle/working/submission.csv'
submission_long.to_csv(submission_file, index=False)

print(f"Submission dosyası kaydedildi: {submission_file}")


