import seaborn as sns
import cv2 

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import glob
import json
import collections
import torch
import torch.nn as nn

import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# read data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')
len(test_desc) #number of test_description.csv rows 


test_desc.head(5)


train_desc.head(5)


train.head(5)


# Function to generate image paths based on directory structure
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = os.listdir(series_dir)
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

# Generate image paths for train and test data
train_image_paths = generate_image_paths(train_desc, f'{train_path}/train_images')
test_image_paths = generate_image_paths(test_desc, f'{train_path}/test_images')


len(train_desc)


len(train_image_paths)


# Define function to reshape a single row of the DataFrame
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    
    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    
    return pd.DataFrame(data)

# Reshape the DataFrame for all rows
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)

# Display the first few rows of the reshaped dataframe
new_train_df.head(5)


# Print columns in a neat way
print("\nColumns in new_train_df:")
print(",".join(new_train_df.columns))

print("\nColumns in label:")
print(",".join(label.columns))

print("\nColumns in test_desc:")
print(",".join(test_desc.columns))

print("\nColumns in sub:")
print(",".join(sub.columns))


# Merge the dataframes on the common columns
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on='series_id', how='inner')


# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id','study_id'], how='inner')
# Display the first few rows of the final merged dataframe
final_merged_df.head(5)


import pandas as pd

# Create the row_id column
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the image_path column
final_merged_df['image_path'] = (
    f'{train_path}/train_images/' + 
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Note: Check image path, since there's 1 instance id, for 1 image, but there's many more images other than the ones labelled in the instance ID. 

# Display the updated dataframe
final_merged_df.head(5)


final_merged_df[final_merged_df["severity"] == "Normal/Mild"].value_counts().sum()


final_merged_df[final_merged_df["severity"] == "Moderate"].value_counts().sum()


final_merged_df[final_merged_df["severity"] == "Severe"].value_counts().sum()


"""
import pandas as pd

# En düşük sınıf sayısını belirleyelim
min_class_count = 3081

# Normal/Mild ve Moderate sınıflarını azaltalım
normal_mild_df = final_merged_df[final_merged_df["severity"] == "Normal/Mild"].sample(n=min_class_count, random_state=42)
moderate_df = final_merged_df[final_merged_df["severity"] == "Moderate"].sample(n=min_class_count, random_state=42)
severe_df = final_merged_df[final_merged_df["severity"] == "Severe"]

# İndeksleri sıfırlayalım
normal_mild_df = normal_mild_df.reset_index(drop=True)
moderate_df = moderate_df.reset_index(drop=True)
severe_df = severe_df.reset_index(drop=True)

# Verileri birleştirelim ve final_merged_df'yi güncelleyelim
final_merged_df = pd.concat([normal_mild_df, moderate_df, severe_df])

# Sonuçları kontrol edelim
print(final_merged_df["severity"].value_counts())
"""


final_merged_df[final_merged_df["severity"] == "Normal/Mild"].value_counts().sum()


final_merged_df[final_merged_df["severity"] == "Moderate"].value_counts().sum()


final_merged_df[final_merged_df["severity"] == "Severe"].value_counts().sum()


# Define the base path for test images
base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Function to get image paths for a series
def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)

# Display the resulting dataframe
expanded_test_desc.head(5)


# change severity column labels
#Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'}
final_merged_df['severity'] = final_merged_df['severity'].map({'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})


test_data = expanded_test_desc
train_data = final_merged_df


train_data.head(5)


train_data['series_description'].value_counts()


def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


import random
import matplotlib.pyplot as plt

# Yeni sıfırlanmış indekslerle rastgele seçim yapalım
final_merged_df_reset = final_merged_df.reset_index(drop=True)

# Rastgele iki indeks seçelim
selected_indices = random.sample(range(len(final_merged_df_reset)), 2)

images = []
row_ids = []

# Seçilen indekslerle görselleri yükleyelim
for i in selected_indices:
    image = load_dicom(final_merged_df_reset['image_path'][i])  # Yeni sıfırlanmış indeksi kullan
    images.append(image)
    row_ids.append(final_merged_df_reset['row_id'][i])  # Yeni sıfırlanmış indeksi kullan

# Görselleri çizdirelim
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_title(f'Row ID: {row_ids[i]}', fontsize=8)
    ax[i].axis('off')
plt.tight_layout()
plt.show()



train_data 


train_data = train_data.dropna()


# Severity columnundaki her bir etiketin sayısını sayalım
severity_counts = train_data['severity'].value_counts()

# Sonuçları yazdır
print(severity_counts)


import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, apply_augmentation_to_all=True):
        self.dataframe = dataframe
        self.transform = transform
        self.apply_augmentation_to_all = apply_augmentation_to_all  # Tüm sınıflara augmentasyon uygula

    def __len__(self):
        return len(self.dataframe) * 2  # Her iki versiyonu kullanarak veri kümesini iki katına çıkarıyoruz

    def __getitem__(self, index):
        original_index = index // 2  # Her iki versiyon için aynı orijinal index'i al
        is_augmented = index % 2  # Orijinal mi, augmentasyonlu mu olduğunu kontrol et

        image_path = self.dataframe['image_path'][original_index]
        image = load_dicom(image_path)  # DICOM dosyasını yükle
        label = self.dataframe['severity'][original_index]  # Etiketi al

        # Orijinal görüntü
        original_image = np.stack([image] * 3, axis=-1)  # 1 kanal olan görüntüyü 3 kanala dönüştür

        # Sadece Severe ve Moderate etiketleri için augmentasyon uygula
        if is_augmented and self.apply_augmentation_to_all and label in ['Severe', 'Moderate']:
            image = self.apply_augmentation(image)  # Sadece augmentasyonlu versiyon için
            augmented_image = np.stack([image] * 3, axis=-1)
        else:
            augmented_image = original_image  # Orijinal görüntü

        # Transformasyonu uygula
        if self.transform:
            original_image = self.transform(original_image)
            augmented_image = self.transform(augmented_image)

        return augmented_image, label  # Augmentasyonlu görüntü ve etiket döndürülür

    # Normalize DICOM görüntüsü
    def normalize_dicom(self, image, window_min=-1000, window_max=400):
        image = np.clip(image, window_min, window_max)
        return (image - window_min) / (window_max - window_min)

    # Augmentasyon uygulama
    def apply_augmentation(self, image):
        augmentations = [
            lambda img: np.rot90(img, k=np.random.randint(1, 4)),  # Random rotation
            lambda img: np.fliplr(img),  # Horizontal flip
            lambda img: np.flipud(img),  # Vertical flip
            lambda img: self.random_crop(img),  # Random crop
            lambda img: self.add_gaussian_noise(img),  # Gaussian noise
            lambda img: self.adjust_brightness(img),  # Brightness adjustment
            lambda img: self.adjust_contrast(img),  # Contrast adjustment
        ]
        
        # Bir augmentasyon seç ve uygula
        augmentation = np.random.choice(augmentations)
        return augmentation(image)

    # Random crop
    def random_crop(self, image, crop_size=(224, 224)):
        h, w = image.shape
        new_h, new_w = crop_size
        
        # Yükseklik ve genişlik farkının negatif olmamasını sağla
        top = np.random.randint(0, max(h - new_h, 1))  # min 1
        left = np.random.randint(0, max(w - new_w, 1))  # min 1
        
        return image[top:top + new_h, left:left + new_w]

    # Gaussian noise ekleme
    def add_gaussian_noise(self, image, mean=0, std=0.01):
        noise = np.random.normal(mean, std, image.shape)
        return np.clip(image + noise, 0, 255)  # 0-255 arasında tutmak

    # Parlaklık ayarı
    def adjust_brightness(self, image, factor_range=(0.95, 1.05)):
        factor = np.random.uniform(*factor_range)
        return np.clip(image * factor, 0, 255)  # 0-255 arasında tutmak

    # Kontrast ayarı
    def adjust_contrast(self, image, factor_range=(0.95, 1.05)):
        mean = np.mean(image)
        factor = np.random.uniform(*factor_range)
        return np.clip((image - mean) * factor + mean, 0, 255)  # 0-255 arasında tutmak


# Function to create datasets and dataloaders for each series description
def create_datasets_and_loaders(df, series_description, transform, batch_size=8):
    # Belirtilen seriye göre filtreleme
    filtered_df = df[df['series_description'] == series_description]
    
    # Tüm verinin %100'ünü kullanıyoruz (isteğe bağlı olarak frac değiştirilebilir)
    filtered_df = filtered_df.sample(frac=1.0, random_state=42)  
    
    # Eğitim ve doğrulama setlerine bölme
    train_df, val_df = train_test_split(filtered_df, test_size=0.3, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Dataset oluşturma (augmentasyon ve transform tüm veriye uygulanır)
    train_dataset = CustomDataset(train_df, transform=transform, apply_augmentation_to_all=True)  # Eğitimde augmentasyon
    val_dataset = CustomDataset(val_df, transform=transform, apply_augmentation_to_all=False)  # Validasyonda augmentasyon yok


    # DataLoader oluşturma
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, len(train_df), len(val_df)


# Define the transforms
transform = transforms.Compose([
    transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create dataloaders for each series description
dataloaders = {}
lengths = {}

# DataLoader'ları oluşturma
trainloader_t1, valloader_t1, len_train_t1, len_val_t1 = create_datasets_and_loaders(
    train_data, 'Sagittal T1', transform
)
trainloader_t2, valloader_t2, len_train_t2, len_val_t2 = create_datasets_and_loaders(
    train_data, 'Axial T2', transform
)
trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir = create_datasets_and_loaders(
    train_data, 'Sagittal T2/STIR', transform
)

dataloaders['Sagittal T1'] = (trainloader_t1, valloader_t1)
dataloaders['Axial T2'] = (trainloader_t2, valloader_t2)
dataloaders['Sagittal T2/STIR'] = (trainloader_t2stir, valloader_t2stir)

lengths['Sagittal T1'] = (len_train_t1, len_val_t1)
lengths['Axial T2'] = (len_train_t2, len_val_t2)
lengths['Sagittal T2/STIR'] = (len_train_t2stir, len_val_t2stir)


# Dictionary mapping labels to indices
label_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}


print(f"Özgün Eğitim seti uzunluğu: {len(trainloader_t1.dataset) / 2}")  # Yarıya bölüyoruz çünkü her örnek iki kere sayılıyor
print(f"Özgün Doğrulama seti uzunluğu: {len(valloader_t1.dataset) / 2}")  # Aynı şekilde
print(f"Eğitim setinin uzunluğu: {len(trainloader_t1.dataset)}")  # Eğitim setinin uzunluğunu yazdır
print(f"Doğrulama setinin uzunluğu: {len(valloader_t1.dataset)}")  # Doğrulama setinin uzunluğunu yazdır
print("---------------------------------------------------------------------\n")
print(f"Özgün Eğitim seti uzunluğu: {len(trainloader_t2.dataset) / 2}")  # Yarıya bölüyoruz çünkü her örnek iki kere sayılıyor
print(f"Özgün Doğrulama seti uzunluğu: {len(valloader_t2.dataset) / 2}")  # Aynı şekilde
print(f"Eğitim setinin uzunluğu: {len(trainloader_t2.dataset)}")  # Eğitim setinin uzunluğunu yazdır
print(f"Doğrulama setinin uzunluğu: {len(valloader_t2.dataset)}")  # Doğrulama setinin uzunluğunu yazdır
print("---------------------------------------------------------------------\n")
print(f"Özgün Eğitim seti uzunluğu: {len(trainloader_t2stir.dataset) / 2}")  # Yarıya bölüyoruz çünkü her örnek iki kere sayılıyor
print(f"Özgün Doğrulama seti uzunluğu: {len(valloader_t2stir.dataset) / 2}")  # Aynı şekilde
print(f"Eğitim setinin uzunluğu: {len(trainloader_t2stir.dataset)}")  # Eğitim setinin uzunluğunu yazdır
print(f"Doğrulama setinin uzunluğu: {len(valloader_t2stir.dataset)}")  # Doğrulama setinin uzunluğunu yazdır


# Etiketlere göre veri sayısını öğrenme
label_counts = train_data['condition'].value_counts()

# label_map ile eşleşen her bir etiketin sayısını yazdırma
for label, index in label_map.items():
    print(f"{label}: {label_counts.get(index, 0)}")



import matplotlib.pyplot as plt

# Function to visualize a batch of images
def visualize_batch(dataloader):
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i]
        img = img.permute(1, 2, 0)  # Convert to HWC for visualization
        ax.imshow(img)
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')
    plt.show()

# Visualize samples from each dataloader
print("Visualizing Sagittal T1 samples")
visualize_batch(trainloader_t1)
print("Visualizing Axial T2 samples")
visualize_batch(trainloader_t2)
print("Visualizing Sagittal T2/STIR samples")
visualize_batch(trainloader_t2stir)


import matplotlib.pyplot as plt

image, label = next(iter(trainloader_t2))
sample = image[1].permute(1, 2, 0)  #sample

# Görüntü ve etiketleri çiz
plt.figure(figsize=(8, 4))
plt.imshow(sample)  # Eğer 3 kanallı ise cmap kullanmaya gerek yok
plt.title(label[1])  # Etiketi yazdır
plt.axis('off')  # Ekseni kapat
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # ResNet50 ağırlıkları için enum'u import et

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomResNet50, self).__init__()
        
        # pretrained=True yerine weights=ResNet50_Weights.IMAGENET1K_V1 kullanarak ağırlıkları yükleyin
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        
        # Eğer manuel ağırlık yolu verilmişse, bu ağırlıkları yükle
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        
        num_ftrs = self.model.fc.in_features  # Son katmanın özellik sayısını al
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Son katmanı değiştirme

    def forward(self, x):
        return self.model(x)

    def unfreeze_middle_layers(self):
        """Orta katmanları çöz."""
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name:  
                param.requires_grad = True
            else:
                param.requires_grad = False

# Modeli başlat
sagittal_t1_model = CustomResNet50(num_classes=3).to(device)
axial_t2_model = CustomResNet50(num_classes=3).to(device)
sagittal_t2stir_model = CustomResNet50(num_classes=3).to(device)

# Orta katmanları çözmek için
for model in [sagittal_t1_model, axial_t2_model, sagittal_t2stir_model]:
    model.unfreeze_middle_layers()  # Orta katmanları çöz

# Eğitim parametreleri
weights = torch.tensor([1.0, 2.0, 4.0])
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Optimizer ayarları
optimizer_sagittal_t1 = torch.optim.Adam(sagittal_t1_model.parameters(), lr=0.001)
optimizer_axial_t2 = torch.optim.Adam(axial_t2_model.parameters(), lr=0.001)
optimizer_sagittal_t2stir = torch.optim.Adam(sagittal_t2stir_model.parameters(), lr=0.001)

# Modelleri ve optimizörleri saklamak için dictionary
models_dict = {
    'Sagittal T1': sagittal_t1_model,
    'Axial T2': axial_t2_model,
    'Sagittal T2/STIR': sagittal_t2stir_model,
}

optimizers_dict = {
    'Sagittal T1': optimizer_sagittal_t1,
    'Axial T2': optimizer_axial_t2,
    'Sagittal T2/STIR': optimizer_sagittal_t2stir,
}

# Eğitim yapılabilir parametrelerin sayısını yazdır
for model_name, model in models_dict.items():
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters for {model_name}: {trainable_params}")



label_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}


for images, labels in trainloader_t2:
    labels = torch.tensor([label_map[label] for label in labels])
    labels = labels.to(device)
    print(labels)
    break


from sklearn.metrics import precision_score, recall_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score



import torch.optim.lr_scheduler as lr_scheduler
from copy import deepcopy
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report
)

def train_model(model, trainloader, valloader, len_train, len_val, optimizer, num_epochs=10, patience=3):
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_val_acc = 0.0
    best_model_wts = deepcopy(model.state_dict())
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0

        with tqdm(trainloader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), torch.tensor([label_map[label] for label in labels]).to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                correct_train += (predicted == labels).sum().item()

                tepoch.set_postfix(epoch=epoch+1)

        scheduler.step()

        train_loss /= len(trainloader)
        train_acc = 100 * correct_train / len_train

        model.eval()
        val_loss, correct_val = 0, 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            with tqdm(valloader, unit="batch") as vepoch:
                for images, labels in vepoch:
                    images, labels = images.to(device), torch.tensor([label_map[label] for label in labels]).to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)

                    if probabilities.dim() == 1:
                        _, predicted = torch.max(probabilities, 0)
                    else:
                        _, predicted = torch.max(probabilities, 1)
                    correct_val += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    vepoch.set_postfix(epoch=epoch+1)

        val_loss /= len(valloader)
        val_acc = 100 * correct_val / len_val

        # Confusion Matrix ve F1 Skoru Hesaplama
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        class_report = classification_report(all_labels, all_predictions)

        # Precision ve Recall
        precision = precision_score(all_labels, all_predictions, average="weighted")
        recall = recall_score(all_labels, all_predictions, average="weighted")

        # Cohen's Kappa
        kappa = cohen_kappa_score(all_labels, all_predictions)

        # Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(all_labels, all_predictions)

        # Balanced Accuracy
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

        # ROC-AUC (Eğer çok sınıflıysa one-vs-rest yaklaşımı kullanılır)
        try:
            roc_auc = roc_auc_score(all_labels, torch.nn.functional.one_hot(torch.tensor(all_predictions), num_classes=len(set(all_labels))).numpy(), multi_class="ovr")
        except ValueError:
            roc_auc = "N/A (ROC-AUC çok sınıflı problemde uygun olmayabilir)"

        # Çıktılar
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"ROC-AUC Score: {roc_auc}")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Classification Report:\n{class_report}")

        # Save the best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = deepcopy(model.state_dict())
            counter = 0
            torch.save(best_model_wts, f'best_model_{epoch+1}.pth')
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_val_acc


#!rm -rf /kaggle/working/*


# Eğitim işlemi
for desc, model in models_dict.items():  # Burada 'models_dict' kullanılmalı
    if desc == 'Sagittal T1':
        trainloader, valloader, len_train, len_val = trainloader_t1, valloader_t1, len_train_t1, len_val_t1
    elif desc == 'Axial T2':
        trainloader, valloader, len_train, len_val = trainloader_t2, valloader_t2, len_train_t2, len_val_t2
    elif desc == 'Sagittal T2/STIR':
        trainloader, valloader, len_train, len_val = trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir
    
    print(f"Training model for {desc}")
    train_model(model, trainloader, valloader, len_train, len_val, optimizers_dict[desc])  # Burada 'optimizers_dict' kullanılmalı


train_data['level'].unique()

