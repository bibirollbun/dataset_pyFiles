# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
counter = 0  # Sayaç başlat
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        counter += 1
        if counter == 15:  # 5 dosya yazdırdıktan sonra dur
            break
    if counter == 15:  # İç döngü kırıldığında dış döngüyü de kır
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns

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


import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        label = self.dataframe['severity'][index]
        
        if self.transform:
            image = self.transform(image)

        return image, label

"""# Function to create datasets and dataloaders for each series description
def create_datasets_and_loaders(df, series_description, transform, batch_size=8):
    filtered_df = df[df['series_description'] == series_description]
    
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(val_df, transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, len(train_df), len(val_df)"""
# Function to create datasets and dataloaders for each series description
def create_datasets_and_loaders(df, series_description, transform, batch_size=8):
    filtered_df = df[df['series_description'] == series_description]
    
    # %5'ini al frac değerini değiştirerek trainde verinin ne kadarını kullanacagını belirlersin
    filtered_df = filtered_df.sample(frac=1.0, random_state=42)  
    
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(val_df, transform)

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

trainloader_t1, valloader_t1, len_train_t1, len_val_t1 = create_datasets_and_loaders(train_data, 'Sagittal T1', transform)
trainloader_t2, valloader_t2, len_train_t2, len_val_t2 = create_datasets_and_loaders(train_data, 'Axial T2', transform)
trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir = create_datasets_and_loaders(train_data, 'Sagittal T2/STIR', transform)

dataloaders['Sagittal T1'] = (trainloader_t1, valloader_t1)
dataloaders['Axial T2'] = (trainloader_t2, valloader_t2)
dataloaders['Sagittal T2/STIR'] = (trainloader_t2stir, valloader_t2stir)

lengths['Sagittal T1'] = (len_train_t1, len_val_t1)
lengths['Axial T2'] = (len_train_t2, len_val_t2)
lengths['Sagittal T2/STIR'] = (len_train_t2stir, len_val_t2stir)

# Dictionary mapping labels to indices
label_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}


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

# Plot images
plt.figsize=(8, 4)
plt.imshow(images[0], cmap='gray')
plt.title(label[0])
plt.axis('off')
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
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomResNet50, self).__init__()
        # pretrained=False ile modelin rastgele ağırlıklarla başlatılmasını sağla
        self.model = models.resnet50(weights=None)  # torchvision 0.13'te 'pretrained' yerine 'weights' kullanılmalı
        
        # Eğer manuel ağırlık yolu verilmişse, bu ağırlıkları yükle
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        
        num_ftrs = self.model.fc.in_features  # Son katmanın özellik sayısını al
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Son katmanı değiştir

    def forward(self, x):
        return self.model(x)

    def unfreeze_middle_layers(self):
        """Orta katmanları çöz."""
        # Orta katmanlar: 3. ve 4. blokları çöz
        for name, param in self.model.named_parameters():
            # Bu katmanlar arasında layer3 ve layer4 çözülür
            if 'layer3' in name or 'layer4' in name:  
                param.requires_grad = True  # Orta katmanlar için requires_grad=True
            else:
                param.requires_grad = False  # Diğer tüm katmanları dondur


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


expanded_test_desc.head(5)


levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

# Function to update row_id with levels
def update_row_id(row, levels):
    level = levels[row.name % len(levels)]
    return f"{row['study_id']}_{row['condition']}_{level}"

# Update row_id in expanded_test_desc to include levels
expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)


expanded_test_desc


# Define a custom test dataset class
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        if self.transform:
            image = self.transform(image)
        return image

# Define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create a test dataset and dataloader
test_dataset = TestDataset(expanded_test_desc, transform)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


for image in testloader:
    print(image.shape)
    break


# Define models in a dictionary (use a different name to avoid conflicts)
model_dict = {
    'Sagittal T1': sagittal_t1_model,
    'Axial T2': axial_t2_model,
    'Sagittal T2/STIR': sagittal_t2stir_model,
}

# Function to get the model based on series_description
def get_model(series_description):
    return model_dict.get(series_description, None)

# Function to make predictions on the test data
def predict_test_data(testloader, expanded_test_desc):
    predictions = []
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []
    
    # Set each model to evaluation mode
    for model in model_dict.values():
        model.eval()

    with torch.no_grad():  # Disable gradient calculation during inference
        for idx, images in enumerate(tqdm(testloader)):  # Iterate through the test data
            images = images.to(device)  # Move images to the device
            series_description = expanded_test_desc.iloc[idx]['series_description']  # Get description from DataFrame
            
            # Get the model corresponding to the series description
            model = get_model(series_description)
            
            if model:  # If a valid model is found
                outputs = model(images)  # Forward pass through the model
                probs = torch.softmax(outputs, dim=1).squeeze(0)  # Get the probabilities for each class
                normal_mild_probs.append(probs[0].item())  # Probability for normal/mild class
                moderate_probs.append(probs[1].item())  # Probability for moderate class
                severe_probs.append(probs[2].item())  # Probability for severe class
                predictions.append(probs)  # Append the full prediction
            else:  # If no model is found for the description
                normal_mild_probs.append(None)
                moderate_probs.append(None)
                severe_probs.append(None)
                predictions.append(None)

    return normal_mild_probs, moderate_probs, severe_probs, predictions

# Make predictions on the test data
normal_mild_probs, moderate_probs, severe_probs, test_predictions = predict_test_data(testloader, expanded_test_desc)



test_predictions[0]


# Add predictions and probabilities to the test DataFrame
expanded_test_desc['normal_mild'] = normal_mild_probs
expanded_test_desc['moderate'] = moderate_probs
expanded_test_desc['severe'] = severe_probs


submission = expanded_test_desc[["row_id","normal_mild","moderate","severe"]]


submission.head(10)


# Group by 'row_id' and sum the values
grouped_submission = submission.groupby('row_id').max().reset_index()

# Normalize the columns
grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']].div(grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0)

# Check the first 3 rows
grouped_submission


len(grouped_submission)


sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]


import os

# Save the DataFrame to "submission.csv" in the desired directory
sub.to_csv("/kaggle/working/submission.csv", index=False)


sub.head(5)

