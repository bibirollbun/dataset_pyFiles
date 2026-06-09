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


import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2



# read data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')
len(test_desc) #number of test_description.csv rows 


print(train_desc.columns)



def generate_image_paths_with_naming(df, data_dir):
    image_paths = []
    for index, row in df.iterrows():
        study_id = str(row['study_id'])
        condition = row['series_description'].replace(' ', '_').lower()  # Series description bilgisi kullanılabilir
        level = row['series_description'].split('/')[-1]  # Seviyeyi serinin adının son kısmından alabiliriz
        instance_number = str(row['instance_number'])

        # Klasör yapısını oluştur
        study_dir = os.path.join(data_dir, study_id)
        series_dir = os.path.join(study_dir, row['series_id'], instance_number + '.dcm')

        if os.path.exists(series_dir):
            image_paths.append(series_dir)

    return image_paths



# Veri ön işleme işlemleri
def load_dicom_images(image_paths, target_size=(128, 128)):
    images = []
    for path in image_paths:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        img = cv2.resize(img, target_size)  # Görselleri boyutlandır
        img = img / np.max(img)  # Normalize et
        images.append(img)
    return np.array(images, dtype=np.float32).reshape(-1, target_size[0], target_size[1], 1)

# Eğitim ve test görüntü yollarını al
train_image_paths = generate_image_paths(train_desc, f'{train_path}/train_images')
test_image_paths = generate_image_paths(test_desc, f'{train_path}/test_images')

# Eğitim ve test görüntülerini yükle
train_images = load_dicom_images(train_image_paths)
test_images = load_dicom_images(test_image_paths)

# Etiketleri encode et
encoder = LabelEncoder()
labels = final_merged_df['severity'].values
labels_encoded = encoder.fit_transform(labels)

# Eğitim ve doğrulama verilerini ayır
X_train, X_val, y_train, y_val = train_test_split(train_images, labels_encoded, test_size=0.2, random_state=42)



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


# import pandas as pd

# # En düşük sınıf sayısını belirleyelim
# min_class_count = 3081

# # Normal/Mild ve Moderate sınıflarını azaltalım
# normal_mild_df = final_merged_df[final_merged_df["severity"] == "Normal/Mild"].sample(n=min_class_count, random_state=42)
# moderate_df = final_merged_df[final_merged_df["severity"] == "Moderate"].sample(n=min_class_count, random_state=42)
# severe_df = final_merged_df[final_merged_df["severity"] == "Severe"]

# # İndeksleri sıfırlayalım
# normal_mild_df = normal_mild_df.reset_index(drop=True)
# moderate_df = moderate_df.reset_index(drop=True)
# severe_df = severe_df.reset_index(drop=True)

# # Verileri birleştirelim ve final_merged_df'yi güncelleyelim
# final_merged_df = pd.concat([normal_mild_df, moderate_df, severe_df])

# # Sonuçları kontrol edelim
# print(final_merged_df["severity"].value_counts())



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
import numpy as np
from PIL import Image
import pydicom  # Assuming you're working with DICOM files

# Define a function to load DICOM images (if not already done)
def load_dicom(image_path):
    # Load the DICOM file (you can use pydicom or another library for this)
    dicom = pydicom.dcmread(image_path)
    
    # Convert the pixel data to a numpy array, then to a PIL image
    image = dicom.pixel_array.astype(np.uint8)  # Assuming the image is in pixel_array
    pil_image = Image.fromarray(image)  # Convert numpy array to PIL image
    return pil_image

# Update the CustomDataset to ensure image is in PIL format and prepared for CapsNet
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Load image using load_dicom
        
        label = self.dataframe['severity'][index]  # Numeric label from the dataframe
        
        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label

# Function to create datasets and dataloaders for each series description
def create_datasets_and_loaders(df, series_description, transform, batch_size=8):
    filtered_df = df[df['series_description'] == series_description]
    
    # %5'ini al frac değerini değiştirerek trainde verinin ne kadarını kullanacağınızı belirleyebilirsiniz
    filtered_df = filtered_df.sample(frac=1.0, random_state=42)  
    
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(val_df, transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, len(train_df), len(val_df)


# Define the transforms for data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CapsNet için uygun boyutta yeniden boyutlandır
    transforms.Grayscale(num_output_channels=3),  # CapsNet RGB ile çalışır, tek kanal gri tonlama görüntüsünü 3 kanala dönüştür
    transforms.ToTensor(),  # PIL -> Tensor dönüşümü, [0, 1] aralığı
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizasyon (opsiyonel)
])

# Create dataloaders for each series description
trainloader_t1, valloader_t1, len_train_t1, len_val_t1 = create_datasets_and_loaders(train_data, 'Sagittal T1', transform)
trainloader_t2, valloader_t2, len_train_t2, len_val_t2 = create_datasets_and_loaders(train_data, 'Axial T2', transform)
trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir = create_datasets_and_loaders(train_data, 'Sagittal T2/STIR', transform)

# Store dataloaders and lengths in dictionaries
dataloaders = {
    'Sagittal T1': (trainloader_t1, valloader_t1),
    'Axial T2': (trainloader_t2, valloader_t2),
    'Sagittal T2/STIR': (trainloader_t2stir, valloader_t2stir)
}

lengths = {
    'Sagittal T1': (len_train_t1, len_val_t1),
    'Axial T2': (len_train_t2, len_val_t2),
    'Sagittal T2/STIR': (len_train_t2stir, len_val_t2stir)
}

# Label mapping for severity
label_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
train_data['severity'] = train_data['severity'].map(label_map)



def visualize_batch(dataloader):
    images, labels = next(iter(dataloader))
    
    # Plot a grid of images (rows = batch size, columns = number of images)
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    
    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i]
        
        img = img.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        
        # Normalize back to [0, 1] for better visualization (assuming Normalize was used)
        img = (img - img.min()) / (img.max() - img.min())
        
        ax.imshow(img)  # Show image in RGB
        ax.set_title(f"Label: {lbl.item()}")  # Convert tensor label to a regular Python number
        ax.axis('off')  # Hide axes

    plt.show()

# Visualize samples from each dataloader
print("Visualizing Sagittal T1 samples")
visualize_batch(trainloader_t1)
print("Visualizing Axial T2 samples")
visualize_batch(trainloader_t2)
print("Visualizing Sagittal T2/STIR samples")
visualize_batch(trainloader_t2stir)



import matplotlib.pyplot as plt

# Bir batch'ten bir örnek alın
images, labels = next(iter(trainloader_t2))
image = images[0]  # İlk görüntüyü seç
label = labels[0]  # İlk görüntünün etiketini seç

# Görüntü boyutlarını HWC formatına dönüştür
image = image.permute(1, 2, 0)  # CHW'den HWC'ye dönüştür

# Görüntüyü çiz
plt.figure(figsize=(8, 4))
plt.imshow(image, cmap='gray')  # Gri tonlamada gösterim
plt.title(f"Label: {label.item()}")  # Etiketi başlık olarak ekle
plt.axis('off')
plt.tight_layout()
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.out_channels = out_channels

        # Define route weights
        self.route_weights = nn.Parameter(
            torch.randn(num_capsules, num_routes, in_channels, out_channels)
        )

    def forward(self, x):
        # x: [batch_size, num_routes, in_channels]
        x = x.unsqueeze(1).unsqueeze(4)  # [batch_size, 1, num_routes, in_channels, 1]

        # Fix: Match dimensions for matmul
        # Use reshape instead of view to ensure correct memory layout
        x = x.reshape(x.size(0), -1, x.size(3))  # [batch_size, num_routes, in_channels]
        
        priors = torch.matmul(self.route_weights, x)  # [num_capsules, num_routes, out_channels]
        
        logits = torch.zeros(*priors.size()).to(x.device)  # [num_capsules, num_routes, out_channels]
        return self.routing(logits, priors)

    def routing(self, logits, priors, iterations=3):
        for i in range(iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
            if i < iterations - 1:
                logits = logits + torch.matmul(priors.transpose(2, 3), outputs)
        return outputs.squeeze(3)

    def squash(self, inputs):
        norm = (inputs ** 2).sum(dim=-1, keepdim=True)
        scale = norm / (1 + norm)
        return scale * inputs / torch.sqrt(norm + 1e-8)


# CapsNet Ana Model
class CapsNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=9, stride=1)
        self.primary_capsules = nn.Conv2d(256, 8*32, kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_routes=32*6*6, in_channels=8, out_channels=16)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)  # İlk Conv katmanı
        x = self.primary_capsules(x)  # [batch_size, 8*32, H, W]
        x = x.view(x.size(0), 32, 8, -1).permute(0, 1, 3, 2)  # [batch_size, 32, H*W, 8]
        x = self.digit_capsules(x)  # [batch_size, num_classes, out_channels]
        return x.norm(dim=-1)  # Vektör normları

# Visualization function for Capsule activations
def visualize_capsules(capsule_activations, num_capsules, title="Capsule Activations"):
    """
    Visualize the activations of capsule networks. 
    Assuming that the capsule activations are 2D or 1D arrays
    """

    fig, axes = plt.subplots(1, num_capsules, figsize=(20, 5))
    for i in range(num_capsules):
        ax = axes[i]
        ax.imshow(capsule_activations[i].cpu().detach().numpy(), cmap='viridis')  # Visualize activations
        ax.set_title(f"Capsule {i+1}")
        ax.axis('off')  # Hide axes
    plt.suptitle(title, fontsize=16)
    plt.show()

# Initialize CapsNet Model
input_dim = 1  # Grayscale images
num_classes = 10  # Number of output capsules (digits, e.g., 0-9 for MNIST)
model = CapsNet(input_dim, num_classes)

# Example forward pass
# Let's assume we're using a batch of images of shape (batch_size, 1, 28, 28)
dummy_input = torch.randn(8, 1, 28, 28)  # Example batch size of 8
output = model(dummy_input)  # Forward pass through CapsNet

# Visualize the capsule activations for the first sample in the batch
# Output is a tensor of shape [batch_size, num_classes, out_channels]
sample_activations = output[0]  # Take the first sample's output

# Visualize activations for all capsules
visualize_capsules(sample_activations, num_classes, title="Capsule Activations for Sample 1")




class MarginLoss(nn.Module):
    def forward(self, labels, predictions):
        left = F.relu(0.9 - predictions, inplace=True) ** 2
        right = F.relu(predictions - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1.0 - labels) * right
        return loss.sum(dim=1).mean()


# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, kayıp ve optimizasyon
model = CapsNet(input_dim=1, num_classes=3).to(device)
criterion = MarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch in trainloader:
        # Eğer batch bir liste (list) türünde ise
        if isinstance(batch, list):
            images, labels = batch
        # Eğer batch bir tuple ise (images, labels) ayıralım
        elif isinstance(batch, tuple):
            images, labels = batch
        # Eğer batch bir dictionary ise
        elif isinstance(batch, dict):
            images = batch['image']  # 'image' anahtarını kullanarak
            labels = batch['label']  # 'label' anahtarını kullanarak
        else:
            raise ValueError(f"Beklenmeyen veri formatı: {type(batch)}")

        # Verileri cihaza taşıyoruz
        if isinstance(images, torch.Tensor):
            images = images.to(device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)

        # Model tahmini
        outputs = model(images)

        # Etiketleri one-hot encoding yapıyoruz
        one_hot_labels = F.one_hot(labels, num_classes=3).float()

        # Kayıp hesaplama
        loss = criterion(one_hot_labels, outputs)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in valloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {100 * correct / total:.2f}%")


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

