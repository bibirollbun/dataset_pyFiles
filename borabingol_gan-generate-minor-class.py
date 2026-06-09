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


csv_path = '/kaggle/input/cropped-image/cropped_labels.csv'
image_dir = '/kaggle/input/cropped-image/cropped_pngs'

dataframe = pd.read_csv(csv_path)
dataframe['image_path'] = dataframe['filename'].apply(lambda x: os.path.join(image_dir, x))


normal_mild_count = dataframe['severity'].value_counts().get('normal_mild', 0)
print(f"normal_mild sayÄ±sÄ±: {normal_mild_count}")


#Deleting normal_mild data
dataframe = dataframe[dataframe['severity'] != 'normal_mild']
normal_mild_count = dataframe['severity'].value_counts().get('normal_mild', 0)
print(f"normal_mild sayÄ±sÄ±: {normal_mild_count}")


dataframe = dataframe.reset_index(drop=True)
print(dataframe.index)  # train_data'nÄ±n indeksini kontrol et


# read data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')
len(test_desc) #number of test_description.csv rows 


dataframe[dataframe["severity"] == "normal_mild"].value_counts().sum()


dataframe[dataframe["severity"] == "moderate"].value_counts().sum()



dataframe[dataframe["severity"] == "severe"].value_counts().sum()



#changed the name of dataframe to prevent conflict, you have to pay attention of using post underscores
train_df = dataframe


train_df


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


test_data = expanded_test_desc
train_data = train_df
# TÃ¼m veriyi kullanmak iÃ§in bu satÄ±rÄ± yorum satÄ±rÄ± yapabilirsin
#train_data = train_data.sample(n=1000, random_state=42).reset_index(drop=True)  # ğŸ”¹ EÄŸitim sÃ¼resini kÄ±saltmak iÃ§in sadece 1000 Ã¶rnek kullan
###############################################################################################################################################


train_data.head(5)


train_data['series_description'].value_counts()


#Yeni pipiline'dan dolayÄ± artÄ±k load_dicom yerine load_png kullanÄ±yoruz.
def load_png(path):
    # PNG gÃ¶rÃ¼ntÃ¼sÃ¼nÃ¼ oku (gri tonlamalÄ± olarak)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # GÃ¶rÃ¼ntÃ¼yÃ¼ normalize et
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    
    # GÃ¶rÃ¼ntÃ¼yÃ¼ 0-255 aralÄ±ÄŸÄ±na dÃ¶nÃ¼ÅŸtÃ¼r
    image = (image * 255).astype(np.uint8)
    
    return image


import random
import matplotlib.pyplot as plt

# Yeni sÄ±fÄ±rlanmÄ±ÅŸ indekslerle rastgele seÃ§im yapalÄ±m
train_data_reset = train_data.reset_index(drop=True)

# Rastgele iki indeks seÃ§elim
selected_indices = random.sample(range(len(train_data_reset)), 2)

images = []
row_ids = []

# SeÃ§ilen indekslerle gÃ¶rselleri yÃ¼kleyelim
for i in selected_indices:
    image = load_png(train_data_reset['image_path'][i])  # Yeni sÄ±fÄ±rlanmÄ±ÅŸ indeksi kullan
    images.append(image)
    row_ids.append(train_data_reset['row_id'][i])  # Yeni sÄ±fÄ±rlanmÄ±ÅŸ indeksi kullan

# GÃ¶rselleri Ã§izdirelim
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_title(f'Row ID: {row_ids[i]}', fontsize=8)
    ax[i].axis('off')
plt.tight_layout()
plt.show()


train_data 


train_data = train_data.dropna()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import pydicom
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LumbarSpinePNGDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.class_labels = {
            f"{condition}_{severity}": idx
            for idx, (condition, severity) in enumerate(
                sorted(dataframe[['condition', 'severity']].drop_duplicates().values.tolist())
            )
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())  # normalize [0,1]
        image = image * 2 - 1  # normalize [-1,1] for Tanh
        image = np.expand_dims(image, axis=0)

        label_key = f"{row['condition']}_{row['severity']}"
        label = self.class_labels[label_key]

        return torch.tensor(image, dtype=torch.float32), label


# ğŸ”¹ Veri YÃ¼kleme
dataset = LumbarSpinePNGDataset(train_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"Toplam {len(dataset)} adet DICOM gÃ¶rÃ¼ntÃ¼sÃ¼ yÃ¼klendi.")


# ğŸ”¹ Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8 -> 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 16 -> 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 32 -> 64
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 64 -> 128
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



# ğŸ”¹ Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.final_feat_size = None  # doÄŸru girintilendi

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)

        # Lazy init of linear layer (ilk forward'da boyuta gÃ¶re inÅŸa edilir)
        if self.final_feat_size is None:
            self.final_feat_size = out.shape[1]
            self.adv_layer = nn.Sequential(nn.Linear(self.final_feat_size, 1), nn.Sigmoid())
            self.adv_layer.to(out.device)

        validity = self.adv_layer(out)
        return validity



#!rm -rf /kaggle/working/*


print(train_data.index)  # train_data'nÄ±n indeksini kontrol et


import os
import pandas as pd
from torchvision.utils import save_image
from tqdm import tqdm  # <-- tqdm'i import et
import matplotlib.image as mpimg


# ğŸ”¹ Model, Loss, Optimizer
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()
loss_function = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

save_dir = "/kaggle/working/generated_images"
os.makedirs(save_dir, exist_ok=True)

generated_data = []
best_g_loss = float("inf")  # BaÅŸlangÄ±Ã§ta en iyi loss sonsuz

# ğŸ”¹ EÄŸitim
num_epochs = 5
for epoch in range(num_epochs):
    epoch_generated = []
    epoch_g_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")  # tqdm ile ilerleme Ã§ubuÄŸu

    for i, (real_images, labels) in enumerate(progress_bar):
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(real_images)
        real_outputs = torch.sigmoid(real_outputs)  # Sigmoid ekledik
        d_real_loss = loss_function(real_outputs, real_labels)

        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        fake_outputs = torch.sigmoid(fake_outputs)  # Sigmoid ekledik
        d_fake_loss = loss_function(fake_outputs, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Generator
        optimizer_G.zero_grad()
        gen_outputs = discriminator(fake_images)
        gen_outputs = torch.sigmoid(gen_outputs)  # Sigmoid ekledik
        g_loss = loss_function(gen_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()

        # GÃ¶rselleri ve bilgileri kaydet
        for j in range(batch_size):
            filename = f"epoch{epoch+1}_batch{i+1}_img{j+1}.png"
            save_path = os.path.join(save_dir, filename)
            save_image((fake_images[j] + 1) / 2, save_path)

            # Gerekli bilgileri ekle
            condition = list(dataset.class_labels.keys())[labels[j].item()].split('_')[0]  # HastalÄ±k tipi
            severity = list(dataset.class_labels.keys())[labels[j].item()].split('_')[1]  # Severity (seviye)
            epoch_generated.append({
                "filename": filename,
                "row_id": f"{train_data['study_id'][j]}_{condition}_{severity}",
                "condition": condition,
                "severity": severity,
                "image_path": save_path
            })

    avg_g_loss = epoch_g_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {avg_g_loss:.4f}")

    # Sadece en iyi epochâ€™un gÃ¶rsellerini kaydet
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        generated_data = epoch_generated.copy()
        print(f"âœ” Yeni en iyi epoch: {epoch+1} | G Loss: {best_g_loss:.4f}")
    else:
        print(f"â�© Epoch {epoch+1} gÃ¶rselleri atlandÄ± (G Loss daha yÃ¼ksek)")

# ğŸ”¹ CSV oluÅŸtur
generated_df = pd.DataFrame(generated_data)
generated_df.to_csv("/kaggle/working/generated_data.csv", index=False)
print("âœ” En iyi epoch gÃ¶rselleri ve CSV kaydedildi.")

# ğŸ”¹ Ã–rnek gÃ¶rselleri gÃ¶ster (en iyi epoch'tan)
n_samples = 6  # KaÃ§ gÃ¶rsel gÃ¶sterilsin

print(f"\nğŸ“¸ En iyi epochâ€™tan Ã¶rnek {n_samples} gÃ¶rsel gÃ¶steriliyor:")
plt.figure(figsize=(15, 5))
for idx, row in enumerate(generated_data[:n_samples]):
    img_path = os.path.join(save_dir, row["filename"])
    img = mpimg.imread(img_path)
    plt.subplot(1, n_samples, idx + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(row["condition"] + " - " + row["severity"])
plt.tight_layout()
plt.show()



generated_df = generated_df.drop('image_path', axis=1)
generated_df.head()


import pandas as pd
import os

# Condition mapping iÃ§in Ã¶rnek veriler
condition_mapping = {
    'Spinal Canal Stenosis': 'Sagittal T2/STIR',
    'Left Neural Foraminal Narrowing': 'Sagittal T1',
    'Right Neural Foraminal Narrowing': 'Sagittal T1',
    'Left Subarticular Stenosis': 'Axial T2',
    'Right Subarticular Stenosis': 'Axial T2'
}

# `generated_df` Ã¼zerinde iÅŸlem yapÄ±yoruz, Ã§Ã¼nkÃ¼ zaten bu veri Ã¼zerinde deÄŸiÅŸiklik yapmayÄ± hedefliyorsunuz
for index, row in generated_df.iterrows():
    condition = row['condition']
    severity = row['severity']
    
    # severity normalse, "normal_mild" olarak gÃ¼ncelle
    if severity == 'normal':
        severity = 'normal_mild'
    
    # MRI TÃ¼rÃ¼nÃ¼ mapping Ã¼zerinden al
    series_desc = condition_mapping.get(condition, 'Unknown MRI Type')

    # `generated_df` Ã¼zerinde her bir satÄ±rda gÃ¼ncellemeler yapÄ±yoruz
    generated_df.at[index, 'study_id'] = 0  # `study_id`'yi 0 yap
    generated_df.at[index, 'level'] = 0  # `level`'i 0 yap
    generated_df.at[index, 'severity'] = severity  # Yeni severity deÄŸerini at
    generated_df.at[index, 'series_id'] = 0  # `series_id`'yi 0 yap
    generated_df.at[index, 'instance_number'] = 0  # `instance_number`'Ä± 0 yap
    generated_df.at[index, 'crop_x'] = 0  # `crop_x`'Ä± 0 yap
    generated_df.at[index, 'crop_y'] = 0  # `crop_y`'Ä± 0 yap
    generated_df.at[index, 'series_description'] = series_desc  # Yeni `series_description` deÄŸerini at

# SÃ¼tun sÄ±ralamasÄ±nÄ± istediÄŸiniz ÅŸekilde yapalÄ±m
column_order = [
    'row_id', 'study_id', 'condition', 'level', 'severity', 
    'series_id', 'instance_number', 'crop_x', 'crop_y', 
    'filename', 'series_description'
]

generated_df = generated_df[column_order]

# GÃ¼ncellenmiÅŸ `generated_df`'yi kaydet
generated_df.to_csv("/kaggle/working/generated.csv", index=False)

print("âœ” Yeni veriler baÅŸarÄ±yla kaydedildi.")



#!rm -rf /kaggle/working/*


train_data['level'].unique()

