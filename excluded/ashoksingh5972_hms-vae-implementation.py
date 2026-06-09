# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install efficientnet_pytorch


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time
import cv2
warnings.filterwarnings('ignore')


start_time = time.time()
BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"


brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}


df = pd.read_csv(f"{BASE_DIR}train.csv")

df_toy = df.sample(frac=0.2, random_state=42)
# Split 80% Train, 20% Temp (Validation + Test)
train_df, temp_df = train_test_split(df_toy, test_size=0.4, random_state=42)

# Split 10% Validation, 10% Test from Temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Splitting done! Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))


class ChunkedBrainActivityDataset(Dataset):
    def __init__(self, csv_file, base_dir, activity_mapping,md):
        self.df = csv_file
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.resize_transform = transforms.Resize((224, 224))
        self.md = md

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spect_id, label, offset = self.df.iloc[idx][["spectrogram_id", "expert_consensus", "spectrogram_label_offset_seconds"]]

        temp_df = pd.read_parquet(f'{self.base_dir}/train_spectrograms/{spect_id}.parquet')
        temp_df.drop(['time'], axis=1, inplace=True)

        start = int(offset) // 2
        temp_df = temp_df[start:start+300]
        temp_df = np.log1p(temp_df)
        temp_df /= temp_df.max()
        temp_arr = np.nan_to_num(temp_df.to_numpy(), nan=1e-4)

        # Use OpenCV to apply a colormap and convert to RGB
        temp_arr_uint8 = np.uint8(255 * temp_arr)
        rgb_image = cv2.applyColorMap(temp_arr_uint8, cv2.COLORMAP_JET)

        # Normalize to [0, 1] and convert to tensor
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image_tensor = torch.tensor(rgb_image).permute(2, 0, 1)  # (C, H, W)
        rgb_image_tensor = self.resize_transform(rgb_image_tensor)
            
        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()
        
        return rgb_image_tensor, y_tensor


# Now create DataLoader with the chunked dataset
# chunk_size = 1000  # Adjust chunk size according to memory constraints

train_dataset = ChunkedBrainActivityDataset(csv_file=train_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")
val_dataset = ChunkedBrainActivityDataset(csv_file=val_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")
test_dataset = ChunkedBrainActivityDataset(csv_file=test_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers= 2, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers= 2, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers= 2, pin_memory=True, prefetch_factor=2)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (224,224) → (112,112)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (112,112) → (56,56)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (56,56) → (28,28)
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 28 * 28, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(128 * 28 * 28, latent_dim)  # Log-variance
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 28 * 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (28,28) → (56,56)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (56,56) → (112,112)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # (112,112) → (224,224)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        x = self.decoder_input(z).view(-1, 128, 28, 28)
        recon_x = self.decoder(x)
        
        return recon_x, mu, logvar



# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss: reconstruction loss + KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, logvar = vae(images)
        loss = vae_loss(recon_x, images, mu, logvar)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}")

# Save trained VAE model
torch.save(vae.state_dict(), "/kaggle/working/vae_model.pth")
print("VAE Model Saved!")



# Load trained VAE model
vae.load_state_dict(torch.load("/kaggle/working/vae_model.pth", map_location=device))
vae.eval()

# Function to get encoded features
def extract_latent_features(dataloader, model, device):
    model.to(device)
    features, labels = [], []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            targets = targets.cpu().numpy()

            _, mu, _ = model(images)  # Extract mean vector as features
            encoded_features = mu.cpu().numpy()

            features.append(encoded_features)
            labels.append(targets)

    features = np.vstack(features)
    labels = np.vstack(labels)
    return features, labels

# Extract features for train, validation, and test sets
X_train, y_train = extract_latent_features(train_loader, vae, device)
X_val, y_val = extract_latent_features(val_loader, vae, device)
X_test, y_test = extract_latent_features(test_loader, vae, device)

# Convert one-hot labels to class indices
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)

# Save extracted features
np.save("vae_train_features.npy", X_train)
np.save("vae_train_labels.npy", y_train)
np.save("vae_val_features.npy", X_val)
np.save("vae_val_labels.npy", y_val)
np.save("vae_test_features.npy", X_test)
np.save("vae_test_labels.npy", y_test)
print("Feature Extraction Done!")



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load extracted features
X_train = np.load("vae_train_features.npy")
y_train = np.load("vae_train_labels.npy")
X_val = np.load("vae_val_features.npy")
y_val = np.load("vae_val_labels.npy")

X_test = np.load("vae_test_features.npy")
y_test = np.load("vae_test_labels.npy")

# Train logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = lr_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=brain_activities))

# Evaluate on test set
y_test_pred = lr_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=brain_activities))





