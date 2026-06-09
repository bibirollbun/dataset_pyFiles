#%% [markdown]
# ## Configuration Kaggle
# Imports et configuration de base pour Kaggle
#%%
import os
import pickle
import time
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL

# Configuration matériel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% [markdown]
# ## Préparation des données (Spécifique Kaggle)
#%%
base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')

df_train = pd.read_csv(train_csv_path)
df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

df_test = pd.read_csv(test_csv_path)
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

# Réduction à 5000 images BALANCÉES (2500 réelles + 2500 générées)
real_samples = df_train[df_train['label'] == 0].sample(n=5500, random_state=42)
fake_samples = df_train[df_train['label'] == 1].sample(n=5500, random_state=42)
df_train = pd.concat([real_samples, fake_samples]).sample(frac=1, random_state=42)  # Mélange

# Split train/val MAINTAINING BALANCE
train_data = df_train.sample(frac=0.9, random_state=42)
val_data = df_train.drop(train_data.index)

#%% [markdown]
# ## Dataset et Augmentations (Optimisé)
#%%
class FireDataset(Dataset):
    def __init__(self, paths, labels, img_size=256, train=True):
        self.paths = paths
        self.labels = labels
        self.train = train
        self.img_size = img_size
        
        self.strong_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.train and random.random() < 0.7:
            img = np.array(img)
            img = self.strong_aug(image=img)['image']
            img = Image.fromarray(img)
        img = img.resize((self.img_size, self.img_size))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

#%% [markdown]
# ## Architecture FIRE (Conforme au papier)
#%%
class FMRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder_mid = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 1)
        )
        self.decoder_mid_c = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x_fft):
        encoded = self.encoder(x_fft)
        m_mid = torch.sigmoid(self.decoder_mid(encoded))
        m_mid_c = torch.sigmoid(self.decoder_mid_c(encoded))
        return m_mid, m_mid_c

class FIRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmre = FMRE()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        self.vae.enable_tiling(False)
        self.vae.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Conv2d(6, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def apply_frequency_mask(self, x, mask):
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        masked_freq = freq * mask
        return torch.fft.ifft2(torch.fft.ifftshift(masked_freq)).real

    def forward(self, x):
        gray_x = torch.mean(x, dim=1, keepdim=True)
        x_fft = torch.fft.fftshift(torch.fft.fft2(gray_x)).abs().log()
        m_mid, m_mid_c = self.fmre(x_fft)
        x_pseudo = self.apply_frequency_mask(x, m_mid_c)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            recon = self.vae(x)[0].float()
            pseudo_recon = self.vae(x_pseudo)[0].float()
        delta_x = (recon - x).abs().detach()
        delta_pseudo = (pseudo_recon - x_pseudo).abs().detach()
        return self.classifier(torch.cat([delta_x, delta_pseudo], dim=1))

#%% [markdown]
# ## Entraînement Kaggle (Optimisé mémoire) + Courbes d'apprentissage
#%%
def train_kaggle():
    train_dataset = FireDataset(train_data['file_name'].values, train_data['label'].values)
    val_dataset = FireDataset(val_data['file_name'].values, val_data['label'].values, train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    model = FIRE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_aucs = []

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs).squeeze()
                loss = F.binary_cross_entropy_with_logits(outputs, labels)
            scaler.scale(loss).backward()
            if (batch_idx + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs.to(device)).squeeze()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.numpy())
        auc = roc_auc_score(val_labels, val_preds)
        val_aucs.append(auc)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUC: {auc:.4f}")

    # Affichage des courbes d'apprentissage
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe de loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Courbe de validation (AUC)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

#%% [markdown]
# ## Génération des soumissions
#%%
def generate_submission(model):
    test_dataset = FireDataset(df_test['id'].values, [0]*len(df_test), train=False)
    test_loader = DataLoader(test_dataset, batch_size=4)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader):
            outputs = model(imgs.to(device)).squeeze()
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    binary_labels = (np.array(predictions) > 0.5).astype(int)
    submission = pd.DataFrame({
        'id': ['test_data_v2/' + os.path.basename(p) for p in df_test['id'].values],
        'label': binary_labels
    })
    submission.to_csv('submission.csv', index=False)
    return submission

#%% [markdown]
# ## Workflow Kaggle
#%%
if __name__ == "__main__":
    trained_model = train_kaggle()
    submission_df = generate_submission(trained_model)
    print(submission_df.head())
    torch.save(trained_model.state_dict(),'model.pth')


