import os
import random
import numpy as np
import pandas as pd
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

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')

df_train = pd.read_csv(train_csv_path)
df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))
df_test = pd.read_csv(test_csv_path)
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

# Échantillonnage équilibré
real_samples = df_train[df_train['label'] == 0].sample(n=25000, random_state=42)
fake_samples = df_train[df_train['label'] == 1].sample(n=25000, random_state=42)
df_train = pd.concat([real_samples, fake_samples]).sample(frac=1, random_state=42)

train_data = df_train.sample(frac=0.9, random_state=42)
val_data = df_train.drop(train_data.index)

# --- DATASET ---
class FireDataset(Dataset):
    def __init__(self, paths, labels, img_size=256, train=True):
        self.paths = paths
        self.labels = labels
        self.train = train
        self.img_size = img_size
        # Correction de CoarseDropout pour Albumentations
        self.strong_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.CoarseDropout(p=0.3)  # Simplification des paramètres
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

# --- MASQUE FRÉQUENTIEL ---
def create_fixed_mask(size=256):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = size // 2
    dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = ((dist >= 40) & (dist <= 120)).float()  # Bande 40-120 pixels
    mask_c = 1.0 - mask
    return mask.unsqueeze(0).unsqueeze(0), mask_c.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

# --- MODULE FMRE ---
class FMRE(nn.Module):
    def __init__(self, init_mask, init_mask_c):
        super().__init__()
        self.register_buffer('M_mid', init_mask)
        self.register_buffer('M_mid_c', init_mask_c)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder_mid = self._build_decoder(self.M_mid)
        self.decoder_mid_c = self._build_decoder(self.M_mid_c)

    def _build_decoder(self, target_mask):
        decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 1)
        )
        with torch.no_grad():
            decoder[-1].weight.data = target_mask.mean() * torch.ones_like(decoder[-1].weight)
            decoder[-1].bias.data.zero_()
        return decoder

    def forward(self, x_fft):
        encoded = self.encoder(x_fft)
        m_mid = torch.sigmoid(self.decoder_mid(encoded))
        m_mid_c = torch.sigmoid(self.decoder_mid_c(encoded))
        # Redimensionnement si besoin
        if m_mid.shape[-1] != x_fft.shape[-1]:
            m_mid = F.interpolate(m_mid, size=x_fft.shape[-2:], mode='bilinear', align_corners=False)
            m_mid_c = F.interpolate(m_mid_c, size=x_fft.shape[-2:], mode='bilinear', align_corners=False)
        return m_mid, m_mid_c

# --- MODÈLE FIRE ---
class FIRE(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        # Initialisation des masques fixes 40-120px
        M_mid, M_mid_c = create_fixed_mask(img_size)
        self.fmre = FMRE(M_mid, M_mid_c)
        
        # VAE du modèle de diffusion Stable Diffusion
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            use_safetensors=True
        ).to(device)
        self.vae.enable_tiling(False)
        self.vae.requires_grad_(False)
        
        # Classifieur léger
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
        if mask.size(1) == 1 and x.size(1) == 3:
            mask = mask.repeat(1, 3, 1, 1)
        freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        masked_freq = freq * mask
        return torch.fft.ifft2(torch.fft.ifftshift(masked_freq, dim=(-2, -1)), dim=(-2, -1)).real

    def forward(self, x):
        # Conversion en niveaux de gris pour l'analyse fréquentielle
        gray_x = torch.mean(x, dim=1, keepdim=True)
        # FFT et masquage
        x_fft = torch.fft.fftshift(torch.fft.fft2(gray_x, dim=(-2, -1)), dim=(-2, -1)).abs().log()
        m_mid, m_mid_c = self.fmre(x_fft)
        # Application du masque complémentaire (x' = image sans fréquences moyennes)
        x_pseudo = self.apply_frequency_mask(x, m_mid_c)
        
        # Reconstruction par VAE - DEUX APPELS AU MÊME VAE
        # 1. Pour l'image originale x
        # 2. Pour l'image filtrée x' (x_pseudo)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Encoder et Decoder séparés pour plus de clarté
            latent_x = self.vae.encode(x).latent_dist.sample()
            latent_pseudo = self.vae.encode(x_pseudo).latent_dist.sample()
            recon_x = self.vae.decode(latent_x).sample.float()
            recon_pseudo = self.vae.decode(latent_pseudo).sample.float()
        
        # Calcul des erreurs de reconstruction
        delta_x = (recon_x - x).abs()
        delta_pseudo = (recon_pseudo - x_pseudo).abs()
        
        # Classification basée sur les deux erreurs
        return self.classifier(torch.cat([delta_x, delta_pseudo], dim=1))

    def compute_loss(self, x, y):
        batch_size = x.size(0)
        gray_x = torch.mean(x, dim=1, keepdim=True)
        x_fft = torch.fft.fftshift(torch.fft.fft2(gray_x, dim=(-2, -1)), dim=(-2, -1)).abs().log()
        m_mid, m_mid_c = self.fmre(x_fft)
        x_mid = self.apply_frequency_mask(x, m_mid)
        x_pseudo = self.apply_frequency_mask(x, m_mid_c)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            latent_x = self.vae.encode(x).latent_dist.sample()
            latent_pseudo = self.vae.encode(x_pseudo).latent_dist.sample()
            recon_x = self.vae.decode(latent_x).sample.float()
            recon_pseudo = self.vae.decode(latent_pseudo).sample.float()
        
        delta_x = (recon_x - x).abs()
        delta_pseudo = (recon_pseudo - x_pseudo).abs()
        
        # 1. L_mid_rec: alignement mid-freq avec erreur
        L_mid_rec = F.mse_loss(x_mid, delta_x)
        
        # 2. L_mask: guidage vers masques prédéfinis (batch-aware)
        M_mid = self.fmre.M_mid.expand(batch_size, -1, -1, -1)
        M_mid_c = self.fmre.M_mid_c.expand(batch_size, -1, -1, -1)
        L_mask = (
            F.mse_loss(m_mid, M_mid) +
            F.mse_loss(m_mid_c, M_mid_c) +
            F.mse_loss(1.0 - m_mid - m_mid_c, torch.zeros_like(m_mid))
        )
        
        # 3. L_ce: perte de classification
        output = self.classifier(torch.cat([delta_x, delta_pseudo], dim=1)).squeeze()
        L_ce = F.binary_cross_entropy_with_logits(output, y)
        
        # Combinaison pondérée des pertes
        total_loss = 0.2 * L_mid_rec + 0.2 * L_mask + 0.6 * L_ce
        
        # Correction: Retourner None au lieu d'un nouveau tenseur
        if torch.isnan(total_loss):
            print("NaN detected! Skipping this batch...")
            return None, {
                "L_mid_rec": 0.0,
                "L_mask": 0.0,
                "L_ce": 0.0
            }
        
        return total_loss, {
            "L_mid_rec": L_mid_rec.item(),
            "L_mask": L_mask.item(),
            "L_ce": L_ce.item()
        }

# --- ENTRAÎNEMENT ---
def train_kaggle():
    train_dataset = FireDataset(train_data['file_name'].values, train_data['label'].values)
    val_dataset = FireDataset(val_data['file_name'].values, val_data['label'].values, train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    model = FIRE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler()

    train_losses = []
    val_aucs = []
    component_losses = {"L_mid_rec": [], "L_mask": [], "L_ce": []}

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        epoch_components = {"L_mid_rec": 0.0, "L_mask": 0.0, "L_ce": 0.0}
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):
            if torch.isnan(imgs).any():
                print(f"NaN in batch {batch_idx}, skipping...")
                continue
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss, components = model.compute_loss(imgs, labels)
                
            # Correction: Vérifier si la loss est None
            if loss is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if (batch_idx + 1) % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                running_loss += loss.item()
                for k in components:
                    epoch_components[k] += components[k]
            else:
                print(f"Skipping batch {batch_idx} due to NaN loss")
                
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        for k in epoch_components:
            component_losses[k].append(epoch_components[k] / max(1, len(train_loader)))
            print(f"Epoch {epoch+1} | {k}: {component_losses[k][-1]:.4f}")
            
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs.to(device)).squeeze()
                outputs = torch.nan_to_num(outputs, nan=0.5)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.numpy())
        try:
            auc = roc_auc_score(val_labels, val_preds)
        except:
            auc = 0.5
        val_aucs.append(auc)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUC: {auc:.4f}")
        
    # Visualisation
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return model

# --- SOUMISSION ---
def generate_submission(model):
    test_dataset = FireDataset(df_test['id'].values, [0]*len(df_test), train=False)
    test_loader = DataLoader(test_dataset, batch_size=4)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader):
            outputs = model(imgs.to(device)).squeeze()
            outputs = torch.nan_to_num(outputs, nan=0.5)
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    binary_labels = (np.array(predictions) > 0.5).astype(int)
    submission = pd.DataFrame({
        'id': ['test_data_v2/' + os.path.basename(p) for p in df_test['id'].values],
        'label': binary_labels
    })
    submission.to_csv('submission.csv', index=False)
    return submission

if __name__ == "__main__":
    trained_model = train_kaggle()
    submission_df = generate_submission(trained_model)
    print(submission_df.head())
    torch.save(trained_model.state_dict(), 'model.pth')


