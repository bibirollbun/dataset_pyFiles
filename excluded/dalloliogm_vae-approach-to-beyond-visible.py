from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
import torch

class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = glob(os.path.join(root_dir, "*", "*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        image = np.load(file_path).astype(np.float32)  # shape (H, W, C)
        image = image / (image.max() + 1e-8)

        label_str = os.path.basename(os.path.dirname(file_path))
        label = int(label_str)
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)

        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, img_channels=250, condition_dim=10, latent_dim=128, hidden_dims=None):
        super(CVAE, self).__init__()
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.condition_embed = nn.Linear(condition_dim, 64)

        # Encoder
        encoder_layers = []
        in_channels = img_channels + 1  # 1 for condition broadcast
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(hidden_dims[-1]*8*8, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*8*8, latent_dim)


        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 64, hidden_dims[-1] * 4 * 4)

        hidden_dims.reverse()
        decoder_layers = []
        
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                                     kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
        
        # 32 → 16 → 125
        decoder_layers.append(nn.ConvTranspose2d(hidden_dims[-1], 64, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

#
    def encode(self, x, c):
        B, _, H, W = x.shape
        c_broadcast = c.argmax(dim=1).view(B, 1, 1, 1).float().expand(-1, 1, H, W)
        x_cond = torch.cat([x, c_broadcast], dim=1)
        x_enc = self.encoder(x_cond)
        x_flat = self.flatten(x_enc)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c_embed = self.condition_embed(c)
        zc = torch.cat([z, c_embed], dim=1)
        x = self.decoder_input(zc)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar



def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld_loss



from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def train_cvae(model, dataloader, device, num_epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            c = F.one_hot(labels, num_classes=10).float().to(device)

            recon, mu, logvar = model(x, c)
            loss = vae_loss(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
     


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HyperspectralDataset(root_dir="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025p2/Train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

model = CVAE(img_channels=125)
train_cvae(model, dataloader, device, num_epochs=10)



def generate_samples(model, disease_level, num_samples=50, latent_dim=128, device='cpu'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.full((num_samples,), disease_level, dtype=torch.long).to(device)
        c = F.one_hot(labels, num_classes=10).float()
        samples = model.decode(z, c)
        return samples.cpu().numpy()  # (B, C, H, W)



import torch

# Normalized SRFs for 125 bands
SRF_GREEN = torch.tensor([
    0.0000,0.0000,0.0000,0.0000,0.0001,0.0002,0.0005,0.0008,0.0014,0.0024,0.0041,
    0.0069,0.0113,0.0180,0.0279,0.0414,0.0583,0.0783,0.1008,0.1252,0.1507,0.1766,
    0.2023,0.2271,0.2505,0.2721,0.2913,0.3079,0.3216,0.3324,0.3404,0.3459,0.3495,
    0.3516,0.3528,0.3533,0.3535,0.3536,0.3538,0.3539,0.3541,0.3542,0.3542,0.3541,
    0.3535,0.3520,0.3491,0.3443,0.3373,0.3277,0.3152,0.2997,0.2811,0.2595,0.2349,
    0.2076,0.1778,0.1462,0.1140,0.0823,0.0524,0.0259,0.0037,0.0003,0.0000,0.0000,
    0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
]).float()
SRF_RED = torch.tensor([
    0.0000,0.0000,0.0000,0.0000,0.0001,0.0002,0.0003,0.0006,0.0012,0.0024,0.0047,
    0.0087,0.0154,0.0255,0.0395,0.0575,0.0786,0.1020,0.1265,0.1505,0.1732,0.1940,
    0.2121,0.2269,0.2381,0.2454,0.2491,0.2494,0.2466,0.2409,0.2326,0.2219,0.2093,
    0.1952,0.1799,0.1639,0.1476,0.1314,0.1157,0.1008,0.0870,0.0744,0.0629,0.0525,
    0.0430,0.0344,0.0266,0.0195,0.0129,0.0070,0.0018,0.0003,0.0000,0.0000,0.0000,
    0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
]).float()
SRF_NIR = torch.tensor([
    0.0000,0.0000,0.0000,0.0000,0.0000,0.0001,0.0002,0.0003,0.0006,0.0011,0.0022,
    0.0041,0.0073,0.0125,0.0204,0.0317,0.0470,0.0666,0.0905,0.1185,0.1500,0.1841,
    0.2196,0.2554,0.2900,0.3219,0.3495,0.3715,0.3870,0.3950,0.3950,0.3872,0.3721,
    0.3503,0.3228,0.2912,0.2573,0.2228,0.1888,0.1563,0.1261,0.0990,0.0755,0.0557,
    0.0395,0.0265,0.0162,0.0082,0.0023,0.0003,0.0000,0.0000,0.0000,0.0000,0.0000,
    0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
]).float()

# Normalize SRFs
SRF_TABLE = {
    'green': SRF_GREEN / SRF_GREEN.sum(),
    'red'  : SRF_RED / SRF_RED.sum(),
    'nir'  : SRF_NIR / SRF_NIR.sum(),
}


import torch.nn.functional as F

def interpolate_srf(srf, target_len):
    srf = srf.view(1, 1, -1)  # (1, 1, original_len)
    srf = F.interpolate(srf, size=target_len, mode='linear', align_corners=False)
    return srf.view(-1)

def hs_to_rgb(hs_img: torch.Tensor) -> torch.Tensor:
    """
    Project a hyperspectral image (C, H, W) to RGB using interpolated SRFs.
    Returns an RGB image of shape (3, H, W)
    """
    assert hs_img.dim() == 3, "Expected (C, H, W)"
    C, H, W = hs_img.shape
    device = hs_img.device

    rgb = []
    for band, srf in zip(['green', 'red', 'nir'], [SRF_GREEN, SRF_RED, SRF_NIR]):
        w = interpolate_srf(srf, C).to(device)
        w = w / (w.sum() + 1e-8)  # normalize
        w = w.view(C, 1, 1)
        channel = (hs_img * w).sum(0)
        rgb.append(channel)

    return torch.stack(rgb)



from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn

class InceptionPool3(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True, transform_input=False).to(device)
        net.eval()

        net.AuxLogits = nn.Identity()  # Remove aux
        self.stem_and_blocks = nn.Sequential(*list(net.children())[:-2])  # Cut off classifier

    def forward(self, x):  # x: (B, 3, H, W)
        with torch.no_grad():
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = self.stem_and_blocks(x)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            return x.view(x.size(0), -1)



from scipy.linalg import sqrtm
import numpy as np

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet Inception Distance."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ 
                        (sigma2 + eps * np.eye(sigma2.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)



import os
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

def generate_submission(model, fid_model, real_eval_dir, output_dir, device, num_levels=10, num_samples=50):
    model.eval()
    fid_model.eval()
    latent_dim = model.latent_dim
    fid_scores = []

    os.makedirs(output_dir, exist_ok=True)

    for level in tqdm(range(num_levels), desc="Generating submission"):
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.full((num_samples,), level, dtype=torch.long).to(device)
        c = F.one_hot(labels, num_classes=num_levels).float().to(device)

        with torch.no_grad():
            fake_hsi = model.decode(z, c).cpu()

        # Save generated .npy files (optional)
        for i in range(num_samples):
            out_path = os.path.join(output_dir, f"gen_{level}_{i}.npy")
            np.save(out_path, fake_hsi[i].numpy())

        # Convert generated HSI to RGB
        fake_rgb = torch.stack([hs_to_rgb(img) for img in fake_hsi]).to(device)

        # Load and convert real HSI to RGB
        real_rgb = []
        real_folder = os.path.join(real_eval_dir, str(level))
        for f in sorted(os.listdir(real_folder))[:num_samples]:
            real = torch.tensor(np.load(os.path.join(real_folder, f))).float()
            real_rgb.append(hs_to_rgb(real))
        real_rgb = torch.stack(real_rgb).to(device)

        # Compute Inception features
        f_fake = fid_model(fake_rgb).cpu().numpy()
        f_real = fid_model(real_rgb).cpu().numpy()

        mu_fake, sigma_fake = f_fake.mean(0), np.cov(f_fake, rowvar=False)
        mu_real, sigma_real = f_real.mean(0), np.cov(f_real, rowvar=False)

        fid = compute_fid(mu_fake, sigma_fake, mu_real, sigma_real)
        fid_scores.append(fid)

    # Pad to 20 rows as required (duplicate second set for now)
    full_fid_scores = fid_scores + fid_scores  # IDs 1–10 and 11–20

    submission_df = pd.DataFrame({
        'ID': list(range(1, 21)),
        'Prediction': full_fid_scores
    })
    submission_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    return submission_df



!ls ../input/beyond-visible-spectrum-ai-for-agriculture-2025p2/evaluation



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fid_model = InceptionPool3(device)

submission = generate_submission(
    model=model,  
    fid_model=fid_model,
    real_eval_dir="../input/beyond-visible-spectrum-ai-for-agriculture-2025p2/evaluation",  # replace with actual path
    output_dir="./generated_submission",
    device=device
)



!ls generated_submission


##!rm generated_submission/*npy


!head generated_submission/submission.csv

