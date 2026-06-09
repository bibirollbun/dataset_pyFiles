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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLP & AUTOENCODER DIVERSITY PIPELINE - COMPLETE WORKING VERSION
Creates diversity using only MLPs and various autoencoding techniques
"""

# ============================================================================
# SECTION 0: INSTALLATIONS (Run these if needed)
# ============================================================================

# No special installations needed - using only standard libraries available in Kaggle

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import warnings
import random
import time
from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

SEED = 42
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ID_COL = "id"
TARGET_COL = "BeatsPerMinute"

BASE_NUM_COLS = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# SECTION 3: AUTOENCODER ARCHITECTURES
# ============================================================================

class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder - learns robust features"""
    def __init__(self, input_dim, encoding_dim=32, noise_factor=0.2):
        super().__init__()
        self.noise_factor = noise_factor
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def add_noise(self, x):
        """Add noise for denoising training"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x
        
    def forward(self, x, decode=False):
        # Add noise during training
        x_noisy = self.add_noise(x)
        
        # Encode
        encoded = self.encoder(x_noisy)
        
        if decode:
            # For autoencoder training
            decoded = self.decoder(encoded)
            return encoded, decoded
        else:
            # For prediction
            return self.predictor(encoded)

class VariationalAutoencoder(nn.Module):
    """VAE - learns probabilistic latent representations"""
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2_mu = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)
        
    def forward(self, x, decode=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        if decode:
            return mu, logvar, self.decode(z)
        else:
            return self.predictor(z)

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder - learns sparse representations"""
    def __init__(self, input_dim, encoding_dim=40, sparsity_param=0.05):
        super().__init__()
        self.sparsity_param = sparsity_param
        
        # Encoder with larger hidden layer for sparsity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 200),
            nn.ReLU(),
            nn.Linear(200, input_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(48, 1)
        )
        
    def forward(self, x, decode=False):
        encoded = self.encoder(x)
        
        if decode:
            decoded = self.decoder(encoded)
            return encoded, decoded
        else:
            return self.predictor(encoded)
            
    def kl_divergence(self, encoded):
        """Calculate KL divergence for sparsity"""
        rho_hat = torch.mean(encoded, dim=0)
        rho = torch.full_like(rho_hat, self.sparsity_param)
        kl = torch.sum(rho * torch.log(rho / (rho_hat + 1e-8)) + 
                      (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8)))
        return kl

class ContractiveAutoencoder(nn.Module):
    """Contractive Autoencoder - learns robust to input perturbations"""
    def __init__(self, input_dim, encoding_dim=30):
        super().__init__()
        
        # Encoder layers (keep separate for Jacobian calculation)
        self.W1 = nn.Linear(input_dim, 150)
        self.W2 = nn.Linear(150, encoding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 150),
            nn.ReLU(),
            nn.Linear(150, input_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
        
    def encode(self, x):
        h = F.relu(self.W1(x))
        return F.relu(self.W2(h))
        
    def forward(self, x, decode=False):
        encoded = self.encode(x)
        
        if decode:
            decoded = self.decoder(encoded)
            return encoded, decoded
        else:
            return self.predictor(encoded)
            
    def jacobian_loss(self, x, encoded):
        """Frobenius norm of Jacobian matrix"""
        # Simplified contractive loss
        return torch.mean(torch.sum(encoded**2, dim=1))

class StackedAutoencoder(nn.Module):
    """Stacked Autoencoder - hierarchical feature learning"""
    def __init__(self, input_dim):
        super().__init__()
        
        # Layer 1
        self.encoder1 = nn.Linear(input_dim, 100)
        self.decoder1 = nn.Linear(100, input_dim)
        
        # Layer 2
        self.encoder2 = nn.Linear(100, 50)
        self.decoder2 = nn.Linear(50, 100)
        
        # Layer 3
        self.encoder3 = nn.Linear(50, 25)
        self.decoder3 = nn.Linear(25, 50)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def encode(self, x):
        h1 = F.relu(self.encoder1(x))
        h2 = F.relu(self.encoder2(h1))
        h3 = F.relu(self.encoder3(h2))
        return h1, h2, h3
        
    def decode(self, h3):
        d2 = F.relu(self.decoder3(h3))
        d1 = F.relu(self.decoder2(d2))
        out = self.decoder1(d1)
        return out
        
    def forward(self, x, decode=False):
        h1, h2, h3 = self.encode(x)
        
        if decode:
            decoded = self.decode(h3)
            return h3, decoded
        else:
            return self.predictor(h3)

# ============================================================================
# SECTION 4: DIVERSE MLP ARCHITECTURES
# ============================================================================

class ShallowMLP(nn.Module):
    """Shallow but wide MLP"""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class DeepMLP(nn.Module):
    """Deep but narrow MLP"""
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        # 8 layers with decreasing width
        for hidden_dim in [128, 96, 64, 48, 32, 24, 16, 8]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        
        # Residual blocks
        self.block1 = self._make_block(256)
        self.block2 = self._make_block(256)
        self.block3 = self._make_block(256)
        
        self.output = nn.Linear(256, 1)
        
    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        
        # Residual connections
        x = x + self.block1(x)
        x = F.relu(x)
        x = x + self.block2(x)
        x = F.relu(x)
        x = x + self.block3(x)
        x = F.relu(x)
        
        return self.output(x)

class PyramidalMLP(nn.Module):
    """MLP with pyramidal structure"""
    def __init__(self, input_dim):
        super().__init__()
        
        # Expanding then contracting
        self.expand = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.contract = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.expand(x)
        return self.contract(x)

class HighwayMLP(nn.Module):
    """MLP with highway connections"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        
        # Highway layers
        self.highways = nn.ModuleList([
            self._make_highway(256) for _ in range(5)
        ])
        
        self.output = nn.Linear(256, 1)
        
    def _make_highway(self, dim):
        return nn.ModuleDict({
            'H': nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ),
            'T': nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        })
        
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        
        for highway in self.highways:
            H = highway['H'](x)
            T = highway['T'](x)
            x = H * T + x * (1 - T)
            
        return self.output(x)

# ============================================================================
# SECTION 5: TRAINING STRATEGIES
# ============================================================================

class AutoencoderTrainer:
    """Diverse training strategies for autoencoders and MLPs"""
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.models = {}
        self.scalers = {}
        
    def train_denoising_ae(self, model, X_train, y_train, X_val, y_val, name='dae'):
        """Train denoising autoencoder"""
        print(f"    Training {name}...")
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = scaler_X.transform(X_val)
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Two-phase training
        # Phase 1: Pretrain autoencoder
        print(f"      Phase 1: Pretraining autoencoder...")
        for epoch in range(30):
            model.train()
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                
                optimizer.zero_grad()
                
                _, reconstructed = model(X_batch, decode=True)
                loss = F.mse_loss(reconstructed, X_batch)
                
                loss.backward()
                optimizer.step()
                
        # Phase 2: Fine-tune for prediction
        print(f"      Phase 2: Fine-tuning for prediction...")
        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Prediction loss
                pred = model(X_batch, decode=False).squeeze()
                pred_loss = F.mse_loss(pred, y_batch)
                
                # Reconstruction loss (auxiliary)
                _, reconstructed = model(X_batch, decode=True)
                recon_loss = F.mse_loss(reconstructed, X_batch)
                
                # Combined loss
                loss = pred_loss + 0.1 * recon_loss
                
                loss.backward()
                optimizer.step()
                
        self.models[name] = model
        
    def train_vae(self, model, X_train, y_train, X_val, y_val, name='vae'):
        """Train variational autoencoder with numerical stability"""
        print(f"    Training {name}...")
        
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        for epoch in range(80):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE forward
                mu, logvar, reconstructed = model(X_batch, decode=True)
                
                # Clamp logvar for numerical stability
                logvar = torch.clamp(logvar, min=-10, max=10)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, X_batch, reduction='sum')
                
                # KL divergence with numerical stability
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=1e10))
                
                # Prediction loss
                pred = model(X_batch, decode=False).squeeze()
                pred_loss = F.mse_loss(pred, y_batch) * X_batch.size(0)
                
                # Total loss with KL annealing
                beta = min(1.0, epoch / 40)
                loss = pred_loss + 0.01 * (recon_loss + beta * kl_loss)
                
                # Check for NaN
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
        self.models[name] = model
        
    def train_sparse_ae(self, model, X_train, y_train, X_val, y_val, name='sparse'):
        """Train sparse autoencoder"""
        print(f"    Training {name}...")
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        for epoch in range(60):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                encoded, reconstructed = model(X_batch, decode=True)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, X_batch)
                
                # Sparsity loss (KL divergence)
                sparsity_loss = model.kl_divergence(encoded)
                
                # Prediction loss
                pred = model(X_batch, decode=False).squeeze()
                pred_loss = F.mse_loss(pred, y_batch)
                
                # Total loss
                loss = pred_loss + 0.1 * recon_loss + 0.001 * sparsity_loss
                
                loss.backward()
                optimizer.step()
                
        self.models[name] = model
        
    def train_contractive_ae(self, model, X_train, y_train, X_val, y_val, name='contractive'):
        """Train contractive autoencoder"""
        print(f"    Training {name}...")
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(60):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                encoded, reconstructed = model(X_batch, decode=True)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, X_batch)
                
                # Contractive loss
                contractive_loss = model.jacobian_loss(X_batch, encoded)
                
                # Prediction loss
                pred = model(X_batch, decode=False).squeeze()
                pred_loss = F.mse_loss(pred, y_batch)
                
                # Total loss
                loss = pred_loss + 0.1 * recon_loss + 0.001 * contractive_loss
                
                loss.backward()
                optimizer.step()
                
        self.models[name] = model
        
    def train_stacked_ae(self, model, X_train, y_train, X_val, y_val, name='stacked'):
        """Train stacked autoencoder"""
        print(f"    Training {name}...")
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(60):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                encoded, reconstructed = model(X_batch, decode=True)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, X_batch)
                
                # Prediction loss
                pred = model(X_batch, decode=False).squeeze()
                pred_loss = F.mse_loss(pred, y_batch)
                
                # Combined loss
                loss = pred_loss + 0.1 * recon_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
        self.models[name] = model
        
    def train_mlp(self, model, X_train, y_train, X_val, y_val, name='mlp', 
                  loss_fn='mse', optimizer_type='adam'):
        """Train MLP with diverse settings"""
        print(f"    Training {name} with {loss_fn} loss and {optimizer_type} optimizer...")
        
        # Different scalers for diversity
        if 'robust' in name:
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        elif 'minmax' in name:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        else:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )
        
        # Different batch sizes
        batch_size = 128 if 'small' in name else 256 if 'medium' in name else 512
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)
        
        # Different optimizers
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            
        # Different loss functions
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            def criterion(pred, target):
                # Quantile loss
                q = 0.5
                errors = target - pred
                return torch.max(q * errors, (q - 1) * errors).mean()
                
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
            # Validation
            if epoch % 10 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        output = model(X_batch).squeeze()
                        loss = F.mse_loss(output, y_batch)
                        val_losses.append(loss.item())
                        
                avg_val_loss = np.mean(val_losses)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break
                        
        if best_state is not None:
            model.load_state_dict(best_state)
        self.models[name] = model
        
    def predict(self, X, name):
        """Generate predictions"""
        model = self.models[name]
        model.eval()
        
        X_scaled = self.scalers[f'{name}_X'].transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            if 'ae' in name or 'vae' in name or 'sparse' in name or 'contractive' in name or 'stacked' in name:
                pred = model(X_tensor, decode=False).cpu().numpy()
            else:
                pred = model(X_tensor).cpu().numpy()
                
            pred = self.scalers[f'{name}_y'].inverse_transform(pred.reshape(-1, 1)).ravel()
            
        return pred

# ============================================================================
# SECTION 6: FEATURE ENGINEERING FOR MLP/AUTOENCODER
# ============================================================================

def create_autoencoder_features(train, test, base_cols):
    """Create features optimized for autoencoders"""
    
    train_fe = train.copy()
    test_fe = test.copy()
    
    # Normalize base features
    for col in base_cols:
        if col in train_fe.columns:
            mean = train_fe[col].mean()
            std = train_fe[col].std()
            train_fe[col] = (train_fe[col] - mean) / (std + 1e-8)
            test_fe[col] = (test_fe[col] - mean) / (std + 1e-8)
            
    # Add polynomial features for MLPs
    for col in base_cols:
        train_fe[f'{col}_sq'] = train_fe[col] ** 2
        test_fe[f'{col}_sq'] = test_fe[col] ** 2
        
    # Add interaction features
    for i, col1 in enumerate(base_cols[:-1]):
        for col2 in base_cols[i+1:]:
            train_fe[f'{col1}_x_{col2}'] = train_fe[col1] * train_fe[col2]
            test_fe[f'{col1}_x_{col2}'] = test_fe[col1] * test_fe[col2]
            
    # Add ratios
    for col1 in base_cols[:3]:
        for col2 in base_cols[3:6]:
            train_fe[f'{col1}_div_{col2}'] = train_fe[col1] / (train_fe[col2] + 1e-8)
            test_fe[f'{col1}_div_{col2}'] = test_fe[col1] / (test_fe[col2] + 1e-8)
            
    # Clean up
    train_fe.fillna(0, inplace=True)
    test_fe.fillna(0, inplace=True)
    train_fe.replace([np.inf, -np.inf], 0, inplace=True)
    test_fe.replace([np.inf, -np.inf], 0, inplace=True)
    
    feature_cols = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
    
    return train_fe, test_fe, feature_cols

# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================

def main_mlp_autoencoder_pipeline():
    """Main pipeline using only MLPs and Autoencoders"""
    
    print("\n" + "="*80)
    print("MLP & AUTOENCODER DIVERSITY PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape: {test.shape}")
    
    # Feature engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 40)
    train_fe, test_fe, feature_cols = create_autoencoder_features(train, test, BASE_NUM_COLS)
    
    X = train_fe[feature_cols].values
    y = train_fe[TARGET_COL].values
    X_test = test_fe[feature_cols].values
    
    print(f"  Features created: {len(feature_cols)}")
    
    target_mean = y.mean()
    target_std = y.std()
    print(f"  Target: Mean={target_mean:.2f}, Std={target_std:.2f}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    # Initialize trainer
    trainer = AutoencoderTrainer()
    
    # Storage for predictions
    all_predictions_oof = []
    all_predictions_test = []
    model_names = []
    
    # Train autoencoders
    print("\n3. TRAINING AUTOENCODERS")
    print("-" * 40)
    
    # Denoising Autoencoder
    dae = DenoisingAutoencoder(X.shape[1], encoding_dim=32, noise_factor=0.2).to(DEVICE)
    trainer.train_denoising_ae(dae, X_train, y_train, X_val, y_val, name='dae')
    pred_oof = trainer.predict(X, 'dae')
    pred_test = trainer.predict(X_test, 'dae')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('DAE')
    print(f"  DAE STD: {np.std(pred_oof):.3f}")
    
    # Variational Autoencoder
    vae = VariationalAutoencoder(X.shape[1], latent_dim=20).to(DEVICE)
    trainer.train_vae(vae, X_train, y_train, X_val, y_val, name='vae')
    pred_oof = trainer.predict(X, 'vae')
    pred_test = trainer.predict(X_test, 'vae')
    
    # Check for NaN and handle
    if np.isnan(pred_oof).any():
        print(f"  Warning: NaN values detected in VAE, using fallback")
        pred_oof = np.full_like(pred_oof, target_mean)
        pred_test = np.full_like(pred_test, target_mean)
    
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('VAE')
    print(f"  VAE STD: {np.std(pred_oof):.3f}")
    
    # Sparse Autoencoder
    sparse_ae = SparseAutoencoder(X.shape[1], encoding_dim=40, sparsity_param=0.05).to(DEVICE)
    trainer.train_sparse_ae(sparse_ae, X_train, y_train, X_val, y_val, name='sparse')
    pred_oof = trainer.predict(X, 'sparse')
    pred_test = trainer.predict(X_test, 'sparse')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Sparse')
    print(f"  Sparse AE STD: {np.std(pred_oof):.3f}")
    
    # Contractive Autoencoder
    contractive_ae = ContractiveAutoencoder(X.shape[1], encoding_dim=30).to(DEVICE)
    trainer.train_contractive_ae(contractive_ae, X_train, y_train, X_val, y_val, name='contractive')
    pred_oof = trainer.predict(X, 'contractive')
    pred_test = trainer.predict(X_test, 'contractive')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Contractive')
    print(f"  Contractive AE STD: {np.std(pred_oof):.3f}")
    
    # Stacked Autoencoder - FIXED: using correct training method
    stacked_ae = StackedAutoencoder(X.shape[1]).to(DEVICE)
    trainer.train_stacked_ae(stacked_ae, X_train, y_train, X_val, y_val, name='stacked')
    pred_oof = trainer.predict(X, 'stacked')
    pred_test = trainer.predict(X_test, 'stacked')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Stacked')
    print(f"  Stacked AE STD: {np.std(pred_oof):.3f}")
    
    # Train diverse MLPs
    print("\n4. TRAINING DIVERSE MLPs")
    print("-" * 40)
    
    # Shallow MLP
    shallow_mlp = ShallowMLP(X.shape[1]).to(DEVICE)
    trainer.train_mlp(shallow_mlp, X_train, y_train, X_val, y_val, 
                     name='shallow', loss_fn='mse', optimizer_type='adam')
    pred_oof = trainer.predict(X, 'shallow')
    pred_test = trainer.predict(X_test, 'shallow')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Shallow')
    print(f"  Shallow MLP STD: {np.std(pred_oof):.3f}")
    
    # Deep MLP
    deep_mlp = DeepMLP(X.shape[1]).to(DEVICE)
    trainer.train_mlp(deep_mlp, X_train, y_train, X_val, y_val,
                     name='deep_robust', loss_fn='huber', optimizer_type='sgd')
    pred_oof = trainer.predict(X, 'deep_robust')
    pred_test = trainer.predict(X_test, 'deep_robust')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Deep')
    print(f"  Deep MLP STD: {np.std(pred_oof):.3f}")
    
    # Residual MLP
    residual_mlp = ResidualMLP(X.shape[1]).to(DEVICE)
    trainer.train_mlp(residual_mlp, X_train, y_train, X_val, y_val,
                     name='residual_minmax', loss_fn='mae', optimizer_type='rmsprop')
    pred_oof = trainer.predict(X, 'residual_minmax')
    pred_test = trainer.predict(X_test, 'residual_minmax')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Residual')
    print(f"  Residual MLP STD: {np.std(pred_oof):.3f}")
    
    # Pyramidal MLP
    pyramidal_mlp = PyramidalMLP(X.shape[1]).to(DEVICE)
    trainer.train_mlp(pyramidal_mlp, X_train, y_train, X_val, y_val,
                     name='pyramidal_small', loss_fn='quantile', optimizer_type='adamw')
    pred_oof = trainer.predict(X, 'pyramidal_small')
    pred_test = trainer.predict(X_test, 'pyramidal_small')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Pyramidal')
    print(f"  Pyramidal MLP STD: {np.std(pred_oof):.3f}")
    
    # Highway MLP
    highway_mlp = HighwayMLP(X.shape[1]).to(DEVICE)
    trainer.train_mlp(highway_mlp, X_train, y_train, X_val, y_val,
                     name='highway_medium', loss_fn='mse', optimizer_type='adam')
    pred_oof = trainer.predict(X, 'highway_medium')
    pred_test = trainer.predict(X_test, 'highway_medium')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Highway')
    print(f"  Highway MLP STD: {np.std(pred_oof):.3f}")
    
    # Create ensemble
    print("\n5. CREATING ENSEMBLE")
    print("-" * 40)
    
    # Stack predictions
    predictions_oof = np.column_stack(all_predictions_oof)
    predictions_test = np.column_stack(all_predictions_test)
    
    # Check diversity
    corr_matrix = np.corrcoef(predictions_oof.T)
    avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix) * (len(corr_matrix) - 1))
    print(f"  Average correlation: {avg_corr:.3f}")
    print(f"  Number of models: {len(model_names)}")
    
    # Simple average ensemble
    final_oof = np.mean(predictions_oof, axis=1)
    final_test = np.mean(predictions_test, axis=1)
    
    # Weighted average based on OOF performance
    oof_errors = []
    for i in range(predictions_oof.shape[1]):
        error = mean_squared_error(y, predictions_oof[:, i])
        oof_errors.append(error)
        
    # Convert errors to weights (inverse)
    weights = 1 / (np.array(oof_errors) + 1e-8)
    weights = weights / weights.sum()
    
    final_oof_weighted = np.average(predictions_oof, weights=weights, axis=1)
    final_test_weighted = np.average(predictions_test, weights=weights, axis=1)
    
    print(f"  Simple ensemble STD: {np.std(final_test):.3f}")
    print(f"  Weighted ensemble STD: {np.std(final_test_weighted):.3f}")
    
    # Choose best ensemble
    final_test = final_test_weighted if np.std(final_test_weighted) > np.std(final_test) * 0.9 else final_test
    
    print(f"  Final STD: {np.std(final_test):.3f}")
    print(f"  STD ratio: {np.std(final_test)/target_std*100:.1f}%")
    
    # Post-processing
    print("\n6. POST-PROCESSING")
    print("-" * 40)
    
    # Clip to reasonable bounds
    lower_clip = np.percentile(y, 1)
    upper_clip = np.percentile(y, 99)
    final_test_clipped = np.clip(final_test, lower_clip, upper_clip)
    
    print(f"  Final clipped STD: {np.std(final_test_clipped):.3f}")
    
    # Save submission
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: final_test_clipped
    })
    
    submission.to_csv("submission_mlp_ae.csv", index=False)
    print("  ✓ Saved to submission_mlp_ae.csv")
    
    # Model analysis
    print("\n7. MODEL DIVERSITY ANALYSIS")
    print("-" * 40)
    print("  Model       | Mean    | STD    | Min    | Max")
    print("  " + "-"*50)
    
    for i, name in enumerate(model_names):
        pred = predictions_test[:, i]
        print(f"  {name:11} | {pred.mean():7.2f} | {pred.std():6.2f} | "
              f"{pred.min():6.2f} | {pred.max():6.2f}")
        
    # Correlation heatmap values
    print("\n  Correlation Matrix (sample):")
    print("  " + "-"*50)
    for i in range(min(5, len(model_names))):
        corr_row = [corr_matrix[i, j] for j in range(min(5, len(model_names)))]
        print(f"  {model_names[i]:11} | " + " | ".join([f"{c:.2f}" for c in corr_row]))
        
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    print("\n" + "="*80)
    print("MLP & AUTOENCODER PIPELINE COMPLETED!")
    print("="*80)
    
    return submission

# ============================================================================
# SECTION 8: EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\nStarting MLP & Autoencoder Diversity Pipeline...")
        print("Using ONLY MLPs and various Autoencoding techniques")
        submission = main_mlp_autoencoder_pipeline()
        print(f"\n✓ Submission shape: {submission.shape}")
        print("✓ Diversity achieved through MLP and Autoencoder variations!")
        print("✓ No errors encountered - pipeline completed successfully!")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

