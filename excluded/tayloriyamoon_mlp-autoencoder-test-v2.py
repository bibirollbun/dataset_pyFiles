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
FIXED ENHANCED MLP & AUTOENCODER DIVERSITY PIPELINE
Complete working version with all bugs fixed
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import warnings
import random
import time
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR

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
# SECTION 3: CUSTOM ACTIVATION FUNCTIONS
# ============================================================================

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# ============================================================================
# SECTION 4: CUSTOM DROPOUT
# ============================================================================

class GaussianDropout(nn.Module):
    """Multiplicative Gaussian Noise"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.p + 1
            return x * noise
        return x

# ============================================================================
# SECTION 5: AUTOENCODER ARCHITECTURES
# ============================================================================

class EnhancedDenoisingAutoencoder(nn.Module):
    """DAE with diverse activation functions"""
    def __init__(self, input_dim, encoding_dim=32, noise_factor=0.2, activation='relu'):
        super().__init__()
        self.noise_factor = noise_factor
        
        # Select activation
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'elu':
            self.act = nn.ELU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'swish':
            self.act = Swish()
        elif activation == 'mish':
            self.act = Mish()
        else:
            self.act = nn.ReLU()
            
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            self.act,
            GaussianDropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.2),
            nn.Linear(128, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, input_dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            self.act,
            nn.Linear(32, 1)
        )
        
    def add_noise(self, x):
        """Add various types of noise"""
        if self.training:
            gaussian_noise = torch.randn_like(x) * self.noise_factor
            uniform_noise = (torch.rand_like(x) - 0.5) * self.noise_factor
            noise = gaussian_noise * 0.7 + uniform_noise * 0.3
            return x + noise
        return x
        
    def forward(self, x, decode=False):
        x_noisy = self.add_noise(x)
        encoded = self.encoder(x_noisy)
        
        if decode:
            decoded = self.decoder(encoded)
            return encoded, decoded
        else:
            return self.predictor(encoded)

class ImprovedVAE(nn.Module):
    """VAE with better numerical stability"""
    def __init__(self, input_dim, latent_dim=20, activation='elu'):
        super().__init__()
        
        # Activation selection
        if activation == 'elu':
            self.act = nn.ELU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.ReLU()
            
        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2_mu = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, input_dim)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            self.act,
            nn.Linear(64, 1)
        )
        
    def encode(self, x):
        h = self.act(self.bn1(self.fc1(x)))
        h = self.act(self.bn2(self.fc2(h)))
        return self.fc2_mu(h), self.fc2_logvar(h)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, min=-4, max=4))
            eps = torch.randn_like(std) * 0.1
            return mu + eps * std
        return mu
        
    def decode(self, z):
        h = self.act(self.bn3(self.fc3(z)))
        h = self.act(self.bn4(self.fc4(h)))
        return self.fc5(h)
        
    def forward(self, x, decode=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        if decode:
            return mu, logvar, self.decode(z)
        else:
            return self.predictor(z)

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder"""
    def __init__(self, input_dim, encoding_dim=40, sparsity_param=0.05):
        super().__init__()
        self.sparsity_param = sparsity_param
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 200),
            nn.ReLU(),
            nn.Linear(200, input_dim)
        )
        
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

# ============================================================================
# SECTION 6: MLP ARCHITECTURES
# ============================================================================

class SELUNet(nn.Module):
    """Deep network with SELU activation"""
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in [256, 192, 128, 96, 64, 32]:
            layer = nn.Linear(current_dim, hidden_dim)
            nn.init.normal_(layer.weight, 0, np.sqrt(1 / current_dim))
            layers.extend([
                layer,
                nn.SELU(),
                nn.AlphaDropout(0.1)
            ])
            current_dim = hidden_dim
            
        final_layer = nn.Linear(current_dim, 1)
        nn.init.normal_(final_layer.weight, 0, np.sqrt(1 / current_dim))
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class SwishNet(nn.Module):
    """Network using Swish activation"""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            Swish(),
            GaussianDropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class MishNet(nn.Module):
    """Network using Mish activation"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.fc4 = nn.Linear(96, 1)
        
        self.mish = Mish()
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = GaussianDropout(0.2)
        
    def forward(self, x):
        x = self.mish(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.mish(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.mish(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x)

class DenseNet(nn.Module):
    """DenseNet-style architecture"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(input_dim + 128, 128)
        self.fc3 = nn.Linear(input_dim + 256, 128)
        self.fc4 = nn.Linear(input_dim + 384, 64)
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        x0 = x
        x1 = F.relu(self.fc1(x))
        x = torch.cat([x0, x1], dim=1)
        
        x2 = F.relu(self.fc2(x))
        x = torch.cat([x0, x1, x2], dim=1)
        
        x3 = F.relu(self.fc3(x))
        x = torch.cat([x0, x1, x2, x3], dim=1)
        
        x = F.relu(self.fc4(x))
        return self.fc_out(x)

class InceptionMLP(nn.Module):
    """Inception-style MLP"""
    def __init__(self, input_dim):
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        
        self.path3 = nn.Sequential(
            nn.Linear(input_dim, 64),
            Swish(),
            nn.Linear(64, 128),
            Swish(),
            nn.Linear(128, 256),
            Swish(),
            nn.Dropout(0.3)
        )
        
        self.merge = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        merged = torch.cat([p1, p2, p3], dim=1)
        return self.merge(merged)

class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.output = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = x + self.block1(x)
        x = F.relu(x)
        x = x + self.block2(x)
        x = F.relu(x)
        return self.output(x)

class PyramidalMLP(nn.Module):
    """MLP with pyramidal structure"""
    def __init__(self, input_dim):
        super().__init__()
        
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

# ============================================================================
# SECTION 7: OPTIMIZERS
# ============================================================================

def get_optimizer(model, opt_name='adam', lr=0.001):
    """Get optimizer by name"""
    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == 'adamax':
        return optim.Adamax(model.parameters(), lr=lr)
    elif opt_name == 'nadam':
        return optim.NAdam(model.parameters(), lr=lr)
    elif opt_name == 'radam':
        return optim.RAdam(model.parameters(), lr=lr)
    else:
        return optim.Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, scheduler_name='none', num_epochs=100, steps_per_epoch=100):
    """Get learning rate scheduler"""
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', patience=5)
    elif scheduler_name == 'cyclic':
        return CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=steps_per_epoch*5)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(optimizer, max_lr=0.01, total_steps=num_epochs*steps_per_epoch)
    else:
        return None

# ============================================================================
# SECTION 8: TRAINER
# ============================================================================

class EnhancedTrainer:
    """Training class for all models"""
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.models = {}
        self.scalers = {}
        
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   name='model', scaler_type='standard', 
                   optimizer_name='adam', scheduler_name='none',
                   loss_fn='mse', batch_size=256, num_epochs=60,
                   use_mixup=False, model_type='mlp'):
        """Train model with various techniques"""
        print(f"    Training {name}...")
        print(f"      Config: {scaler_type} scaling, {optimizer_name} optimizer")
        
        # Select scaler
        if scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type == 'robust':
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        elif scaler_type == 'minmax':
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif scaler_type == 'quantile':
            scaler_X = QuantileTransformer(n_quantiles=100, output_distribution='normal')
            scaler_y = QuantileTransformer(n_quantiles=100, output_distribution='normal')
        elif scaler_type == 'power':
            scaler_X = PowerTransformer(method='yeo-johnson')
            scaler_y = PowerTransformer(method='yeo-johnson')
        else:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)
        
        # Get optimizer
        optimizer = get_optimizer(model, optimizer_name, lr=0.001)
        scheduler = get_scheduler(optimizer, scheduler_name, num_epochs, len(train_loader))
        
        # Loss function
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
            
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Mixup
                if use_mixup and random.random() > 0.5:
                    lam = np.random.beta(0.2, 0.2)
                    batch_size = X_batch.size(0)
                    index = torch.randperm(batch_size).to(self.device)
                    X_batch = lam * X_batch + (1 - lam) * X_batch[index]
                    y_batch = lam * y_batch + (1 - lam) * y_batch[index]
                
                optimizer.zero_grad()
                
                # Forward pass based on model type
                if model_type == 'vae':
                    # VAE specific
                    mu, logvar, reconstructed = model(X_batch, decode=True)
                    pred = model(X_batch, decode=False).squeeze()
                    
                    # VAE loss
                    recon_loss = F.mse_loss(reconstructed, X_batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    pred_loss = criterion(pred, y_batch)
                    
                    loss = pred_loss + 0.01 * (recon_loss + 0.01 * kl_loss)
                    
                elif model_type == 'sparse':
                    # Sparse AE specific
                    encoded, reconstructed = model(X_batch, decode=True)
                    pred = model(X_batch, decode=False).squeeze()
                    
                    recon_loss = F.mse_loss(reconstructed, X_batch)
                    sparsity_loss = model.kl_divergence(encoded)
                    pred_loss = criterion(pred, y_batch)
                    
                    loss = pred_loss + 0.1 * recon_loss + 0.001 * sparsity_loss
                    
                elif model_type == 'ae':
                    # Regular autoencoder
                    encoded, reconstructed = model(X_batch, decode=True)
                    pred = model(X_batch, decode=False).squeeze()
                    
                    recon_loss = F.mse_loss(reconstructed, X_batch)
                    pred_loss = criterion(pred, y_batch)
                    
                    loss = pred_loss + 0.1 * recon_loss
                    
                else:
                    # Regular MLP
                    output = model(X_batch).squeeze()
                    loss = criterion(output, y_batch)
                
                # Check for NaN
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
                
                if scheduler and scheduler_name in ['cyclic', 'onecycle']:
                    scheduler.step()
                    
            # Validation
            if epoch % 5 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        
                        if model_type in ['vae', 'ae', 'sparse']:
                            output = model(X_batch, decode=False).squeeze()
                        else:
                            output = model(X_batch).squeeze()
                            
                        loss = F.mse_loss(output, y_batch)
                        val_losses.append(loss.item())
                        
                avg_val_loss = np.mean(val_losses)
                
                if scheduler and scheduler_name == 'plateau':
                    scheduler.step(avg_val_loss)
                    
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 8:
                        break
                        
            if scheduler and scheduler_name == 'cosine':
                scheduler.step()
                
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
            if hasattr(model, 'decode'):
                pred = model(X_tensor, decode=False).cpu().numpy()
            else:
                pred = model(X_tensor).cpu().numpy()
                
            pred = self.scalers[f'{name}_y'].inverse_transform(pred.reshape(-1, 1)).ravel()
            
        return pred

# ============================================================================
# SECTION 9: FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features(train, test, base_cols):
    """Enhanced feature engineering"""
    
    train_fe = train.copy()
    test_fe = test.copy()
    
    # Base normalization
    for col in base_cols:
        if col in train_fe.columns:
            mean = train_fe[col].mean()
            std = train_fe[col].std()
            train_fe[f'{col}_norm'] = (train_fe[col] - mean) / (std + 1e-8)
            test_fe[f'{col}_norm'] = (test_fe[col] - mean) / (std + 1e-8)
            
    # Polynomial features
    for col in base_cols:
        train_fe[f'{col}_sq'] = train_fe[col] ** 2
        test_fe[f'{col}_sq'] = test_fe[col] ** 2
        train_fe[f'{col}_sqrt'] = np.sqrt(np.abs(train_fe[col]))
        test_fe[f'{col}_sqrt'] = np.sqrt(np.abs(test_fe[col]))
        
    # Log transformations
    for col in base_cols:
        train_fe[f'{col}_log'] = np.log1p(np.abs(train_fe[col]))
        test_fe[f'{col}_log'] = np.log1p(np.abs(test_fe[col]))
        
    # Trigonometric features for first 3 columns
    for col in base_cols[:3]:
        train_fe[f'{col}_sin'] = np.sin(train_fe[col])
        test_fe[f'{col}_sin'] = np.sin(test_fe[col])
        train_fe[f'{col}_cos'] = np.cos(train_fe[col])
        test_fe[f'{col}_cos'] = np.cos(test_fe[col])
        
    # Interaction features (limited to avoid too many features)
    for i, col1 in enumerate(base_cols[:5]):
        for col2 in base_cols[i+1:6]:
            train_fe[f'{col1}_x_{col2}'] = train_fe[col1] * train_fe[col2]
            test_fe[f'{col1}_x_{col2}'] = test_fe[col1] * test_fe[col2]
            
    # Ratios
    for col1 in base_cols[:3]:
        for col2 in base_cols[6:]:
            if train_fe[col2].std() > 0:
                train_fe[f'{col1}_div_{col2}'] = train_fe[col1] / (train_fe[col2] + 1e-8)
                test_fe[f'{col1}_div_{col2}'] = test_fe[col1] / (test_fe[col2] + 1e-8)
            
    # Statistical aggregations
    train_fe['mean_features'] = train_fe[base_cols].mean(axis=1)
    test_fe['mean_features'] = test_fe[base_cols].mean(axis=1)
    train_fe['std_features'] = train_fe[base_cols].std(axis=1)
    test_fe['std_features'] = test_fe[base_cols].std(axis=1)
    train_fe['max_features'] = train_fe[base_cols].max(axis=1)
    test_fe['max_features'] = test_fe[base_cols].max(axis=1)
    train_fe['min_features'] = train_fe[base_cols].min(axis=1)
    test_fe['min_features'] = test_fe[base_cols].min(axis=1)
    
    # Clean up
    train_fe.fillna(0, inplace=True)
    test_fe.fillna(0, inplace=True)
    train_fe.replace([np.inf, -np.inf], 0, inplace=True)
    test_fe.replace([np.inf, -np.inf], 0, inplace=True)
    
    feature_cols = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
    
    return train_fe, test_fe, feature_cols

# ============================================================================
# SECTION 10: MAIN PIPELINE
# ============================================================================

def main_pipeline():
    """Main enhanced pipeline"""
    
    print("\n" + "="*80)
    print("ENHANCED MLP & AUTOENCODER DIVERSITY PIPELINE")
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
    train_fe, test_fe, feature_cols = create_enhanced_features(train, test, BASE_NUM_COLS)
    
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
    trainer = EnhancedTrainer()
    
    # Storage for predictions
    all_predictions_oof = []
    all_predictions_test = []
    model_names = []
    
    # ========== AUTOENCODERS ==========
    print("\n3. TRAINING AUTOENCODERS")
    print("-" * 40)
    
    # DAE with ReLU
    model = EnhancedDenoisingAutoencoder(X.shape[1], encoding_dim=32, activation='relu').to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='dae_relu', scaler_type='standard',
                       optimizer_name='adam', scheduler_name='cosine',
                       loss_fn='mse', batch_size=256, model_type='ae')
    pred_oof = trainer.predict(X, 'dae_relu')
    pred_test = trainer.predict(X_test, 'dae_relu')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('DAE-ReLU')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # DAE with Swish
    model = EnhancedDenoisingAutoencoder(X.shape[1], encoding_dim=40, activation='swish').to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='dae_swish', scaler_type='robust',
                       optimizer_name='adamw', scheduler_name='plateau',
                       loss_fn='huber', batch_size=128, model_type='ae')
    pred_oof = trainer.predict(X, 'dae_swish')
    pred_test = trainer.predict(X_test, 'dae_swish')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('DAE-Swish')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # VAE with ELU
    model = ImprovedVAE(X.shape[1], latent_dim=25, activation='elu').to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='vae_elu', scaler_type='quantile',
                       optimizer_name='radam', scheduler_name='cyclic',
                       loss_fn='mse', batch_size=256, model_type='vae')
    pred_oof = trainer.predict(X, 'vae_elu')
    pred_test = trainer.predict(X_test, 'vae_elu')
    if not np.isnan(pred_oof).any():
        all_predictions_oof.append(pred_oof)
        all_predictions_test.append(pred_test)
        model_names.append('VAE-ELU')
        print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Sparse Autoencoder
    model = SparseAutoencoder(X.shape[1], encoding_dim=35).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='sparse_ae', scaler_type='minmax',
                       optimizer_name='sgd', scheduler_name='cosine',
                       loss_fn='mse', batch_size=384, model_type='sparse')
    pred_oof = trainer.predict(X, 'sparse_ae')
    pred_test = trainer.predict(X_test, 'sparse_ae')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Sparse-AE')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # ========== MLPs ==========
    print("\n4. TRAINING MLPs")
    print("-" * 40)
    
    # SELU Network
    model = SELUNet(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='selu_net', scaler_type='standard',
                       optimizer_name='adamax', scheduler_name='plateau',
                       loss_fn='mse', batch_size=256, model_type='mlp')
    pred_oof = trainer.predict(X, 'selu_net')
    pred_test = trainer.predict(X_test, 'selu_net')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('SELU-Net')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Swish Network
    model = SwishNet(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='swish_net', scaler_type='robust',
                       optimizer_name='adam', scheduler_name='cosine',
                       loss_fn='mae', batch_size=192, model_type='mlp')
    pred_oof = trainer.predict(X, 'swish_net')
    pred_test = trainer.predict(X_test, 'swish_net')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Swish-Net')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Mish Network
    model = MishNet(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='mish_net', scaler_type='power',
                       optimizer_name='rmsprop', scheduler_name='cyclic',
                       loss_fn='huber', batch_size=128, model_type='mlp', use_mixup=True)
    pred_oof = trainer.predict(X, 'mish_net')
    pred_test = trainer.predict(X_test, 'mish_net')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Mish-Net')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Dense Network
    model = DenseNet(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='dense_net', scaler_type='quantile',
                       optimizer_name='nadam', scheduler_name='onecycle',
                       loss_fn='mse', batch_size=256, model_type='mlp')
    pred_oof = trainer.predict(X, 'dense_net')
    pred_test = trainer.predict(X_test, 'dense_net')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Dense-Net')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Inception MLP
    model = InceptionMLP(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='inception_mlp', scaler_type='minmax',
                       optimizer_name='adamw', scheduler_name='plateau',
                       loss_fn='mse', batch_size=192, model_type='mlp')
    pred_oof = trainer.predict(X, 'inception_mlp')
    pred_test = trainer.predict(X_test, 'inception_mlp')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Inception')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Residual MLP
    model = ResidualMLP(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='residual_mlp', scaler_type='standard',
                       optimizer_name='sgd', scheduler_name='cosine',
                       loss_fn='mse', batch_size=384, model_type='mlp')
    pred_oof = trainer.predict(X, 'residual_mlp')
    pred_test = trainer.predict(X_test, 'residual_mlp')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Residual')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # Pyramidal MLP
    model = PyramidalMLP(X.shape[1]).to(DEVICE)
    trainer.train_model(model, X_train, y_train, X_val, y_val,
                       name='pyramidal_mlp', scaler_type='robust',
                       optimizer_name='adam', scheduler_name='plateau',
                       loss_fn='huber', batch_size=256, model_type='mlp')
    pred_oof = trainer.predict(X, 'pyramidal_mlp')
    pred_test = trainer.predict(X_test, 'pyramidal_mlp')
    all_predictions_oof.append(pred_oof)
    all_predictions_test.append(pred_test)
    model_names.append('Pyramidal')
    print(f"      STD: {np.std(pred_oof):.3f}")
    
    # ========== ENSEMBLE ==========
    print("\n5. CREATING ENSEMBLE")
    print("-" * 40)
    
    # Stack predictions
    predictions_oof = np.column_stack(all_predictions_oof)
    predictions_test = np.column_stack(all_predictions_test)
    
    print(f"  Number of models: {len(model_names)}")
    
    # Check diversity
    valid_corr = []
    for i in range(predictions_oof.shape[1]):
        for j in range(i+1, predictions_oof.shape[1]):
            if not (np.isnan(predictions_oof[:, i]).any() or np.isnan(predictions_oof[:, j]).any()):
                corr = np.corrcoef(predictions_oof[:, i], predictions_oof[:, j])[0, 1]
                valid_corr.append(corr)
    
    if valid_corr:
        avg_corr = np.mean(valid_corr)
        print(f"  Average correlation: {avg_corr:.3f}")
    
    # Weighted ensemble
    oof_errors = []
    for i in range(predictions_oof.shape[1]):
        if not np.isnan(predictions_oof[:, i]).any():
            error = mean_squared_error(y, predictions_oof[:, i])
            oof_errors.append(error)
        else:
            oof_errors.append(1e10)
    
    weights = 1 / (np.array(oof_errors) + 1e-8)
    weights = weights / weights.sum()
    
    # Final predictions
    final_test = np.average(predictions_test, weights=weights, axis=1)
    
    print(f"  Final STD: {np.std(final_test):.3f}")
    print(f"  STD ratio: {np.std(final_test)/target_std*100:.1f}%")
    
    # Post-processing
    print("\n6. POST-PROCESSING")
    print("-" * 40)
    
    lower_clip = np.percentile(y, 0.5)
    upper_clip = np.percentile(y, 99.5)
    final_test_clipped = np.clip(final_test, lower_clip, upper_clip)
    
    print(f"  Clipped STD: {np.std(final_test_clipped):.3f}")
    
    # Save submission
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: final_test_clipped
    })
    
    submission.to_csv("submission_enhanced.csv", index=False)
    print("  ✓ Saved to submission_enhanced.csv")
    
    # Model analysis
    print("\n7. MODEL ANALYSIS")
    print("-" * 40)
    print("  Model      | Mean    | STD    | Weight")
    print("  " + "-"*40)
    
    for i, name in enumerate(model_names):
        pred = predictions_test[:, i]
        if not np.isnan(pred).any():
            print(f"  {name:10} | {pred.mean():7.2f} | {pred.std():6.2f} | {weights[i]:.3f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return submission

# ============================================================================
# SECTION 11: EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\nStarting Enhanced Pipeline...")
        print("Features:")
        print("  • Multiple activation functions")
        print("  • Various optimizers and schedulers")
        print("  • Different dropout techniques")
        print("  • Multiple scalers")
        print("  • Advanced architectures")
        
        submission = main_pipeline()
        print(f"\n✓ Submission shape: {submission.shape}")
        print("✓ Pipeline completed successfully!")
        print("✓ Check submission_enhanced.csv for results")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

