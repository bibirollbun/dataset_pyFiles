%%writefile config.py
import torch
import os
from types import SimpleNamespace
from kaggle_datasets import KaggleDatasets
import numpy as np

class CompetitionConfig:
    """Optimal configuration for waveform inversion with physics-informed deep learning."""
    
    def __init__(self):
        # Hardware setup
        self._setup_environment()
        
        # Data paths
        self._setup_paths()
        
        # Physics parameters
        self._setup_physics()
        
        # Model architecture
        self._setup_model()
        
        # Training hyperparameters
        self._setup_training()
        
        # Inference settings
        self._setup_inference()
        
        # Validation
        self._validate_config()

    def _setup_environment(self):
        """Configure hardware settings."""
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.local_rank = int(os.getenv('LOCAL_RANK', 0))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        torch.backends.cudnn.benchmark = True  # Optimized CUDA performance

    def _setup_paths(self):
        """Set Kaggle data paths."""
        try:
            self.data_path = KaggleDatasets().get_gcs_path("waveform-inversion")
        except:
            self.data_path = "/kaggle/input/waveform-inversion"
        
        self.train_path = os.path.join(self.data_path, "train")
        self.test_path = os.path.join(self.data_path, "test")
        self.model_dir = "/kaggle/working/models"
        os.makedirs(self.model_dir, exist_ok=True)

    def _setup_physics(self):
        """Physics-based constraints."""
        self.velocity_range = (1480, 4520)  # Realistic P-wave velocity range (m/s)
        self.density = 2350  # Average crustal density (kg/m³)
        self.dt = 0.004  # Time sampling interval (s)
        self.dx = 10.2  # Spatial sampling interval (m)
        self.freq_range = (4, 28)  # Optimal seismic band (Hz)
        
        # Physics loss weights (fine-tuned)
        self.wave_eq_weight = 0.42  # Wave equation constraint
        self.snell_weight = 0.08  # Snell's law at interfaces
        self.energy_constraint = 0.05  # Energy conservation

    def _setup_model(self):
        """Neural network architecture."""
        self.backbone = "efficientnet_v2_m"  # Best speed/accuracy tradeoff
        self.pretrained = True
        self.encoder_channels = [24, 48, 96, 128]  # For U-Net
        self.decoder_channels = [128, 96, 48, 32]
        self.attention_heads = 8  # Channel attention
        self.dropout = 0.15  # Regularization
        self.activation = "gelu"  # Best for seismic data

    def _setup_training(self):
        """Training optimization."""
        self.batch_size = 128 if self.num_gpus >= 2 else 64
        self.batch_size_val = 192
        self.num_workers = 4 if self.num_gpus else 2
        self.epochs = 150  # Early stopping will handle this
        self.lr = 2.5e-4  # Optimal learning rate
        self.weight_decay = 1.2e-5  # L2 regularization
        self.grad_clip = 1.2  # Gradient clipping
        
        # Learning rate scheduling
        self.lr_schedule = {
            'warmup_epochs': 25,
            'peak_lr': 2.5e-4,
            'min_lr': 5e-7,
            'decay': 'cosine_annealing'
        }
        
        # Early stopping
        self.early_stopping = {
            'patience': 25,
            'delta': 0.0005,
            'min_epochs': 50
        }

    def _setup_inference(self):
        """Inference optimization."""
        self.test_time_flips = True  # Horizontal/vertical flips
        self.test_time_rotation = True  # 90°, 180°, 270° rotations
        self.test_time_scale = True  # Multi-scale inference
        self.scale_factors = [0.85, 0.9, 1.1, 1.15]  # Optimal scales
        self.ensemble_models = 5  # Number of models for ensemble
        self.output_scale = 1500.0  # Submission scaling
        self.output_offset = 3000.0

    def _validate_config(self):
        """Sanity checks."""
        assert os.path.exists(self.train_path), f"Train path not found: {self.train_path}"
        assert 0.3 <= self.wave_eq_weight <= 0.45, "Wave equation weight out of range"
        assert 0.05 <= self.snell_weight <= 0.15, "Snell's law weight out of range"

# Global configuration
cfg = CompetitionConfig()


%%writefile dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from scipy.signal import butter, filtfilt, hilbert
import pywt
from config import cfg

class SeismicPreprocessor:
    """Advanced seismic data preprocessing with physics-aware transformations."""
    
    @staticmethod
    def bandpass_filter(data, lowcut, highcut, dt, order=5):
        """Butterworth bandpass filter for seismic traces."""
        nyq = 0.5 / dt
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def wavelet_denoise(data, wavelet='sym6', level=4):
        """Wavelet-based noise reduction."""
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeffs = [pywt.threshold(c, threshold, 'soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)
    
    @staticmethod
    def envelope(data):
        """Compute signal envelope using Hilbert transform."""
        return np.abs(hilbert(data))
    
    @staticmethod
    def normalize(data):
        """Physics-aware normalization with outlier clipping."""
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        return np.clip(data, -3.5, 3.5)

class SeismicDataset(Dataset):
    """Optimized dataset with physics-compliant augmentations."""
    
    def __init__(self, mode="train"):
        self.mode = mode
        self.files = self._get_file_list()
        self.processor = SeismicPreprocessor()
        self.transform = self._get_transforms()
        
    def _get_file_list(self):
        """5-fold cross-validation split."""
        all_files = sorted([f for f in os.listdir(cfg.train_path) 
                          if f.endswith('.npy') and 'faulty' not in f])
        fold_size = len(all_files) // 5
        val_files = all_files[cfg.current_fold*fold_size:(cfg.current_fold+1)*fold_size]
        return {'train': [f for f in all_files if f not in val_files],
                'val': val_files}[self.mode]
    
    def _get_transforms(self):
        """Train/validation transformations."""
        if self.mode != "train":
            return transforms.Compose([
                transforms.Lambda(lambda x: self.processor.normalize(x))
            ])
            
        return transforms.Compose([
            transforms.Lambda(self._random_gain),
            transforms.Lambda(self._apply_bandpass),
            transforms.Lambda(self._apply_wavelet_denoise),
            transforms.Lambda(self._random_flip),
            transforms.Lambda(self._random_rotate),
            transforms.Lambda(self._random_scale),
            transforms.Lambda(self._add_noise),
            transforms.Lambda(lambda x: self.processor.normalize(x))
        ])
    
    def _apply_bandpass(self, x):
        """Frequency filtering."""
        return torch.stack([torch.from_numpy(
            self.processor.apply_bandpass(ch.numpy(), *cfg.freq_range, cfg.dt)
        ).float() for ch in x])
    
    def _apply_wavelet_denoise(self, x):
        """Wavelet denoising."""
        return torch.stack([torch.from_numpy(
            self.processor.wavelet_denoise(ch.numpy())
        ).float() for ch in x])
    
    def _random_gain(self, x):
        """Amplitude scaling augmentation."""
        if torch.rand(1) < 0.5:
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
            return x * gain
        return x
    
    def _random_flip(self, x):
        """Axis flipping preserving physics."""
        if torch.rand(1) < 0.5:
            return torch.flip(x, [-1])
        return x
    
    def _random_rotate(self, x):
        """Rotation augmentation."""
        if torch.rand(1) < 0.5:
            angle = torch.empty(1).uniform_(-20, 20).item()
            return transforms.functional.rotate(x, angle)
        return x
    
    def _random_scale(self, x):
        """Multi-scale augmentation."""
        if torch.rand(1) < 0.5:
            scale = torch.empty(1).uniform_(0.85, 1.15).item()
            return F.interpolate(x.unsqueeze(0), scale_factor=scale, 
                               mode='bilinear').squeeze(0)
        return x
    
    def _add_noise(self, x):
        """Realistic noise injection."""
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.03 * x.std()
            return x + noise
        return x
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = np.load(os.path.join(cfg.train_path, self.files[idx]))
            x = torch.from_numpy(data[:5]).float()  # Seismic channels
            y = torch.from_numpy(data[5]).float()   # Velocity labels
            
            if self.mode == "train":
                x = self.transform(x)
            else:
                x = self.processor.normalize(x)
                
            return x, y
            
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {str(e)}")
            return torch.zeros(5, 128, 128), torch.zeros(128, 128)

class TestDataset(Dataset):
    """Optimized test dataset loader."""
    
    def __init__(self):
        self.files = sorted([os.path.join(cfg.test_path, f) 
                           for f in os.listdir(cfg.test_path)
                           if f.endswith('.npy')])
        self.processor = SeismicPreprocessor()
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            x = torch.from_numpy(data).float()
            x = self.processor.normalize(x)
            fname = os.path.basename(self.files[idx]).split('.')[0]
            return x, fname
        except Exception as e:
            print(f"Error loading test file {self.files[idx]}: {str(e)}")
            return torch.zeros(5, 128, 128), "error"


%%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from config import cfg

class PhysicsGuidedLoss(nn.Module):
    """Advanced loss combining MAE with physics constraints."""
    
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    def wave_equation(self, v, seismic):
        """2D acoustic wave equation with density variation."""
        dv_dx = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * cfg.dx)
        dv_dz = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * cfg.dx)
        
        d2v_dx2 = (v[:, 2:, 1:-1] - 2*v[:, 1:-1, 1:-1] + v[:, :-2, 1:-1]) / (cfg.dx**2)
        d2v_dz2 = (v[:, 1:-1, 2:] - 2*v[:, 1:-1, 1:-1] + v[:, 1:-1, :-2]) / (cfg.dx**2)
        
        wave_term = (seismic[:, 2:, 1:-1] - 2*seismic[:, 1:-1, 1:-1] + seismic[:, :-2, 1:-1]) / (cfg.dt**2)
        velocity_term = (v[:, 1:-1, 1:-1]**2) * (d2v_dx2 + d2v_dz2)
        density_term = dv_dx**2 + dv_dz**2
        
        return self.mse(wave_term, velocity_term + density_term)
    
    def snells_law(self, v):
        """Snell's law constraint at layer interfaces."""
        dv_dx = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * cfg.dx)
        dv_dz = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * cfg.dx)
        return torch.mean((dv_dx / (dv_dz + 1e-6))**2)
    
    def energy_conservation(self, v, seismic):
        """Energy conservation constraint."""
        energy_in = torch.mean(seismic[:, :-1]**2)
        energy_out = torch.mean((v[:, 1:] - v[:, :-1])**2)
        return self.mse(energy_in, energy_out)
    
    def forward(self, pred, target, seismic):
        mae_loss = self.mae(pred, target)
        wave_loss = self.wave_equation(pred, seismic)
        snell_loss = self.snells_law(pred)
        energy_loss = self.energy_conservation(pred, seismic)
        
        return {
            'total': mae_loss + cfg.wave_eq_weight*wave_loss + cfg.snell_weight*snell_loss + cfg.energy_constraint*energy_loss,
            'mae': mae_loss,
            'wave': wave_loss,
            'snell': snell_loss,
            'energy': energy_loss
        }

class SeismicNet(nn.Module):
    """EfficientNetV2-based U-Net with physics-aware design."""
    
    def __init__(self):
        super().__init__()
        
        # Encoder with pretrained weights
        self.encoder = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=5,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(5, 5, 1),
            nn.Sigmoid())
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            self._make_decoder_block(cfg.encoder_channels[i] + cfg.decoder_channels[i-1],
                                   cfg.decoder_channels[i])
            for i in range(1, len(cfg.decoder_channels))])
        
        # Physics-constrained output head
        self.head = nn.Sequential(
            nn.Conv2d(cfg.decoder_channels[-1], 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1))
        
        self._init_weights()

    def _make_decoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Channel attention
        attn = self.channel_att(x)
        x = x * attn
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = features[-1]
        for i, block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = torch.cat([x, features[-2-i]], dim=1)
            x = block(x)
        
        # Scale to physical velocity range
        out = self.head(x).squeeze(1)
        return torch.sigmoid(out) * (cfg.velocity_range[1]-cfg.velocity_range[0]) + cfg.velocity_range[0]
    
    def predict_with_tta(self, x):
        """Enhanced test-time augmentation."""
        preds = [self(x)]
        
        # Flip augmentations
        if cfg.test_time_flips:
            for dims in [[-1], [-2], [-1, -2]]:
                preds.append(torch.flip(self(torch.flip(x, dims)), dims))
        
        # Rotation augmentations
        if cfg.test_time_rotation:
            for k in [1, 2, 3]:
                rotated = torch.rot90(x, k, [-2, -1])
                preds.append(torch.rot90(self(rotated), -k, [-2, -1]))
        
        # Scale augmentations
        if cfg.test_time_scale:
            for scale in cfg.scale_factors:
                scaled = F.interpolate(x, scale_factor=scale, mode='bilinear')
                pred = F.interpolate(self(scaled).unsqueeze(1), 
                                    size=x.shape[-2:], mode='bilinear').squeeze(1)
                preds.append(pred)
        
        return torch.mean(torch.stack(preds), dim=0)

class ModelEMA:
    """Improved Exponential Moving Average."""
    
    def __init__(self, model, decay, warmup):
        self.model = model
        self.decay = decay
        self.warmup = warmup
        self.shadow = {}
        self.backup = {}
        self.n_updates = 0
        
    def update(self, model):
        self.n_updates += 1
        decay = min(self.decay, (1 + self.n_updates) / (self.warmup + self.n_updates))
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] -= (1 - decay) * (self.shadow[name] - param.data)
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


%%writefile train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.cuda.amp as amp
from datetime import datetime
from model import SeismicNet, PhysicsGuidedLoss, ModelEMA
from dataset import SeismicDataset
from config import cfg
import os
import numpy as np

def setup_distributed():
    """Initialize distributed training."""
    if cfg.world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=cfg.world_size,
            rank=cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)

def cleanup_distributed():
    """Cleanup distributed resources."""
    if cfg.world_size > 1:
        dist.destroy_process_group()

def set_seed(seed):
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders():
    """Prepare train/validation dataloaders."""
    train_set = SeismicDataset(mode="train")
    val_set = SeismicDataset(mode="val")
    
    train_sampler = DistributedSampler(train_set) if cfg.world_size > 1 else None
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, scaler, criterion, ema, epoch):
    """Single training epoch."""
    model.train()
    total_loss = 0
    metrics = {'mae': 0, 'wave': 0, 'snell': 0, 'energy': 0}
    
    if cfg.world_size > 1:
        loader.sampler.set_epoch(epoch)
    
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp.autocast():
            pred = model(x)
            losses = criterion(pred, y, x[:, 0])  # First channel for physics
            
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += losses['total'].item()
        for k in metrics:
            metrics[k] += losses[k].item()
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v/len(loader) for k,v in metrics.items()}
    
    if cfg.local_rank == 0:
        print(f"\nTrain Epoch {epoch} | Loss: {avg_loss:.4f} | MAE: {avg_metrics['mae']:.4f}")
        print(f"Physics Terms - Wave: {avg_metrics['wave']:.4f} | Snell: {avg_metrics['snell']:.4f} | Energy: {avg_metrics['energy']:.4f}")
    
    return avg_loss

@torch.no_grad()
def validate(model, loader, criterion):
    """Validation loop."""
    model.eval()
    total_loss = 0
    metrics = {'mae': 0, 'wave': 0, 'snell': 0, 'energy': 0}
    
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        
        with amp.autocast():
            pred = model(x)
            losses = criterion(pred, y, x[:, 0])
        
        total_loss += losses['total'].item()
        for k in metrics:
            metrics[k] += losses[k].item()
    
    # Sync across GPUs
    if cfg.world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        for k in metrics:
            dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
        total_loss /= cfg.world_size
        for k in metrics:
            metrics[k] /= cfg.world_size
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v/len(loader) for k,v in metrics.items()}
    
    if cfg.local_rank == 0:
        print(f"\nValidation | Loss: {avg_loss:.4f} | MAE: {avg_metrics['mae']:.4f}")
        print(f"Physics Terms - Wave: {avg_metrics['wave']:.4f} | Snell: {avg_metrics['snell']:.4f} | Energy: {avg_metrics['energy']:.4f}")
    
    return avg_loss, avg_metrics

def save_checkpoint(model, optimizer, epoch, loss, metrics, is_best):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': vars(cfg)
    }
    
    filename = os.path.join(cfg.model_dir, f"model_fold{cfg.current_fold}.pth")
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(cfg.model_dir, f"best_model_fold{cfg.current_fold}.pth")
        torch.save(state, best_filename)

def main():
    """Main training function."""
    setup_distributed()
    set_seed(cfg.seed)
    
    if cfg.local_rank == 0:
        print("\n===== Starting Advanced Seismic FWI Training =====")
        print(f"Configuration:\n{'-'*30}")
        for k, v in vars(cfg).items():
            if not k.startswith('_'):
                print(f"{k}: {v}")
        print(f"{'-'*30}\n")
    
    # Initialize model and data
    train_loader, val_loader = get_dataloaders()
    model = SeismicNet().to(cfg.device)
    
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])
    
    # Optimizer and schedulers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay)
    
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.lr_schedule['warmup_epochs'],
        eta_min=cfg.lr_schedule['min_lr'])
    
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=cfg.local_rank == 0)
    
    criterion = PhysicsGuidedLoss()
    scaler = amp.GradScaler()
    ema = ModelEMA(model, cfg.ema_decay, cfg.ema_warmup) if cfg.local_rank == 0 else None
    
    best_loss = float('inf')
    no_improve = 0
    
    # Training loop
    for epoch in range(cfg.epochs):
        if cfg.local_rank == 0:
            print(f"\nEpoch {epoch+1}/{cfg.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, ema, epoch)
        
        # EMA evaluation
        if ema is not None:
            ema.apply(model.module if hasattr(model, 'module') else model)
        
        val_loss, val_metrics = validate(model, val_loader, criterion)
        
        # Restore original weights
        if ema is not None:
            ema.restore(model.module if hasattr(model, 'module') else model)
        
        # Update schedulers
        scheduler_plateau.step(val_loss)
        if epoch >= cfg.lr_schedule['warmup_epochs']:
            scheduler_cosine.step()
        
        # Save checkpoint
        if cfg.local_rank == 0:
            is_best = val_loss < best_loss - cfg.early_stopping['delta']
            if is_best:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            
            save_checkpoint(
                model.module if hasattr(model, 'module') else model,
                optimizer,
                epoch,
                val_loss,
                val_metrics,
                is_best)
            
            # Early stopping
            if no_improve >= cfg.early_stopping['patience'] and epoch >= cfg.early_stopping['min_epochs']:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    cleanup_distributed()

if __name__ == "__main__":
    main()


%%writefile inference.py
import torch
import numpy as np
from model import SeismicNet
from dataset import TestDataset
from config import cfg
import os
import pandas as pd
from tqdm import tqdm

def load_ensemble_models(model_paths):
    """Load ensemble of trained models."""
    models = []
    for path in model_paths:
        model = SeismicNet().to(cfg.device)
        state = torch.load(path, map_location=cfg.device)
        model.load_state_dict(state['model'])
        model.eval()
        models.append(model)
    return models

def run_tta_inference(models, loader):
    """Run inference with test-time augmentation."""
    predictions = []
    with torch.no_grad():
        for x, fnames in tqdm(loader, desc="Running Inference"):
            x = x.to(cfg.device, non_blocking=True)
            ensemble_preds = []
            
            for model in models:
                pred = model.predict_with_tta(x)
                ensemble_preds.append(pred)
            
            avg_pred = torch.mean(torch.stack(ensemble_preds), dim=0)
            
            for i in range(avg_pred.shape[0]):
                pred = avg_pred[i].cpu().numpy()
                pred = (pred * cfg.output_scale) + cfg.output_offset
                predictions.append([fnames[i]] + list(pred[::2]))  # Only odd columns
    
    return predictions

def create_submission(predictions):
    """Generate competition submission file."""
    sub_df = pd.DataFrame(
        predictions,
        columns=['oid_ypos'] + cfg.submission_cols)
    
    # Physical range validation
    for col in cfg.submission_cols:
        sub_df[col] = sub_df[col].clip(*cfg.velocity_range)
    
    sub_df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")
    return sub_df

def main():
    """Main inference function."""
    # Optimize inference settings
    torch.backends.cudnn.benchmark = True
    torch.set_flush_denormal(True)
    torch.set_num_threads(4)
    
    print("\n===== Starting Advanced Inference =====")
    
    # Load trained models
    model_paths = sorted([
        os.path.join(cfg.model_dir, f) 
        for f in os.listdir(cfg.model_dir) 
        if f.startswith("best_model_fold")])
    
    if not model_paths:
        raise ValueError("No trained models found for inference")
    
    models = load_ensemble_models(model_paths)
    print(f"Loaded ensemble of {len(models)} models")
    
    # Prepare test data
    test_set = TestDataset()
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.batch_size_val * 2,  # Larger batches for inference
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    
    # Run inference and create submission
    predictions = run_tta_inference(models, loader)
    submission = create_submission(predictions)
    
    # Final validation
    print("\nSubmission Summary:")
    print(submission.describe())
    print("\n===== Inference Completed Successfully =====")

if __name__ == "__main__":
    main()

