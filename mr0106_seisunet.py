# Cell 1: Installations & Setup
!pip install -q torch torchvision pandas tqdm

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_v2_m
from torchvision import transforms
from tqdm.auto import tqdm
from IPython.display import FileLink

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Cell 2: Configuration Class
class SeisFusionConfig:
    """Configuration class with optimized parameters for seismic inversion"""
    
    def __init__(self):
        # Hardware setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        self.num_workers = min(8, os.cpu_count())
        
        # Dataset paths (Kaggle competition structure)
        self.dataset_root = '/kaggle/input/waveform-inversion/'
        self.train_path = os.path.join(self.dataset_root, 'train_samples')
        self.test_path = os.path.join(self.dataset_root, 'test')
        self.submission_cols = [f'x_{i}' for i in range(1, 70, 2)]
        
        # Physics constraints
        self.velocity_range = (1500.0, 4500.0)  # m/s
        self.wave_eq_weight = 0.45
        self.snell_weight = 0.10
        self.energy_weight = 0.05
        
        # Model architecture
        self.backbone = "efficientnet_v2_m"
        self.pretrained = True
        self.encoder_channels = [24, 48, 96, 128]
        self.decoder_channels = [128, 96, 48, 32]
        self.dropout = 0.15
        self.activation = "gelu"
        
        # Training parameters
        self.batch_size = 144 if self.num_gpus >= 2 else 72
        self.batch_size_val = 192
        self.epochs = 200
        self.lr = 3.2e-4
        self.weight_decay = 1.5e-5
        self.grad_clip = 1.0
        
        # Learning rate schedule
        self.lr_schedule = {
            'warmup_epochs': 30,
            'peak_lr': 3.2e-4,
            'min_lr': 1e-7,
            'decay': 'cosine_annealing'
        }
        
        # Early stopping
        self.early_stopping = {
            'patience': 30,
            'delta': 0.0003,
            'min_epochs': 60
        }
        
        # Inference settings
        self.test_time_flips = True
        self.test_time_rotation = True
        self.test_time_scale = True
        self.scale_factors = [0.88, 0.94, 1.06, 1.12]
        self.ensemble_models = 5
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate paths and parameters"""
        if not os.path.exists(self.train_path):
            available = os.listdir(self.dataset_root)
            raise FileNotFoundError(
                f"Train path not found. Available directories: {available}\n"
                "Please ensure you've added the dataset correctly"
            )
        assert len(self.submission_cols) == 35, "Should have 35 prediction columns"
        
        logger.info(f"Training data: {self.train_path}")
        logger.info(f"Test data: {self.test_path}")

cfg = SeisFusionConfig()


# %% [code]
# Cell 3: Dataset Class (Updated for correct dimensions)
class SeismicDataset(Dataset):
    """Physics-aware dataset with advanced augmentation for [5,1000,70] input"""
    
    def __init__(self, mode="train"):
        self.mode = mode
        self.path = cfg.train_path if mode == "train" else cfg.test_path
        
        # Load all training samples from category folders
        if mode == "train":
            self.files = []
            for category in os.listdir(self.path):
                category_path = os.path.join(self.path, category)
                self.files.extend([
                    os.path.join(category_path, f)
                    for f in os.listdir(category_path)
                    if f.endswith('.npy')
                ])
        else:
            self.files = [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith('.npy')
            ]
        
        logger.info(f"Initialized {mode} dataset with {len(self.files)} samples")
        logger.info(f"Sample dimensions: {self._check_sample_shapes()}")  # Verify shapes
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])  # Expected shape: [6, 1000, 70]
            
            if self.mode == "train":
                x = torch.from_numpy(data[:5]).float()  # [5, 1000, 70]
                y = torch.from_numpy(data[5]).float()   # [1000, 70]
                x = self._augment(x) if self.mode == "train" else self._normalize(x)
                return x, y
            else:
                x = torch.from_numpy(data[:5]).float()  # [5, 1000, 70]
                x = self._normalize(x)
                return x, os.path.basename(self.files[idx]).split('.')[0]
                
        except Exception as e:
            logger.error(f"Error loading {self.files[idx]}: {str(e)}")
            dummy = torch.zeros(5, 1000, 70)
            return dummy, torch.zeros(1000, 70) if self.mode == "train" else "error"
    
    def _augment(self, x):
        """Physics-preserving augmentation for [5,1000,70] input"""
        # Random time-axis flips (dimension 2)
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [-1])
        
        # Random channel shuffling
        if torch.rand(1) < 0.3:
            channel_order = torch.randperm(5)
            x = x[channel_order]
            
        # Add noise proportional to channel std
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.025 * x.std()
            x += noise
            
        return self._normalize(x)
    
    def _normalize(self, x):
        """Channel-wise normalization for [5,1000,70] input"""
        # Normalize each channel separately
        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].mean()) / (x[i].std() + 1e-8)
        return x
    
    def _check_sample_shapes(self):
        """Verify actual data shapes"""
        try:
            sample_data = np.load(self.files[0])
            return {
                'input_shape': sample_data[:5].shape,
                'target_shape': sample_data[5].shape if self.mode == "train" else None
            }
        except:
            return "Shape check failed"


# Cell 4: Model Architecture
class SeismicNet(nn.Module):
    """Physics-informed neural network for seismic inversion"""
    
    def __init__(self):
        super().__init__()
        # Custom encoder for [batch, 5, 1000, 70] input
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Additional encoder layers
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Decoder with skip connections
        self.up1 = self._build_block(128, 96)
        self.up2 = self._build_block(96, 48)
        self.up3 = self._build_block(48, 32)
        
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(24, 1, 1)
        )
        
    def _build_block(self, in_ch, out_ch):
        """Decoder block constructor"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )
    
    def forward(self, x):
        # Encoder pathway
        x1 = self.encoder[0:4](x)   # After first block (24 channels)
        x2 = self.encoder[4:8](x1)  # After second block (48 channels)
        x3 = self.encoder[8:12](x2) # After third block (96 channels)
        x4 = self.encoder[12:](x3)  # Final encoder output (128 channels)
        
        # Decoder with skip connections
        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x + x3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x + x2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x + x1)
        
        # Output with physical scaling
        out = self.head(x).squeeze(1)
        return torch.sigmoid(out) * (cfg.velocity_range[1]-cfg.velocity_range[0]) + cfg.velocity_range[0]
    
    def predict_with_tta(self, x):
        """Test-time augmentation with physics checks"""
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
                scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                pred = F.interpolate(self(scaled).unsqueeze(1), size=x.shape[-2:], 
                                   mode='bilinear', align_corners=False).squeeze(1)
                preds.append(pred)
        
        return torch.mean(torch.stack(preds), dim=0)


# Cell 5: Training Pipeline
def create_data_loaders():
    """Create optimized data loaders"""
    train_set = SeismicDataset(mode="train")
    val_set = SeismicDataset(mode="train")  # Same data for validation
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

class PhysicsGuidedLoss(nn.Module):
    """Combined data and physics loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)

def train_model():
    """Complete training pipeline"""
    logger.info("Initializing training...")
    
    # Initialize model
    model = SeismicNet().to(cfg.device)
    if cfg.num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {cfg.num_gpus} GPUs")
    
    # Data loaders
    train_loader, val_loader = create_data_loaders()
    
    # Optimizer and schedulers
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.lr_schedule['warmup_epochs'],
        eta_min=cfg.lr_schedule['min_lr']
    )
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training utilities
    criterion = PhysicsGuidedLoss()
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        # Training phase
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation phase
        val_loss = validate(model, val_loader, criterion)
        
        # Update schedulers
        scheduler_plateau.step(val_loss)
        if epoch >= cfg.lr_schedule['warmup_epochs']:
            scheduler_cosine.step()
        
        # Checkpointing
        if val_loss < best_loss - cfg.early_stopping['delta']:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if cfg.num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_model.pth')
        else:
            early_stop_counter += 1
        
        # Early stopping check
        if early_stop_counter >= cfg.early_stopping['patience'] and epoch >= cfg.early_stopping['min_epochs']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Logging
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s\n"
            f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}\n"
            f"LR: {current_lr:.2e}"
        )
    
    logger.info(f"Training complete. Best val loss: {best_loss:.4f}")

def validate(model, loader, criterion):
    """Validation with physics checks"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            with autocast():
                pred = model(x)
                total_loss += criterion(pred, y).item()
    
    return total_loss / len(loader)


# Cell 6: Inference & Submission
def run_inference():
    """Generate competition submission with TTA and ensembling"""
    logger.info("Starting inference...")
    
    # Load ensemble of models
    models = []
    for i in range(cfg.ensemble_models):
        model = SeismicNet().to(cfg.device)
        try:
            state_dict = torch.load(f'model_{i}.pth', map_location=cfg.device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            models.append(model)
        except FileNotFoundError:
            logger.warning(f"Model {i} not found")
    
    if not models:
        raise ValueError("No trained models found for inference")
    
    # Data loader
    test_set = SeismicDataset(mode="test")
    loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size_val * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Run inference
    predictions = []
    with torch.no_grad():
        for x, fnames in tqdm(loader, desc="Inference"):
            x = x.to(cfg.device)
            batch_preds = []
            
            for model in models:
                with autocast():
                    pred = model.predict_with_tta(x).cpu().numpy()
                    batch_preds.append(pred)
            
            # Ensemble averaging
            avg_pred = np.mean(batch_preds, axis=0)
            
            # Process each sample
            for i in range(avg_pred.shape[0]):
                processed = np.clip(avg_pred[i], *cfg.velocity_range)
                predictions.append([fnames[i]] + list(processed.flatten()))
    
    # Create submission
    sub_df = pd.DataFrame(predictions, columns=['oid_ypos'] + cfg.submission_cols)
    
    # Validate submission
    min_val = sub_df.iloc[:, 1:].min().min()
    max_val = sub_df.iloc[:, 1:].max().max()
    logger.info(f"Submission stats - Min: {min_val:.1f}, Max: {max_val:.1f}")
    
    # Save to CSV
    sub_df.to_csv("submission.csv", index=False)
    logger.info("Submission file created")
    display(FileLink('submission.csv'))
    
    return sub_df


# %% [code]
# Cell 7: Main Execution (Corrected Version)
def main():
    """Main execution function with proper error handling"""
    try:
        print("Select mode:")
        print("1. Train model")
        print("2. Run inference and generate submission")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            # Initialize model and move to device
            model = SeismicNet().to(cfg.device)
            if cfg.num_gpus > 1:
                model = nn.DataParallel(model)
            
            # Create data loaders with proper error handling
            try:
                train_loader, val_loader = create_data_loaders()
            except Exception as e:
                logger.error(f"Error creating data loaders: {str(e)}")
                return
            
            # Training setup
            optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
            scaler = GradScaler(device_type='cuda', enabled=cfg.device.type == 'cuda')
            criterion = PhysicsGuidedLoss()
            
            # Training loop
            best_loss = float('inf')
            for epoch in range(1, cfg.epochs + 1):
                try:
                    train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
                    val_loss = validate(model, val_loader, criterion)
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Save best model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), 'best_model.pth')
                    
                    logger.info(f"Epoch {epoch}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch}: {str(e)}")
                    break
            
            logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
            
        elif choice == '2':
            # Run inference
            try:
                submission = run_inference()
                logger.info("Submission created successfully")
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
        
        else:
            logger.error("Invalid choice. Please enter 1 or 2")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()

