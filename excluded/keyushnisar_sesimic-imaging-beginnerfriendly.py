import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict
import gc
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    input_channels: int = 5
    output_height: int = 70
    output_width: int = 70
    feature_channels: List[int] = None
    dropout_rate: float = 0.25
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    n_epochs: int = 50
    early_stopping_patience: int = 10
    train_data_path: str = '/kaggle/input/waveform-inversion/train_samples'
    test_data_path: str = '/kaggle/input/waveform-inversion/test'
    output_file: str = 'submission.csv'
    model_save_path: str = 'seismic_wavenet.pth'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0

config = Config(feature_channels=[64, 128, 256, 512])

def setup_environment():
    torch.manual_seed(42)
    np.random.seed(42)
    if config.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA enabled with cuDNN benchmark")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class SeismicDataset(Dataset):
    def __init__(self, input_files: List[Path], output_files: List[Path]):
        self.input_files = input_files
        self.output_files = output_files
        self.n_examples_per_file = 500

    def __len__(self) -> int:
        return len(self.input_files) * self.n_examples_per_file

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file
        try:
            X = np.load(self.input_files[file_idx], mmap_mode='r')
            y = np.load(self.output_files[file_idx], mmap_mode='r')
            X_sample = torch.from_numpy(X[sample_idx].copy()).float()
            y_sample = torch.from_numpy(y[sample_idx].copy()).float()
            return X_sample, y_sample
        except Exception as e:
            logger.error(f"Error loading index {idx}: {e}")
            raise
        finally:
            gc.collect()

class TestDataset(Dataset):
    def __init__(self, test_files: List[Path]):
        self.test_files = test_files

    def __len__(self) -> int:
        return len(self.test_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        try:
            data = np.load(self.test_files[idx])
            tensor_data = torch.from_numpy(data).float()
            return tensor_data, self.test_files[idx].stem
        except Exception as e:
            logger.error(f"Error loading test file {self.test_files[idx]}: {e}")
            raise

class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class ResidualLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels)
        self.conv2 = ConvLayer(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))

class SeismicWaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList()
        in_channels = config.input_channels
        for out_channels in config.feature_channels:
            self.encoder.append(nn.Sequential(
                ConvLayer(in_channels, out_channels),
                ResidualLayer(out_channels),
                nn.MaxPool2d(2)
            ))
            in_channels = out_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(config.feature_channels[-1], config.feature_channels[-2], kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualLayer(config.feature_channels[-2]),
            nn.Conv2d(config.feature_channels[-2], config.feature_channels[-3], kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualLayer(config.feature_channels[-3]),
            nn.Conv2d(config.feature_channels[-3], 1, kernel_size=1)
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        x = self.decoder(x)
        x = self.dropout(x)
        return x * 1000 + 1500

def multi_scale_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    ssim_val = 0
    for scale in [1, 0.5, 0.25]:
        if scale != 1:
            pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=True)
            target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
            pred_np_scaled = pred_scaled.detach().cpu().numpy().squeeze()
            target_np_scaled = target_scaled.detach().cpu().numpy().squeeze()
        else:
            pred_np_scaled, target_np_scaled = pred_np, target_np
        ssim_val += ssim(pred_np_scaled, target_np_scaled, data_range=target_np_scaled.max() - target_np_scaled.min())
    return torch.tensor(1 - ssim_val / 3, device=pred.device)

def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1_loss = F.l1_loss(pred, target)
    ssim_loss = multi_scale_ssim(pred, target)
    return l1_loss + 0.5 * ssim_loss

def get_data_loaders():
    input_files = [f for f in Path(config.train_data_path).rglob('*.npy') if 'seis' in f.stem or 'data' in f.stem]
    output_files = [Path(str(f).replace('seis', 'vel').replace('data', 'model')) for f in input_files]
    if not all(f.exists() for f in output_files):
        missing = [str(f) for f in output_files if not f.exists()]
        logger.error(f"Missing output files: {missing[:5]}")
        raise FileNotFoundError("Missing output files")
    n_train = int(0.8 * len(input_files))
    train_inputs, valid_inputs = input_files[:n_train], input_files[n_train:]
    train_outputs, valid_outputs = output_files[:n_train], output_files[n_train:]
    train_dataset = SeismicDataset(train_inputs, train_outputs)
    valid_dataset = SeismicDataset(valid_inputs, valid_outputs)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    return train_loader, valid_loader

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler) -> float:
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        optimizer.zero_grad()
        with autocast(enabled=config.use_mixed_precision):
            outputs = torch.utils.checkpoint.checkpoint_sequential(model.encoder, len(model.encoder), inputs)
            outputs = model.decoder(outputs)
            loss = compute_loss(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            with autocast(enabled=config.use_mixed_precision):
                outputs = model(inputs)
                loss = compute_loss(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    scaler = GradScaler(enabled=config.use_mixed_precision)
    history = {'train_loss': [], 'valid_loss': [], 'lr': []}
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, config.n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        valid_loss = validate_epoch(model, valid_loader)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, LR={history['lr'][-1]:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.early_stopping_patience:
            logger.info("Early stopping triggered")
            break
        gc.collect()
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load(config.model_save_path))
    return history

def predict(model: nn.Module, test_files: List[Path]):
    model.eval()
    test_dataset = TestDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, num_workers=config.num_workers, pin_memory=True)
    results = []
    with torch.no_grad():
        for inputs, oids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(config.device)
            with autocast(enabled=config.use_mixed_precision):
                outputs = model(inputs)
            outputs = outputs.cpu().numpy().squeeze(axis=1)
            for y_pred, oid in zip(outputs, oids):
                for y_pos in range(config.output_height):
                    row = {'oid_ypos': f"{oid}_y_{y_pos}"}
                    for x_pos in range(1, config.output_width, 2):
                        row[f"x_{x_pos}"] = y_pred[y_pos, x_pos]
                    results.append(row)
    df = pd.DataFrame(results)
    cols = ['oid_ypos'] + [f'x_{i}' for i in range(1, config.output_width, 2)]
    df = df[cols]
    df.to_csv(config.output_file, index=False, float_format='%.4f')
    logger.info(f"Predictions saved to {config.output_file}")

def main():
    setup_environment()
    train_loader, valid_loader = get_data_loaders()
    model = SeismicWaveNet().to(config.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train_model(model, train_loader, valid_loader)
    test_files = list(Path(config.test_data_path).glob('*.npy'))
    predict(model, test_files)
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/kaggle/input/wavenet/wavenet.png')
plt.imshow(img)
plt.axis('off')  
plt.show()




