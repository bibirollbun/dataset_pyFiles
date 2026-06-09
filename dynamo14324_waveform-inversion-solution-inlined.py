# RETAINED FIX: Use a newer version of timm to avoid compatibility issues with Python 3.11+.
!pip install -q pytorch-lightning==2.0.0 torchmetrics==0.11.4 einops==0.6.1 timm==0.9.12 segmentation-models-pytorch albumentations

try:
    import pytorch_lightning
    print('PyTorch Lightning imported successfully.')
except ImportError:
    print('ERROR: Failed to import PyTorch Lightning.')

# --- Inlined Functions from kaggle_gm_automation.py (Retained for robustness) ---
import os, gc, logging, random, warnings
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
try:
    from packaging import version
except ImportError:
    pass
try:
    import importlib.metadata
except ImportError:
    pass
try:
    import pkg_resources
except ImportError:
    pass
try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning.pytorch as pl
    except ImportError:
        print("Warning: PyTorch Lightning not found")
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    pass
try:
    import psutil
except ImportError:
    pass
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InlinedFunctions")

# Helper functions for environment detection and optimization remain unchanged as they are robust.
def get_pytorch_lightning_version() -> str:
    try:
        import importlib.metadata
        try: return importlib.metadata.version("pytorch-lightning")
        except importlib.metadata.PackageNotFoundError: 
            try: return importlib.metadata.version("lightning")
            except: pass
    except: pass
    try:
        import pkg_resources
        try: return pkg_resources.get_distribution("pytorch-lightning").version
        except pkg_resources.DistributionNotFound: 
            try: return pkg_resources.get_distribution("lightning").version
            except: pass
    except: pass
    return "2.0.0"

def detect_device_type() -> str:
    if torch.cuda.is_available(): return "gpu"
    try:
        import torch_xla.core.xla_model as xm
        xm.xla_device()
        return "tpu"
    except: return "cpu"

def set_reproducibility(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, gc, time, glob, logging
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
from einops import rearrange
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WaveformInversion")

# Set random seed for reproducibility
set_reproducibility(42)
class CFG:
    # Competition and data paths
    IN_KAGGLE = os.path.exists('/kaggle/input')
    COMP_PATH = '/kaggle/input/waveform-inversion' if IN_KAGGLE else './input/waveform-inversion'
    OUTPUT_PATH = '/kaggle/working/' if IN_KAGGLE else './'
    TRAIN_PATH = f"{COMP_PATH}/train_samples"
    TEST_PATH = f"{COMP_PATH}/test"
    DATASET_FAMILIES = ['FlatVel_A', 'Fault', 'Style']
    
    # Model and training parameters
    SEED = 42
    DEVICE = detect_device_type()
    PRECISION = "16-mixed" if DEVICE == "gpu" else "bf16-true" if DEVICE == "tpu" else "32-true"
    N_SPLITS = 5 # Initial desired number of splits
    EPOCHS = 40
    BATCH_SIZE = 16
    LR = 1e-3
    WEIGHT_DECAY = 1e-6
    T_0 = 10
    
    # Define which models to train for the ensemble
    MODELS_TO_TRAIN = ["ConvNeXtUNet", "InversionNet"]

print(f"Running in {'Kaggle' if CFG.IN_KAGGLE else 'Local'} environment.")
print(f"Using device: {CFG.DEVICE} with precision: {CFG.PRECISION}")
def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        ToTensorV2(),
    ])
    return train_transform, val_transform

class WaveformDataset(Dataset):
    def __init__(self, data_files, model_files=None, transform=None, is_test=False):
        self.data_files = data_files
        self.model_files = model_files
        self.transform = transform
        self.is_test = is_test
        self.data_mmap = [np.load(f, mmap_mode='r') for f in data_files]
        if not self.is_test:
            self.model_mmap = [np.load(f, mmap_mode='r') for f in model_files]
        self.file_sample_counts = [len(d) for d in self.data_mmap]
        self.cumulative_samples = np.cumsum([0] + self.file_sample_counts)
        self.total_samples = self.cumulative_samples[-1]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_samples, idx + 1) - 1
        sample_idx = idx - self.cumulative_samples[file_idx]
        seismic = self.data_mmap[file_idx][sample_idx].astype(np.float32)
        seismic = (seismic - seismic.mean()) / (seismic.std() + 1e-8)
        seismic = rearrange(seismic, 's t r -> t r s')
        if self.is_test:
            augmented = self.transform(image=seismic)
            return {'seismic': augmented['image'].unsqueeze(0)}
        else:
            velocity = self.model_mmap[file_idx][sample_idx].astype(np.float32)
            if self.transform:
                augmented = self.transform(image=seismic, mask=velocity)
                seismic = augmented['image']
                velocity = augmented['mask']
            seismic = seismic.unsqueeze(0)
            velocity = velocity.unsqueeze(0)
            return {'seismic': seismic, 'velocity': velocity}

def find_data_files():
    data_files, model_files, family_keys = [], [], []
    for fam_idx, family in enumerate(CFG.DATASET_FAMILIES):
        family_path = f"{CFG.TRAIN_PATH}/{family}"
        if not os.path.exists(family_path):
            continue
        seis_pattern = f"{family_path}/**/data*.npy" if family != 'Fault' else f"{family_path}/seis*.npy"
        current_data_files = sorted(glob.glob(seis_pattern, recursive=True))
        for data_file in current_data_files:
            model_file = data_file.replace('data', 'model').replace('seis', 'vel')
            if os.path.exists(model_file):
                data_files.append(data_file)
                model_files.append(model_file)
                family_keys.append(fam_idx)
    logger.info(f"Found {len(data_files)} training data files with matching velocity maps.")
    return np.array(data_files), np.array(model_files), np.array(family_keys)

train_data_files, train_model_files, train_family_keys = find_data_files()

# FIX: Dynamically adjust the number of splits based on the number of available data files.
# REASON: The error log showed a `ValueError` because N_SPLITS (5) was greater than the number of samples found (2).
# This fix makes the code robust by ensuring N_SPLITS is never larger than the sample count, preventing the crash.
n_samples = len(train_data_files)
if n_samples > 0 and n_samples < CFG.N_SPLITS:
    logger.warning(f"Number of samples ({n_samples}) is less than N_SPLITS ({CFG.N_SPLITS}). Adjusting N_SPLITS to {n_samples}.")
    CFG.N_SPLITS = n_samples
# InversionNet (custom U-Net like model)
class InversionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x); p1 = self.pool(e1)
        e2 = self.encoder2(p1); p2 = self.pool(e2)
        e3 = self.encoder3(p2); p3 = self.pool(e3)
        e4 = self.encoder4(p3); p4 = self.pool(e4)
        b = self.bottleneck(p4)
        d4 = self.decoder4(torch.cat((self.upconv4(b), e4), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim=1))
        return self.final_conv(d1)

# ConvNeXt U-Net model
class ConvNeXtUNet(nn.Module):
    def __init__(self, backbone='convnext_tiny', pretrained=True, in_channels=1, out_channels=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
            encoder_depth=4,
        )
        self.model.encoder.set_swish(memory_efficient=True)

    def forward(self, x):
        if self.model.encoder.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.Tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.get_device() or 'cpu')
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class PhysicsLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = self.kernel.to(device)

    def forward(self, velocity_map):
        laplacian = F.conv2d(velocity_map, self.kernel.to(velocity_map.device), padding=1)
        return F.mse_loss(laplacian, torch.zeros_like(laplacian))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.05, device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.physics_loss = PhysicsLoss(device=device)

    def forward(self, y_pred, y_true):
        l1 = self.l1_loss(y_pred, y_true)
        ssim = self.ssim_loss(y_pred, y_true)
        physics = self.physics_loss(y_pred)
        combined = self.alpha * ssim + (1 - self.alpha) * l1 + self.beta * physics
        return combined
class WaveformInversionModule(pl.LightningModule):
    def __init__(self, model_name="ConvNeXtUNet", lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        if model_name == "ConvNeXtUNet":
            self.model = ConvNeXtUNet()
        elif model_name == "InversionNet":
            self.model = InversionNet()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.loss_fn = CombinedLoss(alpha=0.85, beta=0.05, device=self.device)
    
    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        self.loss_fn.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['seismic'], batch['velocity']
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['seismic'], batch['velocity']
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, eta_min=1e-6)
        return [optimizer], [scheduler]
def train_model_loop(model_name, train_loader, val_loader=None, fold=None):
    """A helper function to encapsulate the training logic for a single model."""
    model = WaveformInversionModule(model_name=model_name, lr=CFG.LR)
    
    # Define callbacks
    monitor_metric = 'val_loss' if val_loader else 'train_loss'
    filename_suffix = f'fold{fold}-best' if fold is not None else 'best'
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=CFG.OUTPUT_PATH,
        filename=f'{model_name}-{filename_suffix}',
        save_top_k=1,
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor=monitor_metric, patience=7, verbose=True, mode='min')
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=CFG.DEVICE, devices=1, precision=CFG.PRECISION,
        max_epochs=CFG.EPOCHS, callbacks=[checkpoint_callback, early_stop_callback],
        logger=True
    )
    
    # Start training
    trainer.fit(model, train_loader, val_loader)
    
    # Clean up memory
    del model, trainer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

train_transforms, val_transforms = get_transforms()

if n_samples == 0:
    logger.error("No training data files found. Skipping training and prediction.")
elif CFG.N_SPLITS >= 2:
    logger.info(f"Starting {CFG.N_SPLITS}-fold cross-validation.")
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
    for model_name in CFG.MODELS_TO_TRAIN:
        logger.info(f"\n{'='*50}\nTRAINING MODEL: {model_name}\n{'='*50}")
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data_files, train_family_keys)):
            logger.info(f"\n--- Fold {fold+1}/{CFG.N_SPLITS} ---")
            train_dataset = WaveformDataset(train_data_files[train_idx], train_model_files[train_idx], transform=train_transforms)
            val_dataset = WaveformDataset(train_data_files[val_idx], train_model_files[val_idx], transform=val_transforms)
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)
            train_model_loop(model_name, train_loader, val_loader, fold=fold)
else: # Case where N_SPLITS is 1 (i.e., only 1 sample found)
    logger.warning("Cannot perform cross-validation with < 2 splits. Training on the full dataset.")
    full_dataset = WaveformDataset(train_data_files, train_model_files, transform=train_transforms)
    full_loader = DataLoader(full_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)
    for model_name in CFG.MODELS_TO_TRAIN:
        logger.info(f"\n{'='*50}\nTRAINING MODEL ON FULL DATASET: {model_name}\n{'='*50}")
        train_model_loop(model_name, full_loader, val_loader=None, fold=None)
class TestDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        seismic = np.load(file_path).astype(np.float32)
        seismic = (seismic - seismic.mean()) / (seismic.std() + 1e-8)
        seismic = rearrange(seismic, 's t r -> t r s')
        if self.transform:
            seismic = self.transform(image=seismic)['image']
        return {'seismic': seismic.unsqueeze(0), 'file_id': os.path.basename(file_path).split('.')[0]}

model_paths = glob.glob(f"{CFG.OUTPUT_PATH}/*.ckpt")
if not model_paths:
    logger.error("No trained models found. Cannot create submission file.")
    # Create a dummy submission file if required by the competition
    sample_submission = pd.read_csv(f'{CFG.COMP_PATH}/sample_submission.csv')
    sample_submission.to_csv(f'{CFG.OUTPUT_PATH}/submission.csv', index=False)
else:
    logger.info(f"Found {len(model_paths)} models for ensembling.")
    test_files = sorted(glob.glob(f"{CFG.TEST_PATH}/*.npy"))
    _, test_transforms = get_transforms()
    test_dataset = TestDataset(test_files, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    all_predictions = {}
    device = torch.device('cuda' if CFG.DEVICE == 'gpu' else 'cpu')

    for model_path in tqdm(model_paths, desc="Ensembling Models"):
        model_name = "ConvNeXtUNet" if "ConvNeXtUNet" in model_path else "InversionNet"
        model = WaveformInversionModule.load_from_checkpoint(model_path, model_name=model_name)
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                file_id = batch['file_id'][0]
                x = batch['seismic'].to(device)
                pred_orig = model(x).squeeze().cpu().numpy()
                pred_flipped = model(torch.flip(x, dims=[-1])).squeeze().cpu().numpy()
                avg_pred = (pred_orig + np.fliplr(pred_flipped)) / 2.0
                if file_id not in all_predictions:
                    all_predictions[file_id] = []
                all_predictions[file_id].append(avg_pred)
    
    submission_data = []
    for file_id, preds in tqdm(all_predictions.items(), desc="Averaging Predictions"):
        final_pred = np.mean(preds, axis=0)
        for r in range(final_pred.shape[0]):
            for c in range(final_pred.shape[1]):
                submission_data.append([f"{file_id}_{r}_{c}", final_pred[r, c]])
    
    submission_df = pd.DataFrame(submission_data, columns=['id', 'velocity'])
    submission_df.to_csv(f'{CFG.OUTPUT_PATH}/submission.csv', index=False)
    logger.info("Submission file created successfully!")
    print(submission_df.head())


