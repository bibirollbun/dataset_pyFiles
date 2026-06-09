import os
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torchvision.transforms as transforms
import warnings
import zipfile
import shutil
import json
import subprocess

warnings.filterwarnings("ignore")

# 配置类
class Config:
    DEBUG_MODE = False
    OUTPUT_DIR = '/kaggle/working'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    FS = 32000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224, 224)
    N_MAX = 50 if DEBUG_MODE else None

config = Config()

class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2
    OUTPUT_DIR = '/kaggle/working'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    model_name = 'vit_b_16'
    pretrained = True
    in_channels = 3
    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224, 224)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 16
    criterion = 'BCEWithLogitsLoss'
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]
    optimizer = 'AdamW'
    lr = 1e-4
    weight_decay = 1e-5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs
    aug_prob = 0.5
    mixup_alpha = 0.5

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()

# 调试信息
print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
print(f"Max samples to process: {config.N_MAX if config.N_MAX is not None else 'ALL'}")

# 加载数据
print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

print("Loading training metadata...")
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')
label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')
working_df = train_df[['primary_label', 'rating', 'filename']].copy()
working_df['target'] = working_df.primary_label.map(label2id)
working_df['filepath'] = config.DATA_ROOT + '/train_audio/' + working_df.filename
working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
total_samples = min(len(working_df), config.N_MAX or len(working_df))
print(f'Total samples to process: {total_samples} out of {len(working_df)} available')
print(f'Samples by class:')
print(working_df['class'].value_counts())

# Mel 频谱图生成函数
def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX, power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    mel_spec_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0).repeat(3, 1, 1)
    vit_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return vit_transforms(mel_spec_tensor).numpy()

def process_audio_file(audio_path, cfg):
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            audio_data = np.concatenate([audio_data] * n_copy) if n_copy > 1 else audio_data
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]
        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, (0, target_samples - len(center_audio)), mode='constant')
        return audio2melspec(center_audio, cfg).astype(np.float32)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# 分批生成与压缩
def generate_and_zip_spectrograms(df, cfg, batch_size=50, max_zip_size_mb=500):
    print("开始分批生成并压缩 Mel 频谱图...")
    start_time = time.time()
    output_dir = os.path.join(cfg.OUTPUT_DIR, 'spectrograms_batches')
    os.makedirs(output_dir, exist_ok=True)
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"总批次: {total_batches}")
    
    for batch_idx in range(total_batches):
        total, used, free = shutil.disk_usage(cfg.OUTPUT_DIR)
        free_gb = free / (1024 ** 3)
        print(f"剩余空间: {free_gb:.2f} GB")
        if free_gb < 1:
            print("磁盘空间不足，停止处理！")
            break
            
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        all_bird_data = {}
        all_labels = {}
        
        for i, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"批次 {batch_idx+1}/{total_batches}"):
            samplename = row['samplename']
            filepath = row['filepath']
            label = row['target']
            mel_spec = process_audio_file(filepath, cfg)
            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec
                all_labels[samplename] = label
        
        if not all_bird_data:
            continue
        
        npz_path = os.path.join(output_dir, f'batch_{batch_idx}.npz')
        batch_keys = list(all_bird_data.keys())
        batch_spectrograms = np.array(list(all_bird_data.values()))
        batch_labels = np.array([all_labels[key] for key in batch_keys], dtype=np.int64)
        np.savez(npz_path, spectrograms=batch_spectrograms, keys=batch_keys, labels=batch_labels)
        
        zip_path = os.path.join(cfg.OUTPUT_DIR, f'batch_{batch_idx}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(npz_path, os.path.basename(npz_path))
        
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"批次 {batch_idx+1} 完成，ZIP 大小: {zip_size_mb:.2f} MB")
        
        total, used, free = shutil.disk_usage(cfg.OUTPUT_DIR)
        free_gb = free / (1024 ** 3)
        if free_gb < 0.5:
            print(f"警告：剩余空间仅 {free_gb:.2f} GB，建议停止处理！")
            break
        
        if zip_size_mb > max_zip_size_mb:
            print(f"警告：批次 {batch_idx+1} 的 ZIP 文件 ({zip_size_mb:.2f} MB) 超过 {max_zip_size_mb} MB，建议减小 batch_size。")
        
        os.remove(npz_path)
    
    end_time = time.time()
    print(f"所有批次处理完成，总耗时 {end_time - start_time:.2f} 秒")


# 执行
batch_size = 20
max_zip_size_mb = 500
generate_and_zip_spectrograms(working_df, cfg, batch_size=batch_size, max_zip_size_mb=max_zip_size_mb)




import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
import pandas as pd
import glob
from torch.amp import autocast, GradScaler
import logging
import sys

# 设置日志
logging.basicConfig(filename='/kaggle/working/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 将 print 输出重定向到文件
class Logger:
    def __init__(self, filename='/kaggle/working/output.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# 配置类
class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2
    OUTPUT_DIR = '/kaggle/working'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    model_name = 'vit_b_16'
    pretrained = True
    in_channels = 3
    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224, 224)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5  # 减少 epoch 数，缩短训练时间
    batch_size = 256
    criterion = 'CrossEntropyLoss'
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]
    optimizer = 'AdamW'
    lr = 1e-4
    weight_decay = 1e-5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs
    aug_prob = 0.5
    mixup_alpha = 0.5
    save_per_epoch = False
    early_stopping_patience = 2
    checkpoint_dir = '/kaggle/working/checkpoints'

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 3
            self.selected_folds = [0]

cfg = CFG()
cfg.update_debug_settings()

os.makedirs(cfg.checkpoint_dir, exist_ok=True)

# 自定义数据集类，按需加载 .npz 文件
class BirdCLEFDataset(Dataset):
    def __init__(self, npz_files, transform=None):
        self.npz_files = npz_files
        self.transform = transform
        self.lengths = []
        self.cumulative_lengths = [0]
        self.file_indices = []
        
        for idx, npz_file in enumerate(self.npz_files):
            try:
                with np.load(npz_file) as data:
                    num_samples = len(data['spectrograms'])
                    self.lengths.append(num_samples)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + num_samples)
                    self.file_indices.extend([idx] * num_samples)
            except Exception as e:
                logging.error(f"Error loading {npz_file}: {e}")
                continue
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths) if idx < cum_len) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        try:
            with np.load(self.npz_files[file_idx]) as data:
                spectrogram = torch.from_numpy(data['spectrograms'][local_idx]).float()
                label = data['labels'][local_idx]
        except Exception as e:
            logging.error(f"Error loading sample {idx} from {self.npz_files[file_idx]}: {e}")
            return None
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label

# Vision Transformer 模型
class BirdCLEFViT(nn.Module):
    def __init__(self, num_classes, weights='IMAGENET1K_V1'):
        super(BirdCLEFViT, self).__init__()
        self.model = vit_b_16(weights=weights)
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 保存检查点
def save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss):
    checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'val_loss': val_loss
    }
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f'checkpoint_fold_{fold}.pth')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Saved checkpoint for fold {fold+1} at epoch {epoch+1}')
    print(f'Saved checkpoint for fold {fold+1} at epoch {epoch+1}')

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, fold):
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f'checkpoint_fold_{fold}.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        logging.info(f'Loaded checkpoint for fold {fold+1} from epoch {start_epoch}')
        print(f'Loaded checkpoint for fold {fold+1} from epoch {start_epoch}')
        return start_epoch, val_loss
    return 0, float('inf')

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, fold):
    scaler = GradScaler('cuda')
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, fold)
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, epochs):
        try:
            model.train()
            train_loss = 0
            for spectrograms, labels in tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs} - Training'):
                if spectrograms is None:
                    continue
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    outputs = model(spectrograms)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for spectrograms, labels in tqdm(val_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs} - Validation'):
                    if spectrograms is None:
                        continue
                    spectrograms = spectrograms.to(device)
                    labels = labels.to(device)
                    with autocast('cuda'):
                        outputs = model(spectrograms)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step()
            
            print(f'Fold {fold+1} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            logging.info(f'Fold {fold+1} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, f'best_model_fold_{fold}.pth'))
                print(f'Saved best model for fold {fold+1} with Val Loss: {best_val_loss:.4f}')
                logging.info(f'Saved best model for fold {fold+1} with Val Loss: {best_val_loss:.4f}')
            else:
                epochs_without_improvement += 1
            
            if cfg.save_per_epoch:
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, f'model_fold_{fold}_epoch_{epoch+1}.pth'))
                print(f'Saved model for fold {fold+1} at epoch {epoch+1}')
                logging.info(f'Saved model for fold {fold+1} at epoch {epoch+1}')

            save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss)

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs with no improvement in validation loss')
                logging.info(f'Early stopping triggered after {epoch+1} epochs with no improvement in validation loss')
                break

        except Exception as e:
            logging.error(f"Error during training at epoch {epoch+1} of fold {fold+1}: {e}")
            save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss)
            raise

    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, f'final_model_fold_{fold}.pth'))
    print(f'Saved final model for fold {fold+1} after training')
    logging.info(f'Saved final model for fold {fold+1} after training')

    return model

# 主执行流程
def main():
    train_df = pd.read_csv(f'{cfg.DATA_ROOT}/train.csv')
    label_list = sorted(train_df['primary_label'].unique())
    num_classes = len(label_list)
    print(f'Number of unique species (classes): {num_classes}')
    logging.info(f'Number of unique species (classes): {num_classes}')

    npz_files = glob.glob('/kaggle/input/birdclef-2025-spectrograms/batch_*/*.npz')
    if not npz_files:
        raise FileNotFoundError("No .npz files found in /kaggle/input/birdclef-2025-spectrograms/batch_*/")
    print(f'Found {len(npz_files)} .npz files for training')
    logging.info(f'Found {len(npz_files)} .npz files for training')

    if cfg.debug:
        npz_files = npz_files[:100]
        print(f'Debug mode: Reduced to {len(npz_files)} .npz files')
        logging.info(f'Debug mode: Reduced to {len(npz_files)} .npz files')

    kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    dataset = BirdCLEFDataset(npz_files)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        if fold not in cfg.selected_folds:
            continue
        
        print(f'\nStarting Fold {fold+1}/{cfg.n_fold}')
        logging.info(f'Starting Fold {fold+1}/{cfg.n_fold}')
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )
        
        model = BirdCLEFViT(num_classes=num_classes, weights='IMAGENET1K_V1')
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        model = model.to(cfg.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
        
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg.epochs, cfg.device, fold)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Training interrupted: {e}")
        raise


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
import pandas as pd
import glob
from torch.amp import autocast, GradScaler
import logging
import sys

# 设置日志
logging.basicConfig(filename='/kaggle/working/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 将 print 输出重定向到文件
class Logger:
    def __init__(self, filename='/kaggle/working/output.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# 配置类
class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2
    OUTPUT_DIR = '/kaggle/working'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    model_name = 'vit_b_16'
    pretrained = True
    in_channels = 3
    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224, 224)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    batch_size = 256
    criterion = 'CrossEntropyLoss'
    n_fold = 5
    selected_folds = [4]  # 只训练第 5 折（fold 4）
    optimizer = 'AdamW'
    lr = 1e-4
    weight_decay = 1e-5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs
    aug_prob = 0.5
    mixup_alpha = 0.5
    save_per_epoch = False
    early_stopping_patience = 2
    checkpoint_dir = '/kaggle/working/checkpoints'

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 3
            self.selected_folds = [4]

cfg = CFG()
cfg.update_debug_settings()

os.makedirs(cfg.checkpoint_dir, exist_ok=True)

# 自定义数据集类，按需加载 .npz 文件
class BirdCLEFDataset(Dataset):
    def __init__(self, npz_files, transform=None):
        self.npz_files = npz_files
        self.transform = transform
        self.lengths = []
        self.cumulative_lengths = [0]
        self.file_indices = []
        
        for idx, npz_file in enumerate(self.npz_files):
            try:
                with np.load(npz_file) as data:
                    num_samples = len(data['spectrograms'])
                    self.lengths.append(num_samples)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + num_samples)
                    self.file_indices.extend([idx] * num_samples)
            except Exception as e:
                logging.error(f"Error loading {npz_file}: {e}")
                continue
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths) if idx < cum_len) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        try:
            with np.load(self.npz_files[file_idx]) as data:
                spectrogram = torch.from_numpy(data['spectrograms'][local_idx]).float()
                label = data['labels'][local_idx]
        except Exception as e:
            logging.error(f"Error loading sample {idx} from {self.npz_files[file_idx]}: {e}")
            return None
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label

# Vision Transformer 模型
class BirdCLEFViT(nn.Module):
    def __init__(self, num_classes, weights='IMAGENET1K_V1'):
        super(BirdCLEFViT, self).__init__()
        self.model = vit_b_16(weights=weights)
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 保存检查点
def save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss):
    checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'val_loss': val_loss
    }
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f'checkpoint_fold_{fold}.pth')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Saved checkpoint for fold {fold+1} at epoch {epoch+1}')
    print(f'Saved checkpoint for fold {fold+1} at epoch {epoch+1}')

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, fold):
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f'checkpoint_fold_{fold}.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        logging.info(f'Loaded checkpoint for fold {fold+1} from epoch {start_epoch}')
        print(f'Loaded checkpoint for fold {fold+1} from epoch {start_epoch}')
        return start_epoch, val_loss
    return 0, float('inf')

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, fold):
    scaler = GradScaler('cuda')
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, fold)
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, epochs):
        try:
            model.train()
            train_loss = 0
            for spectrograms, labels in tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs} - Training'):
                if spectrograms is None:
                    continue
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    outputs = model(spectrograms)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for spectrograms, labels in tqdm(val_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs} - Validation'):
                    if spectrograms is None:
                        continue
                    spectrograms = spectrograms.to(device)
                    labels = labels.to(device)
                    with autocast('cuda'):
                        outputs = model(spectrograms)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step()
            
            print(f'Fold {fold+1} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            logging.info(f'Fold {fold+1} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, f'best_model_fold_{fold}.pth'))
                print(f'Saved best model for fold {fold+1} with Val Loss: {best_val_loss:.4f}')
                logging.info(f'Saved best model for fold {fold+1} with Val Loss: {best_val_loss:.4f}')
            else:
                epochs_without_improvement += 1
            
            if cfg.save_per_epoch:
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, f'model_fold_{fold}_epoch_{epoch+1}.pth'))
                print(f'Saved model for fold {fold+1} at epoch {epoch+1}')
                logging.info(f'Saved model for fold {fold+1} at epoch {epoch+1}')

            save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss)

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs with no improvement in validation loss')
                logging.info(f'Early stopping triggered after {epoch+1} epochs with no improvement in validation loss')
                break

        except Exception as e:
            logging.error(f"Error during training at epoch {epoch+1} of fold {fold+1}: {e}")
            save_checkpoint(model, optimizer, scheduler, epoch, fold, val_loss)
            raise

    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, f'final_model_fold_{fold}.pth'))
    print(f'Saved final model for fold {fold+1} after training')
    logging.info(f'Saved final model for fold {fold+1} after training')

    return model

# 主执行流程
def main():
    train_df = pd.read_csv(f'{cfg.DATA_ROOT}/train.csv')
    label_list = sorted(train_df['primary_label'].unique())
    num_classes = len(label_list)
    print(f'Number of unique species (classes): {num_classes}')
    logging.info(f'Number of unique species (classes): {num_classes}')

    npz_files = glob.glob('/kaggle/input/birdclef-2025-spectrograms/batch_*/*.npz')
    if not npz_files:
        raise FileNotFoundError("No .npz files found in /kaggle/input/birdclef-2025-spectrograms/batch_*/")
    print(f'Found {len(npz_files)} .npz files for training')
    logging.info(f'Found {len(npz_files)} .npz files for training')

    if cfg.debug:
        npz_files = npz_files[:100]
        print(f'Debug mode: Reduced to {len(npz_files)} .npz files')
        logging.info(f'Debug mode: Reduced to {len(npz_files)} .npz files')

    kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    dataset = BirdCLEFDataset(npz_files)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        if fold not in cfg.selected_folds:
            continue
        
        print(f'\nStarting Fold {fold+1}/{cfg.n_fold}')
        logging.info(f'Starting Fold {fold+1}/{cfg.n_fold}')
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )
        
        model = BirdCLEFViT(num_classes=num_classes, weights='IMAGENET1K_V1')
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        model = model.to(cfg.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
        
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg.epochs, cfg.device, fold)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Training interrupted: {e}")
        raise


import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16
import glob
import logging

# 设置日志
logging.basicConfig(filename='/kaggle/working/merge.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Vision Transformer 模型
class BirdCLEFViT(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(BirdCLEFViT, self).__init__()
        self.model = vit_b_16(weights=weights)
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 合并所有折叠的模型权重，保存最终模型
def save_final_model(num_classes, fold_model_paths):
    final_model = BirdCLEFViT(num_classes=num_classes, weights=None)  # 不加载预训练权重
    state_dicts = []
    
    # 加载所有折叠的最佳模型
    for fold in range(5):  # 0 到 4 折
        model_path = fold_model_paths[fold]
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            state_dicts.append(state_dict)
            logging.info(f'Loaded best model for fold {fold} from {model_path}')
            print(f'Loaded best model for fold {fold} from {model_path}')
        else:
            logging.warning(f'Best model for fold {fold} not found at {model_path}, skipping...')
            print(f'Best model for fold {fold} not found at {model_path}, skipping...')
            continue
    
    if not state_dicts:
        logging.error("No best models found for any fold, cannot save final model.")
        print("No best models found for any fold, cannot save final model.")
        return
    
    # 平均所有折叠的权重
    avg_state_dict = state_dicts[0].copy()
    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)
    
    final_model.load_state_dict(avg_state_dict)
    final_model_path = '/kaggle/working/final_model.pth'
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f'Saved final model with averaged weights from all folds at {final_model_path}')
    print(f'Saved final model with averaged weights from all folds at {final_model_path}')

# 主执行流程
def main():
    # 类别数
    num_classes = 206  # 根据之前的日志，类别数为 206

    # 模型参数路径
    # 前 4 折从 /kaggle/input/ 加载，第 5 折从 /kaggle/working/ 加载
    fold_model_paths = [
        '/kaggle/input/model/pytorch/default/1/best_model_fold_0.pth',  # 需上传
        '/kaggle/input/model/pytorch/default/1/best_model_fold_1.pth',  # 需上传
        '/kaggle/input/model/pytorch/default/1/best_model_fold_2.pth',  # 需上传
        '/kaggle/input/model/pytorch/default/1/best_model_fold_3.pth',  # 需上传
        '/kaggle/working/best_model_fold_4.pth'  # 第 5 折刚训练完成
    ]

    # 合并所有折叠的模型
    save_final_model(num_classes, fold_model_paths)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error during merging: {e}")
        raise











import os
import numpy as np
import torch
import pandas as pd
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt

# 设置路径
DATASET_PATH = '/kaggle/input/birdclef-2025-spectrograms'
TRAIN_CSV_PATH = '/kaggle/input/birdclef-2025/train.csv'
BATCH_FOLDER = 'batch_0'
NPZ_PATH = os.path.join(DATASET_PATH, BATCH_FOLDER, f'{BATCH_FOLDER}.npz')

# 从 train.csv 计算类别映射
print("Loading training metadata...")
train_df = pd.read_csv(TRAIN_CSV_PATH)
label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))
num_classes = len(label_list)
print(f"Global number of classes: {num_classes}")

# 加载 .npz 文件
data = np.load(NPZ_PATH)
spectrograms = data['spectrograms']  # 形状: (batch_size, 3, 224, 224)
labels = data['labels']  # 形状: (batch_size,)
keys = data['keys']  # 样本名称列表

# 检查数据形状
print(f"Spectrograms shape: {spectrograms.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Keys: {keys[:5]}")

# 加载 Hugging Face 预训练 ViT 模型和图像处理器
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,  # 使用全局类别数
    ignore_mismatched_sizes=True
)

# 如果有训练后的模型，加载它
TRAINED_MODEL_PATH = '/kaggle/working/vit_birdclef_model/final_model'
if os.path.exists(TRAINED_MODEL_PATH):
    print(f"Loading trained model from {TRAINED_MODEL_PATH}")
    model = ViTForImageClassification.from_pretrained(TRAINED_MODEL_PATH)
    processor = ViTImageProcessor.from_pretrained(TRAINED_MODEL_PATH)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 取第一个频谱图进行测试
sample_spectrogram = spectrograms[0]  # 形状: (3, 224, 224)
sample_label = labels[0]
sample_key = keys[0]

# 将频谱图转换为 PIL 图像
sample_spectrogram = sample_spectrogram.transpose(1, 2, 0)  # 转换为 (224, 224, 3)
sample_spectrogram = (sample_spectrogram * 255).astype(np.uint8)
image = Image.fromarray(sample_spectrogram)

# 使用图像处理器处理图像
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# 打印结果
print(f"Sample key: {sample_key}")
print(f"True label: {sample_label}")
print(f"Predicted class: {predicted_class}")

# 可视化频谱图
plt.figure(figsize=(6, 6))
plt.imshow(sample_spectrogram)
plt.title(f"Sample: {sample_key}, True Label: {sample_label}")
plt.axis('off')
plt.show()


print(f"label2id['CSA36385']: {label2id['1139490']}")
print(f"id2label[0]: {id2label[0]}")




