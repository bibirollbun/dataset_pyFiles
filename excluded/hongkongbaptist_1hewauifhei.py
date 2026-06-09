# import os
# import gc
# import logging
# import numpy as np
# import pandas as pd
# import librosa
# import torch
# from sklearn.model_selection import KFold
# import cv2
# from tqdm.auto import tqdm

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class CFG:
#     data_root = '/kaggle/input/birdclef-2025'
#     train_audio = f'{data_root}/train_audio'
#     train_csv = f'{data_root}/train.csv'
#     taxonomy_csv = f'{data_root}/taxonomy.csv'
#     output_dir = '/kaggle/working/processed_data'
    
#     FS = 32000
#     WINDOW_SIZE = 5
#     N_FFT = 1024
#     HOP_LENGTH = 512
#     N_MELS = 128
#     FMIN = 50
#     FMAX = 14000
#     TARGET_SHAPE = (256, 256)
    
#     n_folds = 5
#     selected_folds = [0, 1]
#     seed = 42
#     aug_prob = 0.5

# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def audio2melspec(audio_data, cfg):
#     if np.isnan(audio_data).any():
#         mean_signal = np.nanmean(audio_data)
#         audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    
#     mel_spec = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=cfg.FS,
#         n_fft=cfg.N_FFT,
#         hop_length=cfg.HOP_LENGTH,
#         n_mels=cfg.N_MELS,
#         fmin=cfg.FMIN,
#         fmax=cfg.FMAX,
#         power=2.0
#     )
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
#     return mel_spec_norm

# def spec_augment(mel_spec, cfg):
#     if np.random.rand() > cfg.aug_prob:
#         return mel_spec
#     max_time_mask = int(mel_spec.shape[1] * 0.1)
#     time_mask = np.random.randint(0, max(1, mel_spec.shape[1] - max_time_mask))
#     mel_spec[:, time_mask:time_mask + max_time_mask] = 0
#     max_freq_mask = int(mel_spec.shape[0] * 0.1)
#     freq_mask = np.random.randint(0, max(1, mel_spec.shape[0] - max_freq_mask))
#     mel_spec[freq_mask:freq_mask + max_freq_mask, :] = 0
#     return mel_spec

# def preprocess_data(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)
#     train_df = pd.read_csv(cfg.train_csv)
#     species = pd.read_csv(cfg.taxonomy_csv)['primary_label'].tolist()
#     num_classes = len(species)
    
#     # Create labels
#     labels = np.zeros((len(train_df), num_classes), dtype=np.float32)
#     for idx, row in train_df.iterrows():
#         primary_label = row['primary_label']
#         label_idx = species.index(primary_label)
#         labels[idx, label_idx] = 1.0
    
#     # Process audio files
#     spectrograms = []
#     audio_paths = [os.path.join(cfg.train_audio, fname) for fname in train_df['filename']]
#     for idx, audio_path in enumerate(tqdm(audio_paths, desc="Processing audio")):
#         try:
#             audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
#             target_samples = cfg.FS * cfg.WINDOW_SIZE
#             if len(audio_data) < target_samples:
#                 audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
#             else:
#                 start_idx = np.random.randint(0, max(1, len(audio_data) - target_samples))
#                 audio_data = audio_data[start_idx:start_idx + target_samples]
            
#             mel_spec = audio2melspec(audio_data, cfg)
#             mel_spec = spec_augment(mel_spec, cfg)
#             if mel_spec.shape != cfg.TARGET_SHAPE:
#                 mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
#             mel_spec = mel_spec.astype(np.float32)
#             spectrograms.append(mel_spec)
#         except Exception as e:
#             logging.error(f"Error processing sample {idx}: {e}")
#             mel_spec = np.zeros(cfg.TARGET_SHAPE, dtype=np.float32)
#             spectrograms.append(mel_spec)
    
#     # Save spectrograms and labels
#     spectrograms = np.array(spectrograms)  # Shape: (num_samples, 256, 256)
#     np.save(os.path.join(cfg.output_dir, 'spectrograms.npy'), spectrograms)
#     np.save(os.path.join(cfg.output_dir, 'labels.npy'), labels)
    
#     # Save k-fold indices
#     kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
#     fold_indices = {}
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#         fold_indices[f'fold_{fold}'] = {'train_idx': train_idx, 'val_idx': val_idx}
#     np.save(os.path.join(cfg.output_dir, 'fold_indices.npy'), fold_indices)
    
#     logging.info(f"Processed data saved to {cfg.output_dir}")

# if __name__ == "__main__":
#     cfg = CFG()
#     set_seed(cfg.seed)
#     preprocess_data(cfg)


# import os
# import gc
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import average_precision_score
# from tqdm.auto import tqdm
# import timm
# from torch.cuda.amp import autocast, GradScaler

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class CFG:
#     output_dir = '/kaggle/working/checkpoints'
#     processed_data_dir = '/kaggle/input/bird-regnet/processed_data'
    
#     model_name = 'regnety_008'
#     in_channels = 1
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     n_folds = 5
#     selected_folds = [0, 1]
#     epochs = 10
#     batch_size = 32
#     num_workers = 0
#     lr = 1e-4
#     weight_decay = 1e-5
#     early_stopping_patience = 3
#     seed = 42

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class PreprocessedBirdCLEFDataset(Dataset):
#     def __init__(self, spectrograms, labels, indices):
#         self.spectrograms = spectrograms[indices]
#         self.labels = labels[indices]
    
#     def __len__(self):
#         return len(self.spectrograms)
    
#     def __getitem__(self, idx):
#         mel_spec = self.spectrograms[idx]
#         mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return mel_spec, label

# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=True,
#             in_chans=cfg.in_channels,
#             drop_rate=0.3,
#             drop_path_rate=0.2
#         )
#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')
        
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)
    
#     def forward(self, x):
#         features = self.backbone(x)
#         if isinstance(features, dict):
#             features = features['features']
#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)
#         logits = self.classifier(features)
#         return logits

# def train_epoch(model, loader, criterion, optimizer, scaler, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (spectrograms, labels) in enumerate(tqdm(loader, desc="Training")):
#         spectrograms, labels = spectrograms.to(device), labels.to(device)
#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()
#     return total_loss / len(loader)

# def validate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for spectrograms, labels in tqdm(loader, desc="Validating"):
#             spectrograms, labels = spectrograms.to(device), labels.to(device)
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_preds.append(probs)
#             all_labels.append(labels.cpu().numpy())
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#     mAP = average_precision_score(all_labels, all_preds, average='macro')
#     return total_loss / len(loader), mAP

# def save_checkpoint(model, optimizer, epoch, fold, val_mAP, path):
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'fold': fold,
#         'val_mAP': val_mAP
#     }, path)
#     logging.info(f"Saved checkpoint for fold {fold} at epoch {epoch}")

# def train_model(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)
    
#     # Load preprocessed data
#     spectrograms = np.load(os.path.join(cfg.processed_data_dir, 'spectrograms.npy'))
#     labels = np.load(os.path.join(cfg.processed_data_dir, 'labels.npy'))
#     fold_indices = np.load(os.path.join(cfg.processed_data_dir, 'fold_indices.npy'), allow_pickle=True).item()
#     num_classes = labels.shape[1]
    
#     for fold in cfg.selected_folds:
#         logging.info(f"Training fold {fold}")
#         train_idx = fold_indices[f'fold_{fold}']['train_idx']
#         val_idx = fold_indices[f'fold_{fold}']['val_idx']
        
#         train_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, train_idx)
#         val_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, val_idx)
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True
#         )
        
#         model = BirdCLEFModel(cfg, num_classes).to(cfg.device)
#         criterion = nn.BCEWithLogitsLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
#         scaler = GradScaler()
        
#         best_mAP = 0
#         epochs_without_improvement = 0
        
#         try:
#             for epoch in range(cfg.epochs):
#                 train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, cfg.device)
#                 val_loss, val_mAP = validate_epoch(model, val_loader, criterion, cfg.device)
                
#                 logging.info(f"Fold {fold} Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f}")
                
#                 if val_mAP > best_mAP:
#                     best_mAP = val_mAP
#                     epochs_without_improvement = 0
#                     save_checkpoint(model, optimizer, epoch, fold, val_mAP, os.path.join(cfg.output_dir, f'fold{fold}_best.pth'))
#                 else:
#                     epochs_without_improvement += 1
                
#                 scheduler.step()
                
#                 if epochs_without_improvement >= cfg.early_stopping_patience:
#                     logging.info(f"Early stopping triggered for fold {fold} after epoch {epoch+1}")
#                     break
#         finally:
#             del train_loader, val_loader
#             gc.collect()
#             torch.cuda.empty_cache()
        
#         save_checkpoint(model, optimizer, cfg.epochs, fold, best_mAP, os.path.join(cfg.output_dir, f'fold{fold}_final.pth'))

# if __name__ == "__main__":
#     cfg = CFG()
#     set_seed(cfg.seed)
#     train_model(cfg)














# import os
# import gc
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import KFold
# from tqdm.auto import tqdm
# import timm
# from torch.amp import GradScaler, autocast

# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
# logging.getLogger().setLevel(logging.INFO)

# class CFG:
#     output_dir = '/kaggle/working/checkpoints'
#     processed_data_dir = '/kaggle/input/bird-regnet/processed_data'
    
#     model_name = 'regnety_008'
#     in_channels = 1
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     n_folds = 3  # 减少折数以增加验证集规模
#     selected_folds = [0, 1]
#     epochs = 10
#     batch_size = 32
#     num_workers = 0
#     lr = 1e-4
#     weight_decay = 1e-5
#     early_stopping_patience = 5  # 增加早停耐心
#     seed = 42
#     subset_ratio = 0.2  # 使用 20% 的数据

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class PreprocessedBirdCLEFDataset(Dataset):
#     def __init__(self, spectrograms, labels, indices):
#         self.spectrograms = spectrograms[indices]
#         self.labels = labels[indices]
    
#     def __len__(self):
#         return len(self.spectrograms)
    
#     def __getitem__(self, idx):
#         mel_spec = self.spectrograms[idx]
#         mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return mel_spec, label

# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=True,
#             in_chans=cfg.in_channels,
#             drop_rate=0.3,
#             drop_path_rate=0.2
#         )
#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')
        
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)
    
#     def forward(self, x):
#         features = self.backbone(x)
#         if isinstance(features, dict):
#             features = features['features']
#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)
#         logits = self.classifier(features)
#         return logits

# def train_epoch(model, loader, criterion, optimizer, scaler, device):
#     if scaler is None:
#         raise ValueError("Scaler 未定义，请确保 GradScaler 已正确初始化。")
#     model.train()
#     total_loss = 0
#     for batch_idx, (spectrograms, labels) in enumerate(tqdm(loader, desc="Training")):
#         spectrograms, labels = spectrograms.to(device), labels.to(device)
#         optimizer.zero_grad()
#         with autocast(device_type='cuda'):
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(loader)
#     print(f"DEBUG: train_epoch 完成, avg_loss={avg_loss}")
#     return avg_loss

# def validate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for spectrograms, labels in tqdm(loader, desc="Validating"):
#             spectrograms, labels = spectrograms.to(device), labels.to(device)
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_preds.append(probs)
#             all_labels.append(labels.cpu().numpy())
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
    
#     ap_scores = []
#     for i in range(all_labels.shape[1]):
#         if np.sum(all_labels[:, i]) > 0:
#             ap = average_precision_score(all_labels[:, i], all_preds[:, i])
#             ap_scores.append(ap)
#         else:
#             logging.warning(f"类别 {i} 在验证集中没有正样本，跳过。")
    
#     mAP = np.mean(ap_scores) if ap_scores else 0.0
#     avg_loss = total_loss / len(loader)
#     print(f"DEBUG: validate_epoch 完成, avg_loss={avg_loss}, mAP={mAP}")
#     return avg_loss, mAP

# def save_checkpoint(model, optimizer, epoch, fold, val_mAP, path):
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'fold': fold,
#         'val_mAP': val_mAP
#     }, path)
#     print(f"已保存检查点：fold {fold} 在 epoch {epoch}")

# def select_subset(spectrograms, labels, subset_ratio, seed):
#     num_samples = len(spectrograms)
#     subset_size = int(num_samples * subset_ratio)
#     np.random.seed(seed)
#     label_sums = np.sum(labels, axis=0)
    
#     # 确保每个类别至少有 1 个样本（如果存在）
#     subset_indices = []
#     for cls in range(labels.shape[1]):
#         cls_indices = np.where(labels[:, cls] == 1)[0]
#         if len(cls_indices) > 0:
#             subset_indices.append(np.random.choice(cls_indices, size=1, replace=False))
    
#     subset_indices = np.unique(subset_indices)
#     remaining_size = max(0, subset_size - len(subset_indices))
#     other_indices = np.setdiff1d(np.arange(num_samples), subset_indices)
#     if remaining_size > 0 and len(other_indices) > 0:
#         other_subset = np.random.choice(other_indices, size=remaining_size, replace=False)
#         subset_indices = np.concatenate([subset_indices, other_subset])
    
#     return subset_indices

# def train_model(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)
    
#     spectrograms = np.load(os.path.join(cfg.processed_data_dir, 'spectrograms.npy'))
#     labels = np.load(os.path.join(cfg.processed_data_dir, 'labels.npy'))
    
#     subset_indices = select_subset(spectrograms, labels, cfg.subset_ratio, cfg.seed)
#     spectrograms = spectrograms[subset_indices]
#     labels = labels[subset_indices]
#     num_classes = labels.shape[1]
    
#     class_counts = np.sum(labels, axis=0)
#     print(f"子集类别分布: {class_counts}")
#     zero_classes = np.where(class_counts == 0)[0]
#     if len(zero_classes) > 0:
#         print(f"子集中没有样本的类别: {zero_classes}")
    
#     kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
#     fold_indices = {}
#     for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(spectrograms)))):
#         fold_indices[f'fold_{fold}'] = {'train_idx': train_idx, 'val_idx': val_idx}
    
#     for fold in cfg.selected_folds:
#         print(f"训练 fold {fold}")
#         train_idx = fold_indices[f'fold_{fold}']['train_idx']
#         val_idx = fold_indices[f'fold_{fold}']['val_idx']
        
#         train_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, train_idx)
#         val_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, val_idx)
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True
#         )
        
#         model = BirdCLEFModel(cfg, num_classes).to(cfg.device)
#         criterion = nn.BCEWithLogitsLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
#         scaler = GradScaler()
        
#         best_mAP = 0
#         epochs_without_improvement = 0
#         min_mAP_improvement = 0.001  # 最小 mAP 提升阈值
        
#         try:
#             for epoch in range(cfg.epochs):
#                 train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, cfg.device)
#                 val_loss, val_mAP = validate_epoch(model, val_loader, criterion, cfg.device)
                
#                 print(f"Fold {fold} Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f}", flush=True)
                
#                 if val_mAP > best_mAP + min_mAP_improvement:
#                     best_mAP = val_mAP
#                     epochs_without_improvement = 0
#                     save_checkpoint(model, optimizer, epoch, fold, val_mAP, os.path.join(cfg.output_dir, f'fold{fold}_best.pth'))
#                 else:
#                     epochs_without_improvement += 1
#                     print(f"没有提升的 epoch 数量: {epochs_without_improvement}", flush=True)
                
#                 scheduler.step()
                
#                 if epochs_without_improvement >= cfg.early_stopping_patience:
#                     print(f"早停触发：fold {fold} 在 epoch {epoch+1} 后停止")
#                     break
#         finally:
#             del train_loader, val_loader
#             gc.collect()
#             torch.cuda.empty_cache()
        
#         save_checkpoint(model, optimizer, cfg.epochs, fold, best_mAP, os.path.join(cfg.output_dir, f'fold{fold}_final.pth'))

# if __name__ == "__main__":
#     cfg = CFG()
#     set_seed(cfg.seed)
#     train_model(cfg)


# import os
# import gc
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import KFold
# from tqdm.auto import tqdm
# import timm
# from torch.amp import GradScaler, autocast
# import shutil
# import psutil

# # 配置日志，写入文件
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(os.path.join('/kaggle/working', 'training_log.txt'))
#     ]
# )
# logging.getLogger().setLevel(logging.INFO)

# class CFG:
#     output_dir = '/kaggle/working/model'
#     processed_data_dir = '/kaggle/input/bird-regnet/processed_data'
    
#     model_name = 'regnety_008'
#     in_channels = 1
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     n_folds = 5  # 增加到 5 折
#     selected_folds = [0]
#     epochs = 20  # 增加到 20 个 epoch
#     batch_size = 32
#     num_workers = 0
#     lr = 1e-4  # 微调学习率
#     weight_decay = 1e-5
#     early_stopping_patience = 5
#     seed = 42
#     subset_ratio = 1.0

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class PreprocessedBirdCLEFDataset(Dataset):
#     def __init__(self, spectrograms, labels, indices):
#         self.spectrograms = spectrograms[indices]
#         self.labels = labels[indices]
    
#     def __len__(self):
#         return len(self.spectrograms)
    
#     def spec_augment(self, mel_spec, aug_prob=0.5):
#         if np.random.rand() > aug_prob:
#             return mel_spec
#         max_time_mask = int(mel_spec.shape[1] * 0.1)
#         time_mask = np.random.randint(0, max(1, mel_spec.shape[1] - max_time_mask))
#         mel_spec[:, time_mask:time_mask + max_time_mask] = 0
#         max_freq_mask = int(mel_spec.shape[0] * 0.1)
#         freq_mask = np.random.randint(0, max(1, mel_spec.shape[0] - max_freq_mask))
#         mel_spec[freq_mask:freq_mask + max_freq_mask, :] = 0
#         return mel_spec
    
#     def __getitem__(self, idx):
#         mel_spec = self.spectrograms[idx].copy()
#         mel_spec = self.spec_augment(mel_spec)
#         mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return mel_spec, label

# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=True,
#             in_chans=cfg.in_channels,
#             drop_rate=0.3,
#             drop_path_rate=0.3
#         )
#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')
        
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)
    
#     def forward(self, x):
#         features = self.backbone(x)
#         if isinstance(features, dict):
#             features = features['features']
#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)
#         logits = self.classifier(features)
#         return logits

# def train_epoch(model, loader, criterion, optimizer, scaler, device):
#     if scaler is None:
#         raise ValueError("Scaler 未定义，请确保 GradScaler 已正确初始化。")
#     model.train()
#     total_loss = 0
#     for batch_idx, (spectrograms, labels) in enumerate(tqdm(loader, desc="Training", disable=True)):
#         spectrograms, labels = spectrograms.to(device), labels.to(device)
#         optimizer.zero_grad()
#         with autocast(device_type='cuda'):
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(loader)
#     return avg_loss

# def validate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for spectrograms, labels in tqdm(loader, desc="Validating", disable=True):
#             spectrograms, labels = spectrograms.to(device), labels.to(device)
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_preds.append(probs)
#             all_labels.append(labels.cpu().numpy())
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
    
#     ap_scores = []
#     for i in range(all_labels.shape[1]):
#         if np.sum(all_labels[:, i]) > 0:
#             ap = average_precision_score(all_labels[:, i], all_preds[:, i])
#             ap_scores.append(ap)
#         else:
#             logging.warning(f"类别 {i} 在验证集中没有正样本，跳过。")
    
#     mAP = np.mean(ap_scores) if ap_scores else 0.0
#     avg_loss = total_loss / len(loader)
#     return avg_loss, mAP

# def save_checkpoint(model, optimizer, epoch, fold, val_mAP, path):
#     try:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         temp_path = os.path.join('/tmp', os.path.basename(path))
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': epoch,
#             'fold': fold,
#             'val_mAP': val_mAP
#         }, temp_path)
#         shutil.copy(temp_path, path)
#         print(f"成功保存检查点：{path}", flush=True)
#     except Exception as e:
#         print(f"保存检查点失败：{path}，错误: {e}", flush=True)
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# def get_disk_space(path):
#     disk = psutil.disk_usage(path)
#     return disk.free / (1024**3)

# def select_subset(spectrograms, labels, subset_ratio, seed):
#     num_samples = len(spectrograms)
#     subset_size = int(num_samples * subset_ratio)
#     np.random.seed(seed)
#     label_sums = np.sum(labels, axis=0)
    
#     subset_indices = []
#     for cls in range(labels.shape[1]):
#         cls_indices = np.where(labels[:, cls] == 1)[0]
#         if len(cls_indices) > 0:
#             subset_indices.append(np.random.choice(cls_indices, size=1, replace=False))
    
#     subset_indices = np.unique(subset_indices)
#     remaining_size = max(0, subset_size - len(subset_indices))
#     other_indices = np.setdiff1d(np.arange(num_samples), subset_indices)
#     if remaining_size > 0 and len(other_indices) > 0:
#         other_subset = np.random.choice(other_indices, size=remaining_size, replace=False)
#         subset_indices = np.concatenate([subset_indices, other_subset])
    
#     return subset_indices

# def train_model(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)
    
#     free_space = get_disk_space('/kaggle/working')
#     print(f"/kaggle/working 可用磁盘空间: {free_space:.2f} GB")
#     free_space_tmp = get_disk_space('/tmp')
#     print(f"/tmp 可用磁盘空间: {free_space_tmp:.2f} GB")
    
#     if not os.access(cfg.output_dir, os.W_OK):
#         print(f"警告：{cfg.output_dir} 不可写，将尝试使用 /tmp 目录")
#         cfg.output_dir = '/tmp/model'
#         os.makedirs(cfg.output_dir, exist_ok=True)
    
#     spectrograms = np.load(os.path.join(cfg.processed_data_dir, 'spectrograms.npy'))
#     labels = np.load(os.path.join(cfg.processed_data_dir, 'labels.npy'))
    
#     subset_indices = select_subset(spectrograms, labels, cfg.subset_ratio, cfg.seed)
#     spectrograms = spectrograms[subset_indices]
#     labels = labels[subset_indices]
#     num_classes = labels.shape[1]
    
#     class_counts = np.sum(labels, axis=0)
#     print(f"子集类别分布: {class_counts}")
#     zero_classes = np.where(class_counts == 0)[0]
#     if len(zero_classes) > 0:
#         print(f"子集中没有样本的类别: {zero_classes}")
    
#     kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
#     fold_indices = {}
#     for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(spectrograms)))):
#         fold_indices[f'fold_{fold}'] = {'train_idx': train_idx, 'val_idx': val_idx}
    
#     for fold in cfg.selected_folds:
#         print(f"训练 fold {fold}")
#         train_idx = fold_indices[f'fold_{fold}']['train_idx']
#         val_idx = fold_indices[f'fold_{fold}']['val_idx']
        
#         train_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, train_idx)
#         val_dataset = PreprocessedBirdCLEFDataset(spectrograms, labels, val_idx)
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True
#         )
        
#         model = BirdCLEFModel(cfg, num_classes).to(cfg.device)
#         label_sums = np.sum(labels, axis=0)
#         pos_weight = torch.tensor(np.clip((len(labels) - label_sums) / (label_sums + 1e-8), 0.1, 100.0), device=cfg.device)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
#         scaler = GradScaler()
        
#         best_mAP = 0
#         epochs_without_improvement = 0
#         min_mAP_improvement = 0.001
        
#         try:
#             for epoch in range(cfg.epochs):
#                 train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, cfg.device)
#                 val_loss, val_mAP = validate_epoch(model, val_loader, criterion, cfg.device)
                
#                 print(f"Fold {fold} Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f}", flush=True)
#                 print(f"当前学习率: {scheduler.get_last_lr()}", flush=True)
                
#                 if val_mAP > best_mAP + min_mAP_improvement:
#                     best_mAP = val_mAP
#                     epochs_without_improvement = 0
#                     save_checkpoint(model, optimizer, epoch, fold, val_mAP, os.path.join(cfg.output_dir, f'fold{fold}_best.pth'))
#                 else:
#                     epochs_without_improvement += 1
#                     print(f"没有提升的 epoch 数量: {epochs_without_improvement}", flush=True)
                
#                 scheduler.step(val_loss)
                
#                 if epochs_without_improvement >= cfg.early_stopping_patience:
#                     print(f"早停触发：fold {fold} 在 epoch {epoch+1} 后停止")
#                     break
#         finally:
#             del train_loader, val_loader
#             gc.collect()
#             torch.cuda.empty_cache()
        
#         save_checkpoint(model, optimizer, cfg.epochs, fold, best_mAP, os.path.join(cfg.output_dir, f'fold{fold}_final.pth'))
        
#         if cfg.output_dir.startswith('/tmp'):
#             final_output_dir = '/kaggle/working/model'
#             os.makedirs(final_output_dir, exist_ok=True)
#             for file_name in os.listdir(cfg.output_dir):
#                 src_path = os.path.join(cfg.output_dir, file_name)
#                 dst_path = os.path.join(final_output_dir, file_name)
#                 shutil.copy(src_path, dst_path)
#                 print(f"复制文件：{src_path} 到 {dst_path}", flush=True)

# if __name__ == "__main__":
#     cfg = CFG()
#     set_seed(cfg.seed)
#     train_model(cfg)




















# import os
# import gc
# import logging
# import numpy as np
# import pandas as pd
# import librosa
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import KFold
# from sklearn.metrics import average_precision_score
# import cv2
# from tqdm.auto import tqdm
# import timm
# from torch.cuda.amp import autocast, GradScaler

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class CFG:
#     data_root = '/kaggle/input/birdclef-2025'
#     train_audio = f'{data_root}/train_audio'
#     train_csv = f'{data_root}/train.csv'
#     taxonomy_csv = f'{data_root}/taxonomy.csv'
#     output_dir = '/kaggle/working/checkpoints'
#     model_path = '/kaggle/input/regnrty008/pytorch/default/1'
    
#     FS = 32000
#     WINDOW_SIZE = 5
#     N_FFT = 1024
#     HOP_LENGTH = 512
#     N_MELS = 128
#     FMIN = 50
#     FMAX = 14000
#     TARGET_SHAPE = (256, 256)
    
#     model_name = 'regnety_008'
#     in_channels = 1
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     n_folds = 5
#     selected_folds = [0, 1]
#     epochs = 10
#     batch_size = 32
#     num_workers = 0
#     lr = 1e-4
#     weight_decay = 1e-5
#     early_stopping_patience = 3
#     seed = 42
#     aug_prob = 0.5

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(CFG.seed)

# class BirdCLEFDataset(Dataset):
#     def __init__(self, df, cfg, mode='train'):
#         self.df = df
#         self.cfg = cfg
#         self.mode = mode
#         self.audio_paths = [os.path.join(self.cfg.train_audio, fname) for fname in df['filename']]
#         self.labels = self.create_labels(df)
    
#     def create_labels(self, df):
#         species = pd.read_csv(self.cfg.taxonomy_csv)['primary_label'].tolist()
#         num_classes = len(species)
#         labels = np.zeros((len(df), num_classes), dtype=np.float32)
#         for idx, row in df.iterrows():
#             primary_label = row['primary_label']
#             label_idx = species.index(primary_label)
#             labels[idx, label_idx] = 1.0
#         return labels
    
#     def __len__(self):
#         return len(self.df)
    
#     def audio2melspec(self, audio_data):
#         if np.isnan(audio_data).any():
#             mean_signal = np.nanmean(audio_data)
#             audio_data = np.nan_to_num(audio_data, nan=mean_signal)
        
#         mel_spec = librosa.feature.melspectrogram(
#             y=audio_data,
#             sr=self.cfg.FS,
#             n_fft=self.cfg.N_FFT,
#             hop_length=self.cfg.HOP_LENGTH,
#             n_mels=self.cfg.N_MELS,
#             fmin=self.cfg.FMIN,
#             fmax=self.cfg.FMAX,
#             power=2.0
#         )
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#         mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
#         return mel_spec_norm
    
#     def spec_augment(self, mel_spec):
#         if self.mode != 'train' or np.random.rand() > self.cfg.aug_prob:
#             return mel_spec
#         max_time_mask = int(mel_spec.shape[1] * 0.1)
#         time_mask = np.random.randint(0, max(1, mel_spec.shape[1] - max_time_mask))
#         mel_spec[:, time_mask:time_mask + max_time_mask] = 0
#         max_freq_mask = int(mel_spec.shape[0] * 0.1)
#         freq_mask = np.random.randint(0, max(1, mel_spec.shape[0] - max_freq_mask))
#         mel_spec[freq_mask:freq_mask + max_freq_mask, :] = 0
#         return mel_spec
    
#     def __getitem__(self, idx):
#         try:
#             audio_data, _ = librosa.load(self.audio_paths[idx], sr=self.cfg.FS)
#             target_samples = self.cfg.FS * self.cfg.WINDOW_SIZE
#             if len(audio_data) < target_samples:
#                 audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
#             else:
#                 start_idx = np.random.randint(0, max(1, len(audio_data) - target_samples))
#                 audio_data = audio_data[start_idx:start_idx + target_samples]
            
#             mel_spec = self.audio2melspec(audio_data)
#             mel_spec = self.spec_augment(mel_spec)
#             if mel_spec.shape != self.cfg.TARGET_SHAPE:
#                 mel_spec = cv2.resize(mel_spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
#             mel_spec = mel_spec.astype(np.float32)
#             mel_spec = torch.tensor(mel_spec).unsqueeze(0)
#             label = torch.tensor(self.labels[idx], dtype=torch.float32)
#             return mel_spec, label
#         except Exception as e:
#             logging.error(f"Error loading sample {idx}: {e}")
#             mel_spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
#             mel_spec = cv2.resize(mel_spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
#             mel_spec = torch.tensor(mel_spec).unsqueeze(0)
#             label = torch.zeros(len(self.labels[0]), dtype=torch.float32)
#             return mel_spec, label

# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=True,
#             in_chans=cfg.in_channels,
#             drop_rate=0.3,
#             drop_path_rate=0.2
#         )
#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')
        
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)
    
#     def forward(self, x):
#         features = self.backbone(x)
#         if isinstance(features, dict):
#             features = features['features']
#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)
#         logits = self.classifier(features)
#         return logits

# def train_epoch(model, loader, criterion, optimizer, scaler, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (spectrograms, labels) in enumerate(tqdm(loader, desc="Training")):
#         spectrograms, labels = spectrograms.to(device), labels.to(device)
#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()
#     return total_loss / len(loader)

# def validate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for spectrograms, labels in tqdm(loader, desc="Validating"):
#             spectrograms, labels = spectrograms.to(device), labels.to(device)
#             outputs = model(spectrograms)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_preds.append(probs)
#             all_labels.append(labels.cpu().numpy())
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#     mAP = average_precision_score(all_labels, all_preds, average='macro')
#     return total_loss / len(loader), mAP

# def save_checkpoint(model, optimizer, epoch, fold, val_mAP, path):
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'fold': fold,
#         'val_mAP': val_mAP
#     }, path)
#     logging.info(f"Saved checkpoint for fold {fold} at epoch {epoch}")

# def train_model(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)
#     train_df = pd.read_csv(cfg.train_csv)
#     species = pd.read_csv(cfg.taxonomy_csv)['primary_label'].tolist()
#     num_classes = len(species)
    
#     kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#         if fold not in cfg.selected_folds:
#             continue
        
#         logging.info(f"Training fold {fold}")
#         train_subset = train_df.iloc[train_idx].reset_index(drop=True)
#         val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
#         train_dataset = BirdCLEFDataset(train_subset, cfg, mode='train')
#         val_dataset = BirdCLEFDataset(val_subset, cfg, mode='val')
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=cfg.batch_size,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True
#         )
        
#         model = BirdCLEFModel(cfg, num_classes).to(cfg.device)
#         criterion = nn.BCEWithLogitsLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
#         scaler = GradScaler()
        
#         best_mAP = 0
#         epochs_without_improvement = 0
        
#         try:
#             for epoch in range(cfg.epochs):
#                 train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, cfg.device)
#                 val_loss, val_mAP = validate_epoch(model, val_loader, criterion, cfg.device)
                
#                 logging.info(f"Fold {fold} Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f}")
                
#                 if val_mAP > best_mAP:
#                     best_mAP = val_mAP
#                     epochs_without_improvement = 0
#                     save_checkpoint(model, optimizer, epoch, fold, val_mAP, os.path.join(cfg.output_dir, f'fold{fold}_best.pth'))
#                 else:
#                     epochs_without_improvement += 1
                
#                 scheduler.step()
                
#                 if epochs_without_improvement >= cfg.early_stopping_patience:
#                     logging.info(f"Early stopping triggered for fold {fold} after epoch {epoch+1}")
#                     break
#         finally:
#             del train_loader, val_loader
#             gc.collect()
#             torch.cuda.empty_cache()
        
#         save_checkpoint(model, optimizer, cfg.epochs, fold, best_mAP, os.path.join(cfg.output_dir, f'fold{fold}_final.pth'))

# if __name__ == "__main__":
#     cfg = CFG()
#     train_model(cfg)


# import os
# import psutil
# print(f"Available CPU cores: {os.cpu_count()}")
# mem = psutil.virtual_memory()
# print(f"Memory usage: {mem.percent}%")




