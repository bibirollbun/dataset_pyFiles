import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')


# SimpleTwoBranchLight v5モデル定義
class GaussianNoise(nn.Module):
    """ガウシアンノイズレイヤー（学習時のみ適用）"""
    def __init__(self, sigma=0.09):
        super().__init__()
        self.sigma = sigma
        self.register_buffer('noise', torch.tensor(0))
    
    def forward(self, x):
        if self.training and self.sigma > 0:
            sampled_noise = self.noise.expand(*x.size()).float().normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    """残差接続付きSE-CNNブロック"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += self.skip(residual)
        out = F.relu(out)
        
        return out


class SimpleTwoBranchLight(nn.Module):
    """
    SimpleTwoBranchLight v5 - 改良版Two-Branchモデル
    
    v5の主な改良点:
    - Modality Dropout: 学習時50%でToF/Thermalをドロップ（CMI制約対応）
    - IMU入力次元: 7 → 11（特徴量エンジニアリング追加）
    - ToF入力次元: 320 → 50（統計量20 + 空間特徴量30）
    - Dropout: 段階的調整（0.1→0.3→0.5）
    - RNN: 1層のみ（シンプル化）
    """
    
    def __init__(self, imu_dim=11, tof_dim=50, thermopile_dim=5, 
                 hidden_dim=256, num_classes=18, modality_dropout_rate=0.5):
        super().__init__()
        
        # IMU Branch (ResidualSECNNBlock使用)
        self.imu_branch = nn.Sequential(
            ResidualSECNNBlock(imu_dim, 64, dropout=0.1),  # v5オリジナルサイズ
            nn.MaxPool1d(2),
            ResidualSECNNBlock(64, 128, kernel_size=5, dropout=0.1),  # v5オリジナルサイズ
            nn.MaxPool1d(2)
        )
        
        # ToF/Thermal Branch (シンプルなCNN)
        tof_thermal_dim = tof_dim + thermopile_dim  # 50 + 5 = 55 (v5で拡張)
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_thermal_dim, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Merge dimension
        merged_dim = 256  # 128 + 128 (v5オリジナル)
        
        # Parallel RNN (1層のみ、LB 0.76準拠)
        self.lstm = nn.LSTM(
            merged_dim, hidden_dim // 2, 
            num_layers=1,  # 1層に削減
            batch_first=True, 
            dropout=0,  # 1層なのでLSTM内dropoutは0
            bidirectional=True
        )
        
        self.gru = nn.GRU(
            merged_dim, hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Noise branch (LB 0.76準拠)
        self.gaussian_noise = GaussianNoise(0.09)
        self.noise_branch = nn.Sequential(
            nn.Linear(merged_dim, 16),
            nn.ELU()
        )
        
        # Combined dimension: LSTM + GRU + Noise
        combined_dim = hidden_dim * 2 + 16
        
        # Attention mechanism with LayerNorm for stability
        self.attention_layernorm = nn.LayerNorm(combined_dim)
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head (段階的Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 高め
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 中程度
            
            nn.Linear(128, num_classes)
        )
        
        # Binary BFRB classifier
        self.bfrb_classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )
        
        # Modality Dropout rate (v5.1追加)
        self.modality_dropout_rate = modality_dropout_rate
        
        # 重み初期化
        self._init_weights()
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, _ = x.size()
        
        # Split sensors
        # IMU: 11次元（7 + 4特徴量）
        # ToF統計量: 20次元 + 空間特徴量: 30次元 = 50次元（v5で拡張）
        # Thermopile: 5次元
        imu = x[:, :, :11].transpose(1, 2)  # (B, C, T)
        tof_thermal = x[:, :, 11:].transpose(1, 2)  # (B, C, T)
        
        # === Modality Dropout (v5.1) ===
        # 学習時に指定確率でToF/Thermalをドロップ（CMI制約対応）
        # サンプル単位でドロップアウト（より安定した学習）
        if self.training and self.modality_dropout_rate > 0:
            # Create a mask for samples in the batch to drop
            dropout_mask = (torch.rand(batch_size, 1, 1, device=x.device) < self.modality_dropout_rate)
            tof_thermal = tof_thermal.masked_fill(dropout_mask, 0)
        
        # Branch processing
        imu_feat = self.imu_branch(imu)  # (B, 128, T')
        tof_feat = self.tof_branch(tof_thermal)  # (B, 128, T')
        
        # Ensure same temporal dimension
        min_len = min(imu_feat.size(2), tof_feat.size(2))
        imu_feat = imu_feat[:, :, :min_len]
        tof_feat = tof_feat[:, :, :min_len]
        
        # Merge
        merged = torch.cat([imu_feat, tof_feat], dim=1)  # (B, 256, T')
        merged = merged.transpose(1, 2)  # (B, T', 256)
        
        # Parallel RNN processing
        lstm_out, _ = self.lstm(merged)  # (B, T', hidden_dim)
        gru_out, _ = self.gru(merged)  # (B, T', hidden_dim)
        
        # Noise branch (global pooling + dense)
        merged_with_noise = self.gaussian_noise(merged)
        noise_feat = F.adaptive_avg_pool1d(merged_with_noise.transpose(1, 2), 1).squeeze(-1)
        noise_out = self.noise_branch(noise_feat).unsqueeze(1).expand(-1, merged.size(1), -1)
        
        # Combine all features
        combined = torch.cat([lstm_out, gru_out, noise_out], dim=-1)  # (B, T', combined_dim)
        
        # Attention with stability improvements
        combined = self.attention_layernorm(combined)  # LayerNorm for stability
        attention_scores = self.attention(combined)  # (B, T', 1)
        # Scale attention scores to prevent softmax overflow
        import math
        attention_scores = attention_scores / math.sqrt(combined.size(-1))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(combined * attention_weights, dim=1)  # (B, combined_dim)
        
        if return_features:
            return context
        
        # Classification
        class_logits = self.classifier(context)
        bfrb_logits = self.bfrb_classifier(context)
        
        return class_logits, bfrb_logits
    
    def _init_weights(self):
        """重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.zeros_(param.data)


# SimpleTwoBranchLight v5モデル定義
class GaussianNoise(nn.Module):
    """ガウシアンノイズレイヤー（学習時のみ適用）"""
    def __init__(self, sigma=0.09):
        super().__init__()
        self.sigma = sigma
        self.register_buffer('noise', torch.tensor(0))
    
    def forward(self, x):
        if self.training and self.sigma > 0:
            sampled_noise = self.noise.expand(*x.size()).float().normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    """残差接続付きSE-CNNブロック"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += self.skip(residual)
        out = F.relu(out)
        
        return out


class SimpleTwoBranchLight(nn.Module):
    """
    SimpleTwoBranchLight v5 - 改良版Two-Branchモデル
    
    v5の主な改良点:
    - Modality Dropout: 学習時50%でToF/Thermalをドロップ（CMI制約対応）
    - IMU入力次元: 7 → 11（特徴量エンジニアリング追加）
    - ToF入力次元: 320 → 50（統計量20 + 空間特徴量30）
    - Dropout: 段階的調整（0.1→0.3→0.5）
    - RNN: 1層のみ（シンプル化）
    """
    
    def __init__(self, imu_dim=11, tof_dim=50, thermopile_dim=5, 
                 hidden_dim=256, num_classes=18, modality_dropout_rate=0.5):
        super().__init__()
        
        # IMU Branch (ResidualSECNNBlock使用) - v5.6: 容量を増強
        self.imu_branch = nn.Sequential(
            ResidualSECNNBlock(imu_dim, 128, dropout=0.1),  # 64→128に増強
            nn.MaxPool1d(2),
            ResidualSECNNBlock(128, 256, kernel_size=5, dropout=0.1),  # 128→256に増強
            nn.MaxPool1d(2)
        )
        
        # ToF/Thermal Branch (シンプルなCNN)
        tof_thermal_dim = tof_dim + thermopile_dim  # 50 + 5 = 55 (v5で拡張)
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_thermal_dim, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Merge dimension
        merged_dim = 384  # 256 + 128 (IMU強化により増加)
        
        # Parallel RNN (1層のみ、LB 0.76準拠)
        self.lstm = nn.LSTM(
            merged_dim, hidden_dim // 2, 
            num_layers=1,  # 1層に削減
            batch_first=True, 
            dropout=0,  # 1層なのでLSTM内dropoutは0
            bidirectional=True
        )
        
        self.gru = nn.GRU(
            merged_dim, hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Noise branch (LB 0.76準拠)
        self.gaussian_noise = GaussianNoise(0.09)
        self.noise_branch = nn.Sequential(
            nn.Linear(merged_dim, 16),
            nn.ELU()
        )
        
        # Combined dimension: LSTM + GRU + Noise
        combined_dim = hidden_dim * 2 + 16
        
        # Attention mechanism with LayerNorm for stability
        self.attention_layernorm = nn.LayerNorm(combined_dim)
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head (段階的Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # 高め
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 中程度
            
            nn.Linear(128, num_classes)
        )
        
        # Binary BFRB classifier
        self.bfrb_classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )
        
        # Modality Dropout rate (v5.1追加)
        self.modality_dropout_rate = modality_dropout_rate
        
        # 重み初期化
        self._init_weights()
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, _ = x.size()
        
        # Split sensors
        # IMU: 11次元（7 + 4特徴量）
        # ToF統計量: 20次元 + 空間特徴量: 30次元 = 50次元（v5で拡張）
        # Thermopile: 5次元
        imu = x[:, :, :11].transpose(1, 2)  # (B, C, T)
        tof_thermal = x[:, :, 11:].transpose(1, 2)  # (B, C, T)
        
        # === Modality Dropout (v5.1) ===
        # 学習時に指定確率でToF/Thermalをドロップ（CMI制約対応）
        # サンプル単位でドロップアウト（より安定した学習）
        if self.training and self.modality_dropout_rate > 0:
            # Create a mask for samples in the batch to drop
            dropout_mask = (torch.rand(batch_size, 1, 1, device=x.device) < self.modality_dropout_rate)
            tof_thermal = tof_thermal.masked_fill(dropout_mask, 0)
        
        # Branch processing
        imu_feat = self.imu_branch(imu)  # (B, 256, T') - v5.6で増強
        tof_feat = self.tof_branch(tof_thermal)  # (B, 128, T')
        
        # Ensure same temporal dimension
        min_len = min(imu_feat.size(2), tof_feat.size(2))
        imu_feat = imu_feat[:, :, :min_len]
        tof_feat = tof_feat[:, :, :min_len]
        
        # Merge
        merged = torch.cat([imu_feat, tof_feat], dim=1)  # (B, 384, T') - v5.6
        merged = merged.transpose(1, 2)  # (B, T', 384)
        
        # Parallel RNN processing
        lstm_out, _ = self.lstm(merged)  # (B, T', hidden_dim)
        gru_out, _ = self.gru(merged)  # (B, T', hidden_dim)
        
        # Noise branch (global pooling + dense)
        merged_with_noise = self.gaussian_noise(merged)
        noise_feat = F.adaptive_avg_pool1d(merged_with_noise.transpose(1, 2), 1).squeeze(-1)
        noise_out = self.noise_branch(noise_feat).unsqueeze(1).expand(-1, merged.size(1), -1)
        
        # Combine all features
        combined = torch.cat([lstm_out, gru_out, noise_out], dim=-1)  # (B, T', combined_dim)
        
        # Attention with stability improvements
        combined = self.attention_layernorm(combined)  # LayerNorm for stability
        attention_scores = self.attention(combined)  # (B, T', 1)
        # Scale attention scores to prevent softmax overflow
        import math
        attention_scores = attention_scores / math.sqrt(combined.size(-1))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(combined * attention_weights, dim=1)  # (B, combined_dim)
        
        if return_features:
            return context
        
        # Classification
        class_logits = self.classifier(context)
        bfrb_logits = self.bfrb_classifier(context)
        
        return class_logits, bfrb_logits
    
    def _init_weights(self):
        """重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.zeros_(param.data)


# Configuration
class Config:
    # v5モデルのパス（学習ノートブックの出力）
    model_base_path = Path('/kaggle/input/cmi-bfrb-simple-two-branch-light-v5-training')
    
    # Model parameters
    sequence_length = 150
    imu_dim = 11  # v5では11次元
    tof_dim = 50  # v5では50次元（統計量20 + 空間特徴量30）
    thermopile_dim = 5
    hidden_dim = 256
    num_classes = 18
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 5-fold models
    n_folds = 5

config = Config()
print(f"Device: {config.device}")
print(f"Model base path: {config.model_base_path}")


class CMIInferenceDataset(Dataset):
    """推論用データセット"""
    def __init__(self, sequences, sequence_ids, sequence_length=150):
        self.sequences = sequences
        self.sequence_ids = sequence_ids
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_id = self.sequence_ids[idx]
        
        # パディング/トランケート
        if len(seq) < self.sequence_length:
            pad_length = self.sequence_length - len(seq)
            seq = np.pad(seq, ((0, pad_length), (0, 0)), mode='constant')
        else:
            seq = seq[-self.sequence_length:]
        
        return torch.FloatTensor(seq), seq_id


# グローバル変数
models = []
scalers = []
label_encoders = []
MODEL_LOADED = False

print("=== Loading v5 models (5-fold ensemble) ===")

# CV結果の読み込み
cv_results_path = config.model_base_path / 'models' / 'cv_results_v5_convergence.json'
if not cv_results_path.exists():
    # フラット化されている可能性
    cv_results_path = config.model_base_path / 'cv_results_v5_convergence.json'

if cv_results_path.exists():
    with open(cv_results_path, 'r') as f:
        cv_results = json.load(f)
    print(f"CV Mean Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    print(f"Individual fold scores: {cv_results['cv_scores']}")

# 各foldのモデルを読み込み
for fold in range(config.n_folds):
    # モデルパスの候補
    model_paths = [
        config.model_base_path / 'models' / f'simple_two_branch_light_v5_convergence_fold{fold}_best.pth',
        config.model_base_path / f'simple_two_branch_light_v5_convergence_fold{fold}_best.pth'
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if model_path.exists():
            try:
                print(f"\nLoading fold {fold} from: {model_path}")
                
                # チェックポイントを読み込み
                checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
                
                # モデル作成
                model = SimpleTwoBranchLight(
                    imu_dim=config.imu_dim,
                    tof_dim=config.tof_dim,
                    thermopile_dim=config.thermopile_dim,
                    hidden_dim=config.hidden_dim,
                    num_classes=config.num_classes,
                    modality_dropout_rate=0.0  # 推論時はドロップアウトなし
                ).to(config.device)
                
                # 重みを読み込み
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models.append(model)
                
                # スケーラーとエンコーダー
                scalers.append(checkpoint['scaler'])
                if fold == 0:  # 最初のfoldからラベルエンコーダーを取得
                    label_encoder = checkpoint['label_encoder']
                    print(f"Classes: {label_encoder.classes_}")
                
                print(f"✓ Fold {fold} loaded - Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
                model_loaded = True
                break
                
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    if not model_loaded:
        print(f"Warning: Fold {fold} model not found")

MODEL_LOADED = len(models) > 0

if not MODEL_LOADED:
    print("\nError: No models loaded!")
    print("Using fallback configuration...")
    
    # フォールバック設定
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, 66)  # v5は66次元
    scaler.fit(dummy_data)
    scalers = [scaler]
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
                      'Eyebrow - pull hair', 'Eyelash - pull hair', 'Feel around in tray and pull out an object',
                      'Forehead - pull hairline', 'Forehead - scratch', 'Glasses on/off',
                      'Neck - pinch skin', 'Neck - scratch', 'Pinch knee/leg skin',
                      'Pull air toward your face', 'Scratch knee/leg skin', 'Text on phone',
                      'Wave hello', 'Write name in air', 'Write name on leg'])
else:
    print(f"\n✓ Successfully loaded {len(models)} models for ensemble")

print(f"\nModel loaded: {MODEL_LOADED}")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    CMI評価API用の推論関数（v2形式）
    5-foldアンサンブルで予測
    
    Args:
        sequence: センサーデータ（Polars DataFrame）
        demographics: デモグラフィック情報（Polars DataFrame）
        
    Returns:
        予測されたジェスチャー名
    """
    global models, scalers, label_encoder, MODEL_LOADED
    
    try:
        # モデルがロードされていない場合
        if not MODEL_LOADED:
            return 'Wave hello'
        
        # Polars DataFrameをPandasに変換
        test_df = sequence.to_pandas() if isinstance(sequence, pl.DataFrame) else sequence
        
        # シーケンスIDを取得
        seq_id = test_df['sequence_id'].iloc[0] if 'sequence_id' in test_df.columns else 'unknown'
        
        # --- 特徴量エンジニアリング ---
        # NaN補完
        test_df = impute_sensor_nans(test_df)
        
        # IMU特徴量追加
        test_df = add_imu_features(test_df)
        
        # ToF統計量集約
        test_df, _ = aggregate_tof_features(test_df)
        
        # ToF空間特徴量追加（v5新機能）
        test_df, _ = add_tof_spatial_features(test_df)
        
        # 特徴量カラムの取得
        _, _, _, _, all_feature_cols = get_feature_columns()
        
        # 特徴量抽出
        seq_features = test_df[all_feature_cols].values
        
        # NaN/infチェック
        seq_features = np.nan_to_num(seq_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5-foldアンサンブル予測
        all_probs = []
        
        for fold_idx, (model, scaler) in enumerate(zip(models, scalers)):
            # スケーリング
            seq_normalized = scaler.transform(seq_features)
            
            # データセット作成
            dataset = CMIInferenceDataset(
                [seq_normalized],
                [seq_id],
                sequence_length=config.sequence_length
            )
            
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            
            # 予測
            with torch.no_grad():
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(config.device)
                    
                    # モデル予測
                    class_logits, _ = model(batch_data)
                    
                    # 確率計算
                    probabilities = F.softmax(class_logits, dim=1).cpu().numpy()
                    all_probs.append(probabilities[0])
        
        # アンサンブル（確率の平均）
        ensemble_probs = np.mean(all_probs, axis=0)
        prediction = np.argmax(ensemble_probs)
        
        # ジェスチャー名に変換
        gesture_name = label_encoder.inverse_transform([prediction])[0]
        
        # ジェスチャー名の形式を確認（CMI APIはアンダースコア形式を期待）
        gesture_name = gesture_name.replace(' ', '_')
        
        return gesture_name
        
    except Exception as e:
        print(f"エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
        # エラー時はデフォルト予測
        return 'Wave_hello'

print("Predict function defined with 5-fold ensemble")


# CMI推論サーバーの起動
import os
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

# 推論サーバーの作成
print("推論サーバーを作成中...")
inference_server = CMIInferenceServer(predict)

print("\n=== SimpleTwoBranchLight v5 推論準備完了 ===")
print(f"モデル読み込み状態: {MODEL_LOADED}")
if MODEL_LOADED:
    print(f"{len(models)}-fold アンサンブルで推論を行います")
else:
    print("注意: モデルファイルが見つからないため、デフォルト予測を返します")

# 環境に応じて実行
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("\nKaggle競技環境を検出。推論サーバーを開始します。")
    inference_server.serve()
else:
    print("\n推論サーバーを実行します...")
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

