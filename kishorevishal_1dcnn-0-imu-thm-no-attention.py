import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import polars as pl
MAX_LEN = 100





    
def pad_sequences(sequences, maxlen, padding='pre', truncating='pre', value=0.0):
    result = []
    for seq in sequences:
        seq = np.array(seq) 
        
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if seq.ndim == 1:
                pad_shape = (pad_len,)
            else:
                pad_shape = (pad_len,) + seq.shape[1:]
            
            if padding == 'post':
                seq = np.concatenate([seq, np.full(pad_shape, value)])
            else: 
                seq = np.concatenate([np.full(pad_shape, value), seq])
        
        result.append(seq)
    return np.array(result, dtype=np.float32)




import torch
import torch.nn as nn
import torch.nn.functional as F



class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock1D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels//4, 1)
        self.conv_3x1 = nn.Conv1d(in_channels, out_channels//4, 3, padding=1)
        self.conv_5x1 = nn.Conv1d(in_channels, out_channels//4, 5, padding=2)
        self.conv_7x1 = nn.Conv1d(in_channels, out_channels//4, 7, padding=3)
        
    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x1(x)
        out3 = self.conv_5x1(x)
        out4 = self.conv_7x1(x)
        return torch.cat([out1,out2, out3,out4], dim=1)

class ResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 dropout: float = 0.0):
        super().__init__()

        padding = ((kernel_size - 1) // 2) * dilation

        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Skip (identity) path
        if in_channels != out_channels or stride != 1:
            # Adjust channels and/or temporal length
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop(out)           # optional
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity                # residual connection
        out  = self.activation(out)
        return out



class GestureCNN(nn.Module):
    def __init__(self, input_dim, num_classes, max_len=100, dropout=0.3,hidden_dims=None):
        super(GestureCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_len = max_len

        self.dense1 = nn.Linear(input_dim, 300)
        self.bn_dense = nn.BatchNorm1d(300) 

        self.conv_blocks = nn.ModuleList()
        in_channels = 300
        if hidden_dims == None:
            hidden_dims = [64, 128, 192, 256]
        
        for i, out_channels in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1:
                kernel_size = 3
            else:
                kernel_size = 5 if i == 1 else 3
            layers = []
            if i == 0:
                layers.append(MultiScaleTemporalConv(in_channels, out_channels))
            else:
                layers.extend( [
                    ResidualBlock1D(in_channels, out_channels,
                            kernel_size=kernel_size,
                            # padding=kernel_size // 2,
                            # bias=False if i < len(hidden_dims) - 1 else True,
                            dropout=dropout),
                    # nn.BatchNorm1d(out_channels) if i < len(hidden_dims) - 1 else nn.Identity(),
                    # nn.ReLU(),
                ] )

            # Inject SE block in middle conv layers (e.g., i == 1 or 2)
            # if 0 < i < len(hidden_dims) - 1:
            #     layers.append(SEBlock1D(out_channels))

            # Add pooling conditionally
            if i in [1, 2]:
                layers.append(nn.MaxPool1d(2))
            elif i == len(hidden_dims) - 1:
                layers.append(nn.AdaptiveAvgPool1d(1))

            # dropout_rate = 0.2 + (0.1 * i)
            layers.append(nn.Dropout(dropout))

            conv_block = nn.Sequential(*layers)
            self.conv_blocks.append(conv_block)
            in_channels = out_channels

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]//2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dims[-1]//2, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Layer 1: Dense layer applied to each time step
        # Reshape to apply dense layer: (batch_size * seq_len, features)
        x_reshaped = x.reshape(-1, features)
        x_dense = torch.relu(self.bn_dense(self.dense1(x_reshaped)))
        # Reshape back: (batch_size, seq_len, 300)
        x = x_dense.reshape(batch_size, seq_len, 300)
        
        # Input shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # Convert to (batch_size, features, seq_len)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x



# class GestureTHMCNN(nn.Module):
#     def __init__(self, input_dim, num_classes=18, hidden_dims=None, max_len=100, dropout=0.3):
#         super(GestureTHMCNN, self).__init__()
        
#         self.input_dim = 16
#         self.num_classes = num_classes
#         self.max_len = max_len
#         hidden_dims = [64, 128, 256, 512]

#         # self.temporal_attention = TemporalSelfAttention(input_dim)

#         self.conv_blocks = nn.ModuleList()
#         in_channels = 16
        
#         for i, out_channels in enumerate(hidden_dims):
#             if i == len(hidden_dims) - 1:
#                 kernel_size = 3
#             else:
#                 kernel_size = 5 if i == 1 else 3
#             layers = []
#             if i == 0:
#                 layers.append(MultiScaleTemporalConv(in_channels, out_channels))
#             else:
#                 layers.extend( [
#                     nn.Conv1d(in_channels, out_channels,
#                             kernel_size=kernel_size,
#                             padding=kernel_size // 2,
#                             bias=False if i < len(hidden_dims) - 1 else True),
#                     nn.BatchNorm1d(out_channels) if i < len(hidden_dims) - 1 else nn.Identity(),
#                     nn.GELU() if i == 0 else nn.ReLU(),
#                 ] )

#             # Inject SE block in middle conv layers (e.g., i == 1 or 2)
#             if 0 < i < len(hidden_dims) - 1:
#                 layers.append(SEBlock1D(out_channels))

#             # Add pooling conditionally
#             if i in [1, 2]:
#                 layers.append(nn.MaxPool1d(2))
#             elif i == len(hidden_dims) - 1:
#                 layers.append(nn.AdaptiveAvgPool1d(1))

#             # dropout_rate = 0.2 + (0.1 * i)
#             layers.append(nn.Dropout(dropout))

#             conv_block = nn.Sequential(*layers)
#             self.conv_blocks.append(conv_block)
#             in_channels = out_channels
        
#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dims[-1], hidden_dims[-1]//2),
#             nn.LeakyReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dims[-1]//2, num_classes)
#         )
        
#     def forward(self, x):
#         # Input shape: (batch_size, seq_len, features)
#         x = x.transpose(1, 2)  # Convert to (batch_size, features, seq_len)
        
#         # x = self.temporal_attention(x)
        
#         for conv_block in self.conv_blocks:
#             x = conv_block(x)
        
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.classifier(x)
#         return x




class ModifiedGatedTwoBranchCNN(nn.Module):
    def __init__(self, input_dim, num_classes, max_len=100, dropout=0.3, checkpoint_path=None, hidden_dims=[48, 96, 192,256]):
        super(ModifiedGatedTwoBranchCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.imu_dim = input_dim - 5
        self.tof_dim = 5
        self.hidden_dims = hidden_dims 

        # TOF dense layer (trainable from scratch)
        self.tof_dense = nn.Linear(self.tof_dim, 150)
        self.tof_bn = nn.BatchNorm1d(150)
        
        # IMU feature extractor: Use GestureCNN without classifier
        gesture_cnn = GestureCNN(
            input_dim=self.imu_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims,
            max_len=max_len,
            dropout=dropout
        )
        
        self.imu_dense1 = gesture_cnn.dense1
        self.imu_bn_dense = gesture_cnn.bn_dense
        self.imu_conv_blocks = gesture_cnn.conv_blocks
        
        # TOF branch (trainable from scratch)
        self.tof_branch = nn.Sequential(
            MultiScaleTemporalConv(150, 48),
            ResidualBlock1D(48, 96, kernel_size=3, dropout=dropout),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            ResidualBlock1D(96, 192, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # Auxiliary unimodal networks
        self.aux_imu = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self.aux_tof = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(192, 64),  # Assuming 192 for TOF
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Fusion and main classifier (unchanged)
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(self.hidden_dims[-1]+192, 128),  # Adjust for hidden_dims[-1]
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1]+192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        imu = x[:, :self.imu_dim, :]
        tof = x[:, self.imu_dim:, :]

        # Process IMU with feature extractor (no classifier)
        batch_size, _, seq_len = imu.shape
        imu = imu.transpose(1, 2)  # (batch, seq_len, imu_dim)
        imu_reshaped = imu.reshape(-1, self.imu_dim)
        imu_dense = F.relu(self.imu_bn_dense(self.imu_dense1(imu_reshaped)))
        imu = imu_dense.reshape(batch_size, seq_len, 300)
        imu = imu.transpose(1, 2)  # (batch, 300, seq_len)
        
        for conv_block in self.imu_conv_blocks:
            imu = conv_block(imu)
        
        imu_features = imu  
        
        # Process TOF
        tof = tof.transpose(1, 2)  # (batch, seq_len, tof_dim)
        tof_reshaped = tof.reshape(-1, self.tof_dim)
        tof_dense = F.relu(self.tof_bn(self.tof_dense(tof_reshaped)))
        tof = tof_dense.reshape(batch_size, seq_len, 150)
        tof = tof.transpose(1, 2)  # (batch, 150, seq_len)
        tof_features = self.tof_branch(tof)
        
        # Global pooling
        imu_pooled = self.global_pool(imu_features).flatten(1)
        tof_pooled = self.global_pool(tof_features).flatten(1)
        
        # Auxiliary outputs
        aux_imu_out = self.aux_imu(imu_features)
        aux_tof_out = self.aux_tof(tof_features)
        
        # Fusion
        concatenated = torch.cat([imu_pooled, tof_pooled], dim=1)
        fusion_weights = self.fusion_weight_net(concatenated)
        
        weighted_imu = imu_pooled * fusion_weights[:, 0:1]
        weighted_tof = tof_pooled * fusion_weights[:, 1:2]
        
        fused_features = torch.cat([weighted_imu, weighted_tof], dim=1)
        main_out = self.classifier(fused_features)
        
        return main_out, aux_imu_out, aux_tof_out, fusion_weights





class GestureTHMCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=None, max_len=100, dropout=0.3):
        super(GestureTHMCNN, self).__init__()
        
        self.input_dim = input_dim
        self.non_thm_dim = input_dim - 5
        self.thm_dim = 5
        self.num_classes = num_classes
        self.max_len = max_len
        hidden_dims = [48, 96, 192, 256]

        # Main branch (non-THM features)
        self.main_dense1 = nn.Linear(self.non_thm_dim, 300)
        self.main_bn_dense = nn.BatchNorm1d(300)

        self.main_conv_blocks = nn.ModuleList()
        main_in_channels = 300
        
        for i, out_channels in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1:
                kernel_size = 3
            else:
                kernel_size = 5 if i == 1 else 3
            layers = []
            if i == 0:
                layers.append(MultiScaleTemporalConv(main_in_channels, out_channels))
            else:
                layers.extend([
                    ResidualBlock1D(main_in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    dropout=dropout),
                ])

            # Add pooling conditionally
            if i in [1, 2]:
                layers.append(nn.MaxPool1d(2))
            elif i == len(hidden_dims) - 1:
                layers.append(nn.AdaptiveAvgPool1d(1))

            layers.append(nn.Dropout(dropout))

            conv_block = nn.Sequential(*layers)
            self.main_conv_blocks.append(conv_block)
            main_in_channels = out_channels

        # THM branch (THM features, different structure)
        self.thm_dense1 = nn.Linear(self.thm_dim, 100) if self.thm_dim > 0 else None
        self.thm_bn_dense = nn.BatchNorm1d(100) if self.thm_dim > 0 else None
        
        self.thm_conv_blocks = nn.ModuleList()
        thm_in_channels = 100
        thm_hidden_dims = [32, 64, 128]  # Different (lighter) than main
        
        for i, out_channels in enumerate(thm_hidden_dims):
            kernel_size = 3  # Simpler kernels for THM
            layers = []
            if i == 0:
                layers.append(MultiScaleTemporalConv(thm_in_channels, out_channels))
            else:
                layers.extend([
                    ResidualBlock1D(thm_in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    dropout=dropout),
                ])

            # Different pooling: fewer pools
            if i in [1]:
                layers.append(nn.MaxPool1d(2))
            elif i == len(thm_hidden_dims) - 1:
                layers.append(nn.AdaptiveAvgPool1d(1))

            layers.append(nn.Dropout(dropout))

            conv_block = nn.Sequential(*layers)
            self.thm_conv_blocks.append(conv_block)
            thm_in_channels = out_channels

        # Classifier head with merged dimensions
        main_dim = hidden_dims[-1]
        thm_dim_final = thm_hidden_dims[-1] if self.thm_dim > 0 else 0
        merged_dim = main_dim + thm_dim_final
        self.classifier = nn.Sequential(
            nn.Linear(merged_dim, merged_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(merged_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Input: (batch_size, seq_len, total_features)
        batch_size, seq_len, total_features = x.shape
        
        # Internal split
        main_x = x[:, :, :self.non_thm_dim]
        thm_x = x[:, :, self.non_thm_dim:] if self.thm_dim > 0 else None

        # Main branch processing
        main_reshaped = main_x.reshape(-1, self.non_thm_dim)
        main_dense = torch.relu(self.main_bn_dense(self.main_dense1(main_reshaped)))
        main_x = main_dense.reshape(batch_size, seq_len, 300)
        main_x = main_x.transpose(1, 2)  # (batch_size, 300, seq_len)
        
        for conv_block in self.main_conv_blocks:
            main_x = conv_block(main_x)
        
        main_x = main_x.view(batch_size, -1)  # Flatten

        # THM branch processing
        if self.thm_dim > 0:
            thm_reshaped = thm_x.reshape(-1, self.thm_dim)
            thm_dense = torch.relu(self.thm_bn_dense(self.thm_dense1(thm_reshaped)))
            thm_x = thm_dense.reshape(batch_size, seq_len, 100)
            thm_x = thm_x.transpose(1, 2)  # (batch_size, 100, seq_len)
            
            for conv_block in self.thm_conv_blocks:
                thm_x = conv_block(thm_x)
            
            thm_x = thm_x.view(batch_size, -1)  # Flatten
        else:
            thm_x = torch.empty(batch_size, 0, device=x.device)  # Empty if no THM

        # Concatenate and classify
        merged = torch.cat((main_x, thm_x), dim=1)
        out = self.classifier(merged)
        return out









import pytorch_lightning as pl_

class GestureLightningModule(pl_.LightningModule):
    def __init__(self, input_dim, num_classes=18, max_len=100, dropout=0.3):
        super().__init__()
        self.model = GestureCNN(input_dim, num_classes)
    
    def forward(self, x):
        return self.model(x)

class GestureTHMLightningModule(pl_.LightningModule):
    def __init__(self, input_dim, num_classes=18, max_len=100, dropout=0.3):
        super().__init__()
        self.model = ModifiedGatedTwoBranchCNN(input_dim, num_classes, max_len, dropout)
    
    def forward(self, x):
        return self.model(x)






from pathlib import Path
import torch
from dataclasses import dataclass
import sys
from types import ModuleType

# Define your config classes exactly as they were during training
@dataclass
class DataConfig:
    raw_dir: str = "./input"
    max_len: int = 100
    batch_size: int = 64
    num_workers: int = 8
    mixup_alpha: float = 0.4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass
class ModelConfig:
    input_dim: int = 11
    num_classes: int = 20
    dropout: float = 0.3
    hidden_dims: list = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [48, 96, 192, 256]

@dataclass
class TrainingConfig:
    lr: float = 5e-4
    weight_decay: float = 2e-3
    epochs: int = 350
    patience: int = 40
    label_smoothing: float = 0.1
    gradient_clip_val: float = None
    accumulate_grad_batches: int = 1

@dataclass
class ExperimentConfig:
    seed: int = 1408
    n_folds: int = 5
    export_dir: str = "./models"
    project_name: str = "gesture_recognition"
    experiment_name: str = "cnn_mixup"
    
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

# Register config module
config_module = ModuleType('config')
config_module.DataConfig = DataConfig
config_module.ModelConfig = ModelConfig
config_module.TrainingConfig = TrainingConfig
config_module.ExperimentConfig = ExperimentConfig
sys.modules['config'] = config_module

def load_lightning_checkpoint(model_path, device):
    """Load model from Lightning checkpoint"""
    try:
        model_ = GestureCNN(input_dim=15, num_classes=num_classes)
        model = GestureLightningModule.load_from_checkpoint(
            str(model_path),
            input_dim = 15,
            strict=False,
            
        )
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded Lightning model from {model_path.name}")
        return model.model  # Return the inner GestureCNN model
        
    except Exception as e:
        print(f"❌ Failed to load {model_path.name}: {e}")
        return None

# Loading code
# Define base export path
EXPORT_ROOT = Path("/kaggle/input/cmi-1dcnn/pytorch/default/26")
scaler = joblib.load(EXPORT_ROOT / "scaler.pkl")
gesture_classes = np.load(EXPORT_ROOT / "gesture_classes.npy", allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.n_features_in_
num_classes = len(gesture_classes)
print(input_dim)

fold_models = []
for fold in [0,2,3,4,1]:
    path = f"imu_fold_{fold}.ckpt"
    model_path = EXPORT_ROOT / path
    model = load_lightning_checkpoint(model_path, device)
    if model is not None:
        fold_models.append(model)

print(f"✓ Successfully loaded {len(fold_models)} models")




from pathlib import Path
import torch
from dataclasses import dataclass
import sys
from types import ModuleType

# Loading code
# Define base export path
EXPORT_ROOT = Path("/kaggle/input/cmi-1dcnn/pytorch/default/32")
scaler_all = joblib.load("/kaggle/input/cmi-1dcnn/pytorch/default/32/scaler.pkl")
gesture_classes_all = np.load(EXPORT_ROOT / "gesture_classes.npy", allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler_all.n_features_in_
num_classes = len(gesture_classes_all)
print(input_dim)

# Define your config classes exactly as they were during training
@dataclass
class DataConfig:
    raw_dir: str = "./input"
    max_len: int = 100
    batch_size: int = 64
    num_workers: int = 8
    mixup_alpha: float = 0.4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass
class ModelConfig:
    input_dim: int = 11
    num_classes: int = 20
    dropout: float = 0.3
    hidden_dims: list = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]

@dataclass
class TrainingConfig:
    lr: float = 5e-4
    weight_decay: float = 2e-3
    epochs: int = 350
    patience: int = 40
    label_smoothing: float = 0.1
    gradient_clip_val: float = None
    accumulate_grad_batches: int = 1

@dataclass
class ExperimentConfig:
    seed: int = 1408
    n_folds: int = 5
    export_dir: str = "./models"
    project_name: str = "gesture_recognition"
    experiment_name: str = "cnn_mixup"
    
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

# Register config module
config_module = ModuleType('config')
config_module.DataConfig = DataConfig
config_module.ModelConfig = ModelConfig
config_module.TrainingConfig = TrainingConfig
config_module.ExperimentConfig = ExperimentConfig
sys.modules['config'] = config_module

def load_lightning_checkpoint(model_path, device):
    """Load model from Lightning checkpoint"""
    try:
        model_ = ModifiedGatedTwoBranchCNN(input_dim=input_dim, num_classes=num_classes)
        model = GestureTHMLightningModule.load_from_checkpoint(
            str(model_path),
            input_dim = input_dim,
            strict=False,
            
        )
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded Lightning model from {model_path.name}")
        return model.model  # Return the inner GestureCNN model
        
    except Exception as e:
        print(f"❌ Failed to load {model_path.name}: {e}")
        return None



thm_fold_models = []
for fold in [0,1,2,3,4]:
    path = f"thm_imu_fold_{fold}.ckpt"
    model_path = EXPORT_ROOT / path
    model = load_lightning_checkpoint(model_path, device)
    if model is not None:
        thm_fold_models.append(model)

print(f"✓ Successfully loaded {len(thm_fold_models)} models")



# def load_trained_model(model_path, scaler_path, classes_path, device):
#     """Load trained model and preprocessing components"""
    
#     # Load model checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
    
#     # Load preprocessing components
#     scaler = joblib.load(scaler_path)
#     gesture_classes = np.load(classes_path,allow_pickle=True)
    
#     # Initialize model architecture (same as training)
#     input_dim = scaler.n_features_in_  # Get from fitted scaler
#     num_classes = 18
    
#     model = GestureCNN(input_dim=input_dim, num_classes=num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
    
#     print(f"✓ Model loaded from {model_path}")
#     print(f"✓ Model trained for {len(gesture_classes)} gesture classes")
    
#     return model, scaler, gesture_classes

# EXPORT_DIR = Path("/kaggle/input/cmi-1dcnn/pytorch/default/10")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, scaler, gesture_classes = load_trained_model(
#     model_path=EXPORT_DIR / 'best_gesture_model.pth',
#     scaler_path=EXPORT_DIR / 'scaler.pkl', 
#     classes_path=EXPORT_DIR / 'gesture_classes.npy',
#     device=device
# )












def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    mat = scaler.transform(mat)
    return mat.astype('float32')


import polars as pl
from collections import Counter



def _imu_features(df):
    df = df.with_columns([
    (
            (pl.col("acc_x")**2 + pl.col("acc_y")**2 + pl.col("acc_z")**2).sqrt()
            .fill_null(0)
        ).alias("acc_mag"),
        (2 * pl.col("rot_w").clip(-1, 1).arccos()).alias("rot_angle")
    ])
    
    df = df.with_columns([
        pl.col("acc_mag").diff().over("sequence_id").fill_null(0).alias("acc_mag_jerk"),
        pl.col("rot_angle").diff().over("sequence_id").fill_null(0).alias("rot_angle_vel")
    ])
    return df


def calculate_angular_velocity_from_quat(df, time_delta=1/200): # Assuming 200Hz sampling rate
    from scipy.spatial.transform import Rotation as R

    quat_values = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    n_samples   = quat_values.shape[0]
    angular_vel = np.zeros((n_samples, 3))

    for i in range(n_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
        np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(df):
    """
    Calculates a scalar angular distance for each sample.
    """
    from scipy.spatial.transform import Rotation as R
    quat_values  = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    n_samples    = quat_values.shape[0]
    angular_dist = np.zeros(n_samples)

    for i in range(n_samples - 1):
        q1, q2 = quat_values[i], quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
        np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 
            pass
            
    return angular_dist





acc_cols = [f'acc_{axis}' for axis in ['x', 'y', 'z']]
linear_acc_cols = [f'linear_acc_{axis}' for axis in ['x', 'y', 'z']]
world_acc_cols = [f'world_acc_{axis}' for axis in ['x', 'y', 'z']]
rot_cols = [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']]
thm_cols = [f'thm_{i}' for i in range(1, 5)]
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

imu_cols = linear_acc_cols + rot_cols
imu_cols = [f'linear_acc_{axis}' for axis in ['x', 'y', 'z']] + \
                   [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']] + ['ang_dist','acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel','ang_vel_x', 'ang_vel_y', 'ang_vel_z']
imu_cols_thm = linear_acc_cols + rot_cols + ['ang_dist','acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel','ang_vel_x', 'ang_vel_y', 'ang_vel_z'] + thm_cols

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global scaler, model, device,gesture_classes
    
    # has_thm_data = not sequence.select(thm_cols).null_count().row(0).values().count(sequence.height) == len(thm_cols)
    
    thm_nulls = sequence.select(thm_cols).null_count().row(0)
    thm_non_null_ratio = [
        1.0 - (null_count / sequence.height) for null_count in thm_nulls
    ]
    # Check if all thermal columns have more than 85% non-null data
    has_thm_data = all(ratio > 0.85 for ratio in thm_non_null_ratio)

    sequence = sequence.with_columns([
            # Compute gravity components
            (2.0 * (pl.col('rot_x') * pl.col('rot_z') - pl.col('rot_w') * pl.col('rot_y'))).alias('gx'),
            (2.0 * (pl.col('rot_w') * pl.col('rot_x') + pl.col('rot_y') * pl.col('rot_z'))).alias('gy'),
            (pl.col('rot_w')**2 - pl.col('rot_x')**2 - pl.col('rot_y')**2 + pl.col('rot_z')**2).alias('gz')
        ]).with_columns([
            # Compute norm and scale
            (pl.col('gx')**2 + pl.col('gy')**2 + pl.col('gz')**2).sqrt().alias('norm')
        ]).with_columns([
            # Apply scaling (avoid division by zero)
            pl.when(pl.col('norm') < 1e-6)
            .then(0.0)
            .otherwise(9.81 / pl.col('norm'))
            .alias('scale')
        ]).with_columns([
            # Scale gravity components
            (pl.col('gx') * pl.col('scale')).alias('gravity_x'),
            (pl.col('gy') * pl.col('scale')).alias('gravity_y'),
            (pl.col('gz') * pl.col('scale')).alias('gravity_z')
        ]).with_columns([
            # Remove gravity from accelerometer data
            (pl.col('acc_x') - pl.col('gravity_x')).alias('linear_acc_x'),
            (pl.col('acc_y') - pl.col('gravity_y')).alias('linear_acc_y'),
            (pl.col('acc_z') - pl.col('gravity_z')).alias('linear_acc_z')
        ]).drop(['gx', 'gy', 'gz', 'norm', 'scale', 'gravity_x', 'gravity_y', 'gravity_z'])
    sequence = _imu_features(sequence)
    df_seq = sequence.to_pandas()
    df_seq[["ang_vel_x", "ang_vel_y", "ang_vel_z"]] = calculate_angular_velocity_from_quat(df_seq) 
    df_seq["ang_dist"] = calculate_angular_distance(df_seq)

    if not has_thm_data:
        print("IMU ONLY")
        processed = preprocess_sequence(df_seq, imu_cols,scaler)
        padded_sequence = pad_sequences([processed], maxlen=100, 
                                      padding='pre', truncating='pre')
        input_tensor = torch.tensor(padded_sequence, dtype=torch.float32).to(device)
    
        preds = []
        for model in fold_models:
            model.eval()
            with torch.no_grad():
                logits = model(input_tensor)
                pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
                preds.append(pred_idx)
        final_idx = Counter(preds).most_common(1)[0][0]
        return str(gesture_classes[final_idx])
    else:
        print("THM USED")
        processed = preprocess_sequence(df_seq, imu_cols_thm,scaler_all)
        padded_sequence = pad_sequences([processed], maxlen=100, 
                                      padding='pre', truncating='pre')
        input_tensor = torch.tensor(padded_sequence, dtype=torch.float32).to(device)
        preds = []
        for model in thm_fold_models:
            model.eval()
            with torch.no_grad():
                logits,_,_,_ = model(input_tensor)
                pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
                preds.append(pred_idx)
        final_idx = Counter(preds).most_common(1)[0][0]
        return str(gesture_classes_all[final_idx])
                

    # Majority vote
    







# sequence_id	gesture
# str	str
# "SEQ_000011"	"Eyelash - pull hair"
# "SEQ_000001"	"Eyebrow - pull hair"



# Kaggle competition interface
import kaggle_evaluation.cmi_inference_server
import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


# pl.read_parquet("/kaggle/working/submission.parquet")







