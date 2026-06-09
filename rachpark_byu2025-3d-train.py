import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Define global constants
DATA_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = './'
MODEL_DIR = './models'

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True) 
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True


class TomogramDataset(Dataset):
    """
    Dataset for loading 3D tomograms from stacks of 2D JPG slices.
    Handles both training and test data.
    """
    def __init__(self, root_dir, max_slices=64, target_size=(128, 128), train=True, csv_file=None, exclude_no_motor=False):
        self.root_dir = root_dir
        self.max_slices = max_slices
        self.target_size = target_size
        self.train = train
        self.exclude_no_motor = exclude_no_motor
        
        if train:
            # Training mode - load labels from CSV
            if csv_file is None:
                raise ValueError("csv_file must be provided for training mode")
            self.labels_df = pd.read_csv(csv_file)
            self.process_metadata()
        else:
            # Test mode - get tomogram directories directly
            self.tomo_dirs = sorted([d for d in os.listdir(root_dir) 
                                   if os.path.isdir(os.path.join(root_dir, d))])
        
        # Cache file paths
        self.cache_file_paths()
    
    def process_metadata(self):
        """
        모든 motor를 개별 데이터 포인트로 처리하도록 수정
        각 motor마다 고유한 ID 부여
        exclude_no_motor가 True인 경우 motor가 없는 tomogram 제외
        """
        processed_data = []
        
        # 각 tomogram에 대해
        for tomo_id in self.labels_df['tomo_id'].unique():
            tomo_rows = self.labels_df[self.labels_df['tomo_id'] == tomo_id]
            
            # 기본 tomogram 정보 가져오기
            base_info = {
                'original_tomo_id': tomo_id,  # 원본 tomo_id 보존
                'Array shape (axis 0)': tomo_rows['Array shape (axis 0)'].iloc[0],
                'Array shape (axis 1)': tomo_rows['Array shape (axis 1)'].iloc[0],
                'Array shape (axis 2)': tomo_rows['Array shape (axis 2)'].iloc[0],
                'Voxel spacing': tomo_rows['Voxel spacing'].iloc[0],
                'Number of motors': tomo_rows['Number of motors'].iloc[0]
            }
            
            num_motors = base_info['Number of motors']
            
            if num_motors == 0:
                # motor가 없는 경우
                if not self.exclude_no_motor:  # exclude_no_motor가 False일 때만 추가
                    motor_info = base_info.copy()
                    motor_info.update({
                        'tomo_id': f"{tomo_id}_no_motor",  # 고유 ID 생성
                        'Motor axis 0': -1,
                        'Motor axis 1': -1,
                        'Motor axis 2': -1
                    })
                    processed_data.append(motor_info)
            else:
                # motor가 있는 경우 - 각 motor를 개별 데이터 포인트로 추가
                motor_rows = tomo_rows[tomo_rows['Motor axis 0'] != -1]
                for motor_idx, motor_row in enumerate(motor_rows.iterrows()):
                    motor_info = base_info.copy()
                    _, row = motor_row
                    motor_info.update({
                        'tomo_id': f"{tomo_id}_motor_{motor_idx}",  # 고유 ID 생성
                        'Motor axis 0': row['Motor axis 0'],
                        'Motor axis 1': row['Motor axis 1'],
                        'Motor axis 2': row['Motor axis 2']
                    })
                    processed_data.append(motor_info)
        
        # DataFrame으로 변환
        self.tomo_df = pd.DataFrame(processed_data)
        
        # 데이터셋 정보 출력
        total_samples = len(self.tomo_df)
        motor_samples = len(self.tomo_df[self.tomo_df['Motor axis 0'] != -1])
        print(f"Dataset statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Samples with motors: {motor_samples}")
        print(f"Samples without motors: {total_samples - motor_samples}")
    
    def cache_file_paths(self):
        """
        파일 경로 캐싱 메서드 수정
        original_tomo_id를 사용하여 파일 경로 찾기
        """
        self.slice_files = {}
        
        if self.train:
            for _, row in self.tomo_df.iterrows():
                tomo_id = row['original_tomo_id']  # 원본 tomo_id 사용
                if tomo_id not in self.slice_files:  # 중복 처리 방지
                    tomo_dir = os.path.join(self.root_dir, tomo_id)
                    files = sorted(glob.glob(os.path.join(tomo_dir, '*.jpg')))
                    self.slice_files[tomo_id] = files
        else:
            for tomo_id in self.tomo_dirs:
                tomo_dir = os.path.join(self.root_dir, tomo_id)
                files = sorted(glob.glob(os.path.join(tomo_dir, '*.jpg')))
                self.slice_files[tomo_id] = files
    
    def load_volume(self, tomo_id):
        """
        볼륨 로딩 메서드 수정
        original_tomo_id를 사용하여 파일 로드
        """
        # tomo_id에서 원본 ID 추출
        original_tomo_id = tomo_id.split('_motor_')[0] if '_motor_' in tomo_id else tomo_id.split('_no_motor')[0]
        files = self.slice_files[original_tomo_id]
        
        # Get array shape
        z_shape = len(files)
        if z_shape > 0:
            img = Image.open(files[0])
            x_shape, y_shape = img.size
        else:
            raise ValueError(f"No slices found for tomogram {tomo_id}")
        
        # Store original shape
        original_shape = np.array([z_shape, x_shape, y_shape])
        
        # Determine which slices to load
        if self.max_slices is not None and z_shape > self.max_slices:
            indices = np.linspace(0, z_shape-1, self.max_slices, dtype=int)
            files_to_load = [files[i] for i in indices]
        else:
            files_to_load = files
        
        # Load slices
        slices = []
        for file_path in files_to_load:
            img = Image.open(file_path).convert('L')
            img = img.resize(self.target_size, Image.BILINEAR)
            slices.append(np.array(img))
        
        # Stack slices to form volume
        volume = np.stack(slices)
        
        # Pad if needed
        if self.max_slices is not None and volume.shape[0] < self.max_slices:
            pad_width = self.max_slices - volume.shape[0]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        
        # Normalize to [0, 1]
        volume = volume.astype(np.float32) / 255.0
        
        return volume, original_shape

    def __len__(self):
        return len(self.tomo_df) if self.train else len(self.tomo_dirs)
        
    def __getitem__(self, idx):
        if self.train:
            row = self.tomo_df.iloc[idx]
            tomo_id = row['tomo_id']
            
            # Load volume
            volume, _ = self.load_volume(tomo_id)
            
            # Get labels
            motor_axes = row[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.astype(np.float32)
            has_motor = not (motor_axes == -1).all()
            
            # Process coordinates
            if not has_motor:
                motor_axes = np.zeros(3, dtype=np.float32)
            else:
                # Get original shape
                array_shape = np.array([
                    row['Array shape (axis 0)'],
                    row['Array shape (axis 1)'],
                    row['Array shape (axis 2)']
                ], dtype=np.float32)
                
                # Apply data augmentation (random jitter to coordinates)
                if random.random() < 0.5:
                    jitter_z = np.random.uniform(-0.05, 0.05) * array_shape[0]
                    jitter_x = np.random.uniform(-0.05, 0.05) * array_shape[1]
                    jitter_y = np.random.uniform(-0.05, 0.05) * array_shape[2]
                    
                    motor_axes[0] += jitter_z
                    motor_axes[1] += jitter_x
                    motor_axes[2] += jitter_y
                    
                    # Ensure coordinates are within bounds
                    motor_axes[0] = max(0, min(motor_axes[0], array_shape[0] - 1))
                    motor_axes[1] = max(0, min(motor_axes[1], array_shape[1] - 1))
                    motor_axes[2] = max(0, min(motor_axes[2], array_shape[2] - 1))
                
                # Normalize coordinates to [0, 1]
                motor_axes[0] = motor_axes[0] / array_shape[0]
                motor_axes[1] = motor_axes[1] / array_shape[1]
                motor_axes[2] = motor_axes[2] / array_shape[2]
            
            # Convert to tensor and ensure correct shape
            # volume shape should be [C, D, H, W] for single sample
            volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
            motor_axes = torch.from_numpy(motor_axes)
            has_motor = torch.tensor([float(has_motor)])
            
            return {
                'tomo_id': tomo_id,
                'volume': volume,  # Shape: [1, D, H, W]
                'has_motor': has_motor,
                'motor_axes': motor_axes,
                'original_shape': torch.tensor([
                    row['Array shape (axis 0)'],
                    row['Array shape (axis 1)'],
                    row['Array shape (axis 2)']
                ], dtype=torch.float32),
                'voxel_spacing': torch.tensor([row['Voxel spacing']], dtype=torch.float32)
            }
        else:
            # Test mode
            tomo_id = self.tomo_dirs[idx]
            
            # Load volume
            volume, original_shape = self.load_volume(tomo_id)
            
            # Convert to tensor and ensure correct shape
            volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
            
            return {
                'tomo_id': tomo_id,
                'volume': volume,  # Shape: [1, D, H, W]
                'original_shape': torch.tensor(original_shape, dtype=torch.float32)
            }


from torchvision.ops import StochasticDepth
from typing import List, Dict
from torch import Tensor
#from torchtune.modules import RotaryPositionalEmbeddings

class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv3d(in_features, out_features, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(num_groups=1, num_channels=out_features)
        )

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)),
                                    requires_grad=True)

    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv3d(
                in_features, in_features, kernel_size=(2, 7, 7), padding='same', bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=in_features, num_channels=in_features),
            # wide -> wide
            nn.Conv3d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv3d(expanded_features, out_features, kernel_size=1),
        )
        #self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        #self.drop_path = StochasticDepth(drop_p, mode="batch")


    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        #x = self.layer_scaler(x)
        #x = self.drop_path(x)
        x += res
        return x

class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=in_features, num_channels=in_features),
                nn.Conv3d(in_features, out_features, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class ConvNextClassifier(nn.Module):
    def __init__(self, max_slices):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=max_slices, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512),
                                     )
        self.l1 = nn.Linear(512, max_slices) # (exists, not exists, exists, ... )

    def forward(self, x, label=None):
        x = self.encoder(x)
        x = self.flatten(x)
        l1 = self.l1(x)

        return l1
    
class ConvNextRegressor(nn.Module):
    def __init__(self, max_slices):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=max_slices, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512),
                                     )
        self.l1 = nn.Linear(512, 3)

    def forward(self, x, label=None):
        x = self.encoder(x)
        x = self.flatten(x)
        l1 = self.l1(x)

        return l1


class FlagellarMotorClassificationLoss(nn.Module):
    """
    Loss function for motor presence detection (classification task).
    Uses binary cross-entropy loss.
    """
    def __init__(self):
        super(FlagellarMotorClassificationLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, presence_pred, presence_true):
        """
        Args:
            presence_pred (Tensor): Predicted probability of motor presence [B, 1]
            presence_true (Tensor): Ground truth motor presence [B, 1]
        Returns:
            Tensor: Classification loss
        """
        return self.bce_loss(presence_pred, presence_true)

class FlagellarMotorRegressorLoss(nn.Module):
    """
    Loss function for motor location regression.
    Uses MSE loss only for samples with motors present.
    """
    def __init__(self):
        super(FlagellarMotorRegressorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, location_pred, location_true, presence_true):
        """
        Args:
            location_pred (Tensor): Predicted motor coordinates [B, 4]
            location_true (Tensor): Ground truth motor coordinates [B, 4]
        Returns:
            Tuple[Tensor, Tensor]: (Location loss, Average Euclidean distance)
        """
        # Initialize loss and distance
        location_loss = torch.tensor(0.0, device=location_pred.device)
        avg_euclidean_dist = torch.tensor(0.0, device=location_pred.device)
        
        # Only compute loss for samples with motors
        has_motor = presence_true.squeeze() > 0.5
        if torch.sum(has_motor) > 0:
            location_pred_with_motor = location_pred[has_motor]
            location_true_with_motor = location_true[has_motor]
            
            # Calculate MSE loss
            location_loss = self.mse_loss(location_pred_with_motor, location_true_with_motor)
            
            # Calculate Euclidean distance for monitoring
            euclidean_dist = torch.sqrt(torch.sum(
                (location_pred_with_motor - location_true_with_motor) ** 2, 
                dim=1
            ))
            avg_euclidean_dist = euclidean_dist.mean()
        
        return location_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        volume = batch['volume'].to(device)
        has_motor = batch['has_motor'].to(device)
        motor_axes = batch['motor_axes'].to(device)
        
        # Forward pass
        location_pred = model(volume)
        
        # Calculate loss
        loss = criterion(
            location_pred, motor_axes, has_motor
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    
    return {
        'loss': avg_loss
    }



# Validation Function

def validate(model, dataloader, criterion, device, threshold=0.5):
    """Validate the model"""
    model.eval()
    epoch_loss = 0
    
    # Track predictions for F-beta score
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            volume = batch['volume'].to(device)
            has_motor = batch['has_motor'].to(device)
            motor_axes = batch['motor_axes'].to(device)
            original_shape = batch['original_shape'].to(device)
            voxel_spacing = batch['voxel_spacing'].to(device)
            
            # Forward pass
            location_pred = model(volume)
            
            # Calculate loss
            loss = criterion(
                location_pred, motor_axes, has_motor
            )
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate F-beta metrics
            for i in range(len(location_pred)):
                # Check if model predicts a motor
                pred_has_motor = True
                true_has_motor = True
                
                if pred_has_motor and true_has_motor:
                    # Convert normalized coordinates back to original space
                    pred_coords = location_pred[i].cpu().numpy()
                    true_coords = motor_axes[i].cpu().numpy()
                    shape = original_shape[i].cpu().numpy()
                    spacing = voxel_spacing[i].item()
                    
                    # Denormalize coordinates
                    pred_coords_orig = np.array([
                        pred_coords[0] * shape[0],
                        pred_coords[1] * shape[1],
                        pred_coords[2] * shape[2]
                    ])
                    
                    true_coords_orig = np.array([
                        true_coords[0] * shape[0],
                        true_coords[1] * shape[1],
                        true_coords[2] * shape[2]
                    ])
                    
                    # Calculate Euclidean distance in Angstroms
                    dist = np.sqrt(np.sum((pred_coords_orig - true_coords_orig) ** 2)) * spacing
                    
                    # Check if prediction is within threshold (1000 Angstroms)
                    if dist <= 1000:
                        true_positives += 1
                    else:
                        false_positives += 1
                        false_negatives += 1
                elif pred_has_motor and not true_has_motor:
                    false_positives += 1
                elif not pred_has_motor and true_has_motor:
                    false_negatives += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
            })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    
    # Calculate F-beta score (beta=2)
    beta = 2
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
    
    if precision + recall > 0:
        f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
    
    return {
        'loss': avg_loss,
        'f_beta': f_beta,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }




## Prediction function
def predict(model, dataloader, device, threshold=0.5):
    """Generate predictions for test set"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move data to device
            volume = batch['volume'].to(device)
            tomo_ids = batch['tomo_id']
            original_shape = batch['original_shape'].to(device)
            
            # Forward pass
            location_pred = model(volume)
            
            # Process predictions
            for i in range(len(location_pred)):
                tomo_id = tomo_ids[i]
                pred_has_motor = True
                
                if pred_has_motor:
                    # Convert normalized coordinates back to original space
                    pred_coords = location_pred[i].cpu().numpy()
                    shape = original_shape[i].cpu().numpy()
                    
                    # Denormalize coordinates
                    pred_coords_orig = np.array([
                        pred_coords[0] * shape[0],
                        pred_coords[1] * shape[1],
                        pred_coords[2] * shape[2]
                    ])
                    
                    predictions.append({
                        'tomo_id': tomo_id,
                        'Motor axis 0': pred_coords_orig[0],
                        'Motor axis 1': pred_coords_orig[1],
                        'Motor axis 2': pred_coords_orig[2]
                    })
                else:
                    predictions.append({
                        'tomo_id': tomo_id,
                        'Motor axis 0': -1,
                        'Motor axis 1': -1,
                        'Motor axis 2': -1
                    })
    
    return pd.DataFrame(predictions)





def run():
    """Train the model and save checkpoints"""
    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 2,
        'max_slices': 32,
        'target_size': (128, 128),
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'epochs': 5, 
        'presence_weight': 1.0,
        'location_weight': 3.0,
        'threshold': 0.5,
        'validation_split': 0.2
    }
    
    # Load and preprocess data
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Get unique tomograms
    tomo_ids = train_df['tomo_id'].unique()
    
    # Split tomograms into train and validation sets
    train_tomo_ids, val_tomo_ids = train_test_split(
        tomo_ids, 
        test_size=config['validation_split'], 
        random_state=RANDOM_SEED,
        stratify=train_df.drop_duplicates('tomo_id')['Number of motors'] > 0  # Stratify by motor presence
    )
    
    # Filter train_df to get only the relevant tomograms
    train_set_df = train_df[train_df['tomo_id'].isin(train_tomo_ids)]
    val_set_df = train_df[train_df['tomo_id'].isin(val_tomo_ids)]
    
    # Create temporary CSVs for the datasets
    train_csv = os.path.join(OUTPUT_DIR, 'train_set.csv')
    val_csv = os.path.join(OUTPUT_DIR, 'val_set.csv')
    
    train_set_df.to_csv(train_csv, index=False)
    val_set_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = TomogramDataset(
        csv_file=train_csv,
        root_dir=TRAIN_DIR,
        train=True,
        max_slices=config['max_slices'],
        target_size=config['target_size'],
        exclude_no_motor=True
    )
    
    val_dataset = TomogramDataset(
        csv_file=val_csv,
        root_dir=TRAIN_DIR,
        train=True,
        max_slices=config['max_slices'],
        target_size=config['target_size'],
        exclude_no_motor=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = ConvNextRegressor(
        max_slices=config['max_slices']
    ).to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize loss function
    criterion = FlagellarMotorRegressorLoss()
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_f_beta = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, DEVICE, threshold=config['threshold'])
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F-beta (β=2): {val_metrics['f_beta']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

        # Save best model (by loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(MODEL_DIR, 'best_model_loss.pth'))
            
            print(f"Saved best model by loss: {best_val_loss:.4f}")
        
        # Save best model (by F-beta)
        if val_metrics['f_beta'] > best_f_beta:
            best_f_beta = val_metrics['f_beta']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(MODEL_DIR, 'best_model_fbeta.pth'))
            
            print(f"Saved best model by F-beta: {best_f_beta:.4f}")
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    
    print("\nTraining completed!")
    return model



model = run()




def generate_predictions(model_path=None):
    """Generate predictions for test set"""
    # Configuration
    config = {
        'batch_size': 4,
        'num_workers': 2,
        'max_slices': 32,
        'target_size': (128, 128),
        'threshold': 0.5
    }
    
    # Use specified model path or default
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_model_fbeta.pth')
    
    # Create test dataset
    test_dataset = TomogramDataset(
        root_dir=TEST_DIR,
        max_slices=config['max_slices'],
        target_size=config['target_size'],
        train=False
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model or create a new one if not found
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Initialize model
        model = ConvNextRegressor(
            config["max_slices"]
        ).to(DEVICE)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']+1} epochs")
        print(f"Validation metrics at checkpoint: F-beta = {checkpoint['val_metrics']['f_beta']:.4f}")
    else:
        print(f"Model not found at {model_path}, creating new model")
        model = ConvNextRegressor(
            config["max_slices"]
        ).to(DEVICE)
    
    # Generate predictions
    predictions_df = predict(model, test_loader, DEVICE, threshold=config['threshold'])
    
    # Save predictions
    output_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    predictions_df.to_csv(output_file, index=False)
    
    # Print statistics
    motor_count = (predictions_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] != -1).all(axis=1).sum()
    print(f"Created submission file with {len(predictions_df)} predictions")
    print(f"Number of motors predicted: {motor_count}")
    print(f"Percentage of motors predicted: {motor_count / len(predictions_df) * 100:.2f}%")
    
    return predictions_df





def visualize_predictions(predictions_df, sample_count=3):
    """Visualize a few sample predictions"""
    # Select samples with and without motors
    motors_present = predictions_df[predictions_df['Motor axis 0'] != -1].sample(min(sample_count, len(predictions_df[predictions_df['Motor axis 0'] != -1])))
    motors_absent = predictions_df[predictions_df['Motor axis 0'] == -1].sample(min(sample_count, len(predictions_df[predictions_df['Motor axis 0'] == -1])))
    
    # Combine the samples
    samples = pd.concat([motors_present, motors_absent])
    
    # Display predictions
    print("Sample predictions:")
    for _, row in samples.iterrows():
        tomo_id = row['tomo_id']
        if row['Motor axis 0'] == -1:
            print(f"Tomogram {tomo_id}: No motor detected")
        else:
            coords = (row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'])
            print(f"Tomogram {tomo_id}: Motor detected at coordinates {coords}")

    # You could add code here to visualize specific tomogram slices with overlaid predictions
    # This would require loading the tomograms and plotting slices near the predicted motor location





