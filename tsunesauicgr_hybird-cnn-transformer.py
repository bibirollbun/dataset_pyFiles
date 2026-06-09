import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Utilities
import os
import json
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


class Config:
    DEBUG = False 
    DEBUG_SIZE = 64
    
    # Paths
    DATA_DIR = '../input/stanford-covid-vaccine/'
    BPPS_DIR = DATA_DIR + 'bpps/'
    WORKING_DIR = '/kaggle/working/'
    
    # Target columns
    TARGET_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
    ALL_TARGET_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
    ERROR_COLS = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 
                  'deg_error_pH10', 'deg_error_50C']
    
    # Sequence parameters
    MAX_SEQ_LENGTH = 130
    SEQ_LENGTH_TRAIN = 107
    SEQ_SCORED_TRAIN = 68
    
    # Model hyperparameters
    D_MODEL = 128
    NHEAD = 8
    NUM_ENCODER_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.2
    EMBED_DIM = 64  # For sequence embedding
    
    # Thêm optimizer configs
    OPTIMIZER_LR = 3e-3
    OPTIMIZER_WEIGHT_DECAY = 1e-6
    OPTIMIZER_USE_GC = False
    
    # Scheduler configs
    SCHEDULER_MILESTONES = [100, 300]
    SCHEDULER_GAMMA = 0.5
    
    # Model configs mới
    MODEL_LAYERS = 6 
    BASE_DIM = 64
    
    # Training hyperparameters
    N_FOLDS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    MAX_EPOCHS = 100
    PATIENCE = 15
    
    # Data filtering
    SIGNAL_TO_NOISE_THRESHOLD = 1.0
    USE_SN_FILTER = True

    # AE Pretraining configs
    AE_EPOCHS = 100  # Tăng lên vì có early stopping
    AE_LR = 1e-3
    AE_DROPOUT = 0.3
    AE_USE_CONTRASTIVE = True
    AE_CONTRASTIVE_WEIGHT = 0.1
    AE_NOISE_PROB = 0.3  # Noise probability for denoising
    
    # Dynamic batch scheduling
    DYNAMIC_BATCH = True
    EPOCHS_LIST = [30, 10, 6, 6, 8, 8]  # Số epochs mỗi stage
    BATCH_SIZE_LIST = [8, 16, 32, 64, 64, 64]  # Batch size tương ứng
    TOTAL_EPOCHS = sum(EPOCHS_LIST)  # 56 epochs
    
    # Override các giá trị cũ
    MAX_EPOCHS = TOTAL_EPOCHS
    BATCH_SIZE = BATCH_SIZE_LIST[0]  # Initial batch size
    PATIENCE = 20  # Tăng patience vì train lâu hơn
    
    # Random seed
    SEED = 42
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seeds
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)

print(f"\n{'='*80}")
print("CONFIGURATION")
print(f"{'='*80}")
print(f"Device: {Config.DEVICE}")
print(f"Model dimensions: d_model={Config.D_MODEL}, nhead={Config.NHEAD}")
print(f"Training: {Config.N_FOLDS}-fold CV, batch_size={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}")
print(f"{'='*80}\n")


print(f"{'='*80}")
print("LOADING DATA")
print(f"{'='*80}\n")

# Load training data
train = pd.read_json(Config.DATA_DIR + 'train.json', lines=True)
print(f"✓ Training data loaded: {train.shape}")

# Load test data
test = pd.read_json(Config.DATA_DIR + 'test.json', lines=True)
print(f"✓ Test data loaded: {test.shape}")

# Load sample submission
sample_sub = pd.read_csv(Config.DATA_DIR + 'sample_submission.csv')
print(f"✓ Sample submission loaded: {sample_sub.shape}")

if Config.DEBUG:
    print(f"\n{'!'*40}")
    print(f"WARNING: RUNNING IN DEBUG MODE (Size: {Config.DEBUG_SIZE})")
    print(f"{'!'*40}\n")
    train = train.head(Config.DEBUG_SIZE)
    test = test.head(Config.DEBUG_SIZE)

# Display basic info
print(f"\nTraining columns: {train.columns.tolist()}")
print(f"\nFirst training sample:")
print(train.iloc[0][['id', 'sequence', 'structure', 'seq_length', 'seq_scored']])


aug_df = pd.read_csv('/kaggle/input/augmented-data-for-stanford-covid-vaccine/aug_data1.csv')
display(aug_df.head())
aug_df.shape


def aug_data(df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
                         
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = pd.concat([df, new_df[df.columns]], ignore_index=True)
    return df

print("Merging data...")

if not Config.DEBUG:
    train = aug_data(train)
    # Update test (optional, if you are augmenting test data too)
    # test = aug_data(test, new_df)
    pass
else:
    print("Skipping augmentation in Debug mode.")

print(f"✓ Merge completed. New Train shape: {train.shape}")


print("\n" + "="*80)
print("SIGNAL TO NOISE FILTERING")
print("="*80 + "\n")

def calculate_signal_to_noise(row):
    """Calculate signal to noise ratio for each sample"""
    signal = []
    for col in Config.TARGET_COLS:
        values = np.array(row[col])
        signal.append(np.mean(values))
    
    noise = []
    for col in [Config.ERROR_COLS[i] for i in [0, 1, 3]]:  # reactivity, Mg_pH10, Mg_50C
        values = np.array(row[col])
        noise.append(np.mean(values))
    
    signal_mean = np.mean(signal)
    noise_mean = np.mean(noise)
    
    return signal_mean / (noise_mean + 1e-8)

train['signal_to_noise'] = train.apply(calculate_signal_to_noise, axis=1)
train['SN_filter'] = (train['signal_to_noise'] >= Config.SIGNAL_TO_NOISE_THRESHOLD).astype(int)

print(f"Signal to Noise statistics:")
print(f"  Mean: {train['signal_to_noise'].mean():.3f}")
print(f"  Median: {train['signal_to_noise'].median():.3f}")
print(f"  Min: {train['signal_to_noise'].min():.3f}")
print(f"  Max: {train['signal_to_noise'].max():.3f}")
print(f"\nSamples passing filter (SN >= {Config.SIGNAL_TO_NOISE_THRESHOLD}): {train['SN_filter'].sum()}/{len(train)}")



class Covid19Dataset(Dataset):
    def __init__(self, X, bpps, mat, seq_length, scored_length, 
                 label=None, label_error=None, signal_to_noise=None, SN_filter_mask=None):
        
        # Xử lý input: Tách Sequence và Features
        # X shape gốc: (num_samples, max_len, 4)
        self.X_seq = X[:, :, 0].astype(np.int64) 
        self.X_feat = X[:, :, 2:].astype(np.int64) # structure(idx 2) & loop(idx 3)
        
        self.bpps = bpps.astype(np.float32)
        self.mat = mat.astype(np.float32)
        
        if label is not None:
            self.label = label.astype(np.float32)
            self.label_error = label_error.astype(np.float32)
            self.signal_to_noise = signal_to_noise.astype(np.float32)
            self.SN_filter_mask = SN_filter_mask
        else:
            self.label = None
            
        # Create masks
        self.mask = np.zeros([len(X), Config.MAX_SEQ_LENGTH], dtype=bool)
        self.scored_mask = np.ones([len(X), Config.MAX_SEQ_LENGTH], dtype=bool)
        
        for i in range(len(seq_length)):
            if seq_length[i] < Config.MAX_SEQ_LENGTH:
                self.mask[i, seq_length[i]:] = True
            if scored_length[i] < Config.MAX_SEQ_LENGTH:
                self.scored_mask[i, scored_length[i]:] = False
                
        self.seq_length = seq_length
        self.scored_length = scored_length
        
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        N = self.seq_length[idx]
        
        X_seq = self.X_seq[idx, :N]
        X_feat = self.X_feat[idx, :N, :] 
        bpps = self.bpps[idx, :N, :N]
        mat = self.mat[idx, :N, :N]
        mask = self.mask[idx, :N]
        scored_mask = self.scored_mask[idx, :N]
        
        if self.label is not None:
            # --- SỬA LỖI TẠI ĐÂY: TRẢ VỀ ĐỦ 10 BIẾN ---
            return (X_seq, X_feat, bpps, mat, mask, scored_mask, 
                    self.label[idx, :N], self.label_error[idx, :N],
                    self.signal_to_noise[idx], self.SN_filter_mask[idx])
        else:
            # Test set trả về 6 biến
            return X_seq, X_feat, bpps, mat, mask, scored_mask


def collate_fn(batch):
    """
    Custom collate function xử lý:
    - Sequence (padding)
    - Features (padding)
    - BPPS/Mat (padding 2D)
    - Labels & Errors (padding)
    - SN & SNF (stacking)
    """
    # Kiểm tra xem batch là train (có label) hay test
    # Train batch từ Dataset trả về 10 phần tử (do đã tách X_seq, X_feat)
    if len(batch[0]) >= 7: 
        # Train: X_seq, X_feat, bpps, mat, mask, scored_mask, label, label_error, sn, snf
        # Lưu ý: Thứ tự này phải khớp 100% với return của Dataset.__getitem__
        is_train = True
        X_seq_list, X_feat_list, bpps_list, mat_list, mask_list, scored_mask_list, \
        label_list, label_error_list, sn_list, snf_list = zip(*batch)
    else:
        # Test: X_seq, X_feat, bpps, mat, mask, scored_mask
        is_train = False
        X_seq_list, X_feat_list, bpps_list, mat_list, mask_list, scored_mask_list = zip(*batch)
    
    batch_size = len(X_seq_list)
    max_len = max([len(x) for x in X_seq_list])
    
    # 1. Padding Inputs
    X_seq_pad = np.zeros((batch_size, max_len), dtype=np.int64)
    X_feat_pad = np.zeros((batch_size, max_len, 2), dtype=np.int64)
    bpps_pad = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    mat_pad = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    mask_pad = np.ones((batch_size, max_len), dtype=bool)
    scored_mask_pad = np.zeros((batch_size, max_len), dtype=bool)
    
    for i in range(batch_size):
        L = len(X_seq_list[i])
        X_seq_pad[i, :L] = X_seq_list[i]
        X_feat_pad[i, :L] = X_feat_list[i]
        bpps_pad[i, :L, :L] = bpps_list[i]
        mat_pad[i, :L, :L] = mat_list[i]
        mask_pad[i, :L] = mask_list[i]
        scored_mask_pad[i, :L] = scored_mask_list[i]
    
    # Convert inputs to Tensor
    X_seq_t = torch.from_numpy(X_seq_pad)
    X_feat_t = torch.from_numpy(X_feat_pad)
    bpps_t = torch.from_numpy(bpps_pad)
    mat_t = torch.from_numpy(mat_pad)
    mask_t = torch.from_numpy(mask_pad)
    scored_mask_t = torch.from_numpy(scored_mask_pad)
    
    if is_train:
        # 2. Padding Labels & Errors
        # Shape: (batch, max_len, num_targets)
        num_targets = label_list[0].shape[1]
        label_pad = np.zeros((batch_size, max_len, num_targets), dtype=np.float32)
        label_error_pad = np.zeros((batch_size, max_len, num_targets), dtype=np.float32)
        
        for i in range(batch_size):
            L = len(X_seq_list[i])
            label_pad[i, :L] = label_list[i]
            label_error_pad[i, :L] = label_error_list[i]
            
        label_t = torch.from_numpy(label_pad)
        label_error_t = torch.from_numpy(label_error_pad)
        
        # 3. Stack SN & SNF (Scalars)
        # Chuyển list thành tensor
        sn_t = torch.tensor(sn_list, dtype=torch.float32)
        snf_t = torch.tensor(snf_list, dtype=torch.bool) # hoặc float tùy nhu cầu
        
        return X_seq_t, X_feat_t, bpps_t, mat_t, mask_t, scored_mask_t, \
               label_t, label_error_t, sn_t, snf_t
    else:
        return X_seq_t, X_feat_t, bpps_t, mat_t, mask_t, scored_mask_t

def collate_fn_ae(batch):
    """
    Collate function CHUYÊN BIỆT cho AutoEncoder.
    Nó cắt mọi sample về đúng 6 phần tử input đầu tiên.
    Bất kể là Train hay Test data, ta đều coi như không có nhãn.
    """
    # Chỉ lấy 6 phần tử đầu tiên của mỗi sample:
    # 0: X_seq, 1: X_feat, 2: bpps, 3: mat, 4: mask, 5: scored_mask
    cleaned_batch = [x[:6] for x in batch]
    
    X_seq_list, X_feat_list, bpps_list, mat_list, mask_list, scored_mask_list = zip(*cleaned_batch)
    
    batch_size = len(X_seq_list)
    max_len = max([len(x) for x in X_seq_list])
    
    # Padding Inputs
    X_seq_pad = np.zeros((batch_size, max_len), dtype=np.int64)
    X_feat_pad = np.zeros((batch_size, max_len, 2), dtype=np.int64)
    bpps_pad = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    mat_pad = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    mask_pad = np.ones((batch_size, max_len), dtype=bool)
    scored_mask_pad = np.zeros((batch_size, max_len), dtype=bool)
    
    for i in range(batch_size):
        L = len(X_seq_list[i])
        X_seq_pad[i, :L] = X_seq_list[i]
        X_feat_pad[i, :L] = X_feat_list[i]
        bpps_pad[i, :L, :L] = bpps_list[i]
        mat_pad[i, :L, :L] = mat_list[i]
        mask_pad[i, :L] = mask_list[i]
        scored_mask_pad[i, :L] = scored_mask_list[i]
    
    # Convert to Tensor
    X_seq_t = torch.from_numpy(X_seq_pad)
    X_feat_t = torch.from_numpy(X_feat_pad)
    bpps_t = torch.from_numpy(bpps_pad)
    mat_t = torch.from_numpy(mat_pad)
    mask_t = torch.from_numpy(mask_pad)
    scored_mask_t = torch.from_numpy(scored_mask_pad)
    
    # LUÔN TRẢ VỀ 6 PHẦN TỬ
    return X_seq_t, X_feat_t, bpps_t, mat_t, mask_t, scored_mask_t


def encode_sequence(sequence):
    """Encode RNA sequence to integers: A=0, C=1, G=2, U=3"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    return [mapping.get(nuc, 0) for nuc in sequence]

def encode_structure(structure):
    """Encode structure to matrix: ( and ) are paired"""
    mat = np.zeros((len(structure), len(structure)), dtype=np.float32)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            mat[i, j] = 1.0
            mat[j, i] = 1.0
    return mat

def prepare_data_for_dataset(df, is_train=True):
    """
    Prepare data với encoding đầy đủ cho structure và loop type
    """
    print(f"Preparing {'training' if is_train else 'test'} data...")
    
    num_samples = len(df)
    max_len = Config.MAX_SEQ_LENGTH
    
    # Initialize arrays - Thêm chiều cho structure type và loop type
    X = np.zeros((num_samples, max_len, 4), dtype=np.int64)  # 0: nucleotide, 1-3: structure/loop info
    bpps = np.zeros((num_samples, max_len, max_len), dtype=np.float32)
    mat = np.zeros((num_samples, max_len, max_len), dtype=np.float32)
    
    seq_length = np.zeros(num_samples, dtype=int)
    scored_length = np.zeros(num_samples, dtype=int)
    
    if is_train:
        label = np.zeros((num_samples, max_len, len(Config.TARGET_COLS)), dtype=np.float32)
        label_error = np.zeros((num_samples, max_len, len(Config.TARGET_COLS)), dtype=np.float32)
        signal_to_noise = np.zeros(num_samples, dtype=np.float32)
        SN_filter_mask = np.zeros(num_samples, dtype=bool)
    
    # Mapping for structure and loop type
    structure_map = {'.': 0, '(': 1, ')': 2}
    loop_map = {'S': 0, 'M': 1, 'I': 2, 'B': 3, 'H': 4, 'E': 5, 'X': 6}
    
    # Process each sample
    for idx in range(num_samples):
        row = df.iloc[idx]
        
        # Encode sequence (nucleotide)
        seq_encoded = encode_sequence(row['sequence'])
        seq_len = len(seq_encoded)
        X[idx, :seq_len, 0] = seq_encoded
        
        # Encode structure type
        structure = row['structure']
        for i, char in enumerate(structure):
            X[idx, i, 2] = structure_map.get(char, 0)
        
        # Encode loop type (nếu có)
        if 'predicted_loop_type' in row:
            loop_type = row['predicted_loop_type']
            for i, char in enumerate(loop_type):
                X[idx, i, 3] = loop_map.get(char, 6)
        
        seq_length[idx] = seq_len
        scored_length[idx] = row['seq_scored']
        
        # Load BPPS
        bpps_data = np.load(Config.BPPS_DIR + f"{row['id']}.npy")
        bpps[idx, :seq_len, :seq_len] = bpps_data
        
        # Encode structure matrix
        mat_data = encode_structure(row['structure'])
        mat[idx, :seq_len, :seq_len] = mat_data
        
        # Training data specific
        if is_train:
            signal_to_noise[idx] = row['signal_to_noise']
            SN_filter_mask[idx] = row['SN_filter'] == 1
            
            # Targets
            for i, target_col in enumerate(Config.TARGET_COLS):
                target_vals = row[target_col]
                label[idx, :len(target_vals), i] = target_vals
                
            # Errors
            error_cols_subset = [Config.ERROR_COLS[i] for i in [0, 1, 3]]
            for i, error_col in enumerate(error_cols_subset):
                error_vals = row[error_col]
                label_error[idx, :len(error_vals), i] = error_vals
    
    print(f"  Processed {num_samples} samples")
    print(f"  X shape: {X.shape}")
    
    if is_train:
        return X, bpps, mat, seq_length, scored_length, label, label_error, signal_to_noise, SN_filter_mask
    else:
        return X, bpps, mat, seq_length, scored_length


print(f"\n{'='*80}")
print("DATA PREPARATION")
print(f"{'='*80}\n")

# Prepare training data
X_train, bpps_train, mat_train, seq_length_train, scored_length_train, \
    label_train, label_error_train, signal_to_noise_train, SN_filter_train = \
    prepare_data_for_dataset(train, is_train=True)

# Prepare test data
X_test, bpps_test, mat_test, seq_length_test, scored_length_test = \
    prepare_data_for_dataset(test, is_train=False)

print(f"\n✓ Data preparation completed!")


# Training dataset
train_dataset = Covid19Dataset(
    X_train, bpps_train, mat_train, seq_length_train, scored_length_train,
    label_train, label_error_train, signal_to_noise_train, SN_filter_train
)

test_dataset = Covid19Dataset(
    X_test, bpps_test, mat_test, seq_length_test, scored_length_test
)

print(f"\nDataset sizes:")
print(f"  Training: {len(train_dataset)}")
print(f"  Test: {len(test_dataset)}")


class DynamicBatchScheduler:
    def __init__(self, epochs_list, batch_size_list):
        # epochs_list = [30, 10, 3, 3, 5, 5]
        # batch_size_list = [8, 16, 32, 64, 128, 256]
        assert len(epochs_list) == len(batch_size_list), "Lists must have same length"
        
        self.epochs_list = epochs_list
        self.batch_size_list = batch_size_list
        
        # Tính mốc bắt đầu của từng stage: [0, 30, 40, 43, 46, 51]
        self.stage_starts = [0] + list(np.cumsum(epochs_list)[:-1])
        self.total_epochs = sum(epochs_list)

    def get_batch_size(self, epoch):
        # Tìm stage hiện tại. Duyệt ngược từ stage cuối về đầu
        for i, start in enumerate(reversed(self.stage_starts)):
            real_idx = len(self.stage_starts) - 1 - i
            if epoch >= start:
                return self.batch_size_list[real_idx]
        return self.batch_size_list[0]

    def get_stage(self, epoch):
        for i, start in enumerate(reversed(self.stage_starts)):
            real_idx = len(self.stage_starts) - 1 - i
            if epoch >= start:
                return real_idx
        return 0

    def is_stage_start(self, epoch):
        return epoch in self.stage_starts


class Mish(nn.Module):
    """Mish activation function - hiệu quả hơn ReLU"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ChannelSELayer(nn.Module):
    """
    Squeeze-and-Excitation layer cho 1D (sequence)
    Giúp model học được channel importance
    """
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        input_tensor: (batch, num_channels, W)
        """
        batch_size, num_channels, W = input_tensor.size()
        # Average along sequence dimension
        squeeze_tensor = input_tensor.mean(-1)
        
        # Channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1))
        return output_tensor


class ChannelSELayer2d(nn.Module):
    """
    Squeeze-and-Excitation layer cho 2D (BPPS matrix)
    """
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer2d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        input_tensor: (batch, num_channels, H, W)
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along spatial dimensions
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        
        # Channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1))
        return output_tensor


class PositionalEmbedding(nn.Module):
    """Positional encoding - thay thế PositionalEncoding cũ"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self):
        return self.pe


class RNAEmbedding(nn.Module):
    """
    Thay thế One-hot encoding bằng Learnable Embeddings
    """
    def __init__(self, d_model=128):
        super().__init__()
        # Seq: 4 bases (A,G,C,U) + 1 padding
        self.seq_embed = nn.Embedding(5, d_model, padding_idx=0)
        # Structure: 3 types ((, ), .) + 1 padding
        self.struct_embed = nn.Embedding(4, d_model // 2, padding_idx=0)
        # Loop: 7 types + 1 padding
        self.loop_embed = nn.Embedding(8, d_model // 2, padding_idx=0)
        
        self.proj = nn.Linear(d_model + (d_model // 2) * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq, x_feat):
        # x_seq: (batch, len)
        # x_feat: (batch, len, 2)
        s = self.seq_embed(x_seq)
        st = self.struct_embed(x_feat[:, :, 0])
        lp = self.loop_embed(x_feat[:, :, 1])
        
        x = torch.cat([s, st, lp], dim=-1)
        x = self.proj(x)
        return self.norm(x)
        
class BiasedMultiheadAttention(nn.Module):
    """
    Attention Mechanism có tích hợp BPPS Matrix làm Bias
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_bias=None, key_padding_mask=None):
        batch_size, len_q, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, len_q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(batch_size, len_q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(batch_size, len_q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add BPPS Bias
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias.unsqueeze(1)
            
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, len_q, self.d_model)
        
        return self.out_proj(output)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = BiasedMultiheadAttention(d_model, nhead, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed Forward block (có thể giữ nguyên CNN hoặc dùng Linear)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Mish(), # Dùng Mish như cũ của bạn rất tốt
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, src, bpps_bias, src_key_padding_mask=None):
        # src: (batch, seq, dim) - LƯU Ý: batch first
        src2 = self.self_attn(src, src, src, attn_bias=bpps_bias, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class ImprovedAutoEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.d_model = config.D_MODEL # Nên tăng lên 128 hoặc 256
        self.nhead = 8
        
        # 1. New Embedding Layer
        self.embedding = RNAEmbedding(self.d_model)
        self.pos_enc = PositionalEmbedding(self.d_model)
        
        # 2. BPPS Processor (Tạo bias cho Attention)
        # Biến đổi BPPS (Batch, Seq, Seq) thành (Batch, Seq, Seq) bias
        self.bpps_proj = nn.Sequential(
            nn.Linear(1, 1), # Học trọng số cho BPPS matrix
            nn.ReLU()
        )
        
        # 3. Encoder Layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                self.d_model, self.nhead, 
                dim_feedforward=self.d_model*4, 
                dropout=0.1
            ) 
            for _ in range(config.MODEL_LAYERS)
        ])
        
        # 4. Decoder Head (For Pretraining)
        self.decoder_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            Mish(),
            nn.Linear(self.d_model, 14) # Output 14 channels (4 seq + 3 struct + 7 loop)
        )
        
    def forward(self, X_seq, X_feat, bpps, mat, src_key_padding_mask=None):
        # X_seq: (Batch, Seq)
        x = self.embedding(X_seq, X_feat)
        x = x + self.pos_enc()[:, :x.size(1), :]
        
        # Xử lý BPPS làm bias
        # bpps: (Batch, Seq, Seq) -> Thêm dimension cuối -> Linear -> Bỏ dimension
        bpps_bias = self.bpps_proj(bpps.unsqueeze(-1)).squeeze(-1)
        # Nếu BPPS = 0, bias nên rất âm (để attention = 0), nếu BPPS = 1, bias = 0 hoặc dương
        # Trick: log(bpps + epsilon) cũng là một cách tốt
        
        # Encoder Flow (Batch First)
        for layer in self.layers:
            x = layer(x, bpps_bias, src_key_padding_mask)
            
        encoded = x
        
        # Decoder Flow
        reconstructed = self.decoder_head(encoded)
        
        return reconstructed, encoded


# ==================================================================================
# 4. ADVANCED MODELING ARCHITECTURE (SOTA STYLE)
# ==================================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv1dBlock(nn.Module):
    """
    Khối CNN 1D để bắt các đặc trưng cục bộ (Local motifs) của RNA
    như GC-pairs, vòng lặp nhỏ...
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding='same', dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = Swish()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Len, Dim) -> (Batch, Dim, Len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        # (Batch, Dim, Len) -> (Batch, Len, Dim)
        return x.transpose(1, 2)

class RelativePositionalBias(nn.Module):
    """
    Học mối quan hệ khoảng cách tương đối thay vì vị trí tuyệt đối.
    RNA base ở vị trí i tương tác với j phụ thuộc vào |i-j|.
    """
    def __init__(self, num_heads, max_dist=130):
        super().__init__()
        self.num_heads = num_heads
        # Tạo bảng lookup cho khoảng cách từ -max_dist đến +max_dist
        self.bias_table = nn.Embedding(2 * max_dist + 1, num_heads)
        self.max_dist = max_dist

    def forward(self, length):
        # Tạo ma trận khoảng cách (Len, Len)
        range_vec = torch.arange(length)
        # distance[i, j] = i - j
        distance_mat = range_vec[None, :] - range_vec[:, None]
        
        # Clip khoảng cách và shift để làm index dương
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_dist, self.max_dist)
        final_mat = distance_mat_clipped + self.max_dist
        
        # Lookup bias: (Len, Len, NumHeads) -> (NumHeads, Len, Len)
        bias = self.bias_table(final_mat.to(self.bias_table.weight.device)).permute(2, 0, 1)
        return bias.unsqueeze(0) # (1, NumHeads, Len, Len)

class GatedFeedForward(nn.Module):
    """
    SwiGLU FeedForward Network - Hiện đại hơn ReLU FFN truyền thống.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gated mechanism: (xW1 * Sigmoid(xW1)) * xW2 -> W3
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        return self.dropout(self.w3(hidden))

class ConformerLayer(nn.Module):
    """
    Kết hợp Convolution và Transformer (Macaron style).
    Flow: 1/2 FFN -> Attention -> Conv -> 1/2 FFN
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = GatedFeedForward(d_model, d_ff, dropout) # Half-step FFN
        
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.conv = Conv1dBlock(d_model, d_model, kernel_size=5, dropout=dropout)
        
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn2 = GatedFeedForward(d_model, d_ff, dropout) # Half-step FFN
        
        self.dropout = nn.Dropout(dropout)
        # Scale parameter cho residual connection
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, src, attn_bias=None, key_padding_mask=None):
        # 1. First Macaron FFN (Half Step)
        src = src + self.scale * self.ffn1(self.norm1(src))
        
        # 2. Attention
        # MultiheadAttention trong PyTorch: attn_mask cộng vào logits
        # attn_bias shape: (Batch*NumHeads, Len, Len) hoặc (Len, Len)
        src2 = self.norm2(src)
        src2, _ = self.attn(src2, src2, src2, attn_mask=attn_bias, key_padding_mask=key_padding_mask)
        src = src + self.dropout(src2)
        
        # 3. Convolution Block
        src = src + self.conv(self.norm3(src))
        
        # 4. Second Macaron FFN (Half Step)
        src = src + self.scale * self.ffn2(self.norm4(src))
        
        return src

class AdvancedRNAEmbedding(nn.Module):
    """
    Embedding phức hợp: Sequence + Structure + Loops + Local Conv Context
    """
    def __init__(self, d_model=128):
        super().__init__()
        # Basic Embeddings
        self.seq_embed = nn.Embedding(5, d_model, padding_idx=0)
        self.struct_embed = nn.Embedding(4, d_model // 4, padding_idx=0)
        self.loop_embed = nn.Embedding(8, d_model // 4, padding_idx=0)
        
        # Project to d_model
        input_dim = d_model + (d_model // 4) * 2
        self.proj = nn.Linear(input_dim, d_model)
        
        # Local Context Extractors (Multi-scale CNN)
        self.conv3 = Conv1dBlock(d_model, d_model, kernel_size=3)
        self.conv5 = Conv1dBlock(d_model, d_model, kernel_size=5)
        self.conv7 = Conv1dBlock(d_model, d_model, kernel_size=7)
        
        self.fusion = nn.Linear(d_model * 4, d_model) # Fusion gốc + 3 convs
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_seq, x_feat):
        # 1. Basic Embedding
        s = self.seq_embed(x_seq)
        st = self.struct_embed(x_feat[:, :, 0])
        lp = self.loop_embed(x_feat[:, :, 1])
        
        x = torch.cat([s, st, lp], dim=-1)
        x = self.proj(x) # (Batch, Len, Dim)
        
        # 2. Multi-scale Convolution
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        
        # 3. Fusion
        x_concat = torch.cat([x, c3, c5, c7], dim=-1)
        x_out = self.fusion(x_concat)
        
        return self.norm(self.dropout(x_out))

class ImprovedAutoEncoder(nn.Module):
    """
    Phiên bản nâng cấp mạnh mẽ của AE với Conformer & Relative Attention
    """
    def __init__(self, config=Config):
        super().__init__()
        self.d_model = config.D_MODEL # 128
        self.nhead = config.NHEAD     # 8
        self.n_layers = config.MODEL_LAYERS # 6
        
        # 1. Advanced Embedding
        self.embedding = AdvancedRNAEmbedding(self.d_model)
        
        # 2. Relative Position Bias
        self.rel_pos = RelativePositionalBias(self.nhead, max_dist=config.MAX_SEQ_LENGTH)
        
        # 3. BPPS Processing (CNN 2D -> Bias)
        # Thay vì Linear đơn giản, dùng 2 lớp để học phi tuyến tính từ BPPS
        self.bpps_proj = nn.Sequential(
            nn.Linear(1, 16),
            Swish(),
            nn.Linear(16, 1) # Output 1 scalar bias per pair
        )
        
        # 4. Encoder Layers (Conformer)
        self.layers = nn.ModuleList([
            ConformerLayer(
                self.d_model, self.nhead, 
                d_ff=config.DIM_FEEDFORWARD, # 512
                dropout=config.DROPOUT
            ) 
            for _ in range(self.n_layers)
        ])
        
        # 5. Decoder Heads (Reconstruction)
        self.decoder_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            Swish(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 14) # 4 seq + 3 struct + 7 loop
        )
        
    def forward(self, X_seq, X_feat, bpps, mat, src_key_padding_mask=None):
        # Input Embedding
        x = self.embedding(X_seq, X_feat) # (Batch, Len, Dim)
        
        # --- Attention Bias Setup ---
        batch_size, seq_len = x.shape[:2]
        
        # 1. Relative Position Bias
        rel_bias = self.rel_pos(seq_len) # (1, Heads, Len, Len)
        rel_bias = rel_bias.repeat(batch_size, 1, 1, 1) # (Batch, Heads, Len, Len)
        
        # 2. BPPS Bias
        # bpps: (Batch, Len, Len)
        # Project bpps to bias space
        bpps_feat = self.bpps_proj(bpps.unsqueeze(-1)).squeeze(-1) # (Batch, Len, Len)
        # Thêm chiều heads
        bpps_bias = bpps_feat.unsqueeze(1).repeat(1, self.nhead, 1, 1) # (Batch, Heads, Len, Len)
        
        # 3. Combine Biases
        # PyTorch MultiheadAttention yêu cầu attn_mask shape: (Batch*Heads, Len, Len)
        combined_bias = rel_bias + bpps_bias
        
        # Masking Padding locations in Attention Matrix
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (Batch, Len) - True là padding
            # Tạo mask (Batch, 1, 1, Len) -> Broadcast
            pad_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2) 
            combined_bias = combined_bias.masked_fill(pad_mask, float('-inf'))
            
        combined_bias = combined_bias.view(batch_size * self.nhead, seq_len, seq_len)
        
        # --- Encoder Loop ---
        for layer in self.layers:
            # Lưu ý: ConformerLayer tự xử lý attn_bias trong MultiheadAttention
            x = layer(x, attn_bias=combined_bias)
            
        encoded = x
        
        # Reconstruction
        reconstructed = self.decoder_head(encoded)
        
        return reconstructed, encoded

class HybridModelFromImprovedAE(nn.Module):
    """
    Model chính cho Regression, sử dụng lại Encoder xịn sò ở trên.
    """
    def __init__(self, ae_model=None, config=Config):
        super().__init__()
        
        self.d_model = config.D_MODEL if ae_model is None else ae_model.d_model
        
        # 1. Load Shared Components
        if ae_model is not None:
            self.embedding = ae_model.embedding
            self.rel_pos = ae_model.rel_pos
            self.bpps_proj = ae_model.bpps_proj
            self.layers = ae_model.layers
            self.nhead = ae_model.nhead
        else:
            # Init fresh if no AE provided (Fallback)
            self.embedding = AdvancedRNAEmbedding(self.d_model)
            self.rel_pos = RelativePositionalBias(8)
            self.bpps_proj = nn.Sequential(nn.Linear(1, 16), Swish(), nn.Linear(16, 1))
            self.layers = nn.ModuleList([ConformerLayer(self.d_model, 8, 512) for _ in range(6)])
            self.nhead = 8
            
        # 2. Advanced Regression Head
        # Thay vì LSTM, dùng GRU nhẹ hơn + Attention Pooling hoặc Sequence Pooling
        self.head_rnn = nn.GRU(
            self.d_model, 
            self.d_model // 2, 
            bidirectional=True, 
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        
        self.head_norm = nn.LayerNorm(self.d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(self.d_model, self.d_model // 2),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, len(config.TARGET_COLS))
        )
        
        # Learnable scaling
        self.label_mean = nn.Parameter(torch.zeros(len(config.TARGET_COLS)), requires_grad=False)
        self.label_std = nn.Parameter(torch.ones(len(config.TARGET_COLS)), requires_grad=False)

    def forward(self, X_seq, X_feat, bpps, mat, src_key_padding_mask=None):
        # --- Encoder Pass (Giống hệt AE) ---
        x = self.embedding(X_seq, X_feat)
        batch_size, seq_len = x.shape[:2]
        
        rel_bias = self.rel_pos(seq_len).repeat(batch_size, 1, 1, 1)
        bpps_bias = self.bpps_proj(bpps.unsqueeze(-1)).squeeze(-1).unsqueeze(1).repeat(1, self.nhead, 1, 1)
        combined_bias = rel_bias + bpps_bias
        
        if src_key_padding_mask is not None:
            pad_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            combined_bias = combined_bias.masked_fill(pad_mask, float('-inf'))
            
        combined_bias = combined_bias.view(batch_size * self.nhead, seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, attn_bias=combined_bias)
            
        # --- Regression Head Pass ---
        # x: (Batch, Len, Dim)
        # GRU giúp làm mượt thông tin chuỗi cuối cùng
        self.head_rnn.flatten_parameters()
        x, _ = self.head_rnn(x) 
        
        x = self.head_norm(x)
        y = self.mlp(x) # (Batch, Len, Targets)
        
        # De-normalize output
        y = y * self.label_std + self.label_mean
        return y

    def set_label_stats(self, mean_vals, std_vals):
        self.label_mean.data = torch.tensor(mean_vals, dtype=torch.float32).to(self.label_mean.device)
        self.label_std.data = torch.tensor(std_vals, dtype=torch.float32).to(self.label_std.device)

print("✓ SOTA Architecture (Conformer + RelativeBias + SwiGLU) loaded.")


class Ranger(optim.Optimizer):
    """
    Ranger optimizer = RAdam + Lookahead + Gradient Centralization
    Hiệu quả hơn Adam thông thường
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5,
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,
                 use_gc=True, gc_conv_only=False):
        
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')
        
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                       N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.use_gc = use_gc
        self.gc_gradient_threshold = 3 if gc_conv_only else 1
        
        print(f"Ranger optimizer loaded. Gradient Centralization usage = {self.use_gc}")
        if self.use_gc and self.gc_gradient_threshold == 1:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_gradient_threshold == 3:
            print(f"GC applied to conv layers only")
    
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')
                
                p_data_fp32 = p.data.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Gradient Centralization
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                
                state['step'] += 1
                
                # Compute variance and mean moving averages
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                buffered = self.radam_buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply learning rate
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                
                p.data.copy_(p_data_fp32)
                
                # Lookahead
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
        
        return loss


def mcrmse_metric(y_true, y_pred):
    """Compute MCRMSE (Mean Columnwise Root Mean Squared Error)"""
    y_true_scored = y_true[:, :68]
    y_pred_scored = y_pred[:, :68]
    colwise_rmse = np.sqrt(np.mean((y_true_scored - y_pred_scored)**2, axis=0))
    return np.mean(colwise_rmse)

def calculate_metrics(y_true, y_pred, target_names):
    """
    Compute RMSE, MAE, R², NSE per target and overall.
    Input shape: (num_samples * seq_len, num_targets) -> Flattened 2D
    """
    metrics = {}
    
    for i, target in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Basic metrics
        metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_t, y_p))
        metrics[f'{target}_mae'] = mean_absolute_error(y_t, y_p)
        metrics[f'{target}_r2'] = r2_score(y_t, y_p)
        
        # NSE metric
        numerator = np.sum((y_t - y_p) ** 2)
        denominator = np.sum((y_t - np.mean(y_t)) ** 2)
        if denominator == 0:
            metrics[f'{target}_nse'] = 0.0
        else:
            metrics[f'{target}_nse'] = 1 - (numerator / denominator)
    
    # Aggregate metrics
    metrics['overall_rmse'] = np.mean([metrics[f'{t}_rmse'] for t in target_names])
    metrics['overall_mae'] = np.mean([metrics[f'{t}_mae'] for t in target_names])
    metrics['overall_r2'] = np.mean([metrics[f'{t}_r2'] for t in target_names])
    metrics['overall_nse'] = np.mean([metrics[f'{t}_nse'] for t in target_names])
    
    return metrics

class MCRMSELoss(nn.Module):
    def __init__(self, seq_len_target=68):
        super(MCRMSELoss, self).__init__()
        self.seq_len_target = seq_len_target

    def forward(self, y_pred, y_true, scored_mask=None):
        # y_pred, y_true shape: (batch, seq_len, 3)
        y_true_scored = y_true[:, :self.seq_len_target, :]
        y_pred_scored = y_pred[:, :self.seq_len_target, :]
        
        # Calculate MSE
        mse = (y_true_scored - y_pred_scored) ** 2
        
        # Mean over batch and sequence length, preserve targets
        rmse_per_target = torch.sqrt(torch.mean(mse, dim=(0, 1)))
        
        # Mean over targets
        loss = torch.mean(rmse_per_target)
        return loss


class AEReconstructionLoss(nn.Module):
    """
    Custom loss cho Denoising AutoEncoder
    Kết hợp:
    - BCE loss cho categorical features (sequence, structure, loop)
    - MSE loss cho continuous features (nếu có)
    """
    def __init__(self):
        super(AEReconstructionLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, reconstructed, target, mask):
        """
        Args:
            reconstructed: (batch, seq, 14) - output của decoder
            target: (batch, seq, 14) - target one-hot features
            mask: (batch, seq) - True for padding positions
        
        Returns:
            loss: scalar loss value
        """
        # Calculate BCE loss
        loss = self.bce(reconstructed, target)
        
        # Mask out padding positions
        # mask shape: (batch, seq) -> expand to (batch, seq, 14)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        
        # Zero out loss at padded positions
        loss = loss * (~mask_expanded).float()
        
        # Average over non-padded positions
        num_valid = (~mask_expanded).float().sum()
        loss = loss.sum() / (num_valid + 1e-8)
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss cho encoded representations
    Giúp model học better representations
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, encoded, mask):
        """
        Args:
            encoded: (batch, seq, dim) - encoded representations
            mask: (batch, seq) - padding mask
        
        Returns:
            loss: contrastive loss value
        """
        batch_size, seq_len, dim = encoded.shape
        
        # Get mean representation per sequence (excluding padding)
        mask_expanded = (~mask).float().unsqueeze(-1)  # (batch, seq, 1)
        seq_repr = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        # seq_repr: (batch, dim)
        
        # Compute pairwise similarity
        similarity_matrix = torch.matmul(seq_repr, seq_repr.T) / self.temperature
        
        # Remove self-similarity (diagonal)
        mask_diag = torch.eye(batch_size, device=encoded.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask_diag, -1e9)
        
        # Contrastive loss: maximize similarity with augmented versions, minimize with others
        # For simplicity, treat all pairs as negatives (can be improved with positive pairs)
        loss = -torch.log_softmax(similarity_matrix, dim=1).diag().mean()
        
        return loss


class CombinedAELoss(nn.Module):
    def __init__(self, use_contrastive=False, contrastive_weight=0.1):
        super(CombinedAELoss, self).__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, reconstructed, encoded, target_full, mask):
        """
        reconstructed: (batch, seq, 14) - Raw logits
        target_full: (batch, seq, 14) - One-hot targets
        mask: (batch, seq) - Padding mask (True where padding)
        """
        # Tách logits và targets cho từng phần
        # 0-4: Sequence, 4-7: Structure, 7-14: Loop
        pred_seq = reconstructed[:, :, :4]
        pred_struct = reconstructed[:, :, 4:7]
        pred_loop = reconstructed[:, :, 7:]
        
        target_seq = target_full[:, :, :4]
        target_struct = target_full[:, :, 4:7]
        target_loop = target_full[:, :, 7:]
        
        # Convert targets from one-hot back to indices for CrossEntropy
        idx_seq = torch.argmax(target_seq, dim=-1)
        idx_struct = torch.argmax(target_struct, dim=-1)
        idx_loop = torch.argmax(target_loop, dim=-1)
        
        # Calculate loss (chỉ tính ở những vị trí không padding)
        # mask is True for padding -> use ~mask for valid positions
        active_mask = ~mask
        
        loss_seq = self.ce_loss(pred_seq.permute(0, 2, 1), idx_seq)
        loss_struct = self.ce_loss(pred_struct.permute(0, 2, 1), idx_struct)
        loss_loop = self.ce_loss(pred_loop.permute(0, 2, 1), idx_loop)
        
        # Masking padding
        loss_seq = (loss_seq * active_mask).sum() / active_mask.sum()
        loss_struct = (loss_struct * active_mask).sum() / active_mask.sum()
        loss_loop = (loss_loop * active_mask).sum() / active_mask.sum()
        
        recon_loss = loss_seq + loss_struct + loss_loop
        
        # Contrastive Loss (Optional - placeholder logic simple MSE reg)
        contrast_loss = torch.tensor(0.0, device=reconstructed.device)
        if self.use_contrastive:
             # Đơn giản hóa: L2 regularization trên encoded features để tránh overfitting
             contrast_loss = torch.mean(encoded ** 2)
        
        total_loss = recon_loss + self.contrastive_weight * contrast_loss
        
        return total_loss, recon_loss, contrast_loss


def add_noise_to_features(X_seq, X_feat, noise_prob=0.3):
    """
    Thêm noise vào cả sequence và features (structure/loop types).
    Thay thế ngẫu nhiên bằng các giá trị hợp lệ khác.
    """
    # 1. Noise cho Sequence (0-3)
    X_seq_noisy = X_seq.clone()
    mask_seq = torch.rand_like(X_seq.float()) < noise_prob
    rand_seq = torch.randint_like(X_seq, 0, 4)
    X_seq_noisy[mask_seq] = rand_seq[mask_seq]
    
    # 2. Noise cho Features
    X_feat_noisy = X_feat.clone()
    
    # Structure type (index 0, range 0-2)
    mask_struct = torch.rand_like(X_feat[:,:,0].float()) < noise_prob
    rand_struct = torch.randint_like(X_feat[:,:,0], 0, 3)
    X_feat_noisy[:,:,0][mask_struct] = rand_struct[mask_struct]
    
    # Loop type (index 1, range 0-6)
    mask_loop = torch.rand_like(X_feat[:,:,1].float()) < noise_prob
    rand_loop = torch.randint_like(X_feat[:,:,1], 0, 7)
    X_feat_noisy[:,:,1][mask_loop] = rand_loop[mask_loop]
    
    return X_seq_noisy, X_feat_noisy

def train_autoencoder_epoch(ae_model, data_loader, optimizer, device, criterion):
    ae_model.train()
    total_loss_sum = 0
    recon_loss_sum = 0
    num_batches = 0
    
    for batch_data in data_loader:
        # AE chỉ nhận 6 biến input, không quan tâm label
        X_seq, X_feat, bpps, mat, mask, scored_mask = batch_data
        
        X_seq = X_seq.to(device)
        X_feat = X_feat.to(device)
        bpps = bpps.to(device)
        mat = mat.to(device)
        mask = mask.to(device)
        
        # Add noise & Create Targets (Code cũ)
        X_seq_noisy, X_feat_noisy = add_noise_to_features(X_seq, X_feat, noise_prob=0.3)
        
        t_seq = F.one_hot(X_seq, 4).float()
        t_struct = F.one_hot(X_feat[:,:,0], 3).float()
        t_loop = F.one_hot(X_feat[:,:,1], 7).float()
        target_full = torch.cat([t_seq, t_struct, t_loop], dim=-1).to(device)
        
        optimizer.zero_grad()
        reconstructed, encoded = ae_model(
            X_seq_noisy, X_feat_noisy, bpps, mat, src_key_padding_mask=mask
        )
        
        total_loss, recon_loss, contrast_loss = criterion(
            reconstructed, encoded, target_full, mask
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        num_batches += 1
    
    return total_loss_sum / num_batches, recon_loss_sum / num_batches, 0.0

def validate_autoencoder(ae_model, data_loader, device, criterion):
    ae_model.eval()
    total_loss_sum = 0
    recon_loss_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Chỉ nhận 6 biến
            X_seq, X_feat, bpps, mat, mask, scored_mask = batch_data
            
            X_seq = X_seq.to(device)
            X_feat = X_feat.to(device)
            bpps = bpps.to(device)
            mat = mat.to(device)
            mask = mask.to(device)
            
            t_seq = F.one_hot(X_seq, 4).float()
            t_struct = F.one_hot(X_feat[:,:,0], 3).float()
            t_loop = F.one_hot(X_feat[:,:,1], 7).float()
            target_full = torch.cat([t_seq, t_struct, t_loop], dim=-1).to(device)
            
            reconstructed, encoded = ae_model(
                X_seq, X_feat, bpps, mat, src_key_padding_mask=mask
            )
            
            total_loss, recon_loss, contrast_loss = criterion(
                reconstructed, encoded, target_full, mask
            )
            
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            num_batches += 1
            
    return total_loss_sum / num_batches, recon_loss_sum / num_batches, 0.0


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    for batch_data in train_loader:
        # Unpack ĐẦY ĐỦ 10 biến (theo thứ tự trong collate_fn)
        X_seq, X_feat, bpps, mat, mask, scored_mask, label, label_error, sn, snf = batch_data 
        
        # Move inputs & labels to device
        X_seq = X_seq.to(device)
        X_feat = X_feat.to(device)
        bpps = bpps.to(device)
        mat = mat.to(device)
        mask = mask.to(device)
        scored_mask = scored_mask.to(device)
        label = label.to(device)
        
        # (Optional) Move metadata to device nếu cần dùng trong loss tùy chỉnh
        # label_error = label_error.to(device)
        # sn = sn.to(device)
        
        optimizer.zero_grad()
        
        # Model forward
        outputs = model(X_seq, X_feat, bpps, mat, src_key_padding_mask=mask)
        
        # 1. Tính Loss (MCRMSE)
        loss = criterion(outputs, label, scored_mask)
        
        # 2. Tính MAE thực tế (Chỉ tính trên 68 vị trí đầu - Seq Scored)
        with torch.no_grad():
            # Lấy phần được chấm điểm
            out_scored = outputs[:, :Config.SEQ_SCORED_TRAIN, :]
            lbl_scored = label[:, :Config.SEQ_SCORED_TRAIN, :]
            # Tính Mean Absolute Error
            mae_batch = torch.mean(torch.abs(out_scored - lbl_scored))
            total_mae += mae_batch.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    return total_loss / num_batches, total_mae / num_batches

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0 # <--- Biến cộng dồn MAE
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            X_seq, X_feat, bpps, mat, mask, scored_mask, label, label_error, sn, snf = batch_data
            
            X_seq = X_seq.to(device)
            X_feat = X_feat.to(device)
            bpps = bpps.to(device)
            mat = mat.to(device)
            mask = mask.to(device)
            label = label.to(device)
            
            outputs = model(X_seq, X_feat, bpps, mat, src_key_padding_mask=mask)
            
            # 1. Tính Loss
            loss = criterion(outputs, label, scored_mask)
            
            # 2. Tính MAE
            out_scored = outputs[:, :Config.SEQ_SCORED_TRAIN, :]
            lbl_scored = label[:, :Config.SEQ_SCORED_TRAIN, :]
            mae_batch = torch.mean(torch.abs(out_scored - lbl_scored))
            total_mae += mae_batch.item()
            
            total_loss += loss.item()
            num_batches += 1
    
    # Trả về cả Loss và MAE
    return total_loss / num_batches, total_mae / num_batches

def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Xử lý linh hoạt cho cả Train set (để tính OOF) và Test set
            if len(batch_data) == 10:  # Train/Val data (có label, err, sn...)
                X_seq, X_feat, bpps, mat, mask, scored_mask, _, _, _, _ = batch_data
            else:  # Test data (chỉ có input)
                X_seq, X_feat, bpps, mat, mask, scored_mask = batch_data
            
            X_seq = X_seq.to(device)
            X_feat = X_feat.to(device)
            bpps = bpps.to(device)
            mat = mat.to(device)
            mask = mask.to(device)
            
            outputs = model(X_seq, X_feat, bpps, mat, src_key_padding_mask=mask)
            all_predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)

print("✓ Training functions updated")


# Helper functions for freezing/unfreezing
def freeze_encoder(model):
    print("Locked Encoder weights.")
    for param in model.embedding.parameters(): param.requires_grad = False
    for param in model.rel_pos.parameters(): param.requires_grad = False
    for param in model.bpps_proj.parameters(): param.requires_grad = False
    for param in model.layers.parameters(): param.requires_grad = False

def unfreeze_encoder(model):
    print("Unlocked Encoder weights.")
    for param in model.parameters(): param.requires_grad = True


# ==================================================================================
# TRAINING PIPELINE: STEP 1 (AE) & STEP 2 (FINETUNE)
# ==================================================================================

# 1. Pretrain AutoEncoder (Sử dụng class ImprovedAutoEncoder)
print(f"\n{'='*80}\nSTEP 1: AUTOENCODER PRETRAINING\n{'='*80}")

# Re-initialize dataloaders for AE
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
train_size = int(0.9 * len(combined_dataset))
ae_train_ds, ae_val_ds = torch.utils.data.random_split(
    combined_dataset, [train_size, len(combined_dataset) - train_size],
    generator=torch.Generator().manual_seed(Config.SEED)
)
ae_train_loader = DataLoader(ae_train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_ae, num_workers=2)
ae_val_loader = DataLoader(ae_val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_ae, num_workers=2)

# Init & Train AE
ae_model = ImprovedAutoEncoder(Config).to(Config.DEVICE)
criterion_ae = CombinedAELoss(use_contrastive=True, contrastive_weight=0.1)
optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=Config.AE_LR)

# AE Training Loop
for epoch in range(Config.AE_EPOCHS):
    train_loss, _, _ = train_autoencoder_epoch(ae_model, ae_train_loader, optimizer_ae, Config.DEVICE, criterion_ae)
    val_loss, _, _ = validate_autoencoder(ae_model, ae_val_loader, Config.DEVICE, criterion_ae)
    if (epoch+1) % 10 == 0:
        print(f"AE Epoch {epoch+1}: Loss={train_loss:.4f}/{val_loss:.4f}")

# Save AE Model
torch.save(ae_model.state_dict(), 'ae_pretrained.pth') 
print("✓ AE Pretraining Done & Saved to 'ae_pretrained.pth'!") 

# 2. Finetune Main Model
print(f"\n{'='*80}\nSTEP 2: SUPERVISED FINETUNING (FREEZE -> UNFREEZE)\n{'='*80}")

# --- KHỞI TẠO BIẾN LỊCH SỬ (SỬA LỖI NAME ERROR) ---
history_dict = {
    'fold': [], 'epoch': [], 'stage': [], 'batch_size': [],
    'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []
}

# Initialize scheduler với Config của bạn
batch_scheduler = DynamicBatchScheduler(
    epochs_list=Config.EPOCHS_LIST,
    batch_size_list=Config.BATCH_SIZE_LIST
)

# Calculate Stats for Scaling (SỬA LỖI TYPE ERROR)
print("Calculating target statistics...")
means = []
stds = []

for col in Config.TARGET_COLS:
    values = np.concatenate(train[col].values) # Phẳng hóa list thành array
    values = values[~np.isnan(values)] # Bỏ nan
    means.append(np.mean(values))
    stds.append(np.std(values))

target_mean = np.array(means)
target_std = np.array(stds)

print(f"Target Mean: {target_mean}")
print(f"Target Std: {target_std}")

# Init Metrics containers
oof_predictions = np.zeros((len(train_dataset), Config.MAX_SEQ_LENGTH, len(Config.TARGET_COLS)))
test_predictions_list = []
fold_metrics = []

kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
    print(f"\n{'='*40} FOLD {fold + 1}/{Config.N_FOLDS} {'='*40}")
    
    # Data Setup
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    # Init Model with AE weights
    model = HybridModelFromImprovedAE(ae_model=ae_model, config=Config).to(Config.DEVICE)
    model.set_label_stats(target_mean, target_std) # Set correct scaling
    
    # Metrics
    criterion = MCRMSELoss(seq_len_target=Config.SEQ_SCORED_TRAIN)
    
    # --- PHASE 1: WARM UP (FROZEN ENCODER) ---
    print(f">>> Phase 1: Training Head only (Encoder Frozen)")
    freeze_encoder(model)
    
    # Optimizer cho Phase 1
    head_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Ranger(head_params, lr=Config.OPTIMIZER_LR)
    
    # Init Loaders
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    WARMUP_EPOCHS = 5
    for epoch in range(WARMUP_EPOCHS):
        t_loss, t_mae = train_one_epoch(model, train_loader, criterion, optimizer, None, Config.DEVICE)
        v_loss, v_mae = validate(model, val_loader, criterion, Config.DEVICE)
        print(f"   [Warmup {epoch+1}] Loss: {t_loss:.4f}/{v_loss:.4f}")
        
        # LOG HISTORY (Warmup)
        history_dict['fold'].append(fold + 1)
        history_dict['epoch'].append(epoch + 1)
        history_dict['stage'].append(0) # Stage 0 for warmup
        history_dict['batch_size'].append(Config.BATCH_SIZE)
        history_dict['train_loss'].append(t_loss)
        history_dict['val_loss'].append(v_loss)
        history_dict['train_mae'].append(t_mae)
        history_dict['val_mae'].append(v_mae)
        
    # --- PHASE 2: FULL FINETUNING (UNFROZEN) ---
    print(f">>> Phase 2: Full Finetuning")
    unfreeze_encoder(model)
    
    # Re-init optimizer cho toàn bộ model
    optimizer = Ranger(model.parameters(), lr=Config.OPTIMIZER_LR / 2.0, weight_decay=1e-5)
    
    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.OPTIMIZER_LR, 
        epochs=Config.TOTAL_EPOCHS - WARMUP_EPOCHS, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    
    # Main Loop
    # Lưu ý: Epoch chạy tiếp từ WARMUP_EPOCHS
    for epoch in range(WARMUP_EPOCHS, Config.TOTAL_EPOCHS):
        # Dynamic Batch check
        new_batch_size = batch_scheduler.get_batch_size(epoch)
        current_stage = batch_scheduler.get_stage(epoch)
        
        if train_loader.batch_size != new_batch_size:
            print(f"   [Batch Switch] -> {new_batch_size}")
            train_loader = DataLoader(train_subset, batch_size=new_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
            # Cập nhật steps cho scheduler nếu cần (OneCycleLR khó cập nhật động, ta chấp nhận sai số nhỏ hoặc reset scheduler)
        
        # Train & Val
        t_loss, t_mae = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, Config.DEVICE)
        v_loss, v_mae = validate(model, val_loader, criterion, Config.DEVICE)
        
        # LOG HISTORY (Main Loop)
        history_dict['fold'].append(fold + 1)
        history_dict['epoch'].append(epoch + 1)
        history_dict['stage'].append(current_stage + 1)
        history_dict['batch_size'].append(new_batch_size)
        history_dict['train_loss'].append(t_loss)
        history_dict['val_loss'].append(v_loss)
        history_dict['train_mae'].append(t_mae)
        history_dict['val_mae'].append(v_mae)
        
        # Save best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            print(f"   Epoch {epoch+1}: *Best* Loss={v_loss:.4f} | MAE={v_mae:.4f}")
        else:
            print(f"   Epoch {epoch+1}: Loss={v_loss:.4f}")
            
    # Load Best & Predict OOF
    model.load_state_dict(torch.load(f'best_model_fold_{fold+1}.pth'))
    val_preds = predict(model, val_loader, Config.DEVICE)
    oof_predictions[val_idx, :val_preds.shape[1], :] = val_preds
    
    # Calc Fold Metrics
    y_true = label_train[val_idx, :Config.SEQ_SCORED_TRAIN, :]
    y_pred = val_preds[:, :Config.SEQ_SCORED_TRAIN, :]
    fold_metrics.append(calculate_metrics(y_true.reshape(-1, 3), y_pred.reshape(-1, 3), Config.TARGET_COLS))
    
    # Predict Test
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_preds = predict(model, test_loader, Config.DEVICE)
    test_predictions_list.append(test_preds)
    
    # Clear memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

print("\nTRAINING COMPLETED!")


# Average test predictions
test_predictions = np.mean(test_predictions_list, axis=0)

# Trích xuất danh sách các giá trị RMSE từ list các dictionary
rmse_values = [m['overall_rmse'] for m in fold_metrics]

# Tính trung bình và độ lệch chuẩn
overall_rmse = np.mean(rmse_values)
overall_std = np.std(rmse_values)

print(f"\n{'='*80}")
print(f"Overall CV RMSE: {overall_rmse:.5f} ± {overall_std:.5f}")
print(f"{'='*80}\n")

print("Calculating global metrics...")

# 1. Lấy dữ liệu thực tế và dự đoán (chỉ lấy phần được chấm điểm - 68 vị trí đầu)
y_train_scored = label_train[:, :Config.SEQ_SCORED_TRAIN, :]
oof_scored = oof_predictions[:, :Config.SEQ_SCORED_TRAIN, :]

# 2. Làm phẳng (Flatten) dữ liệu để tính toán
y_train_flat = y_train_scored.reshape(-1, 3)
oof_flat = oof_scored.reshape(-1, 3)

# 3. Tính toán metrics (Tạo biến cv_metrics tại đây)
cv_metrics = calculate_metrics(y_train_flat, oof_flat, Config.TARGET_COLS)

print(f"Metrics calculated. Overall RMSE: {cv_metrics['overall_rmse']:.5f}")


print(f"\nGenerating submission...")

# Ensemble
test_predictions = np.mean(test_predictions_list, axis=0)

submission_data = []
for i in range(len(test)):
    mol_id = test.iloc[i]['id']
    seq_scored = test.iloc[i]['seq_scored']
    
    for pos in range(seq_scored):
        submission_data.append({
            'id_seqpos': f"{mol_id}_{pos}",
            'reactivity': test_predictions[i, pos, 0],
            'deg_Mg_pH10': test_predictions[i, pos, 1],
            'deg_Mg_50C': test_predictions[i, pos, 2],
            'deg_pH10': 0.0,
            'deg_50C': 0.0
        })

submission_df = pd.DataFrame(submission_data)
submission_final = sample_sub[['id_seqpos']].merge(submission_df, on='id_seqpos', how='left').fillna(0)

# Use overall_rmse from the calculated cv_metrics
sub_filename = f'submission_hybrid_cv{cv_metrics["overall_rmse"]:.5f}.csv'
submission_final.to_csv(sub_filename, index=False)

print(f"✓ Submission saved to: {sub_filename}")
print("DONE!")


print(f"\n{'='*80}")
print("EVALUATION REPORT")
print(f"{'='*80}\n")

# 1. Calculate Global CV Metrics (Fixes NameError)
# Get scored positions for all training data
y_train_scored = label_train[:, :Config.SEQ_SCORED_TRAIN, :]
oof_scored = oof_predictions[:, :Config.SEQ_SCORED_TRAIN, :]

# Flatten
y_train_flat = y_train_scored.reshape(-1, 3)
oof_flat = oof_scored.reshape(-1, 3)

# Calculate metrics
cv_metrics = calculate_metrics(y_train_flat, oof_flat, Config.TARGET_COLS)
print(f"Overall CV RMSE: {cv_metrics['overall_rmse']:.5f}")

# 2. Create Summary Table
summary_data = []
for target in Config.TARGET_COLS:
    row = {
        'Target': target,
        'RMSE': f"{cv_metrics[f'{target}_rmse']:.5f} ± {np.std([m[f'{target}_rmse'] for m in fold_metrics]):.5f}",
        'MAE': f"{cv_metrics[f'{target}_mae']:.5f} ± {np.std([m[f'{target}_mae'] for m in fold_metrics]):.5f}",
        'R²': f"{cv_metrics[f'{target}_r2']:.5f} ± {np.std([m[f'{target}_r2'] for m in fold_metrics]):.5f}",
        'NSE': f"{cv_metrics[f'{target}_nse']:.5f} ± {np.std([m[f'{target}_nse'] for m in fold_metrics]):.5f}"
    }
    summary_data.append(row)

# Add overall row
summary_data.append({
    'Target': 'Overall',
    'RMSE': f"{cv_metrics['overall_rmse']:.5f} ± {np.std([m['overall_rmse'] for m in fold_metrics]):.5f}",
    'MAE': f"{cv_metrics['overall_mae']:.5f} ± {np.std([m['overall_mae'] for m in fold_metrics]):.5f}",
    'R²': f"{cv_metrics['overall_r2']:.5f} ± {np.std([m['overall_r2'] for m in fold_metrics]):.5f}",
    'NSE': f"{cv_metrics['overall_nse']:.5f} ± {np.std([m['overall_nse'] for m in fold_metrics]):.5f}"
})

summary_df = pd.DataFrame(summary_data)
print("\nMETRICS SUMMARY:")
print(summary_df.to_string(index=False))
summary_df.to_csv('cv_metrics_summary.csv', index=False)

# 3. Visualizations
# 3a. Training History
history_df = pd.DataFrame(history_dict)
if not history_df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Loss
    for fold in range(1, Config.N_FOLDS + 1):
        fold_data = history_df[history_df['fold'] == fold]
        axes[0, 0].plot(fold_data['epoch'], fold_data['train_loss'], alpha=0.5, label=f'Fold {fold}')
        axes[0, 1].plot(fold_data['epoch'], fold_data['val_loss'], alpha=0.5, label=f'Fold {fold}')
    
    axes[0, 0].set_title('Training Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].set_title('Validation Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)
    
    # Plot MAE
    for fold in range(1, Config.N_FOLDS + 1):
        fold_data = history_df[history_df['fold'] == fold]
        axes[1, 0].plot(fold_data['epoch'], fold_data['train_mae'], alpha=0.5, label=f'Fold {fold}')
        axes[1, 1].plot(fold_data['epoch'], fold_data['val_mae'], alpha=0.5, label=f'Fold {fold}')

    axes[1, 0].set_title('Training MAE'); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].set_title('Validation MAE'); axes[1, 1].legend(); axes[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

# 3b. Metrics per Fold
metrics_comparison = pd.DataFrame(fold_metrics)
metrics_comparison['fold'] = range(1, Config.N_FOLDS + 1)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics_to_plot = ['overall_rmse', 'overall_mae', 'overall_r2', 'overall_nse']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    ax.bar(metrics_comparison['fold'], metrics_comparison[metric], color='steelblue', alpha=0.7)
    ax.axhline(metrics_comparison[metric].mean(), color='red', linestyle='--')
    ax.set_title(metric.upper())
    ax.set_xticks(range(1, Config.N_FOLDS + 1))
plt.tight_layout()
plt.show()

# 3c. Scatter Plots (Pred vs True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, target in enumerate(Config.TARGET_COLS):
    ax = axes[idx]
    # Sample 5000 points to avoid slow plotting
    indices = np.random.choice(len(y_train_flat), size=min(5000, len(y_train_flat)), replace=False)
    y_t = y_train_flat[indices, idx]
    y_p = oof_flat[indices, idx]
    
    ax.scatter(y_t, y_p, alpha=0.3, s=10)
    min_val, max_val = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_title(f'{target} (R2={cv_metrics[f"{target}_r2"]:.3f})')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
plt.tight_layout()
plt.show()


# 3d. Batch Size Schedule Visualization
print("\nGenerating batch size schedule visualization...")

history_df = pd.DataFrame(history_dict)

if not history_df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Loss với batch size color coding
    for fold in range(1, Config.N_FOLDS + 1):
        fold_data = history_df[history_df['fold'] == fold]
        scatter = axes[0, 0].scatter(
            fold_data['epoch'], 
            fold_data['train_loss'],
            c=fold_data['batch_size'],
            cmap='viridis',
            alpha=0.6,
            s=20,
            label=f'Fold {fold}' if fold == 1 else ""
        )
    
    axes[0, 0].set_title('Training Loss vs Epoch (colored by batch size)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[0, 0])
    cbar.set_label('Batch Size')
    
    # Plot 2: Validation Loss
    for fold in range(1, Config.N_FOLDS + 1):
        fold_data = history_df[history_df['fold'] == fold]
        axes[0, 1].plot(fold_data['epoch'], fold_data['val_loss'], alpha=0.6, label=f'Fold {fold}')
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Batch Size Schedule
    fold_1_data = history_df[history_df['fold'] == 1]
    axes[1, 0].step(fold_1_data['epoch'], fold_1_data['batch_size'], where='post', linewidth=2)
    axes[1, 0].set_title('Batch Size Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Batch Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add stage boundaries
    cumsum_epochs = np.cumsum([0] + Config.EPOCHS_LIST)
    for i, boundary in enumerate(cumsum_epochs[1:-1], 1):
        axes[1, 0].axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].text(boundary, Config.BATCH_SIZE_LIST[-1] * 0.9, 
                       f'Stage {i+1}', rotation=90, va='top')
    
    # Plot 4: Stage Performance
    stage_performance = history_df.groupby('stage').agg({
        'train_loss': 'mean',
        'val_loss': 'mean'
    }).reset_index()
    
    x = np.arange(len(stage_performance))
    width = 0.35
    axes[1, 1].bar(x - width/2, stage_performance['train_loss'], width, label='Train', alpha=0.7)
    axes[1, 1].bar(x + width/2, stage_performance['val_loss'], width, label='Val', alpha=0.7)
    axes[1, 1].set_title('Average Loss per Stage')
    axes[1, 1].set_xlabel('Stage')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Stage {i}' for i in stage_performance['stage']])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_dynamics_with_batch_schedule.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Batch schedule visualization saved")

# Print stage-wise summary
print("\n" + "="*80)
print("STAGE-WISE PERFORMANCE SUMMARY")
print("="*80)

stage_summary = history_df.groupby(['stage', 'batch_size']).agg({
    'train_loss': ['mean', 'std', 'min'],
    'val_loss': ['mean', 'std', 'min']
}).round(5)

stage_summary.columns = ['_'.join(col).strip() for col in stage_summary.columns.values]
print(stage_summary)
stage_summary.to_csv('stage_wise_performance.csv')

