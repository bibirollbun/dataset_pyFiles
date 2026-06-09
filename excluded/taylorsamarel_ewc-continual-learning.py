# MABe Challenge - Memory-Optimized Neural Network
# Reduced memory footprint while maintaining core functionality

validate_or_submit = 'submit'
verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os
import gc
from collections import defaultdict, deque
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks

warnings.filterwarnings('ignore')
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== CONFIGURATION (MEMORY OPTIMIZED) ====================
class Config:
    """Memory-optimized configuration"""
    # Chunk processing - REDUCED
    chunk_size = 5
    replay_buffer_size = 3000  # Down from 15000
    max_chunks_per_action = 10  # Down from 20
    diversity_k = 30  # Down from 100
    
    # Validation
    val_ratio = 0.2
    convergence_patience = 4
    min_improvement = 0.005
    
    # Neural network architecture - REDUCED
    mlp_hidden_dims = [256, 128]  # Down from [512, 256, 128]
    
    # Temporal convolution - REDUCED
    use_temporal_conv = True
    temporal_conv_channels = [32, 64, 32]  # Down from [64, 128, 64]
    temporal_kernel_sizes = [5, 5, 3]
    
    # LSTM - REDUCED
    use_lstm = True
    lstm_hidden_size = 64  # Down from 128
    lstm_num_layers = 2
    
    # Attention - REDUCED
    use_attention = True
    attention_heads = 2  # Down from 4
    attention_dim = 128
    
    # Graph network - DISABLED
    use_graph_net = False
    graph_hidden_dim = 64
    graph_layers = 2
    
    dropout = 0.35
    learning_rate = 0.0005
    batch_size = 128  # Down from 256
    epochs_per_chunk = 15
    early_stopping_patience = 5
    
    # EWC - REDUCED
    ewc_lambda = 2000
    fisher_sample_size = 500  # Down from 2000
    
    # Advanced feature engineering - DISABLED
    use_polynomial_features = False  # Was True
    poly_degree = 2
    poly_interaction_only = True
    
    use_wavelets = False  # Was True
    use_phase_features = True
    use_higher_order_stats = False  # Was True
    
    # Feature windows - REDUCED
    temporal_windows = [5, 15, 30, 60, 120]  # Down from 11 windows
    long_range_windows = [60, 120, 240]  # Down from 8 windows
    skip_distances = [15, 30, 60, 120]  # Down from 8 distances
    entropy_windows = [10, 30, 60]  # Down from 5
    spectral_windows = [30, 60]  # Down from 3
    
    drop_body_parts = [
        'headpiece_bottombackleft', 'headpiece_bottombackright',
        'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
        'headpiece_topbackleft', 'headpiece_topbackright',
        'headpiece_topfrontleft', 'headpiece_topfrontright',
        'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
    ]

config = Config()

# ==================== SCORING FUNCTIONS ====================
class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames = defaultdict(set)
    prediction_frames = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels = set(json.loads(active_labels))
        predicted_mouse_pairs = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for same frame')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution missing {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission missing {col}')

    solution = pl.DataFrame(solution)
    submission = pl.DataFrame(submission)
    
    solution = solution.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8),
                      pl.col('target_id').cast(pl.Utf8), pl.col('action')], separator='_').alias('label_key'))
    submission = submission.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8),
                      pl.col('target_id').cast(pl.Utf8), pl.col('action')], separator='_').alias('prediction_key'))

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# ==================== ADVANCED NEURAL NETWORKS ====================

class TemporalConvBlock(nn.Module):
    """1D Convolutional block for temporal features"""
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for feature importance"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1)
        return x

class AdvancedBehaviorNet(nn.Module):
    """Advanced neural network with multiple architectural components"""
    
    def __init__(self, input_dim, 
                 mlp_hidden=[256, 128],
                 use_temporal_conv=True, temporal_channels=[32, 64, 32],
                 use_lstm=True, lstm_hidden=64, lstm_layers=2,
                 use_attention=True, attention_heads=2,
                 dropout=0.35):
        super().__init__()
        
        self.use_temporal_conv = use_temporal_conv
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        
        current_dim = input_dim
        
        if use_temporal_conv:
            conv_layers = []
            in_ch = 1
            for out_ch in temporal_channels:
                conv_layers.append(TemporalConvBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            self.temporal_conv = nn.Sequential(*conv_layers)
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            current_dim += temporal_channels[-1]
        
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, 
                               batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
            current_dim += lstm_hidden
        
        if use_attention:
            self.attention_proj = nn.Linear(input_dim, attention_heads * 32)
            self.attention = MultiHeadAttention(attention_heads * 32, attention_heads)
            current_dim += attention_heads * 32
        
        mlp_layers = []
        prev_dim = current_dim
        for hidden_dim in mlp_hidden:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, x):
        features = [x]
        
        if self.use_temporal_conv:
            x_temp = x.unsqueeze(1)
            x_temp = self.temporal_conv(x_temp)
            x_temp = self.temporal_pool(x_temp).squeeze(-1)
            features.append(x_temp)
        
        if self.use_lstm:
            x_lstm = x.unsqueeze(1)
            _, (h_n, _) = self.lstm(x_lstm)
            x_lstm = h_n[-1]
            features.append(x_lstm)
        
        if self.use_attention:
            x_attn = self.attention_proj(x)
            x_attn = self.attention(x_attn)
            features.append(x_attn)
        
        x_combined = torch.cat(features, dim=1)
        output = self.mlp(x_combined)
        
        return output

class EWCModel:
    """Neural network with EWC for continual learning"""
    
    def __init__(self, input_dim, 
                 mlp_hidden=[256, 128],
                 use_temporal_conv=True, temporal_channels=[32, 64, 32],
                 use_lstm=True, lstm_hidden=64, lstm_layers=2,
                 use_attention=True, attention_heads=2,
                 dropout=0.35, learning_rate=0.0005, 
                 ewc_lambda=2000, device='cpu'):
        
        self.device = device
        self.model = AdvancedBehaviorNet(
            input_dim, mlp_hidden, use_temporal_conv, temporal_channels,
            use_lstm, lstm_hidden, lstm_layers, use_attention, attention_heads,
            dropout
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        self.ewc_lambda = ewc_lambda
        
        self.fisher_matrices = {}
        self.optimal_params = {}
        self.task_count = 0
        
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
    def compute_fisher_matrix(self, dataloader, sample_size=500):
        """Compute Fisher Information Matrix"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        samples_seen = 0
        for inputs, targets in dataloader:
            if samples_seen >= sample_size:
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            
            log_probs = torch.log(probs + 1e-8) * (targets == 1).float() + \
                       torch.log(1 - probs + 1e-8) * (targets == 0).float()
            loss = -log_probs.mean()
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(dataloader.dataset)
            
            samples_seen += len(inputs)
        
        return fisher
    
    def ewc_loss(self):
        """Compute EWC regularization"""
        loss = 0
        for task_id in range(self.task_count):
            for n, p in self.model.named_parameters():
                fisher = self.fisher_matrices[task_id][n]
                optimal = self.optimal_params[task_id][n]
                loss += (fisher * (p - optimal) ** 2).sum()
        return self.ewc_lambda * loss
    
    def fit(self, X, y, val_X=None, val_y=None, epochs=15, batch_size=128, 
            early_stopping_patience=5):
        """Train model"""
        
        if not self.scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if val_X is not None and val_y is not None:
            val_X_scaled = self.scaler.transform(val_X)
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(val_X_scaled),
                torch.FloatTensor(val_y)
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_dataloader = None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                
                bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = ((1 - pt) ** 2 * bce_loss).mean()
                
                loss = focal_loss + self.ewc_loss()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"      Early stop at epoch {epoch+1}")
                    break
        
        self.update_ewc(dataloader)
    
    def update_ewc(self, dataloader):
        """Update Fisher and optimal params"""
        fisher = self.compute_fisher_matrix(dataloader, config.fisher_sample_size)
        optimal = {n: p.data.clone() for n, p in self.model.named_parameters()}
        
        self.fisher_matrices[self.task_count] = fisher
        self.optimal_params[self.task_count] = optimal
        self.task_count += 1
    
    def evaluate(self, dataloader):
        """Evaluate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs).squeeze()
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])

# ==================== REPLAY BUFFER ====================
class DiverseReplayBuffer:
    """Diverse replay buffer"""
    def __init__(self, max_size: int, diversity_k: int):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.diversity_k = diversity_k
        self.action_counts = defaultdict(int)
    
    def add(self, X: pd.DataFrame, y: np.ndarray, video_id: int):
        """Add samples"""
        unique_actions = np.unique(y)
        for action_val in unique_actions:
            mask = (y == action_val)
            X_action = X[mask]
            
            if len(X_action) == 0:
                continue
            
            if len(X_action) > self.diversity_k:
                indices = np.linspace(0, len(X_action)-1, self.diversity_k, dtype=int)
                X_selected = X_action.iloc[indices]
            else:
                X_selected = X_action
            
            for idx, row in X_selected.iterrows():
                self.buffer.append({
                    'X': row.to_dict(),
                    'y': action_val,
                    'video_id': video_id
                })
                self.action_counts[action_val] += 1
    
    def sample(self, n_samples: int, columns: list):
        """Sample from buffer"""
        if len(self.buffer) == 0:
            return None, None
        
        sample_size = min(n_samples, len(self.buffer))
        step = max(1, len(self.buffer) // sample_size)
        indices = list(range(0, len(self.buffer), step))[:sample_size]
        
        samples = [self.buffer[i] for i in indices]
        X_list, y_list = [], []
        
        for sample in samples:
            try:
                row = [sample['X'].get(col, np.nan) for col in columns]
                X_list.append(row)
                y_list.append(sample['y'])
            except:
                continue
        
        if len(X_list) > 0:
            return pd.DataFrame(X_list, columns=columns), np.array(y_list)
        return None, None

# ==================== FEATURE ENGINEERING (MEMORY OPTIMIZED) ====================

def safe_entropy(x):
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < 2:
        return 0.0
    try:
        hist, _ = np.histogram(x_clean, bins=10)
        return entropy(hist + 1e-10)
    except:
        return 0.0

def safe_fft_features(x):
    """Extract FFT features"""
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < 4:
        return 0.0, 0.0, 0.0
    try:
        fft_vals = np.abs(fft(x_clean)[:len(x_clean)//2])
        freqs = fftfreq(len(x_clean), 1.0)[:len(x_clean)//2]
        
        peak_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
        peak_power = fft_vals.max()
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
        
        return peak_freq, peak_power, spectral_centroid
    except:
        return 0.0, 0.0, 0.0

def safe_welch_features(x, window=30):
    """Power spectral density via Welch method"""
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < window:
        return 0.0, 0.0
    try:
        f, psd = welch(x_clean, nperseg=min(window, len(x_clean)))
        
        dominant_freq = f[np.argmax(psd)]
        spectral_entropy = entropy(psd + 1e-10)
        
        return dominant_freq, spectral_entropy
    except:
        return 0.0, 0.0

def add_spectral_features(X, data, windows=[30, 60]):
    """Add spectral features - REDUCED"""
    for part in ['nose', 'body_center', 'tail_base']:
        if part in data.columns.get_level_values(0):
            for axis in ['x', 'y']:
                if axis in data[part].columns:
                    for window in windows:
                        peak_freq = data[part][axis].rolling(window, min_periods=1).apply(
                            lambda x: safe_fft_features(x)[0], raw=True
                        )
                        X[f'{part}_{axis}_peak_freq_{window}'] = peak_freq.astype(np.float32)
                        
                        peak_power = data[part][axis].rolling(window, min_periods=1).apply(
                            lambda x: safe_fft_features(x)[1], raw=True
                        )
                        X[f'{part}_{axis}_peak_power_{window}'] = peak_power.astype(np.float32)
    
    return X

def add_phase_features(X, data):
    """Add phase and circular statistics"""
    if 'nose' in data.columns.get_level_values(0) and 'tail_base' in data.columns.get_level_values(0):
        dx = data['nose']['x'] - data['tail_base']['x']
        dy = data['nose']['y'] - data['tail_base']['y']
        angle = np.arctan2(dy, dx)
        
        X['body_angle_rad'] = angle.astype(np.float32)
        X['body_angle_sin'] = np.sin(angle).astype(np.float32)
        X['body_angle_cos'] = np.cos(angle).astype(np.float32)
        
        X['angular_vel'] = angle.diff().astype(np.float32)
        X['angular_accel'] = X['angular_vel'].diff().astype(np.float32)
        
        for window in [10, 30, 60]:
            angle_var = angle.rolling(window, min_periods=1).var()
            X[f'phase_coherence_{window}'] = (1 / (1 + angle_var)).astype(np.float32)
    
    return X

def add_interaction_features(X, data):
    """Add interaction terms between key body parts"""
    available_parts = data.columns.get_level_values(0).unique()
    
    if 'nose' in available_parts and 'tail_base' in available_parts:
        nose_speed = np.sqrt(data['nose']['x'].diff()**2 + data['nose']['y'].diff()**2)
        tail_speed = np.sqrt(data['tail_base']['x'].diff()**2 + data['tail_base']['y'].diff()**2)
        
        X['nose_tail_speed_product'] = (nose_speed * tail_speed).astype(np.float32)
        X['nose_tail_speed_ratio'] = (nose_speed / (tail_speed + 1e-6)).astype(np.float32)
    
    if 'ear_left' in available_parts and 'ear_right' in available_parts:
        ear_dist = np.sqrt(
            (data['ear_left']['x'] - data['ear_right']['x'])**2 +
            (data['ear_left']['y'] - data['ear_right']['y'])**2
        )
        X['ear_dist'] = ear_dist.astype(np.float32)
        X['ear_dist_change'] = ear_dist.diff().astype(np.float32)
        X['ear_dist_accel'] = ear_dist.diff().diff().astype(np.float32)
    
    return X

def add_autoregressive_features(X, data, lags=[1, 3, 5, 10]):
    """Add autoregressive features - REDUCED"""
    if 'body_center' in data.columns.get_level_values(0):
        center_x = data['body_center']['x']
        center_y = data['body_center']['y']
        
        for lag in lags:
            X[f'ar_x_lag{lag}'] = center_x.shift(lag).astype(np.float32)
            X[f'ar_y_lag{lag}'] = center_y.shift(lag).astype(np.float32)
            
            X[f'autocorr_x_{lag}'] = (center_x * center_x.shift(lag)).astype(np.float32)
            X[f'autocorr_y_{lag}'] = (center_y * center_y.shift(lag)).astype(np.float32)
    
    return X

def transform_single(single_mouse, body_parts_tracked):
    """Comprehensive single mouse features - MEMORY OPTIMIZED"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    X = pd.DataFrame()
    
    # Base pairwise distances
    for part1, part2 in itertools.combinations(body_parts_tracked, 2):
        if part1 in available_body_parts and part2 in available_body_parts:
            X[f"{part1}_{part2}"] = np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False).astype(np.float32)
    
    # Speed features - REDUCED
    if 'ear_left' in single_mouse.columns and 'ear_right' in single_mouse.columns:
        for dt in [1, 5, 10]:
            shifted = single_mouse[['ear_left', 'ear_right']].shift(dt)
            X[f'speed_left_{dt}'] = np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False).astype(np.float32)
            X[f'speed_right_{dt}'] = np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False).astype(np.float32)
    
    # Body geometry
    if 'nose' in available_body_parts and 'body_center' in available_body_parts and 'tail_base' in available_body_parts:
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        
        dot_product = (v1['x'] * v2['x'] + v1['y'] * v2['y'])
        norm_v1 = np.sqrt(v1['x']**2 + v1['y']**2)
        norm_v2 = np.sqrt(v2['x']**2 + v2['y']**2)
        
        X['body_angle'] = (dot_product / (norm_v1 * norm_v2 + 1e-6)).astype(np.float32)
        
        body_angle = np.arctan2(v1['y'], v1['x']) - np.arctan2(v2['y'], v2['x'])
        X['angle_change'] = body_angle.diff().astype(np.float32)
        X['angle_std'] = body_angle.rolling(15, min_periods=1).std().astype(np.float32)
    
    # Body length dynamics
    if 'nose' in available_body_parts and 'tail_base' in available_body_parts:
        body_length = np.sqrt(
            (single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
            (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2
        ).astype(np.float32)
        
        X['body_length'] = body_length
        X['length_change'] = body_length.diff().astype(np.float32)
        X['length_std'] = body_length.rolling(20, min_periods=1).std().astype(np.float32)
        X['length_accel'] = body_length.diff().diff().astype(np.float32)
    
    # Multi-scale temporal features
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']
        
        for dt in [1, 3, 5, 10, 15]:
            velocity = np.sqrt(
                (center_x - center_x.shift(dt))**2 + 
                (center_y - center_y.shift(dt))**2
            ) / dt
            X[f'velocity_{dt}'] = velocity.astype(np.float32)
            X[f'accel_{dt}'] = velocity.diff().astype(np.float32)
        
        for window in config.temporal_windows:
            X[f'cx_mean_{window}'] = center_x.rolling(window, min_periods=1, center=True).mean().astype(np.float32)
            X[f'cy_mean_{window}'] = center_y.rolling(window, min_periods=1, center=True).mean().astype(np.float32)
            X[f'cx_std_{window}'] = center_x.rolling(window, min_periods=1, center=True).std().astype(np.float32)
            X[f'cy_std_{window}'] = center_y.rolling(window, min_periods=1, center=True).std().astype(np.float32)
            
            X[f'x_range_{window}'] = (center_x.rolling(window, min_periods=1, center=True).max() - 
                                      center_x.rolling(window, min_periods=1, center=True).min()).astype(np.float32)
            X[f'y_range_{window}'] = (center_y.rolling(window, min_periods=1, center=True).max() - 
                                      center_y.rolling(window, min_periods=1, center=True).min()).astype(np.float32)
        
        for window in [15, 30, 60]:
            dist_traveled = np.sqrt(center_x.diff()**2 + center_y.diff()**2).rolling(window, min_periods=1).sum()
            displacement = np.sqrt(
                (center_x - center_x.shift(window))**2 + 
                (center_y - center_y.shift(window))**2
            )
            X[f'tortuosity_{window}'] = (dist_traveled / (displacement + 1e-6)).astype(np.float32)
    
    # Head dynamics
    if 'nose' in available_body_parts:
        nose_x = single_mouse['nose']['x']
        nose_y = single_mouse['nose']['y']
        
        X['head_vel'] = np.sqrt(nose_x.diff()**2 + nose_y.diff()**2).astype(np.float32)
        X['head_accel'] = X['head_vel'].diff().astype(np.float32)
        X['head_jerk'] = X['head_accel'].diff().astype(np.float32)
        
        head_dir = np.arctan2(nose_y.diff(), nose_x.diff())
        X['head_dir_change'] = head_dir.diff().astype(np.float32)
        X['head_dir_std'] = head_dir.rolling(10, min_periods=1).std().astype(np.float32)
    
    # Long-range features
    for window in config.long_range_windows:
        if 'body_center' in available_body_parts:
            center_x = single_mouse['body_center']['x']
            center_y = single_mouse['body_center']['y']
            
            X[f'long_disp_{window}'] = np.sqrt(
                (center_x - center_x.shift(window))**2 + 
                (center_y - center_y.shift(window))**2
            ).astype(np.float32)
            
            dx = center_x.diff()
            dy = center_y.diff()
            speed = np.sqrt(dx**2 + dy**2)
            
            X[f'burst_{window}'] = (
                speed.rolling(window, min_periods=1).max() / 
                (speed.rolling(window, min_periods=1).mean() + 1e-6)
            ).astype(np.float32)
    
    # Skip-gram features
    for skip in config.skip_distances:
        if 'nose' in available_body_parts:
            X[f'nose_skip_{skip}'] = np.sqrt(
                (single_mouse['nose']['x'] - single_mouse['nose']['x'].shift(skip))**2 +
                (single_mouse['nose']['y'] - single_mouse['nose']['y'].shift(skip))**2
            ).astype(np.float32)
    
    # Behavior transitions
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']
        vx = center_x.diff()
        vy = center_y.diff()
        speed = np.sqrt(vx**2 + vy**2)
        acceleration = speed.diff()
        
        for window in [30, 60, 120]:
            X[f'move_var_{window}'] = speed.rolling(window, min_periods=1).std().astype(np.float32)
            
            active_threshold = speed.quantile(0.7) if len(speed) > 0 and not speed.isna().all() else 0
            is_active = (speed > active_threshold).astype(float)
            transitions = is_active.diff().abs()
            X[f'transitions_{window}'] = transitions.rolling(window, min_periods=1).sum().astype(np.float32)
            X[f'active_persist_{window}'] = is_active.rolling(window, min_periods=1).mean().astype(np.float32)
            X[f'accel_changes_{window}'] = acceleration.rolling(window, min_periods=1).std().astype(np.float32)
    
    # Advanced features (conditional)
    if config.use_phase_features:
        X = add_phase_features(X, single_mouse)
    
    X = add_spectral_features(X, single_mouse, config.spectral_windows)
    X = add_interaction_features(X, single_mouse)
    X = add_autoregressive_features(X, single_mouse)
    
    # Entropy features
    for window in config.entropy_windows:
        if 'body_center' in single_mouse.columns.get_level_values(0):
            for axis in ['x', 'y']:
                if axis in single_mouse['body_center'].columns:
                    ent = single_mouse['body_center'][axis].rolling(window, min_periods=1).apply(
                        safe_entropy, raw=True
                    )
                    X[f'entropy_{axis}_{window}'] = ent.astype(np.float32)
    
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Comprehensive pair features - MEMORY OPTIMIZED"""
    available_body_parts_A = mouse_pair['A'].columns.get_level_values(0)
    available_body_parts_B = mouse_pair['B'].columns.get_level_values(0)
    
    X = pd.DataFrame()
    
    # Pairwise distances
    for part1, part2 in itertools.product(body_parts_tracked, repeat=2):
        if part1 in available_body_parts_A and part2 in available_body_parts_B:
            X[f"AB_{part1}_{part2}"] = np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False).astype(np.float32)
    
    # Relative speeds - REDUCED
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        for dt in [1, 5, 10]:
            shifted_A = mouse_pair['A']['ear_left'].shift(dt)
            shifted_B = mouse_pair['B']['ear_left'].shift(dt)
            X[f'speed_A_{dt}'] = np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False).astype(np.float32)
            X[f'speed_B_{dt}'] = np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False).astype(np.float32)
            X[f'speed_diff_{dt}'] = (X[f'speed_A_{dt}'] - X[f'speed_B_{dt}']).astype(np.float32)
    
    # Relative orientation
    if 'nose' in available_body_parts_A and 'tail_base' in available_body_parts_A and \
       'nose' in available_body_parts_B and 'tail_base' in available_body_parts_B:
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        
        dot_product = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y'])
        norm_A = np.sqrt(dir_A['x']**2 + dir_A['y']**2)
        norm_B = np.sqrt(dir_B['x']**2 + dir_B['y']**2)
        
        X['rel_orient'] = (dot_product / (norm_A * norm_B + 1e-6)).astype(np.float32)
        
        A_to_B = mouse_pair['B']['body_center'] - mouse_pair['A']['body_center'] if 'body_center' in available_body_parts_A else mouse_pair['B']['nose'] - mouse_pair['A']['nose']
        dot_A = (dir_A['x'] * A_to_B['x'] + dir_A['y'] * A_to_B['y'])
        dot_B = (-dir_B['x'] * A_to_B['x'] + -dir_B['y'] * A_to_B['y'])
        
        X['A_facing'] = (dot_A / (norm_A * np.sqrt(A_to_B['x']**2 + A_to_B['y']**2) + 1e-6)).astype(np.float32)
        X['B_facing'] = (dot_B / (norm_B * np.sqrt(A_to_B['x']**2 + A_to_B['y']**2) + 1e-6)).astype(np.float32)
        X['mutual_face'] = (X['A_facing'] * X['B_facing']).astype(np.float32)
    
    # Approach dynamics - REDUCED
    if 'nose' in available_body_parts_A and 'nose' in available_body_parts_B:
        nose_dist = np.sqrt(
            (mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
            (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2
        ).astype(np.float32)
        
        for dt in [1, 5, 10]:
            X[f'nose_change_{dt}'] = (nose_dist - nose_dist.shift(dt)).astype(np.float32)
            X[f'approach_{dt}'] = (-X[f'nose_change_{dt}'] / dt).astype(np.float32)
        
        # Proximity zones
        for thresh in [3, 5, 10, 15, 20]:
            is_close = (nose_dist < thresh).astype(np.float32)
            for window in [5, 10, 30, 60]:
                X[f'close_{thresh}_{window}'] = is_close.rolling(window, min_periods=1).mean().astype(np.float32)
    
    # Distance features
    if 'body_center' in available_body_parts_A and 'body_center' in available_body_parts_B:
        center_dist = np.sqrt(
            (mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
            (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2
        ).astype(np.float32)
        
        X['very_close'] = (center_dist < 5.0).astype(np.float32)
        X['close'] = ((center_dist >= 5.0) & (center_dist < 10.0)).astype(np.float32)
        X['medium'] = ((center_dist >= 10.0) & (center_dist < 20.0)).astype(np.float32)
        X['far'] = (center_dist >= 20.0).astype(np.float32)
        
        for window in config.temporal_windows:
            X[f'dist_mean_{window}'] = center_dist.rolling(window, min_periods=1, center=True).mean().astype(np.float32)
            X[f'dist_std_{window}'] = center_dist.rolling(window, min_periods=1, center=True).std().astype(np.float32)
            X[f'dist_min_{window}'] = center_dist.rolling(window, min_periods=1, center=True).min().astype(np.float32)
            X[f'dist_max_{window}'] = center_dist.rolling(window, min_periods=1, center=True).max().astype(np.float32)
            
            dist_var = center_dist.rolling(window, min_periods=1, center=True).var()
            X[f'interact_{window}'] = (1 / (1 + dist_var)).astype(np.float32)
        
        # Coordinated movement
        A_vel_x = mouse_pair['A']['body_center']['x'].diff()
        A_vel_y = mouse_pair['A']['body_center']['y'].diff()
        B_vel_x = mouse_pair['B']['body_center']['x'].diff()
        B_vel_y = mouse_pair['B']['body_center']['y'].diff()
        
        vel_alignment = (A_vel_x * B_vel_x + A_vel_y * B_vel_y) / (
            np.sqrt(A_vel_x**2 + A_vel_y**2) * np.sqrt(B_vel_x**2 + B_vel_y**2) + 1e-6
        )
        
        X['vel_align'] = vel_alignment.astype(np.float32)
        
        for window in [5, 10, 30, 60]:
            X[f'vel_align_mean_{window}'] = vel_alignment.rolling(window, min_periods=1, center=True).mean().astype(np.float32)
            X[f'vel_align_std_{window}'] = vel_alignment.rolling(window, min_periods=1, center=True).std().astype(np.float32)
        
        # Following behavior - REDUCED
        for lag in [5, 10, 15]:
            A_vel_x_lag = A_vel_x.shift(lag)
            A_vel_y_lag = A_vel_y.shift(lag)
            
            follow_score = (A_vel_x_lag * B_vel_x + A_vel_y_lag * B_vel_y) / (
                np.sqrt(A_vel_x_lag**2 + A_vel_y_lag**2) * np.sqrt(B_vel_x**2 + B_vel_y**2) + 1e-6
            )
            X[f'follow_{lag}'] = follow_score.astype(np.float32)
        
        # Long-range interaction
        for long_window in config.long_range_windows:
            X[f'dist_long_{long_window}'] = center_dist.rolling(long_window, min_periods=1, center=True).mean().astype(np.float32)
    
    # Add features for mouse A
    X = add_spectral_features(X, mouse_pair['A'], config.spectral_windows)
    X = add_interaction_features(X, mouse_pair['A'])
    X = add_autoregressive_features(X, mouse_pair['A'])
    
    if config.use_phase_features:
        X = add_phase_features(X, mouse_pair['A'])
    
    return X

# ==================== DATA GENERATOR ====================
def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """Generate mouse data"""
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'):
            continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except:
            continue
            
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@config.drop_body_parts)")
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid
        gc.collect()
        
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = (pvid / row.pix_per_cm_approx).astype(np.float32)

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label, video_id
                    else:
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions, video_id
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    if len(vid_agent_actions) == 0:
                        continue
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label, video_id
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions, video_id

# ==================== CHUNK LEARNING MANAGER ====================
class ChunkLearningManager:
    """Manages chunk-based learning with advanced neural networks"""
    
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.action_models = {}
        self.replay_buffers = {}
        self.feature_columns = {}
        self.action_thresholds = {}
        self.convergence_history = defaultdict(list)
        self.train_videos = set()
        self.val_videos = set()
        self.imputer_fitted = False
        self.imputer_columns = []
    
    def split_videos(self, video_ids):
        """Split for validation"""
        video_ids = list(video_ids)
        np.random.shuffle(video_ids)
        split_idx = int(len(video_ids) * (1 - config.val_ratio))
        self.train_videos = set(video_ids[:split_idx])
        self.val_videos = set(video_ids[split_idx:])
        if verbose:
            print(f"      Train/Val: {len(self.train_videos)}/{len(self.val_videos)}")
    
    def check_convergence(self, action):
        """Check convergence"""
        history = self.convergence_history[action]
        if len(history) < config.convergence_patience + 1:
            return False
        
        recent_best = max(history[-config.convergence_patience:])
        overall_best = max(history[:-config.convergence_patience]) if len(history) > config.convergence_patience else 0
        improvement = recent_best - overall_best
        
        return improvement < config.min_improvement
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """Optimize threshold"""
        best_f1, best_thresh = 0, 0.27
        for thresh in np.arange(0.15, 0.5, 0.05):
            f1 = f1_score(y_true, y_pred_proba >= thresh, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        return best_thresh, best_f1
    
    def train_in_chunks(self, mode_type: str, body_parts: list, train_dataset: pd.DataFrame):
        """Train with chunk-based processing"""
        
        print(f"  Training {mode_type} models with advanced neural networks...")
        
        # Collect actions and videos
        all_actions = set()
        video_ids = []
        all_feature_cols = set()
        
        for switch, data, meta, label, vid_id in generate_mouse_data(train_dataset, 'train',
                                                                      generate_single=(mode_type == 'single'),
                                                                      generate_pair=(mode_type == 'pair')):
            if switch == mode_type:
                all_actions.update(label.columns)
                video_ids.append(vid_id)
                
                # Collect feature columns
                if mode_type == 'single':
                    X_sample = transform_single(data, body_parts)
                else:
                    X_sample = transform_pair(data, body_parts)
                all_feature_cols.update(X_sample.columns)
                
                del data, meta, label, X_sample
                gc.collect()
        
        print(f"    Found {len(all_actions)} actions, {len(set(video_ids))} videos")
        
        if len(set(video_ids)) == 0:
            return
        
        # Store all possible feature columns and fit imputer
        self.imputer_columns = sorted(list(all_feature_cols))
        dummy_data = pd.DataFrame(0, index=[0], columns=self.imputer_columns)
        self.imputer.fit(dummy_data)
        self.imputer_fitted = True
        
        self.split_videos(set(video_ids))
        
        # Train each action
        for action in all_actions:
            if verbose:
                print(f"    Training: {action}")
            
            self.replay_buffers[action] = DiverseReplayBuffer(config.replay_buffer_size, config.diversity_k)
            
            chunk_count = 0
            val_X_list, val_y_list = [], []
            ewc_model = None
            
            for switch, data, meta, label, vid_id in generate_mouse_data(train_dataset, 'train',
                                                                          generate_single=(mode_type == 'single'),
                                                                          generate_pair=(mode_type == 'pair')):
                if switch != mode_type or action not in label.columns:
                    continue
                
                try:
                    # Extract features
                    if mode_type == 'single':
                        X = transform_single(data, body_parts)
                    else:
                        X = transform_pair(data, body_parts)
                    
                    del data
                    gc.collect()
                    
                    # Get labels
                    mask = ~label[action].isna().values
                    X_action = X[mask]
                    y_action = label[action][mask].values.astype(int)
                    
                    if len(X_action) == 0:
                        continue
                    
                    # Store feature columns for this action
                    if action not in self.feature_columns:
                        self.feature_columns[action] = X_action.columns.tolist()
                    
                    # Align features to imputer columns
                    X_action = X_action.reindex(columns=self.imputer_columns, fill_value=0)
                    
                    # Impute
                    X_imputed = pd.DataFrame(
                        self.imputer.transform(X_action),
                        columns=self.imputer_columns,
                        index=X_action.index
                    )
                    
                    # Select only features for this action
                    X_imputed = X_imputed[self.feature_columns[action]]
                    
                    # Split train/val
                    if vid_id in self.val_videos:
                        val_X_list.append(X_imputed)
                        val_y_list.append(y_action)
                        del X, X_action, X_imputed
                        gc.collect()
                        continue
                    
                    if vid_id not in self.train_videos:
                        continue
                    
                    # Initialize model
                    if ewc_model is None:
                        input_dim = len(self.feature_columns[action])
                        ewc_model = EWCModel(
                            input_dim=input_dim,
                            mlp_hidden=config.mlp_hidden_dims,
                            use_temporal_conv=config.use_temporal_conv,
                            temporal_channels=config.temporal_conv_channels,
                            use_lstm=config.use_lstm,
                            lstm_hidden=config.lstm_hidden_size,
                            lstm_layers=config.lstm_num_layers,
                            use_attention=config.use_attention,
                            attention_heads=config.attention_heads,
                            dropout=config.dropout,
                            learning_rate=config.learning_rate,
                            ewc_lambda=config.ewc_lambda,
                            device=device
                        )
                    
                    # Get replay
                    X_replay, y_replay = self.replay_buffers[action].sample(1500, self.feature_columns[action])
                    
                    # Combine
                    if X_replay is not None and len(X_replay) > 0:
                        X_combined = pd.concat([X_imputed, X_replay], ignore_index=True)
                        y_combined = np.concatenate([y_action, y_replay])
                    else:
                        X_combined = X_imputed
                        y_combined = y_action
                    
                    # Train
                    if len(val_X_list) > 0:
                        val_X_concat = pd.concat(val_X_list, ignore_index=True)
                        val_y_concat = np.concatenate(val_y_list)
                        ewc_model.fit(
                            X_combined.values, y_combined,
                            val_X=val_X_concat.values, val_y=val_y_concat,
                            epochs=config.epochs_per_chunk,
                            batch_size=config.batch_size,
                            early_stopping_patience=config.early_stopping_patience
                        )
                    else:
                        ewc_model.fit(
                            X_combined.values, y_combined,
                            epochs=config.epochs_per_chunk,
                            batch_size=config.batch_size
                        )
                    
                    # Validate
                    if len(val_X_list) > 0:
                        try:
                            val_pred = ewc_model.predict_proba(val_X_concat.values)[:, 1]
                            _, val_f1 = self.optimize_threshold(val_y_concat, val_pred)
                            self.convergence_history[action].append(val_f1)
                            
                            if verbose:
                                print(f"      Chunk {chunk_count+1}: Val F1={val_f1:.3f}")
                        except:
                            pass
                    
                    # Add to replay
                    self.replay_buffers[action].add(X_imputed, y_action, vid_id)
                    
                    del X, X_action, X_imputed, X_combined, y_combined
                    gc.collect()
                    
                    chunk_count += 1
                    
                    if self.check_convergence(action):
                        if verbose:
                            print(f"      Converged at chunk {chunk_count}")
                        break
                    
                    if chunk_count >= config.max_chunks_per_action:
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"      Error: {e}")
                    continue
            
            # Optimize threshold
            if len(val_X_list) > 0 and ewc_model is not None:
                val_X_concat = pd.concat(val_X_list, ignore_index=True)
                val_y_concat = np.concatenate(val_y_list)
                
                try:
                    val_pred = ewc_model.predict_proba(val_X_concat.values)[:, 1]
                    best_thresh, best_f1 = self.optimize_threshold(val_y_concat, val_pred)
                    self.action_thresholds[action] = best_thresh
                    if verbose:
                        print(f"      Threshold: {best_thresh:.2f}, F1: {best_f1:.3f}, Tasks: {ewc_model.task_count}")
                except:
                    self.action_thresholds[action] = 0.27
            else:
                self.action_thresholds[action] = 0.27
            
            if ewc_model is not None:
                self.action_models[action] = ewc_model
            
            # Aggressive cleanup
            del val_X_list, val_y_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def predict(self, mode_type: str, body_parts: list, test_dataset: pd.DataFrame) -> list:
        """Predict on test data"""
        submission_parts = []
        
        for switch, data, meta, actions, vid_id in generate_mouse_data(test_dataset, 'test',
                                                                        generate_single=(mode_type == 'single'),
                                                                        generate_pair=(mode_type == 'pair')):
            if switch != mode_type:
                continue
            
            try:
                if mode_type == 'single':
                    X = transform_single(data, body_parts)
                else:
                    X = transform_pair(data, body_parts)
                
                del data
                gc.collect()
                
                # CRITICAL FIX: Align features to imputer columns BEFORE imputation
                X_aligned = X.reindex(columns=self.imputer_columns, fill_value=0)
                
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X_aligned),
                    columns=self.imputer_columns,
                    index=X_aligned.index
                )
                del X, X_aligned
                gc.collect()
                
                probs = pd.DataFrame(index=meta.video_frame)
                
                for action in actions:
                    if action in self.action_models:
                        model = self.action_models[action]
                        
                        if action in self.feature_columns:
                            cols = self.feature_columns[action]
                            X_action = X_imputed.reindex(columns=cols, fill_value=0)
                        else:
                            X_action = X_imputed
                        
                        try:
                            pred = model.predict_proba(X_action.values)
                            probs[action] = pred[:, 1]
                        except:
                            pass
                
                del X_imputed
                gc.collect()
                
                if probs.shape[1] > 0:
                    sub_part = self._predictions_to_submission(probs, meta)
                    submission_parts.append(sub_part)
            
            except Exception as e:
                if verbose:
                    print(f'  ERROR: {e}')
        
        return submission_parts
    
    def _predictions_to_submission(self, probs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """Convert to submission"""
        ama = np.argmax(probs.values, axis=1)
        max_probs = probs.values.max(axis=1)
        thresh_array = np.array([self.action_thresholds.get(probs.columns[i], 0.27) for i in ama])
        ama = np.where(max_probs >= thresh_array, ama, -1)
        
        ama = pd.Series(ama, index=meta.video_frame)
        changes_mask = (ama != ama.shift(1)).values
        ama_changes = ama[changes_mask]
        meta_changes = meta[changes_mask]
        mask = ama_changes.values >= 0
        mask[-1] = False
        
        submission_part = pd.DataFrame({
            'video_id': meta_changes['video_id'][mask].values,
            'agent_id': meta_changes['agent_id'][mask].values,
            'target_id': meta_changes['target_id'][mask].values,
            'action': probs.columns[ama_changes[mask].values],
            'start_frame': ama_changes.index[mask],
            'stop_frame': ama_changes.index[1:][mask[:-1]]
        })
        
        stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
        stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
        stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
        
        for i in range(len(submission_part)):
            video_id = submission_part.video_id.iloc[i]
            agent_id = submission_part.agent_id.iloc[i]
            target_id = submission_part.target_id.iloc[i]
            if i < len(stop_video_id):
                if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                    new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                    submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
            else:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        
        return submission_part

# ==================== ROBUSTIFY ====================
def robustify(submission, dataset):
    """Clean submission"""
    if len(submission) == 0:
        return submission
    
    submission = submission[submission.start_frame < submission.stop_frame]
    
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])
    
    submission = pd.concat(group_list) if group_list else pd.DataFrame()
    submission = submission.reset_index(drop=True)
    return submission

# ==================== MAIN ====================
def main():
    """Main execution"""
    print("Loading data...")
    train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
    test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
    
    body_parts_list = list(np.unique(train.body_parts_tracked))
    submission_list = []
    
    for section in range(1, len(body_parts_list)):
        body_parts_tracked_str = body_parts_list[section]
        
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
            print(f"\n{section}. Processing {len(body_parts_tracked)} body parts")
            
            if len(body_parts_tracked) > 5:
                body_parts_tracked = [b for b in body_parts_tracked if b not in config.drop_body_parts]
            
            train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
            test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
            
            if len(test_subset) == 0:
                print("  No test videos")
                continue
            
            # Single mouse
            if len(train_subset) > 0:
                manager_single = ChunkLearningManager()
                manager_single.train_in_chunks('single', body_parts_tracked, train_subset)
                
                if len(manager_single.action_models) > 0:
                    print(f"  Predicting on {len(test_subset)} test videos")
                    parts = manager_single.predict('single', body_parts_tracked, test_subset)
                    submission_list.extend(parts)
                
                del manager_single
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Pairs
            if len(train_subset) > 0:
                manager_pair = ChunkLearningManager()
                manager_pair.train_in_chunks('pair', body_parts_tracked, train_subset)
                
                if len(manager_pair.action_models) > 0:
                    print(f"  Predicting on {len(test_subset)} test videos")
                    parts = manager_pair.predict('pair', body_parts_tracked, test_subset)
                    submission_list.extend(parts)
                
                del manager_pair
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f'Exception: {e}')
            import traceback
            traceback.print_exc()
    
    # Final submission
    if len(submission_list) > 0:
        submission = pd.concat(submission_list, ignore_index=True)
        submission_robust = robustify(submission, test)
    else:
        submission_robust = pd.DataFrame({
            'video_id': [438887472],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'start_frame': [278],
            'stop_frame': [500]
        })
    
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('/kaggle/working/submission.csv')
    print(f"\nFinal submission: {len(submission_robust)} predictions")

if __name__ == '__main__':
    main()

