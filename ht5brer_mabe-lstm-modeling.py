import pandas as pd
import math
import numpy as np
from pathlib import Path
from typing import Literal, Union, Optional
from tqdm.auto import tqdm
import itertools
import os
import json
import yaml
import warnings
warnings.filterwarnings("ignore")

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict
from collections import Counter


class RawTableLoader:    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.train_path = root_path / "train.csv"
        self.test_path = root_path / "test.csv"
        self.submission_path = root_path / "sample_submission.csv"
        self.drop_body_parts = [
            'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
            'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
            'spine_1', 'spine_2',
            'tail_middle_1', 'tail_middle_2', 'tail_midpoint', "tail_tip", "neck"
        ]

    def read_train(self):
        df = pd.read_csv(self.train_path)
        return df

    def read_test(self):
        return pd.read_csv(self.test_path)

    def read_submission(self):
        return pd.read_csv(self.submission_path)

    def read_tracking(self, data_type: Literal["train", "test"], lab_id, video_id):
        track = pd.read_parquet(self.root_path / f"{data_type}_tracking" / lab_id / f"{video_id}.parquet")
        track = track[~track["bodypart"].isin(self.drop_body_parts)]
        track_piv = track.pivot_table(index=["video_frame"], columns=["mouse_id", "bodypart"], values=["x", "y"])
        track_piv = track_piv.reorder_levels([1,0,2], axis=1)

        full_range = range(track_piv.index.min(), track_piv.index.max() + 1)
        track_piv = track_piv.reindex(full_range)
        
        return track_piv

    def read_annotation(self, data_type: Literal["train"], lab_id, video_id):
        return pd.read_parquet(self.root_path / f"{data_type}_annotation" / lab_id / f"{video_id}.parquet")


def interpolate_missing_values(data: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """
    時系列データの欠損値を線形補間で補完する
    
    Parameters:
    -----------
    data : pd.DataFrame
        時系列データ（各列が変数、各行が時点）
    max_gap : int, default=10
        補完する最大連続欠損フレーム数
    
    Returns:
    --------
    pd.DataFrame
        補完後のデータフレーム
    """
    result = data.copy()
    
    for column in result.columns:
        result[column] = interpolate_series(result[column], max_gap)
    
    return result

def interpolate_series(series: pd.Series, max_gap: int = 10) -> pd.Series:
    """
    単一の時系列に対する欠損値補間
    
    Parameters:
    -----------
    series : pd.Series
        時系列データ
    max_gap : int
        補完する最大連続欠損フレーム数
    
    Returns:
    --------
    pd.Series
        補間後の時系列データ
    """
    result = series.copy()
    is_null = result.isnull()
    
    if not is_null.any():
        return result
    
    # 連続する欠損区間を特定
    null_groups = identify_null_groups(is_null)
    
    for start_idx, end_idx in null_groups:
        gap_length = end_idx - start_idx + 1
        
        # 指定フレーム数以内の欠損のみ補完
        if gap_length <= max_gap:
            # 前の有効値を取得
            before_val = get_last_valid_before(result, start_idx)
            # 後の有効値を取得
            after_val = get_first_valid_after(result, end_idx)
            
            if before_val is not None and after_val is not None:
                # 線形補間を実行
                interpolated_values = linear_interpolate(
                    before_val, after_val, gap_length
                )
                result.iloc[start_idx:end_idx+1] = interpolated_values
    
    return result

def identify_null_groups(is_null: pd.Series) -> list:
    """
    連続する欠損値の区間を特定
    
    Returns:
    --------
    list of tuple
        (開始インデックス, 終了インデックス) のリスト
    """
    groups = []
    start = None
    
    for i, null_val in enumerate(is_null):
        if null_val and start is None:
            start = i
        elif not null_val and start is not None:
            groups.append((start, i - 1))
            start = None
    
    # 最後まで欠損が続く場合
    if start is not None:
        groups.append((start, len(is_null) - 1))
    
    return groups

def get_last_valid_before(series: pd.Series, index: int) -> Optional[float]:
    """指定インデックス前の最後の有効値を取得"""
    if index == 0:
        return None
    
    for i in range(index - 1, -1, -1):
        if not pd.isna(series.iloc[i]):
            return series.iloc[i]
    return None

def get_first_valid_after(series: pd.Series, index: int) -> Optional[float]:
    """指定インデックス後の最初の有効値を取得"""
    if index == len(series) - 1:
        return None
    
    for i in range(index + 1, len(series)):
        if not pd.isna(series.iloc[i]):
            return series.iloc[i]
    return None

def linear_interpolate(start_val: float, end_val: float, num_points: int) -> np.ndarray:
    """
    線形補間による値の生成
    
    Parameters:
    -----------
    start_val : float
        開始値
    end_val : float
        終了値
    num_points : int
        補間する点数
    
    Returns:
    --------
    np.ndarray
        補間された値の配列
    """
    if num_points == 1:
        return np.array([(start_val + end_val) / 2], dtype=np.float32)
    
    # 線形補間：y = start_val + (end_val - start_val) * t
    # t は 1/(num_points+1) から num_points/(num_points+1) まで
    t_values = np.linspace(1, num_points, num_points) / (num_points + 1)
    interpolated = np.float32(start_val + (end_val - start_val) * t_values)
    
    return interpolated


class TrackingDataset:
    def __init__(self, lab_id, video_id, phase, pix_per_cm_approx, tracking_df, annotation_df, behaviors_labeled):
        self.lab_id = lab_id
        self.video_id = video_id
        self.phase = phase
        self.pix_per_cm_approx = pix_per_cm_approx
        self.tracking_df = tracking_df
        self.annotaion_df = annotation_df
        self.behaviors_labeled = behaviors_labeled

    def get_active_action(self, agent_id, target_id):
        if agent_id == target_id:
            at_key = f"mouse{agent_id},self"
        else:    
            at_key = f"mouse{agent_id},mouse{target_id}"

        active_actions = []
        for action in self.behaviors_labeled:
            if at_key in action:
                active_actions.append(action.split(",")[-1])
        return active_actions        

    @classmethod
    def from_row(cls, row, phase, loader):
        tracking_df = loader.read_tracking(phase, row.lab_id, row.video_id)
        tracking_df /= row.pix_per_cm_approx
        if phase == "train":
            annotation_df = loader.read_annotation(phase, row.lab_id, row.video_id)
        else:
            annotation_df = None
        behaviors_labeled = list(set(json.loads(row.behaviors_labeled)))
        behaviors_labeled = sorted([b.replace("'", "") for b in behaviors_labeled])
        return cls(row.lab_id, row.video_id, phase, row.pix_per_cm_approx, tracking_df, annotation_df, behaviors_labeled)

    def get_data_for_single_mouse_model(self, is_imputation):
        if is_imputation:
            tracking_df = interpolate_missing_values(self.tracking_df, max_gap=90)
        else:
            tracking_df = self.tracking_df.copy()

        data_for_single_mouse_model = {}

        for mouse_id in tracking_df.columns.get_level_values(0).unique().tolist():
            single_mouse_tracking = tracking_df[mouse_id]

        
            single_mouse_tracking_w_action = single_mouse_tracking.copy()
            single_mouse_tracking_w_action.columns = [f"{bp}_{xy}" for xy, bp in single_mouse_tracking_w_action.columns]

            if self.annotaion_df is not None:
                single_mouse_annotation = self.annotaion_df[(self.annotaion_df["agent_id"] == mouse_id) & (self.annotaion_df["target_id"] == mouse_id)]
                current_cols = single_mouse_tracking_w_action.columns.tolist()
                for action, start_frame, stop_frame in single_mouse_annotation[["action", "start_frame", "stop_frame"]].values:
                    if action not in current_cols:
                        single_mouse_tracking_w_action[action] = 2
                        current_cols.append(action)
                
                    single_mouse_tracking_w_action.loc[start_frame, action] = 0
                    if start_frame < stop_frame:
                        single_mouse_tracking_w_action.loc[start_frame+1: stop_frame+1, action] = 1

            dict_key = f"{self.lab_id}-{self.video_id}-{mouse_id}-{mouse_id}"
            data_for_single_mouse_model[dict_key] = {
                "tracking": single_mouse_tracking_w_action,
                "active_action": self.get_active_action(mouse_id, mouse_id),
                "video_id": self.video_id
            }
        return data_for_single_mouse_model

    def get_data_for_double_mouse_model(self, is_imputation):
        if is_imputation:
            tracking_df = interpolate_missing_values(self.tracking_df, max_gap=90)
        else:
            tracking_df = self.tracking_df.copy()

        data_for_double_mouse_model = {}

        for agent_mouse_id, target_mouse_id in itertools.permutations(tracking_df.columns.get_level_values(0).unique().tolist(), 2):
            agent_mouse_tracking = tracking_df[agent_mouse_id]
            agent_mouse_tracking.columns = [f"agent_{bp}_{xy}" for xy, bp in agent_mouse_tracking.columns]
            target_mouse_tracking = tracking_df[target_mouse_id]
            target_mouse_tracking.columns = [f"target_{bp}_{xy}" for xy, bp in target_mouse_tracking.columns]
            double_mouse_tracking_w_action = pd.concat([agent_mouse_tracking, target_mouse_tracking], axis=1)

            if self.annotaion_df is not None:
                double_mouse_annotation = self.annotaion_df[(self.annotaion_df["agent_id"] == agent_mouse_id) & (self.annotaion_df["target_id"] == target_mouse_id)]
                current_cols = double_mouse_tracking_w_action.columns.tolist()
                for action, start_frame, stop_frame in double_mouse_annotation[["action", "start_frame", "stop_frame"]].values:
                    if action not in current_cols:
                        double_mouse_tracking_w_action[action] = 2
                        current_cols.append(action)
                
                    double_mouse_tracking_w_action.loc[start_frame, action] = 0
                    try:
                        if start_frame < stop_frame:
                            double_mouse_tracking_w_action.loc[start_frame+1: stop_frame+1, action] = 1
                    except:
                        display(double_mouse_tracking_w_action)
                        print(start_frame, stop_frame, self.video_id)
            dict_key = f"{self.lab_id}-{self.video_id}-{agent_mouse_id}-{target_mouse_id}"
            data_for_double_mouse_model[dict_key] = {
                "tracking": double_mouse_tracking_w_action,
                "active_action": self.get_active_action(agent_mouse_id, target_mouse_id),
                "video_id": self.video_id,
            }
        return data_for_double_mouse_model


def get_feature_and_label_cols(trackings):
    feature_cols = []
    target_cols = []

    tracking_columns = []
    for _, tracking in trackings.items():
        tracking_columns += tracking["tracking"].columns.tolist()
    tracking_columns = list(set(tracking_columns))

    for col in tracking_columns:
        if ("_x" in col) or ("_y" in col):
            feature_cols.append(col)
        else:
            target_cols.append(col)
    return feature_cols, target_cols


SINGLE_LAB_LIST = [
    "AdaptableSnail",
    # "BoisterousParrot",
    # "CautiousGiraffe", # Only Muli
    # "DeliriousFly",
    # "ElegantMink",
    "GroovyShrew",
    # "InvincibleJellyfish",
    # "JovialSwallow",
    "LyricalHare",
    "NiftyGoldfinch",
    # "PleasantMeerkat",
    # "ReflectiveManatee",
    # "SparklingTapir",
    "TranquilPanther",
    "UppityFerret"
]

DOUBLE_LAB_LIST = [
    "AdaptableSnail",
    "BoisterousParrot",
    "CautiousGiraffe", # Only Muli
    "DeliriousFly",
    "ElegantMink",
    "GroovyShrew",
    "InvincibleJellyfish",
    "JovialSwallow",
    "LyricalHare",
    "NiftyGoldfinch",
    "PleasantMeerkat",
    "ReflectiveManatee",
    "SparklingTapir",
    "TranquilPanther",
    "UppityFerret"
]


# SEQ_LEN = 90
# WINDOWS = 20
# TRAIN_RATE = 0.5
# LAB_ID = SINGLE_LAB_LIST[5]
# PHASE = "train"
# ACTION_TYPE = "double"
# if ACTION_TYPE == "single":
#     LAB_LIST = SINGLE_LAB_LIST
# elif ACTION_TYPE == "double":
#     LAB_LIST = DOUBLE_LAB_LIST
# LAB_ID = LAB_LIST[1]


def load_tracking_dataset(lab_id, phase, action_type, meta_df):
    trackings = {}
    for _, row in tqdm(meta_df[meta_df["lab_id"] == lab_id].iterrows()):
        if "MABe22" in row.lab_id: continue
        try:
            tracking_dataset = TrackingDataset.from_row(row, phase, loader)
            features_col = tracking_dataset.tracking_df.columns
            if action_type == "single":
                trackings |= tracking_dataset.get_data_for_single_mouse_model(is_imputation=True)
            elif action_type == "double":
                trackings |= tracking_dataset.get_data_for_double_mouse_model(is_imputation=True)
        except FileNotFoundError as e:
            print(e)
    return trackings


def standarization_columns(trackings, feature_cols, target_cols):
    standarized_trackings = []
    for _, tracking_dict in trackings.items():
        tracking = tracking_dict["tracking"]
        # 説明変数のカラム数が足りないデータに対し、欠損列を追加
        for f_col in feature_cols:
            if f_col not in tracking.columns:
                tracking[f_col] = np.nan
    
        if len(set(tracking.columns) & set(target_cols)) and (len(target_cols) > 0) == 0:
            continue
    
        # 目的変数のカラムが足りないデータに対し、非アクション値を追加
        for t_col in target_cols:
            if t_col not in tracking.columns:
                tracking[t_col] = -1
    
        fixed_columns = feature_cols + target_cols
        standarized_trackings.append(tracking[fixed_columns])
    return standarized_trackings

def get_max_cm(trackings, feature_cols):
    max_cm = 0
    for tracking in trackings:
        _max_cm = tracking[feature_cols].fillna(0).max().max()
        if max_cm < _max_cm:
            max_cm = _max_cm
    return max_cm

def standarization_tracks_vals(trackings, feature_cols, max_cm=None):
    if max_cm is None:
        max_cm = get_max_cm(trackings, feature_cols)
    standarized_trackings = []
    for tracking in trackings:
        tracking[feature_cols] /= max_cm
        standarized_trackings.append(tracking)
    return standarized_trackings, max_cm

def train_test_split(trackings, train_rate):
    tracking_train_list = []
    tracking_valid_list = []
    
    for tracking in trackings:
        L = len(tracking)
        tracking_train_list.append(tracking[tracking.index < L * train_rate].values)
        tracking_valid_list.append(tracking[tracking.index >= L * train_rate].values)

    return tracking_train_list, tracking_valid_list


def get_weight_torch(trackings, target_cols):
    weights = {k: {0: 0, 1: 0, 2: 0, "total": 0} for k in target_cols}
    for tracking in trackings:
        for k in target_cols:
            count = Counter(tracking[k].tolist())
            for v, count in count.items():
                if v == -1:
                    continue
                weights[k][v] += count
                weights[k]["total"] += count

    torch_weight = torch.ones((len(target_cols), 3))
    for ki, k in enumerate(target_cols):
        action_torch_weights = torch.ones(3)
        for v in range(3):
            action_torch_weights[v] = weights[k]["total"] / weights[k][v]
        torch_weight[ki] = action_torch_weights
    return torch_weight



class MouseBehaviorDataset(Dataset):
    def __init__(self, data_list: List[np.ndarray], sequence_length: int = 150, n_cols=1, windows=1):
        """
        data_list: List of arrays, each with shape (N_k, C+target_dims)
        sequence_length: Window size for training
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        for mouse_data in data_list:
            is_end = False
            # Assume last columns are multi-label targets
            features = mouse_data[:, :-n_cols]  # C features (with missing values)
            targets = mouse_data[:, -n_cols:]   # 3 actions (a, b, c)
            
            # Create sliding windows
            step = math.ceil((len(mouse_data) - sequence_length) / windows) + 1
            for i in range(step):
                seq_features = features[i*windows:i*windows+sequence_length]
                seq_targets = targets[i*windows:i*windows+sequence_length]
                if len(seq_features) < sequence_length:
                    seq_features = features[-sequence_length:]
                    seq_targets = targets[-sequence_length:]
                # if len(seq_features) < sequence_length:
                #     seq_features = features[-sequence_length:]
                #     seq_targets = targets[-sequence_length:]
                #     is_end = True
                
                self.sequences.append(seq_features.astype(np.float32))
                self.targets.append(seq_targets.astype(np.int64))
                # if is_end:
                #     break
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.targets[idx])


class MouseBehaviorTestDataset(Dataset):
    def __init__(self, data_list: List[np.ndarray], sequence_length: int = 150, windows=1):
        """
        data_list: List of arrays, each with shape (N_k, C+target_dims)
        sequence_length: Window size for training
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        for mouse_data in data_list:
            is_end = False
            # Assume last columns are multi-label targets
            
            # Create sliding windows
            step = math.ceil((len(mouse_data) - sequence_length) / windows) + 1
            for i in range(step):
                seq_features = mouse_data[i*windows:i*windows+sequence_length]
                if len(seq_features) < sequence_length:
                    seq_features = mouse_data[-sequence_length:]
                
                self.sequences.append(seq_features.astype(np.float32))
                # if is_end:
                #     break
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])


class MaskedBiLSTM(nn.Module):
    def __init__(self, 
                 input_dim: int, spatial_hidden_dim=32, 
                 lstm_input_dim=32, lstm_hidden_dim: int = 32, 
                 num_actions: int = 1, num_layers: int = 4):
        super().__init__()
        # self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, spatial_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(spatial_hidden_dim, lstm_input_dim),
            nn.ReLU()
        )
        
        # BiLSTM layers
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        
        # Action classifiers (3 actions: a, b, c)
        self.action_classifiers = nn.ModuleList([
            nn.Linear(lstm_hidden_dim * 2, 3) for _ in range(num_actions)  # 3 classes: B, I, O
        ])

        self.layer_norm = nn.LayerNorm(lstm_hidden_dim*2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.1)

    def _init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)

            elif "weight_hh" in name:
                nn.init.orthognal_(param.data)

            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(9)
                param.data[n//4:n//2].fill(1)
        for classifier in self.action_classifiers:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(classifier.bias)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, input_dim = x.shape
        
        # Create mask for missing values (assume NaN represents missing)
        if mask is None:
            mask = ~torch.isnan(x).any(dim=-1)  # (batch, seq_len)
        
        # Replace NaN with zeros
        x = torch.nan_to_num(x, -1.0)
        x = self.spatial_encoder(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # # Apply mask to LSTM output
        # if mask is not None:
        #     mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
        #     lstm_out = lstm_out * mask_expanded.float()
        
        # Multi-label classification for each action
        action_logits = []
        for classifier in self.action_classifiers:
            logits = classifier(lstm_out)  # (batch, seq_len, 3)
            action_logits.append(logits)
        
        return torch.stack(action_logits, dim=-2)


def train_model(train_data: List[np.ndarray], 
                val_data: List[np.ndarray],
                torch_weight,
                input_dim: int = None,
                sequence_length: int = 150,
                epochs: int = 2,
                batch_size: int = 512,
                learning_rate: float = 0.001, n_cols=1, windows=1, suffix=""):
    
    # Create datasets
    train_dataset = MouseBehaviorDataset(train_data, sequence_length, n_cols, windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,      # GPU転送を高速化
    num_workers=4)
    
    val_dataset = MouseBehaviorDataset(val_data, sequence_length, n_cols, windows)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,      # GPU転送を高速化
    num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedBiLSTM(input_dim, num_actions=n_cols)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
      # Ignore masked positions

    patience = 10
    minibatch_count = 0
    best_val_acc = -np.inf
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_features)  # (batch, seq, num_actions, 3)
            
            # Calculate loss for each action
            loss = 0
            for action_idx in range(n_cols):
                action_pred = predictions[:, :, action_idx]  # (batch, seq, 3)
                action_target = batch_targets[:, :, action_idx]  # (batch, seq)
                criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch_weight[action_idx].to(device))
                _loss = criterion(action_pred.reshape(-1, 3), action_target.reshape(-1))
                loss += _loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
        
        if val_data:
            tr_f1, _ = evaluate_model(model, train_loader, device, n_cols)
            print(f'Train F1: {tr_f1:.4f}') 
            val_f1, val_loss = evaluate_model(model, val_loader, device, n_cols)
            print(f'Validation F1: {val_f1:.4f}') 
            # torch.save(model.state_dict(), f"model_{epoch}_{suffix}.pth")
            scheduler.step(val_loss)
    torch.save(model.state_dict(), f"model_{suffix}.pth")
    return model


def predict(model, test_data: List[np.ndarray], sequence_length: int = 150, batch_size: int = 32, windows=1):
    """Inference function"""
    test_dataset = MouseBehaviorTestDataset(test_data, sequence_length, windows)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,      # GPU転送を高速化
    num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            
            # Forward pass
            bio_predictions = model(batch_features)  # (batch, seq, num_actions, 3)
            bio_predictions = F.softmax(bio_predictions, dim=-1)
            
            # Convert to multi-label predictions
            # multilabel_pred = bio_to_multilabel(bio_predictions)
            all_predictions.append(bio_predictions.cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)

def evaluate_model(model, data_loader, device, n_cols):
    """Evaluation function"""
    model.eval()
    action_f1s = {i: {"tps": 0, "fps": 0, "fns": 0} for i in range(n_cols)}
    beta = 1
    val_loss = 0
    
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = model(batch_features)
            bio_pred = predictions.argmax(dim=-1)

            for i in range(n_cols):
                targets_flat = batch_targets[:, :, i].flatten()
                pred_flat = bio_pred[:, :, i].flatten()

                criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch_weight[i].to(device))
                _loss = criterion(predictions[:, :, i].reshape(-1, 3), targets_flat.reshape(-1))
                val_loss += _loss / len(data_loader)
    
                target_active = (targets_flat == 0) | (targets_flat == 1)
                target_inactive = (targets_flat == 2)
                pred_active = (pred_flat == 0) | (pred_flat == 1)
                pred_inactive = (pred_flat == 2)
                action_f1s[i]["tps"] += (target_active & pred_active).sum().item()
                action_f1s[i]["fns"] += (target_active & pred_inactive).sum().item()
                action_f1s[i]["fps"] += (target_inactive & pred_active).sum().item()

    fscore = 0
    for i in range(n_cols):
        try:
            fbeta = (1 + beta**2) * action_f1s[i]["tps"] / ((1 + beta**2) * action_f1s[i]["tps"] + beta**2 * action_f1s[i]["fns"] + action_f1s[i]["fps"])
        except:
            fbeta = 0
        fscore += fbeta / n_cols
    return fscore, val_loss


def fixed_pred_per_frame(frame_len, prediction, target_cols, seq_len, windows):
    final_prob = np.zeros((frame_len, len(target_cols), 3))
    step = math.ceil((frame_len - seq_len) // windows) + 1
    p_index = 0
    for i in range(step):
        if len(final_prob[i*WINDOWS:i*WINDOWS+SEQ_LEN, :, :]) == SEQ_LEN:
            final_prob[i*WINDOWS:i*WINDOWS+SEQ_LEN, :, :] += prediction[p_index]
        else:
            final_prob[-SEQ_LEN:, :, :] += prediction[p_index]
        p_index += 1
    final_prob = final_prob / final_prob.sum(axis=-1, keepdims=True)
    return final_prob


# Robustify function
def robustify(submission, meta_df):

    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame').reset_index(drop=True)
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        non_fixed = True
        while non_fixed:
            last_stop_frame = 0
            for i, (_, row) in enumerate(group.iterrows()):
                if row['start_frame'] < last_stop_frame:
                    if row['stop_frame'] < last_stop_frame:
                        mask[i] = False
                    else:
                        group.loc[i, "start_frame"] = last_stop_frame
                        break
                else:
                    last_stop_frame = row['stop_frame']
            else:
                non_fixed = False
        group_list.append(group[mask])
    submission = pd.concat(group_list)
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    submission = submission.reset_index(drop=True)
    return submission


# SEQ_LEN = 90
# TRAIN_RATE = 0.5
# PHASE = "train"
# thres_info = {}
# label_name_info = {}

# root_dir = Path("/kaggle/input/MABe-mouse-behavior-detection")
# loader = RawTableLoader(root_dir)
# train_df = loader.read_train()

# for action_type in ["single", "double"]:
#     if action_type == "single":
#         LAB_LIST = SINGLE_LAB_LIST
#         WINDOWS = 1
#     elif action_type == "double":
#         LAB_LIST = DOUBLE_LAB_LIST
#         WINDOWS = 20

#     for lab_id in LAB_LIST:
#         trackings = load_tracking_dataset(lab_id=lab_id, phase=PHASE, action_type=action_type, meta_df=train_df)
#         feature_cols, target_cols = get_feature_and_label_cols(trackings)
        
#         trackings_for_modeling = standarization_columns(trackings, feature_cols, target_cols)
#         trackings_for_modeling, max_cp = standarization_tracks_vals(trackings_for_modeling, feature_cols)
#         label_name_info[f"{lab_id}_{action_type}"] = {
#             "feature_cols": feature_cols,
#             "label_cols": target_cols,
#             "max_cp": max_cp
#         }
#         train_tracking, valid_tracking = train_test_split(trackings_for_modeling, train_rate=TRAIN_RATE)
#         torch_weight = get_weight_torch(trackings_for_modeling, target_cols)
    
#         model = train_model(
#             train_tracking, 
#             valid_tracking, 
#             torch_weight=torch_weight,
#             input_dim=len(feature_cols),
#             sequence_length=SEQ_LEN, 
#             epochs=10, 
#             batch_size=256, 
#             learning_rate=0.001,
#             n_cols=len(target_cols),
#             windows=WINDOWS,
#             suffix = f"{action_type}_{lab_id}",
#         )

#         scores = []
#         for valid_tracking_val in valid_tracking:
#             prediction = predict(
#                 model, 
#                 [valid_tracking_val[:, :-len(target_cols)]], 
#                 sequence_length=SEQ_LEN, 
#                 batch_size=1024, 
#                 n_cols=len(target_cols),
#                 windows=WINDOWS
#             )
        
#             pred_per_frame = fixed_pred_per_frame(len(valid_tracking_val), prediction, target_cols, SEQ_LEN, WINDOWS)
        
#             score_info = {}
#             for i, target_col in enumerate(target_cols):
#                 score_info[f"{target_col}_pred"] = 1 - pred_per_frame[:, i, 2]
#                 score_info[f"{target_col}_correct"] = valid_tracking_val[:, -len(target_cols) + i]
                
#             score = pd.DataFrame(score_info)
#             scores.append(score)
#         scores = pd.concat(scores)
        
#         fbeta = 0
#         thres_info_per_lab = {}
#         for target_col in target_cols:
#             best_fbeta_per_action = 0
#             best_fbeta_thres_per_action = -1
#             for thres in np.arange(0, 1, 0.01):
#                 tps = len(scores[(scores[f"{target_col}_pred"] > thres) & scores[f"{target_col}_correct"].isin([0,1])])
#                 fps = len(scores[(scores[f"{target_col}_pred"] > thres) & scores[f"{target_col}_correct"].isin([2])])
#                 fns = len(scores[(scores[f"{target_col}_pred"] <= thres) & scores[f"{target_col}_correct"].isin([0,1])])
#                 beta = 1
#                 _fbeta = (1 + beta**2) * tps / ((1 + beta**2) * tps + beta**2 * fns + fps)
#                 if best_fbeta_per_action < _fbeta:
#                     best_fbeta_thres_per_action = thres
#                     best_fbeta_per_action = _fbeta
#             thres_info_per_lab[target_col] = float(best_fbeta_thres_per_action)
#             fbeta += best_fbeta_per_action / len(target_cols)
#             print(f"action: {target_col}, thres: {best_fbeta_thres_per_action}, fbeta: {best_fbeta_per_action}")
#         thres_info[f"{lab_id}_{action_type}"] = thres_info_per_lab
#         print(f"fbeta: {fbeta}")

# with open('config.yaml', 'w') as f:
#     yaml.dump(label_name_info, f, default_flow_style=False, sort_keys=False)

# with open('thres_info.yaml', 'w') as f:
#     yaml.dump(thres_info, f, default_flow_style=False, sort_keys=False)


SEQ_LEN = 90
MODEL_DIR = "/kaggle/input/mabe-lstm-models"
TRAIN_RATE = 0.5
PHASE = "test"

root_dir = Path("/kaggle/input/MABe-mouse-behavior-detection")
loader = RawTableLoader(root_dir)
test_df = loader.read_test()

with open(os.path.join(MODEL_DIR, 'config.yaml'), 'r', encoding='utf-8') as f:
    label_name_info = yaml.safe_load(f)

with open(os.path.join(MODEL_DIR, 'thres_info.yaml'), 'r', encoding='utf-8') as f:
    thres_info = yaml.safe_load(f)

predictions = []

for action_type in ["single", "double"]:
    if action_type == "single":
        LAB_LIST = SINGLE_LAB_LIST
        WINDOWS = 1
    elif action_type == "double":
        LAB_LIST = DOUBLE_LAB_LIST
        WINDOWS = 20

    for lab_id in LAB_LIST:
        trackings = load_tracking_dataset(lab_id=lab_id, phase=PHASE, action_type=action_type, meta_df=test_df)
        feature_cols = label_name_info[f"{lab_id}_{action_type}"]["feature_cols"]
        target_cols = label_name_info[f"{lab_id}_{action_type}"]["label_cols"]
        max_cm = label_name_info[f"{lab_id}_{action_type}"]["max_cp"]
        thres_info_per_lab = thres_info[f"{lab_id}_{action_type}"]

        
        trackings_for_modeling = standarization_columns(trackings, feature_cols, [])
        trackings_for_modeling, _ = standarization_tracks_vals(trackings_for_modeling, feature_cols, max_cm=max_cm)

        if len(trackings_for_modeling) == 0:
            continue
        trackings_for_modeling = [tracking_for_modeling.values for tracking_for_modeling in trackings_for_modeling]
        input_dim = trackings_for_modeling[0].shape[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        suffix = f"{action_type}_{lab_id}"
        state_dict = torch.load(os.path.join(MODEL_DIR, f"model_{suffix}.pth"), map_location='cpu')
        model = MaskedBiLSTM(input_dim, num_actions=len(target_cols))
        model.load_state_dict(state_dict)
        model.to(device)
        
        scores = []
        for tracking_for_modeling, (keys, tracking_info) in zip(trackings_for_modeling, trackings.items()):
            _, video_id, agent_id, target_id = keys.split("-")
            prediction = predict(
                model, 
                [tracking_for_modeling], 
                sequence_length=SEQ_LEN, 
                batch_size=1024, 
                windows=WINDOWS
            )

            pred_per_frame = fixed_pred_per_frame(len(tracking_for_modeling), prediction, target_cols, SEQ_LEN, WINDOWS)
            active_action = tracking_info["active_action"]

            assert len(set(target_cols) | set(active_action)) == len(target_cols)
            for i, target_col in enumerate(target_cols):
                if target_col not in active_action:
                    continue
                score_info = {}
                score_info["video_frame"] = tracking_info["tracking"].index
                score_info["action"] = target_col
                score_info["pred"] = ((1 - pred_per_frame[:, i, 2]) > thres_info_per_lab[target_col]) * 1
                score = pd.DataFrame(score_info)

                score = score[score["pred"] == 1]
                score = score.sort_values("video_frame")
                score['frame_diff'] = score["video_frame"].diff()
                
                # 新しいセグメントの開始点を特定（差分が1より大きい場合）
                score['new_segment'] = (score['frame_diff'] > 1) | (score['frame_diff'].isna())
                score['segment_id'] = score['new_segment'].cumsum()
                
                # セグメントごとに開始・終了フレームを取得
                for segment_id in score['segment_id'].unique():
                    segment_frames = score[score['segment_id'] == segment_id]["video_frame"]
                    start_frame = segment_frames.min()
                    stop_frame = segment_frames.max()
                    
                    predictions.append({
                        "video_id": video_id,
                        "agent_id": f"mouse{agent_id}",
                        "target_id": f"mouse{target_id}" if agent_id != target_id else "self",
                        'action': target_col,
                        'start_frame': start_frame,
                        'stop_frame': stop_frame
                    })
            
                        

predictions = pd.DataFrame(predictions).reset_index(drop=True)
robustify_predictions = robustify(predictions, test_df)
robustify_predictions.index.name = 'row_id'

if len(prediction) == 0:
    predictions = pd.DataFrame(
        dict(
            video_id=438887472,
            agent_id='mouse1',
            target_id='self',
            action='rear',
            start_frame='278',
            stop_frame='500'
        ), index=[44]
    )
robustify_predictions.to_csv('submission.csv')


# import polars as pl
# class HostVisibleError(Exception):
#     pass

# def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
#     label_frames: defaultdict[str, set[int]] = defaultdict(set)
#     prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

#     for row in lab_solution.to_dicts():
#         label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

#     for video in lab_solution['video_id'].unique():
#         active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
#         active_labels: set[str] = set(json.loads(active_labels))
#         predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

#         for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
#             if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
#                 continue
           
#             new_frames = set(range(row['start_frame'], row['stop_frame']))
#             new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
#             prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
#             if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
#                 print(prediction_pair)
#                 print(predicted_mouse_pairs[prediction_pair])
#                 print(new_frames)
#                 raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
#             prediction_frames[row['prediction_key']].update(new_frames)
#             predicted_mouse_pairs[prediction_pair].update(new_frames)

#     tps = defaultdict(int)
#     fns = defaultdict(int)
#     fps = defaultdict(int)
#     for key, pred_frames in prediction_frames.items():
#         action = key.split('_')[-1]
#         matched_label_frames = label_frames[key]
#         tps[action] += len(pred_frames.intersection(matched_label_frames))
#         fns[action] += len(matched_label_frames.difference(pred_frames))
#         fps[action] += len(pred_frames.difference(matched_label_frames))

#     distinct_actions = set()
#     for key, frames in label_frames.items():
#         action = key.split('_')[-1]
#         distinct_actions.add(action)
#         if key not in prediction_frames:
#             fns[action] += len(frames)

#     action_f1s = []
#     for action in distinct_actions:
#         if tps[action] + fns[action] + fps[action] == 0:
#             action_f1s.append(0)
#         else:
#             action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
#     return sum(action_f1s) / len(action_f1s)

# def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
#     if len(solution) == 0 or len(submission) == 0:
#         raise ValueError('Missing solution or submission data')

#     expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

#     for col in expected_cols:
#         if col not in solution.columns:
#             raise ValueError(f'Solution is missing column {col}')
#         if col not in submission.columns:
#             raise ValueError(f'Submission is missing column {col}')

#     solution: pl.DataFrame = pl.DataFrame(solution)
#     submission: pl.DataFrame = pl.DataFrame(submission)
#     assert (solution['start_frame'] <= solution['stop_frame']).all()
#     assert (submission['start_frame'] <= submission['stop_frame']).all()
#     solution_videos = set(solution['video_id'].unique())
#     submission = submission.filter(pl.col('video_id').is_in(solution_videos))

#     solution = solution.with_columns(
#         pl.concat_str(
#             [
#                 pl.col('video_id').cast(pl.Utf8),
#                 pl.col('agent_id').cast(pl.Utf8),
#                 pl.col('target_id').cast(pl.Utf8),
#                 pl.col('action'),
#             ],
#             separator='_',
#         ).alias('label_key'),
#     )
#     submission = submission.with_columns(
#         pl.concat_str(
#             [
#                 pl.col('video_id').cast(pl.Utf8),
#                 pl.col('agent_id').cast(pl.Utf8),
#                 pl.col('target_id').cast(pl.Utf8),
#                 pl.col('action'),
#             ],
#             separator='_',
#         ).alias('prediction_key'),
#     )

#     lab_scores = []
#     for lab in solution['lab_id'].unique():
#         lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
#         lab_videos = set(lab_solution['video_id'].unique())
#         lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
#         lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

#     return sum(lab_scores) / len(lab_scores)

# def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
#     solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
#     submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
#     return mouse_fbeta(solution, submission, beta=beta)

