!pip install torch_geometric


import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.debug = True

#paths
cfg.output_dir = f"/kaggle/working/"
cfg.data_folder = f"/kaggle/input/MABe-mouse-behavior-detection"
cfg.train_df = f'/kaggle/input/MABe-mouse-behavior-detection/test.csv'

# stages
cfg.test = False
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.seed = 1994

    
# Sampling strategy for long videos
cfg.min_windows_per_video = 5  # Minimum windows to sample per video
cfg.max_windows_per_video = 1000 # Maximum windows to sample per video
cfg.windows_per_epoch_ratio = 1.0 # Sample 30% of possible windows per epoch

cfg.MASTER_SKELETON = [
    'nose',
    'ear_left',
    'ear_right',
    'head_center',
    'body_center',
    'tail_base'
]
cfg.NUM_MASTER_KEYPOINTS = len(cfg.MASTER_SKELETON)
cfg.MASTER_SKELETON_MAP = {name: i for i, name in enumerate(cfg.MASTER_SKELETON)}

cfg.tracked_bodyparts = ['body_center', 'ear_left', 'ear_right', 'forepaw_left', 'forepaw_right', 
    'head', 'headpiece_bottombackleft', 'headpiece_bottombackright', 
    'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft',
    'headpiece_topfrontright', 'hindpaw_left', 'hindpaw_right', 'hip_left',
    'hip_right', 'lateral_left', 'lateral_right', 'neck', 'nose', 'spine_1', 
    'spine_2', 'tail_base', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint', 'tail_tip'
]

cfg.set_behavior_classes = ['allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'biteobject',
    'chase', 'chaseattack', 'climb', 'defend', 'dig', 'disengage', 'dominance', 'dominancegroom', 
    'dominancemount', 'ejaculate', 'escape', 'exploreobject', 'flinch', 'follow', 'freeze', 'genitalgroom',
    'huddle', 'intromit', 'mount', 'rear', 'reciprocalsniff', 'rest', 'run', 'self', 'selfgroom', 
    'shepherd', 'sniff', 'sniffbody', 'sniffface', 'sniffgenital', 'submit', 'tussle', 'no_action'
]

cfg.set_mice = ['mouse1', 'mouse2', 'mouse3', 'mouse4']
cfg.max_pairs = len(cfg.set_mice) * len(cfg.set_mice)  # 16 if 4 mice

cfg.mouse_id_map = {1:0, 2:1, 3:2, 4:3}
cfg.mouse_id_to_string = {0:'mouse1', 1:'mouse2', 2:'mouse3', 3:'mouse4'}

cfg.action_id_map = {'allogroom': 0, 'approach': 1, 'attack': 2, 'attemptmount': 3, 'avoid': 4,
                     'biteobject': 5, 'chase': 6, 'chaseattack': 7, 'climb': 8, 'defend': 9, 'dig': 10,
                     'disengage': 11, 'dominance': 12, 'dominancegroom': 13, 'dominancemount': 14,
                     'ejaculate': 15, 'escape': 16, 'exploreobject': 17, 'flinch': 18, 'follow': 19,
                     'freeze': 20, 'genitalgroom': 21, 'huddle': 22, 'intromit': 23, 'mount': 24,
                     'rear': 25, 'reciprocalsniff': 26, 'rest': 27, 'run': 28, 'self': 29,
                     'selfgroom': 30, 'shepherd': 31, 'sniff': 32, 'sniffbody': 33, 'sniffface': 34,
                     'sniffgenital': 35, 'submit': 36, 'tussle': 37, 'no_action': 38}

cfg.id_to_action_map = {v:k for k,v in cfg.action_id_map.items()}

# DETECTED FROM ACTUAL DATASET - actions that only occur when agent_id == target_id
cfg.self_only_actions = ['biteobject', 'climb', 'dig', 'exploreobject', 'freeze', 'genitalgroom', 'huddle', 'rear', 'rest', 'run', 'selfgroom']

# DETECTED FROM ACTUAL DATASET - actions that only occur when agent_id != target_id
cfg.social_only_actions = ['allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'chase', 'chaseattack', 
                           'defend', 'disengage', 'dominance', 'dominancegroom', 'dominancemount', 'ejaculate', 
                           'escape', 'flinch', 'follow', 'intromit', 'mount', 'reciprocalsniff', 'shepherd',
                           'sniff', 'sniffbody', 'sniffface', 'sniffgenital', 'submit', 'tussle']
cfg.reverse_time = False

cfg.window_size = 512
cfg.feature_dim = 232 #280
cfg.per_mouse_feature_dim = cfg.feature_dim // len(cfg.set_mice)
cfg.bias_prob = 0.5  # Probability of sampling behavior-rich window during training
cfg.oversample_factor = 10
cfg.preprocessing_basedir = "/kaggle/working/preprocessed"

#model
cfg.model = "mdl_1"

encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=256
encoder_config.encoder_dim=256
encoder_config.num_layers=4
encoder_config.num_attention_heads= 16
encoder_config.feed_forward_expansion_factor=2
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.0
encoder_config.feed_forward_dropout_p= 0.0
encoder_config.attention_dropout_p= 0.0
encoder_config.conv_dropout_p= 0.0
encoder_config.conv_kernel_size= 51

cfg.encoder_config = encoder_config

cfg.use_bn= True
cfg.use_gnn = True 
cfg.cnn_extractor = False


# LOSS SETTINGS (for handling sparse labels)
cfg.use_focal_loss = True  # Start with weighted BCE, simpler to debug
cfg.focal_alpha = 0.25  # Weight for positive/negative samples in focal loss
cfg.focal_gamma = 2.0  # Focusing parameter for focal loss
cfg.pos_weight = 10.0  # High weight for positive samples due to extreme sparsity (0.1% positive)

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 100
cfg.eval_epochs = 5
cfg.lr = 1e-4  # Reduced learning rate for stability with larger windows
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.05
cfg.clip_grad = 0.
cfg.warmup = 5
cfg.batch_size = 64
cfg.batch_size_val = 64
cfg.mixed_precision = True # True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 4


#EVAL
cfg.calc_metric = True
cfg.metric = "metric_1"
cfg.save_val_data = True


import pandas as pd
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys
import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')

DATA_DIR = '/kaggle/input/MABe-mouse-behavior-detection'
PREPROC_DIR = '/kaggle/working//preprocessed'
os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(os.path.join(PREPROC_DIR, 'test'), exist_ok=True)

test_meta = pd.read_csv(os.path.join(DATA_DIR, '/kaggle/input/MABe-mouse-behavior-detection/test.csv'))
all_meta = test_meta.copy()  # For this example, only train

MASTER_SKELETON = [
    'nose',
    'ear_left',
    'ear_right',
    'head_center',
    'body_center',
    'tail_base'
]
NUM_MASTER_KEYPOINTS = len(MASTER_SKELETON)
MASTER_SKELETON_MAP = {name: i for i, name in enumerate(MASTER_SKELETON)}

fixed_bodyparts = cfg.tracked_bodyparts
num_bodyparts = len(fixed_bodyparts)
fixed_actions = cfg.set_behavior_classes
num_actions = len(fixed_actions)
action_map = cfg.action_id_map

NO_ACTION_IDX = action_map['no_action']  # GET THE NO_ACTION INDEX
print(f"Fixed actions ({num_actions}): {fixed_actions}")
print(f"no_action index: {NO_ACTION_IDX}")
# Max mice and pairs from cfg
set_mice = cfg.set_mice
max_mice = len(set_mice)  # 4 if 4 mice
max_pairs = cfg.max_pairs  # 16 if 4 mice

# Mouse map from cfg
mouse_id_map = cfg.mouse_id_map  # Assume {1: 0, 2: 1, 3: 2, 4: 3} or similar

# Frame counts
frame_counts = {'test': {}}

def process_tracking_video(row, split):
    lab_id = row['lab_id']
    video_id = row['video_id']
    pix_per_cm = row.get('pix_per_cm_approx', 1.0)
    
    if pd.isna(pix_per_cm) or pix_per_cm <= 0:
        print(f"Warning {video_id}: Invalid pix_per_cm ({pix_per_cm}), defaulting to 1.0")
        pix_per_cm = 1.0
    
    # Check for no mice from metadata
    num_mice = sum(1 for i in range(1, max_mice + 1) if pd.notna(row.get(f'mouse{i}_id')))
    if num_mice == 0:
        print(f"Skipping {video_id}: no mice")
        return
    
    # Load tracking parquet
    path = os.path.join(DATA_DIR, f'{split}_tracking', lab_id, f'{video_id}.parquet')
    if not os.path.exists(path):
        print(f"Warning: No tracking file for {video_id}")
        return
    df = pd.read_parquet(path)
    if df.empty:
        print(f"Warning: Empty tracking for {video_id}")
        return
    
    # Handle types
    df['video_frame'] = df['video_frame'].astype(int)
    df['mouse_id'] = df['mouse_id'].astype(int)
    max_frame = df['video_frame'].max()
    
    # Pivot to wide
    df_x = df.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    df_y = df.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    
    # Fill missing frames
    full_index = pd.Index(range(0, max_frame + 1), name='video_frame')
    df_x = df_x.reindex(full_index, fill_value=np.nan)
    df_y = df_y.reindex(full_index, fill_value=np.nan)
    num_frames = len(full_index)
    
    # Map mouse_ids to 0-based using cfg map
    current_mouse_levels = df_x.columns.levels[0]
    new_mouse_levels = [mouse_id_map.get(level, -1) for level in current_mouse_levels]
    if any(l < 0 for l in new_mouse_levels):
        print(f"Warning: Unknown mouse_ids in {video_id}")
    df_x.columns = df_x.columns.set_levels(new_mouse_levels, level=0)
    df_y.columns = df_y.columns.set_levels(new_mouse_levels, level=0)
    
    # Helper to get bodypart array [frames, mice], reindexed to all mice with NaN fill
    def get_bp_array(bp, is_x=True, fill_nan=True):
        df = df_x if is_x else df_y
        if bp not in df.columns.levels[1]:
            return np.full((num_frames, max_mice), np.nan, dtype=np.float32)
        bp_df = df.xs(bp, level=1, axis=1)
        bp_df = bp_df.reindex(columns=range(max_mice), fill_value=np.nan)
        return bp_df.values.astype(np.float32)
    
    # For master skeleton, create keypoints array
    keypoints = np.full((num_frames, max_mice, NUM_MASTER_KEYPOINTS, 2), np.nan, dtype=np.float32)
    
    # Vectorized mapping
    # Nose
    nose_x = get_bp_array('nose', is_x=True)
    nose_y = get_bp_array('nose', is_x=False)
    keypoints[:, :, MASTER_SKELETON_MAP['nose'], 0] = nose_x
    keypoints[:, :, MASTER_SKELETON_MAP['nose'], 1] = nose_y
    # Special fallback for labs like GroovyShrew
    if lab_id == 'GroovyShrew':
        head_x = get_bp_array('head', is_x=True)
        head_y = get_bp_array('head', is_x=False)
        keypoints[:, :, MASTER_SKELETON_MAP['nose'], 0] = np.where(np.isnan(nose_x), head_x, nose_x)
        keypoints[:, :, MASTER_SKELETON_MAP['nose'], 1] = np.where(np.isnan(nose_y), head_y, nose_y)
    
    # Ear left/right
    keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 0] = get_bp_array('ear_left', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 1] = get_bp_array('ear_left', is_x=False)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 0] = get_bp_array('ear_right', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 1] = get_bp_array('ear_right', is_x=False)
    
    # Tail base
    keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 0] = get_bp_array('tail_base', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 1] = get_bp_array('tail_base', is_x=False)
    
    # Head_Center: first head, then neck, then average ear_left/right
    head_x = get_bp_array('head', is_x=True)
    head_y = get_bp_array('head', is_x=False)
    neck_x = get_bp_array('neck', is_x=True)
    neck_y = get_bp_array('neck', is_x=False)
    ear_left_x = keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 0]
    ear_left_y = keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 1]
    ear_right_x = keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 0]
    ear_right_y = keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 1]
    
    avg_ear_x = np.nanmean(np.stack([ear_left_x, ear_right_x], axis=-1), axis=-1)
    avg_ear_y = np.nanmean(np.stack([ear_left_y, ear_right_y], axis=-1), axis=-1)
    
    head_center_x = np.where(~np.isnan(head_x), head_x,
                    np.where(~np.isnan(neck_x), neck_x, avg_ear_x))
    head_center_y = np.where(~np.isnan(head_y), head_y,
                    np.where(~np.isnan(neck_y), neck_y, avg_ear_y))
    keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 0] = head_center_x
    keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 1] = head_center_y
    
    # Body_Center: body_center, then average [spine_1, spine_2, hip_left, hip_right, neck], then (head_center + tail_base)/2
    body_center_x = get_bp_array('body_center', is_x=True)
    body_center_y = get_bp_array('body_center', is_x=False)
    
    spine1_x = get_bp_array('spine_1', is_x=True)
    spine1_y = get_bp_array('spine_1', is_x=False)
    spine2_x = get_bp_array('spine_2', is_x=True)
    spine2_y = get_bp_array('spine_2', is_x=False)
    hip_left_x = get_bp_array('hip_left', is_x=True)
    hip_left_y = get_bp_array('hip_left', is_x=False)
    hip_right_x = get_bp_array('hip_right', is_x=True)
    hip_right_y = get_bp_array('hip_right', is_x=False)
    neck_x = get_bp_array('neck', is_x=True)  # Reuse
    neck_y = get_bp_array('neck', is_x=False)
    
    avg_body_x = np.nanmean(np.stack([spine1_x, spine2_x, hip_left_x, hip_right_x, neck_x], axis=-1), axis=-1)
    avg_body_y = np.nanmean(np.stack([spine1_y, spine2_y, hip_left_y, hip_right_y, neck_y], axis=-1), axis=-1)
    
    tail_base_x = keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 0]
    tail_base_y = keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 1]
    head_center_x = keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 0]  # Updated now
    head_center_y = keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 1]
    fallback_x = (head_center_x + tail_base_x) / 2
    fallback_y = (head_center_y + tail_base_y) / 2
    fallback_x[np.isnan(fallback_x)] = np.nan  # If either missing
    fallback_y[np.isnan(fallback_y)] = np.nan
    
    final_body_x = np.where(~np.isnan(body_center_x), body_center_x,
                   np.where(~np.isnan(avg_body_x), avg_body_x, fallback_x))
    final_body_y = np.where(~np.isnan(body_center_y), body_center_y,
                   np.where(~np.isnan(avg_body_y), avg_body_y, fallback_y))
    keypoints[:, :, MASTER_SKELETON_MAP['body_center'], 0] = final_body_x
    keypoints[:, :, MASTER_SKELETON_MAP['body_center'], 1] = final_body_y
    
    # Create validity mask before imputation (1: valid/original non-NaN, 0: invalid/NaN)
    mask = (~np.isnan(keypoints[..., 0])).astype(np.uint8)  # [frames, mice, bp] since x/y same
    
    keypoints /= pix_per_cm
    
    # print(f"Processed {video_id}: {num_frames} frames, feature_dim {keypoints.shape[1]}")
    return keypoints.astype(np.float32), num_frames


%%writefile model_1.py

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0]
        x = x_skip.view(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0]
        x = x_skip.view(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d

class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()   

        self.in_channels = in_channels = (32//2) * n_landmarks
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):
        xc = data.permute(0,2,1,3)  # B,C,T,F
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x

class MABeFeatureExtractor(nn.Module):
    def __init__(self, input_dim=708, encoder_dim=144, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Project input features to encoder dimension
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Optional: Add a small CNN for local temporal patterns
        self.use_conv = True
        if self.use_conv:
            self.conv1 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv_norm = nn.LayerNorm(encoder_dim)
            self.conv_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: Input features (batch, seq_len, 708)
            mask: Attention mask (batch, seq_len)
        Returns:
            Encoded features (batch, seq_len, encoder_dim)
        """
        # Project to encoder dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        if self.use_conv:
            # Apply temporal convolutions
            x_conv = x.transpose(1, 2)  # (batch, encoder_dim, seq_len)
            x_conv = F.relu(self.conv1(x_conv))
            x_conv = F.relu(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, encoder_dim)

            # Residual connection
            x = x + self.conv_dropout(self.conv_norm(x_conv))

        # Apply mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        return x

import torch.nn.functional as F
class GraphAggregation(nn.Module):
    def __init__(self, node_dim):
        super(GraphAggregation, self).__init__()
        M = 4  # Fixed num_mice
        self.linear = nn.Linear(node_dim, node_dim)  # Message transformation
        self.adjacency = nn.Parameter(torch.eye(M))  # Learnable adj matrix with self-loops

    def forward(self, nodes):
        """
        nodes: [B*T, M, node_dim] batched per-mouse features.
        Returns aggregated [B*T, M, node_dim].
        """
        messages = self.linear(nodes)  # [B*T, M, node_dim]
        adj_batch = self.adjacency.unsqueeze(0).expand(nodes.size(0), -1, -1)  # [B*T, M, M]
        aggregated = torch.bmm(adj_batch, messages)  # [B*T, M, M] @ [B*T, M, dim] = [B*T, M, dim]
        return aggregated

class BehaviorClassificationHead(nn.Module):

    def __init__(self, encoder_dim=144, num_pairs=16, num_actions=39, dropout=0.0):
        super().__init__()

        self.num_pairs = num_pairs  # Number of mouse pairs (e.g., 4 mice -> 16 directed pairs)
        self.num_actions = num_actions  # Number of behavior classes

        # Shared layers for all pairs
        self.shared_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Single head that outputs all pairs x actions
        # This allows better parameter sharing across pairs
        self.classifier = nn.Linear(encoder_dim, num_pairs * num_actions)

    def forward(self, x, mask=None):
        """
        Args:
            x: Encoder output (batch, seq_len, encoder_dim)
            mask: Attention mask (batch, seq_len)
        Returns:
            Logits (batch, seq_len, num_pairs, num_actions)
        """
        batch_size, seq_len, _ = x.shape

        # Shared transformation
        x = self.shared_proj(x)  # (batch, seq_len, encoder_dim)

        # Single classifier for all pairs
        logits = self.classifier(x)  # (batch, seq_len, num_pairs * num_actions)

        # Reshape to separate pairs and actions
        logits = logits.view(batch_size, seq_len, self.num_pairs, self.num_actions)

        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = GraphAggregation(self.encoder_dim // 4) if getattr(cfg, 'use_gnn', False) else None

        # Feature extractor
        if self.cfg.cnn_extractor:
            self.feature_extractor = FeatureExtractor(
                n_landmarks=cfg.per_mouse_feature_dim // 4,
                out_dim=cfg.encoder_config.encoder_dim)
        else:
            self.feature_extractor = MABeFeatureExtractor(
                input_dim=cfg.feature_dim,
                encoder_dim=self.encoder_dim,
                dropout=cfg.encoder_config.input_dropout_p
            )

        # Squeezeformer encoder
        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        # Classification head
        self.classifier = BehaviorClassificationHead(
            encoder_dim=self.encoder_dim,
            num_pairs=self.num_pairs,
            num_actions=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask']  # (batch, seq_len)
        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            B, T, M, _ = x.shape
            x = self.feature_extractor(x, mask)  # (batch, seq_len, encoder_dim)
        else:
            x = batch['input']  # (batch, seq_len, 708)
            B, T, _ = x.shape
            x = self.feature_extractor(x, mask)  # (batch, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask)  # (batch, seq_len, encoder_dim)

        if self.gnn is not None:
            B, T, D = x.shape
            M = 4  # num_mice
            node_dim = D // M
            # Reshape to per-mouse nodes [B*T, M, node_dim]
            x_resh = x.view(B * T, M, node_dim)
            # Apply batched GNN
            x_gnn = self.gnn(x_resh)  # [B*T, M, node_dim]
            x_gnn = x_gnn.view(B, T, D)  # Back to original shape
            x = x + x_gnn  # Residual add for fusion

        # Classify
        logits = self.classifier(x, mask)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_gnn.py

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d

class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()   

        self.in_channels = in_channels = (32//2) * n_landmarks
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):
        xc = data.permute(0,2,1,3)  # B,C,T,F
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x

class MABeFeatureExtractor(nn.Module):
    def __init__(self, input_dim=708, encoder_dim=144, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Project input features to encoder dimension
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Optional: Add a small CNN for local temporal patterns
        self.use_conv = True
        if self.use_conv:
            self.conv1 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv_norm = nn.LayerNorm(encoder_dim)
            self.conv_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: Input features (batch, seq_len, 708)
            mask: Attention mask (batch, seq_len)
        Returns:
            Encoded features (batch, seq_len, encoder_dim)
        """
        # Project to encoder dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        if self.use_conv:
            # Apply temporal convolutions
            x_conv = x.transpose(1, 2)  # (batch, encoder_dim, seq_len)
            x_conv = F.relu(self.conv1(x_conv))
            x_conv = F.relu(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, encoder_dim)

            # Residual connection
            x = x + self.conv_dropout(self.conv_norm(x_conv))

        # Apply mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        return x

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        # 1. Project Raw Features (Crucial Step)
        # We need to turn coordinates into "semantic" embeddings before Attention
        self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. TransformerConv Layers
        # We use beta=True for a gating mechanism (improves deep GNNs)
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm1 = LayerNorm(out_channels) # LayerNorm is often more stable for Transformers
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        # If you want self-loops (mice look at themselves), add them here, 
        # but TransformerConv usually adds them automatically or handles them separately.

        # Repeat for the whole batch
        # We need to offset indices by num_mice (4) for each subsequent frame
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create the massive edge list
        # shape: [Edges_Per_Frame, Batch_Time]
        src = base_src.unsqueeze(1) + offsets.unsqueeze(0)
        dst = base_dst.unsqueeze(1) + offsets.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x):
        """
        Input: [Batch, Frames, Mice, Features]
        Output: [Batch, Frames, Mice, Out_Channels]
        """
        B, T, M, D = x.shape
        
        # Flatten Batch and Time: treat every frame as an independent graph
        # New shape: [Total_Graphs * Mice, Features]
        x_flat = x.view(B * T * M, D)
        
        # 1. Linear Projection
        x_emb = F.relu(self.embedding(x_flat))
        
        # 2. Create Edges on the fly
        # (Optimized: we only calculate this once per forward pass structure)
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 3. GNN Layer 1
        # TransformerConv expects [Num_Nodes, Dim]
        h = self.conv1(x_emb, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 4. GNN Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back (NO POOLING)
        # We want to keep the mice separate for the Squeezeformer
        return h.view(B, T, M, -1)


class BehaviorClassificationHead(nn.Module):

    def __init__(self, encoder_dim=144, num_pairs=16, num_actions=39, dropout=0.0):
        super().__init__()

        self.num_pairs = num_pairs  # Number of mouse pairs (e.g., 4 mice -> 16 directed pairs)
        self.num_actions = num_actions  # Number of behavior classes

        self.shared_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(encoder_dim, num_pairs * num_actions)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.shared_proj(x)  # (batch, seq_len, encoder_dim)
        logits = self.classifier(x)  # (batch, seq_len, num_pairs * num_actions)
        logits = logits.view(batch_size, seq_len, self.num_pairs, self.num_actions)

        return logits

class PairwiseBehaviorHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        self.in_features = encoder_dim * 2 
        
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output is logits for ONE pair
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - The unflattened Squeezeformer output
        Returns:
            logits: [Batch, Time, Num_Pairs (16), Num_Classes]
        """
        B, T, M, D = x.shape
        
        # We need to form 16 pairs. 
        # Efficient way using broadcasting:
        
        # 1. Expand M1 to represent the "Source" mouse
        # Shape: [B, T, M, 1, D] -> Repeat on dim 3 -> [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        
        # 2. Expand M2 to represent the "Target" mouse
        # Shape: [B, T, 1, M, D] -> Repeat on dim 2 -> [B, T, M, M, D]
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # 3. Concatenate to get pair features
        # Shape: [B, T, M, M, 2*D]
        pair_features = torch.cat([m1, m2], dim=-1)
        
        # 4. Flatten the mouse dimensions to get the list of 16 pairs
        # Shape: [B, T, M*M, 2*D] -> [B, T, 16, 2*D]
        pair_features_flat = pair_features.view(B, T, M*M, -1)
        
        # 5. Classify
        logits = self.layers(pair_features_flat)
        
        return logits

class NetGNN(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(NetGNN, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        # Feature extractor
        if self.cfg.cnn_extractor:
            self.feature_extractor = FeatureExtractor(
                n_landmarks=cfg.per_mouse_feature_dim // 4,
                out_dim=cfg.encoder_config.encoder_dim)
        else:
            self.feature_extractor = MABeFeatureExtractor(
                input_dim=cfg.feature_dim,
                encoder_dim=self.encoder_dim,
                dropout=cfg.encoder_config.input_dropout_p
            )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = BehaviorClassificationHead(
            encoder_dim=self.encoder_dim,
            num_pairs=self.num_pairs,
            num_actions=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        self.classifier = PairwiseBehaviorHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x = self.gnn(x)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)
        else:
            x = batch['input']  # (batch, seq_len, 708)
            B, T, _ = x.shape
            x = self.feature_extractor(x, mask)  # (batch, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_inter.py


import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np
from torch_geometric.nn import TransformerConv, LayerNorm


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        return x

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        # 1. Project Raw Features (Crucial Step)
        # Check if we need to project (if dim changes) or just use as is
        self.proj_input = (in_channels != out_channels)
        if self.proj_input:
            self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. TransformerConv Layers
        # We use beta=True for a gating mechanism (improves deep GNNs)
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm1 = LayerNorm(out_channels) # LayerNorm is often more stable for Transformers
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        # If you want self-loops (mice look at themselves), add them here, 
        # but TransformerConv usually adds them automatically or handles them separately.

        # Repeat for the whole batch
        # We need to offset indices by num_mice (4) for each subsequent frame
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create the massive edge list
        # shape: [Edges_Per_Frame, Batch_Time]
        src = base_src.unsqueeze(1) + offsets.unsqueeze(0)
        dst = base_dst.unsqueeze(1) + offsets.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x):
        """
        Input: [Batch, Frames, Mice, Features]
        Output: [Batch, Frames, Mice, Out_Channels]
        """
        B, T, M, D = x.shape
        
        # Flatten Batch and Time: treat every frame as an independent graph
        # New shape: [Total_Graphs * Mice, Features]
        x_flat = x.reshape(B * T * M, D)
        
        # 1. Linear Projection
        if self.proj_input:
            x_emb = F.relu(self.embedding(x_flat))
        else:
            x_emb = x_flat
        
        # 2. Create Edges on the fly
        # (Optimized: we only calculate this once per forward pass structure)
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 3. GNN Layer 1
        # TransformerConv expects [Num_Nodes, Dim]
        h = self.conv1(x_emb, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 4. GNN Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back (NO POOLING)
        # We want to keep the mice separate for the Squeezeformer
        return h.reshape(B, T, M, -1)

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks with Interleaved GNN Layers
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
        gnn_interval: int = 4, # Inject GNN every N layers
        num_mice: int = 4,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_interval = gnn_interval
        self.num_mice = num_mice
        self.encoder_dim = encoder_dim

        self.blocks = nn.ModuleList()
        self.gnn_layers = nn.ModuleList()

        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )
            
            # Interleaved GNN Injection
            if (idx + 1) % gnn_interval == 0:
                self.gnn_layers.append(
                    SpatialMouseGNN(
                        in_channels=encoder_dim, 
                        out_channels=encoder_dim, 
                        num_mice=num_mice,
                        heads=4, 
                        dropout=0.1
                    )
                )
            else:
                self.gnn_layers.append(None)

    def forward(self, x: Tensor, mask: Tensor):
        # x input: [Batch * Mice, Time, Dim]
        # mask input: [Batch * Mice, Time]
        BM, T, D = x.shape
        M = self.num_mice
        B = BM // M

        for idx, (block, gnn) in enumerate(zip(self.blocks, self.gnn_layers)):
            x = block(x, mask)

            if gnn is not None:
                # 1. Reshape to [Batch, Time, Mice, Dim]
                x_spatial = x.view(B, M, T, D).permute(0, 2, 1, 3)
                
                # 2. Apply GNN with Residual Connection
                gnn_out = gnn(x_spatial)
                x_spatial = x_spatial + gnn_out
                
                # 3. Flatten back to [Batch*Mice, Time, Dim]
                x = x_spatial.permute(0, 2, 1, 3).reshape(BM, T, D)
                
        return x

from timm.layers.norm_act import BatchNormAct2d

class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()   

        self.in_channels = in_channels = (32//2) * n_landmarks
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):
        xc = data.permute(0,2,1,3)  # B,C,T,F
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x

class MABeFeatureExtractor(nn.Module):
    def __init__(self, input_dim=708, encoder_dim=144, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Project input features to encoder dimension
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Optional: Add a small CNN for local temporal patterns
        self.use_conv = True
        if self.use_conv:
            self.conv1 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv_norm = nn.LayerNorm(encoder_dim)
            self.conv_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: Input features (batch, seq_len, 708)
            mask: Attention mask (batch, seq_len)
        Returns:
            Encoded features (batch, seq_len, encoder_dim)
        """
        # Project to encoder dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        if self.use_conv:
            # Apply temporal convolutions
            x_conv = x.transpose(1, 2)  # (batch, encoder_dim, seq_len)
            x_conv = F.relu(self.conv1(x_conv))
            x_conv = F.relu(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, encoder_dim)

            # Residual connection
            x = x + self.conv_dropout(self.conv_norm(x_conv))

        # Apply mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        return x

class BehaviorClassificationHead(nn.Module):

    def __init__(self, encoder_dim=144, num_pairs=16, num_actions=39, dropout=0.0):
        super().__init__()

        self.num_pairs = num_pairs  # Number of mouse pairs (e.g., 4 mice -> 16 directed pairs)
        self.num_actions = num_actions  # Number of behavior classes

        self.shared_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(encoder_dim, num_pairs * num_actions)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.shared_proj(x)  # (batch, seq_len, encoder_dim)
        logits = self.classifier(x)  # (batch, seq_len, num_pairs * num_actions)
        logits = logits.view(batch_size, seq_len, self.num_pairs, self.num_actions)

        return logits

class PairwiseBehaviorHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        self.in_features = encoder_dim * 2 
        
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output is logits for ONE pair
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - The unflattened Squeezeformer output
        Returns:
            logits: [Batch, Time, Num_Pairs (16), Num_Classes]
        """
        B, T, M, D = x.shape
        
        # We need to form 16 pairs. 
        # Efficient way using broadcasting:
        
        # 1. Expand M1 to represent the "Source" mouse
        # Shape: [B, T, M, 1, D] -> Repeat on dim 3 -> [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        
        # 2. Expand M2 to represent the "Target" mouse
        # Shape: [B, T, 1, M, D] -> Repeat on dim 2 -> [B, T, M, M, D]
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # 3. Concatenate to get pair features
        # Shape: [B, T, M, M, 2*D]
        pair_features = torch.cat([m1, m2], dim=-1)
        
        # 4. Flatten the mouse dimensions to get the list of 16 pairs
        # Shape: [B, T, M*M, 2*D] -> [B, T, 16, 2*D]
        pair_features_flat = pair_features.view(B, T, M*M, -1)
        
        # 5. Classify
        logits = self.layers(pair_features_flat)
        
        return logits

class NetInter(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(NetInter, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        # Feature extractor
        if self.cfg.cnn_extractor:
            self.feature_extractor = FeatureExtractor(
                n_landmarks=cfg.per_mouse_feature_dim // 4,
                out_dim=cfg.encoder_config.encoder_dim)
        else:
            self.feature_extractor = MABeFeatureExtractor(
                input_dim=cfg.feature_dim,
                encoder_dim=self.encoder_dim,
                dropout=cfg.encoder_config.input_dropout_p
            )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn,
            gnn_interval = 2, # Interleave GNN every 2 layers
            num_mice = 4
        )

        self.classifier = BehaviorClassificationHead(
            encoder_dim=self.encoder_dim,
            num_pairs=self.num_pairs,
            num_actions=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        self.classifier = PairwiseBehaviorHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x = self.gnn(x)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)
        else:
            x = batch['input']  # (batch, seq_len, 708)
            B, T, _ = x.shape
            x = self.feature_extractor(x, mask)  # (batch, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_238.py


import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.reshape(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.reshape(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        # 1. Project Raw Features (Crucial Step)
        # We need to turn coordinates into "semantic" embeddings before Attention
        self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. TransformerConv Layers
        # We use beta=True for a gating mechanism (improves deep GNNs)
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm1 = LayerNorm(out_channels) # LayerNorm is often more stable for Transformers
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        # If you want self-loops (mice look at themselves), add them here, 
        # but TransformerConv usually adds them automatically or handles them separately.

        # Repeat for the whole batch
        # We need to offset indices by num_mice (4) for each subsequent frame
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create the massive edge list
        # shape: [Edges_Per_Frame, Batch_Time]
        src = base_src.unsqueeze(1) + offsets.unsqueeze(0)
        dst = base_dst.unsqueeze(1) + offsets.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x):
        """
        Input: [Batch, Frames, Mice, Features]
        Output: [Batch, Frames, Mice, Out_Channels]
        """
        B, T, M, D = x.shape
        
        # Flatten Batch and Time: treat every frame as an independent graph
        # New shape: [Total_Graphs * Mice, Features]
        x_flat = x.reshape(B * T * M, D)
        
        # 1. Linear Projection
        x_emb = F.relu(self.embedding(x_flat))
        
        # 2. Create Edges on the fly
        # (Optimized: we only calculate this once per forward pass structure)
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 3. GNN Layer 1
        # TransformerConv expects [Num_Nodes, Dim]
        h = self.conv1(x_emb, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 4. GNN Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back (NO POOLING)
        # We want to keep the mice separate for the Squeezeformer
        return h.reshape(B, T, M, -1)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationAugmentedHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        # --- 1. The Relation Network Module (Global Context) ---
        # g_theta: Processes every pair to find relationships
        # Input: [Mouse_A + Mouse_B] -> 2 * encoder_dim
        self.g_theta = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU()
        )
        
        # f_phi: Processes the SUM of all relations (The "Cage Vibe")
        self.f_phi = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. The Pairwise Classifier ---
        # Input: [Mouse_A + Mouse_B + Global_Context]
        self.in_features = (encoder_dim * 2) + (encoder_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - Output from Squeezeformer/GNN
        """
        B, T, M, D = x.shape
        
        # --- Step A: Form All Pairs (Same as before) ---
        # [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # The specific pairs we want to classify
        # Shape: [B, T, M, M, 2*D]
        specific_pairs = torch.cat([m1, m2], dim=-1)
        
        # --- Step B: The Relation Network (Global "Thinking") ---
        # 1. Apply g_theta to ALL pairs
        # Shape: [B, T, M, M, D/2]
        relations = self.g_theta(specific_pairs)
        
        # 2. Sum over all pairs (M*M) to get ONE vector per frame
        # This represents the "Global Social State" of the cage at time T
        # Shape: [B, T, D/2]
        global_sum = relations.sum(dim=(2, 3)) 
        
        # 3. Apply f_phi (Reasoning over the global state)
        # Shape: [B, T, D/2]
        global_context = self.f_phi(global_sum)
        
        # --- Step C: Combine Local + Global ---
        # We need to broadcast the global context to every specific pair
        # Global: [B, T, 1, 1, D/2]
        global_context_expanded = global_context.unsqueeze(2).unsqueeze(3).expand(-1, -1, M, M, -1)
        
        # Concatenate: [Specific_Pair (2D) | Global_Context (0.5D)]
        # Shape: [B, T, M, M, 2.5D]
        combined_features = torch.cat([specific_pairs, global_context_expanded], dim=-1)
        
        # --- Step D: Final Classification ---
        # Flatten pairs: [B, T, 16, 2.5D]
        combined_flat = combined_features.view(B, T, M*M, -1)
        
        # Predict
        logits = self.classifier(combined_flat)
        
        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = RelationAugmentedHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x = self.gnn(x)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_242.py

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=6, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        
        # 1. Project Raw Node Features
        self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. Project Raw Edge Features (New)
        # Projects your 6 social features to match the GNN's internal dimension
        self.edge_embedding = nn.Linear(edge_dim, edge_dim)
        
        # 3. TransformerConv Layers
        # Added edge_dim argument so it knows what to expect
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True, edge_dim=edge_dim)
        self.norm1 = LayerNorm(out_channels)
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True, edge_dim=edge_dim)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        
        # Offsets for the whole batch
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create [Batch_Time, Edges_Per_Frame]
        # We use (BT, 1) + (1, 12) to get (BT, 12)
        # This ensures edges are ordered: [Graph0_Edges, Graph1_Edges, ...]
        src = offsets.unsqueeze(1) + base_src.unsqueeze(0)
        dst = offsets.unsqueeze(1) + base_dst.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x, x_edges):
        """
        x: [Batch, Frames, Mice, Node_Features]
        x_edges: [Batch, Frames, Mice, Mice, Edge_Features]
        """
        B, T, M, D_n = x.shape
        x_edges = x_edges[:, :, :, :, :6]  # [B, T, M, M, Edge_Features]
        _, _, _, _, D_e = x_edges.shape
        
        # 1. Flatten Nodes: [Total_Mice, Dim]
        x_flat = x.view(B * T * M, D_n)
        x_emb = F.relu(self.embedding(x_flat))
        
        # 2. Flatten Edge Attributes to match Edge Index
        # Reshape to [Batch*Time, M, M, Edge_Dim]
        edge_feats_flat = x_edges.view(B * T, M, M, D_e)
        
        # We need to extract the specific 12 edges (0->1, 0->2...) per graph
        # Base indices for one graph
        base_src_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=x.device)
        base_dst_idx = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=x.device)
        
        # Select attributes for valid edges
        # Shape: [Batch*Time, 12, Edge_Dim]
        valid_edge_attrs = edge_feats_flat[:, base_src_idx, base_dst_idx, :]
        
        # Flatten to [Total_Edges, Edge_Dim]
        # This matches the order of _get_fully_connected_edge_index
        edge_attr = valid_edge_attrs.view(-1, D_e)
        
        # Project Edge Attributes
        edge_attr = F.relu(self.edge_embedding(edge_attr))
        
        # 3. Create Edge Index
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 4. GNN Layers
        h = self.conv1(x_emb, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back
        return h.view(B, T, M, -1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationAugmentedHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        # --- 1. The Relation Network Module (Global Context) ---
        # g_theta: Processes every pair to find relationships
        # Input: [Mouse_A + Mouse_B] -> 2 * encoder_dim
        self.g_theta = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU()
        )
        
        # f_phi: Processes the SUM of all relations (The "Cage Vibe")
        self.f_phi = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. The Pairwise Classifier ---
        # Input: [Mouse_A + Mouse_B + Global_Context]
        self.in_features = (encoder_dim * 2) + (encoder_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - Output from Squeezeformer/GNN
        """
        B, T, M, D = x.shape
        
        # --- Step A: Form All Pairs (Same as before) ---
        # [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # The specific pairs we want to classify
        # Shape: [B, T, M, M, 2*D]
        specific_pairs = torch.cat([m1, m2], dim=-1)
        
        # --- Step B: The Relation Network (Global "Thinking") ---
        # 1. Apply g_theta to ALL pairs
        # Shape: [B, T, M, M, D/2]
        relations = self.g_theta(specific_pairs)
        
        # 2. Sum over all pairs (M*M) to get ONE vector per frame
        # This represents the "Global Social State" of the cage at time T
        # Shape: [B, T, D/2]
        global_sum = relations.sum(dim=(2, 3)) 
        
        # 3. Apply f_phi (Reasoning over the global state)
        # Shape: [B, T, D/2]
        global_context = self.f_phi(global_sum)
        
        # --- Step C: Combine Local + Global ---
        # We need to broadcast the global context to every specific pair
        # Global: [B, T, 1, 1, D/2]
        global_context_expanded = global_context.unsqueeze(2).unsqueeze(3).expand(-1, -1, M, M, -1)
        
        # Concatenate: [Specific_Pair (2D) | Global_Context (0.5D)]
        # Shape: [B, T, M, M, 2.5D]
        combined_features = torch.cat([specific_pairs, global_context_expanded], dim=-1)
        
        # --- Step D: Final Classification ---
        # Flatten pairs: [B, T, 16, 2.5D]
        combined_flat = combined_features.view(B, T, M*M, -1)
        
        # Predict
        logits = self.classifier(combined_flat)
        
        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = RelationAugmentedHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x_nodes = batch['node_feats']  # Shape [B, T, 4, Node_Dim]
            x_edges = batch['edge_feats']  # Shape [B, T, 4, 4, Edge_Dim]
            x = self.gnn(x_nodes, x_edges)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_243.py


import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=8, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        
        self.hidden_edge_dim = 32  # Give the edges some bandwidth!
        # 1. Project Raw Node Features
        self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. Project Raw Edge Features (New)
        # Projects your 6 social features to match the GNN's internal dimension
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_edge_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_edge_dim, self.hidden_edge_dim) # Extra depth
        )
        
        # 3. TransformerConv Layers
        # Added edge_dim argument so it knows what to expect
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True, edge_dim=self.hidden_edge_dim)
        self.norm1 = LayerNorm(out_channels)
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True, edge_dim=self.hidden_edge_dim)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        
        # Offsets for the whole batch
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create [Batch_Time, Edges_Per_Frame]
        # We use (BT, 1) + (1, 12) to get (BT, 12)
        # This ensures edges are ordered: [Graph0_Edges, Graph1_Edges, ...]
        src = offsets.unsqueeze(1) + base_src.unsqueeze(0)
        dst = offsets.unsqueeze(1) + base_dst.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x, x_edges):
        """
        x: [Batch, Frames, Mice, Node_Features]
        x_edges: [Batch, Frames, Mice, Mice, Edge_Features]
        """
        B, T, M, D_n = x.shape
        _, _, _, _, D_e = x_edges.shape
        
        # 1. Flatten Nodes: [Total_Mice, Dim]
        x_flat = x.view(B * T * M, D_n)
        x_emb = F.relu(self.embedding(x_flat))
        
        # 2. Flatten Edge Attributes to match Edge Index
        # Reshape to [Batch*Time, M, M, Edge_Dim]
        edge_feats_flat = x_edges.view(B * T, M, M, D_e)
        
        # We need to extract the specific 12 edges (0->1, 0->2...) per graph
        # Base indices for one graph
        base_src_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=x.device)
        base_dst_idx = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=x.device)
        
        # Select attributes for valid edges
        # Shape: [Batch*Time, 12, Edge_Dim]
        valid_edge_attrs = edge_feats_flat[:, base_src_idx, base_dst_idx, :]
        
        # Flatten to [Total_Edges, Edge_Dim]
        # This matches the order of _get_fully_connected_edge_index
        edge_attr = valid_edge_attrs.view(-1, D_e)
        
        # Project Edge Attributes
        edge_attr = self.edge_embedding(edge_attr)

        # 3. Create Edge Index
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 4. GNN Layers
        h = self.conv1(x_emb, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back
        return h.view(B, T, M, -1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationAugmentedHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        # --- 1. The Relation Network Module (Global Context) ---
        # g_theta: Processes every pair to find relationships
        # Input: [Mouse_A + Mouse_B] -> 2 * encoder_dim
        self.g_theta = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU()
        )
        
        # f_phi: Processes the SUM of all relations (The "Cage Vibe")
        self.f_phi = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. The Pairwise Classifier ---
        # Input: [Mouse_A + Mouse_B + Global_Context]
        self.in_features = (encoder_dim * 2) + (encoder_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - Output from Squeezeformer/GNN
        """
        B, T, M, D = x.shape
        
        # --- Step A: Form All Pairs (Same as before) ---
        # [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # The specific pairs we want to classify
        # Shape: [B, T, M, M, 2*D]
        specific_pairs = torch.cat([m1, m2], dim=-1)
        
        # --- Step B: The Relation Network (Global "Thinking") ---
        # 1. Apply g_theta to ALL pairs
        # Shape: [B, T, M, M, D/2]
        relations = self.g_theta(specific_pairs)
        
        # 2. Sum over all pairs (M*M) to get ONE vector per frame
        # This represents the "Global Social State" of the cage at time T
        # Shape: [B, T, D/2]
        global_sum = relations.sum(dim=(2, 3)) 
        
        # 3. Apply f_phi (Reasoning over the global state)
        # Shape: [B, T, D/2]
        global_context = self.f_phi(global_sum)
        
        # --- Step C: Combine Local + Global ---
        # We need to broadcast the global context to every specific pair
        # Global: [B, T, 1, 1, D/2]
        global_context_expanded = global_context.unsqueeze(2).unsqueeze(3).expand(-1, -1, M, M, -1)
        
        # Concatenate: [Specific_Pair (2D) | Global_Context (0.5D)]
        # Shape: [B, T, M, M, 2.5D]
        combined_features = torch.cat([specific_pairs, global_context_expanded], dim=-1)
        
        # --- Step D: Final Classification ---
        # Flatten pairs: [B, T, 16, 2.5D]
        combined_flat = combined_features.view(B, T, M*M, -1)
        
        # Predict
        logits = self.classifier(combined_flat)
        
        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = RelationAugmentedHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x_nodes = batch['node_feats']  # Shape [B, T, 4, Node_Dim]
            x_edges = batch['edge_feats']  # Shape [B, T, 4, 4, Edge_Dim]
            x = self.gnn(x_nodes, x_edges)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


%%writefile model_244.py


import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        # 1. Project Raw Features (Crucial Step)
        # We need to turn coordinates into "semantic" embeddings before Attention
        self.embedding = nn.Linear(in_channels, out_channels)
        
        # 2. TransformerConv Layers
        # We use beta=True for a gating mechanism (improves deep GNNs)
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm1 = LayerNorm(out_channels) # LayerNorm is often more stable for Transformers
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """
        # Base edges for one frame of 4 mice (0-3)
        # 0->1, 0->2, 0->3, 1->0, ...
        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)
        # If you want self-loops (mice look at themselves), add them here, 
        # but TransformerConv usually adds them automatically or handles them separately.

        # Repeat for the whole batch
        # We need to offset indices by num_mice (4) for each subsequent frame
        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        # Broadcasting to create the massive edge list
        # shape: [Edges_Per_Frame, Batch_Time]
        src = base_src.unsqueeze(1) + offsets.unsqueeze(0)
        dst = base_dst.unsqueeze(1) + offsets.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x):
        """
        Input: [Batch, Frames, Mice, Features]
        Output: [Batch, Frames, Mice, Out_Channels]
        """
        B, T, M, D = x.shape
        
        # Flatten Batch and Time: treat every frame as an independent graph
        # New shape: [Total_Graphs * Mice, Features]
        x_flat = x.view(B * T * M, D)
        
        # 1. Linear Projection
        x_emb = F.relu(self.embedding(x_flat))
        
        # 2. Create Edges on the fly
        # (Optimized: we only calculate this once per forward pass structure)
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        # 3. GNN Layer 1
        # TransformerConv expects [Num_Nodes, Dim]
        h = self.conv1(x_emb, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 4. GNN Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        
        # 5. Reshape back (NO POOLING)
        # We want to keep the mice separate for the Squeezeformer
        return h.view(B, T, M, -1)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationAugmentedHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        # --- 1. The Relation Network Module (Global Context) ---
        # g_theta: Processes every pair to find relationships
        # Input: [Mouse_A + Mouse_B] -> 2 * encoder_dim
        self.g_theta = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU()
        )
        
        # f_phi: Processes the SUM of all relations (The "Cage Vibe")
        self.f_phi = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. The Pairwise Classifier ---
        # Input: [Mouse_A + Mouse_B + Global_Context]
        self.in_features = (encoder_dim * 2) + (encoder_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - Output from Squeezeformer/GNN
        """
        B, T, M, D = x.shape
        
        # --- Step A: Form All Pairs (Same as before) ---
        # [B, T, M, M, D]
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # The specific pairs we want to classify
        # Shape: [B, T, M, M, 2*D]
        specific_pairs = torch.cat([m1, m2], dim=-1)
        
        # --- Step B: The Relation Network (Global "Thinking") ---
        # 1. Apply g_theta to ALL pairs
        # Shape: [B, T, M, M, D/2]
        relations = self.g_theta(specific_pairs)
        
        # 2. Sum over all pairs (M*M) to get ONE vector per frame
        # This represents the "Global Social State" of the cage at time T
        # Shape: [B, T, D/2]
        global_sum = relations.sum(dim=(2, 3)) 
        
        # 3. Apply f_phi (Reasoning over the global state)
        # Shape: [B, T, D/2]
        global_context = self.f_phi(global_sum)
        
        # --- Step C: Combine Local + Global ---
        # We need to broadcast the global context to every specific pair
        # Global: [B, T, 1, 1, D/2]
        global_context_expanded = global_context.unsqueeze(2).unsqueeze(3).expand(-1, -1, M, M, -1)
        
        # Concatenate: [Specific_Pair (2D) | Global_Context (0.5D)]
        # Shape: [B, T, M, M, 2.5D]
        combined_features = torch.cat([specific_pairs, global_context_expanded], dim=-1)
        
        # --- Step D: Final Classification ---
        # Flatten pairs: [B, T, 16, 2.5D]
        combined_flat = combined_features.view(B, T, M*M, -1)
        
        # Predict
        logits = self.classifier(combined_flat)
        
        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = RelationAugmentedHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                # Fallback if cfg is wrong, assume last class is no_action
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        # Optional: Multi-class focal loss
        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_ego_mice']
            x = self.gnn(x)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        # APPLY BEHAVIOR MASK if provided (for evaluation)
        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            # Expand to match logits shape
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            # For multi-class, we need to ensure at least one action is valid per pair
            # Set masked positions to very negative value
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        # Calculate loss if labels provided
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            # Convert one-hot to class indices if needed
            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                # Convert to class indices
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            # Calculate loss
            if self.use_focal_loss:
                # Multi-class focal loss
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                # Standard weighted cross-entropy
                loss = self.loss_fn(logits_flat, labels_flat)

            # Reshape loss back
            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)  # Multi-class probabilities
        output['predictions'] = probs
        output['logits'] = logits

        # Store additional info for evaluation
        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output


import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import random
from tqdm import tqdm
import ast
import warnings
import gc

warnings.filterwarnings('ignore', message='Mean of empty slice')


def batch_to_device(batch, device):
    batch_dict = {}
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch_dict[key] = batch[key].to(device)
        else:
            batch_dict[key] = batch[key]
    return batch_dict


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug

        self.base_data_dir = '/kaggle/input/MABe-mouse-behavior-detection'
        valid_videos = []
        self.video_info = {}
        self.video_keypoints = {}
        
        num_mice = len(cfg.set_mice)
        num_pairs = num_mice * num_mice
        num_actions = len(cfg.set_behavior_classes)
        mouse_to_idx = {'mouse1': 0, 'mouse2': 1, 'mouse3': 2, 'mouse4': 3}

        for idx in tqdm(range(len(self.df)), desc="Scanning videos and pre-loading labels"):
            row = self.df.iloc[idx]
            lab_id = row['lab_id']
            video_id = row['video_id']

            try:
                keypoints, num_frames = process_tracking_video(row, 'test')
                if keypoints is None or num_frames == 0:
                    continue

                if video_id not in self.video_keypoints.keys():
                    self.video_keypoints[video_id] = keypoints
                else:
                    self.video_keypoints[video_id+len(self.video_keypoints)] = keypoints
                    print(len(self.video_keypoints), self.video_keypoints.keys())
                
                behaviors_labeled = ast.literal_eval(row['behaviors_labeled']) if isinstance(row['behaviors_labeled'], str) else row['behaviors_labeled']
                if not isinstance(behaviors_labeled, list):
                    continue
                behavior_mask = np.zeros((num_pairs, num_actions), dtype=bool)
                for behavior_str in behaviors_labeled:
                    parts = behavior_str.split(',')
                    if len(parts) != 3:
                        continue
                    agent, target, action = parts
                    agent_idx = mouse_to_idx.get(agent, -1)
                    if target == 'self':
                        target_idx = agent_idx
                    else:
                        target_idx = mouse_to_idx.get(target, -1)
                    if agent_idx == -1 or target_idx == -1:
                        continue
                    pair_idx = agent_idx * num_mice + target_idx
                    action_idx = cfg.action_id_map.get(action, -1)
                    if action_idx == -1:
                        continue
                    behavior_mask[pair_idx, action_idx] = True
                
                no_action_idx = cfg.action_id_map.get('no_action', -1)
                if no_action_idx != -1:
                    behavior_mask[:, no_action_idx] = True

                self.video_info[video_id] = {
                    'num_frames': num_frames,
                    'duration_sec': row.get('video_duration_sec', num_frames / 30),
                    'lab_id': row['lab_id'],
                    'behavior_mask': behavior_mask
                }
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue

        self.create_sampling_schedule()

    def create_sampling_schedule(self):
        self.sampling_schedule = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            video_id = row['video_id']
            lab_id = row['lab_id']
            if video_id not in self.video_info:
                continue

            num_frames = self.video_info[video_id]['num_frames']
            if num_frames <= self.cfg.window_size:
                num_windows = 1
            else:
                stride = self.cfg.stride if hasattr(self.cfg, 'stride') else self.cfg.window_size // 2
                total_possible_windows = (num_frames // stride ) + 1

                if self.mode == 'train':
                    num_windows = max(
                        self.cfg.min_windows_per_video,
                        min(
                            self.cfg.max_windows_per_video,
                            int(total_possible_windows * self.cfg.windows_per_epoch_ratio)
                        )
                    )
                else:
                    num_windows = total_possible_windows

            # Add entries to schedule
            for win_idx in range(num_windows):
                self.sampling_schedule.append({
                    'video_id': video_id,
                    'df_idx': idx,
                    'win_idx': win_idx,
                    'num_frames': num_frames
                })

        print(f"Created sampling schedule with {len(self.sampling_schedule)} windows")
        if len(self.df) > 0:
            print(f"Average windows per video: {len(self.sampling_schedule) / len(self.df):.2f}")

    def __len__(self):
        return len(self.sampling_schedule)

    def __getitem__(self, idx):
        sample_info = self.sampling_schedule[idx]
        video_id = sample_info['video_id']
        num_frames = sample_info['num_frames']

        row = self.df.iloc[sample_info['df_idx']]
        # Get video info
        video_data = self.video_info[video_id]
        
        if self.mode == 'train':
            positive_frames = video_data.get('behavior_frames', np.array([], dtype=np.int64))
            
            if random.random() < self.cfg.bias_prob and len(positive_frames) > 0:
                center_frame = random.choice(positive_frames)
                offset = random.randint(0, self.cfg.window_size - 1)
                start = max(0, min(center_frame - offset, num_frames - self.cfg.window_size))
            else:
                start = random.randint(0, max(0, num_frames - self.cfg.window_size))
        else:  # val mode: deterministic sliding window
            stride = self.cfg.stride if hasattr(self.cfg, 'stride') else self.cfg.window_size // 2
            start = sample_info['win_idx'] * stride
            if num_frames <= self.cfg.window_size:
                start = 0

        if start + self.cfg.window_size > num_frames:
            start = num_frames - self.cfg.window_size

        if self.cfg.window_size >= num_frames:
            start = 0
    
        end = min(start + self.cfg.window_size, num_frames)
        features = self.video_keypoints[video_id][start:end]  # Direct slice
        features = torch.from_numpy(features)
        labels = None

        actual_len = features.shape[0]
        if actual_len < self.cfg.window_size:
            pad_len = self.cfg.window_size - actual_len
            features = torch.cat([features, torch.full((pad_len, *features.shape[1:]), float('nan'))], dim=0)
            if labels is not None:
                labels = torch.cat([labels, torch.zeros(pad_len, *labels.shape[1:])], dim=0)
                no_action_idx = self.cfg.action_id_map.get('no_action', 38)
                labels[actual_len:, :, no_action_idx] = 1.0 

        # Create mask
        mask = torch.ones(self.cfg.window_size)
        if actual_len < self.cfg.window_size:
            mask[actual_len:] = 0

        # Apply time reversal if configured
        if self.cfg.reverse_time:
            features = torch.flip(features, dims=[0])
            if labels is not None:
                labels = torch.flip(labels, dims=[0])
            mask = torch.flip(mask, dims=[0])

        ego_features_copy = features.clone()
        features, per_mouse_features, node_feats, edge_feats = self.compute_feature(features)
        ego_features, ego_per_mouse_features = self.compute_feature_ego(ego_features_copy)

        item = {
            'input': features,
            'input_ego': ego_features,
            'input_mice': per_mouse_features,
            'input_ego_mice': ego_per_mouse_features,
            'input_mask': mask,
            'node_feats': node_feats,
            'edge_feats': edge_feats,
            'labels': labels,
            'video_id': video_id,
            'start_frame': torch.tensor(start),
            'num_frames': torch.tensor(num_frames)  # Total video frames
        }

        # Add behavior_mask (per video)
        if 'behavior_mask' in video_data:
            item['behavior_mask'] = torch.from_numpy(video_data['behavior_mask']).float()

        return item

    def compute_egocentric_transform(self, keypoints, bp_to_idx):
        T, M, B, C = keypoints.shape
        device = keypoints.device
        
        centers = keypoints[:, :, bp_to_idx['body_center']].clone() # [T, M, 2]
        
        heads = keypoints[:, :, bp_to_idx['head_center']]
        tails = keypoints[:, :, bp_to_idx['tail_base']]
        
        spines = heads - tails # [T, M, 2]

        spines = torch.nan_to_num(spines, 0.0)
        
        spine_norms = torch.norm(spines, dim=-1, keepdim=True)
        tiny_mask = spine_norms < 1e-6
        spines = torch.where(tiny_mask, torch.tensor([1.0, 0.0], device=device), spines)
        spine_norms = torch.where(tiny_mask, torch.tensor(1.0, device=device), spine_norms)
        
        cos_theta = spines[..., 0:1] / spine_norms
        sin_theta = spines[..., 1:2] / spine_norms
        

        row1 = torch.cat([cos_theta, sin_theta], dim=-1)
        row2 = torch.cat([-sin_theta, cos_theta], dim=-1)
        
        rot_matrices = torch.stack([row1, row2], dim=-2)
        
        return centers, rot_matrices

    def compute_feature_ego(self, keypoints):
        T, M, B, C = keypoints.shape
        device = keypoints.device
        
        # Configuration
        FPS = 30
        MAX_VELOCITY = 100.0
        MAX_ACCEL = 500.0
        MAX_JERK = 1000.0
        EPSILON = 1e-6
        
        bp_to_idx = self.cfg.MASTER_SKELETON_MAP
        
        def safe_divide(a, b, eps=EPSILON):
            return a / (b + eps)
        
        # Helper function for angle computation
        def compute_angle_3points(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            norm_v1 = torch.norm(v1, dim=-1, keepdim=True)
            norm_v2 = torch.norm(v2, dim=-1, keepdim=True)
            cos_angle = torch.sum(v1 * v2, dim=-1) / (norm_v1.squeeze(-1) * norm_v2.squeeze(-1) + EPSILON)
            return torch.acos(torch.clamp(cos_angle, -1.0 + EPSILON, 1.0 - EPSILON))
        
        all_features = []
        
        # ============== KINEMATICS (GLOBAL) ==============
        velocities = torch.diff(keypoints, dim=0, prepend=keypoints[0:1]) * FPS
        velocities = torch.clamp(velocities, -MAX_VELOCITY, MAX_VELOCITY)
        accelerations = torch.diff(velocities, dim=0, prepend=velocities[0:1])
        accelerations = torch.clamp(accelerations, -MAX_ACCEL, MAX_ACCEL)
        jerks = torch.diff(accelerations, dim=0, prepend=accelerations[0:1])
        jerks = torch.clamp(jerks, -MAX_JERK, MAX_JERK)
        
        speeds_normalized = torch.norm(velocities, dim=-1) / MAX_VELOCITY
        accel_normalized = torch.norm(accelerations, dim=-1) / MAX_ACCEL
        jerk_normalized = torch.norm(jerks, dim=-1) / MAX_JERK
        
        all_features.append(speeds_normalized.reshape(T, M * B))
        all_features.append(accel_normalized.reshape(T, M * B))
        all_features.append(jerk_normalized.reshape(T, M * B))
        
        # Angular velocities
        angular_velocities = torch.zeros(T, M, B).to(device)
        for m in range(M):
            for b in range(B):
                vx, vy = velocities[:, m, b, 0], velocities[:, m, b, 1]
                angles = torch.atan2(vy, vx)
                angular_velocities[1:, m, b] = torch.diff(angles)
                angular_velocities[:, m, b] = (angular_velocities[:, m, b] + torch.pi) % (2 * torch.pi) - torch.pi
        
        angular_velocities_normalized = angular_velocities / torch.pi
        all_features.append(angular_velocities_normalized.reshape(T, M * B))
        
        # ============== CENTROID & SHAPE ==============
        centroid_parts = ['head_center', 'body_center', 'tail_base']
        centroid_indices = torch.tensor([bp_to_idx[bp] for bp in centroid_parts]).to(device)
        mouse_centroids = torch.mean(keypoints[:, :, centroid_indices, :], dim=2)
        
        centroid_velocity = torch.diff(mouse_centroids, dim=0, prepend=mouse_centroids[0:1]) * FPS
        centroid_velocity = torch.clamp(centroid_velocity, -MAX_VELOCITY, MAX_VELOCITY)
        centroid_speed_normalized = torch.norm(centroid_velocity, dim=-1) / MAX_VELOCITY
        
        centroid_accel = torch.diff(centroid_velocity, dim=0, prepend=centroid_velocity[0:1])
        centroid_accel_normalized = torch.norm(torch.clamp(centroid_accel, -MAX_ACCEL, MAX_ACCEL), dim=-1) / MAX_ACCEL
        
        all_features.extend([centroid_speed_normalized, centroid_accel_normalized])
        
        movement_heading = torch.atan2(centroid_velocity[:, :, 1], centroid_velocity[:, :, 0])
        movement_heading_normalized = movement_heading / torch.pi
        all_features.append(movement_heading_normalized)
        
        # Shape / Body Configuration
        nose_pos = keypoints[:, :, bp_to_idx['nose'], :]
        tail_base_pos = keypoints[:, :, bp_to_idx['tail_base'], :]
        head_center_pos = keypoints[:, :, bp_to_idx['head_center'], :]
        body_center_pos = keypoints[:, :, bp_to_idx['body_center'], :]
        ear_left_pos = keypoints[:, :, bp_to_idx['ear_left'], :]
        ear_right_pos = keypoints[:, :, bp_to_idx['ear_right'], :]
        
        body_length = torch.norm(nose_pos - tail_base_pos, dim=-1)
        body_length_95 = torch.quantile(body_length[~torch.isnan(body_length)], 0.95) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0
        body_length_normalized = torch.clamp(body_length / (body_length_95 + EPSILON), 0, 2)
        all_features.append(body_length_normalized)
        
        body_length_change = torch.diff(body_length, dim=0, prepend=body_length[0:1])
        body_length_change_normalized = torch.clamp(body_length_change / (body_length_95 * 0.1 + EPSILON), -2, 2)
        all_features.append(body_length_change_normalized)
        
        ear_spread = torch.norm(ear_left_pos - ear_right_pos, dim=-1)
        ear_spread_95 = torch.quantile(ear_spread[~torch.isnan(ear_spread)], 0.95) if ear_spread[~torch.isnan(ear_spread)].numel() > 0 else 1.0
        ear_spread_normalized = torch.clamp(ear_spread / (ear_spread_95 + EPSILON), 0, 2)
        all_features.append(ear_spread_normalized)
        
        body_elongation = safe_divide(body_length, ear_spread)
        body_elongation_normalized = torch.clamp(body_elongation, 0, 10) / 10
        all_features.append(body_elongation_normalized)
        
        # Angles
        body_curvature = compute_angle_3points(nose_pos, body_center_pos, tail_base_pos)
        body_curvature_normalized = body_curvature / torch.pi
        all_features.append(body_curvature_normalized)
        
        head_angle = compute_angle_3points(nose_pos, head_center_pos, body_center_pos)
        head_angle_normalized = head_angle / torch.pi
        all_features.append(head_angle_normalized)
        
        tail_angle = compute_angle_3points(head_center_pos, body_center_pos, tail_base_pos)
        tail_angle_normalized = tail_angle / torch.pi
        all_features.append(tail_angle_normalized)
        
        body_vector = nose_pos - tail_base_pos
        body_orientation = torch.atan2(body_vector[:, :, 1], body_vector[:, :, 0])
        body_orientation_normalized = body_orientation / torch.pi
        all_features.append(body_orientation_normalized)
        
        unwrapped_orientation = torch.from_numpy(np.unwrap(body_orientation.cpu().numpy(), axis=0)).to(device)
        body_angular_velocity = torch.diff(unwrapped_orientation, dim=0, prepend=unwrapped_orientation[0:1])
        body_angular_velocity = torch.clamp(body_angular_velocity, -torch.pi, torch.pi) / torch.pi
        all_features.append(body_angular_velocity)
        
        nose_centroid_dist = torch.norm(nose_pos - mouse_centroids, dim=-1)
        nose_centroid_normalized = torch.clamp(safe_divide(nose_centroid_dist, body_length), 0, 2)
        tail_centroid_dist = torch.norm(tail_base_pos - mouse_centroids, dim=-1)
        tail_centroid_normalized = torch.clamp(safe_divide(tail_centroid_dist, body_length), 0, 2)
        all_features.extend([nose_centroid_normalized, tail_centroid_normalized])
        
        nose_velocity = velocities[:, :, bp_to_idx['nose'], :]
        tail_velocity = velocities[:, :, bp_to_idx['tail_base'], :]
        nose_rel_speed = torch.norm(nose_velocity - centroid_velocity, dim=-1) / MAX_VELOCITY
        tail_rel_speed = torch.norm(tail_velocity - centroid_velocity, dim=-1) / MAX_VELOCITY
        all_features.extend([nose_rel_speed, tail_rel_speed])
        
        # ============== EGOCENTRIC ALIGNMENT (THE FIX) ==============
        ego_centers, ego_rot_mats = self.compute_egocentric_transform(keypoints, bp_to_idx)
        
        typical_mouse_size = torch.median(body_length[~torch.isnan(body_length)]) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0

        # ============== INTER-MOUSE FEATURES ==============
        inter_features_list = []
        
        # Per-mouse lists
        mouse_features = [[] for _ in range(M)]
        
        # 1. Add Intra-mouse features
        bp_specific = [speeds_normalized, accel_normalized, jerk_normalized, angular_velocities_normalized]
        for feat in bp_specific:
            for m in range(M):
                mouse_features[m].append(feat[:, m, :].reshape(T, B))
        
        mouse_specific = [
            centroid_speed_normalized, centroid_accel_normalized, movement_heading_normalized,
            body_length_normalized, body_length_change_normalized, ear_spread_normalized, body_elongation_normalized,
            body_curvature_normalized, head_angle_normalized, tail_angle_normalized, body_orientation_normalized, body_angular_velocity,
            nose_centroid_normalized, tail_centroid_normalized, nose_rel_speed, tail_rel_speed
        ]
        for feat in mouse_specific:
            for m in range(M):
                mouse_features[m].append(feat[:, m].unsqueeze(-1))
        
        # 2. Add Egocentric Interaction Features
        for i in range(M): # Agent
            R_i = ego_rot_mats[:, i] # [T, 2, 2]
            Center_i = ego_centers[:, i] # [T, 2]
            
            for j in range(M): # Target
                if i == j:
                    continue
                
                # --- Basic Distances (Existing) ---
                centroid_i = mouse_centroids[:, i, :]
                centroid_j = mouse_centroids[:, j, :]
                dist_centroids = torch.norm(centroid_j - centroid_i, dim=-1)
                dist_centroids_normalized = torch.clamp(dist_centroids / (typical_mouse_size * 5 + EPSILON), 0, 2)
                mouse_features[i].append(dist_centroids_normalized.unsqueeze(-1))
                inter_features_list.append(dist_centroids_normalized.unsqueeze(-1))
                
                dist_change = torch.diff(dist_centroids, dim=0, prepend=dist_centroids[0:1])
                dist_change_normalized = torch.clamp(dist_change / (typical_mouse_size + EPSILON), -2, 2) / 2
                mouse_features[i].append(dist_change_normalized.unsqueeze(-1))
                inter_features_list.append(dist_change_normalized.unsqueeze(-1))
                
                nose_i = nose_pos[:, i, :]
                nose_j = nose_pos[:, j, :]
                tail_j = tail_base_pos[:, j, :]
                
                nose_nose_dist = torch.norm(nose_i - nose_j, dim=-1)
                nose_nose_normalized = torch.clamp(nose_nose_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                mouse_features[i].append(nose_nose_normalized.unsqueeze(-1))
                inter_features_list.append(nose_nose_normalized.unsqueeze(-1))
                
                nose_tail_dist = torch.norm(nose_i - tail_j, dim=-1)
                nose_tail_normalized = torch.clamp(nose_tail_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                mouse_features[i].append(nose_tail_normalized.unsqueeze(-1))
                inter_features_list.append(nose_tail_normalized.unsqueeze(-1))
                
                rel_speed = (centroid_speed_normalized[:, i] - centroid_speed_normalized[:, j])
                mouse_features[i].append(rel_speed.unsqueeze(-1))
                inter_features_list.append(rel_speed.unsqueeze(-1))
                
                vec_to_j = centroid_j - centroid_i
                heading_vec_i = body_vector[:, i, :]
                dot_prod = torch.sum(heading_vec_i * vec_to_j, dim=-1)
                norm_prod = torch.norm(heading_vec_i, dim=-1) * torch.norm(vec_to_j, dim=-1)
                approach_angle = torch.acos(torch.clamp(safe_divide(dot_prod, norm_prod), -1.0 + EPSILON, 1.0 - EPSILON))
                approach_angle_normalized = approach_angle / torch.pi
                mouse_features[i].append(approach_angle_normalized.unsqueeze(-1))
                inter_features_list.append(approach_angle_normalized.unsqueeze(-1))
                
                # --- NEW EGOCENTRIC FEATURES (High SOTA Impact) ---
                target_pts = {
                    'nose': nose_pos[:, j, :],
                    'body': mouse_centroids[:, j, :],
                    'tail': tail_base_pos[:, j, :]
                }
                
                for pt_name, pt_tensor in target_pts.items():
                    rel_vec = pt_tensor - Center_i # [T, 2]
                    
                    rel_vec_ego = torch.matmul(R_i, rel_vec.unsqueeze(-1)).squeeze(-1) # [T, 2]
                    
                    norm_scale = typical_mouse_size * 5.0 + EPSILON
                    
                    ego_x = torch.clamp(rel_vec_ego[:, 0] / norm_scale, -2.0, 2.0)
                    ego_y = torch.clamp(rel_vec_ego[:, 1] / norm_scale, -2.0, 2.0)
                    
                    # Add to features
                    mouse_features[i].append(ego_x.unsqueeze(-1))
                    mouse_features[i].append(ego_y.unsqueeze(-1))
                    
                    # Also add to flattened inter_features_list for global stream
                    inter_features_list.append(ego_x.unsqueeze(-1))
                    inter_features_list.append(ego_y.unsqueeze(-1))

        if inter_features_list:
            inter_features = torch.cat(inter_features_list, dim=-1)
            all_features.append(inter_features)
        
        per_mouse_feats = [torch.cat(feats, dim=-1) for feats in mouse_features]
        
        features = torch.cat(all_features, dim=-1)
        per_mouse_feats = torch.stack(per_mouse_feats, dim=1)

        features = torch.clamp(features, -10, 10)
        per_mouse_feats = torch.clamp(per_mouse_feats, -10, 10)
        
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        per_mouse_feats = torch.nan_to_num(per_mouse_feats, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return features, per_mouse_feats

    def compute_feature(self, keypoints):
            T, M, B, C = keypoints.shape
            device = keypoints.device
            
            # Configuration
            FPS = 30  
            WINDOW_SIZES = [3, 5, 10, 20]  # For rolling statistics
            
            # Clipping thresholds to prevent explosions
            MAX_VELOCITY = 100.0  # cm/s
            MAX_ACCEL = 500.0     # cm/s^2
            MAX_JERK = 1000.0     # cm/s^3
            EPSILON = 1e-6
            
            # Helper function for safe division
            def safe_divide(a, b, eps=EPSILON):
                return a / (b + eps)
            
            # Helper function for angle computation
            def compute_angle_3points(p1, p2, p3):
                """Compute angle at p2 formed by p1-p2-p3"""
                v1 = p1 - p2
                v2 = p3 - p2
                norm_v1 = torch.norm(v1, dim=-1, keepdim=True)
                norm_v2 = torch.norm(v2, dim=-1, keepdim=True)
                cos_angle = torch.sum(v1 * v2, dim=-1) / (norm_v1.squeeze(-1) * norm_v2.squeeze(-1) + EPSILON)
                return torch.acos(torch.clamp(cos_angle, -1.0 + EPSILON, 1.0 - EPSILON))
            
            # Helper for rolling statistics
            def rolling_window_torch(tensor, window_size, operation='mean'):
                """Apply rolling window operation"""
                result = torch.zeros_like(tensor)
                pad = window_size // 2
                for i in range(tensor.shape[0]):
                    start = max(0, i - pad)
                    end = min(tensor.shape[0], i + pad + 1)
                    if operation == 'mean':
                        result[i] = torch.mean(tensor[start:end], dim=0)
                    elif operation == 'std':
                        if end - start > 1:
                            result[i] = torch.std(tensor[start:end], dim=0)
                        else:
                            result[i] = 0
                return result
            
            all_features = []
            bp_to_idx = self.cfg.MASTER_SKELETON_MAP
            
            # ============== KINEMATICS (NORMALIZED) ==============
            
            # Velocities with clipping
            velocities = torch.diff(keypoints, dim=0, prepend=keypoints[0:1]) * FPS
            velocities = torch.clamp(velocities, -MAX_VELOCITY, MAX_VELOCITY)
            
            # Accelerations with clipping
            accelerations = torch.diff(velocities, dim=0, prepend=velocities[0:1])
            accelerations = torch.clamp(accelerations, -MAX_ACCEL, MAX_ACCEL)
            
            # Jerks with clipping
            jerks = torch.diff(accelerations, dim=0, prepend=accelerations[0:1])
            jerks = torch.clamp(jerks, -MAX_JERK, MAX_JERK)
            
            # Speed and acceleration magnitudes (normalized by max values)
            speeds = torch.norm(velocities, dim=-1)  # [T, M, B]
            speeds_normalized = speeds / MAX_VELOCITY  # Normalize to [0, 1]
            
            accel_mags = torch.norm(accelerations, dim=-1)  # [T, M, B]
            accel_normalized = accel_mags / MAX_ACCEL
            
            jerk_mags = torch.norm(jerks, dim=-1)  # [T, M, B]
            jerk_normalized = jerk_mags / MAX_JERK
            
            all_features.append(speeds_normalized.reshape(T, M * B))
            all_features.append(accel_normalized.reshape(T, M * B))
            all_features.append(jerk_normalized.reshape(T, M * B))
            
            # Angular velocities (already bounded [-pi, pi] naturally)
            angular_velocities = torch.zeros(T, M, B).to(device)
            for m in range(M):
                for b in range(B):
                    vx, vy = velocities[:, m, b, 0], velocities[:, m, b, 1]
                    angles = torch.atan2(vy, vx)
                    angular_velocities[1:, m, b] = torch.diff(angles)
                    # Wrap to [-pi, pi]
                    angular_velocities[:, m, b] = (angular_velocities[:, m, b] + torch.pi) % (2 * torch.pi) - torch.pi
            
            # Normalize angular velocities to [-1, 1]
            angular_velocities_normalized = angular_velocities / torch.pi
            all_features.append(angular_velocities_normalized.reshape(T, M * B))
            
            # ============== CENTER OF MASS FEATURES ==============
            
            # Centroid using key body parts
            centroid_parts = ['head_center', 'body_center', 'tail_base']
            centroid_indices = torch.tensor([bp_to_idx[bp] for bp in centroid_parts]).to(device)
            mouse_centroids = torch.mean(keypoints[:, :, centroid_indices, :], dim=2)  # [T, M, 2]
            
            # Centroid kinematics (normalized)
            centroid_velocity = torch.diff(mouse_centroids, dim=0, prepend=mouse_centroids[0:1]) * FPS
            centroid_velocity = torch.clamp(centroid_velocity, -MAX_VELOCITY, MAX_VELOCITY)
            centroid_speed = torch.norm(centroid_velocity, dim=-1)  # [T, M]
            centroid_speed_normalized = centroid_speed / MAX_VELOCITY
            
            centroid_accel = torch.diff(centroid_velocity, dim=0, prepend=centroid_velocity[0:1])
            centroid_accel = torch.clamp(centroid_accel, -MAX_ACCEL, MAX_ACCEL)
            centroid_accel_mag = torch.norm(centroid_accel, dim=-1)
            centroid_accel_normalized = centroid_accel_mag / MAX_ACCEL
            
            all_features.extend([centroid_speed_normalized, centroid_accel_normalized])
            
            # Movement direction (already bounded [-pi, pi])
            movement_heading = torch.atan2(centroid_velocity[:, :, 1], centroid_velocity[:, :, 0])
            movement_heading_normalized = movement_heading / torch.pi  # [-1, 1]
            all_features.append(movement_heading_normalized)
            
            # ============== BODY CONFIGURATION (NORMALIZED) ==============
            
            # Key positions
            nose_pos = keypoints[:, :, bp_to_idx['nose'], :]
            tail_base_pos = keypoints[:, :, bp_to_idx['tail_base'], :]
            head_center_pos = keypoints[:, :, bp_to_idx['head_center'], :]
            body_center_pos = keypoints[:, :, bp_to_idx['body_center'], :]
            ear_left_pos = keypoints[:, :, bp_to_idx['ear_left'], :]
            ear_right_pos = keypoints[:, :, bp_to_idx['ear_right'], :]
            
            # Body length (normalize by percentile or known max)
            body_length = torch.norm(nose_pos - tail_base_pos, dim=-1)  # [T, M]
            # Use 95th percentile for robust normalization
            body_length_95 = torch.quantile(body_length[~torch.isnan(body_length)], 0.95) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0
            body_length_normalized = torch.clamp(body_length / (body_length_95 + EPSILON), 0, 2)  # Allow some values > 1
            all_features.append(body_length_normalized)
            
            # Body length rate of change (normalized)
            body_length_change = torch.diff(body_length, dim=0, prepend=body_length[0:1])
            body_length_change_normalized = torch.clamp(body_length_change / (body_length_95 * 0.1 + EPSILON), -2, 2)
            all_features.append(body_length_change_normalized)
            
            # Ear spread (normalized)
            ear_spread = torch.norm(ear_left_pos - ear_right_pos, dim=-1)  # [T, M]
            ear_spread_95 = torch.quantile(ear_spread[~torch.isnan(ear_spread)], 0.95) if ear_spread[~torch.isnan(ear_spread)].numel() > 0 else 1.0
            ear_spread_normalized = torch.clamp(ear_spread / (ear_spread_95 + EPSILON), 0, 2)
            all_features.append(ear_spread_normalized)
            
            # Body elongation ratio (already a ratio, just clamp)
            body_elongation = safe_divide(body_length, ear_spread)
            body_elongation_normalized = torch.clamp(body_elongation, 0, 10) / 10  # Typical range 1-5
            all_features.append(body_elongation_normalized)
            
            # ============== BODY ANGLES (ALREADY NORMALIZED) ==============
            
            # All angles are in radians, normalize to [-1, 1]
            body_curvature = compute_angle_3points(nose_pos, body_center_pos, tail_base_pos)  # [T, M]
            body_curvature_normalized = body_curvature / torch.pi
            all_features.append(body_curvature_normalized)
            
            head_angle = compute_angle_3points(nose_pos, head_center_pos, body_center_pos)  # [T, M]
            head_angle_normalized = head_angle / torch.pi
            all_features.append(head_angle_normalized)
            
            tail_angle = compute_angle_3points(head_center_pos, body_center_pos, tail_base_pos)  # [T, M]
            tail_angle_normalized = tail_angle / torch.pi
            all_features.append(tail_angle_normalized)
            
            # Body orientation angle
            body_vector = nose_pos - tail_base_pos
            body_orientation = torch.atan2(body_vector[:, :, 1], body_vector[:, :, 0])  # [T, M]
            body_orientation_normalized = body_orientation / torch.pi
            all_features.append(body_orientation_normalized)
            
            # Angular velocity of body orientation
            unwrapped_orientation = torch.from_numpy(np.unwrap(body_orientation.cpu().numpy(), axis=0)).to(device)
            body_angular_velocity = torch.diff(unwrapped_orientation, dim=0, prepend=unwrapped_orientation[0:1])
            body_angular_velocity = torch.clamp(body_angular_velocity, -torch.pi, torch.pi) / torch.pi
            all_features.append(body_angular_velocity)
            
            # ============== EXTREMITY FEATURES (NORMALIZED) ==============
            
            # Normalize distances by body length
            nose_centroid_dist = torch.norm(nose_pos - mouse_centroids, dim=-1)  # [T, M]
            nose_centroid_normalized = safe_divide(nose_centroid_dist, body_length)
            nose_centroid_normalized = torch.clamp(nose_centroid_normalized, 0, 2)
            
            tail_centroid_dist = torch.norm(tail_base_pos - mouse_centroids, dim=-1)  # [T, M]
            tail_centroid_normalized = safe_divide(tail_centroid_dist, body_length)
            tail_centroid_normalized = torch.clamp(tail_centroid_normalized, 0, 2)
            
            all_features.extend([nose_centroid_normalized, tail_centroid_normalized])
            
            # Relative velocities (normalized)
            nose_velocity = velocities[:, :, bp_to_idx['nose'], :]
            tail_velocity = velocities[:, :, bp_to_idx['tail_base'], :]
            nose_rel_velocity = nose_velocity - centroid_velocity
            tail_rel_velocity = tail_velocity - centroid_velocity
            nose_rel_speed = torch.norm(nose_rel_velocity, dim=-1) / MAX_VELOCITY
            tail_rel_speed = torch.norm(tail_rel_velocity, dim=-1) / MAX_VELOCITY
            all_features.extend([nose_rel_speed, tail_rel_speed])
                    
            # ============== INTER-MOUSE FEATURES (NORMALIZED) ==============
            
            inter_features_list = []
            
            # Get typical mouse size for normalization
            typical_mouse_size = torch.median(body_length[~torch.isnan(body_length)]) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0
            

            ######### THIS IS PER MOUSE ADAPTATION
            # Initialize per-mouse feature lists
            mouse_features = [[] for _ in range(M)]
            
            # Add bodypart-specific features per mouse
            bp_specific = [speeds_normalized, accel_normalized, jerk_normalized, angular_velocities_normalized]
            for feat in bp_specific:  # each [T, M, B]
                for m in range(M):
                    mouse_features[m].append(feat[:, m, :].reshape(T, B))  # [T, 6]
            
            # Add mouse-specific features per mouse
            mouse_specific = [
                centroid_speed_normalized, centroid_accel_normalized, movement_heading_normalized,
                body_length_normalized, body_length_change_normalized, ear_spread_normalized, body_elongation_normalized,
                body_curvature_normalized, head_angle_normalized, tail_angle_normalized, body_orientation_normalized, body_angular_velocity,
                nose_centroid_normalized, tail_centroid_normalized, nose_rel_speed, tail_rel_speed
            ]
            for feat in mouse_specific:  # each [T, M]
                for m in range(M):
                    mouse_features[m].append(feat[:, m].unsqueeze(-1))  # [T, 1]

            pure_node_feats_list = [torch.cat(feats, dim=-1) for feats in mouse_features]
            pure_node_feats = torch.stack(pure_node_feats_list, dim=1) # [T, M, Node_Dim]
            pure_node_feats = torch.clamp(pure_node_feats, -10, 10)
            pure_node_feats = torch.nan_to_num(pure_node_feats, nan=0.0, posinf=10.0, neginf=-10.0)

            edge_attrs_matrix = torch.zeros(T, M, M, 8).to(device)
            
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    
                    # Distances normalized by typical mouse size
                    centroid_i = mouse_centroids[:, i, :]
                    centroid_j = mouse_centroids[:, j, :]
                    dist_centroids = torch.norm(centroid_j - centroid_i, dim=-1)
                    dist_centroids_normalized = torch.clamp(dist_centroids / (typical_mouse_size * 5 + EPSILON), 0, 2)
                    
                    mouse_features[i].append(dist_centroids_normalized.unsqueeze(-1))  # From i's perspective
                    inter_features_list.append(dist_centroids_normalized.unsqueeze(-1))
                    
                    # Rate of distance change (normalized)
                    dist_change = torch.diff(dist_centroids, dim=0, prepend=dist_centroids[0:1])
                    dist_change_normalized = torch.clamp(dist_change / (typical_mouse_size + EPSILON), -2, 2) / 2
                    
                    mouse_features[i].append(dist_change_normalized.unsqueeze(-1))
                    inter_features_list.append(dist_change_normalized.unsqueeze(-1))
                    
                    nose_i = nose_pos[:, i, :]
                    nose_j = nose_pos[:, j, :]
                    tail_j = tail_base_pos[:, j, :]
                    
                    nose_nose_dist = torch.norm(nose_i - nose_j, dim=-1)
                    nose_nose_normalized = torch.clamp(nose_nose_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                    mouse_features[i].append(nose_nose_normalized.unsqueeze(-1))
                    
                    nose_tail_dist = torch.norm(nose_i - tail_j, dim=-1)
                    nose_tail_normalized = torch.clamp(nose_tail_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                    mouse_features[i].append(nose_tail_normalized.unsqueeze(-1))
                    
                    inter_features_list.extend([
                        nose_nose_normalized.unsqueeze(-1),
                        nose_tail_normalized.unsqueeze(-1)
                    ])
                    
                    rel_speed = (centroid_speed_normalized[:, i] - centroid_speed_normalized[:, j])
                    mouse_features[i].append(rel_speed.unsqueeze(-1))
                    inter_features_list.append(rel_speed.unsqueeze(-1))
                    
                    # Approach angle (normalized)
                    vec_to_j = centroid_j - centroid_i
                    heading_vec_i = body_vector[:, i, :]
                    dot_prod = torch.sum(heading_vec_i * vec_to_j, dim=-1)
                    norm_prod = torch.norm(heading_vec_i, dim=-1) * torch.norm(vec_to_j, dim=-1)
                    approach_angle = torch.acos(torch.clamp(safe_divide(dot_prod, norm_prod), -1.0 + EPSILON, 1.0 - EPSILON))
                    approach_angle_normalized = approach_angle / torch.pi
                    
                    inter_features_list.append(approach_angle_normalized.unsqueeze(-1))
                    mouse_features[i].append(approach_angle_normalized.unsqueeze(-1))

                    heading_vec_j = body_vector[:, j, :]
                    dot_align = torch.sum(heading_vec_i * heading_vec_j, dim=-1)
                    norm_h_i = torch.norm(heading_vec_i, dim=-1) + EPSILON
                    norm_h_j = torch.norm(heading_vec_j, dim=-1) + EPSILON
                    alignment = torch.clamp(dot_align / (norm_h_i * norm_h_j), -1.0, 1.0)
                    
                    vec_to_i = -vec_to_j # Vector from J to I
                    dot_aspect = torch.sum(heading_vec_j * vec_to_i, dim=-1)
                    norm_vec_i = torch.norm(vec_to_i, dim=-1) + EPSILON
                    target_aspect = torch.acos(torch.clamp(dot_aspect / (norm_h_j * norm_vec_i), -1.0 + EPSILON, 1.0 - EPSILON))
                    target_aspect_norm = target_aspect / torch.pi

                    edge_attrs_matrix[:, i, j, 0] = dist_centroids_normalized
                    edge_attrs_matrix[:, i, j, 1] = dist_change_normalized
                    edge_attrs_matrix[:, i, j, 2] = nose_nose_normalized
                    edge_attrs_matrix[:, i, j, 3] = nose_tail_normalized
                    edge_attrs_matrix[:, i, j, 4] = rel_speed
                    edge_attrs_matrix[:, i, j, 5] = approach_angle_normalized
                    edge_attrs_matrix[:, i, j, 6] = alignment
                    edge_attrs_matrix[:, i, j, 7] = target_aspect_norm
                    
            if inter_features_list:
                inter_features = torch.cat(inter_features_list, dim=-1)
                all_features.append(inter_features)
            
            per_mouse_feats = [torch.cat(feats, dim=-1) for feats in mouse_features]  # List of [T, 58]
            # ============== CONCATENATE AND FINAL NORMALIZATION ==============
            
            features = torch.cat(all_features, dim=-1)
            per_mouse_feats = torch.stack(per_mouse_feats, dim=1)  # [T, M, 58]

            features = torch.clamp(features, -10, 10)
            per_mouse_feats = torch.clamp(per_mouse_feats, -10, 10)
            
            # Handle NaN/Inf
            features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            per_mouse_feats = torch.nan_to_num(per_mouse_feats, nan=0.0, posinf=10.0, neginf=-10.0)
            edge_attrs_matrix = torch.nan_to_num(edge_attrs_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

            return features, per_mouse_feats, pure_node_feats, edge_attrs_matrix
                


def collate_fn(batch):
    """Custom collate function for batching"""
    inputs = torch.stack([item['input'] for item in batch])
    inputs_ego = torch.stack([item['input_ego'] for item in batch]) if 'input_ego' in batch[0] else None
    inputs_mice = torch.stack([item['input_mice'] for item in batch]) if 'input_mice' in batch[0] else None
    inputs_ego_mice = torch.stack([item['input_ego_mice'] for item in batch]) if 'input_ego_mice' in batch[0] else None
    node_feats = torch.stack([item['node_feats'] for item in batch]) if 'node_feats' in batch[0] else None
    edge_feats = torch.stack([item['edge_feats'] for item in batch]) if 'edge_feats' in batch[0] else None
    masks = torch.stack([item['input_mask'] for item in batch])
    behavior_masks = torch.stack([item['behavior_mask'] for item in batch]) if 'behavior_mask' in batch[0] else None

    labels = torch.stack([item['labels'] for item in batch]) if batch[0]['labels'] is not None else None

    video_ids = [item['video_id'] for item in batch]
    start_frames = torch.stack([item['start_frame'] for item in batch])
    num_frames = torch.stack([item['num_frames'] for item in batch])

    collated = {
        'input': inputs,
        'input_ego': inputs_ego,
        'input_mice': inputs_mice,
        'input_ego_mice': inputs_ego_mice,
        'node_feats': node_feats,
        'edge_feats': edge_feats,
        'input_mask': masks,
        'labels': labels,
        'video_id': video_ids,
        'start_frame': start_frames,
        'num_frames': num_frames
    }
    if behavior_masks is not None:
        collated['behavior_mask'] = behavior_masks

    return collated
    
# For compatibility
tr_collate_fn = collate_fn
val_collate_fn = collate_fn


def process_one_video_preds_filters(cfg, video_preds, video_id, lab_id, num_frames,
                                       mouse_map, action_map, num_pairs, num_actions,
                                       min_duration, smooth_sigma=0.0, conf_thresh=0.0,
                                       max_gap=0, use_median_filter=False):


    if smooth_sigma > 0:
        smoothed_preds = np.zeros_like(video_preds)
        for pair in range(num_pairs):
            for act in range(num_actions):
                smoothed_preds[:, pair, act] = gaussian_filter1d(video_preds[:, pair, act], sigma=smooth_sigma)
        video_preds = smoothed_preds / np.sum(smoothed_preds, axis=2, keepdims=True)  # Re-normalize

    # Get best actions
    best_action_indices = np.argmax(video_preds, axis=2)  # [num_frames, num_pairs]
    best_probs = np.max(video_preds, axis=2)  # [num_frames, num_pairs] for confidence

    # Optional: Median filter on labels (after argmax)
    if use_median_filter:
        for pair in range(num_pairs):
            best_action_indices[:, pair] = median_filter(best_action_indices[:, pair], size=5)  # Window size tunable

    video_predictions = []
    num_mice = int(np.sqrt(num_pairs))

    for pair_idx in range(num_pairs):
        agent_idx = pair_idx // num_mice
        target_idx = pair_idx % num_mice
        agent_id = mouse_map[agent_idx]
        target_id = mouse_map[target_idx] if agent_idx != target_idx else 'self'

        for action_idx in range(num_actions):
            action = action_map[action_idx]
            if action == 'no_action':
                continue
            if (agent_idx == target_idx and action in cfg.social_only_actions) or \
               (agent_idx != target_idx and action in cfg.self_only_actions):
                continue

            behavior_frames = (best_action_indices[:, pair_idx] == action_idx)

            # Find contiguous segments
            padded = np.concatenate(([False], behavior_frames, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0] - 1

            # Step 2: Merge gaps (for same action)
            if len(starts) > 1 and max_gap > 0:
                merged_starts, merged_ends = [starts[0]], [ends[0]]
                for i in range(1, len(starts)):
                    if starts[i] - merged_ends[-1] - 1 <= max_gap:
                        merged_ends[-1] = ends[i]  # Merge
                    else:
                        merged_starts.append(starts[i])
                        merged_ends.append(ends[i])
                starts, ends = np.array(merged_starts), np.array(merged_ends)

            # Filter by min_duration and confidence
            durations = ends - starts + 1
            valid = durations >= min_duration
            for start, end in zip(starts[valid], ends[valid]):
                seg_probs = video_preds[start:end+1, pair_idx, action_idx]  # Use action-specific probs
                mean_conf = np.mean(seg_probs)
                if mean_conf < conf_thresh:
                    continue
                video_predictions.append({
                    'video_id': video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': action,
                    'start_frame': start,
                    'stop_frame': end,
                    'confidence': mean_conf
                })

    if video_predictions:
        df = pd.DataFrame(video_predictions)
        resolved_predictions = []
        for (agent_id, target_id), group in df.groupby(['agent_id', 'target_id']):
            segments = group.to_dict('records')
            segments.sort(key=lambda x: x['start_frame'])
            # Check for overlap
            overlap_found = False
            for k in range(1, len(segments)):
                if segments[k]['start_frame'] <= segments[k-1]['stop_frame']:
                    overlap_found = True
                    break
            if overlap_found:
                print(f"Warning: Overlaps found in {video_id} for pair {agent_id}-{target_id}. Resolving with NMS.")
                # NMS: Sort by descending confidence, greedily select non-overlapping
                segments.sort(key=lambda x: -x['confidence'])
                selected = []
                for seg in segments:
                    if all(max(seg['start_frame'], sel['start_frame']) > min(seg['stop_frame'], sel['stop_frame']) for sel in selected):
                        selected.append(seg)
                resolved_predictions.extend(selected)
            else:
                resolved_predictions.extend(segments)
        if resolved_predictions:
            return pd.DataFrame(resolved_predictions)
    return None



import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import argparse
import torch
from torch.amp import autocast, GradScaler
import json
from joblib import Parallel, delayed
import ast
from time import time
from torch.utils.data import DataLoader
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

import model_1
import model_gnn
import model_inter
import model_238 as model_238_code
import model_242 as model_242_code
import model_243 as model_243_code
import model_244 as model_244_code

warnings.filterwarnings('ignore')

start_time = time()

cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {cfg.device}")

window_size = cfg.window_size
cfg.stride = window_size // 2  # 50% overlap
min_duration = 5 # Minimum frames for a behavior

cfg_233 = copy(cfg)
cfg_233.reverse_time = False
cfg_233.use_gnn = False
cfg_233.use_bn = True
cfg_233.cnn_extractor = True
cfg_233.encoder_config.input_dim=256
cfg_233.encoder_config.encoder_dim=256
cfg_233.encoder_config.num_layers=4
cfg_233.encoder_config.num_attention_heads=4
cfg_233.per_mouse_feature_dim = cfg_233.feature_dim // 4


model_233 = model_gnn.NetGNN(cfg_233).to(cfg_233.device)
model_233 = torch.compile(model_233)
weights_path_233 = "/kaggle/input/mabe-weights/MAB-233_best.pth"
weights_233 = torch.load(weights_path_233, map_location=cfg_233.device, weights_only=False)#['model_state_dict']
model_233.load_state_dict(weights_233)
model_233.eval()


cfg_238 = copy(cfg)
cfg_238.cnn_extractor = True
cfg_238.use_gnn = True
cfg_238.use_bn = True
cfg_238.encoder_config.input_dim=256
cfg_238.encoder_config.encoder_dim=256
cfg_238.encoder_config.num_layers=4
cfg_238.encoder_config.num_attention_heads=4
cfg_238.per_mouse_feature_dim = cfg_238.feature_dim // 4

model_238 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_238 = torch.compile(model_238)
weights_path_238 = "/kaggle/input/mabe-weights/MAB-238_best.pth"
weights_238 = torch.load(weights_path_238, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_238.load_state_dict(weights_238)
model_238.eval()

model_289 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_289 = torch.compile(model_289)
weights_path_289 = "/kaggle/input/mabe-weights/MAB-289_best.pth"
weights_289 = torch.load(weights_path_289, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_289.load_state_dict(weights_289)
model_289.eval()

model_290 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_290 = torch.compile(model_290)
weights_path_290 = "/kaggle/input/mabe-weights/MAB-290_best.pth"
weights_290 = torch.load(weights_path_290, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_290.load_state_dict(weights_290)
model_290.eval()

model_293 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_293 = torch.compile(model_293)
weights_path_293 = "/kaggle/input/mabe-weights/MAB-293_best.pth"
weights_293 = torch.load(weights_path_293, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_293.load_state_dict(weights_293)
model_293.eval()

model_292 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_292 = torch.compile(model_292)
weights_path_292 = "/kaggle/input/mabe-weights/MAB-292_best.pth"
weights_292 = torch.load(weights_path_292, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_292.load_state_dict(weights_292)
model_292.eval()

model_245 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_245 = torch.compile(model_245)
weights_path_245 = "/kaggle/input/mabe-weights/MAB-245_best.pth"
weights_245 = torch.load(weights_path_245, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_245.load_state_dict(weights_245)
model_245.eval()

model_264 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_264 = torch.compile(model_264)
weights_path_264 = "/kaggle/input/mabe-weights/MAB-264_best.pth"
weights_264 = torch.load(weights_path_264, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_264.load_state_dict(weights_264)
model_264.eval()

model_266 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_266 = torch.compile(model_266)
weights_path_266 = "/kaggle/input/mabe-weights/MAB-266_best.pth"
weights_266 = torch.load(weights_path_266, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_266.load_state_dict(weights_266)
model_266.eval()

model_267 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_267 = torch.compile(model_267)
weights_path_267 = "/kaggle/input/mabe-weights/MAB-267_best.pth"
weights_267 = torch.load(weights_path_267, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_267.load_state_dict(weights_267)
model_267.eval()

model_269 = model_238_code.Net(cfg_238).to(cfg_238.device)
model_269 = torch.compile(model_269)
weights_path_269 = "/kaggle/input/mabe-weights/MAB-269_best.pth"
weights_269 = torch.load(weights_path_269, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_269.load_state_dict(weights_269)
model_269.eval()

cfg_244 = copy(cfg_238)
cfg_244.per_mouse_feature_dim = 304 // 4
model_244 = model_244_code.Net(cfg_244).to(cfg_244.device)
model_244 = torch.compile(model_244)
weights_path_244 = "/kaggle/input/mabe-weights/MAB-244_best.pth"
weights_244 = torch.load(weights_path_244, map_location=cfg_244.device, weights_only=False)#['model_state_dict']
model_244.load_state_dict(weights_244)
model_244.eval()

cfg_234 = copy(cfg)
cfg_234.reverse_time = False
cfg_234.use_gnn = False
cfg_234.use_bn = True
cfg_234.cnn_extractor = True
cfg_234.encoder_config.input_dim=256
cfg_234.encoder_config.encoder_dim=256
cfg_234.encoder_config.num_layers=4
cfg_234.encoder_config.num_attention_heads=4
cfg_234.per_mouse_feature_dim = cfg_234.feature_dim // 4

model_234 = model_inter.NetInter(cfg_234).to(cfg_234.device)
model_234 = torch.compile(model_234)
weights_path_234 = "/kaggle/input/mabe-weights/MAB-234_best.pth"
weights_234 = torch.load(weights_path_234, map_location=cfg_234.device, weights_only=False)#['model_state_dict']
model_234.load_state_dict(weights_234)
model_234.eval()

cfg_240 = copy(cfg)
cfg_240.cnn_extractor = True
cfg_240.use_gnn = False
cfg_240.use_bn = True
cfg_240.encoder_config.input_dim=256
cfg_240.encoder_config.encoder_dim=256
cfg_240.encoder_config.num_layers=8
cfg_240.encoder_config.num_attention_heads=4
cfg_240.per_mouse_feature_dim = cfg_240.feature_dim // 4

model_240 = model_inter.NetInter(cfg_240).to(cfg_240.device)
model_240 = torch.compile(model_240)
weights_path_240 = "/kaggle/input/mabe-weights/MAB-240_best.pth"
weights_240 = torch.load(weights_path_240, map_location=cfg_240.device, weights_only=False)#['model_state_dict']
model_240.load_state_dict(weights_240)
model_240.eval()

cfg_241 = copy(cfg)
cfg_241.cnn_extractor = True
cfg_241.use_gnn = False
cfg_241.use_bn = True
cfg_241.encoder_config.input_dim=256
cfg_241.encoder_config.encoder_dim=256
cfg_241.encoder_config.num_layers=4
cfg_241.encoder_config.num_attention_heads=4
cfg_241.per_mouse_feature_dim = cfg_241.feature_dim // 4

model_241 = model_inter.NetInter(cfg_241).to(cfg_241.device)
model_241 = torch.compile(model_241)
weights_path_241 = "/kaggle/input/mabe-weights/MAB-241_best.pth"
weights_241 = torch.load(weights_path_241, map_location=cfg_241.device, weights_only=False)#['model_state_dict']
model_241.load_state_dict(weights_241)
model_241.eval()

cfg_242 = copy(cfg)
cfg_242.cnn_extractor = True
cfg_242.use_gnn = True
cfg_242.use_bn = True
cfg_242.encoder_config.input_dim=256
cfg_242.encoder_config.encoder_dim=256
cfg_242.encoder_config.num_layers=4
cfg_242.encoder_config.num_attention_heads=4
cfg_242.per_mouse_feature_dim = 40 #cfg_242.feature_dim // 4

model_242 = model_242_code.Net(cfg_242).to(cfg_242.device)
model_242 = torch.compile(model_242)
weights_path_242 = "/kaggle/input/mabe-weights/MAB-242_best.pth"
weights_242 = torch.load(weights_path_242, map_location=cfg_242.device, weights_only=False)#['model_state_dict']
model_242.load_state_dict(weights_242)
model_242.eval()

cfg_243 = copy(cfg)
cfg_243.cnn_extractor = True
cfg_243.use_gnn = True
cfg_243.use_bn = True
cfg_243.encoder_config.input_dim=256
cfg_243.encoder_config.encoder_dim=256
cfg_243.encoder_config.num_layers=4
cfg_243.encoder_config.num_attention_heads=4
cfg_243.per_mouse_feature_dim = 40 #cfg.feature_dim // len(cfg.set_mice)

model_243 = model_243_code.Net(cfg_243).to(cfg_243.device)
model_243 = torch.compile(model_243)
weights_path_243 = "/kaggle/input/mabe-weights/MAB-243_best.pth"
weights_243 = torch.load(weights_path_243, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_243.load_state_dict(weights_243)
model_243.eval()

model_294 = model_243_code.Net(cfg_243).to(cfg_243.device)
model_294 = torch.compile(model_294)
weights_path_294 = "/kaggle/input/mabe-weights/MAB-294_best.pth"
weights_294 = torch.load(weights_path_294, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_294.load_state_dict(weights_294)
model_294.eval()

model_295 = model_243_code.Net(cfg_243).to(cfg_243.device)
model_295 = torch.compile(model_295)
weights_path_295 = "/kaggle/input/mabe-weights/MAB-295_best.pth"
weights_295 = torch.load(weights_path_295, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_295.load_state_dict(weights_295)
model_295.eval()

model_297 = model_243_code.Net(cfg_243).to(cfg_243.device)
model_297 = torch.compile(model_297)
weights_path_297 = "/kaggle/input/mabe-weights/MAB-297_best.pth"
weights_297 = torch.load(weights_path_297, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_297.load_state_dict(weights_297)
model_297.eval()


cfg_256 = copy(cfg)
cfg_256.cnn_extractor = True
cfg_256.use_gnn = True
cfg_256.use_bn = True
cfg_256.encoder_config.input_dim=256
cfg_256.encoder_config.encoder_dim=256
cfg_256.encoder_config.num_layers=8
cfg_256.encoder_config.num_attention_heads=4

model_256 = model_inter.NetInter(cfg_256).to(cfg_256.device)
model_256 = torch.compile(model_256)
weights_path_256 = "/kaggle/input/mabe-weights/MAB-256_best.pth"
weights_256 = torch.load(weights_path_256, map_location=cfg_256.device, weights_only=False)#['model_state_dict']
model_256.load_state_dict(weights_256)
model_256.eval()

cfg_273 = copy(cfg)
cfg_273.cnn_extractor = True
cfg_273.use_gnn = False
cfg_273.use_bn = True
cfg_273.encoder_config.input_dim=256
cfg_273.encoder_config.encoder_dim=256
cfg_273.encoder_config.num_layers=4
cfg_273.encoder_config.num_attention_heads=4
cfg_273.per_mouse_feature_dim = cfg_273.feature_dim // 4

model_273 = model_gnn.NetGNN(cfg_273).to(cfg_273.device)
model_273 = torch.compile(model_273)
weights_path_273 = "/kaggle/input/mabe-weights/MAB-273_best.pth"
weights_273 = torch.load(weights_path_273, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_273.load_state_dict(weights_273)
model_273.eval() 

model_274 = model_gnn.NetGNN(cfg_273).to(cfg_273.device)
model_274 = torch.compile(model_274)
weights_path_274 = "/kaggle/input/mabe-weights/MAB-274_best.pth"
weights_274 = torch.load(weights_path_274, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_274.load_state_dict(weights_274)
model_274.eval()

model_275 = model_gnn.NetGNN(cfg_273).to(cfg_273.device)
model_275 = torch.compile(model_275)
weights_path_275 = "/kaggle/input/mabe-weights/MAB-275_best.pth"
weights_275 = torch.load(weights_path_275, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_275.load_state_dict(weights_275)
model_275.eval()

model_276 = model_gnn.NetGNN(cfg_273).to(cfg_273.device)
model_276 = torch.compile(model_276)
weights_path_276 = "/kaggle/input/mabe-weights/MAB-276_best.pth"
weights_276 = torch.load(weights_path_276, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_276.load_state_dict(weights_276)
model_276.eval()


cfg_278 = copy(cfg)
cfg_278.cnn_extractor = True
cfg_278.use_gnn = False
cfg_278.use_bn = True
cfg_278.encoder_config.input_dim=256
cfg_278.encoder_config.encoder_dim=256
cfg_278.encoder_config.num_layers=4
cfg_278.encoder_config.num_attention_heads=4
cfg_278.per_mouse_feature_dim = cfg_278.feature_dim // 4

model_278 = model_inter.NetInter(cfg_278).to(cfg_278.device)
model_278 = torch.compile(model_278)
weights_path_278 = "/kaggle/input/mabe-weights/MAB-278_best.pth"
weights_278 = torch.load(weights_path_278, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_278.load_state_dict(weights_278)
model_278.eval()

model_279 = model_inter.NetInter(cfg_278).to(cfg_278.device)
model_279 = torch.compile(model_279)
weights_path_279 = "/kaggle/input/mabe-weights/MAB-279_best.pth"
weights_279 = torch.load(weights_path_279, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_279.load_state_dict(weights_279)
model_279.eval()

model_284 = model_inter.NetInter(cfg_278).to(cfg_278.device)
model_284 = torch.compile(model_284)
weights_path_284 = "/kaggle/input/mabe-weights/MAB-284_best.pth"
weights_284 = torch.load(weights_path_284, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_284.load_state_dict(weights_284)
model_284.eval()

model_285 = model_inter.NetInter(cfg_278).to(cfg_278.device)
model_285 = torch.compile(model_285)
weights_path_285 = "/kaggle/input/mabe-weights/MAB-285_best.pth"
weights_285 = torch.load(weights_path_285, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_285.load_state_dict(weights_285)
model_285.eval()


# Get action/mouse info from config
num_mice = len(cfg.set_mice)
num_pairs = num_mice * num_mice
actions = cfg.set_behavior_classes
num_actions = len(actions)
action_map = cfg.id_to_action_map

mouse_map_str = cfg.mouse_id_to_string
int_to_str_map = {1: 'mouse1', 2: 'mouse2', 3: 'mouse3', 4: 'mouse4'}

scaler = GradScaler(device=cfg.device) if cfg.mixed_precision else None
test_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

chunk_size = 10  # Adjust based on memory limits; e.g., process 50 videos at a time
all_predictions_dfs = []

for i in range(0, len(test_df), chunk_size):
    chunk_df = test_df.iloc[i:i + chunk_size]
    temp_prediction_dfs = []
    val_dataset = CustomDataset(chunk_df, cfg, aug=None, mode="val")

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.batch_size_val if hasattr(cfg, 'batch_size_val') else cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn if val_collate_fn else tr_collate_fn,
        drop_last=False
    )

    # --- PHASE 1: GPU-BOUND WORK (Inference) ---
    video_accum = {}  # video_id -> {'all_preds': np.array, 'counts': np.array, 'lab_id': str, 'num_frames': int}
    for batch in tqdm(val_dataloader, desc="Validation"):
        batch = batch_to_device(batch, cfg.device)
        with torch.no_grad():
            # output_233 = model_233(batch)
            # preds_batch_233 = output_233['predictions'].cpu().numpy()
    
            # output_234 = model_234(batch)
            # preds_batch_234 = output_234['predictions'].cpu().numpy() 
    
            # output_238 = model_238(batch)
            # preds_batch_238 = output_238['predictions'].cpu().numpy()  
    
            output_240 = model_240(batch)
            preds_batch_240 = output_240['predictions'].cpu().numpy()
        
            output_242 = model_242(batch)
            preds_batch_242 = output_242['predictions'].cpu().numpy()
                
            # output_243 = model_243(batch)
            # preds_batch_243 = output_243['predictions'].cpu().numpy()
                
            output_244 = model_244(batch)
            preds_batch_244 = output_244['predictions'].cpu().numpy() 
                
            # output_245 = model_245(batch)
            # preds_batch_245 = output_245['predictions'].cpu().numpy() 
                 
            output_256 = model_256(batch)
            preds_batch_256 = output_256['predictions'].cpu().numpy()  
            
            # ##################
            output_264 = model_264(batch)
            preds_batch_264 = output_264['predictions'].cpu().numpy()

            output_266 = model_266(batch)
            preds_batch_266 = output_266['predictions'].cpu().numpy()

            output_267 = model_267(batch)
            preds_batch_267 = output_267['predictions'].cpu().numpy()

            output_269 = model_269(batch)
            preds_batch_269 = output_269['predictions'].cpu().numpy()

            preds_batch_245fold = (preds_batch_264 + preds_batch_266 
                                + preds_batch_267 + preds_batch_269) / 4.0    
            ###############
            output_273 = model_273(batch)
            preds_batch_273 = output_273['predictions'].cpu().numpy()

            output_274 = model_274(batch)
            preds_batch_274 = output_274['predictions'].cpu().numpy()

            output_275 = model_275(batch)
            preds_batch_275 = output_275['predictions'].cpu().numpy()

            output_276 = model_276(batch)
            preds_batch_276 = output_276['predictions'].cpu().numpy()

            preds_batch_233fold = (preds_batch_273 + preds_batch_274 
                                + preds_batch_275 + preds_batch_276) / 4.0            
            ###############
            output_278 = model_278(batch)
            preds_batch_278 = output_278['predictions'].cpu().numpy()

            output_279 = model_279(batch)
            preds_batch_279 = output_279['predictions'].cpu().numpy()

            output_284 = model_284(batch)
            preds_batch_284 = output_284['predictions'].cpu().numpy()

            output_285 = model_285(batch)
            preds_batch_285 = output_285['predictions'].cpu().numpy()
            
            preds_batch_234fold = (preds_batch_278 + preds_batch_279 
                                + preds_batch_284 + preds_batch_285) / 4.0
            #################
            output_289 = model_289(batch)
            preds_batch_289 = output_289['predictions'].cpu().numpy()
            
            output_290 = model_290(batch)
            preds_batch_290 = output_290['predictions'].cpu().numpy()

            output_293 = model_293(batch)
            preds_batch_293 = output_293['predictions'].cpu().numpy()

            output_292 = model_292(batch)
            preds_batch_292 = output_292['predictions'].cpu().numpy()
            
            preds_batch_238fold = (preds_batch_289 + preds_batch_290 
                                   + preds_batch_293 + preds_batch_292) / 4.0
            ##################
            output_294 = model_294(batch)
            preds_batch_294 = output_294['predictions'].cpu().numpy()
            
            output_295 = model_295(batch)
            preds_batch_295 = output_295['predictions'].cpu().numpy()
            
            output_297 = model_297(batch)
            preds_batch_297 = output_297['predictions'].cpu().numpy()
            
            
            preds_batch_243fold = (preds_batch_294 + preds_batch_295 
                                   + preds_batch_297) / 3.0

            
            preds_batch = (preds_batch_233fold + preds_batch_234fold
                + preds_batch_238fold + preds_batch_240 + preds_batch_242 + preds_batch_243fold
                + preds_batch_244 + preds_batch_245fold + preds_batch_256) / 9.0
        

        
        bs = preds_batch.shape[0]
        for i in range(bs):
            video_id = batch['video_id'][i]
            if video_id not in video_accum:
                num_frames = batch['num_frames'][i].item()
                lab_id = val_dataset.video_info[video_id]['lab_id']
                video_accum[video_id] = {
                    'all_preds': np.zeros((num_frames, num_pairs, num_actions), dtype=np.float32),
                    'counts': np.zeros((num_frames, 1, 1), dtype=np.float32),
                    'lab_id': lab_id,
                    'num_frames': num_frames
                }
            
            start = batch['start_frame'][i].item()
            mask = batch['input_mask'][i].cpu().numpy()  # [seq_len]
            actual_len = int(np.sum(mask))
            preds = preds_batch[i][:actual_len]
            # No global cfg.reverse_time check needed; flipping handled above
            
            end = min(start + actual_len, video_accum[video_id]['num_frames'])
            video_accum[video_id]['all_preds'][start:end] += preds[: (end - start)]
            video_accum[video_id]['counts'][start:end] += 1

    # Average predictions
    for video_id in video_accum:
        acc = video_accum[video_id]
        counts = np.maximum(acc['counts'], 1)
        acc['all_preds'] /= counts

    # Prepare data for post-processing
    all_video_data_for_processing = [
        {
            'video_preds': video_accum[video_id]['all_preds'],
            'video_id': video_id,
            'lab_id': video_accum[video_id]['lab_id'],
            'num_frames': video_accum[video_id]['num_frames']
        }
        for video_id in video_accum
    ]

    # --- PHASE 2: Post-Processing ---
    results = Parallel(n_jobs=-1)(
        delayed(process_one_video_preds_filters)(
            cfg,
            data['video_preds'],
            data['video_id'],
            data['lab_id'],
            data['num_frames'],
            mouse_map_str,
            action_map,
            num_pairs,
            num_actions,
            min_duration,
            smooth_sigma=0.0, 
            conf_thresh=0.0,
            max_gap=0, 
            use_median_filter=False
        ) for data in tqdm(all_video_data_for_processing, desc="Phase 2: Post-Processing")
    )
    temp_prediction_dfs = [df for df in results if df is not None]
    all_predictions_dfs.extend(temp_prediction_dfs)
    del val_dataloader
    del val_dataset
    gc.collect()
FINAL_SUBMISSION_COLS = [
    'row_id',
    'video_id',
    'agent_id',
    'target_id',
    'action',
    'start_frame',
    'stop_frame'
]
##END
if all_predictions_dfs:
    predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    predictions_df['row_id'] = range(len(predictions_df))
    predictions_df = predictions_df[FINAL_SUBMISSION_COLS]
else:
    predictions_df = pd.DataFrame(columns=FINAL_SUBMISSION_COLS)


predictions_df.to_csv("submission.csv", index=False)

submission_file = pd.read_csv('submission.csv')
print(f"Saved the submission file with {len(submission_file)} rows.")
end_time = time()
print(f"Total inference and post-processing time: {end_time - start_time:.2f} seconds")

