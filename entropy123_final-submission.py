### -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Installation Block (Choose based on environment)
# ----------------------------------------------------------------------------

!pip install \
  /kaggle/input/gcn-related/whl/torch_geometric/torch_geometric/pyg_lib-0.4.0-cp310-cp310-linux_x86_64.whl \
  /kaggle/input/gcn-related/whl/torch_geometric/torch_geometric/torch_cluster-1.6.3-cp310-cp310-linux_x86_64.whl \
  /kaggle/input/gcn-related/whl/torch_geometric/torch_geometric/torch_scatter-2.1.2-cp310-cp310-linux_x86_64.whl \
  /kaggle/input/gcn-related/whl/torch_geometric/torch_geometric/torch_sparse-0.6.18-cp310-cp310-linux_x86_64.whl \
  /kaggle/input/gcn-related/whl/torch_geometric/torch_geometric/torch_spline_conv-1.2.2-cp310-cp310-linux_x86_64.whl \
  /kaggle/input/gcn-related/geometric1/torch_geometric-2.6.1-py3-none-any.whl\
  --no-index \
  --find-links=../input/gcn-related/whl/torch_geometric/torch_geometric


!pip install \/kaggle/input/gcn-related/iterative/iterative_stratification-0.1.9-py3-none-any.whl\
    --no-index \
  --find-links=../input/gcn-related/iterative/
!pip install torch_geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html


print("Installed libraries using pip.")

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import sys
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from copy import deepcopy as dp
import seaborn as sns
from scipy.sparse import coo_matrix

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset # Explicitly import

# import optuna # No longer needed

import warnings
warnings.filterwarnings('ignore')

# Check versions (optional, but good practice)
print(f"\n--- Versions ---")
print(f"Python Version: {sys.version.split(' ')[0]}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"PyTorch Version: {torch.__version__}")
try:
    import torch_geometric
    print(f"Torch Geometric Version: {torch_geometric.__version__}")
except ImportError:
    print("Torch Geometric not found.")
try:
    import iterstrat
    print(f"Iterative Stratification Version: {iterstrat.__version__}")
except ImportError:
    print("Iterative Stratification not found.")
print("-" * 16)

# ----------------------------------------------------------------------------
# Helper Functions (Copied from original)
# ----------------------------------------------------------------------------

def norm_fit(df_1, saveM=True, sc_name='zsco'):
    """Fits a scaler to the data."""
    ss_1_dic = {'zsco': StandardScaler(),
                'mima': MinMaxScaler(),
                'maxb': MaxAbsScaler(),
                'robu': RobustScaler(),
                'norm': Normalizer(),
                'quan': QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal"),
                'powe': PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1), index=df_1.index, columns=df_1.columns)
    if saveM == False:
        return df_2
    else:
        return df_2, ss_1

def norm_tra(df_1, ss_x):
    """Transforms data using a pre-fitted scaler."""
    df_2 = pd.DataFrame(ss_x.transform(df_1), index=df_1.index, columns=df_1.columns)
    return df_2

def g_table(list1):
    """Creates a frequency table (dictionary) from a list."""
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return table_dic

def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in CuDNN convolutions
    torch.backends.cudnn.deterministic = True
    # Disabling benchmark mode might slightly slow down computation but improves reproducibility
    torch.backends.cudnn.benchmark = False


def get_edge(y_train, p, tau):
    """
    Calculates graph edges based on label co-occurrence and conditional probability.

    Args:
        y_train (np.ndarray): Training target labels (samples x labels).
        p (float): Probability transition parameter for GCN edge weights.
        tau (float): Conditional probability threshold for edge creation.

    Returns:
        tuple: (edge_index, edge_weight) as torch tensors.
    """
    co_occur = y_train.T @ y_train
    N = np.maximum(y_train.sum(axis=0), 1e-8) # Sum occurrences for each label, avoid division by zero
    P_cond = co_occur / N[:, np.newaxis] # Conditional probability P(j|i) = C(i,j) / N(i)

    A = (P_cond >= tau).astype(float) # Adjacency matrix based on threshold tau
    num_labels = A.shape[0]
    for i in range(num_labels):
        A[i][i] = 0.0 # Remove self-loops in the initial adjacency

    degree = A.sum(axis=1)
    adj = np.zeros_like(A, dtype=float) # Initialize weighted adjacency matrix

    for i in range(num_labels):
        if degree[i] > 0:
            for j in range(num_labels):
                if j == i:
                    adj[i][i] = 1 - p # Self-loop weight
                else:
                    adj[i][j] = p * A[i][j] / degree[i] # Neighbor weight (normalized)
        else: # Handle nodes with degree 0 (isolated nodes)
            adj[i,i] = 1.0 # Self-loop with weight 1

    coo = coo_matrix(adj)
    edge_index_np = np.vstack((coo.row, coo.col))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    edge_weight_np = coo.data
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float)

    return edge_index, edge_weight

# ----------------------------------------------------------------------------
# Model Definitions (Copied from original)
# ----------------------------------------------------------------------------


class Model(nn.Module):
    """CNN part of the architecture."""
    def __init__(self, num_features, num_targets, hidden_size): # num_targets here is effectively output dim before GCN
        super(Model, self).__init__()
        # Precompute layer shapes based on hidden_size (assumed fixed at 4096)
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        # Check if hidden_size is compatible with cha_1
        if hidden_size % cha_1 != 0:
             raise ValueError(f"hidden_size ({hidden_size}) must be divisible by cha_1 ({cha_1})")

        cha_1_reshape = int(hidden_size / cha_1)
        # Check if pooling sizes are valid
        if cha_1_reshape <= 0:
             raise ValueError(f"Resulting reshape dimension ({cha_1_reshape}) is too small.")

        cha_po_1 = int(cha_1_reshape / 2) # Output size for AdaptiveAvgPool1d
        if cha_po_1 <= 0:
             raise ValueError(f"Resulting pooling dimension cha_po_1 ({cha_po_1}) is too small.")

        # MaxPool1d kernel=4, stride=2, padding=1 doubles effective size before halving
        # Input size to MaxPool1d is cha_po_1
        # Output size calculation: floor((input_size + 2*padding - kernel_size)/stride + 1)
        # Output size = floor((cha_po_1 + 2*1 - 4)/2 + 1) = floor((cha_po_1 - 2)/2 + 1)
        pooled_size = np.floor((cha_po_1 + 2*1 - 4)/2 + 1).astype(int)
        if pooled_size <= 0:
            raise ValueError(f"Resulting MaxPool1d output dimension ({pooled_size}) is too small.")

        cha_po_2 = pooled_size * cha_3 # Final flattened size

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2 # This is the output dimension of the CNN part

        # Fixed dropout rates from original code
        dropout_1 = 0.1
        dropout_c1 = 0.1
        dropout_c2 = 0.1
        dropout_c2_1 = 0.3
        dropout_c2_2 = 0.2

        # Batch Norm -> Dropout -> Dense
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout_1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        # Reshape -> Conv1d Block 1
        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(dropout_c1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False), dim=None)

        # Adaptive Pooling
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        # Conv1d Block 2 (Residual connection style)
        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(dropout_c2)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        # Conv1d Block 3 (part of residual path)
        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(dropout_c2_1)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        # Conv1d Block 4 (part of residual path)
        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(dropout_c2_2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None)

        # Max Pooling
        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        # Flatten
        self.flt = nn.Flatten()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06) # CELU activation

        # Reshape for 1D Convolutions
        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        # Conv Block 1
        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x)) # ReLU activation

        x = self.ave_po_c1(x) # Adaptive Average Pooling

        # Conv Block 2 (Main path for residual)
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x)) # ReLU activation
        x_s = x # Store for residual connection

        # Conv Block 3
        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x)) # ReLU activation

        # Conv Block 4
        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x)) # ReLU activation

        # Residual Connection (element-wise multiplication)
        x = x * x_s

        # Max Pooling
        x = self.max_po_c2(x)

        # Flatten for final output
        x = self.flt(x)
        return x # Shape: [batch_size, self.cha_po_2]

class GcnClassifier(nn.Module):
    """GCN part of the architecture."""
    def __init__(self, word_emd_tensor, embedding_dim, hidden_dim, classifier_dim,
                 edge_index, edge_weight, drop_p, drop_p2, negative_slope):
        super().__init__()
        # Register word embeddings as non-trainable buffer (as per user's provided code)
        # If embeddings should be trainable, change to nn.Parameter(word_emd_tensor)
        self.register_buffer('word_emd_buf', word_emd_tensor)
        # Register edge index and weights as non-trainable buffers
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # GCN Layers
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, classifier_dim) # Output matches CNN feature dim

        # Dropout layers
        self.dropout = nn.Dropout(p=drop_p)
        self.dropout2 = nn.Dropout(p=drop_p2) # Kept for consistency, though not used after gcn2
        self.negative_slope = negative_slope # Slope for LeakyReLU

    def forward(self):
        # Use the registered buffer for word embeddings
        x = self.word_emd_buf
        # GCN Layer 1 -> LeakyReLU -> Dropout
        x = F.leaky_relu(self.gcn1(x, self.edge_index, self.edge_weight), negative_slope=self.negative_slope)
        x = self.dropout(x)

        # GCN Layer 2 (Output layer, no activation/dropout applied here as per original structure)
        x = self.gcn2(x, self.edge_index, self.edge_weight)

        return x # Shape: [num_targets, classifier_dim]


class AllModel(nn.Module):
    """Combines CNN feature extractor and GCN classifier."""
    def __init__(self, cnn, gcn):
        super().__init__()
        self.cnn = cnn
        self.gcn = gcn

    def forward(self, x):
        # Get sample features from CNN
        cnn_features = self.cnn(x) # Shape: [batch_size, cnn_output_dim]

        # Get target classifiers (weights) from GCN
        gcn_classifiers = self.gcn() # Shape: [num_targets, gcn_classifier_dim]

        # Matrix multiplication to get predictions
        # Requires gcn_classifier_dim == cnn_output_dim
        predictions = cnn_features @ gcn_classifiers.t() # Shape: [batch_size, num_targets]

        return predictions

# ----------------------------------------------------------------------------
# Loss, Datasets, Training/Validation Functions (Copied from original)
# ----------------------------------------------------------------------------


class SmoothBCEwLogits(_WeightedLoss):
    """Binary Cross Entropy loss with label smoothing and optional pos_weight."""
    def __init__(self, weight=None, reduction='mean', smoothing=0.0, pos_weight=None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight # Sample weights (usually None)
        self.reduction = reduction
        self.pos_weight = pos_weight # Per-class positive weights

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        """Applies label smoothing."""
        assert 0 <= smoothing < 1
        with torch.no_grad():
            # targets = 1 * (1 - smoothing) + 0.5 * smoothing = 1 - 0.5 * smoothing
            # targets = 0 * (1 - smoothing) + 0.5 * smoothing = 0.5 * smoothing
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        # Apply smoothing to targets
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)

        # Calculate BCEWithLogitsLoss, passing pos_weight
        loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight, pos_weight=self.pos_weight)

        # Apply reduction
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        # else: reduction='none', do nothing

        return loss


class TrainDataset(Dataset):
    """Dataset wrapper for training data."""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct

class TestDataset(Dataset):
    """Dataset wrapper for test data."""
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    """Performs one epoch of training."""
    model.train()
    final_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None: # OneCycleLR steps per batch
            scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)
    return final_loss

def valid_fn(model, loss_fn, dataloader, device):
    """Performs one epoch of validation."""
    model.eval()
    final_loss = 0
    valid_preds = []
    with torch.no_grad(): # Disable gradient calculation
        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            final_loss += loss.item()
            # Store predictions (apply sigmoid to get probabilities)
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds) # Combine predictions from all batches
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    """Performs inference on test data."""
    model.eval()
    preds = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data['x'].to(device)
            outputs = model(inputs)
            # Store predictions (apply sigmoid)
            preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds

# ----------------------------------------------------------------------------
# Training Execution Functions (Using hardcoded best parameters)
# ----------------------------------------------------------------------------

def run_training(fold, seed, params, # Pass best params dict
                 train, test_, target_cols, target_nonsc_cols2, # Pass data related vars
                 feature_cols0, feat_dic, word_emd, DEVICE, # Pass pre-loaded word_emd
                 y_train1_full, PRETRAINED_MODEL_DIR): # Pass full targets and pretrain dir
    """
    Runs training and validation for a single fold using fixed hyperparameters.

    Args:
        fold (int): The current fold number.
        seed (int): Random seed for this run.
        params (dict): Dictionary containing the best hyperparameters.
        train (pd.DataFrame): Training dataframe with 'kfold' column.
        test_ (pd.DataFrame): Test features dataframe.
        target_cols (list): List of scored target column names.
        target_nonsc_cols2 (list): List of selected non-scored target column names (not used in training).
        feature_cols0 (list): List of original feature column names (before PCA).
        feat_dic (dict): Dictionary mapping feature types ('gene', 'cell') to column names.
        word_emd (np.ndarray): Pre-loaded word embeddings for GCN.
        DEVICE (torch.device): CPU or CUDA device.
        y_train1_full (np.ndarray): Target values for all non-control samples (used for edge calculation).
        PRETRAINED_MODEL_DIR (str): Path to the directory containing pre-trained CNN models.


    Returns:
        tuple: (oof_preds, test_preds, val_idx)
               - oof_preds (np.ndarray): Out-of-fold predictions for the validation set.
               - test_preds (np.ndarray): Predictions for the test set.
               - val_idx (np.ndarray): Indices of the validation samples.
    """
    seed_everything(seed)
    print(f"Starting Fold: {fold}, Seed: {seed}")

    folds = train.copy() # Use the train df with kfold column
    feature_cols = dp(feature_cols0) # Use deepcopy of original features

    # Get train/validation indices for this fold
    trn_idx = folds[folds['kfold'] != fold].index
    val_idx = folds[folds['kfold'] == fold].index

    # Create train/validation dataframes
    train_df = folds.loc[trn_idx].reset_index(drop=True)
    valid_df = folds.loc[val_idx].reset_index(drop=True)

    # Extract features and targets for this fold
    # Using .copy() to avoid SettingWithCopyWarning later
    x_train_all_feats = train_df[feature_cols].copy()
    y_train = train_df[target_cols].values # Scored targets for training
    x_valid_all_feats = valid_df[feature_cols].copy()
    y_valid = valid_df[target_cols].values # Scored targets for validation
    x_test_all_feats = test_[feature_cols].copy()

    # ------------ Feature Scaling (Quantile Normalization) --------------
    print("  Applying Quantile Normalization...")
    col_num = list(set(feat_dic.get('gene', []) + feat_dic.get('cell', [])) & set(feature_cols))
    col_num.sort()

    # Check if there are numerical columns to normalize
    if col_num:
        x_train_norm, ss = norm_fit(x_train_all_feats[col_num], True, 'quan')
        x_valid_norm = norm_tra(x_valid_all_feats[col_num], ss)
        x_test_norm = norm_tra(x_test_all_feats[col_num], ss)

        # Update dataframes with normalized columns
        x_train_all_feats.loc[:, col_num] = x_train_norm # Use .loc for assignment
        x_valid_all_feats.loc[:, col_num] = x_valid_norm # Use .loc for assignment
        x_test_all_feats.loc[:, col_num] = x_test_norm   # Use .loc for assignment
    else:
        print("  Warning: No numerical columns found for Quantile Normalization.")


    # ------------ PCA Feature Engineering --------------
    print("  Applying PCA...")
    n_comp_gene = 50 # Fixed PCA components for genes
    n_comp_cell = 15 # Fixed PCA components for cells

    def pca_pre(tr, va, te, n_comp, feat_raw, feat_new):
        """Applies PCA and handles potential component mismatch."""
        # Ensure only existing columns are used for PCA
        feat_raw_present = [f for f in feat_raw if f in tr.columns]
        if not feat_raw or not feat_raw_present:
             print(f"  Skipping PCA for {feat_new[0].split('-')[0]}: No raw features provided or found.")
             # Return dataframes of zeros with correct columns
             tr2 = pd.DataFrame(np.zeros((len(tr), n_comp)), columns=feat_new, index=tr.index)
             va2 = pd.DataFrame(np.zeros((len(va), n_comp)), columns=feat_new, index=va.index)
             te2 = pd.DataFrame(np.zeros((len(te), n_comp)), columns=feat_new, index=te.index)
             return tr2, va2, te2

        pca = PCA(n_components=n_comp, random_state=42)
        try:
            tr_pca = pca.fit_transform(tr[feat_raw_present])
            va_pca = pca.transform(va[feat_raw_present])
            te_pca = pca.transform(te[feat_raw_present])
        except ValueError as e:
             print(f"  Error during PCA for {feat_new[0].split('-')[0]}: {e}. Returning zeros.")
             tr2 = pd.DataFrame(np.zeros((len(tr), n_comp)), columns=feat_new, index=tr.index)
             va2 = pd.DataFrame(np.zeros((len(va), n_comp)), columns=feat_new, index=va.index)
             te2 = pd.DataFrame(np.zeros((len(te), n_comp)), columns=feat_new, index=te.index)
             return tr2, va2, te2


        # Pad if n_comp > actual components found (can happen if variance is low)
        if tr_pca.shape[1] < n_comp:
            print(f"  Warning: PCA for {feat_new[0].split('-')[0]} found {tr_pca.shape[1]} components, padding to {n_comp}.")
            pad_width = ((0, 0), (0, n_comp - tr_pca.shape[1]))
            tr_pca = np.pad(tr_pca, pad_width, mode='constant', constant_values=0)
            va_pca = np.pad(va_pca, pad_width, mode='constant', constant_values=0)
            te_pca = np.pad(te_pca, pad_width, mode='constant', constant_values=0)

        tr2 = pd.DataFrame(tr_pca, columns=feat_new, index=tr.index)
        va2 = pd.DataFrame(va_pca, columns=feat_new, index=va.index)
        te2 = pd.DataFrame(te_pca, columns=feat_new, index=te.index)
        return tr2, va2, te2

    # PCA for Gene features
    pca_feat_g = [f'pca_G-{i}' for i in range(n_comp_gene)]
    gene_features = feat_dic.get('gene', [])
    x_tr_g_pca, x_va_g_pca, x_te_g_pca = pca_pre(x_train_all_feats, x_valid_all_feats, x_test_all_feats,
                                                 n_comp_gene, gene_features, pca_feat_g)
    x_train_processed = pd.concat([x_train_all_feats, x_tr_g_pca], axis=1)
    x_valid_processed = pd.concat([x_valid_all_feats, x_va_g_pca], axis=1)
    x_test_processed = pd.concat([x_test_all_feats, x_te_g_pca], axis=1)

    # PCA for Cell features
    pca_feat_c = [f'pca_C-{i}' for i in range(n_comp_cell)]
    cell_features = feat_dic.get('cell', [])
    x_tr_c_pca, x_va_c_pca, x_te_c_pca = pca_pre(x_train_processed, x_valid_processed, x_test_processed,
                                                 n_comp_cell, cell_features, pca_feat_c)
    x_train_processed = pd.concat([x_train_processed, x_tr_c_pca], axis=1)
    x_valid_processed = pd.concat([x_valid_processed, x_va_c_pca], axis=1)
    x_test_processed = pd.concat([x_test_processed, x_te_c_pca], axis=1)

    # Update feature list to include PCA features
    current_feature_cols = feature_cols + pca_feat_g + pca_feat_c
    print(f"  Total features after PCA: {len(current_feature_cols)}")

    # Select final features for model input (as numpy arrays)
    x_train = x_train_processed[current_feature_cols].values
    x_valid = x_valid_processed[current_feature_cols].values
    x_test = x_test_processed[current_feature_cols].values

    # --- HyperParameters (Using Best Found Params) ---
    # These are hardcoded from the provided best trial results
    EPOCHS = 25 # Increased epochs for potentially better convergence in final run
    BATCH_SIZE = 128 # Fixed batch size
    WEIGHT_DECAY_CNN = params['weight_decay_cnn']
    WEIGHT_DECAY_GCN = params['weight_decay_gcn']
    LEARNING_RATE_CNN = params['lr_cnn']
    LEARNING_RATE_GCN = params['lr_gcn']
    PCT_START = params['pct_start']
    DIV_FACTOR = params['div_factor']
    SMOOTHING = params['smoothing']
    HIDDEN_DIM_GCN = params['hidden_dim_gcn']
    DROP_P_GCN = params['drop_p_gcn']
    DROP_P2_GCN = params['drop_p2_gcn'] # Keep param even if layer inactive
    NEG_SLOPE_GCN = params['negative_slope_gcn']
    P_EDGE = params['p_edge']
    TAU_EDGE = params['tau_edge']
    # EMBEDD_DIM is derived from the loaded word_emd.shape[1]

    EARLY_STOPPING_STEPS = 5 # Early stopping patience
    EARLY_STOP = False # Enable early stopping for standard training

    num_features = x_train.shape[1] # Number of input features for CNN
    num_targets = len(target_cols) # Number of output targets
    hidden_size_cnn = 4096 # Fixed hidden size for the CNN's first dense layer

    # --- Calculate Positive Weights for Loss Function ---
    # Based on target frequency in the *current training fold*
    print("  Calculating positive weights for loss...")
    tar_freq = np.maximum(train_df[target_cols].values.sum(axis=0), 1) # Sum of 1s per target, min 1
    # Log scaling: Assign lower weight to more frequent targets
    tar_weight0 = np.log(tar_freq + 100) # Add constant before log
    if len(tar_weight0) > 0 and np.all(tar_weight0 > 0): # Check for non-positive values before division
        tar_weight0_min = np.min(tar_weight0)
        tar_weight = tar_weight0_min / tar_weight0 # Invert: higher freq -> lower weight
        tar_weight = np.clip(tar_weight, 0.1, 10.0) # Clip weights to reasonable range
    else:
        print("  Warning: Could not calculate target weights properly (zero/negative log freq?). Using default weights of 1.")
        tar_weight = np.ones(num_targets) # Default to 1 if calculation fails
    pos_weight = torch.tensor(tar_weight, dtype=torch.float).to(DEVICE)


    # --- Edge Calculation (using full non-control target data) ---
    print("  Calculating graph edges...")
    edge_index, edge_weight = get_edge(y_train1_full, P_EDGE, TAU_EDGE)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    print(f"  Edge index shape: {edge_index.shape}, Edge weight shape: {edge_weight.shape}")


    # --- Model Initialization ---
    print("  Initializing models...")
    # Initialize CNN part
    cnn_model = Model(
            num_features=num_features, # Correct number of input features
            num_targets=2048, # This is the expected *output* dimension of the CNN part
            hidden_size=hidden_size_cnn,
        )

    # --- Load Pre-trained CNN Weights ---
    # Construct path using the provided PRETRAINED_MODEL_DIR
    pretrained_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"FOLD_mod11_{seed}_{fold}_.pth")

    if os.path.exists(pretrained_model_path):
        try:
            print(f"  Attempting to load pre-trained CNN weights from: {pretrained_model_path}")
            # Load weights, mapping to the correct device (e.g., CPU if saved on GPU)
            pretrained_state_dict = torch.load(pretrained_model_path, map_location=DEVICE)

            # Load weights, ignoring missing/unexpected keys (e.g., final classifier layer if it existed)
            # strict=False is important if the saved model differs slightly (e.g., had layers for non-scored targets)
            load_result = cnn_model.load_state_dict(pretrained_state_dict, strict=False)
            print(f"  Loaded pre-trained CNN weights. Load result: {load_result}")

        except FileNotFoundError:
            print(f"  Warning: Pre-trained CNN model file not found at {pretrained_model_path}. CNN will train from scratch.")
        except RuntimeError as e:
            print(f"  Warning: RuntimeError loading pre-trained CNN state_dict (may indicate size mismatch): {e}. CNN training from scratch.")
        except Exception as e:
            print(f"  Warning: Generic error loading pre-trained CNN model: {e}. CNN training from scratch.")
    else:
         print(f"  Warning: Pre-trained CNN model path not found: {pretrained_model_path}. CNN will train from scratch.")


    # Initialize GCN Classifier
    word_emd_tensor = torch.Tensor(word_emd).to(DEVICE) # Use the pre-loaded word_emd
    embedding_dim_gcn = word_emd.shape[1] # Dimension from loaded embedding
    # The GCN classifier output dimension must match the CNN feature dimension
    gcn_classifier_dim = cnn_model.cha_po_2

    print(f"  GCN Embedding Dim: {embedding_dim_gcn}, GCN Hidden Dim: {HIDDEN_DIM_GCN}, GCN Classifier Dim: {gcn_classifier_dim}")

    gcn_model = GcnClassifier(
        word_emd_tensor=word_emd_tensor,
        embedding_dim=embedding_dim_gcn,
        hidden_dim=HIDDEN_DIM_GCN,
        classifier_dim=gcn_classifier_dim, # Match CNN output dim
        edge_index=edge_index,
        edge_weight=edge_weight,
        drop_p=DROP_P_GCN,
        drop_p2=DROP_P2_GCN,
        negative_slope=NEG_SLOPE_GCN
    )

    # Combine CNN and GCN
    model = AllModel(cnn_model, gcn_model)
    model.to(DEVICE)

    # --- Optimizer and Scheduler ---
    print("  Setting up optimizer and scheduler...")
    # Define parameter groups for different learning rates/weight decays
    optimizer_grouped_parameters = [
        {'params': model.cnn.parameters(), 'lr': LEARNING_RATE_CNN, 'weight_decay': WEIGHT_DECAY_CNN},
        {'params': model.gcn.parameters(), 'lr': LEARNING_RATE_GCN, 'weight_decay': WEIGHT_DECAY_GCN}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # DataLoaders (using num_workers=0 for safety in notebooks)
    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) # drop_last=True can help with BatchNorm stability if last batch is small
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Scheduler (OneCycleLR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              pct_start=PCT_START,
                                              div_factor=DIV_FACTOR,
                                              max_lr=[LEARNING_RATE_CNN, LEARNING_RATE_GCN], # Max LR for each param group
                                              epochs=EPOCHS,
                                              steps_per_epoch=len(trainloader)) # Use len(trainloader)

    # --- Loss Function ---
    # Use smoothing and pos_weight for training loss
    loss_tr = SmoothBCEwLogits(smoothing=SMOOTHING, pos_weight=pos_weight).to(DEVICE)
    # Use standard BCE for validation loss
    loss_va = nn.BCEWithLogitsLoss().to(DEVICE)

    # --- Training Loop ---
    print(f"  Starting training loop for {EPOCHS} epochs...")
    oof = np.zeros((len(valid_df), num_targets)) # Initialize OOF predictions for this fold
    best_loss = np.inf
    early_step = 0
    best_epoch = -1

    # Temporary model save path
    model_save_path = f"model_fold{fold}_seed{seed}_best.pth"

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        current_lr_cnn = optimizer.param_groups[0]['lr']
        current_lr_gcn = optimizer.param_groups[1]['lr']
        print(f"    Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, LR_cnn: {current_lr_cnn:.4e}, LR_gcn: {current_lr_gcn:.4e}")

        # Check for improvement for early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            oof = valid_preds # Store best OOF predictions
            best_epoch = epoch
            # Save the best model state for this fold
            torch.save(model.state_dict(), model_save_path)
            # print(f"      Validation loss improved to {best_loss:.6f}. Model state saved.")
            early_step = 0 # Reset early stopping counter
        elif EARLY_STOP:
            early_step += 1
            if early_step >= EARLY_STOPPING_STEPS:
                print(f"    Early stopping triggered at epoch {epoch+1} (best loss: {best_loss:.6f} at epoch {best_epoch+1})")
                break # Exit training loop

    # --- Prediction on Test Set ---
    print("  Generating test predictions using best model...")
    # Load the best model state saved during training
    if os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print(f"  Loaded best model state from epoch {best_epoch+1} (Valid Loss: {best_loss:.6f}).")
        except Exception as e:
            print(f"  Warning: Could not load best model state from {model_save_path}: {e}. Using model from last epoch.")
    else:
         print(f"  Warning: Best model checkpoint {model_save_path} not found. Using model from last epoch for test prediction.")

    testdataset = TestDataset(x_test)
    testloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    predictions = inference_fn(model, testloader, DEVICE)

    # # Cleanup saved model file
    # if os.path.exists(model_save_path):
    #     os.remove(model_save_path)

    print(f"Finished Fold: {fold}, Seed: {seed}, Best Valid Loss: {best_loss:.6f} at Epoch {best_epoch+1}")
    return oof, predictions, val_idx # Return validation indices along with predictions

def run_k_fold(NFOLDS, seed, params, # Pass best params
               train, test, target_cols, target_nonsc_cols2, # Pass data
               feature_cols0, feat_dic, word_emd, DEVICE, # Pass loaded word_emd
               y_train1_full, PRETRAINED_MODEL_DIR): # Pass full targets and pretrain dir
    """
    Runs k-fold cross-validation for a given seed and parameters.

    Args:
        NFOLDS (int): Number of folds.
        seed (int): Random seed.
        params (dict): Dictionary of best hyperparameters.
        train (pd.DataFrame): Training dataframe with 'kfold' column.
        test (pd.DataFrame): Test features dataframe.
        target_cols (list): List of scored target column names.
        target_nonsc_cols2 (list): List of selected non-scored target column names.
        feature_cols0 (list): List of original feature column names.
        feat_dic (dict): Dictionary mapping feature types to names.
        word_emd (np.ndarray): Pre-loaded word embeddings.
        DEVICE (torch.device): CPU or CUDA device.
        y_train1_full (np.ndarray): Target values for all non-control samples.
        PRETRAINED_MODEL_DIR (str): Path to the directory containing pre-trained CNN models.

    Returns:
        tuple: (oof, predictions)
               - oof (np.ndarray): Aggregated out-of-fold predictions for the entire training set for this seed.
               - predictions (np.ndarray): Averaged predictions for the test set across all folds for this seed.
    """
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    # Ensure train has the 'kfold' column before starting folds
    if 'kfold' not in train.columns:
        raise ValueError("The 'train' dataframe must have the 'kfold' column for cross-validation.")

    for fold in range(NFOLDS):
        print(f"\n===== Fold {fold+1} / {NFOLDS} | Seed {seed} =====")
        oof_, pred_, val_idx = run_training(fold, seed, params,
                                            train, test, target_cols, target_nonsc_cols2,
                                            feature_cols0, feat_dic, word_emd, DEVICE,
                                            y_train1_full, PRETRAINED_MODEL_DIR) # Pass pretrain dir

        # --- Validation Check ---
        if val_idx.max() >= len(oof):
             raise IndexError(f"Fold {fold}, Seed {seed}: Validation indices {val_idx.max()} out of bounds for OOF array size {len(oof)}")
        if len(val_idx) != len(oof_):
             raise ValueError(f"Fold {fold}, Seed {seed}: Length mismatch - validation indices ({len(val_idx)}) vs OOF predictions ({len(oof_)})")
        # --- End Validation Check ---

        oof[val_idx] = oof_ # Assign OOF predictions to the correct rows using validation indices
        predictions += pred_ / NFOLDS # Accumulate test predictions, average by number of folds
    
    oof_filename = f'/kaggle/working/oof_seed{seed}.npy'
    np.save(oof_filename, oof)
    print(f'Saved OOF predictions for seed {seed} to {oof_filename}')
    return oof, predictions

# ----------------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------------



# --- Configuration ---
SEED_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  
#SEED_LIST=[0]
NFOLDS = 5                  # Use 5 folds for cross-validation
KFOLD_SEED = 42             # Seed for creating the KFold splits (consistent across runs)

print(f"--- Configuration ---")
print(f"Seeds: {SEED_LIST}")
print(f"Number of Folds: {NFOLDS}")
print(f"KFold Seed: {KFOLD_SEED}")

# --- Paths (ADJUST BASED ON YOUR ENVIRONMENT: Kaggle vs. Local) ---
if os.path.exists('/kaggle/input'):
     print("Detected Kaggle environment.")
     INPUT_DIR = '/kaggle/input/lish-moa/'
     EMBEDDING_DIR = '/kaggle/input/embedding-666' # Directory containing embedding files (e.g., embedding3)
     # Path to directory containing pre-trained CNN models (IMPORTANT!)
     # Using the exact path provided by the user
     PRETRAINED_MODEL_DIR = '/kaggle/input/gcn-related/results/results (1)'
else:
     print("Assuming local environment.")
     # Adjust these paths for your local setup
     INPUT_DIR = './lish-moa/'
     EMBEDDING_DIR = './embedding3/' # Make sure this exists locally
     PRETRAINED_MODEL_DIR = './results/results (1)/' # Make sure this exists locally
     # Create directories if they don't exist for local run
     os.makedirs(INPUT_DIR, exist_ok=True)
     os.makedirs(EMBEDDING_DIR, exist_ok=True)
     os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
     # TODO: Add logic here to download data or create dummy files if needed for local testing.

print(f"Using Input Directory: {INPUT_DIR}")
print(f"Using Embedding Directory: {EMBEDDING_DIR}")
print(f"Using Pre-trained Model Directory: {PRETRAINED_MODEL_DIR}")
print("-" * 20)


# --- Best Hyperparameters (Hardcoded from Optuna study) ---
best_params = {
    'p_edge': 0.1,
    'tau_edge': 0.55,
    'hidden_dim_gcn': 256,
    'negative_slope_gcn': 0.07426199834556609,
    'lr_cnn': 0.00010668196415401104,
    'weight_decay_cnn': 2.5486706762832486e-05,
    'weight_decay_gcn': 2.8811942386039456e-06,
    'lr_gcn': 0.002975845239015901,
    'drop_p_gcn': 0.25106267877438987,
    'drop_p2_gcn': 0.368935002894421,
    'smoothing': 0.00024852192187681233,
    'pct_start': 0.24079542373592375,
    'div_factor': 21.91052238898028,
    'embedd': 512 # This determines which embedding file to load
}
print("\n--- Using Best Hyperparameters ---")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print("-" * 32)


# --- Device Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {DEVICE}")

# --- Load Word Embeddings (based on best_params['embedd']) ---
best_embed_dim = best_params['embedd']
embedding_filename = f'embeddings_{best_embed_dim}.txt'
embedding_path = os.path.join(EMBEDDING_DIR, embedding_filename)
try:
    print(f"Loading embeddings (dim={best_embed_dim}) from: {embedding_path}")
    # Assuming space-separated values, no header. Adjust read_csv if format differs.
    word_emd = pd.read_csv(embedding_path, sep=' ', header=None).values
    print(f"Embeddings loaded successfully, shape: {word_emd.shape}")
    # Validate loaded dimension matches expected dimension
    if word_emd.shape[1] != best_embed_dim:
        print(f"  FATAL ERROR: Loaded embedding dimension ({word_emd.shape[1]}) differs from parameter ({best_embed_dim}).")
        sys.exit(1)
except FileNotFoundError:
    print(f"FATAL ERROR: Required embedding file not found at {embedding_path}")
    print("Please ensure the embedding file exists in the specified directory.")
    sys.exit(1) # Exit if essential embedding file is missing
except Exception as e:
    print(f"FATAL ERROR loading embedding file {embedding_path}: {e}")
    sys.exit(1)


# --- Data Loading and Initial Preprocessing ---
print("\nLoading data...")
try:
    train_features = pd.read_csv(os.path.join(INPUT_DIR, 'train_features.csv'))
    train_targets_scored = pd.read_csv(os.path.join(INPUT_DIR, 'train_targets_scored.csv'))
    train_targets_nonscored = pd.read_csv(os.path.join(INPUT_DIR, 'train_targets_nonscored.csv'))
    test_features = pd.read_csv(os.path.join(INPUT_DIR, 'test_features.csv'))
    sample_submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
    train_drug = pd.read_csv(os.path.join(INPUT_DIR, 'train_drug.csv'))
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"FATAL ERROR loading data file: {e}")
    print(f"Please ensure all input files are present in: {INPUT_DIR}")
    sys.exit(1)

# Extract target column names
target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()
print(f"Number of scored targets: {len(target_cols)}")
print(f"Number of non-scored targets (initial): {len(target_nonsc_cols)}")

# --- Feature Engineering & Selection (Consistent with Optuna setup) ---
print("\nPerforming feature engineering...")
feat_dic = {}
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS
print(f"Found {len(GENES)} gene features and {len(CELLS)} cell features.")

# Select informative non-scored targets based on correlation (logic from Optuna setup)
nonctr_mask = train_features['cp_type'] != 'ctl_vehicle'
nonctr_ids = train_features.loc[nonctr_mask, 'sig_id'].unique()
train_targets_scored_nonctl = train_targets_scored[train_targets_scored['sig_id'].isin(nonctr_ids)].set_index('sig_id')
train_targets_nonscored_nonctl = train_targets_nonscored[train_targets_nonscored['sig_id'].isin(nonctr_ids)].set_index('sig_id')
# Align indices before correlation
common_ids = train_targets_scored_nonctl.index.intersection(train_targets_nonscored_nonctl.index)
y_train_nonctl_scored = train_targets_scored_nonctl.loc[common_ids].values
y_train_nonctl_nonscored = train_targets_nonscored_nonctl.loc[common_ids].values

target_nonsc_cols2 = [] # Initialize
if y_train_nonctl_scored.shape[0] > 0 and y_train_nonctl_nonscored.shape[0] > 0:
    if np.any(np.std(y_train_nonctl_scored, axis=0) == 0) or np.any(np.std(y_train_nonctl_nonscored, axis=0) == 0):
         print("  Warning: Zero variance detected in some non-control targets. Correlation might be unreliable.")
    try:
         mat_cor = np.corrcoef(y_train_nonctl_scored.T, y_train_nonctl_nonscored.T)
         if not np.isnan(mat_cor).all():
              num_scored = y_train_nonctl_scored.shape[1]
              mat_cor2 = pd.DataFrame(mat_cor[num_scored:, :num_scored], index=target_nonsc_cols, columns=target_cols)
              mat_cor2 = mat_cor2.dropna(axis=0, how='all').dropna(axis=1, how='all')
              if not mat_cor2.empty:
                  mat_cor2 = mat_cor2.fillna(0)
                  mat_cor2_max = mat_cor2.abs().max(axis=1)
                  q_n_cut = 0.9
                  if len(mat_cor2_max) > 0:
                      quantile_value = np.quantile(mat_cor2_max, q_n_cut)
                      target_nonsc_cols2 = mat_cor2_max[mat_cor2_max > quantile_value].index.tolist()
                  else: print("  Warning: No non-scored targets remaining after correlation processing.")
              else: print("  Warning: Correlation matrix became empty after removing NaN targets.")
         else: print("  Warning: Correlation matrix calculation resulted in all NaNs.")
    except Exception as e: print(f"  Error during non-scored target correlation analysis: {e}")
else: print("  Warning: No non-control samples found for non-scored target correlation analysis.")
print(f"Selected {len(target_nonsc_cols2)} non-scored targets (not used in final model loss).")

# Apply sample-wise quantile normalization/scaling
print("Applying sample-wise normalization...")
for df in [train_features, test_features]:
    g_cols = [col for col in feat_dic.get('gene', []) if col in df.columns]
    c_cols = [col for col in feat_dic.get('cell', []) if col in df.columns]
    if g_cols:
        q25_g = df[g_cols].quantile(0.25, axis=1)
        q75_g = df[g_cols].quantile(0.75, axis=1)
        df[g_cols] = df[g_cols].sub((q25_g + q75_g) / 2, axis=0)
    if c_cols:
        q25_c = df[c_cols].quantile(0.25, axis=1)
        q72_c = df[c_cols].quantile(0.72, axis=1) # Note 0.72
        df[c_cols] = df[c_cols].sub((q25_c + q72_c) / 2, axis=0)
        scale_c = df[c_cols].abs().quantile(0.75, axis=1) + 4.0
        scale_c[scale_c == 0] = 1e-6 # Avoid division by zero
        df[c_cols] = df[c_cols].div(scale_c, axis=0)

# Merge targets and remove control samples
print("Merging data and removing controls...")
train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
print(f"Training samples after removing controls: {len(train)}")
print(f"Test samples after removing controls: {len(test)}")

# Define y_train1_full: Targets for *all* non-control samples (used for consistent edge calculation)
y_train1_full = train[target_cols].values.astype(float)
print(f"Shape of y_train1_full (for edge calculation): {y_train1_full.shape}")

# Prepare final training and test dataframes (dropping cp_type)
target = train[['sig_id'] + target_cols]
train0 = train.drop(['cp_type','sig_id'], axis=1) # Drop sig_id here as it's added back with kfold merge
test_processed = test.drop(['cp_type'], axis=1).copy() # Keep sig_id in test_processed for submission merge

# Add drug_id for KFold splitting
train_sig_ids = train['sig_id'].tolist()
train_drug_filtered = train_drug[train_drug['sig_id'].isin(train_sig_ids)].copy()
target_with_drug = target.merge(train_drug_filtered, on='sig_id', how='left')
target_with_drug['drug_id'] = target_with_drug['drug_id'].fillna('UNKNOWN_DRUG')

# --- KFold Setup (Leave One Drug Out + Stratified Split) ---
print(f"\nSetting up {NFOLDS} Folds using seed {KFOLD_SEED}...")
target_with_drug['kfold'] = -1
drug_counts = target_with_drug['drug_id'].value_counts()
stratify_threshold = NFOLDS # Use NFOLDS as the threshold
drugs_leave_out = drug_counts[drug_counts <= stratify_threshold].index
drugs_stratify = drug_counts[drug_counts > stratify_threshold].index

# Assign folds for leave-drug-out drugs
fold_map_leave_out = {drug: i % NFOLDS for i, drug in enumerate(drugs_leave_out)}
target_with_drug['kfold'] = target_with_drug['drug_id'].map(fold_map_leave_out).fillna(target_with_drug['kfold'])

# Assign folds for stratify drugs
stratify_indices = target_with_drug[target_with_drug['drug_id'].isin(drugs_stratify)].index
if len(stratify_indices) > 0:
    mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=KFOLD_SEED)
    stratify_targets = target_with_drug.loc[stratify_indices, target_cols].values
    for fold_num, (_, val_indices_skf) in enumerate(mskf.split(stratify_indices, stratify_targets)):
        original_val_indices = stratify_indices[val_indices_skf]
        target_with_drug.loc[original_val_indices, 'kfold'] = fold_num

unassigned_count = (target_with_drug['kfold'] == -1).sum()
if unassigned_count > 0:
    print(f"  Warning: {unassigned_count} samples remained unassigned. Assigning them to fold 0.")
    target_with_drug.loc[target_with_drug['kfold'] == -1, 'kfold'] = 0
target_with_drug.kfold = target_with_drug.kfold.astype(int)

# Merge fold information back into the main training dataframe (train0 lacks sig_id now)
# Need to merge based on original index or add sig_id back temporarily
train_with_folds = train0.copy()
train_with_folds['sig_id'] = train['sig_id'] # Add sig_id back for merge
train_with_folds = train_with_folds.merge(target_with_drug[['sig_id', 'kfold']], on='sig_id', how='left')
train_with_folds = train_with_folds.drop('sig_id', axis=1) # Drop sig_id again after merge

if train_with_folds['kfold'].isna().any():
     print("FATAL ERROR: kfold column contains NaNs after merging.")
     sys.exit(1)
print("Fold distribution:")
print(train_with_folds['kfold'].value_counts().sort_index())


# --- Define Feature List ---
feature_cols = []
if 'gene' in feat_dic: feature_cols += feat_dic['gene']
if 'cell' in feat_dic: feature_cols += feat_dic['cell']
print(f"Total base features (before PCA): {len(feature_cols)}")
feature_cols0 = dp(feature_cols) # Store original list

# --- Run Training Across Seeds ---
print(f"\n--- Starting {NFOLDS}-Fold Training across {len(SEED_LIST)} seeds ---")
overall_oof = np.zeros((len(train_with_folds), len(target_cols)))
overall_predictions = np.zeros((len(test_processed), len(target_cols))) # test_processed still has controls removed

for seed in SEED_LIST:
    print(f"\n===== Running Training for Seed: {seed} =====")
    # Pass PRETRAINED_MODEL_DIR to run_k_fold
    oof_seed, pred_seed = run_k_fold(NFOLDS, seed, best_params,
                                     train_with_folds, test_processed, target_cols, target_nonsc_cols2,
                                     feature_cols0, feat_dic, word_emd, DEVICE,
                                     y_train1_full, PRETRAINED_MODEL_DIR)

    # Accumulate results across seeds
    overall_oof += oof_seed # OOF preds are collected fold-wise, just sum per seed
    overall_predictions += pred_seed # Test preds are averaged over folds in run_k_fold, just sum per seed

# Average the accumulated OOF and predictions by the number of seeds
overall_oof /= len(SEED_LIST)
overall_predictions /= len(SEED_LIST)
print("-" * 60)

# --- Calculate Final OOF Score ---
print("\nCalculating Final OOF LogLoss...")
final_y_true = train_with_folds[target_cols].values
final_losses = []
oof_scores = {}
for i, target_name in enumerate(target_cols):
    oof_clipped = np.clip(overall_oof[:, i], 1e-15, 1 - 1e-15)
    try:
        loss = log_loss(final_y_true[:, i], oof_clipped)
        final_losses.append(loss)
        oof_scores[target_name] = loss
    except ValueError as e:
        print(f"  Warning: ValueError calculating log_loss for target {i} ({target_name}): {e}. Skipping.")
        final_losses.append(10.0)
        oof_scores[target_name] = 10.0

if not final_losses:
    print("Error: No valid OOF losses could be calculated.")
    final_mean_logloss = float('inf')
else:
    final_mean_logloss = np.mean(final_losses)
    print(f"\n>>> Final Mean OOF LogLoss across {len(SEED_LIST)} seeds and {NFOLDS} folds: {final_mean_logloss:.6f} <<<")
    # Compare with target score
    target_score = 0.013785
    print(f"    Target Score from Optuna: {target_score:.6f}")
    print(f"    Difference: {final_mean_logloss - target_score:+.6f}")

# Optional: Save detailed OOF scores
# oof_scores_df = pd.DataFrame.from_dict(oof_scores, orient='index', columns=['LogLoss'])
# oof_scores_df.to_csv('final_oof_scores_per_target.csv')
# print("Detailed OOF scores saved to final_oof_scores_per_target.csv")

# Optional: Save final OOF predictions
# oof_df = pd.DataFrame(overall_oof, columns=target_cols)
# oof_df['sig_id'] = train['sig_id'] # Add sig_id back from original non-control train df
# oof_df = oof_df[['sig_id'] + target_cols]
# oof_df.to_csv('final_oof_predictions.csv', index=False)
# print("Final OOF predictions saved to final_oof_predictions.csv")


# --- Prepare Submission File ---
print("\nGenerating submission file...")
sub = sample_submission.copy()

# Create a dataframe with predictions for the non-control test samples
# test_processed includes sig_id, test does not necessarily if controls were dropped earlier
pred_df = pd.DataFrame(overall_predictions, columns=target_cols)
pred_df['sig_id'] = test_processed['sig_id'].values # Add sig_id from the test set *after* controls were removed

# Merge predictions with the sample submission format
sub = sub[['sig_id']].merge(pred_df, on='sig_id', how='left')

# Fill predictions for control samples (which were removed during processing) with 0
control_mask_original = test_features['cp_type'] == 'ctl_vehicle'
control_sig_ids_original = test_features.loc[control_mask_original, 'sig_id'].tolist()
# Set target columns to 0 for these control sig_ids in the submission file
sub.loc[sub['sig_id'].isin(control_sig_ids_original), target_cols] = 0

# Final check for any remaining NaNs (e.g., if merge failed unexpectedly)
if sub[target_cols].isnull().values.any():
    nan_count = sub[target_cols].isnull().sum().sum()
    print(f"Warning: {nan_count} NaNs found in submission file target columns after merging/filling controls. Filling remaining NaNs with 0.")
    sub.fillna(0, inplace=True)

# Ensure column order matches sample submission
sub = sub[sample_submission.columns]

# Save the submission file
final_oof_filename = 'GCN.npy'
np.save(final_oof_filename, overall_oof)
print(f'Final OOF saved to {final_oof_filename}')
#submission_filename = 'submission.csv'
#sub.to_csv(submission_filename, index=False)
#print(f"Submission file saved to '{submission_filename}'")

print("\nScript finished.")


test_counts = pd.read_csv('../input/moa-test-target-means/target_counts.csv')
test_counts = test_counts.sort_values('train_ct').reset_index(drop=True)


#sub = pd.read_csv('submission_tabnet_01830.csv')
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')


# CONVERT PROBABILITIES TO ODDS, APPLY MULTIPLIER, CONVERT BACK TO PROBABILITIES
def scale(x,k):
    x = x.copy()
    idx = np.where(x!=1)[0]
    y = k * x[idx] / (1-x[idx])
    x[idx] =  y/(1+y)
    return x


public = pd.read_csv('../input/moa-test-target-means/sample_public_submission.csv').sig_id.values
SZ = 3982; SZ2 = 3982*3

for c in test_counts.target.values:
    
    m2 = sub.loc[sub.sig_id.isin(public),c].mean()
    m3 = sub.loc[~sub.sig_id.isin(public),c].mean()
    m = test_counts.loc[test_counts.target==c,'public_ct'].values[0]/SZ /m2
    sub.loc[sub.sig_id.isin(public),c] = scale(sub.loc[sub.sig_id.isin(public),c].values,m)
    
    if len(sub)>len(public):
        
        m = test_counts.loc[test_counts.target==c,'private_ct'].values[0]/SZ2 /m3
        sub.loc[~sub.sig_id.isin(public),c] = scale(sub.loc[~sub.sig_id.isin(public),c].values,m)
                    
print('Updated',len(test_counts),'targets')


# # SAVE SUBMISSION FILE
# s = sub.loc[(test.cp_type=='ctl_vehicle').values].mean().sum()
# print('Control rows in submission have mean = %i'%s)

sub.to_csv('submission.csv',index=False)
sub.head()

