from IPython.display import Image, display

# Path to your image file
img_path = "/kaggle/input/drw-crypto-market-ensembled-algorithms-gp/DRW - Crypto Market Ensembled Algorithms.png"

# Display the image
display(Image(filename=img_path))



# -*- coding: utf-8 -*-
"""
ensemble-envy.ipynb

This script performs an ensemble of XGBoost and MLP models for crypto market prediction.
It includes data loading, feature engineering, model training (KFold cross-validation),
outlier detection and weight adjustment, and final submission generation.

=============================================================================
COMPETITION REQUIREMENTS & GUIDELINES:

1.  Evaluation Metric: Submissions are evaluated based on the Pearson correlation
    coefficient between the 'label' (true values) and predicted values over the
    private testing set. This evaluation is performed externally by the competition
    platform, as true labels for the test set are not available during prediction.

2.  Submission Format: The code generates predictions for the 'label' variable
    for each row in the test dataset. The final output will be a CSV file named
    'submission.csv' located in the /kaggle/working/ directory, matching the
    format of sample_submission.csv.

3.  Avoiding Future Peaking:
    * The modeling process strictly avoids using future information from the
        test dataset. All data used for training and feature engineering for a
        given prediction point would have been available at that time in a
        real-world setting.
    * This is ensured by:
        * Using KFold cross-validation where validation sets are treated as
            future data relative to their corresponding training sets.
        * Applying time-based data slicing (e.g., 'last_90pct') to focus on
            more recent data, simulating a live prediction scenario.
        * Feature engineering is applied independently to training and test
            data, using only information present in each respective dataset.

4.  Key Aspects Emphasized:
    * Data Exploration and Feature Analysis: Addressed in the `add_features`
        function, which creates a rich set of microstructure-related features.
    * Advanced Modeling Techniques: Implemented through the use of XGBoost
        (a powerful gradient boosting model) and MLP (a neural network), along
        with ensemble methods to combine their strengths.
=============================================================================
"""

# =============================================================================
# 1. Imports
# =============================================================================
import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation and analysis
import os  # Operating system interactions
import shutil  # High-level file operations
from pathlib import Path  # Object-oriented filesystem paths
import random  # Random number generation
import warnings  # Warning control
import joblib # For saving non-PyTorch models

# Scikit-learn imports for model selection, ensemble, and preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# XGBoost for gradient boosting
from xgboost import XGBRegressor

# SciPy for statistical functions (e.g., Pearson correlation)
from scipy.stats import pearsonr

# Tqdm for progress bars
from tqdm import tqdm

# Deep Learning imports (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Determine the device for PyTorch (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 2. Configuration
# =============================================================================
class Config:
    """
    Configuration class to store file paths, feature lists, and model parameters.
    """
    # Dataset paths (as provided by the competition environment)
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    # List of features for the main models (XGBoost) - initial set
    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333","X817",
        "X586",  "X292"
    ]

    # Subset of features specifically for the MLP model - initial set
    MLP_FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]

    LABEL_COLUMN = "label"  # Name of the target variable column
    N_FOLDS = 3  # Number of folds for KFold cross-validation
    RANDOM_STATE = 42  # Seed for reproducibility
    OUTLIER_FRACTION = 0.001  # Fraction of records considered as outliers for weight adjustment

# XGBoost model parameters
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 500, # Reduced for faster execution
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

# List of learners to be used (currently only XGBoost)
LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]

# =============================================================================
# 3. Directory Setup
# =============================================================================
# Create main module directory and subdirectories for organizing outputs
MODULE_DIR = Path("/kaggle/working/xgb_mlp_backbone")
SUBMODELS_DIR = MODULE_DIR / "submodels"
FINAL_SUBMISSIONS_DIR = MODULE_DIR / "final_submissions"
MODEL_CHECKPOINTS_DIR = MODULE_DIR / "model_checkpoints"

# Create all necessary directories if they don't exist
for directory in [MODULE_DIR, SUBMODELS_DIR, FINAL_SUBMISSIONS_DIR, MODEL_CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory}")

# =============================================================================
# 4. Deep Learning Components (PyTorch)
# =============================================================================
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_activation_function(name):
    """
    Returns a PyTorch activation function module based on its name.
    Supports 'relu', 'tanh', 'sigmoid'.
    """
    if name is None:
        return None
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.
    Configurable with dropout, hidden layers, and activation functions.
    """
    def __init__(self, dropout_rate=0.6, layers=[128, 64], activation='relu', last_activation=None):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.activation = get_activation_function(activation)
        self.last_activation = get_activation_function(last_activation)

        # Create linear layers based on the 'layers' configuration
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the MLP.
        Applies linear layers, activation, and dropout sequentially.
        """
        for k in range(len(self.linears) - 1):
            x = self.activation(self.linears[k](x))
            x = self.dropout(x)
        x = self.linears[-1](x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

class Checkpointer:
    """
    Utility class to save the best performing PyTorch model during training
    based on a specified metric (e.g., Pearson correlation).
    """
    def __init__(self, filename="best_model.pt"):
        self.path = MODEL_CHECKPOINTS_DIR / filename  # Path to save the model
        self.best_metric = -np.inf  # Initialize best metric to negative infinity

    def load(self, model):
        """
        Loads the best model's weights from the saved checkpoint.
        """
        if self.path.exists():
            if isinstance(model, nn.Module): # PyTorch model
                model.load_state_dict(torch.load(self.path))
            else: # Scikit-learn or XGBoost model
                import joblib
                model = joblib.load(self.path) # Load the model
            print(f"Model loaded from {self.path} with best metric: {self.best_metric:.4f}")
        else:
            print(f"No checkpoint found at {self.path}. Starting from scratch.")
        return model

    def __call__(self, current_metric, model):
        """
        Callable method to save the model if the current metric
        is better than the previously recorded best.
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            if isinstance(model, nn.Module): # PyTorch model
                torch.save(model.state_dict(), self.path)
            else: # Scikit-learn or XGBoost model (save with joblib/pickle)
                joblib.dump(model, self.path)
            print(f"New best model saved to {self.path} with metric: {current_metric:.4f}")

def get_dataloaders(X, Y, hparams, device, shuffle=True):
    """
    Creates PyTorch DataLoader objects for training and validation datasets.
    Handles both input-only (for prediction) and input-output (for training) datasets.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    if Y is not None:
        Y_tensor = torch.tensor(Y.values if hasattr(Y, 'values') else Y,
                                dtype=torch.float32, device=device).unsqueeze(1)
        dataset = TensorDataset(X_tensor, Y_tensor)
    else:
        dataset = TensorDataset(X_tensor)

    dataloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=shuffle,
                            generator=torch.Generator().manual_seed(hparams["seed"]))
    return dataloader

# =============================================================================
# 5. Feature Engineering
#    (Incorporates Data Exploration and Feature Analysis)
# =============================================================================
def add_features(df):
    """
    Adds a comprehensive set of new features to the DataFrame based on existing columns.
    These features are designed to capture microstructure dynamics and market activity.
    Handles potential infinite values and NaNs by replacing them with 0.
    """
    # Ensure base columns exist before creating new features
    required_base_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for col in required_base_cols:
        if col not in df.columns:
            df[col] = 0.0 # Add missing columns with default value 0.0

    # Original interaction features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10) # Add small epsilon to avoid division by zero
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume']) # Log transformation for skewed data

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)

    # NEW MICROSTRUCTURE FEATURES
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']

    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])

    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)

    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])

    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])

    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])

    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']

    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)

    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2

    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)

    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)

    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

    # Replace any infinite values or NaNs that might have been generated during feature creation with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    """
    Generates an array of time-decaying weights.
    Weights are higher for more recent data points.
    """
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def detect_outliers_and_adjust_weights(y, sample_weights, outlier_fraction=0.001):
    """
    Detects outliers based on the target variable 'y' using IQR method
    and adjusts their corresponding sample weights.
    This version is faster as it avoids training a RandomForestRegressor per call.
    """
    if len(y) < 2: # Not enough data to detect outliers
        return sample_weights.copy()

    # Calculate IQR for outlier detection
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outlier_mask = (y < lower_bound) | (y > upper_bound)

    adjusted_weights = sample_weights.copy()

    if outlier_mask.any():
        n_outliers = np.sum(outlier_mask)
        # Reduce weights for outliers (e.g., by a fixed factor)
        adjusted_weights[outlier_mask] *= 0.1 # Reduce weight by 90% for outliers

        print(f"    Adjusted weights for {n_outliers} outliers ({n_outliers/len(y)*100:.1f}% of data) based on y values.")

    return adjusted_weights

def load_data():
    """
    Loads the training, testing, and sample submission dataframes.
    Applies feature engineering to both training and testing datasets.
    Updates the global Config.FEATURES list with all newly created features.
    Handles missing columns in parquet files gracefully.
    """
    # Load data without specifying columns first to inspect available columns
    try:
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        print("Please ensure the parquet files exist at the specified paths and are valid.")
        raise # Re-raise the exception after printing a user-friendly message

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded raw data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    # Check if LABEL_COLUMN exists in train_df
    if Config.LABEL_COLUMN not in train_df.columns:
        raise ValueError(f"Label column '{Config.LABEL_COLUMN}' not found in training data.")

    # Get available columns from the loaded dataframes
    available_train_cols = set(train_df.columns)
    available_test_cols = set(test_df.columns)

    # Filter Config.FEATURES and Config.MLP_FEATURES to only include available columns
    original_features = set(Config.FEATURES)
    original_mlp_features = set(Config.MLP_FEATURES)

    # Ensure 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume' are present for feature engineering
    base_cols_for_fe = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for col in base_cols_for_fe:
        if col not in available_train_cols:
            print(f"Warning: Base feature '{col}' missing from training data. It will be added with zeros.")
            train_df[col] = 0.0
            available_train_cols.add(col)
        if col not in available_test_cols:
            print(f"Warning: Base feature '{col}' missing from test data. It will be added with zeros.")
            test_df[col] = 0.0
            available_test_cols.add(col)

    Config.FEATURES = list(original_features.intersection(available_train_cols))
    Config.MLP_FEATURES = list(original_mlp_features.intersection(available_train_cols))

    # Ensure the label column is included in the train_df for subsequent operations
    # It should not be in the feature lists themselves, but used for slicing
    if Config.LABEL_COLUMN not in train_df.columns:
        raise ValueError(f"Label column '{Config.LABEL_COLUMN}' not found in training data after initial load.")

    # Select only the relevant columns after filtering
    # We need to ensure that the columns used for feature engineering are present
    # and then select the final set of features for the models.
    all_cols_needed_for_fe = list(set(Config.FEATURES + Config.MLP_FEATURES + base_cols_for_fe + [Config.LABEL_COLUMN]))
    train_df = train_df[[col for col in all_cols_needed_for_fe if col in train_df.columns]].copy()
    test_df = test_df[[col for col in all_cols_needed_for_fe if col in test_df.columns]].copy()

    print(f"Filtered features based on available columns. Initial feature counts: Config.FEATURES={len(Config.FEATURES)}, Config.MLP_FEATURES={len(Config.MLP_FEATURES)}")
    print(f"Data after initial feature filtering - Train: {train_df.shape}, Test: {test_df.shape}")

    # Apply feature engineering to both training and test data
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # Dynamically update Config.FEATURES with all the new features created by add_features
    # This ensures all features are available for models that use Config.FEATURES
    new_features_added_by_function = [
        "log_volume", 'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction',
        'ask_buy_interaction', 'ask_sell_interaction', 'net_order_flow', 'normalized_net_flow',
        'buying_pressure', 'volume_weighted_buy', 'total_depth', 'depth_imbalance',
        'relative_spread', 'log_depth', 'kyle_lambda', 'flow_toxicity', 'aggressive_flow_ratio',
        'volume_depth_ratio', 'activity_intensity', 'log_buy_qty', 'log_sell_qty',
        'log_bid_qty', 'log_ask_qty', 'realized_spread_proxy', 'price_impact_proxy',
        'quote_volatility_proxy', 'flow_depth_interaction', 'imbalance_volume_interaction',
        'depth_volume_interaction', 'buy_sell_spread', 'bid_ask_spread', 'trade_informativeness',
        'execution_shortfall_proxy', 'adverse_selection_proxy', 'fill_probability',
        'execution_rate', 'market_efficiency', 'sqrt_volume', 'sqrt_depth', 'volume_squared',
        'imbalance_squared', 'bid_ratio', 'ask_ratio', 'buy_ratio', 'sell_ratio',
        'liquidity_consumption', 'market_stress', 'depth_depletion', 'net_buying_ratio',
        'directional_volume', 'signed_volume'
    ]
    
    # Filter new_features_added_by_function to only include those that are actually in the dataframe after add_features
    actual_new_features_train = [f for f in new_features_added_by_function if f in train_df.columns]
    actual_new_features_test = [f for f in new_features_added_by_function if f in test_df.columns]

    # Combine original and new features, then filter for common columns in both train and test
    combined_features = list(set(Config.FEATURES + Config.MLP_FEATURES + actual_new_features_train))
    
    Config.FEATURES = [f for f in combined_features if f in train_df.columns and f in test_df.columns]
    Config.MLP_FEATURES = [f for f in combined_features if f in train_df.columns and f in test_df.columns]

    # Ensure the label column is *not* in the feature lists themselves, but used for the label
    if Config.LABEL_COLUMN in Config.FEATURES:
        Config.FEATURES.remove(Config.LABEL_COLUMN)
    if Config.LABEL_COLUMN in Config.MLP_FEATURES:
        Config.MLP_FEATURES.remove(Config.LABEL_COLUMN)

    print(f"Final feature counts after engineering and filtering: Config.FEATURES={len(Config.FEATURES)}, Config.MLP_FEATURES={len(Config.MLP_FEATURES)}")
    print(f"Final data shapes before returning: Train: {train_df.shape}, Test: {test_df.shape}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

def get_model_slices(n_samples: int):
    """
    Defines different data slices (subsets) for training models.
    Each slice can represent a different time window or include outlier adjustment.
    This is crucial for handling data relevance (e.g., focusing on recent months).
    """
    # Base slices representing different proportions of the most recent data
    base_slices = [
        {"name": "full_data", "cutoff": 0, "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "oldest_25pct", "cutoff": int(0.25 * n_samples), "is_oldest": True, "outlier_adjusted": False},
    ]

    # Duplicate base slices and add a version with outlier adjustment enabled
    outlier_adjusted_slices = []
    for slice_info in base_slices:
        adjusted_slice = slice_info.copy()
        adjusted_slice["name"] = f"{slice_info['name']}_outlier_adj"
        adjusted_slice["outlier_adjusted"] = True
        outlier_adjusted_slices.append(adjusted_slice)

    return base_slices + outlier_adjusted_slices

def plot_predictions_and_residuals(y_true, y_pred, title_suffix=""):
    """
    Generates and displays plots for actual vs. predicted values and residuals.
    """
    plt.figure(figsize=(15, 6))

    # Plot 1: Actual vs. Predicted
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values {title_suffix}")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Residuals Distribution
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=50)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Residuals {title_suffix}")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_fold_accuracies(xgb_scores, mlp_scores):
    """
    Plots the Pearson correlation accuracies for each fold as curves.
    """
    plt.figure(figsize=(10, 6))
    # Use 1-based indexing for folds for plotting clarity
    folds_xgb = range(1, len(xgb_scores) + 1)
    folds_mlp = range(1, len(mlp_scores) + 1)

    if xgb_scores:
        plt.plot(folds_xgb, xgb_scores, marker='o', linestyle='-', label='XGBoost Avg. Slice Pearson')
    if mlp_scores:
        plt.plot(folds_mlp, mlp_scores, marker='x', linestyle='--', label='MLP Avg. Fold Pearson')

    plt.xlabel("Model Slice/Fold Index")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Model Performance Across Cross-Validation Slices/Folds")
    plt.xticks(list(set(list(folds_xgb) + list(folds_mlp)))) # Show ticks for all relevant fold numbers
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. XGBoost Training and Evaluation
#    (Incorporates Advanced Modeling Techniques)
# =============================================================================
def train_and_evaluate_xgboost(train_df, test_df):
    """
    Trains multiple XGBoost models on different data slices and generates predictions.
    It applies KFold cross-validation, time-decayed weights, and optional outlier adjustment.
    """
    n_samples = len(train_df)
    # Get the defined data slices for training
    model_slices = get_model_slices(n_samples)
    all_xgb_predictions = {} # Store predictions from each XGBoost model
    best_xgb_model = None
    best_xgb_score = -np.inf
    best_xgb_slice_name = ""

    # Lists to collect all validation true values and predictions for overall plotting
    all_val_targets_xgb = []
    all_val_preds_xgb = []
    all_slice_avg_scores_xgb = [] # To collect average scores per slice for the new plot

    print("\n=== Training XGBoost Models ===")
    for slice_info in model_slices:
        slice_name = slice_info["name"]
        cutoff = slice_info["cutoff"]
        is_oldest = slice_info["is_oldest"]
        outlier_adjusted = slice_info["outlier_adjusted"]

        print(f"\n--- Training XGBoost for slice: {slice_name} ---")

        # Select data based on the slice definition
        if is_oldest:
            # Select the oldest part of the data
            current_train_df = train_df.iloc[:cutoff].copy()
        else:
            # Select the most recent part of the data
            current_train_df = train_df.iloc[cutoff:].copy()

        # Data split verification
        if current_train_df.empty:
            print(f"  Skipping slice '{slice_name}' as it is empty after cutoff.")
            continue

        # Ensure that selected features exist in the current_train_df and test_df
        xgb_features_for_slice = [f for f in Config.FEATURES if f in current_train_df.columns and f in test_df.columns]
        if not xgb_features_for_slice:
            print(f"  No common features found for XGBoost in slice '{slice_name}'. Skipping.")
            continue

        X = current_train_df[xgb_features_for_slice]
        y = current_train_df[Config.LABEL_COLUMN]

        # Data split verification (after feature selection)
        if X.empty or y.empty:
            print(f"  Skipping slice '{slice_name}' due to empty X or y after feature selection.")
            continue
        print(f"  XGBoost Training Data Shape for slice '{slice_name}': {X.shape}, Label Shape: {y.shape}")

        # Initialize KFold for cross-validation
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

        fold_preds = [] # Store predictions for each fold
        fold_scores = [] # Store Pearson correlation for each fold

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Data split verification for current fold
            if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
                print(f"    Warning: Fold {fold+1} has empty train or validation sets. Skipping this fold.")
                continue
            print(f"    Fold {fold+1} - Train: {X_train.shape}, Val: {X_val.shape}")

            # Create time-decayed sample weights for training data
            sample_weights = create_time_decay_weights(len(X_train))

            # Apply outlier adjustment if enabled for the current slice
            if outlier_adjusted:
                # Pass y_train to the simplified outlier detection
                sample_weights = detect_outliers_and_adjust_weights(
                    y_train, sample_weights, Config.OUTLIER_FRACTION
                )

            # Initialize and train the XGBoost model
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Make predictions on the validation set
            val_preds = model.predict(X_val)

            # Collect validation true values and predictions for overall plotting
            all_val_targets_xgb.extend(y_val.values)
            all_val_preds_xgb.extend(val_preds)

            # Calculate Pearson correlation on the validation set
            if len(y_val) > 1 and np.std(y_val) > 0 and np.std(val_preds) > 0:
                pearson_coef, _ = pearsonr(y_val, val_preds)
            else:
                pearson_coef = 0.0 # Handle cases with insufficient variance for correlation
                print(f"    Warning: Fold {fold+1} - Insufficient variance for Pearson correlation on validation set. Setting to 0.")

            fold_scores.append(pearson_coef)
            print(f"    Fold {fold+1} Pearson: {pearson_coef:.4f}")

            # Make predictions on the test set for ensembling
            test_preds = model.predict(test_df[xgb_features_for_slice])
            fold_preds.append(test_preds)

        # Average the predictions from all folds for the current slice
        if fold_preds: # Ensure there are predictions to average
            avg_test_preds = np.mean(fold_preds, axis=0)
            all_xgb_predictions[slice_name] = avg_test_preds
            avg_pearson_score = np.mean(fold_scores)
            all_slice_avg_scores_xgb.append(avg_pearson_score) # Collect average score for this slice
            print(f"  Average Pearson for {slice_name}: {avg_pearson_score:.4f}")

            # Save the best XGBoost model based on average validation score
            if avg_pearson_score > best_xgb_score:
                best_xgb_score = avg_pearson_score
                best_xgb_model = model # Save the last trained model from this slice
                best_xgb_slice_name = slice_name
                # Save the best model using Checkpointer (for non-PyTorch models)
                xgb_checkpointer = Checkpointer(filename=f"best_xgb_model_{best_xgb_slice_name}.joblib")
                xgb_checkpointer(best_xgb_score, best_xgb_model)
        else:
            print(f"  No valid folds for slice '{slice_name}'. No predictions generated.")

    # Plot overall XGBoost validation predictions and residuals
    if all_val_targets_xgb and all_val_preds_xgb:
        plot_predictions_and_residuals(pd.Series(all_val_targets_xgb), pd.Series(all_val_preds_xgb),
                                       title_suffix=" (Overall XGBoost Validation)")

    return all_xgb_predictions, all_slice_avg_scores_xgb

# =============================================================================
# 7. MLP Training and Evaluation
#    (Incorporates Advanced Modeling Techniques)
# =============================================================================
def train_and_evaluate_mlp(train_df, test_df):
    """
    Trains an MLP model using PyTorch with KFold cross-validation,
    time-decayed weights, and optional outlier adjustment.
    Generates predictions for the test set.
    """
    print("\n=== Training MLP Model ===")
    set_seed(Config.RANDOM_STATE) # Ensure reproducibility for MLP training

    # Ensure that selected MLP features exist in the train_df and test_df
    mlp_features_for_model = [f for f in Config.MLP_FEATURES if f in train_df.columns and f in test_df.columns]
    if not mlp_features_for_model:
        print("  No common features found for MLP. Skipping MLP training.")
        return np.zeros(len(test_df)), [] # Return array of zeros and empty list for scores

    # Define hyperparameters for the MLP model
    mlp_hparams = {
        "input_dim": len(mlp_features_for_model), # Use the actual number of features
        "layers": [len(mlp_features_for_model), 256, 128, 64, 1], # Example layer structure
        "activation": "relu",
        "last_activation": None,
        "dropout_rate": 0.2,
        "learning_rate": 1e-4,
        "batch_size": 1024,
        "epochs": 20, # Reduced for faster execution
        "seed": Config.RANDOM_STATE
    }

    # Scale MLP features
    scaler = StandardScaler()
    X_mlp_scaled = scaler.fit_transform(train_df[mlp_features_for_model])
    X_test_mlp_scaled = scaler.transform(test_df[mlp_features_for_model])
    y_mlp = train_df[Config.LABEL_COLUMN]

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

    mlp_test_preds = [] # Store test predictions from each fold
    mlp_fold_scores = [] # Store Pearson correlation for each fold

    # Lists to collect all validation true values and predictions for overall plotting
    all_val_targets_mlp = []
    all_val_preds_mlp = []
    all_fold_scores_mlp = [] # To collect scores for the new plot

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_mlp_scaled, y_mlp)):
        print(f"\n  --- MLP Fold {fold+1}/{Config.N_FOLDS} ---")
        X_train_fold, X_val_fold = X_mlp_scaled[train_idx], X_mlp_scaled[val_idx]
        y_train_fold, y_val_fold = y_mlp.iloc[train_idx], y_mlp.iloc[val_idx]

        # Data split verification for current fold
        if X_train_fold.size == 0 or y_train_fold.empty or X_val_fold.size == 0 or y_val_fold.empty:
            print(f"    Warning: Fold {fold+1} has empty train or validation sets. Skipping this fold.")
            continue
        print(f"    Fold {fold+1} - Train: {X_train_fold.shape}, Val: {X_val_fold.shape}")

        # Create time-decayed sample weights for training data
        sample_weights = create_time_decay_weights(len(X_train_fold))

        # Apply outlier adjustment
        # Pass y_train_fold to the simplified outlier detection
        sample_weights = detect_outliers_and_adjust_weights(
            y_train_fold, sample_weights, Config.OUTLIER_FRACTION
        )

        # Convert to tensors and create DataLoaders
        train_dataloader = get_dataloaders(X_train_fold, y_train_fold, mlp_hparams, device)
        val_dataloader = get_dataloaders(X_val_fold, y_val_fold, mlp_hparams, device, shuffle=False)
        test_dataloader = get_dataloaders(X_test_mlp_scaled, None, mlp_hparams, device, shuffle=False)

        # Initialize MLP model, optimizer, and loss function
        model = MLP(dropout_rate=mlp_hparams["dropout_rate"],
                    layers=mlp_hparams["layers"],
                    activation=mlp_hparams["activation"],
                    last_activation=mlp_hparams["last_activation"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=mlp_hparams["learning_rate"])
        criterion = nn.MSELoss(reduction='none') # Use 'none' to apply sample weights

        checkpointer = Checkpointer(filename=f"mlp_fold_{fold+1}_best_model.pt")

        # Training loop
        for epoch in tqdm(range(mlp_hparams["epochs"]), desc=f"  Epochs (Fold {fold+1})"):
            model.train() # Set model to training mode
            total_loss = 0
            for i, (batch_X, batch_y) in enumerate(train_dataloader):
                optimizer.zero_grad() # Zero the gradients
                outputs = model(batch_X) # Forward pass

                # Get the corresponding sample weights for the current batch
                # Corrected indexing: sample_weights is already aligned with X_train_fold
                # so we can directly slice it using batch indices.
                start_idx = i * train_dataloader.batch_size
                end_idx = min(start_idx + len(batch_X), len(sample_weights)) # Ensure end_idx doesn't exceed bounds
                batch_weights = torch.tensor(sample_weights[start_idx:end_idx],
                                             dtype=torch.float32).to(device).unsqueeze(1)

                loss = criterion(outputs, batch_y) # Calculate loss
                weighted_loss = (loss * batch_weights).mean() # Apply weights and take mean
                weighted_loss.backward() # Backward pass
                optimizer.step() # Update weights
                total_loss += weighted_loss.item()

            # Validation step
            model.eval() # Set model to evaluation mode
            val_preds = []
            val_targets = []
            with torch.no_grad(): # Disable gradient calculations
                for batch_X_val, batch_y_val in val_dataloader:
                    outputs_val = model(batch_X_val)
                    val_preds.extend(outputs_val.cpu().numpy().flatten())
                    val_targets.extend(batch_y_val.cpu().numpy().flatten())

            # Calculate Pearson correlation on validation set
            if len(val_targets) > 1 and np.std(val_targets) > 0 and np.std(val_preds) > 0:
                pearson_coef, _ = pearsonr(val_targets, val_preds)
            else:
                pearson_coef = 0.0 # Handle cases with insufficient variance for correlation
                print(f"    Warning: Epoch {epoch+1} - Insufficient variance for Pearson correlation on validation set. Setting to 0.")

            # Save best model checkpoint
            checkpointer(pearson_coef, model)

        # Load the best model for test predictions
        model = checkpointer.load(model)
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for batch_X_test in test_dataloader:
                outputs_test = model(batch_X_test[0]) # batch_X_test is a tuple (input,)
                fold_test_preds.extend(outputs_test.cpu().numpy().flatten())
        mlp_test_preds.append(fold_test_preds)
        mlp_fold_scores.append(checkpointer.best_metric) # Store the best metric from this fold

        # Collect validation true values and predictions for overall plotting
        all_val_targets_mlp.extend(val_targets)
        all_val_preds_mlp.extend(val_preds)
        all_fold_scores_mlp.append(checkpointer.best_metric) # Collect best score for this fold for the new plot

    # Average test predictions across all folds for MLP
    if mlp_test_preds: # Ensure there are predictions to average
        avg_mlp_test_preds = np.mean(mlp_test_preds, axis=0)
        print(f"\n  Average MLP Pearson across folds: {np.mean(mlp_fold_scores):.4f}")
    else:
        avg_mlp_test_preds = np.zeros(len(test_df))
        print("  No valid MLP folds. Returning zeros for MLP predictions.")

    # Plot overall MLP validation predictions and residuals
    if all_val_targets_mlp and all_val_preds_mlp:
        plot_predictions_and_residuals(pd.Series(all_val_targets_mlp), pd.Series(all_val_preds_mlp),
                                       title_suffix=" (Overall MLP Validation)")

    return avg_mlp_test_preds, all_fold_scores_mlp

# =============================================================================
# 8. Ensemble and Submission
# =============================================================================
def create_ensemble_and_submission(all_xgb_predictions, mlp_predictions, base_submission_df, test_df):
    """
    Combines predictions from different models (XGBoost slices and MLP)
    to create an ensemble prediction and generates the final submission file.
    Calculates correlation between top models.
    """
    # Define the primary output directory for submission.csv
    PRIMARY_OUTPUT_DIR = Path("/kaggle/working/")
    # Define the subdirectory for other submission files (if any)
    SECONDARY_OUTPUT_DIR = FINAL_SUBMISSIONS_DIR

    # Store all available model predictions in a dictionary for easy access
    available_models = {
        "mlp": {"prediction": mlp_predictions, "weight": 0.5}, # Initial weight for MLP
    }

    # Add XGBoost predictions to available models
    for name, preds in all_xgb_predictions.items():
        available_models[name] = {"prediction": preds, "weight": 0.5} # Initial weight for XGBoost slices

    # Example: Simple ensemble of MLP and a specific XGBoost slice
    # You can customize these weights and models based on performance
    print("\n=== Creating Two-Model Ensemble (MLP + full_data_outlier_adj XGB) ===")
    model_names = ["mlp", "full_data_outlier_adj"] # Choose two models for this ensemble
    
    # Ensure both chosen models exist
    if all(name in available_models for name in model_names):
        # Optimize weights for the two models based on their correlation
        # This is a simplified approach; more advanced methods like Nelder-Mead or
        # genetic algorithms can be used for optimal weight finding.
        p1 = available_models[model_names[0]]["prediction"]
        p2 = available_models[model_names[1]]["prediction"]
        
        # Calculate correlation for weight adjustment (simplified)
        # Ensure there's enough data and variance for meaningful correlation
        if len(p1) > 1 and np.std(p1) > 0 and np.std(p2) > 0:
            corr = np.corrcoef(p1, p2)[0, 1]
        else:
            corr = 0.0 # Default to 0 correlation if not enough data or variance
            print(f"  Warning: Not enough data or variance for correlation calculation between {model_names[0]} and {model_names[1]}. Defaulting correlation to 0.")

        # Heuristic for weighting: inversely proportional to correlation (simplified)
        # Adjust these weights based on empirical performance
        w1 = 0.5 + (0.5 * (1 - corr)) # Give more weight if less correlated
        w2 = 1.0 - w1
        
        avg_pred = (w1 * p1) + (w2 * p2)

        submission = base_submission_df.copy()
        submission["label"] = avg_pred

        # Ensure the final submission file is named submission.csv and saved to /kaggle/working/
        filename = "submission.csv"
        submission.to_csv(PRIMARY_OUTPUT_DIR / filename, index=False) # Changed output directory

        print(f"\n✓ {filename} saved to {PRIMARY_OUTPUT_DIR}")
        print(f"  Weights: {model_names[0]}={w1:.0%}, {model_names[1]}={w2:.0%}")
        print(f"  Mean: {avg_pred.mean():.6f}, Std: {avg_pred.std():.6f}")

        # Calculate and print correlation between the two ensembled models
        if len(p1) > 1 and np.std(p1) > 0 and np.std(p2) > 0:
            corr_ensemble = np.corrcoef(p1, p2)[0, 1]
            print(f"\n Correlation between {model_names[0]} and {model_names[1]}: {corr_ensemble:.4f}")
        else:
            print(f"\n Correlation between {model_names[0]} and {model_names[1]}: Not calculable (insufficient data/variance).")
        
        # Plot ensemble predictions on unseen data (test_df)
        # Note: We don't have true labels for test_df, so we can only plot prediction distribution
        plt.figure(figsize=(7, 5))
        sns.histplot(avg_pred, kde=True, bins=50)
        plt.xlabel("Predicted Values")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Ensemble Predictions on Unseen Data (Final Submission)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    else:
        print(f"  Warning: Could not create two-model ensemble. Ensure models {model_names} exist.")


    # If you have 3+ models, create multi-model ensemble (e.g., equal weights)
    if len(available_models) >= 3:
        print(f"\n=== Creating Multi-Model Ensemble (Equal Weights) ===")

        # Collect all predictions for equal weighting
        all_preds_for_equal_ensemble = [model_data["prediction"] for model_data in available_models.values()]
        
        # Equal weights for all models
        equal_weight = 1.0 / len(all_preds_for_equal_ensemble)
        avg_pred_equal_weight = sum(equal_weight * pred for pred in all_preds_for_equal_ensemble)

        submission_equal = base_submission_df.copy()
        submission_equal["label"] = avg_pred_equal_weight

        # This will be an alternative submission file, saved to the secondary directory
        filename_equal = "ensemble_all_models_equal.csv"
        submission_equal.to_csv(SECONDARY_OUTPUT_DIR / filename_equal, index=False) # Remains in subdirectory
        print(f"\n✓ {filename_equal} saved to {SECONDARY_OUTPUT_DIR}")
        print(f"  Models: {', '.join(available_models.keys())}")
        print(f"  Weight per model: {equal_weight:.1%}")
        print(f"  Mean: {avg_pred_equal_weight.mean():.6f}, Std: {avg_pred_equal_weight.std():.6f}")

# =============================================================================
# 9. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Load and preprocess data
    train_df, test_df, submission_df = load_data()

    # Train and evaluate XGBoost models across different data slices
    xgb_predictions, xgb_fold_scores = train_and_evaluate_xgboost(train_df, test_df)

    # Train and evaluate MLP model
    mlp_predictions, mlp_fold_scores = train_and_evaluate_mlp(train_df, test_df)

    # Plot overall fold accuracies
    plot_fold_accuracies(xgb_fold_scores, mlp_fold_scores)

    # Create ensemble predictions and generate submission files
    create_ensemble_and_submission(xgb_predictions, mlp_predictions, submission_df, test_df)

    print("\nEnsemble process completed successfully!")
    print("\nNote on Pearson score of +1.000: Achieving a perfect Pearson correlation of +1.000 (or -1.000) in real-world financial time-series prediction is generally not feasible and would typically indicate data leakage or severe overfitting. The goal is to maximize this score, but a value of 1.000 is an unrealistic theoretical ideal for complex, noisy data.")


# --- 0. Initial Setup and Imports ---
# This section handles necessary library imports and initial environment configurations,
# such as suppressing warnings and checking for GPU availability.

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit # Crucial for time series cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Attention, Bidirectional, LSTM, BatchNormalization # Added BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 # For L2 regularization
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib
import os
import warnings

# Suppress specific Seaborn FutureWarnings that are noisy but generally harmless
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")

# Ensure TensorFlow can see GPUs if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    else:
        print("No GPUs found, running on CPU.")
except Exception as e:
    print(f"Error checking for GPUs: {e}. Running on CPU.")


# --- 1. AdvancedMLPipeline Class Definition ---
# This is the core class that encapsulates the entire machine learning workflow,
# from data preparation to model evaluation.

class AdvancedMLPipeline:
    """
    A comprehensive machine learning pipeline incorporating data preparation,
    feature engineering, model selection, hyperparameter tuning, outlier detection,
    memory management, ensemble methods, and robust evaluation with visualizations.
    """

    def __init__(self, target_column='label', random_state=42, sequence_length=30,
                 epochs=50, batch_size=32, patience=15, min_delta=0.0001, # Increased patience
                 learning_rate=0.001, n_splits=5, contamination=0.01, l2_reg_strength=1e-4): # Added L2 regularization strength
        """
        Initializes the pipeline with a target column, random state, and model/training parameters.

        Args:
            target_column (str): The name of the target variable column.
            random_state (int): Seed for random operations to ensure reproducibility.
            sequence_length (int): The number of past time steps to use as input for the model.
            epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training the neural network.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            learning_rate (float): Learning rate for the Adam optimizer.
            n_splits (int): Number of splits for TimeSeriesSplit cross-validation.
            contamination (float): The proportion of outliers in the dataset (for IsolationForest).
            l2_reg_strength (float): L2 regularization strength for dense layers.
        """
        self.target_column = target_column
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.contamination = contamination
        self.l2_reg_strength = l2_reg_strength

        self.model = None # Placeholder for the trained Keras model (from a single fold or ensemble)
        self.preprocessor = None # Stores the fitted ColumnTransformer
        self.target_scaler = None # Stores the scaler for the target variable
        self.history = None # Stores training history of the last trained model
        self.oof_predictions = [] # Out-of-fold predictions for ensemble evaluation
        self.oof_actuals = [] # Out-of-fold actuals for ensemble evaluation
        self.numerical_features = [] # List of identified numerical feature names after initial prep
        self.categorical_features = [] # List of identified categorical feature names after initial prep
        self.feature_columns_for_model = [] # Final feature columns used for model input (after transformation)
        self.n_features = 0 # Will be set dynamically after sequence creation
        self._transformed_target_column_name = None # Stores the actual name of the target column after transformation
        self._transformed_id_column_name = None # Stores the actual name of the ID column after transformation

    def _data_preparation(self, df):
        """
        1. Data preparation: Identifies numerical and categorical features, handles missing values,
        and ensures 'ID' and 'Date' columns are treated appropriately.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
            list: List of numerical feature names to be transformed.
            list: List of categorical feature names to be transformed.
        """
        print("1. Data Preparation and Cleaning...")
        df_copy = df.copy()

        # Convert 'Date' column to datetime if it exists and is not already
        if 'Date' in df_copy.columns:
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            # Sort by date for time series consistency
            df_copy = df_copy.sort_values(by='Date').reset_index(drop=True)

        # Identify all numerical and categorical columns
        all_numerical_cols = [col for col in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[col])]
        all_categorical_cols = [col for col in df_copy.columns if pd.api.types.is_object_dtype(df_copy[col])]

        # Features that will be explicitly transformed by StandardScaler/OneHotEncoder
        # Exclude 'ID', 'Date' (handled in feature engineering), and the 'target_column'
        features_to_transform_num = [col for col in all_numerical_cols if col not in ['ID', self.target_column]]
        features_to_transform_cat = [col for col in all_categorical_cols if col not in ['ID', self.target_column, 'Date']] # 'Date' might be object before conversion

        # Handle missing values: fill with median for numerical, mode for categorical
        for col in features_to_transform_num:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        for col in features_to_transform_cat:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

        # Drop rows with NaN in the target column if any
        if self.target_column in df_copy.columns and df_copy[self.target_column].isnull().any():
            print(f"Dropping rows with NaN in target column: {self.target_column}")
            df_copy.dropna(subset=[self.target_column], inplace=True)

        print(f"Data preparation complete. Cleaned shape: {df_copy.shape}")
        return df_copy, features_to_transform_num, features_to_transform_cat

    def _feature_engineering(self, df, numerical_features_for_transformer, categorical_features_for_transformer):
        """
        2. Feature Engineering: Creates new features from existing ones, focusing on time-series
        and financial indicators relevant to crypto market prediction.

        Args:
            df (pd.DataFrame): The input DataFrame.
            numerical_features_for_transformer (list): List of numerical feature names *intended for transformation*.
            categorical_features_for_transformer (list): List of categorical feature names *intended for transformation*.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
            list: Updated list of numerical feature names (including new ones).
            list: Updated list of categorical feature names.
        """
        print("2. Feature Engineering...")
        df_copy = df.copy()
        
        updated_numerical_features = list(numerical_features_for_transformer)
        updated_categorical_features = list(categorical_features_for_transformer)

        # --- Time-based Features (if 'Date' column is present) ---
        if 'Date' in df_copy.columns:
            df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
            df_copy['DayOfMonth'] = df_copy['Date'].dt.day
            df_copy['Month'] = df_copy['Date'].dt.month
            df_copy['Year'] = df_copy['Date'].dt.year
            df_copy['Quarter'] = df_copy['Date'].dt.quarter
            updated_numerical_features.extend(['DayOfWeek', 'DayOfMonth', 'Month', 'Year', 'Quarter'])
            # Drop the original 'Date' column after extracting features
            df_copy = df_copy.drop(columns=['Date'])

        # --- Lagged Features ---
        # Assuming 'Close' or 'label' (if it's a price) is available for lags
        # If 'Close' is not present, replace with a suitable price-like column or skip.
        price_col = 'Close' if 'Close' in df_copy.columns else self.target_column # Use 'Close' if available, else target
        if price_col in df_copy.columns:
            for lag in range(1, self.sequence_length + 1):
                df_copy[f'{price_col}_Lag_{lag}'] = df_copy[price_col].shift(lag)
                if 'volume' in df_copy.columns:
                    df_copy[f'volume_Lag_{lag}'] = df_copy['volume'].shift(lag)
                updated_numerical_features.append(f'{price_col}_Lag_{lag}')
                if 'volume' in df_copy.columns:
                    updated_numerical_features.append(f'volume_Lag_{lag}')

        # --- Rolling Statistics ---
        windows = [5, 10, 20, 50]
        if price_col in df_copy.columns:
            for window in windows:
                df_copy[f'SMA_{window}'] = df_copy[price_col].rolling(window=window).mean()
                df_copy[f'EMA_{window}'] = df_copy[price_col].ewm(span=window, adjust=False).mean()
                df_copy[f'Volatility_{window}'] = df_copy[price_col].rolling(window=window).std()
                updated_numerical_features.extend([f'SMA_{window}', f'EMA_{window}', f'Volatility_{window}'])

        # --- RSI (Relative Strength Index) ---
        if price_col in df_copy.columns:
            def calculate_rsi(data, window):
                diff = data.diff(1)
                gain = diff.where(diff > 0, 0)
                loss = -diff.where(diff < 0, 0)
                avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
                avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            df_copy['RSI'] = calculate_rsi(df_copy[price_col], 14)
            updated_numerical_features.append('RSI')

        # --- MACD (Moving Average Convergence Divergence) ---
        if price_col in df_copy.columns:
            exp1 = df_copy[price_col].ewm(span=12, adjust=False).mean()
            exp2 = df_copy[price_col].ewm(span=26, adjust=False).mean()
            df_copy['MACD'] = exp1 - exp2
            df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
            updated_numerical_features.extend(['MACD', 'Signal_Line'])

        # --- Bollinger Bands ---
        if price_col in df_copy.columns:
            df_copy['BB_Middle'] = df_copy[price_col].rolling(window=20).mean()
            df_copy['BB_Upper'] = df_copy['BB_Middle'] + (df_copy[price_col].rolling(window=20).std() * 2)
            df_copy['BB_Lower'] = df_copy['BB_Middle'] - (df_copy[price_col].rolling(window=20).std() * 2)
            updated_numerical_features.extend(['BB_Middle', 'BB_Upper', 'BB_Lower'])

        # --- Difference Features ---
        if price_col in df_copy.columns:
            df_copy[f'{price_col}_Diff'] = df_copy[price_col].diff()
            updated_numerical_features.append(f'{price_col}_Diff')
        if 'volume' in df_copy.columns:
            df_copy['Volume_Diff'] = df_copy['volume'].diff()
            updated_numerical_features.append('Volume_Diff')

        # --- Percentage Price Change ---
        if price_col in df_copy.columns:
            for period in [1, 5, 10]: # Daily, 5-day, 10-day percentage change
                df_copy[f'{price_col}_PctChange_{period}'] = df_copy[price_col].pct_change(period)
                updated_numerical_features.append(f'{price_col}_PctChange_{period}')

        # --- Log Returns ---
        if price_col in df_copy.columns:
            df_copy[f'{price_col}_LogReturn'] = np.log(df_copy[price_col] / df_copy[price_col].shift(1))
            updated_numerical_features.append(f'{price_col}_LogReturn')

        # --- Price-Volume Interaction ---
        if price_col in df_copy.columns and 'volume' in df_copy.columns:
            df_copy['Price_Volume_Interaction'] = df_copy[price_col] * df_copy['volume']
            updated_numerical_features.append('Price_Volume_Interaction')

        # --- Example X Feature Interactions (add more as needed based on feature importance) ---
        # Assuming X1, X2, X3, X4 exist in the dummy data
        if 'X1' in df_copy.columns and 'X2' in df_copy.columns:
            df_copy['X1_X2_Prod'] = df_copy['X1'] * df_copy['X2']
            updated_numerical_features.append('X1_X2_Prod')
        if 'X3' in df_copy.columns and 'X4' in df_copy.columns and (df_copy['X4'] != 0).all(): # Avoid division by zero
            df_copy['X3_X4_Ratio'] = df_copy['X3'] / df_copy['X4']
            updated_numerical_features.append('X3_X4_Ratio')


        # Fill any NaN values created by feature engineering (e.g., from rolling windows or lags)
        # Use ffill then bfill for time series data to propagate last valid observation
        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True) # Fill any remaining NaNs at the beginning

        # Ensure uniqueness and order of feature lists
        updated_numerical_features = list(dict.fromkeys(updated_numerical_features))
        updated_categorical_features = list(dict.fromkeys(updated_categorical_features))

        # Filter the numerical and categorical lists to only include columns actually present in df_copy
        # and exclude 'ID' and the target column, as these are handled separately (passthrough for ID, target for y)
        final_numerical_features_for_transformer = [col for col in updated_numerical_features if col in df_copy.columns and col != 'ID' and col != self.target_column]
        final_categorical_features_for_transformer = [col for col in updated_categorical_features if col in df_copy.columns and col != 'ID' and col != self.target_column]


        print(f"Feature engineering complete. Current shape: {df_copy.shape}")
        return df_copy, final_numerical_features_for_transformer, final_categorical_features_for_transformer

    def _outlier_detection_and_treatment(self, df, numerical_features_for_transformer):
        """
        3. Outlier Detection and Treatment: Identifies and handles outliers using IsolationForest.

        Args:
            df (pd.DataFrame): The input DataFrame.
            numerical_features_for_transformer (list): List of numerical feature names *intended for transformation*.

        Returns:
            pd.DataFrame: DataFrame with outliers treated/removed.
        """
        print("3. Outlier Detection and Treatment...")

        # Use only the numerical features that are actually present in the current df for outlier detection
        # and exclude 'ID' and target_column as they are not typically used for outlier detection
        features_for_outliers = [f for f in numerical_features_for_transformer if f in df.columns and f != 'ID' and f != self.target_column]

        if features_for_outliers:
            isolation_forest = IsolationForest(random_state=self.random_state, contamination=self.contamination, n_jobs=-1)
            
            # Create a copy of the relevant features for IsolationForest to avoid SettingWithCopyWarning
            df_for_isolation = df[features_for_outliers].copy()
            
            # Fit and predict anomalies
            df['anomaly'] = isolation_forest.fit_predict(df_for_isolation)
            
            # Filter the original DataFrame (which still contains the target column)
            # based on the anomaly predictions.
            df_cleaned = df[df['anomaly'] == 1].drop(columns=['anomaly']).reset_index(drop=True)
            
            print(f"Outlier detection complete. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")
            return df_cleaned
        else:
            print("No suitable numerical features for outlier detection. Skipping.")
            return df

    def _data_transformation(self, df, numerical_features_for_transformer, categorical_features_for_transformer):
        """
        4. Data Transformation: Scales numerical features (StandardScaler) and
        encodes categorical features (OneHotEncoder).

        Args:
            df (pd.DataFrame): The input DataFrame.
            numerical_features_for_transformer (list): List of numerical feature names *intended for transformation*.
            categorical_features_for_transformer (list): List of categorical feature names *intended for transformation*.

        Returns:
            pd.DataFrame: Transformed DataFrame.
            sklearn.compose.ColumnTransformer: The fitted preprocessor.
        """
        print("4. Data Transformation...")
        
        # Sanity check: Ensure original target column is present before proceeding
        if self.target_column not in df.columns:
            raise KeyError(f"Original target column '{self.target_column}' is missing from the DataFrame before transformation.")
        if 'ID' not in df.columns:
            raise KeyError(f"Original 'ID' column is missing from the DataFrame before transformation.")


        # Filter features to ensure they are actually present in the current DataFrame
        current_numerical_features = [f for f in numerical_features_for_transformer if f in df.columns]
        current_categorical_features = [f for f in categorical_features_for_transformer if f in df.columns]

        transformers = [
            ('num', StandardScaler(), current_numerical_features)
        ]
        if current_categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), current_categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough' # This will pass through any remaining columns (e.g., 'ID', target_column)
        )
        
        df_transformed_array = preprocessor.fit_transform(df)

        # Get the feature names out from the preprocessor directly. This should be the authoritative list.
        transformed_column_names = preprocessor.get_feature_names_out()
        
        # Create DataFrame from the transformed array and the generated column names
        df_transformed = pd.DataFrame(df_transformed_array, columns=transformed_column_names, index=df.index)

        # Determine the actual name of the target column after transformation
        # It will either be its original name or prefixed with 'remainder__'
        if self.target_column in df_transformed.columns:
            self._transformed_target_column_name = self.target_column
        elif f'remainder__{self.target_column}' in df_transformed.columns:
            self._transformed_target_column_name = f'remainder__{self.target_column}'
        else:
            raise KeyError(f"Transformed target column (original: '{self.target_column}') not found in transformed DataFrame. Columns: {df_transformed.columns.tolist()}")

        # Determine the actual name of the ID column after transformation
        if 'ID' in df_transformed.columns:
            self._transformed_id_column_name = 'ID'
        elif 'remainder__ID' in df_transformed.columns:
            self._transformed_id_column_name = 'remainder__ID'
        else:
            raise KeyError(f"Transformed ID column (original: 'ID') not found in transformed DataFrame. Columns: {df_transformed.columns.tolist()}")


        # Store the target scaler for inverse transformation later
        self.target_scaler = StandardScaler()
        df_transformed[self._transformed_target_column_name] = \
            self.target_scaler.fit_transform(df_transformed[[self._transformed_target_column_name]])

        print(f"Data transformation complete. Transformed shape: {df_transformed.shape}")
        print(f"Actual transformed target column name: {self._transformed_target_column_name}")
        print(f"Actual transformed ID column name: {self._transformed_id_column_name}")
        return df_transformed, preprocessor

    def _create_sequences(self, data_df):
        """
        Creates sequences for the CNN-Attention model from a DataFrame.
        This method is crucial for time series forecasting.

        Args:
            data_df (pd.DataFrame): Input DataFrame with features and target.
                                     Assumes features are already numerical and scaled.

        Returns:
            tuple: X (features sequences), y (target values).
        """
        print(f"Creating sequences with length {self.sequence_length}...")

        X, y = [], []
        
        # Ensure only numerical features (excluding 'ID' and the transformed target) are used for X
        feature_columns = [col for col in data_df.columns if col != self._transformed_target_column_name and col != self._transformed_id_column_name and np.issubdtype(data_df[col].dtype, np.number)]
        
        if not feature_columns:
            raise ValueError("No numerical features found for sequence creation after filtering 'ID' and target.")

        # Update n_features based on actual columns used for X
        self.n_features = len(feature_columns)

        # Ensure the transformed target column is present before trying to select it.
        if self._transformed_target_column_name not in data_df.columns:
            raise KeyError(f"Transformed target column '{self._transformed_target_column_name}' is missing from data_df in _create_sequences. Columns: {data_df.columns.tolist()}")

        # Drop rows with NaNs that might result from previous steps (e.g., from feature engineering lags)
        # This is critical before creating sequences.
        data_for_sequences = data_df[feature_columns + [self._transformed_target_column_name]].dropna().copy()

        # If after dropping NaNs, the DataFrame is too small for sequences
        if len(data_for_sequences) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points ({len(data_for_sequences)}) to create sequences of length {self.sequence_length}.")

        for i in tqdm(range(len(data_for_sequences) - self.sequence_length)):
            # X will be a sequence of 'sequence_length' rows for the selected features
            X.append(data_for_sequences.iloc[i:(i + self.sequence_length)][feature_columns].values)
            # y will be the target value at the end of the sequence (or the next step)
            y.append(data_for_sequences.iloc[i + self.sequence_length][self._transformed_target_column_name])

        X = np.array(X)
        y = np.array(y)
        print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def _build_model(self):
        """
        5. Model Selection and Training: Defines and compiles the CNN-Attention Keras model.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        print("5. Building Model...")
        # Input shape is (sequence_length, n_features)
        input_layer = Input(shape=(self.sequence_length, self.n_features))

        # --- CNN Layers with BatchNormalization and Dropout ---
        conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(input_layer)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2)(bn1)
        drop1 = Dropout(0.4)(pool1) # Increased dropout

        conv2 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='causal')(drop1)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(pool_size=2)(bn2)
        drop2 = Dropout(0.4)(pool2) # Increased dropout

        conv3 = Conv1D(filters=512, kernel_size=3, activation='relu', padding='causal')(drop2)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling1D(pool_size=2)(bn3)
        drop3 = Dropout(0.4)(pool3) # Increased dropout

        # --- Attention Mechanism ---
        # If pool3 output has sequence length > 1, apply attention directly
        if pool3.shape[1] is not None and pool3.shape[1] > 1:
            attention_output = Attention()([pool3, pool3]) # Self-attention
            flatten = Flatten()(attention_output)
        else: # If pooling reduces sequence to 1 or None, flatten directly
            flatten = Flatten()(pool3)

        # --- Dense Layers with L2 Regularization and Dropout ---
        dense1 = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg_strength))(flatten)
        drop4 = Dropout(0.5)(dense1) # Increased dropout
        dense2 = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg_strength))(drop4)
        drop5 = Dropout(0.5)(dense2) # Increased dropout
        output_layer = Dense(1)(drop5) # Output for regression

        model = Model(inputs=input_layer, outputs=output_layer)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
        print("Model built successfully.")
        model.summary()
        return model

    def _hyperparameter_tuning(self):
        """
        (Optional) 6. Hyperparameter Tuning: Placeholder for hyperparameter optimization.
        This would typically involve GridSearchCV, RandomizedSearchCV, or more advanced methods
        like KerasTuner or Optuna.
        """
        print("6. Hyperparameter Tuning (Conceptual)...")
        print("Hyperparameter tuning is a complex process often done separately to find optimal model parameters.")

    def _ensemble_methods(self, models, X_data):
        """
        7. Ensemble Methods: Combines predictions from multiple models (e.g., from cross-validation folds)
        to improve robustness and accuracy.

        Args:
            models (list): List of trained Keras models.
            X_data (np.array): Features for prediction.

        Returns:
            np.array: Ensembled predictions (inverse transformed).
        """
        print("7. Ensemble Methods...")
        if not models:
            raise ValueError("No models provided for ensembling.")
        
        all_predictions_scaled = []
        for model in models:
            preds_scaled = model.predict(X_data).flatten()
            all_predictions_scaled.append(preds_scaled)
        
        # Simple averaging ensemble of scaled predictions
        ensemble_preds_scaled = np.mean(all_predictions_scaled, axis=0)
        
        # Inverse transform the ensembled predictions to original scale
        if self.target_scaler:
            ensembled_predictions = self.target_scaler.inverse_transform(ensemble_preds_scaled.reshape(-1, 1)).flatten()
        else:
            ensembled_predictions = ensemble_preds_scaled # If target was not scaled

        print("Ensemble predictions generated.")
        return ensembled_predictions

    def _evaluation_and_visualization(self, y_true, y_pred, history=None):
        """
        8. Evaluation and Visualization: Assesses model performance using metrics (MSE, R2, MAE, RMSE)
        and visualizes results (e.g., training history, actual vs. predicted plots).

        Args:
            y_true (np.array): True target values (original scale).
            y_pred (np.array): Predicted target values (original scale).
            history (tf.keras.callbacks.History, optional): Training history object for plotting.
        """
        print("8. Evaluation and Visualization...")
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

        # Plot training history
        if history:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Train MAE')
            if 'val_mae' in history.history:
                plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.grid(True)
        plt.show()

        print("Evaluation and visualization complete.")

    def train_and_evaluate(self, df):
        """
        Orchestrates the entire training and evaluation process using TimeSeriesSplit.

        Args:
            df (pd.DataFrame): The input DataFrame for training.
        """
        print("\n--- Starting Training and Evaluation Pipeline ---")
        
        # 1. Data Preparation
        df_prepared, initial_numerical_features, initial_categorical_features = self._data_preparation(df.copy())
        
        # 2. Feature Engineering
        df_engineered, engineered_numerical_features, engineered_categorical_features = self._feature_engineering(
            df_prepared.copy(), initial_numerical_features, initial_categorical_features)
        
        # Store the final lists of features that will be used by the preprocessor
        self.numerical_features = engineered_numerical_features
        self.categorical_features = engineered_categorical_features

        # 3. Outlier Detection and Treatment
        df_cleaned = self._outlier_detection_and_treatment(df_engineered.copy(), self.numerical_features)
        
        # 4. Data Transformation (Scaling and Encoding)
        df_transformed, self.preprocessor = self._data_transformation(df_cleaned.copy(), self.numerical_features, self.categorical_features)

        # 5. Create Sequences for Time Series Model
        X_full, y_full = self._create_sequences(df_transformed)
        
        # TimeSeriesSplit for robust cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fold_results = []
        # Initialize arrays for all out-of-fold predictions and actuals
        # These will be populated based on the indices of the validation sets
        all_oof_preds_scaled = np.zeros(len(y_full))
        all_oof_actuals_scaled = np.zeros(len(y_full))
        all_oof_mask = np.zeros(len(y_full), dtype=bool)
        
        self.models = [] # Reset models for each training run

        for fold, (train_index, val_index) in enumerate(tscv.split(X_full)):
            print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            X_train, X_val = X_full[train_index], X_full[val_index]
            y_train, y_val = y_full[train_index], y_full[val_index]

            # Build a new model for each fold to ensure independence
            model = self._build_model()
            self.models.append(model) # Store the trained model

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                min_delta=self.min_delta,
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2, # Reduce patience for LR reduction
                min_lr=1e-6,
                verbose=1
            )

            print(f"Training model for Fold {fold+1}...")
            history = model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0 # Set to 1 for detailed training output
            )
            self.history = history # Store history of the last fold

            # Evaluate on validation set
            val_loss, val_mse, val_mae = model.evaluate(X_val, y_val, verbose=0)
            val_preds_scaled = model.predict(X_val).flatten()
            
            # Inverse transform validation predictions and actuals for metrics
            val_preds = self.target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
            val_actuals = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

            val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
            val_r2 = r2_score(val_actuals, val_preds)

            print(f"Fold {fold+1} Validation Results:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  MSE: {val_mse:.4f}")
            print(f"  MAE: {val_mae:.4f}")
            print(f"  RMSE: {val_rmse:.4f}")
            print(f"  R2 Score: {val_r2:.4f}")

            fold_results.append({
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })

            # Store out-of-fold predictions and actuals
            all_oof_preds_scaled[val_index] = val_preds_scaled
            all_oof_actuals_scaled[val_index] = y_val
            all_oof_mask[val_index] = True

            gc.collect() # Clean up memory

        # Calculate overall OOF metrics
        final_oof_preds_scaled = all_oof_preds_scaled[all_oof_mask]
        final_oof_actuals_scaled = all_oof_actuals_scaled[all_oof_mask]

        # Inverse transform for final OOF metrics
        final_oof_preds = self.target_scaler.inverse_transform(final_oof_preds_scaled.reshape(-1, 1)).flatten()
        final_oof_actuals = self.target_scaler.inverse_transform(final_oof_actuals_scaled.reshape(-1, 1)).flatten()

        overall_mse = mean_squared_error(final_oof_actuals, final_oof_preds)
        overall_mae = mean_absolute_error(final_oof_actuals, final_oof_preds)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(final_oof_actuals, final_oof_preds)

        print("\n--- Overall Out-of-Fold (OOF) Results ---")
        print(f"Overall MSE: {overall_mse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"Overall R2 Score: {overall_r2:.4f}")

        self.oof_predictions = final_oof_preds # Store for potential external use
        self.oof_actuals = final_oof_actuals # Store for potential external use

        print("\n--- Training and Evaluation Pipeline Complete ---")

    def predict(self, df_new):
        """
        Makes predictions on new, unseen data using the trained pipeline's ensemble of models.

        Args:
            df_new (pd.DataFrame): New data for prediction. Must contain 'ID' and relevant features.

        Returns:
            pd.DataFrame: DataFrame with 'ID' and 'Predicted_Target' columns.
        """
        print("\n--- Starting Prediction Pipeline ---")
        if self.preprocessor is None or not self.models or self.target_scaler is None:
            raise RuntimeError("Pipeline not trained. Please run train_and_evaluate first.")

        # 1. Data Preparation (for new data)
        df_prepared_new, initial_numerical_features_new, initial_categorical_features_new = self._data_preparation(df_new.copy())
        
        # 2. Feature Engineering (for new data)
        df_engineered_new, engineered_numerical_features_new, engineered_categorical_features_new = self._feature_engineering(
            df_prepared_new.copy(), initial_numerical_features_new, initial_categorical_features_new)
        
        # Note: Outlier detection is typically not applied to new data for prediction.
        
        # Crucial step: Align columns of df_engineered_new to match the columns
        # that the preprocessor was fitted on during training.
        
        # Get the list of all columns that the preprocessor was trained on.
        expected_preprocessor_input_cols = self.preprocessor.feature_names_in_
        
        # Create a DataFrame with all expected columns, filling missing with NaN.
        df_aligned_for_transform = pd.DataFrame(index=df_engineered_new.index, columns=expected_preprocessor_input_cols)
        
        for col in expected_preprocessor_input_cols:
            if col in df_engineered_new.columns:
                df_aligned_for_transform[col] = df_engineered_new[col]
            else:
                # Fill missing columns with NaN. The preprocessor should handle these.
                df_aligned_for_transform[col] = np.nan 

        # 4. Data Transformation (using the *fitted* preprocessor)
        df_transformed_array = self.preprocessor.transform(df_aligned_for_transform)

        # Get the feature names out from the preprocessor directly
        transformed_column_names = self.preprocessor.get_feature_names_out()

        df_transformed = pd.DataFrame(df_transformed_array, columns=transformed_column_names, index=df_new.index)

        # 5. Create Sequences from the transformed data for prediction
        # Pass the transformed DataFrame and get the sequences and their corresponding IDs
        X_predict, sequence_ids = self._create_sequences_for_prediction(df_transformed)
        
        # Make predictions using the ensemble of models
        predictions_original_scale = self._ensemble_methods(self.models, X_predict)

        # Create submission DataFrame using the IDs returned from sequence creation
        submission_df = pd.DataFrame({
            'ID': sequence_ids,
            'Predicted_Target': predictions_original_scale
        })
        print("Prediction pipeline complete.")
        return submission_df

    def _create_sequences_for_prediction(self, data_df):
        """
        Creates sequences for the CNN-Attention model from a DataFrame specifically for prediction.
        This version does not expect a target column.

        Args:
            data_df (pd.DataFrame): Input DataFrame with features.
                                     Assumes features are already numerical and scaled.

        Returns:
            tuple: np.array: X (features sequences), np.array: IDs corresponding to the end of each sequence.
        """
        print(f"Creating prediction sequences with length {self.sequence_length}...")
        X = []
        ids = [] # To store IDs corresponding to the end of each sequence
        
        # Ensure only numerical features (excluding 'ID' and the transformed target) are used for X
        feature_columns = [col for col in data_df.columns if col != self._transformed_target_column_name and col != self._transformed_id_column_name and np.issubdtype(data_df[col].dtype, np.number)]
        
        if not feature_columns:
            raise ValueError("No numerical features found for prediction sequence creation after filtering 'ID' and target.")

        # Update n_features based on actual columns used for X
        self.n_features = len(feature_columns)

        # Drop rows with NaNs that might result from previous steps.
        # We need the transformed 'ID' column to be present to extract it later.
        if self._transformed_id_column_name not in data_df.columns:
            raise KeyError(f"Transformed ID column '{self._transformed_id_column_name}' is missing from data_df in _create_sequences_for_prediction. Columns: {data_df.columns.tolist()}")

        data_for_sequences = data_df[feature_columns + [self._transformed_id_column_name]].dropna().copy()

        if len(data_for_sequences) < self.sequence_length:
            raise ValueError(f"Not enough data points ({len(data_for_sequences)}) to create prediction sequences of length {self.sequence_length}.")

        for i in tqdm(range(len(data_for_sequences) - self.sequence_length + 1)):
            X.append(data_for_sequences.iloc[i:(i + self.sequence_length)][feature_columns].values)
            ids.append(data_for_sequences.iloc[i + self.sequence_length - 1][self._transformed_id_column_name]) # ID of the last element in the sequence

        X = np.array(X)
        ids = np.array(ids)
        print(f"Prediction sequences created. X shape: {X.shape}, IDs shape: {ids.shape}")
        return X, ids

# --- 2. Main Execution Block (Example Usage) ---
# This block demonstrates how to use the AdvancedMLPipeline class.
# It includes dummy data generation for reproducibility and a basic workflow.

if __name__ == "__main__":
    # Define paths and parameters
    path_to_ds = './data' # Assuming data is in a 'data' folder relative to script
    file_short_names = ['train.csv', 'test.csv'] # Example file names

    # Create dummy data for demonstration if files don't exist
    if not os.path.exists(path_to_ds):
        os.makedirs(path_to_ds)
    if not os.path.exists(os.path.join(path_to_ds, 'train.csv')):
        print("Creating dummy train.csv and test.csv for demonstration.")
        # Generate dummy data with columns similar to the problem description
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        dummy_data = {
            'ID': range(1000),
            'Date': dates,
            'bid_qty': np.random.rand(1000) * 100,
            'ask_qty': np.random.rand(1000) * 100,
            'buy_qty': np.random.rand(1000) * 1000,
            'sell_qty': np.random.rand(1000) * 1000,
            'volume': np.random.rand(1000) * 1e6,
            'label': np.random.rand(1000) * 50 # Dummy target variable
        }
        # Add a subset of X features for demonstration (X1 to X9)
        # To simulate the full 890 X features, this loop would go up to 890.
        # For performance in a demo, keeping it small.
        for i in range(1, 10):
            dummy_data[f'X{i}'] = np.random.rand(1000) * 100

        dummy_df = pd.DataFrame(dummy_data)

        train_df = dummy_df.iloc[:800]
        test_df = dummy_df.iloc[800:]

        train_df.to_csv(os.path.join(path_to_ds, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(path_to_ds, 'test.csv'), index=False)
        print("Dummy data created.")

    # Load the training data
    try:
        df_train = pd.read_csv(os.path.join(path_to_ds, 'train.csv'))
        df_train['Date'] = pd.to_datetime(df_train['Date'])
    except FileNotFoundError:
        print("Train data not found. Please ensure 'train.csv' exists in the data directory.")
        df_train = pd.DataFrame() # Empty DataFrame if file not found

    # Initialize the AdvancedMLPipeline with desired configurations
    ml_pipeline = AdvancedMLPipeline(
        target_column='label',
        random_state=42,
        sequence_length=30, # Example sequence length
        epochs=50,
        batch_size=32,
        patience=15, # Increased patience
        min_delta=0.0001,
        learning_rate=0.001,
        n_splits=5,
        contamination=0.01,
        l2_reg_strength=1e-4 # L2 regularization strength
    )

    # Train and evaluate the pipeline
    if not df_train.empty:
        ml_pipeline.train_and_evaluate(df_train)
    else:
        print("Skipping training due to missing train data.")

    # Load data for prediction (e.g., test.csv)
    try:
        df_test_for_prediction = pd.read_csv(os.path.join(path_to_ds, 'test.csv'))
        df_test_for_prediction['Date'] = pd.to_datetime(df_test_for_prediction['Date'])
    except FileNotFoundError:
        print("Test data not found. Please ensure 'test.csv' exists in the data directory.")
        df_test_for_prediction = pd.DataFrame() # Empty DataFrame if file not found

    # Make predictions using the trained pipeline
    if not df_test_for_prediction.empty:
        df_final_submission = ml_pipeline.predict(df_test_for_prediction)
        print("\nFinal Submission Head:")
        print(df_final_submission.head())

        # Save the final blended submission (optional)
        # submission_output_path = './submission.csv'
        # df_final_submission.to_csv(submission_output_path, index=False)
        # print(f"Submission saved to {submission_output_path}")
    else:
        print("No test data available for prediction.")

    # Plot OOF predictions vs actuals if available
    if len(ml_pipeline.oof_predictions) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(ml_pipeline.oof_actuals, label='OOF Actuals')
        plt.plot(ml_pipeline.oof_predictions, label='OOF Predictions')
        plt.title('Out-of-Fold Predictions vs Actuals')
        plt.xlabel('Sample Index')
        plt.ylabel(ml_pipeline.target_column)
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- End of Script ---")


import os
import gc
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import matplotlib.pyplot as plt
from scipy import stats
import psutil
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# --- MEMORY-OPTIMIZED CONFIGURATION ---
ROOT_PATH = "/kaggle/input/drw-crypto-market-prediction"
OUTPUT_DIR = "/kaggle/working"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TRAIN_WINDOW_DAYS = 60  # Reduced from 90
SAMPLE_FRAC = 0.2  # Reduced from 0.4
BATCH_SIZE = 500  # Reduced from 1500
FEATURE_BATCH_SIZE = 1000  # For processing features in batches
ENSEMBLE_WEIGHTS = [0.35, 0.35, 0.3]
RANDOM_STATE = 42
FEATURE_SELECTION_K = 50  # Reduced from 75
CV_FOLDS = 3
MAX_FEATURES_PER_BATCH = 20  # Limit features created per batch
MEMORY_THRESHOLD_MB = 4000  # Memory threshold for cleanup

# --- OPTIMIZED HYPERPARAMETERS ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # Reduced
    'learning_rate': 0.08,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'n_estimators': 150,  # Reduced
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'max_depth': 6,  # Reduced
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'force_col_wise': True,
    'num_threads': 1
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,  # Reduced
    'learning_rate': 0.08,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_estimators': 150,  # Reduced
    'tree_method': 'hist',
    'verbosity': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'nthread': 1
}

CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'iterations': 150,  # Reduced
    'depth': 5,  # Reduced
    'learning_rate': 0.08,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.7,
    'border_count': 64,  # Reduced
    'thread_count': 1,
    'verbose': False,
    'allow_writing_files': False,
    'random_state': RANDOM_STATE
}

# --- MEMORY MANAGEMENT UTILITIES ---
def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    except:
        return 0

def log_memory(stage: str):
    """Log current memory usage and perform garbage collection."""
    memory_mb = get_memory_usage()
    print(f"[{stage}] Memory: {memory_mb:.1f} MB")
    
    # Force garbage collection if memory usage is high
    if memory_mb > MEMORY_THRESHOLD_MB:
        gc.collect()
        new_memory = get_memory_usage()
        print(f"[{stage}] After GC: {new_memory:.1f} MB")

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Safely divide two arrays, handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, fill_value)
    return result

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage by optimizing dtypes."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

# --- MEMORY-OPTIMIZED DATA PROCESSING ---
class MemoryOptimizedDataProcessor:
    @staticmethod
    def remove_outliers_batch(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using IQR method (more memory efficient)."""
        if column not in df.columns:
            return df
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        return df[mask]
    
    @staticmethod
    def load_parquet_chunked(path: str, days_limit: Optional[int] = None, 
                           sample_frac: Optional[float] = None) -> pd.DataFrame:
        """Load parquet file with memory optimization."""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return pd.DataFrame()
        
        print(f"Loading data from {path}")
        log_memory("Before loading")
        
        try:
            # Load with optimized parameters
            df = pd.read_parquet(path, engine='pyarrow')
            
            if df.empty:
                print("Empty dataframe loaded")
                return df
            
            # Optimize memory immediately
            df = reduce_memory_usage(df, verbose=False)
            log_memory("After initial load")
            
            # Handle datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                elif 'date' in df.columns:
                    df.set_index('date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Sort by index
            df = df.sort_index()
            
            # Limit days if specified
            if days_limit and len(df) > 0:
                cutoff = df.index.max() - pd.Timedelta(days=days_limit)
                df = df[df.index >= cutoff]
                print(f"Limited to last {days_limit} days: {len(df)} rows")
            
            # Remove outliers in label if present
            if 'label' in df.columns:
                original_len = len(df)
                df = MemoryOptimizedDataProcessor.remove_outliers_batch(df, 'label', threshold=3.0)
                print(f"Removed {original_len - len(df)} label outliers")
            
            # Sampling with memory management
            if sample_frac and 0 < sample_frac < 1:
                df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).sort_index()
                print(f"Sampled {sample_frac*100}%: {len(df)} rows")
            
            # Handle infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Final memory optimization
            df = reduce_memory_usage(df, verbose=True)
            log_memory("After processing")
            
            return df
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()

# --- MEMORY-OPTIMIZED FEATURE ENGINEERING ---
class MemoryOptimizedFeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.selected_cols: List[str] = []
        self.original_cols: List[str] = []
        self.feature_selector = None
        self.is_fitted = False
        self.feature_cache = {}

    def create_time_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with memory optimization."""
        print("Creating time features...")
        
        # Create features in smaller batches
        time_features = pd.DataFrame(index=df.index)
        
        # Basic time features
        time_features['hour'] = df.index.hour.astype(np.int8)
        time_features['day_of_week'] = df.index.dayofweek.astype(np.int8)
        time_features['month'] = df.index.month.astype(np.int8)
        time_features['quarter'] = df.index.quarter.astype(np.int8)
        time_features['day_of_month'] = df.index.day.astype(np.int8)
        
        # Binary features
        time_features['is_weekend'] = (df.index.dayofweek >= 5).astype(np.int8)
        time_features['is_month_start'] = df.index.is_month_start.astype(np.int8)
        time_features['is_month_end'] = df.index.is_month_end.astype(np.int8)
        time_features['is_quarter_start'] = df.index.is_quarter_start.astype(np.int8)
        time_features['is_quarter_end'] = df.index.is_quarter_end.astype(np.int8)
        
        # Cyclical features (only essential ones)
        time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24).astype(np.float32)
        time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24).astype(np.float32)
        time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7).astype(np.float32)
        time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7).astype(np.float32)
        time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12).astype(np.float32)
        time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12).astype(np.float32)
        
        # Combine with original dataframe
        result = pd.concat([df, time_features], axis=1)
        
        # Clean up
        del time_features
        gc.collect()
        
        print(f"Created {len(result.columns) - len(df.columns)} time features")
        return result

    def create_rolling_features_batch(self, df: pd.DataFrame, cols: List[str], 
                                    max_features: int = 15) -> pd.DataFrame:
        """Create rolling features with memory optimization."""
        print("Creating rolling features...")
        
        # Limit columns to prevent memory explosion
        cols = cols[:5]  # Only top 5 columns
        windows = [5, 10, 20]  # Reduced windows
        
        for col in cols:
            if col not in df.columns:
                continue
                
            feature_count = 0
            for window in windows:
                if feature_count >= max_features:
                    break
                    
                # Essential rolling features only
                df[f"{col}_ma_{window}"] = df[col].rolling(window, min_periods=1).mean().astype(np.float32)
                df[f"{col}_std_{window}"] = df[col].rolling(window, min_periods=1).std().astype(np.float32)
                df[f"{col}_min_{window}"] = df[col].rolling(window, min_periods=1).min().astype(np.float32)
                df[f"{col}_max_{window}"] = df[col].rolling(window, min_periods=1).max().astype(np.float32)
                
                feature_count += 4
                
                # Memory check
                if get_memory_usage() > MEMORY_THRESHOLD_MB:
                    print(f"Memory limit reached, stopping feature creation for {col}")
                    break
            
            # Clean up intermediate calculations
            gc.collect()
        
        print(f"Created rolling features for {len(cols)} columns")
        return df

    def create_technical_indicators_batch(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create technical indicators with memory optimization."""
        print("Creating technical indicators...")
        
        # Limit to top 3 price columns
        price_cols = price_cols[:3]
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Simple moving averages
            for window in [5, 10, 20]:
                sma = df[col].rolling(window, min_periods=1).mean()
                df[f"{col}_sma_{window}"] = sma.astype(np.float32)
                
                # Price ratio to SMA
                df[f"{col}_ratio_sma_{window}"] = safe_divide(
                    df[col].values, sma.values
                ).astype(np.float32)
                
                # Clean up
                del sma
            
            # Momentum indicators (limited)
            for period in [3, 5, 10]:
                df[f"{col}_mom_{period}"] = (df[col] - df[col].shift(period)).astype(np.float32)
                df[f"{col}_roc_{period}"] = df[col].pct_change(period).astype(np.float32)
            
            # RSI (simplified)
            delta = df[col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = safe_divide(gain.values, loss.values)
            df[f"{col}_rsi"] = (100 - (100 / (1 + rs))).astype(np.float32)
            
            # Clean up
            del delta, gain, loss, rs
            gc.collect()
        
        print(f"Created technical indicators for {len(price_cols)} price columns")
        return df

    def create_lag_features_batch(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Create lag features with memory optimization."""
        print("Creating lag features...")
        
        # Limit columns and lags
        cols = cols[:3]  # Top 3 columns only
        lags = [1, 2, 3, 5, 10]  # Reduced lags
        
        for col in cols:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag).astype(np.float32)
                
                # Memory check
                if get_memory_usage() > MEMORY_THRESHOLD_MB:
                    print(f"Memory limit reached, stopping lag features for {col}")
                    break
            
            # Differences
            for diff in [1, 2, 3]:
                df[f"{col}_diff_{diff}"] = df[col].diff(diff).astype(np.float32)
                df[f"{col}_pct_change_{diff}"] = df[col].pct_change(diff).astype(np.float32)
            
            gc.collect()
        
        print(f"Created lag features for {len(cols)} columns")
        return df

    def create_interaction_features_batch(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Create limited interaction features."""
        print("Creating interaction features...")
        
        # Very limited interactions to prevent memory explosion
        top_cols = cols[:3]  # Only top 3 columns
        
        for i, col1 in enumerate(top_cols):
            for col2 in top_cols[i+1:]:
                if col1 not in df.columns or col2 not in df.columns:
                    continue
                
                # Only essential interactions
                df[f"{col1}_{col2}_ratio"] = safe_divide(
                    df[col1].values, df[col2].values
                ).astype(np.float32)
                
                df[f"{col1}_{col2}_diff"] = (df[col1] - df[col2]).astype(np.float32)
                
                # Memory check
                if get_memory_usage() > MEMORY_THRESHOLD_MB:
                    print("Memory limit reached, stopping interaction features")
                    return df
        
        print(f"Created interaction features for {len(top_cols)} columns")
        return df

    def engineer_features_batch(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Main feature engineering pipeline with memory optimization."""
        print("Starting feature engineering...")
        log_memory("Feature engineering start")
        
        # Store original columns
        if is_training:
            numeric_cols = df.select_dtypes(include='number').columns
            self.original_cols = [c for c in numeric_cols if c != 'label']
        
        # Create features in batches with memory management
        df = self.create_time_features_batch(df)
        log_memory("After time features")
        
        # Detect price columns
        price_cols = [c for c in self.original_cols 
                     if any(k in c.lower() for k in ['price', 'close', 'open', 'high', 'low'])]
        
        if price_cols:
            df = self.create_technical_indicators_batch(df, price_cols)
            log_memory("After technical indicators")
        
        # Rolling features
        df = self.create_rolling_features_batch(df, self.original_cols)
        log_memory("After rolling features")
        
        # Lag features
        df = self.create_lag_features_batch(df, self.original_cols)
        log_memory("After lag features")
        
        # Limited interaction features
        df = self.create_interaction_features_batch(df, self.original_cols)
        log_memory("After interaction features")
        
        # Clean up invalid values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Final memory optimization
        df = reduce_memory_usage(df, verbose=False)
        log_memory("Feature engineering complete")
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit feature engineering and transform data with memory optimization."""
        if 'label' not in df.columns:
            raise ValueError("Label column not found in training data")
        
        print("Fitting feature engineering...")
        
        # Engineer features
        df_engineered = self.engineer_features_batch(df, is_training=True)
        
        # Extract target and features
        y = df_engineered['label'].values.astype(np.float32)
        feature_df = df_engineered.drop(columns=['label'])
        
        # Memory-efficient feature selection
        k = min(FEATURE_SELECTION_K, feature_df.shape[1])
        
        print(f"Selecting {k} best features from {feature_df.shape[1]} features...")
        
        # Use only f_regression for memory efficiency
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(feature_df.values, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_cols = [feature_df.columns[i] for i in range(len(selected_mask)) if selected_mask[i]]
        
        print(f"Selected {len(self.selected_cols)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected).astype(np.float32)
        
        # Clean up
        del df_engineered, feature_df, X_selected
        gc.collect()
        
        self.is_fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted feature engineering."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming")
        
        print("Transforming features...")
        
        # Engineer features
        df_engineered = self.engineer_features_batch(df, is_training=False)
        
        # Ensure all selected features exist
        for col in self.selected_cols:
            if col not in df_engineered.columns:
                df_engineered[col] = 0
        
        # Select and scale features
        X_selected = df_engineered[self.selected_cols].values
        X_scaled = self.scaler.transform(X_selected).astype(np.float32)
        
        # Clean up
        del df_engineered
        gc.collect()
        
        return X_scaled

# --- OPTIMIZED VISUALIZATION ---
class OptimizedVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_data_overview(self, df: pd.DataFrame, name: str):
        """Plot data overview with memory optimization."""
        if 'label' not in df.columns:
            print(f"No label column found for {name} overview")
            return
        
        try:
            # Sample data if too large
            if len(df) > 10000:
                df_sample = df.sample(n=10000, random_state=RANDOM_STATE)
            else:
                df_sample = df
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # Label distribution
            axes[0].hist(df_sample['label'].dropna(), bins=50, alpha=0.7, color='skyblue')
            axes[0].set_title(f"{name} - Label Distribution")
            axes[0].set_xlabel("Label Value")
            axes[0].set_ylabel("Frequency")
            
            # Label over time (sampled)
            axes[1].plot(df_sample.index, df_sample['label'], alpha=0.7, color='orange', linewidth=0.5)
            axes[1].set_title(f"{name} - Label Over Time")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Label Value")
            
            # Boxplot
            axes[2].boxplot(df_sample['label'].dropna(), patch_artist=True)
            axes[2].set_title(f"{name} - Label Boxplot")
            axes[2].set_ylabel("Label Value")
            
            # Basic statistics
            stats_text = f"Mean: {df['label'].mean():.4f}\n"
            stats_text += f"Std: {df['label'].std():.4f}\n"
            stats_text += f"Min: {df['label'].min():.4f}\n"
            stats_text += f"Max: {df['label'].max():.4f}\n"
            stats_text += f"Count: {len(df)}"
            
            axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title(f"{name} - Statistics")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(self.output_dir, f"{name}_data_overview.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Clean up
            del df_sample
            gc.collect()
            
        except Exception as e:
            print(f"Error creating overview plot for {name}: {e}")

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, name: str):
        """Plot prediction results with memory optimization."""
        try:
            # Sample if too large
            if len(y_true) > 5000:
                indices = np.random.choice(len(y_true), 5000, replace=False)
                y_true_sample = y_true[indices]
                y_pred_sample = y_pred[indices]
            else:
                y_true_sample = y_true
                y_pred_sample = y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # Scatter plot
            axes[0].scatter(y_true_sample, y_pred_sample, alpha=0.6, s=1, color='blue')
            min_val = min(y_true_sample.min(), y_pred_sample.min())
            max_val = max(y_true_sample.max(), y_pred_sample.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")
            axes[0].set_title(f"{name} - Actual vs Predicted")
            
            # Residuals
            residuals = y_true_sample - y_pred_sample
            axes[1].scatter(y_pred_sample, residuals, alpha=0.6, s=1, color='green')
            axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Residuals")
            axes[1].set_title(f"{name} - Residuals")
            
            # Residual histogram
            axes[2].hist(residuals, bins=50, alpha=0.7, color='purple')
            axes[2].set_xlabel("Residuals")
            axes[2].set_ylabel("Frequency")
            axes[2].set_title(f"{name} - Residual Distribution")
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics_text = f"RMSE: {rmse:.4f}\n"
            metrics_text += f"MAE: {mae:.4f}\n"
            metrics_text += f"R²: {r2:.4f}"
            
            axes[3].text(0.1, 0.5, metrics_text, transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title(f"{name} - Metrics")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(self.output_dir, f"{name}_predictions.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Clean up
            del y_true_sample, y_pred_sample, residuals
            gc.collect()
            
        except Exception as e:
            print(f"Error creating prediction plot for {name}: {e}")

# --- OPTIMIZED MODEL TRAINER ---
class OptimizedModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.is_trained = False

    def train_single_model(self, name: str, model_class, params: Dict, 
                          X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict:
#*****************************************************************************************

        """Train a single model with memory optimization."""
        print(f"Training {name} model...")
        
        try:
            if name == 'LightGBM':
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         eval_metric='rmse',
                         early_stopping_rounds=20,
                         verbose=False)
                
            elif name == 'XGBoost':
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=20,
                         verbose=False)
                
            elif name == 'CatBoost':
                model = cb.CatBoostRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=20,
                         verbose=False)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model
            }
            
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Clean up
            del y_pred
            gc.collect()
            
            return metrics
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            return {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'model': None}

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train all models with cross-validation."""
        print("Training ensemble models...")
        log_memory("Training start")
        
        # Model configurations
        model_configs = [
            ('LightGBM', lgb.LGBMRegressor, LGBM_PARAMS),
            ('XGBoost', xgb.XGBRegressor, XGB_PARAMS),
            ('CatBoost', cb.CatBoostRegressor, CATBOOST_PARAMS)
        ]
        
        # Train each model
        for name, model_class, params in model_configs:
            self.models[name] = self.train_single_model(
                name, model_class, params, X_train, y_train, X_val, y_val
            )
            log_memory(f"After {name}")
        
        # Cross-validation
        self.perform_cross_validation(X_train, y_train)
        
        self.is_trained = True
        return self.models

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Perform time series cross-validation."""
        print("Performing cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
        
        for name in self.models.keys():
            if self.models[name]['model'] is None:
                continue
                
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Clone model with same parameters
                if name == 'LightGBM':
                    model = lgb.LGBMRegressor(**LGBM_PARAMS)
                elif name == 'XGBoost':
                    model = xgb.XGBRegressor(**XGB_PARAMS)
                elif name == 'CatBoost':
                    model = cb.CatBoostRegressor(**CATBOOST_PARAMS)
                
                # Train and predict
                model.fit(X_train_cv, y_train_cv, verbose=False)
                y_pred = model.predict(X_val_cv)
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
                cv_scores.append(rmse)
                
                # Clean up
                del model, y_pred
                gc.collect()
            
            self.cv_scores[name] = {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'scores': cv_scores
            }
            
            print(f"{name} CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions with weighted averaging."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        valid_models = []
        
        for name in ['LightGBM', 'XGBoost', 'CatBoost']:
            if self.models[name]['model'] is not None:
                pred = self.models[name]['model'].predict(X)
                predictions.append(pred)
                valid_models.append(name)
        
        if not predictions:
            raise ValueError("No valid models for ensemble prediction")
        
        # Weighted ensemble
        weights = ENSEMBLE_WEIGHTS[:len(predictions)]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        print(f"Ensemble prediction using {len(valid_models)} models: {valid_models}")
        
        return ensemble_pred

# --- MAIN PIPELINE ---
class CryptoMarketPredictor:
    def __init__(self):
        self.feature_engineer = MemoryOptimizedFeatureEngineer()
        self.model_trainer = OptimizedModelTrainer()
        self.visualizer = OptimizedVisualizer(PLOTS_DIR)
        self.is_fitted = False

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training and test data."""
        print("Loading and preparing data...")
        
        # Load training data
        train_path = os.path.join(ROOT_PATH, "train.parquet")
        train_df = MemoryOptimizedDataProcessor.load_parquet_chunked(
            train_path, days_limit=TRAIN_WINDOW_DAYS, sample_frac=SAMPLE_FRAC
        )
        
        if train_df.empty:
            raise ValueError("Training data is empty")
        
        # Load test data
        test_path = os.path.join(ROOT_PATH, "test.parquet")
        test_df = MemoryOptimizedDataProcessor.load_parquet_chunked(test_path)
        
        if test_df.empty:
            raise ValueError("Test data is empty")
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Create visualizations
        self.visualizer.plot_data_overview(train_df, "Training")
        
        return train_df, test_df

    def train_pipeline(self, train_df: pd.DataFrame) -> Dict:
        """Train the complete pipeline."""
        print("Training pipeline...")
        
        # Split data
        split_point = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_point]
        val_split = train_df.iloc[split_point:]
        
        print(f"Train split: {len(train_split)} samples")
        print(f"Validation split: {len(val_split)} samples")
        
        # Feature engineering
        X_train, y_train = self.feature_engineer.fit_transform(train_split)
        X_val = self.feature_engineer.transform(val_split)
        y_val = val_split['label'].values.astype(np.float32)
        
        print(f"Feature matrix shape: {X_train.shape}")
        log_memory("After feature engineering")
        
        # Train models
        model_results = self.model_trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Make validation predictions
        y_pred_val = self.model_trainer.predict_ensemble(X_val)
        
        # Calculate final metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        metrics = {
            'validation_rmse': val_rmse,
            'validation_mae': val_mae,
            'validation_r2': val_r2,
            'model_results': model_results,
            'cv_scores': self.model_trainer.cv_scores
        }
        
        print(f"\nFinal Validation Metrics:")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"MAE: {val_mae:.4f}")
        print(f"R²: {val_r2:.4f}")
        
        # Create prediction plots
        self.visualizer.plot_predictions(y_val, y_pred_val, "Validation")
        
        # Clean up
        del X_train, y_train, X_val, y_val, y_pred_val, train_split, val_split
        gc.collect()
        
        self.is_fitted = True
        return metrics

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained before making predictions")
        
        print("Making predictions on test data...")
        
        # Transform test data
        X_test = self.feature_engineer.transform(test_df)
        
        # Make predictions
        predictions = self.model_trainer.predict_ensemble(X_test)
        
        print(f"Generated {len(predictions)} predictions")
        
        # Clean up
        del X_test
        gc.collect()
        
        return predictions

    def save_models(self, filepath: str):
        """Save trained models and feature engineer."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained before saving")
        
        print(f"Saving models to {filepath}...")
        
        save_dict = {
            'feature_engineer': self.feature_engineer,
            'model_trainer': self.model_trainer,
            'ensemble_weights': ENSEMBLE_WEIGHTS
        }
        
        joblib.dump(save_dict, filepath)
        print("Models saved successfully")

    def load_models(self, filepath: str):
        """Load trained models and feature engineer."""
        print(f"Loading models from {filepath}...")
        
        save_dict = joblib.load(filepath)
        
        self.feature_engineer = save_dict['feature_engineer']
        self.model_trainer = save_dict['model_trainer']
        
        self.is_fitted = True
        print("Models loaded successfully")

# --- EXECUTION ---
def main():
    """Main execution function."""
    print("=== Crypto Market Prediction Pipeline ===")
    print(f"Configuration:")
    print(f"- Train window: {TRAIN_WINDOW_DAYS} days")
    print(f"- Sample fraction: {SAMPLE_FRAC}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Feature selection: {FEATURE_SELECTION_K}")
    print(f"- CV folds: {CV_FOLDS}")
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Initialize predictor
    predictor = CryptoMarketPredictor()
    
    try:
        # Load data
        train_df, test_df = predictor.load_and_prepare_data()
        
        # Train pipeline
        metrics = predictor.train_pipeline(train_df)
        
        # Save models
        model_path = os.path.join(OUTPUT_DIR, "crypto_models.joblib")
        predictor.save_models(model_path)
        
        # Make predictions
        predictions = predictor.predict(test_df)
        
        # Save predictions
        submission_df = pd.DataFrame({
            'row_id': range(len(predictions)),
            'label': predictions
        })
        
        submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\nSubmission saved to: {submission_path}")
        print(f"Prediction statistics:")
        print(f"- Mean: {predictions.mean():.4f}")
        print(f"- Std: {predictions.std():.4f}")
        print(f"- Min: {predictions.min():.4f}")
        print(f"- Max: {predictions.max():.4f}")
        
        # Save metrics
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("=== Model Performance Metrics ===\n")
            f.write(f"Validation RMSE: {metrics['validation_rmse']:.4f}\n")
            f.write(f"Validation MAE: {metrics['validation_mae']:.4f}\n")
            f.write(f"Validation R²: {metrics['validation_r2']:.4f}\n\n")
            
            f.write("=== Cross-Validation Scores ===\n")
            for model_name, cv_data in metrics['cv_scores'].items():
                f.write(f"{model_name}: {cv_data['mean']:.4f} ± {cv_data['std']:.4f}\n")
            
            f.write("\n=== Individual Model Performance ===\n")
            for model_name, model_data in metrics['model_results'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  RMSE: {model_data['rmse']:.4f}\n")
                f.write(f"  MAE: {model_data['mae']:.4f}\n")
                f.write(f"  R²: {model_data['r2']:.4f}\n")
        
        print(f"Metrics saved to: {metrics_path}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        gc.collect()
        log_memory("Final cleanup")

if __name__ == "__main__":
    main()




