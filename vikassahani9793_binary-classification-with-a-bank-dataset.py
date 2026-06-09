# Advanced Bank Marketing Binary Classification - Kaggle Competition
# Playground Series - Season 5, Episode 8
# Optimized for Tesla T4 Dual GPU (15GB each)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# GPU and Multi-processing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from itertools import combinations

# Utilities
import gc
import pickle
import joblib
from datetime import datetime
import os

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

print("=== ADVANCED BANK MARKETING CLASSIFICATION (COMPLETE FIXED) ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# =============================================================================
# DATA LOADING AND EXPLORATION
# =============================================================================

def load_and_explore_data():
    """Load and perform comprehensive EDA"""
    print("\nğŸ”� Loading and exploring data...")
    
    # Load data - handle file paths properly
    try:
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
        sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
    except FileNotFoundError:
        # Fallback for local development
        print("Kaggle data not found, creating synthetic data for testing...")
        train_df = create_synthetic_data(750000, include_target=True)
        test_df = create_synthetic_data(250000, include_target=False)
        sample_sub = pd.DataFrame({
            'id': range(750000, 1000000),
            'y': np.zeros(250000)
        })
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")
    
    if 'y' in train_df.columns:
        print(f"\nTarget distribution:")
        print(train_df['y'].value_counts(normalize=True))
    
    # Check for missing values
    print(f"\nMissing values in train: {train_df.isnull().sum().sum()}")
    print(f"Missing values in test: {test_df.isnull().sum().sum()}")
    
    # Data types
    print(f"\nData types:")
    print(train_df.dtypes.value_counts())
    
    return train_df, test_df, sample_sub

def create_synthetic_data(n_samples, include_target=True):
    """Create synthetic data for testing when real data unavailable"""
    np.random.seed(RANDOM_STATE)
    
    data = {
        'id': range(n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'job': np.random.choice(['management', 'technician', 'entrepreneur', 'blue-collar', 
                               'unknown', 'retired', 'admin.', 'services', 'self-employed', 
                               'unemployed', 'housemaid', 'student'], n_samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
        'education': np.random.choice(['university.degree', 'high.school', 'basic.9y', 
                                     'professional.course', 'basic.4y', 'basic.6y', 
                                     'unknown', 'illiterate'], n_samples),
        'default': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.8, 0.1, 0.1]),
        'housing': np.random.choice(['yes', 'no', 'unknown'], n_samples, p=[0.6, 0.3, 0.1]),
        'loan': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.7, 0.2, 0.1]),
        'contact': np.random.choice(['cellular', 'telephone'], n_samples, p=[0.8, 0.2]),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], n_samples),
        'duration': np.random.exponential(200, n_samples).astype(int),
        'campaign': np.random.poisson(2, n_samples) + 1,
        'pdays': np.random.choice([999] + list(range(0, 400)), n_samples, p=[0.8] + [0.2/400]*400),
        'previous': np.random.poisson(0.5, n_samples),
        'poutcome': np.random.choice(['nonexistent', 'failure', 'success'], 
                                   n_samples, p=[0.8, 0.15, 0.05]),
        'emp.var.rate': np.random.normal(0.1, 1.5, n_samples),
        'cons.price.idx': np.random.normal(93.5, 1, n_samples),
        'cons.conf.idx': np.random.normal(-40, 5, n_samples),
        'euribor3m': np.random.normal(3.5, 2, n_samples),
        'nr.employed': np.random.normal(5190, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    if include_target:
        # Create synthetic target with realistic patterns
        prob = (
            0.05 +  # Base probability
            0.1 * (df['age'] > 60).astype(int) +  # Older people more likely
            0.05 * (df['education'] == 'university.degree').astype(int) +  # Education effect
            0.03 * (df['job'] == 'management').astype(int) +  # Job effect
            0.02 * (df['poutcome'] == 'success').astype(int)  # Previous success
        )
        df['y'] = np.random.binomial(1, prob, n_samples)
    
    return df

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        
    def create_interaction_features(self, df, max_interactions=20):
        """Create interaction features between numerical columns"""
        print("Creating interaction features...")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        cols_to_remove = ['y', 'id']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]
        
        interaction_features = []
        count = 0
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                if count >= max_interactions:
                    break
                    
                # Multiplication
                df[f'{col1}_{col2}_mult'] = df[col1] * df[col2]
                # Division (with small epsilon to avoid division by zero)
                df[f'{col1}_{col2}_div'] = df[col1] / (df[col2] + 1e-8)
                # Addition
                df[f'{col1}_{col2}_add'] = df[col1] + df[col2]
                # Subtraction  
                df[f'{col1}_{col2}_sub'] = df[col1] - df[col2]
                
                interaction_features.extend([
                    f'{col1}_{col2}_mult', f'{col1}_{col2}_div',
                    f'{col1}_{col2}_add', f'{col1}_{col2}_sub'
                ])
                count += 4
                
        print(f"Created {len(interaction_features)} interaction features")
        return df, interaction_features
    
    def create_polynomial_features(self, df, degree=2):
        """Create polynomial features"""
        print("Creating polynomial features...")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        cols_to_remove = ['y', 'id']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]
        
        poly_features = []
        for col in numerical_cols[:5]:  # Limit to prevent feature explosion
            # Handle negative values properly
            abs_col = np.abs(df[col])
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
            df[f'{col}_sqrt'] = np.sqrt(abs_col)
            df[f'{col}_log'] = np.log1p(abs_col)
            
            poly_features.extend([
                f'{col}_squared', f'{col}_cubed', 
                f'{col}_sqrt', f'{col}_log'
            ])
            
        print(f"Created {len(poly_features)} polynomial features")
        return df, poly_features
    
    def create_statistical_features(self, df):
        """Create statistical aggregation features"""
        print("Creating statistical features...")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        cols_to_remove = ['y', 'id']
        numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]
        
        statistical_features = []
        
        if len(numerical_cols) > 1:
            # Row-wise statistics
            df_numeric = df[numerical_cols]
            df['row_mean'] = df_numeric.mean(axis=1)
            df['row_std'] = df_numeric.std(axis=1)
            df['row_median'] = df_numeric.median(axis=1)
            df['row_min'] = df_numeric.min(axis=1)
            df['row_max'] = df_numeric.max(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
            statistical_features.extend([
                'row_mean', 'row_std', 'row_median', 
                'row_min', 'row_max', 'row_range'
            ])
        
        print(f"Created {len(statistical_features)} statistical features")
        return df, statistical_features
    
    def encode_categorical_features(self, train_df, test_df):
        """Advanced categorical encoding"""
        print("Encoding categorical features...")
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col == 'id':  # Skip ID column if it's object type
                continue
                
            # Label encoding
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
            le.fit(combined_values)
            
            train_df[f'{col}_encoded'] = le.transform(train_df[col].astype(str))
            test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
            self.label_encoders[col] = le
            
            # Frequency encoding
            freq_map = train_df[col].value_counts().to_dict()
            train_df[f'{col}_freq'] = train_df[col].map(freq_map)
            test_df[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)
            
            # Target encoding (only if target exists)
            if 'y' in train_df.columns:
                target_map = train_df.groupby(col)['y'].mean().to_dict()
                train_df[f'{col}_target_enc'] = train_df[col].map(target_map)
                test_df[f'{col}_target_enc'] = test_df[col].map(target_map).fillna(train_df['y'].mean())
        
        return train_df, test_df
    
    def feature_selection(self, X, y, k=100):
        """Select top k features using multiple methods"""
        print(f"Selecting top {k} features...")
        
        if X.shape[1] <= k:
            print(f"Feature count ({X.shape[1]}) <= k ({k}), using all features")
            return np.arange(X.shape[1])
        
        # Method 1: Univariate selection
        selector_f = SelectKBest(f_classif, k=min(k//2, X.shape[1]))
        selector_f.fit(X, y)
        selected_features_f = selector_f.get_support(indices=True)
        
        # Method 2: Mutual information
        selector_mi = SelectKBest(mutual_info_classif, k=min(k//2, X.shape[1]))
        selector_mi.fit(X, y)
        selected_features_mi = selector_mi.get_support(indices=True)
        
        # Combine both methods
        selected_features = np.unique(np.concatenate([selected_features_f, selected_features_mi]))
        
        # If we still have too few features, add top variance features
        if len(selected_features) < k and X.shape[1] > len(selected_features):
            remaining_features = np.setdiff1d(np.arange(X.shape[1]), selected_features)
            variances = np.var(X[:, remaining_features], axis=0)
            top_var_idx = np.argsort(variances)[::-1][:k-len(selected_features)]
            selected_features = np.concatenate([selected_features, remaining_features[top_var_idx]])
        
        print(f"Selected {len(selected_features)} features")
        return selected_features[:k]  # Ensure we don't exceed k
    
    def fit_transform(self, train_df, test_df):
        """Complete feature engineering pipeline"""
        print("\nğŸ”§ Starting advanced feature engineering...")
        
        # Make copies to avoid modifying originals
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Categorical encoding
        train_df, test_df = self.encode_categorical_features(train_df, test_df)
        
        # Remove original categorical columns (except id)
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        if 'id' in categorical_cols:
            categorical_cols.remove('id')
        
        train_df = train_df.drop(columns=categorical_cols)
        test_df = test_df.drop(columns=categorical_cols)
        
        # Create interaction features
        train_df, interaction_features = self.create_interaction_features(train_df)
        test_df, _ = self.create_interaction_features(test_df)
        
        # Create polynomial features
        train_df, poly_features = self.create_polynomial_features(train_df)
        test_df, _ = self.create_polynomial_features(test_df)
        
        # Create statistical features
        train_df, stat_features = self.create_statistical_features(train_df)
        test_df, _ = self.create_statistical_features(test_df)
        
        print(f"Total features after engineering: {train_df.shape[1]}")
        
        return train_df, test_df

# =============================================================================
# CORRECTED DEEP LEARNING MODELS
# =============================================================================

class TabularNN(nn.Module):
    """Fixed TabNet-inspired neural network"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(TabularNN, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Feature transformer
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.feature_transformer = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(prev_dim, num_heads=8, batch_first=True, dropout=dropout)
        
        # Final classifier - FIXED: Use BCEWithLogitsLoss, so no sigmoid here
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, 1)  # No sigmoid - will be handled by loss function
        )
        
    def forward(self, x):
        x = self.input_bn(x)
        features = self.feature_transformer(x)
        
        # Add attention (reshape for attention mechanism)
        features_reshaped = features.unsqueeze(1)
        attended, _ = self.attention(features_reshaped, features_reshaped, features_reshaped)
        attended = attended.squeeze(1)
        
        # Residual connection
        features = features + attended
        
        output = self.classifier(features)
        return output.squeeze()

def train_neural_network(X_train, y_train, X_val, y_val, device, class_weight_ratio=7.3):
    """CORRECTED: Train neural network with proper architecture and loss handling"""
    print("Training neural network...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)  # Reduced workers for stability
    
    # Initialize model
    model = TabularNN(X_train.shape[1], hidden_dims=[512, 256, 128], dropout=0.3).to(device)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    # FIXED: Use BCEWithLogitsLoss with proper class weighting
    pos_weight = torch.tensor([class_weight_ratio]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_auc = 0
    patience_counter = 0
    patience = 15
    
    print("Starting neural network training...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass - model outputs logits, loss function handles sigmoid
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits)  # Apply sigmoid for evaluation
            val_auc = roc_auc_score(y_val, val_probs.cpu().numpy())
        
        scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            avg_train_loss = train_loss / max(batch_count, 1)
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Best validation AUC: {best_val_auc:.6f}")
    
    return model

# =============================================================================
# GRADIENT BOOSTING MODELS WITH PROPER ERROR HANDLING
# =============================================================================

def train_lightgbm_gpu(X_train, y_train, X_val, y_val):
    """Train LightGBM with GPU acceleration and proper error handling"""
    print("Training LightGBM...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'random_state': RANDOM_STATE,
        'force_row_wise': True  # For GPU compatibility
    }
    
    try:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
    except Exception as e:
        print(f"GPU training failed, falling back to CPU: {e}")
        params['device'] = 'cpu'
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
    
    return model

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    """Train XGBoost with GPU acceleration and fallback"""
    print("Training XGBoost...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'verbosity': 1
    }
    
    try:
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
    except Exception as e:
        print(f"GPU training failed, falling back to CPU: {e}")
        params['tree_method'] = 'hist'
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
    
    return model

def train_catboost_gpu(X_train, y_train, X_val, y_val):
    """Train CatBoost with GPU acceleration and fallback"""
    print("Training CatBoost...")
    
    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'random_strength': 1,
        'one_hot_max_size': 2,
        'random_seed': RANDOM_STATE,
        'verbose': 100,
        'early_stopping_rounds': 50
    }
    
    if torch.cuda.is_available():
        params.update({
            'task_type': 'GPU',
            'devices': '0'
        })
    
    try:
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100
        )
    except Exception as e:
        print(f"GPU training failed, falling back to CPU: {e}")
        params_cpu = {k: v for k, v in params.items() if k not in ['task_type', 'devices']}
        model = cb.CatBoostClassifier(**params_cpu)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100
        )
    
    return model

# =============================================================================
# COMPLETE ENSEMBLE IMPLEMENTATION
# =============================================================================

class SimplifiedEnsemble:
    """Complete ensemble with proper error handling"""
    
    def __init__(self, device):
        self.base_models = {}
        self.model_weights = {}
        self.device = device
        
    def add_model(self, model, name, weight=1.0):
        """Add a model to the ensemble"""
        self.base_models[name] = model
        self.model_weights[name] = weight
        print(f"Added model '{name}' with weight {weight}")
        
    def predict(self, X):
        """Make ensemble predictions using weighted average"""
        predictions = []
        total_weight = 0
        
        for name, model in self.base_models.items():
            try:
                weight = self.model_weights[name]
                
                if name == 'neural_network':
                    # Handle neural network prediction
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        pred = torch.sigmoid(model(X_tensor)).cpu().numpy()
                elif hasattr(model, 'predict_proba'):
                    # Sklearn-style models
                    pred = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'predict'):
                    # XGBoost/LightGBM style
                    if name in ['xgboost']:
                        dtest = xgb.DMatrix(X)
                        pred = model.predict(dtest)
                    else:
                        pred = model.predict(X)
                else:
                    print(f"Warning: Could not make prediction with model {name}")
                    continue
                
                predictions.append(pred * weight)
                total_weight += weight
                
            except Exception as e:
                print(f"Error predicting with model {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model!")
        
        # Weighted average
        final_predictions = np.sum(predictions, axis=0) / total_weight
        return final_predictions

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline"""
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load and explore data
    train_df, test_df, sample_sub = load_and_explore_data()
    
    # Feature engineering
    feature_engineer = AdvancedFeatureEngineer()
    train_processed, test_processed = feature_engineer.fit_transform(train_df, test_df)
    
    # Prepare features and target
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'y']]
    X = train_processed[feature_cols].values
    y = train_processed['y'].values if 'y' in train_processed.columns else None
    X_test = test_processed[feature_cols].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    
    if y is not None:
        print(f"Target distribution: {np.bincount(y.astype(int))}")
        
        # Feature selection
        selected_features = feature_engineer.feature_selection(X, y, k=150)
        X = X[:, selected_features]
        X_test = X_test[:, selected_features]
        
        print(f"Selected feature matrix shape: {X.shape}")
        
        # Handle infinite and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Calculate class weights
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        class_weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Class imbalance ratio: {class_weight_ratio:.2f}")
        
        # Initialize ensemble
        ensemble = SimplifiedEnsemble(device)
        
        # Train models
        print("\nğŸš€ Starting model training...")
        
        # 1. Neural Network
        try:
            print("\n1. Training Neural Network...")
            nn_model = train_neural_network(X_train, y_train, X_val, y_val, device, class_weight_ratio)
            ensemble.add_model(nn_model, 'neural_network', weight=0.3)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Neural network training failed: {e}")
        
        # 2. LightGBM
        try:
            print("\n2. Training LightGBM...")
            lgb_model = train_lightgbm_gpu(X_train, y_train, X_val, y_val)
            ensemble.add_model(lgb_model, 'lightgbm', weight=0.25)
            
        except Exception as e:
            print(f"LightGBM training failed: {e}")
        
        # 3. XGBoost
        try:
            print("\n3. Training XGBoost...")
            xgb_model = train_xgboost_gpu(X_train, y_train, X_val, y_val)
            ensemble.add_model(xgb_model, 'xgboost', weight=0.25)
            
        except Exception as e:
            print(f"XGBoost training failed: {e}")
        
        # 4. CatBoost
        try:
            print("\n4. Training CatBoost...")
            cat_model = train_catboost_gpu(X_train, y_train, X_val, y_val)
            ensemble.add_model(cat_model, 'catboost', weight=0.2)
            
        except Exception as e:
            print(f"CatBoost training failed: {e}")
        
        # Ensemble validation
        print("\nğŸ“Š Ensemble Validation...")
        try:
            val_predictions = ensemble.predict(X_val)
            val_auc = roc_auc_score(y_val, val_predictions)
            print(f"Ensemble Validation AUC: {val_auc:.6f}")
            
            # Classification report
            val_pred_binary = (val_predictions > 0.5).astype(int)
            print("\nClassification Report:")
            print(classification_report(y_val, val_pred_binary))
            
        except Exception as e:
            print(f"Ensemble validation failed: {e}")
        
        # Generate predictions for test set
        print("\nğŸ”® Generating test predictions...")
        try:
            test_predictions = ensemble.predict(X_test)
            
            # Create submission
            submission = sample_sub.copy()
            submission['y'] = test_predictions
            
            # Save submission
            submission.to_csv('advanced_ensemble_submission.csv', index=False)
            print("Submission saved as 'advanced_ensemble_submission.csv'")
            
            # Display prediction statistics
            print(f"\nTest Prediction Statistics:")
            print(f"Min: {test_predictions.min():.6f}")
            print(f"Max: {test_predictions.max():.6f}")
            print(f"Mean: {test_predictions.mean():.6f}")
            print(f"Std: {test_predictions.std():.6f}")
            
            # Show distribution
            print(f"\nPrediction Distribution:")
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                count = np.sum(test_predictions > threshold)
                pct = count / len(test_predictions) * 100
                print(f"Predictions > {threshold}: {count} ({pct:.2f}%)")
                
        except Exception as e:
            print(f"Test prediction failed: {e}")
    
    else:
        print("No target variable found - this appears to be test-only data")
    
    print("\nâœ… Pipeline completed!")

# =============================================================================
# CROSS-VALIDATION EVALUATION
# =============================================================================

def cross_validate_ensemble(X, y, device, cv_folds=5):
    """Perform cross-validation on the ensemble"""
    print(f"\nğŸ”„ Performing {cv_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features for this fold
        scaler = RobustScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        # Calculate class weights
        pos_count = np.sum(y_train_fold == 1)
        neg_count = np.sum(y_train_fold == 0)
        class_weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Initialize fold ensemble
        fold_ensemble = SimplifiedEnsemble(device)
        
        # Train models for this fold
        try:
            # Quick neural network (reduced complexity for CV)
            nn_model = train_neural_network(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                device, class_weight_ratio
            )
            fold_ensemble.add_model(nn_model, 'neural_network', weight=0.4)
        except Exception as e:
            print(f"NN failed in fold {fold + 1}: {e}")
        
        try:
            # LightGBM
            lgb_model = train_lightgbm_gpu(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            fold_ensemble.add_model(lgb_model, 'lightgbm', weight=0.3)
        except Exception as e:
            print(f"LightGBM failed in fold {fold + 1}: {e}")
        
        try:
            # XGBoost
            xgb_model = train_xgboost_gpu(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            fold_ensemble.add_model(xgb_model, 'xgboost', weight=0.3)
        except Exception as e:
            print(f"XGBoost failed in fold {fold + 1}: {e}")
        
        # Get fold predictions
        try:
            fold_predictions = fold_ensemble.predict(X_val_fold)
            fold_auc = roc_auc_score(y_val_fold, fold_predictions)
            cv_scores.append(fold_auc)
            print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
        except Exception as e:
            print(f"Fold {fold + 1} prediction failed: {e}")
            cv_scores.append(0.5)  # Worst case score
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\nğŸ“ˆ Cross-validation Results:")
    print(f"Mean AUC: {mean_cv_score:.6f} Â± {std_cv_score:.6f}")
    print(f"Individual fold scores: {[f'{score:.6f}' for score in cv_scores]}")
    
    return mean_cv_score, std_cv_score

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def plot_feature_importance(models_dict, feature_names, top_k=20):
    """Plot feature importance from different models"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(models_dict.items()):
            if idx >= 4:
                break
                
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'get_feature_importance'):
                    importances = model.get_feature_importance()
                else:
                    continue
                
                # Get top features
                indices = np.argsort(importances)[::-1][:top_k]
                top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                              for i in indices]
                top_importances = importances[indices]
                
                # Plot
                axes[idx].barh(range(len(top_importances)), top_importances[::-1])
                axes[idx].set_yticks(range(len(top_importances)))
                axes[idx].set_yticklabels(top_features[::-1])
                axes[idx].set_title(f'{name} Feature Importance')
                axes[idx].set_xlabel('Importance')
                
            except Exception as e:
                print(f"Could not plot importance for {name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error plotting\n{name}', 
                             transform=axes[idx].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping feature importance plots")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

def save_model_artifacts(ensemble, feature_engineer, scaler):
    """Save trained models and preprocessing artifacts"""
    try:
        # Create artifacts directory
        os.makedirs('model_artifacts', exist_ok=True)
        
        # Save feature engineer
        joblib.dump(feature_engineer, 'model_artifacts/feature_engineer.pkl')
        
        # Save scaler
        joblib.dump(scaler, 'model_artifacts/scaler.pkl')
        
        # Save individual models
        for name, model in ensemble.base_models.items():
            if name == 'neural_network':
                torch.save(model.state_dict(), f'model_artifacts/{name}_state_dict.pth')
            else:
                joblib.dump(model, f'model_artifacts/{name}_model.pkl')
        
        # Save model weights
        joblib.dump(ensemble.model_weights, 'model_artifacts/ensemble_weights.pkl')
        
        print("Model artifacts saved successfully!")
        
    except Exception as e:
        print(f"Error saving model artifacts: {e}")

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Enable multiprocessing for sklearn
    os.environ['LOKY_MAX_CPU_COUNT'] = str(min(mp.cpu_count(), 8))
    
    # Run main pipeline
    main()

