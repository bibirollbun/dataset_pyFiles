# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline

# Standard Models
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                  LassoLars, OrthogonalMatchingPursuit, HuberRegressor,
                                  RANSACRegressor, TheilSenRegressor, ARDRegression,
                                  PassiveAggressiveRegressor, SGDRegressor)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, VotingRegressor,
                             GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
                             HistGradientBoostingRegressor, StackingRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, Matern, DotProduct
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge

# Advanced Models
try:
    from xgboost import XGBRegressor
except:
    print("XGBoost not available")
    
try:
    from lightgbm import LGBMRegressor
except:
    print("LightGBM not available")
    
try:
    from catboost import CatBoostRegressor
except:
    print("CatBoost not available")

# FLAML AutoML
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except:
    print("FLAML not available - install with: pip install flaml")
    FLAML_AVAILABLE = False

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except:
    print("TabNet not available - install with: pip install pytorch-tabnet")
    TABNET_AVAILABLE = False

# NGBoost
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except:
    print("NGBoost not available - install with: pip install ngboost")
    NGBOOST_AVAILABLE = False

# GANDALF - would need custom implementation or specific package
# For now, we'll use a placeholder
GANDALF_AVAILABLE = False

# Feature engineering
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import skew, kurtosis
import scipy.special

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except:
    print("PyTorch not available")
    TORCH_AVAILABLE = False

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)

# ===== CUSTOM NEURAL NETWORK IMPLEMENTATIONS =====
if TORCH_AVAILABLE:
    class CustomMLP(nn.Module):
        """Advanced MLP with residual connections and dropout"""
        def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
            super(CustomMLP, self).__init__()
            self.layers = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                self.dropouts.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            self.output_layer = nn.Linear(prev_dim, 1)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
                x = layer(x)
                x = bn(x)
                x = self.activation(x)
                x = dropout(x)
            return self.output_layer(x)
    
    class MLPWrapper:
        """Sklearn-compatible wrapper for PyTorch MLP"""
        def __init__(self, hidden_dims=[128, 64, 32], dropout=0.3, epochs=100, lr=0.001):
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.epochs = epochs
            self.lr = lr
            self.model = None
            self.scaler = StandardScaler()
            
        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = CustomMLP(X_scaled.shape[1], self.hidden_dims, self.dropout)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            self.model.train()
            for epoch in range(self.epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            return self
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).numpy().flatten()
            
            return predictions

# ===== 1. LOAD DATA =====
print("="*80)
print("CALIFORNIA HOMELESSNESS PREDICTION - ADVANCED MODEL DIVERSITY")
print("="*80)

print("\n1. Loading data...")
train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Separate features and target
train_ids = train['ID']
test_ids = test['ID']
y_train = train['HOMELESS_RATE']
X_train = train.drop(['ID', 'HOMELESS_RATE'], axis=1)
X_test = test.drop(['ID'], axis=1)

# ===== 2. ADVANCED FEATURE ENGINEERING =====
print("\n2. Advanced Feature Engineering")
print("-"*50)

def create_comprehensive_features(df):
    """Create comprehensive features including non-linear transformations"""
    df_new = df.copy()
    
    # === Basic Ratios and Indices ===
    df_new['age_vulnerability'] = (df['AGE_U18_PCT'] + df['AGE_65_69_PCT'] + 
                                   df['AGE_70_79_PCT'] + df['AGE_80_PLUS_PCT'])
    df_new['working_age_ratio'] = (df['AGE_25_34_PCT'] + df['AGE_35_44_PCT'] + 
                                   df['AGE_45_54_PCT'])
    df_new['youth_adult_ratio'] = df['AGE_18_24_PCT'] / (df['AGE_25_PLUS_PCT'] + 1e-5)
    
    # === Family and Household Features ===
    df_new['family_stability'] = df['FAMILY_HH_TOTAL'] - df['NONFAMILY_SINGLE_MALE_PCT'] - df['NONFAMILY_SINGLE_FEMALE_PCT']
    df_new['single_vulnerability'] = df['NONFAMILY_SINGLE_MALE_PCT'] + df['NONFAMILY_SINGLE_FEMALE_PCT']
    df_new['single_gender_imbalance'] = np.abs(df['NONFAMILY_SINGLE_MALE_PCT'] - df['NONFAMILY_SINGLE_FEMALE_PCT'])
    
    # === Race and Diversity Features ===
    df_new['minority_pct'] = 100 - df['RACE_WHITE_NH_PCT']
    race_cols = ['RACE_WHITE_NH_PCT', 'RACE_BLACK_NH_PCT', 'RACE_NATIVE_NH_PCT', 
                 'RACE_ASIAN_NH_PCT', 'RACE_PACIFIC_NH_PCT', 'RACE_HISPANIC_ANY_PCT']
    
    # Multiple diversity indices
    df_new['diversity_herfindahl'] = 1 - (df[race_cols] / 100).pow(2).sum(axis=1)
    race_props = df[race_cols] / 100
    df_new['diversity_shannon'] = -(race_props * np.log(race_props + 1e-10)).sum(axis=1)
    df_new['diversity_simpson'] = 1 - ((race_props * (race_props - 1/len(race_cols))).sum(axis=1))
    
    # === Advanced Transformations ===
    # Polynomial features
    df_new['family_hh_squared'] = df['FAMILY_HH_TOTAL'] ** 2
    df_new['family_hh_cubed'] = df['FAMILY_HH_TOTAL'] ** 3
    df_new['single_female_squared'] = df['NONFAMILY_SINGLE_FEMALE_PCT'] ** 2
    df_new['household_pct_squared'] = df['TOTAL_HOUSEHOLDS_PCT'] ** 2
    
    # Exponential and logarithmic
    for col in ['TOTAL_HOUSEHOLDS_PCT', 'VETERAN_POP_PCT', 'DISABILITY_POP_PCT']:
        df_new[f'{col}_log'] = np.log1p(df[col])
        df_new[f'{col}_exp'] = np.expm1(df[col] / 100)  # Scale before exp
    
    # Trigonometric features (cyclical encoding)
    df_new['age_cycle_sin'] = np.sin(2 * np.pi * df['AGE_25_34_PCT'] / 100)
    df_new['age_cycle_cos'] = np.cos(2 * np.pi * df['AGE_25_34_PCT'] / 100)
    
    # Interaction features
    df_new['veteran_disability_interact'] = df['VETERAN_POP_PCT'] * df['DISABILITY_POP_PCT']
    df_new['elderly_disability_interact'] = (df['AGE_65_69_PCT'] + df['AGE_70_79_PCT'] + 
                                            df['AGE_80_PLUS_PCT']) * df['DISABILITY_POP_PCT']
    df_new['family_household_interact'] = df['FAMILY_HH_TOTAL'] * df['TOTAL_HOUSEHOLDS_PCT']
    df_new['age25_34_family_interact'] = df['AGE_25_34_PCT'] * df['FAMILY_HH_TOTAL']
    
    # Ratios
    df_new['disability_ratio'] = df['DISABILITY_POP_PCT'] / (df['NODISABILITY_POP_PCT'] + 1e-5)
    df_new['veteran_ratio'] = df['VETERAN_POP_PCT'] / (df['NONVETERAN_POP_PCT'] + 1e-5)
    df_new['single_male_female_ratio'] = df['NONFAMILY_SINGLE_MALE_PCT'] / (df['NONFAMILY_SINGLE_FEMALE_PCT'] + 1e-5)
    
    # Statistical aggregations across age groups
    age_groups = ['AGE_U18_PCT', 'AGE_18_24_PCT', 'AGE_25_34_PCT', 'AGE_35_44_PCT', 
                  'AGE_45_54_PCT', 'AGE_55_59_PCT', 'AGE_60_61_PCT', 'AGE_62_64_PCT',
                  'AGE_65_69_PCT', 'AGE_70_79_PCT', 'AGE_80_PLUS_PCT']
    
    df_new['age_std'] = df[age_groups].std(axis=1)
    df_new['age_entropy'] = -(df[age_groups] / 100 * np.log(df[age_groups] / 100 + 1e-10)).sum(axis=1)
    
    # Binning features
    df_new['high_family_households'] = (df['FAMILY_HH_TOTAL'] > df['FAMILY_HH_TOTAL'].median()).astype(int)
    df_new['high_single_female'] = (df['NONFAMILY_SINGLE_FEMALE_PCT'] > df['NONFAMILY_SINGLE_FEMALE_PCT'].median()).astype(int)
    
    return df_new

# Apply feature engineering
X_train_eng = create_comprehensive_features(X_train)
X_test_eng = create_comprehensive_features(X_test)

print(f"Features after engineering: {X_train_eng.shape[1]}")

# Remove highly correlated features
correlation_matrix = X_train_eng.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
X_train_eng = X_train_eng.drop(columns=to_drop)
X_test_eng = X_test_eng.drop(columns=to_drop)

print(f"Features after correlation removal: {X_train_eng.shape[1]}")

# ===== 3. MODEL DEFINITIONS =====
print("\n3. Initializing Diverse Model Suite")
print("-"*50)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Initialize models dictionary
models = {}

# === Linear Models ===
models.update({
    'Ridge': Ridge(alpha=5.0),
    'Lasso': Lasso(alpha=0.01, max_iter=2000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=2000),
    'BayesianRidge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6),
    'ARDRegression': ARDRegression(alpha_1=1e-6, alpha_2=1e-6),
    'LassoLars': LassoLars(alpha=0.01),
    'HuberRegressor': HuberRegressor(epsilon=1.35, max_iter=200),
    'PassiveAggressive': PassiveAggressiveRegressor(C=1.0, max_iter=1000, random_state=42),
    'SGDRegressor': SGDRegressor(loss='huber', penalty='elasticnet', alpha=0.01, 
                                 l1_ratio=0.5, max_iter=1000, random_state=42),
})

# === Robust Regressors ===
models.update({
    'RANSACRegressor': RANSACRegressor(random_state=42),
    'TheilSenRegressor': TheilSenRegressor(random_state=42, max_iter=300),
})

# === Kernel Methods ===
models.update({
    'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.001),
    'SVR_Linear': SVR(kernel='linear', C=1.0, epsilon=0.001),
    'SVR_Poly': SVR(kernel='poly', C=10, degree=2, epsilon=0.001),
    'NuSVR': NuSVR(kernel='rbf', C=10, gamma='scale'),
    'KernelRidge': KernelRidge(alpha=1.0, kernel='rbf', gamma='scale'),
})

# === Tree-based Models ===
models.update({
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=4, 
                                         min_samples_split=5, min_samples_leaf=3, 
                                         random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=4,
                                     min_samples_split=5, min_samples_leaf=3,
                                     random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                  learning_rate=0.05, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, max_depth=3,
                                                          learning_rate=0.05, random_state=42),
    'AdaBoost': AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), 
                                 n_estimators=100, learning_rate=0.5, random_state=42),
    'Bagging': BaggingRegressor(DecisionTreeRegressor(max_depth=3),
                               n_estimators=50, random_state=42),
})

# === Advanced Boosting (if available) ===
if 'XGBRegressor' in globals():
    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42)

if 'LGBMRegressor' in globals():
    models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                       num_leaves=20, random_state=42, verbose=-1)

if 'CatBoostRegressor' in globals():
    models['CatBoost'] = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05,
                                          l2_leaf_reg=10, random_state=42, verbose=False)

# === Gaussian Process ===
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
models['GaussianProcess'] = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, 
                                                     normalize_y=True, random_state=42)

# === Neural Networks ===
models.update({
    'MLP_Basic': MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu',
                             solver='lbfgs', alpha=0.1, random_state=42, max_iter=1000),
    'MLP_Deep': MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                            solver='adam', alpha=0.01, learning_rate_init=0.001,
                            random_state=42, max_iter=1000),
    'MLP_Wide': MLPRegressor(hidden_layer_sizes=(200, 100), activation='tanh',
                            solver='adam', alpha=0.001, learning_rate_init=0.001,
                            random_state=42, max_iter=1000),
})

# === Advanced Neural Networks (if PyTorch available) ===
if TORCH_AVAILABLE:
    models['CustomMLP'] = MLPWrapper(hidden_dims=[128, 64, 32], dropout=0.3, epochs=100)
    models['DeepMLP'] = MLPWrapper(hidden_dims=[256, 128, 64, 32], dropout=0.4, epochs=150)

# === TabNet (if available) ===
if TABNET_AVAILABLE:
    models['TabNet'] = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                                      lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
                                      optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                      mask_type='entmax', seed=42, verbose=0)

# === NGBoost (if available) ===
if NGBOOST_AVAILABLE:
    models['NGBoost'] = NGBRegressor(n_estimators=100, learning_rate=0.01,
                                    minibatch_frac=1.0, random_state=42)

# === Other Models ===
models.update({
    'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance', p=2),
    'KNN_Uniform': KNeighborsRegressor(n_neighbors=5, weights='uniform', p=1),
})

print(f"Total models initialized: {len(models)}")

# ===== 4. FLAML AUTOML =====
if FLAML_AVAILABLE:
    print("\n4. Running FLAML AutoML")
    print("-"*50)
    
    automl = AutoML()
    automl_settings = {
        "time_budget": 60,  # 60 seconds
        "metric": 'rmse',
        "task": 'regression',
        "n_jobs": -1,
        "seed": 42,
        "model_history": True,
        "ensemble": True,
        "max_iter": 100,
        "early_stop": True,
        "eval_method": "cv",
        "n_splits": 5,
    }
    
    # Run FLAML
    automl.fit(X_train_scaled, y_train, **automl_settings)
    
    print(f"FLAML Best Model: {automl.best_estimator}")
    print(f"FLAML Best RMSE: {automl.best_loss:.6f}")
    
    # Add FLAML to models
    models['FLAML_AutoML'] = automl

# ===== 5. CROSS-VALIDATION AND EVALUATION =====
print("\n5. Model Training and Evaluation")
print("-"*50)

# Evaluation function
def evaluate_model(y_true, y_pred):
    """Calculate multiple evaluation metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MedAE': median_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    return metrics

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
trained_models = {}
predictions_dict = {}

print("\nTraining models with 5-fold cross-validation...")
print("-"*100)
print(f"{'Model':<25} | {'RMSE':>10} | {'MAE':>10} | {'R²':>10} | {'Time (s)':>10}")
print("-"*100)

import time

for name, model in models.items():
    try:
        start_time = time.time()
        
        # Cross-validation predictions
        cv_predictions = np.zeros(len(y_train))
        cv_metrics = {'RMSE': [], 'MAE': [], 'R2': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Special handling for TabNet
            if name == 'TabNet' and TABNET_AVAILABLE:
                model_clone = model.fit(X_fold_train, y_fold_train.values.reshape(-1, 1),
                                      eval_set=[(X_fold_val, y_fold_val.values.reshape(-1, 1))],
                                      max_epochs=100, patience=20, batch_size=16, 
                                      virtual_batch_size=8, eval_metric=['rmse'])
                fold_pred = model_clone.predict(X_fold_val).flatten()
            else:
                # Standard sklearn interface
                model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                model_clone.fit(X_fold_train, y_fold_train)
                fold_pred = model_clone.predict(X_fold_val)
            
            cv_predictions[val_idx] = fold_pred
            
            # Calculate metrics
            fold_metrics = evaluate_model(y_fold_val, fold_pred)
            for metric in ['RMSE', 'MAE', 'R2']:
                cv_metrics[metric].append(fold_metrics[metric])
        
        # Store CV results
        cv_results[name] = {
            'RMSE_mean': np.mean(cv_metrics['RMSE']),
            'RMSE_std': np.std(cv_metrics['RMSE']),
            'MAE_mean': np.mean(cv_metrics['MAE']),
            'MAE_std': np.std(cv_metrics['MAE']),
            'R2_mean': np.mean(cv_metrics['R2']),
            'R2_std': np.std(cv_metrics['R2']),
            'cv_predictions': cv_predictions
        }
        
        # Train on full dataset
        if name == 'TabNet' and TABNET_AVAILABLE:
            model.fit(X_train_scaled, y_train.values.reshape(-1, 1),
                     max_epochs=100, patience=20, batch_size=16, 
                     virtual_batch_size=8)
        else:
            model.fit(X_train_scaled, y_train)
        
        trained_models[name] = model
        
        # Make test predictions
        if name == 'TabNet' and TABNET_AVAILABLE:
            test_pred = model.predict(X_test_scaled).flatten()
        else:
            test_pred = model.predict(X_test_scaled)
        
        predictions_dict[name] = test_pred
        
        elapsed_time = time.time() - start_time
        
        print(f"{name:<25} | {cv_results[name]['RMSE_mean']:>10.6f} | "
              f"{cv_results[name]['MAE_mean']:>10.6f} | "
              f"{cv_results[name]['R2_mean']:>10.4f} | {elapsed_time:>10.2f}")
        
    except Exception as e:
        print(f"{name:<25} | Failed: {str(e)[:50]}...")

# ===== 6. STACKING ENSEMBLE =====
print("\n6. Creating Stacking Ensemble")
print("-"*50)

# Select top models for stacking
sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['RMSE_mean'])
top_n = min(10, len(sorted_models))
top_model_names = [name for name, _ in sorted_models[:top_n]]

print(f"\nTop {top_n} models for stacking:")
for i, (name, results) in enumerate(sorted_models[:top_n], 1):
    print(f"  {i}. {name} (RMSE: {results['RMSE_mean']:.6f})")

# Create base estimators list
base_estimators = [(name, trained_models[name]) for name in top_model_names if name in trained_models]

# Create stacking ensemble
stacking = StackingRegressor(
    estimators=base_estimators[:5],  # Use top 5 for stacking
    final_estimator=BayesianRidge(),
    cv=5,
    n_jobs=-1
)

# Train stacking
print("\nTraining stacking ensemble...")
stacking.fit(X_train_scaled, y_train)
stacking_pred = stacking.predict(X_test_scaled)

# Add to predictions
predictions_dict['StackingEnsemble'] = stacking_pred

# ===== 7. WEIGHTED ENSEMBLE =====
print("\n7. Creating Weighted Ensemble")
print("-"*50)

# Create weighted ensemble from top models
ensemble_predictions = []
ensemble_weights = []

for model_name in top_model_names[:15]:  # Use top 15 models
    if model_name in predictions_dict:
        pred = predictions_dict[model_name]
        pred = np.maximum(pred, 0)  # Ensure non-negative
        ensemble_predictions.append(pred)
        
        # Weight inversely proportional to RMSE
        weight = 1 / (cv_results[model_name]['RMSE_mean'] + 1e-6)
        ensemble_weights.append(weight)

# Normalize weights
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()

print(f"\nEnsemble includes {len(ensemble_predictions)} models")
print("\nTop 5 weights:")
for i, (model, weight) in enumerate(zip(top_model_names[:5], ensemble_weights[:5])):
    print(f"  {model}: {weight:.4f}")

# Final predictions
final_predictions = np.average(ensemble_predictions, weights=ensemble_weights, axis=0)

# Post-processing
upper_bound = np.percentile(y_train, 99) * 1.2
final_predictions = np.clip(final_predictions, 0, upper_bound)

# ===== 8. RESIDUAL ANALYSIS FOR TOP MODELS =====
print("\n8. Residual Analysis")
print("-"*50)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.ravel()

for idx, (model_name, results) in enumerate(sorted_models[:5]):
    if model_name in cv_results:
        cv_pred = cv_results[model_name]['cv_predictions']
        residuals = y_train - cv_pred
        
        # Residual plot
        ax = axes[idx]
        ax.scatter(cv_pred, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name} - Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[idx + 5]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{model_name} - Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Histogram
        ax = axes[idx + 10]
        ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} - Residual Distribution')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 9. MODEL COMPARISON VISUALIZATION =====
print("\n9. Model Performance Visualization")
print("-"*50)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'RMSE': [cv_results[m]['RMSE_mean'] for m in cv_results],
    'MAE': [cv_results[m]['MAE_mean'] for m in cv_results],
    'R2': [cv_results[m]['R2_mean'] for m in cv_results]
}).sort_values('RMSE')

# Print top 20 models
print("\nTop 20 Models by RMSE:")
print(comparison_df.head(20).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# RMSE comparison
ax = axes[0, 0]
top_20 = comparison_df.head(20)
colors = ['darkgreen' if i < 5 else 'skyblue' for i in range(len(top_20))]
bars = ax.bar(range(len(top_20)), top_20['RMSE'], color=colors)
ax.set_xticks(range(len(top_20)))
ax.set_xticklabels(top_20['Model'], rotation=45, ha='right')
ax.set_ylabel('RMSE')
ax.set_title('Top 20 Models - RMSE (Lower is Better)')
ax.grid(axis='y', alpha=0.3)

# R² comparison  
ax = axes[0, 1]
scatter = ax.scatter(comparison_df['RMSE'], comparison_df['R2'], 
                    s=100, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
for i, model in enumerate(comparison_df['Model'][:10]):
    ax.annotate(model, (comparison_df['RMSE'].iloc[i], comparison_df['R2'].iloc[i]),
                fontsize=8, ha='right')
ax.set_xlabel('RMSE')
ax.set_ylabel('R² Score')
ax.set_title('RMSE vs R² Score')
ax.grid(True, alpha=0.3)

# Model category performance
ax = axes[1, 0]
categories = {
    'Linear': ['Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 'ARDRegression'],
    'Robust': ['HuberRegressor', 'RANSACRegressor', 'TheilSenRegressor'],
    'SVM': ['SVR_RBF', 'SVR_Linear', 'SVR_Poly', 'NuSVR'],
    'Tree': ['RandomForest', 'ExtraTrees', 'GradientBoosting'],
    'Boosting': ['XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost'],
    'Neural': ['MLP_Basic', 'MLP_Deep', 'MLP_Wide', 'CustomMLP', 'TabNet'],
    'Other': ['KNN', 'GaussianProcess', 'FLAML_AutoML']
}

category_performance = {}
for cat, models_list in categories.items():
    cat_rmse = [cv_results[m]['RMSE_mean'] for m in models_list if m in cv_results]
    if cat_rmse:
        category_performance[cat] = np.mean(cat_rmse)

if category_performance:
    cats = list(category_performance.keys())
    perfs = list(category_performance.values())
    ax.bar(cats, perfs, color='coral')
    ax.set_ylabel('Average RMSE')
    ax.set_title('Performance by Model Category')
    ax.grid(axis='y', alpha=0.3)

# Prediction variance
ax = axes[1, 1]
top_5_preds = [predictions_dict[m] for m in top_model_names[:5] if m in predictions_dict]
if top_5_preds:
    pred_std = np.std(top_5_preds, axis=0)
    ax.hist(pred_std, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Prediction Standard Deviation')
    ax.set_ylabel('Frequency')
    ax.set_title('Ensemble Prediction Uncertainty')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 10. FINAL SUBMISSION =====
print("\n10. Creating Final Submission")
print("-"*50)

submission = pd.DataFrame({
    'ID': test_ids,
    'HOMELESS_RATE': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nSubmission Statistics:")
print(f"Mean: {final_predictions.mean():.6f}")
print(f"Std: {final_predictions.std():.6f}")
print(f"Min: {final_predictions.min():.6f}")
print(f"Max: {final_predictions.max():.6f}")

# ===== 11. COMPREHENSIVE REPORT =====
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS REPORT")
print("="*80)

print(f"\nDataset Information:")
print(f"  Training samples: {len(train)}")
print(f"  Test samples: {len(test)}")
print(f"  Original features: {len(X_train.columns)}")
print(f"  Engineered features: {len(X_train_eng.columns)}")

print(f"\nModel Diversity:")
print(f"  Total models evaluated: {len(models)}")
print(f"  Linear models: {len([m for m in models if any(x in m for x in ['Ridge', 'Lasso', 'Linear'])])}")
print(f"  Tree-based models: {len([m for m in models if any(x in m for x in ['Forest', 'Tree', 'Boost'])])}")
print(f"  Neural networks: {len([m for m in models if 'MLP' in m or 'TabNet' in m])}")
print(f"  SVM variants: {len([m for m in models if 'SVR' in m or 'SVM' in m])}")

print(f"\nEnsemble Details:")
print(f"  Models in weighted ensemble: {len(ensemble_predictions)}")
print(f"  Best single model: {comparison_df.iloc[0]['Model']} (RMSE: {comparison_df.iloc[0]['RMSE']:.6f})")
print(f"  Stacking ensemble included: {'Yes' if 'StackingEnsemble' in predictions_dict else 'No'}")
if FLAML_AVAILABLE:
    print(f"  FLAML AutoML included: Yes")

print(f"\nTop 5 Performing Models:")
for i, row in comparison_df.head(5).iterrows():
    print(f"  {i+1}. {row['Model']:<20} | RMSE: {row['RMSE']:.6f} | R²: {row['R2']:.4f}")

print("\n✓ Analysis complete!")
print("✓ Submission saved to: submission.csv")
print("✓ Visualizations saved")

# Save results
import pickle
results_summary = {
    'cv_results': cv_results,
    'comparison_df': comparison_df,
    'ensemble_weights': dict(zip(top_model_names[:len(ensemble_weights)], ensemble_weights)),
    'predictions': final_predictions,
    'model_categories': categories if 'categories' in locals() else None
}

with open('advanced_analysis_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("✓ Results saved to: advanced_analysis_results.pkl")

