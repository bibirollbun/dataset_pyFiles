# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files_names = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files_names.append(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# COMPLETE DEPRESSION PREDICTION PIPELINE
# Advanced ML solution with multiple models and ensemble

# =====================================================
# INSTALL PACKAGES
# =====================================================
!pip install -q catboost
!pip install -q lightgbm
!pip install -q xgboost
!pip install -q pytorch-tabnet
!pip install -q optuna

# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from scipy.stats import rankdata
import gc

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pytorch_tabnet.tab_model import TabNetClassifier

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================
# 1. DATA LOADING
# =====================================================
print("\n" + "="*60)
print("1. DATA LOADING")
print("="*60)

# Load data
train_df = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
test_df = pd.read_csv('/kaggle/input/moc-competition-mental-health/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Save IDs and target
test_ids = test_df['id'].values
y_train = train_df['Depression'].values

# Remove IDs and names
train_df = train_df.drop(['id', 'Name'], axis=1)
test_df = test_df.drop(['id', 'Name'], axis=1)

# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================
print("\n" + "="*60)
print("2. FEATURE ENGINEERING")
print("="*60)

class FeatureEngineer:
    """Advanced feature engineering"""
    
    def __init__(self):
        self.feature_stats = {}
        
    def create_features(self, df):
        """Create advanced features"""
        # Combine pressure and satisfaction
        df['Study_Work_Pressure'] = df[['Academic Pressure', 'Work Pressure']].fillna(method='ffill').mean(axis=1)
        df['Study_Job_Satisfaction'] = df[['Study Satisfaction', 'Job Satisfaction']].fillna(method='ffill').mean(axis=1)
        
        # Advanced ratios and interactions
        df['Pressure_Satisfaction_Ratio'] = (df['Study_Work_Pressure'] + 1) / (df['Study_Job_Satisfaction'] + 1)
        df['Work_Life_Balance'] = df['Study_Job_Satisfaction'] - df['Study_Work_Pressure']
        df['Stress_Index'] = df['Financial Stress'].fillna(df['Financial Stress'].median()) + df['Study_Work_Pressure']
        
        # Polynomial features
        df['Age_squared'] = df['Age'] ** 2
        df['CGPA_squared'] = df['CGPA'].fillna(df['CGPA'].median()) ** 2
        df['Stress_Work_Interaction'] = df['Financial Stress'].fillna(df['Financial Stress'].median()) * df['Work/Study Hours'].fillna(df['Work/Study Hours'].median())
        
        # Logarithmic transformations
        df['Log_Age'] = np.log1p(df['Age'])
        df['Log_Work_Hours'] = np.log1p(df['Work/Study Hours'].fillna(df['Work/Study Hours'].median()))
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 22, 25, 28, 32, 100], 
                                labels=['teen', 'early20s', 'mid20s', 'late20s', 'early30s', 'older'])
        
        # CGPA categories
        df['CGPA_Category'] = pd.cut(df['CGPA'].fillna(df['CGPA'].median()), 
                                     bins=[0, 5.5, 6.5, 7.5, 8.5, 10],
                                     labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Risk factors
        df['Low_CGPA'] = (df['CGPA'].fillna(df['CGPA'].median()) < 6.5).astype(int)
        df['High_CGPA'] = (df['CGPA'].fillna(df['CGPA'].median()) > 8.0).astype(int)
        df['Low_Sleep'] = (df['Sleep Duration'] == 'Less than 5 hours').astype(int)
        df['High_Work_Hours'] = (df['Work/Study Hours'].fillna(df['Work/Study Hours'].median()) > 10).astype(int)
        df['High_Financial_Stress'] = (df['Financial Stress'].fillna(df['Financial Stress'].median()) > 3).astype(int)
        df['Low_Satisfaction'] = (df['Study_Job_Satisfaction'] < 2).astype(int)
        
        # Protective factors
        df['Good_Sleep'] = (df['Sleep Duration'].isin(['7-8 hours', 'More than 8 hours'])).astype(int)
        df['Balanced_Life'] = ((df['Work_Life_Balance'] > -1) & (df['Work_Life_Balance'] < 1)).astype(int)
        
        # Risk score
        df['Risk_Score'] = (
            df['Low_CGPA'] + df['Low_Sleep'] + df['High_Work_Hours'] + 
            df['High_Financial_Stress'] + df['Low_Satisfaction'] -
            df['Good_Sleep'] - df['Balanced_Life']
        )
        
        return df
    
    def fit_transform(self, train_df, test_df):
        """Apply feature engineering to train and test"""
        train_df = self.create_features(train_df)
        test_df = self.create_features(test_df)
        
        # Drop original columns that were combined
        drop_cols = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']
        train_df = train_df.drop(drop_cols, axis=1)
        test_df = test_df.drop(drop_cols, axis=1)
        
        return train_df, test_df

# Apply feature engineering
fe = FeatureEngineer()
train_df, test_df = fe.fit_transform(train_df, test_df)

# =====================================================
# 3. DATA PREPROCESSING
# =====================================================
print("\n" + "="*60)
print("3. DATA PREPROCESSING")
print("="*60)

# Handle missing values
for df in [train_df, test_df]:
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != 'Depression':  # Skip target
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('missing')

# Encoding categorical variables
# Binary encoding
binary_map = {
    'Yes': 1, 'No': 0, 'missing': 0,
    'Male': 0, 'Female': 1,
    'Student': 0, 'Working Professional': 1
}

binary_cols = ['Gender', 'Working Professional or Student', 
               'Have you ever had suicidal thoughts ?', 
               'Family History of Mental Illness']

for col in binary_cols:
    train_df[col] = train_df[col].map(binary_map).fillna(0)
    test_df[col] = test_df[col].map(binary_map).fillna(0)

# Ordinal encoding
sleep_map = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4,
    'missing': 2.5
}
train_df['Sleep Duration'] = train_df['Sleep Duration'].map(sleep_map).fillna(2.5)
test_df['Sleep Duration'] = test_df['Sleep Duration'].map(sleep_map).fillna(2.5)

diet_map = {
    'Unhealthy': 0,
    'Moderate': 1,
    'Healthy': 2,
    'missing': 1
}
train_df['Dietary Habits'] = train_df['Dietary Habits'].map(diet_map).fillna(1)
test_df['Dietary Habits'] = test_df['Dietary Habits'].map(diet_map).fillna(1)

# Label encoding for remaining categoricals
label_cols = ['City', 'Profession', 'Degree', 'Age_Group', 'CGPA_Category']
for col in label_cols:
    le = LabelEncoder()
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Prepare final datasets
X_train = train_df.drop('Depression', axis=1)
X_test = test_df

print(f"Final train shape: {X_train.shape}")
print(f"Final test shape: {X_test.shape}")

# Store original DataFrames for CatBoost
X_train_df = X_train.copy()
X_test_df = X_test.copy()

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 4. NEURAL NETWORK ARCHITECTURES
# =====================================================
print("\n" + "="*60)
print("4. DEFINING NEURAL NETWORK ARCHITECTURES")
print("="*60)

# Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# 4.1 Fixed Enhanced GANDALF
class EnhancedGANDALF(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Gated layers with residual connections
        self.gated_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            # Gate
            self.gated_layers.append(nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.Sigmoid()
            ))
            
            # Transform
            self.residual_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            
            prev_dim = hidden_dim
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = F.gelu(x)
        
        for gate, transform in zip(self.gated_layers, self.residual_layers):
            # Apply gate
            gated = x * gate(x)
            
            # Get the output dimension from the first Linear layer in transform
            out_dim = transform[0].out_features
            
            # Transform with residual connection
            residual = F.adaptive_avg_pool1d(x.unsqueeze(1), out_dim).squeeze(1)
            x = transform(gated) + residual
        
        return self.output(x).squeeze()

# 4.2 Simple Deep Neural Network
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

# =====================================================
# 5. TRAINING FUNCTION
# =====================================================
print("\n" + "="*60)
print("5. SETTING UP TRAINING")
print("="*60)

def train_neural_network(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train a neural network model"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.BCELoss()
    
    best_val_acc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                predicted = (outputs > 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                val_probs.extend(outputs.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_true, val_probs)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

# =====================================================
# 6. HYPERPARAMETER OPTIMIZATION
# =====================================================
print("\n" + "="*60)
print("6. HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Prepare data splits
X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

# Also prepare DataFrame versions for tree models
X_opt_train_df, X_opt_val_df = X_train_df.iloc[:len(X_opt_train)], X_train_df.iloc[len(X_opt_train):]

def optimize_lgb(trial):
    """Optimize LightGBM hyperparameters"""
    params = {
        'n_estimators': 1000,
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': SEED,
        'verbosity': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_opt_train, y_opt_train, 
              eval_set=[(X_opt_val, y_opt_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    pred = model.predict_proba(X_opt_val)[:, 1]
    return roc_auc_score(y_opt_val, pred)

# Run optimization
print("Optimizing LightGBM...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
study.optimize(optimize_lgb, n_trials=10)
print(f"Best LightGBM AUC: {study.best_value:.4f}")

# =====================================================
# 7. TRAIN ALL MODELS
# =====================================================
print("\n" + "="*60)
print("7. TRAINING ALL MODELS")
print("="*60)

# Prepare data loaders for neural networks
train_dataset = TabularDataset(X_opt_train, y_opt_train)
val_dataset = TabularDataset(X_opt_val, y_opt_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Store all models
all_models = {}
model_performance = {}

# 7.1 Train Enhanced GANDALF
print("\n--- Training Enhanced GANDALF ---")
gandalf = EnhancedGANDALF(X_train_scaled.shape[1]).to(device)
gandalf, gandalf_acc = train_neural_network(gandalf, train_loader, val_loader, epochs=50)
all_models['GANDALF'] = gandalf
model_performance['GANDALF'] = gandalf_acc

# 7.2 Train Deep Neural Network
print("\n--- Training Deep Neural Network ---")
deep_nn = DeepNN(X_train_scaled.shape[1]).to(device)
deep_nn, deep_nn_acc = train_neural_network(deep_nn, train_loader, val_loader, epochs=50)
all_models['DeepNN'] = deep_nn
model_performance['DeepNN'] = deep_nn_acc

# 7.3 Train TabNet
print("\n--- Training TabNet ---")
tabnet = TabNetClassifier(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    momentum=0.98,
    seed=SEED,
    verbose=0
)

tabnet.fit(
    X_opt_train, y_opt_train,
    eval_set=[(X_opt_val, y_opt_val)],
    max_epochs=50,
    patience=15,
    batch_size=256,
    virtual_batch_size=128
)

tabnet_pred = tabnet.predict_proba(X_opt_val)[:, 1]
tabnet_acc = accuracy_score(y_opt_val, tabnet_pred > 0.5)
all_models['TabNet'] = tabnet
model_performance['TabNet'] = tabnet_acc
print(f"TabNet Validation Accuracy: {tabnet_acc:.4f}")

# 7.4 Train CatBoost
print("\n--- Training CatBoost ---")
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=SEED,
    verbose=False
)

cat_model.fit(X_opt_train_df, y_opt_train, 
              eval_set=(X_opt_val_df, y_opt_val),
              early_stopping_rounds=50, verbose=False)
              
cat_acc = accuracy_score(y_opt_val, cat_model.predict(X_opt_val_df))
all_models['CatBoost'] = cat_model
model_performance['CatBoost'] = cat_acc
print(f"CatBoost Validation Accuracy: {cat_acc:.4f}")

# 7.5 Train LightGBM with best params
print("\n--- Training LightGBM ---")
lgb_params = study.best_params
lgb_params.update({'n_estimators': 1500, 'random_state': SEED, 'verbosity': -1})
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_opt_train, y_opt_train, 
              eval_set=[(X_opt_val, y_opt_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
              
lgb_acc = accuracy_score(y_opt_val, lgb_model.predict(X_opt_val))
all_models['LightGBM'] = lgb_model
model_performance['LightGBM'] = lgb_acc
print(f"LightGBM Validation Accuracy: {lgb_acc:.4f}")

# 7.6 Train XGBoost
print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=SEED,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_opt_train, y_opt_train, 
              eval_set=[(X_opt_val, y_opt_val)],
              early_stopping_rounds=50, verbose=False)
              
xgb_acc = accuracy_score(y_opt_val, xgb_model.predict(X_opt_val))
all_models['XGBoost'] = xgb_model
model_performance['XGBoost'] = xgb_acc
print(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")

# 7.7 Train Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1
)
rf_model.fit(X_opt_train, y_opt_train)
rf_acc = accuracy_score(y_opt_val, rf_model.predict(X_opt_val))
all_models['RandomForest'] = rf_model
model_performance['RandomForest'] = rf_acc
print(f"Random Forest Validation Accuracy: {rf_acc:.4f}")

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
for model, acc in sorted(model_performance.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:20s}: {acc:.4f}")

# =====================================================
# 8. ENSEMBLE WITH STACKING
# =====================================================
print("\n" + "="*60)
print("8. CREATING ENSEMBLE")
print("="*60)

# Get validation predictions from all models
val_predictions = np.zeros((len(X_opt_val), len(all_models)))

for i, (name, model) in enumerate(all_models.items()):
    if name in ['GANDALF', 'DeepNN']:
        # Neural network predictions
        model.eval()
        test_dataset_val = TabularDataset(X_opt_val)
        test_loader_val = DataLoader(test_dataset_val, batch_size=256)
        
        preds = []
        with torch.no_grad():
            for X_batch in test_loader_val:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds.extend(outputs.cpu().numpy())
        val_predictions[:, i] = np.array(preds)
        
    elif name == 'TabNet':
        val_predictions[:, i] = model.predict_proba(X_opt_val)[:, 1]
    elif name == 'CatBoost':
        val_predictions[:, i] = model.predict_proba(X_opt_val_df)[:, 1]
    else:
        val_predictions[:, i] = model.predict_proba(X_opt_val)[:, 1]

# Simple meta-learner
class MetaLearner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

# Train meta-learner
print("Training meta-learner...")
meta_model = MetaLearner(len(all_models)).to(device)
meta_dataset = TabularDataset(val_predictions, y_opt_val)
meta_loader = DataLoader(meta_dataset, batch_size=64, shuffle=True)

optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(30):
    meta_model.train()
    for X_batch, y_batch in meta_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = meta_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# =====================================================
# 9. FINAL PREDICTIONS
# =====================================================
print("\n" + "="*60)
print("9. GENERATING FINAL PREDICTIONS")
print("="*60)

# Get test predictions from all models
test_predictions = np.zeros((len(X_test), len(all_models)))

for i, (name, model) in enumerate(all_models.items()):
    print(f"Getting predictions from {name}...")
    
    if name in ['GANDALF', 'DeepNN']:
        # Neural network predictions
        model.eval()
        test_dataset = TabularDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        preds = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds.extend(outputs.cpu().numpy())
        test_predictions[:, i] = np.array(preds)
        
    elif name == 'TabNet':
        test_predictions[:, i] = model.predict_proba(X_test_scaled)[:, 1]
    elif name == 'CatBoost':
        test_predictions[:, i] = model.predict_proba(X_test_df)[:, 1]
    else:
        test_predictions[:, i] = model.predict_proba(X_test_scaled)[:, 1]

# Apply meta-learner
meta_model.eval()
with torch.no_grad():
    meta_input = torch.FloatTensor(test_predictions).to(device)
    ensemble_pred = meta_model(meta_input).cpu().numpy()

# Additional ensemble: weighted average
weighted_pred = (0.3 * test_predictions[:, model_performance['CatBoost']] + 
                 0.25 * test_predictions[:, model_performance['LightGBM']] +
                 0.2 * test_predictions[:, model_performance['XGBoost']] +
                 0.15 * test_predictions[:, model_performance['RandomForest']] +
                 0.1 * np.mean([test_predictions[:, i] for i in range(test_predictions.shape[1])], axis=0))

# Combine meta-learner and weighted average
final_pred = 0.6 * ensemble_pred + 0.4 * weighted_pred

# Convert to binary
final_binary = (final_pred > 0.5).astype(int)

# =====================================================
# 10. CREATE SUBMISSION
# =====================================================
print("\n" + "="*60)
print("10. CREATING SUBMISSION")
print("="*60)

submission = pd.DataFrame({
    'id': test_ids,
    'Depression': final_binary
})

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

# Save probabilities
prob_submission = pd.DataFrame({
    'id': test_ids,
    'Depression_probability': final_pred
})
prob_submission.to_csv('submission_probabilities.csv', index=False)

# Save individual model predictions
individual_predictions = pd.DataFrame(test_predictions, columns=list(all_models.keys()))
individual_predictions['id'] = test_ids
individual_predictions.to_csv('individual_model_predictions.csv', index=False)

# Final statistics
print(f"\nPrediction distribution:")
print(f"Depression = 0: {(final_binary == 0).sum()} ({(final_binary == 0).mean():.2%})")
print(f"Depression = 1: {(final_binary == 1).sum()} ({(final_binary == 1).mean():.2%})")

print("\n" + "="*60)
print("PIPELINE COMPLETE! ðŸš€")
print("="*60)

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

