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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train_ids = train['id']
test_ids = test['id']
target = train['accident_risk'].values

train_features = train.drop(['id', 'accident_risk'], axis=1)
test_features = test.drop(['id'], axis=1)

# Comprehensive Feature Engineering
def create_ultra_features(df):
    df = df.copy()
    
    # Basic interactions
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    df['lanes_squared'] = df['num_lanes'] ** 2
    
    # Advanced interactions
    df['lanes_speed_curvature'] = df['num_lanes'] * df['speed_limit'] * df['curvature']
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 0.1)
    df['curvature_speed_ratio'] = df['curvature'] / (df['speed_limit'] + 1)
    
    # Statistical features
    df['curvature_log'] = np.log1p(df['curvature'])
    df['speed_log'] = np.log1p(df['speed_limit'])
    df['curvature_sqrt'] = np.sqrt(df['curvature'])
    
    # Risk indicators
    for q in [0.25, 0.5, 0.75, 0.9]:
        df[f'high_curvature_{int(q*100)}'] = (df['curvature'] > df['curvature'].quantile(q)).astype(int)
        df[f'high_speed_{int(q*100)}'] = (df['speed_limit'] > df['speed_limit'].quantile(q)).astype(int)
    
    df['single_lane'] = (df['num_lanes'] == 1).astype(int)
    df['two_lanes'] = (df['num_lanes'] == 2).astype(int)
    df['multi_lane'] = (df['num_lanes'] >= 3).astype(int)
    df['wide_road'] = (df['num_lanes'] >= 4).astype(int)
    
    # Weather features
    df['is_foggy'] = (df['weather'] == 'foggy').astype(int)
    df['is_rainy'] = (df['weather'] == 'rainy').astype(int)
    df['is_clear'] = (df['weather'] == 'clear').astype(int)
    
    # Lighting features
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_dim'] = (df['lighting'] == 'dim').astype(int)
    df['is_daylight'] = (df['lighting'] == 'daylight').astype(int)
    
    # Visibility combinations
    df['bad_visibility'] = ((df['weather'] == 'foggy') | (df['lighting'] == 'night')).astype(int)
    df['worst_visibility'] = ((df['weather'] == 'foggy') & (df['lighting'] == 'night')).astype(int)
    df['poor_weather'] = ((df['weather'] == 'rainy') & (df['lighting'] == 'dim')).astype(int)
    df['excellent_visibility'] = ((df['weather'] == 'clear') & (df['lighting'] == 'daylight')).astype(int)
    df['moderate_visibility'] = ((df['weather'] == 'clear') & (df['lighting'] == 'dim')).astype(int)
    
    # Road type features
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_rural'] = (df['road_type'] == 'rural').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    
    # Time features
    df['is_morning'] = (df['time_of_day'] == 'morning').astype(int)
    df['is_afternoon'] = (df['time_of_day'] == 'afternoon').astype(int)
    df['is_evening'] = (df['time_of_day'] == 'evening').astype(int)
    df['rush_hour'] = ((df['time_of_day'] == 'morning') | (df['time_of_day'] == 'evening')).astype(int)
    
    # Accident history features
    if 'num_reported_accidents' in df.columns:
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
        df['accidents_log'] = np.log1p(df['num_reported_accidents'])
        df['accidents_squared'] = df['num_reported_accidents'] ** 2
        df['high_accidents'] = (df['num_reported_accidents'] > 2).astype(int)
        df['very_high_accidents'] = (df['num_reported_accidents'] > 4).astype(int)
        df['no_accidents'] = (df['num_reported_accidents'] == 0).astype(int)
        df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
        df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
        df['accidents_visibility'] = df['num_reported_accidents'] * df['bad_visibility']
    
    # Boolean combinations
    df['signs_public'] = (df['road_signs_present'] & df['public_road']).astype(int)
    df['no_signs_public'] = (~df['road_signs_present'] & df['public_road']).astype(int)
    df['signs_private'] = (df['road_signs_present'] & ~df['public_road']).astype(int)
    df['no_signs_private'] = (~df['road_signs_present'] & ~df['public_road']).astype(int)
    
    df['holiday_school'] = (df['holiday'] & df['school_season']).astype(int)
    df['holiday_no_school'] = (df['holiday'] & ~df['school_season']).astype(int)
    df['school_no_holiday'] = (~df['holiday'] & df['school_season']).astype(int)
    df['no_holiday_no_school'] = (~df['holiday'] & ~df['school_season']).astype(int)
    
    # Complex risk scores
    df['danger_score'] = (
        df['worst_visibility'] * 3 +
        df['is_foggy'] * 2 +
        df['is_rainy'] * 1.5 +
        df['is_night'] * 2 +
        (df['curvature'] > 0.7).astype(int) * 3 +
        (df['speed_limit'] > 60).astype(int) * 2 +
        df['single_lane'] * 2 +
        df.get('very_high_accidents', 0) * 4
    )
    
    df['safety_score'] = (
        df['excellent_visibility'] * 3 +
        df['is_clear'] * 2 +
        df['is_daylight'] * 2 +
        (df['curvature'] < 0.3).astype(int) * 2 +
        df['multi_lane'] * 2 +
        df['signs_public'] * 1.5 +
        df.get('no_accidents', 0) * 3
    )
    
    df['risk_balance'] = df['danger_score'] - df['safety_score']
    df['risk_ratio'] = df['danger_score'] / (df['safety_score'] + 1)
    
    # Road-specific interactions
    df['highway_speed'] = df['is_highway'] * df['speed_limit']
    df['highway_curvature'] = df['is_highway'] * df['curvature']
    df['rural_curvature'] = df['is_rural'] * df['curvature']
    df['urban_congestion'] = df['is_urban'] * (df['speed_limit'] < 40).astype(int)
    
    # Weather-road interactions
    df['rainy_highway'] = df['is_rainy'] * df['is_highway']
    df['foggy_rural'] = df['is_foggy'] * df['is_rural']
    df['night_highway'] = df['is_night'] * df['is_highway']
    df['night_rural'] = df['is_night'] * df['is_rural']
    
    # Complex multi-way interactions
    df['speed_visibility_curvature'] = df['speed_limit'] * df['bad_visibility'] * df['curvature']
    df['weather_time_speed'] = df['is_rainy'] * df['rush_hour'] * df['speed_limit']
    df['highway_rush_weather'] = df['is_highway'] * df['rush_hour'] * (df['is_rainy'] + df['is_foggy'])
    
    return df

print("\nCreating features...")
train_features = create_ultra_features(train_features)
test_features = create_ultra_features(test_features)

# Encode categoricals
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in categorical_cols:
    le = LabelEncoder()
    train_features[col] = le.fit_transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))

bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train_features[col] = train_features[col].astype(int)
    test_features[col] = test_features[col].astype(int)

print(f"Total features: {train_features.shape[1]}")

# Neural Network Definition
class RoadAccidentNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(RoadAccidentNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

# Training function for NN
def train_nn_fold(X_train, y_train, X_val, y_val, params):
    # Scale features
    scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=SEED)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    model = RoadAccidentNN(
        input_dim=X_train.shape[1],
        hidden_dims=params['hidden_dims'],
        dropout_rate=params['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= 20:
            break
    
    model.load_state_dict(best_model_state)
    return model, scaler, np.sqrt(best_val_loss)

# Optuna optimization for GBM models
def optimize_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 4, 8),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 35),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 0.5),
        'verbose': -1,
        'random_state': SEED
    }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

print("\nOptimizing LightGBM hyperparameters...")
# Quick optimization on subset
sample_size = min(100000, len(train_features))
sample_idx = np.random.choice(len(train_features), sample_size, replace=False)
X_sample = train_features.iloc[sample_idx]
y_sample = target[sample_idx]

train_idx, val_idx = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_sample))[0]
X_opt_train, X_opt_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
y_opt_train, y_opt_val = y_sample[train_idx], y_sample[val_idx]

study = optuna.create_study(direction='minimize', study_name='lgb_optimization')
study.optimize(lambda trial: optimize_lgb(trial, X_opt_train, y_opt_train, X_opt_val, y_opt_val), n_trials=30)

best_lgb_params = study.best_params
best_lgb_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': SEED
})

print(f"Best LightGBM RMSE: {study.best_value:.5f}")
print(f"Best params: {best_lgb_params}")

# Updated model parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'random_state': SEED,
    'tree_method': 'hist'
}

cat_params = {
    'iterations': 10000,
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 5,
    'random_seed': SEED,
    'verbose': False,
    'loss_function': 'RMSE'
}

nn_params = {
    'hidden_dims': [256, 128, 64, 32],
    'dropout': 0.3,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 512,
    'epochs': 200
}

# Cross-validation
n_folds = 12
target_bins = pd.qcut(target, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train_features))
oof_xgb = np.zeros(len(train_features))
oof_cat = np.zeros(len(train_features))
oof_nn = np.zeros(len(train_features))

pred_lgb = np.zeros(len(test_features))
pred_xgb = np.zeros(len(test_features))
pred_cat = np.zeros(len(test_features))
pred_nn = np.zeros(len(test_features))

print(f"\nTraining with {n_folds}-fold CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, target_bins)):
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{n_folds}")
    print(f"{'='*60}")
    
    X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
    y_train, y_val = target[train_idx], target[val_idx]
    
    # LightGBM
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model_lgb = lgb.train(
        best_lgb_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    pred_lgb += model_lgb.predict(test_features) / n_folds
    
    # XGBoost
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dval, 'val')],
        early_stopping_rounds=200,
        verbose_eval=0
    )
    
    oof_xgb[val_idx] = model_xgb.predict(dval)
    pred_xgb += model_xgb.predict(xgb.DMatrix(test_features)) / n_folds
    
    # CatBoost
    print("Training CatBoost...")
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=False)
    
    oof_cat[val_idx] = model_cat.predict(X_val)
    pred_cat += model_cat.predict(test_features) / n_folds
    
    # Neural Network
    print("Training Neural Network...")
    model_nn, scaler_nn, nn_score = train_nn_fold(X_train.values, y_train, X_val.values, y_val, nn_params)
    
    model_nn.eval()
    with torch.no_grad():
        X_val_scaled = scaler_nn.transform(X_val.values)
        oof_nn[val_idx] = model_nn(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        
        X_test_scaled = scaler_nn.transform(test_features.values)
        pred_nn += model_nn(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy() / n_folds
    
    # Fold scores
    lgb_score = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    xgb_score = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    cat_score = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    
    print(f"LightGBM: {lgb_score:.5f} | XGBoost: {xgb_score:.5f}")
    print(f"CatBoost: {cat_score:.5f} | NN: {nn_score:.5f}")

# Final OOF scores
print(f"\n{'='*60}")
print("FINAL OOF SCORES:")
print(f"{'='*60}")
print(f"LightGBM: {np.sqrt(mean_squared_error(target, oof_lgb)):.5f}")
print(f"XGBoost:  {np.sqrt(mean_squared_error(target, oof_xgb)):.5f}")
print(f"CatBoost: {np.sqrt(mean_squared_error(target, oof_cat)):.5f}")
print(f"NN:       {np.sqrt(mean_squared_error(target, oof_nn)):.5f}")

# Optimal ensemble
from scipy.optimize import differential_evolution

def ensemble_score(weights):
    w = np.abs(weights)
    w = w / w.sum()
    pred = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_nn
    return np.sqrt(mean_squared_error(target, pred))

result = differential_evolution(ensemble_score, [(0, 1)] * 4, seed=SEED, maxiter=2000)
weights = np.abs(result.x)
weights = weights / weights.sum()

print(f"\n{'='*60}")
print("OPTIMAL ENSEMBLE WEIGHTS:")
print(f"{'='*60}")
print(f"LightGBM: {weights[0]:.4f}")
print(f"XGBoost:  {weights[1]:.4f}")
print(f"CatBoost: {weights[2]:.4f}")
print(f"NN:       {weights[3]:.4f}")
print(f"\nEnsemble OOF RMSE: {result.fun:.5f}")

# Final predictions
final_pred = weights[0] * pred_lgb + weights[1] * pred_xgb + weights[2] * pred_cat + weights[3] * pred_nn
final_pred = np.clip(final_pred, 0, 1)

submission = pd.DataFrame({'id': test_ids, 'accident_risk': final_pred})
submission.to_csv('submission.csv', index=False)

print(f"\n{'='*60}")
print("SUBMISSION CREATED!")
print(f"{'='*60}")
print(f"Range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
print(f"Mean: {final_pred.mean():.4f} | Target Mean: {target.mean():.4f}")
print(f"{'='*60}")

