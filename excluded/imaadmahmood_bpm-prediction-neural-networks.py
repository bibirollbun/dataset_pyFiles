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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports (available in Kaggle)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def advanced_feature_engineering(df):
    """
    Create advanced features based on domain knowledge and successful patterns
    """
    # 1. Duration transformations
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['LogDuration'] = np.log1p(df['TrackDurationMs'])
    df['SqrtDuration'] = np.sqrt(df['TrackDurationMs'])
    
    # 2. Ratio features (from reference + improvements)
    df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-8)
    df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-8)
    df['Rhythm_Energy_Ratio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
    df['Mood_Energy_Ratio'] = df['MoodScore'] / (df['Energy'] + 1e-8)
    
    # 3. Interaction features (from reference + new ones)
    df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
    df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']
    df['VocalEnergy'] = df['VocalContent'] * df['Energy']
    df['LoudnessEnergy'] = df['AudioLoudness'] * df['Energy']
    
    # 4. Polynomial features for key predictors
    df['RhythmScore_squared'] = df['RhythmScore'] ** 2
    df['Energy_squared'] = df['Energy'] ** 2
    df['MoodScore_squared'] = df['MoodScore'] ** 2
    
    # 5. Composite scores
    df['IntensityScore'] = (df['Energy'] + df['AudioLoudness'] / -10) / 2
    df['ComplexityScore'] = df['AcousticQuality'] + df['InstrumentalScore'] + df['VocalContent']
    df['PerformanceProfile'] = df['LivePerformanceLikelihood'] * df['Energy'] * df['RhythmScore']
    
    # 6. Binned features for non-linear patterns
    df['Energy_binned'] = pd.cut(df['Energy'], bins=5, labels=[0,1,2,3,4]).astype(int)
    df['Duration_binned'] = pd.cut(df['TrackDurationMin'], bins=5, labels=[0,1,2,3,4]).astype(int)
    df['Rhythm_binned'] = pd.cut(df['RhythmScore'], bins=5, labels=[0,1,2,3,4]).astype(int)
    
    return df

def handle_outliers(df, features_to_clip):
    """
    Apply winsorization (outlier clipping) to specified features
    """
    for col in features_to_clip:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Winsorization (capping outliers)
            df[col] = np.where(df[col] < lower, lower,
                              np.where(df[col] > upper, upper, df[col]))
    return df

# Neural Network Architecture
class BPMNet(nn.Module):
    def __init__(self, input_dim):
        super(BPMNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
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
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, y_train, X_val, y_val, input_dim):
    """
    Train neural network with early stopping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # Initialize model
    model = BPMNet(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(200):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def predict_neural_network(model, X):
    """
    Make predictions with neural network
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions

def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess data with advanced feature engineering
    """
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Original features to handle outliers
    outlier_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                       'InstrumentalScore', 'Energy', 'TrackDurationMs']
    
    # Handle outliers
    train = handle_outliers(train, outlier_features)
    test = handle_outliers(test, outlier_features)
    
    # Feature engineering
    train = advanced_feature_engineering(train)
    test = advanced_feature_engineering(test)
    
    # Get all features (exclude id and target)
    feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
    
    # Extract target and IDs
    y = train['BeatsPerMinute'].values
    train_ids = train['id'].values
    test_ids = test['id'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    
    return X_train, y, X_test, test_ids, feature_cols

def train_xgboost_optimized(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with optimized parameters
    """
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)], 
             early_stopping_rounds=50, 
             verbose=False)
    
    return model

def train_lightgbm_optimized(X_train, y_train, X_val, y_val):
    """
    Train LightGBM with optimized parameters
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)], 
                 callbacks=callbacks)
    except:
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, 
                 verbose=False)
    
    return model

def optimize_ensemble_weights(predictions, y_true):
    """
    Find optimal weights for ensemble
    """
    def objective(weights):
        final_pred = np.average(predictions, axis=0, weights=weights)
        return np.sqrt(mean_squared_error(y_true, final_pred))
    
    # Constraints: weights sum to 1, all weights >= 0
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(predictions)
    
    # Initial guess
    initial_weights = [1/len(predictions)] * len(predictions)
    
    # Optimize
    result = minimize(objective, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def main():
    # File paths
    train_path = '/kaggle/input/playground-series-s5e9/train.csv'
    test_path = '/kaggle/input/playground-series-s5e9/test.csv'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_ids, features = load_and_preprocess_data(train_path, test_path)
    print(f"Training with {len(features)} features")
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize prediction arrays
    oof_preds = {'xgb': np.zeros(len(y_train)),
                 'lgb': np.zeros(len(y_train)),
                 'nn': np.zeros(len(y_train))}
    
    test_preds = {'xgb': np.zeros(len(X_test)),
                  'lgb': np.zeros(len(X_test)),
                  'nn': np.zeros(len(X_test))}
    
    print("Training models with cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}/5")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train models
        print("  Training XGBoost...")
        xgb_model = train_xgboost_optimized(X_tr, y_tr, X_val, y_val)
        
        print("  Training LightGBM...")
        lgb_model = train_lightgbm_optimized(X_tr, y_tr, X_val, y_val)
        
        print("  Training Neural Network...")
        nn_model = train_neural_network(X_tr, y_tr, X_val, y_val, X_train.shape[1])
        
        # Out-of-fold predictions
        oof_preds['xgb'][val_idx] = xgb_model.predict(X_val)
        oof_preds['lgb'][val_idx] = lgb_model.predict(X_val)
        oof_preds['nn'][val_idx] = predict_neural_network(nn_model, X_val)
        
        # Test predictions (average across folds)
        test_preds['xgb'] += xgb_model.predict(X_test) / 5
        test_preds['lgb'] += lgb_model.predict(X_test) / 5
        test_preds['nn'] += predict_neural_network(nn_model, X_test) / 5
    
    # Calculate individual model OOF scores
    print("\nOut-of-fold RMSE scores:")
    for model_name, preds in oof_preds.items():
        rmse = np.sqrt(mean_squared_error(y_train, preds))
        print(f"{model_name.upper()}: {rmse:.4f}")
    
    # Optimize ensemble weights
    print("\nOptimizing ensemble weights...")
    pred_list = [oof_preds['xgb'], oof_preds['lgb'], oof_preds['nn']]
    optimal_weights = optimize_ensemble_weights(pred_list, y_train)
    
    print(f"Optimal weights: XGB={optimal_weights[0]:.3f}, "
          f"LGB={optimal_weights[1]:.3f}, NN={optimal_weights[2]:.3f}")
    
    # Final ensemble prediction
    final_oof = np.average(pred_list, axis=0, weights=optimal_weights)
    final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
    print(f"Final ensemble OOF RMSE: {final_rmse:.4f}")
    
    # Generate test predictions
    test_pred_list = [test_preds['xgb'], test_preds['lgb'], test_preds['nn']]
    final_test_pred = np.average(test_pred_list, axis=0, weights=optimal_weights)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': final_test_pred
    })
    
    submission.to_csv('/kaggle/working/neural_enhanced_submission.csv', index=False)
    print("Enhanced submission saved!")
    
    return submission

if __name__ == "__main__":
    submission = main()

