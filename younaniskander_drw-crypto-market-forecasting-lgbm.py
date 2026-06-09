# DRW Crypto Market Prediction - Memory Optimized (Fixed)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import gc
from tqdm import tqdm

# Configuration
class Config:
    LAGS = [1, 5, 15, 60]  # Reduced lags for memory safety
    CORE_FEATURES = ['bid_qty', 'ask_qty', 'volume']  # Core features only
    N_FOLDS = 3
    SAMPLE_FRAC = 0.5  # Use 50% of data if memory constrained

# Memory-safe processing
def process_data():
    """Load and process data with memory safeguards"""
    # 1. Load data with memory monitoring
    # print("Loading data...")
    train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    
    # Subsample if needed
    if Config.SAMPLE_FRAC < 1:
        train = train.sample(frac=Config.SAMPLE_FRAC, random_state=42)
    
    # 2. Create lag features in batches
    # print("Creating lag features...")
    for feat in Config.CORE_FEATURES:
        for lag in Config.LAGS:
            train[f'{feat}_lag_{lag}'] = train[feat].shift(lag).astype(np.float32)
            gc.collect()
    
    # 3. Drop NA and reduce memory
    train = train.dropna()
    for col in train.columns:
        if train[col].dtype == 'float64':
            train[col] = train[col].astype(np.float32)
    
    return train, test

# Main execution
def main():
    # 1. Process data
    train, test = process_data()
    
    # 2. Feature engineering
    # print("Engineering features...")
    train['imbalance'] = (train['bid_qty'] - train['ask_qty']) / (train['bid_qty'] + train['ask_qty'] + 1e-6)
    features = Config.CORE_FEATURES + [
        col for col in train.columns 
        if any(feat in col for feat in ['lag_', 'imbalance'])
    ]
    
    # 3. Train model
    # print("Training model...")
    models = []
    kf = GroupKFold(n_splits=Config.N_FOLDS)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train, groups=np.arange(len(train)))):
        X_train, X_val = train[features].iloc[train_idx], train[features].iloc[val_idx]
        y_train, y_val = train['label'].iloc[train_idx], train['label'].iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=100,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        models.append(model)
        gc.collect()
    
    # 4. Create submission
    print("Creating submission...")
    test_pred = np.zeros(len(test))
    for feat in Config.CORE_FEATURES:
        for lag in Config.LAGS:
            test[f'{feat}_lag_{lag}'] = train[feat].iloc[-lag:].values[0]  # Safe lag for test
    
    test['imbalance'] = (test['bid_qty'] - test['ask_qty']) / (test['bid_qty'] + test['ask_qty'] + 1e-6)
    test_pred = np.mean([model.predict(test[features]) for model in models], axis=0)
    
    submission = pd.DataFrame({
        'row_id': test.index,
        'label': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    # # print("Submission created successfully!")

if __name__ == "__main__":
    main()

