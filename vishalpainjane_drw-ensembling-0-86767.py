import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
import os
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")




def optimize_memory(df, verbose=True):
    """
    Optimize memory usage by downcasting numeric types where possible.
    """
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory usage before optimization: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
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
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory usage after optimization: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

def preprocess_data_chunked(raw_df, chunk_size=10):
    """
    Preprocess data with memory-efficient chunked lag creation.
    """
    assert len(raw_df.shape) == 2

    y = raw_df['label'].to_numpy().astype(np.float32) if 'label' in raw_df.columns else None

    # Original features (keeping your existing selection)
    cols = [
        'X363', 'X405', 'X321',
        'X175', 'X179', 'X137', 'X197', 'X22', 'X40', 'X181',
        'X28', 'X169', 'X198', 'X173',
        'X338', 'X288', 'X385', 'X344', 'X427', 'X587', 'X450',
        'X97', 'X52', 'X444',
        'X598', 'X379', 'X696', 'X297', 'X138',
        'X572', 'X343', 'X586', 'X466', 'X438', 'X452', 'X459',
        'X435', 'X386', 'X55', 'X341', 'X683', 'X428', 'X605',
        'X445', 'X272', 'X180', 'X593', 'X680',
        'X686', 'X692', 'X695',
        "X603", "X674", "X421", "X333",
        "X415", "X345", "X174", "X302", "X178", "X168", "X612",
        'X298', 'X45', 'X46', 'X39', 'X752', 'X759', 'X41', 'X42',
        "buy_qty", "sell_qty", "volume",
        "bid_qty", "ask_qty",
        'X758', 'X296', 'X611', 'X780', 'X451', 'X25', 'X591',
    ]
    
    # Remove duplicates while preserving order
    cols = list(dict.fromkeys(cols))
    available_cols = [col for col in cols if col in raw_df.columns]
    
    print(f"Using {len(available_cols)} features")

    # Select and optimize base features
    df = raw_df[available_cols].copy()
    df = optimize_memory(df, verbose=True)
    assert df.isna().sum().sum() == 0

    # Extended lag features (keeping your existing lags)
    lag_periods = [1, 3, 5, 6, 7, 8, 9, 12, 15, 18, 20, 30, 120, 150, 365]
    
    print("Creating lagged features in chunks...")
    result_df = df.copy()
    
    for i in range(0, len(lag_periods), chunk_size):
        chunk_lags = lag_periods[i:i+chunk_size]
        print(f"  Processing lags: {chunk_lags}")
        
        chunk_dfs = []
        for lag in chunk_lags:
            lagged = df.shift(-lag).add_suffix(f'_lead_{lag}')
            lagged = lagged.fillna(0.0).astype(np.float32)
            chunk_dfs.append(lagged)
        
        if chunk_dfs:
            chunk_combined = pd.concat(chunk_dfs, axis=1)
            result_df = pd.concat([result_df, chunk_combined], axis=1)
            del chunk_dfs, chunk_combined
            gc.collect()
    
    result_df = optimize_memory(result_df, verbose=True)
    
    if y is not None:
        assert 'label' not in result_df.columns
        assert raw_df.shape[0] == result_df.shape[0]
        assert result_df.isna().sum().sum() == 0
        assert result_df.shape[0] == y.shape[0]
    
    print(f"Final feature count: {result_df.shape[1]}")
    
    return result_df, y

class FastEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.cv_scores = {}
        
    def fit(self, X, y, cv_folds=3):
        print("Training Fast Ensemble with GPU acceleration...")
        
        # GPU-optimized models
        self.models = {
            # 'ridge': Ridge(alpha=0.5, copy_X=False, solver='lsqr'),
            # 'ridge_heavy': Ridge(alpha=5.0, copy_X=False, solver='lsqr'),
            # 'lasso': Lasso(alpha=0.05, copy_X=False, max_iter=500),
            # 'elastic': ElasticNet(alpha=0.3, l1_ratio=0.7, copy_X=False, max_iter=500),
            'lgb_1': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                reg_alpha=0.1,
                reg_lambda=0.1
            ),
            'lgb_2': lgb.LGBMRegressor(
                n_estimators=80,
                learning_rate=0.15,
                max_depth=5,
                num_leaves=20,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=123,
                verbosity=-1,
                n_jobs=-1,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                reg_alpha=0.05,
                reg_lambda=0.05
            ),
            'xgb_1': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='rmse',
                reg_alpha=0.1,
                reg_lambda=0.1
            ),
            'xgb_2': xgb.XGBRegressor(
                n_estimators=80,
                learning_rate=0.15,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=123,
                verbosity=0,
                n_jobs=-1,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='rmse',
                reg_alpha=0.05,
                reg_lambda=0.05
            )
        
        }
        
        # Cross-validation for model evaluation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold + 1}/{cv_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y[val_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                print(f"    Training {name}...")
                
                # Clone model for this fold
                if name == 'lgb':
                    fold_model = lgb.LGBMRegressor(**model.get_params())
                elif name == 'xgb':
                    fold_model = xgb.XGBRegressor(**model.get_params())
                else:
                    fold_model = type(model)(**model.get_params())
                
                # Fit and predict
                fold_model.fit(X_train_fold, y_train_fold)
                oof_predictions[val_idx, model_idx] = fold_model.predict(X_val_fold)
        
        # Calculate CV scores for each model
        for model_idx, name in enumerate(self.models.keys()):
            score = mean_squared_error(y, oof_predictions[:, model_idx])
            self.cv_scores[name] = score
            print(f"  {name} CV MSE: {score:.6f}")
        
        # Calculate optimal weights using inverse MSE
        mse_values = np.array(list(self.cv_scores.values()))
        inverse_mse = 1.0 / (mse_values + 1e-8)  # Add small epsilon to avoid division by zero
        self.weights = inverse_mse / inverse_mse.sum()
        
        print("  Optimal weights:")
        for i, (name, weight) in enumerate(zip(self.models.keys(), self.weights)):
            print(f"    {name}: {weight:.4f}")
        
        # Fit final models on full data
        print("  Fitting final models on full data...")
        for name, model in self.models.items():
            print(f"    Fitting {name}...")
            model.fit(X, y)
        
        # Calculate ensemble CV score
        ensemble_pred = np.average(oof_predictions, weights=self.weights, axis=1)
        ensemble_score = mean_squared_error(y, ensemble_pred)
        print(f"  Ensemble CV MSE: {ensemble_score:.6f}")
        
        return self
    
    def predict(self, X, batch_size=50000):
        """
        Make predictions with the ensemble.
        """
        print("Making ensemble predictions...")
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.models)))
        
        # Process in batches to save memory
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X.iloc[i:end_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                predictions[i:end_idx, model_idx] = model.predict(X_batch)
            
            if i // batch_size % 10 == 0:  # Print every 10 batches
                print(f"  Processed {end_idx}/{n_samples} samples")
        
        # Weighted average
        final_predictions = np.average(predictions, weights=self.weights, axis=1)
        return final_predictions.astype(np.float32)


# Set memory-efficient options for pandas
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None

# Load and preprocess training data
print("Loading training data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')

print(f"\nTotal columns in training data: {len(train_df.columns)}")
print(f"Sample columns: {list(train_df.columns[:20])}")

print("\nOptimizing memory for raw training data...")
train_df = optimize_memory(train_df, verbose=True)

X_train, y_train = preprocess_data_chunked(train_df, chunk_size=10)

del train_df
gc.collect()

print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")

# Train the ensemble model
ensemble = FastEnsemble()
ensemble.fit(X_train, y_train, cv_folds=3)

# Feature importance from best performing model
best_model_name = min(ensemble.cv_scores, key=ensemble.cv_scores.get)
best_model = ensemble.models[best_model_name]

print(f"\nBest single model: {best_model_name}")

# Plot feature importance for interpretable models
if hasattr(best_model, 'coef_'):
    print("\nPlotting feature importance...")
    coef_series = pd.Series(best_model.coef_, index=X_train.columns).abs().sort_values(ascending=False)
    top_features = coef_series.head(50)  # Reduced for faster plotting
    
    plt.figure(figsize=(12, 15))
    top_features.sort_values().plot(kind='barh')
    plt.title(f'Top 50 Feature Coefficients ({best_model_name})')
    plt.xlabel('Coefficient Magnitude')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 20 most important features:")
    for i, (feat, coef) in enumerate(coef_series.head(20).items(), 1):
        print(f"{i:2d}. {feat:30s} {coef:.6f}")

# Clean up training data
del X_train, y_train
gc.collect()

# Load and process test data
print("\nLoading test data...")
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

print("\nOptimizing memory for raw test data...")
test_df = optimize_memory(test_df, verbose=True)

# Timestamp reconstruction (keeping your existing logic)
timestamp_recon_path = '/kaggle/input/closest-rows/closest_rows.csv'
use_timestamp_reconstruction = os.path.exists(timestamp_recon_path)

if use_timestamp_reconstruction:
    print("Found timestamp reconstruction file, loading...")
    t = pd.Series(pd.read_csv(timestamp_recon_path)['0'].to_numpy())
    assert t.shape == (test_df.shape[0],)
    print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))

    # Process timestamp reconstruction
    t -= 10080
    t[t < 0] = 538149
    t = t.sort_values()
    t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
    t = t.sort_index()
    t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()

    # Sort test dataset
    test_df = test_df.iloc[t.to_numpy()]
else:
    print("WARNING: Timestamp reconstruction file not found!")
    t = pd.Series(np.arange(len(test_df)))

# Preprocess test data
print("\nPreprocessing test data...")
X_test, _ = preprocess_data_chunked(test_df, chunk_size=10)

del test_df
gc.collect()

print(f"Test data shape: {X_test.shape}")

# Make ensemble predictions
y_pred = ensemble.predict(X_test, batch_size=50000)

del X_test
gc.collect()

# Display prediction statistics
print("\nPrediction statistics:")
print(pd.Series(y_pred).describe())

# Plot results (reduced plotting for speed)
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.plot(np.cumsum(y_pred))
plt.title('Cumulative Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Sum')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(y_pred, bins=50, alpha=0.7, edgecolor='black')
plt.title('Prediction Distribution')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.plot(y_pred[:1000])
plt.title('First 1000 Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')

plt.subplot(2, 2, 4)
# Plot model weights
names = list(ensemble.models.keys())
weights = ensemble.weights
plt.bar(names, weights)
plt.title('Ensemble Model Weights')
plt.ylabel('Weight')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Prepare submission
print("\nPreparing submission...")
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

if use_timestamp_reconstruction:
    submission = submission.iloc[t.to_numpy()]
    submission['prediction'] = y_pred
    submission = submission.sort_index()
else:
    submission['prediction'] = y_pred

submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")

print("\nSubmission preview:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction range: [{submission['prediction'].min():.6f}, {submission['prediction'].max():.6f}]")

# Print ensemble summary
print("\n" + "="*50)
print("ENSEMBLE SUMMARY")
print("="*50)
for name, score in ensemble.cv_scores.items():
    weight = ensemble.weights[list(ensemble.models.keys()).index(name)]
    print(f"{name:12s}: CV MSE = {score:.6f}, Weight = {weight:.4f}")

gc.collect()
print("\nDone!")


# Load test data
print("\nLoading test data...")
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# Optimize memory for test data
print("\nOptimizing memory for raw test data...")
test_df = optimize_memory_gpu(test_df, verbose=True)

# Try to load precomputed timestamp reconstruction data
timestamp_recon_path = '/kaggle/input/closest-rows/closest_rows.csv'
use_timestamp_reconstruction = os.path.exists(timestamp_recon_path)

if use_timestamp_reconstruction:
    print("Found timestamp reconstruction file, loading...")
    
    # Load precomputed timestamp reconstruction data
    t = pd.Series(pd.read_csv(timestamp_recon_path)['0'].to_numpy())
    assert t.shape == (test_df.shape[0],)
    print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))

    # Visualize the reconstructed timestamps
    plt.figure(figsize=(16, 4))
    plt.plot(t.sort_values().to_numpy())
    plt.title('Sorted Reconstructed Timestamps')
    plt.show()

    plt.figure(figsize=(16, 4))
    plt.plot(t[t >= 0].sort_values().iloc[:1000].to_numpy())
    plt.axhline(10080, color='r', linestyle='--')
    plt.title('First 1000 Valid Reconstructed Timestamps')
    plt.show()

    # Process timestamp reconstruction
    t -= 10080
    t[t < 0] = 538149

    t = t.sort_values()
    t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
    t = t.sort_index()

    t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()

    # Visualize test data before sorting
    if 'X656' in test_df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(test_df['X656'].to_numpy())
        plt.title('Test Data Feature X656 - Before Sorting')
        plt.show()

    # Sort test dataset by reconstructed time order
    test_df = test_df.iloc[t.to_numpy()]

    # Visualize test data after sorting
    if 'X656' in test_df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(test_df['X656'].to_numpy())
        plt.title('Test Data Feature X656 - After Sorting')
        plt.show()
else:
    print("WARNING: Timestamp reconstruction file not found!")
    print(f"Expected path: {timestamp_recon_path}")
    print("Proceeding without timestamp reconstruction...")
    print("This may significantly impact model performance since lagged features assume temporal order.")
    
    t = pd.Series(np.arange(len(test_df)))

# Preprocess test data
# ...existing code for loading and timestamp reconstruction...

# Preprocess test data with enhanced features
print("\nPreprocessing test data with enhanced features...")
X_test, _ = preprocess_data_chunked(test_df, chunk_size=10, create_advanced=True, top_features=top_features, enhanced=True)

# Apply same feature selection
X_test_selected = X_test[selected_features_list]

# Clean up test dataframe
del test_df, X_test
gpu_memory_cleanup()
gc.collect()

print(f"Test data shape after processing: {X_test_selected.shape}")

# Make predictions in smaller batches
print("\nMaking predictions...")
batch_size = 40000  # Reduced from 80000
n_samples = X_test_selected.shape[0]
y_pred = np.zeros(n_samples, dtype=np.float32)

for i in range(0, n_samples, batch_size):
    end_idx = min(i + batch_size, n_samples)
    print(f"  Predicting batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    
    batch_data = X_test_selected.iloc[i:end_idx]
    y_pred[i:end_idx] = final_model.predict(batch_data).astype(np.float32)
    
    # Cleanup every few batches
    if (i // batch_size + 1) % 3 == 0:
        aggressive_memory_cleanup()

# Clean up test features
del X_test_selected
aggressive_memory_cleanup()

# Display prediction statistics
print("\nPrediction statistics:")
print(pd.Series(y_pred).describe())

# Plot cumulative predictions
plt.figure(figsize=(16, 4))
plt.plot(np.cumsum(y_pred))
plt.title('Cumulative Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Sum')
plt.grid(True, alpha=0.3)
plt.show()

# Plot prediction distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(y_pred, bins=100, alpha=0.7, edgecolor='black')
plt.title('Prediction Distribution')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(y_pred[:1000])
plt.title('First 1000 Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.tight_layout()
plt.show()

# Prepare submission
print("\nPreparing submission...")
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

if use_timestamp_reconstruction:
    # Reorder submission to match original test order
    submission = submission.iloc[t.to_numpy()]
    submission['prediction'] = y_pred
    submission = submission.sort_index()
else:
    # If no timestamp reconstruction, just use predictions in order
    submission['prediction'] = y_pred

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")

# Display submission
print("\nSubmission preview:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction range: [{submission['prediction'].min():.6f}, {submission['prediction'].max():.6f}]")

# Final memory cleanup
gc.collect()
print("\nDone!")




