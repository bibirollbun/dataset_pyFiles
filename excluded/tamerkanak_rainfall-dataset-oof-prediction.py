pip install -q tabpfn


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tabpfn import TabPFNClassifier
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    print("CUDA is not currently available. Please check the following:")
    print("- Do you have a CUDA-capable GPU?")
    print("- Are CUDA and cuDNN libraries installed?")
    print("- Did you install the CUDA-enabled version of PyTorch? (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")
else:
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', torch.version.cuda)
    print('CUDA device count:', torch.cuda.device_count())
    print('Current CUDA device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))



train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall'].astype(int).values
X_test = test.drop(['id'], axis=1)


COLS = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
        'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Mean rainfall value
m = train.rainfall.mean()

# Concatenate all data (for feature engineering if needed)
all_data = pd.concat([X, X_test], axis=0, ignore_index=True)

# Feature engineering: for each feature, calculate mean rainfall per value
for c in COLS:
    n = f"{c}_mean_target"
    # Calculate mean rainfall per feature value in train data
    feature_means = train.groupby(c)['rainfall'].mean().to_dict()
    
    # Add to train
    X[n] = X[c].map(feature_means)
    X[n] = X[n].fillna(m)
    
    # Add to test
    X_test[n] = X_test[c].map(feature_means)
    X_test[n] = X_test[n].fillna(m)

# Check for NULL values (very important!)
null_train = X.isnull().sum().sum()
null_test = X_test.isnull().sum().sum()
if null_train > 0 or null_test > 0:
    print(f"WARNING: There are NULL values after feature engineering! Train NULL: {null_train}, Test NULL: {null_test}")
    # Fill NULLs if any
    X = X.fillna(m)
    X_test = X_test.fillna(m)
    print("NULL values have been filled with the mean rainfall.")
else:
    print("No NULL values after feature engineering.")

print(f"After feature engineering - Train shape: {X.shape}")
print(f"After feature engineering - Test shape: {X_test.shape}")


n_folds = 6
folds = np.zeros(len(X))

fold_size = 365
for fold_idx in range(n_folds):
    start_idx = fold_idx * fold_size
    end_idx = start_idx + fold_size
    # For the last fold, stop at the end of the data
    if end_idx > len(X):
        end_idx = len(X)
    folds[start_idx:end_idx] = fold_idx


print("\n=== FOLD INFORMATION ===")
for fold in range(n_folds):
    fold_indices = np.where(folds == fold)[0]
    fold_ids = train.iloc[fold_indices]['id']
    fold_days = train.iloc[fold_indices]['day']
    fold_X = X.iloc[fold_indices]
    nan_count = fold_X.isna().sum().sum()
    
    print(f"Fold {fold}:")
    print(f"  - Size: {len(fold_indices)}")
    print(f"  - ID range: {fold_ids.min()} - {fold_ids.max()}")
    print(f"  - Day range: {fold_days.min()} - {fold_days.max()}")
    print(f"  - Target distribution: {np.bincount(y[fold_indices])}")
    print(f"  - Number of NaNs: {nan_count}")


models = {
    'xgb': xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, 
                             tree_method='hist', verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    'cat': cb.CatBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=0),
    'tabpfn': TabPFNClassifier(n_estimators=32, random_state=42, device="cuda"),
    'lr': LogisticRegression(random_state=42, max_iter=5000),
    'svc': SVC(probability=True, random_state=42, kernel='rbf')
}


print(f"\n=== MODEL TRAINING ===")

from sklearn.preprocessing import StandardScaler

# 6 - Cross validation with out-of-fold prediction
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros((n_folds, len(X_test))) for name in models}

for fold in range(n_folds):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Fit and apply scaler in each fold
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    
    for name, model in models.items():
        print(f"  Training {name}...")
        # Scaling may not be necessary for tree-based models, but we apply it to all here
        model.fit(X_tr_scaled, y_tr)
        
        # OOF predictions
        oof_preds[name][val_idx] = model.predict_proba(X_val_scaled)[:, 1]
        
        # Test predictions
        test_preds[name][fold] = model.predict_proba(X_test_scaled)[:, 1]
        
        # Validation score
        val_auc = roc_auc_score(y_val, oof_preds[name][val_idx])
        print(f"    {name} Fold {fold} OOF AUC: {val_auc:.4f}")


print(f"\n=== OVERALL OOF SCORES ===")
for name in models.keys():
    overall_auc = roc_auc_score(y, oof_preds[name])
    print(f"{name} Overall OOF AUC: {overall_auc:.4f}")


print(f"\n=== ENSEMBLE ===")

# For test predictions, take the mean across folds for each model
test_preds_mean = {name: test_preds[name].mean(axis=0) for name in models}

# Equal-weighted ensemble
ensemble_test = np.mean([test_preds_mean[name] for name in models], axis=0)
ensemble_oof = np.mean([oof_preds[name] for name in models], axis=0)

print(f"Ensemble OOF AUC: {roc_auc_score(y, ensemble_oof):.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sample['rainfall'] = ensemble_test
sample.to_csv('submission.csv', index=False)

print(f"\nSubmission file created: submission.csv")
print(f"Ensemble prediction summary:")
print(f"  - Min: {ensemble_test.min():.4f}")
print(f"  - Max: {ensemble_test.max():.4f}")
print(f"  - Mean: {ensemble_test.mean():.4f}")


# Train on the entire training set and predict X_test (for each model separately)
final_test_preds = {}

# Fit scaler on the entire training set
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
X_test_scaled_full = scaler_full.transform(X_test)

# Recreate models
models_full = {
    'xgb': xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, 
                             tree_method='hist', verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    'cat': cb.CatBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=0),
    'tabpfn': TabPFNClassifier(n_estimators=32, random_state=42, device="cuda"),
    'lr': LogisticRegression(random_state=42, max_iter=5000),
    'svc': SVC(probability=True, random_state=42, kernel='rbf')
}

for name, model in models_full.items():
    print(f"\nTraining {name} model on the full training set and making test predictions...")
    model.fit(X_scaled_full, y)
    final_test_preds[name] = model.predict_proba(X_test_scaled_full)[:, 1]
    print(f"{name} test predictions: min={final_test_preds[name].min():.4f}, max={final_test_preds[name].max():.4f}, mean={final_test_preds[name].mean():.4f}")



# Create a separate submission file for each model in final_test_preds
for name, preds in final_test_preds.items():
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
    sample_sub['rainfall'] = preds
    out_path = f'submission_{name}.csv'
    sample_sub.to_csv(out_path, index=False)
    print(f"Submission file created for {name}: {out_path}")


