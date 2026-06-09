import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import os
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


INPUT_DIR = '/kaggle/input/playground-s5-e8'
ORIGINAL_DATA_DIR = '/kaggle/input/playground-series-s5e8'

X = pd.read_feather(os.path.join(INPUT_DIR, 'train_processed_ohe.feather'))
X_test = pd.read_feather(os.path.join(INPUT_DIR, 'test_processed_ohe.feather'))
y = pd.read_csv(os.path.join(INPUT_DIR, 'y_train.csv'))['y']


test_ids = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, 'test.csv'))['id']

print(f"Data loaded. Train shape: {X.shape}, Test shape: {X_test.shape}")

# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")


def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'seed': 42,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight,
        
        # Hyperparameters to be tuned by Optuna
        'eta': trial.suggest_float('eta', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
    }

    N_SPLITS_TUNE = 5
    skf = StratifiedKFold(n_splits=N_SPLITS_TUNE, shuffle=True, random_state=42)
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


study = optuna.create_study(direction='maximize', study_name='xgb_tuning')
study.optimize(objective, n_trials=250, timeout=60*60*8)


best_xgb_params = study.best_params
print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best trial AUC: {study.best_value:.5f}")
print("Best hyperparameters found:")
print(best_xgb_params)


best_xgb_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'seed': 42,
    'n_jobs': -1,
    'scale_pos_weight': scale_pos_weight
})


N_SPLITS_FINAL = 10
skf_final = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=42)


oof_xgb_preds = np.zeros(len(X))
test_xgb_preds = np.zeros(len(X_test))


MODELS_DIR = 'models/xgb'
os.makedirs(MODELS_DIR, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X, y)):
    print(f"===== Fold {fold+1}/{N_SPLITS_FINAL} =====")
    
    # Split data for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_xgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)])
    
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_xgb_preds[val_idx] = val_preds
    
    test_xgb_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS_FINAL
    
    model_path = os.path.join(MODELS_DIR, f'xgb_fold_{fold}.pkl')
    joblib.dump(model, model_path)
    print(f"Fold {fold+1} model saved to: {model_path}")
    
    del X_train, y_train, X_val, y_val, model
    gc.collect()


final_oof_auc = roc_auc_score(y, oof_xgb_preds)
print("\n--- Final OOF Training Complete ---")
print(f"Final XGBoost OOF ROC AUC: {final_oof_auc:.5f}")

# Save OOF predictions for ensembling later
np.save('oof_xgb_preds.npy', oof_xgb_preds)
print("OOF predictions saved to oof_xgb_preds.npy")


submission_df = pd.DataFrame({'id': test_ids, 'y': test_xgb_preds})
submission_path = 'submission_xgb_tuned.csv'
submission_df.to_csv(submission_path, index=False)


print(f"Submission file created successfully: {submission_path}")
print("Submission file head:")
print(submission_df.head())

