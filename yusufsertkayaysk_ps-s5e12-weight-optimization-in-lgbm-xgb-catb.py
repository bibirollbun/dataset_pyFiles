import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata 
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]

# --- 2. Categorical Handling ---
# Cast objects to category for LGBM; CatBoost handles them via indices; XGBoost handles via enable_categorical
cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# --- 3. CV Setup ---
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# --- 4. Training Function (GPU Enabled) ---
def train_model(model_type, X, y, test_data, params, cat_features=None):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_data))
    
    print(f"--- Training {model_type.upper()} on GPU ---")
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params, enable_categorical=True, early_stopping_rounds=100)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
        elif model_type == 'cat':
            model = cb.CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, 
                      cat_features=cat_features, early_stopping_rounds=100)

        # Predict
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(test_data)[:, 1] / FOLDS
        
    score = roc_auc_score(y, oof_preds)
    print(f"{model_type.upper()} OOF AUC: {score:.5f}")
    return oof_preds, test_preds

# --- 5. Hyperparameters (GPU Enabled) ---
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.059216255749261655, 
    'num_leaves': 26,     
    'max_depth': 4,
    'lambda_l1': 1.3404844864067962,
    'lambda_l2': 3.1381681073903975e-07,
    'min_child_samples': 95,
    'n_estimators': 5000,  
    'colsample_bytree': 0.5645863195919457,
    'subsample': 0.9745291249731525,
    'random_state': 133,
    'n_jobs': -1,
    'verbosity': -1,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.05488248216649515,
    'max_depth': 3,
    'min_child_weight': 9,
    'subsample': 0.9324156371717225,
    'gamma': 0.5502813883351498,
    'reg_alpha': 2.755473920545449e-07,
    'colsample_bytree': 0.5974622135152583,
    'reg_lambda': 3.1929676333640495,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'gpu_hist',
    'device': 'cuda'
}

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.24708071253037928,
    'l2_leaf_reg': 4.287313623485399,
    'random_strength': 0.01441624319702462,
    'depth': 4,
    'bagging_temperature': 0.09057002851116619,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'allow_writing_files': False,
    'task_type': 'GPU',
    'devices': '0'
}

# --- 6. Train Models ---
oof_lgb, pred_lgb = train_model('lgb', X, y, test, lgb_params)
oof_xgb, pred_xgb = train_model('xgb', X, y, test, xgb_params)
oof_cat, pred_cat = train_model('cat', X, y, test, cat_params, cat_features=cat_cols)

# --- 7. Rank Averaging (The Fix) ---
print("\n--- Running Rank Averaging ---")

# Stack predictions
oof_preds_all = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_preds_all = np.column_stack([pred_lgb, pred_xgb, pred_cat])

# Rank Transformation
# This converts raw probabilities (e.g., 0.85, 0.40) into ranks (e.g., 0.99, 0.45)
# This fixes calibration issues and forces models to agree on the "order" of patients.
oof_ranks = np.zeros_like(oof_preds_all)
test_ranks = np.zeros_like(test_preds_all)

for i in range(oof_preds_all.shape[1]):
    oof_ranks[:, i] = rankdata(oof_preds_all[:, i]) / len(oof_preds_all)
    test_ranks[:, i] = rankdata(test_preds_all[:, i]) / len(test_preds_all)

# Weights: Give slightly more to LGBM since it was your best single model, 
# but force XGB and CAT to contribute significantly (30% each).
# Previous Hill Climb effectively used [0.94, 0.06, 0.0], which caused overfitting.
weights = [0.40, 0.30, 0.30] 

print(f"Applying weights: LGB={weights[0]}, XGB={weights[1]}, CAT={weights[2]}")

# Weighted Average of Ranks
avg_rank_oof = np.average(oof_ranks, axis=1, weights=weights)
avg_rank_test = np.average(test_ranks, axis=1, weights=weights)

score_rank = roc_auc_score(y, avg_rank_oof)
print(f"Rank Averaging OOF AUC: {score_rank:.5f}")

# --- 8. Submission ---
submission = pd.DataFrame({
    "id": pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")["id"],
    "diagnosed_diabetes": avg_rank_test
})

submission.to_csv("submission_rank_avg.csv", index=False)
print("Saved submission_rank_avg.csv")

