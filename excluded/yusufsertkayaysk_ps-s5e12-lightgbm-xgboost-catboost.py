import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb  # Added CatBoost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]




# --- IMPROVEMENT 1: Better Categorical Handling ---
# Instead of .cat.codes, we cast to 'category' type for LGBM and let CatBoost handle objects.
# For XGBoost, we still use codes (or OneHot), but we prefer EnableCategorical=True if version supports it.
cat_cols = X.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")




# --- CV Setup ---
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


def train_model(model_type, X, y, test_data, params, cat_features=None):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_data))
    models = []
    
    print(f"--- Training {model_type.upper()} ---")
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        if model_type == 'lgb':
            # LGBM handles categories natively if dtype is 'category'
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            
        elif model_type == 'xgb':
            # XGBoost needs enable_categorical=True for category dtypes
            model = xgb.XGBClassifier(**params, enable_categorical=True)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
            
        elif model_type == 'cat':
            # CatBoost needs specific list of cat features
            model = cb.CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, 
                      cat_features=cat_features, early_stopping_rounds=100)

        # Predictions
        # Best_iteration_ is handled automatically by predict_proba in modern sklearn APIs for these libs
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(test_data)[:, 1] / FOLDS
        models.append(model)
        
    score = roc_auc_score(y, oof_preds)
    print(f"{model_type.upper()} OOF AUC: {score:.5f}")
    return oof_preds, test_preds


# --- Configuration ---

# LightGBM: Native categorical support
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02, # Lower LR for better convergence
    'num_leaves': 31,      # Slightly smaller leaves to prevent Overfitting
    'n_estimators': 5000,  # High cap, controlled by Early Stopping
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# XGBoost: Enable categorical
xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist' # Faster training
}

# CatBoost: The new addition
cat_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'allow_writing_files': False,
    'task_type': 'CPU' 
}

# --- Execution ---
oof_lgb, pred_lgb = train_model('lgb', X, y, test, lgb_params)
oof_xgb, pred_xgb = train_model('xgb', X, y, test, xgb_params)
oof_cat, pred_cat = train_model('cat', X, y, test, cat_params, cat_features=cat_cols)


# --- Weighted Ensemble ---
# Weights can be tuned using Scipy.optimize, but a heuristic starting point:
# CatBoost usually handles categorical data best, so we give it slightly more weight.
w_lgb = 0.30
w_xgb = 0.30
w_cat = 0.40

final_oof = (w_lgb * oof_lgb) + (w_xgb * oof_xgb) + (w_cat * oof_cat)
ensemble_score = roc_auc_score(y, final_oof)
print(f"Weighted Ensemble OOF AUC: {ensemble_score:.5f}")

# Generate Submission
test_pred = (w_lgb * pred_lgb) + (w_xgb * pred_xgb) + (w_cat * pred_cat)

submission = pd.DataFrame({
    "id": pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")["id"],
    "diagnosed_diabetes": test_pred
})
submission.to_csv("submission_optimized.csv", index=False)
print("Saved submission_optimized.csv")

