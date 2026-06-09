import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub1=pd.read_csv('/kaggle/input/diabetes-prediction-vault/submission.csv')


train['is_train'] = 1  # No quotes
test['is_train'] = 0   # No quotes


full= pd.concat([train, test], axis=0).reset_index(drop=True)


cat_cols = full.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    full[col] = full[col].astype('category')


X = full[full['is_train'] == 1].drop(['diagnosed_diabetes', 'is_train'], axis=1)
y = full[full['is_train'] == 1]['diagnosed_diabetes']
X_test = full[full['is_train'] == 0].drop(['diagnosed_diabetes', 'is_train'], axis=1)


import optuna


"""""def objective(trial):
    # Define the ranges for the hyperparameters
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'device': 'gpu',          # Use 'gpu' here, NOT 'task_type'
        'gpu_platform_id': 0,     # Often needed on Kaggle
        'gpu_device_id': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'n_estimators': 2000 # Use a smaller number during tuning for speed
    }

    # Use a simple 3-fold CV for tuning speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)"""""


"""""study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Increase to 100 for better results

print("Best Score:", study.best_value)
print("Best Params:", study.best_params)

# Update your best_params variable with the results
best_params = study.best_params
best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'n_estimators': 5000})"""""


best_params = {
        'learning_rate': 0.08803846682124453, 
        'num_leaves': 282, 
        'max_depth': 3, 
        'min_child_samples': 92, 
        'subsample': 0.8709267365781787, 
        'colsample_bytree': 0.6668191550393563, 
        'lambda_l1': 5.846589576875182e-05, 
        'lambda_l2': 0.7513901267452217,
        'device': 'gpu',
        'gpu_platform_id': 0,     # Often needed on Kaggle
        'gpu_device_id': 0,
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': 5000
    }


N_FOLDS  = 10


 skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Containers for results
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
importances_list = [] # For the feature importance plot we discussed

print(f"--- Starting {N_FOLDS}-Fold CV ---")

# 2. The Loop
# Note: Using X_processed (the array) and y (the series)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Use [idx] for NumPy arrays and .iloc[idx] for Pandas Series
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize model
    model = lgb.LGBMClassifier(**best_params)
    
    # 3. Training
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0) # 'period' instead of just '0'0        ]
        ])
    
    # 4. Save results
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Collect importance for the plot
    importances_list.append(model.feature_importances_)
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} complete. AUC: {fold_auc:.5f}")

# 5. Final Metric
final_auc = roc_auc_score(y, oof_preds)
print(f"\nFinal Out-Of-Fold AUC: {final_auc:.5f}")


test_id=test['id']


submission= pd.DataFrame({
    'id': test_id,
    'diagnosed_diabetes': test_preds
})



final=(submission*0.5) + (sub1*0.5)


final.to_csv('submission.csv', index=False)


submission.head(2)

