# ===================================================================
#   Kaggle S5E10 - Road Accident Risk (Expert Level)
#   Model: LightGBM with Optuna Hyperparameter Tuning & K-Fold CV
#   Goal: To beat the top score on the leaderboard.
# ===================================================================

# -------------------------------------------------------------------
# Step 1: Zaroori Libraries Install aur Import Karna
# -------------------------------------------------------------------
# Optuna ko install karein
!pip install optuna -q

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

print("Step 1: Libraries successfully imported!")


# -------------------------------------------------------------------
# Step 2: Data Load Karna
# -------------------------------------------------------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e10/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Step 2: Data successfully loaded!")


# -------------------------------------------------------------------
# Step 3: Feature Engineering
# -------------------------------------------------------------------
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

features = [col for col in train_df.columns if col != 'accident_risk']
X_train = train_df[features]
y_train = train_df['accident_risk']
X_test = test_df[features]

print("Step 3: Feature engineering complete.")


# -------------------------------------------------------------------
# Step 4: Optuna (Tez Version)
# -------------------------------------------------------------------

def objective(trial):
    """Optuna ke liye objective function jo best parameters dhoondta hai."""
    
    # Hyperparameters ka search space
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'device': 'gpu',  # <-- CHANGE 1: GPU istemal karein
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    
    N_SPLITS = 3 # <-- CHANGE 2: Folds kam kar diye
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_rmse_scores = []
    
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        oof_rmse_scores.append(rmse)
        
    return np.mean(oof_rmse_scores)

print("Step 4: Starting FASTER hyperparameter tuning with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15) # <-- CHANGE 3: Trials kam kar diye

best_params = study.best_params
# Best parameters mein GPU ki setting add karna na bhoolein
best_params['device'] = 'gpu' 
print("Best parameters found:", best_params)


# -------------------------------------------------------------------
# Step 5: Final Model ko Best Parameters ke sath Train Karna
# -------------------------------------------------------------------
# Best parameters mein kuch zaroori parameters shamil karna
final_params = best_params
final_params['objective'] = 'rmse'
final_params['metric'] = 'rmse'
final_params['n_estimators'] = 2000 # Final training ke liye estimators barha dein
final_params['verbosity'] = -1
final_params['n_jobs'] = -1
final_params['seed'] = 42
final_params['boosting_type'] = 'gbdt'

N_SPLITS = 10 # Final training ke liye 10 folds
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_predictions = np.zeros(X_test.shape[0])

print("\nStep 5: Training final model with best parameters on 10 folds...")
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    print(f"===== Final Fold {fold+1} =====")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    test_predictions += model.predict(X_test) / N_SPLITS

print("Final model training complete!")


# -------------------------------------------------------------------
# Step 6: Submission File Banana
# -------------------------------------------------------------------
test_predictions[test_predictions < 0] = 0
test_predictions[test_predictions > 1] = 1

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': test_predictions})
submission_df.to_csv('road_accident_risk.csv', index=False)

print("\nStep 6: 'road_accident_risk.csv' file has been created successfully!")
display(submission_df.head())

