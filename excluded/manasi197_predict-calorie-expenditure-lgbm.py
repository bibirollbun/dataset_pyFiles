# pip install optuna-integration[lightgbm]


# Load & Clean Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Base Preprocessing
train = train.drop_duplicates().reset_index(drop=True)
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})


# Feature Engineering
def add_cross_terms(df, features):
    cross_terms = {}
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            cross_terms[f"{features[i]}_x_{features[j]}"] = df[features[i]] * df[features[j]]
    return pd.concat([df, pd.DataFrame(cross_terms)], axis=1)

numerical = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
train = add_cross_terms(train, numerical)
test = add_cross_terms(test, numerical)


# Additional Features
for df in [train, test]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df['Heart_Rate'] / (df['Duration'] + 1)  # Avoid division by zero
    df['Log_Body_Temp'] = np.log1p(df['Body_Temp'])
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_Heart_Rate'] = np.log1p(df['Heart_Rate'])
    df['Age_Adjusted_Heart_Rate'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']


# Prepare Data
X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])  # Log-transform target
X_test = test.drop(columns=["id"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Stratified KFold
y_bins = pd.qcut(train['Calories'], q=5, labels=False)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Hyperparameter Tuning with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1500, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 40, 100),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 5.0, log=True),
        'random_state': 42,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    oof_preds = np.zeros(len(X))
    for train_idx, valid_idx in kf.split(X, y_bins):
        X_tr, X_val = X_scaled[train_idx], X_scaled[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(150), LightGBMPruningCallback(trial, "rmse")]
        )
        
        oof_preds[valid_idx] = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
best_params.update({'random_state': 42, 'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})


# Final Training
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
print("\nTraining final LightGBM model...")

for fold, (train_idx, valid_idx) in tqdm(enumerate(kf.split(X, y_bins)), total=kf.n_splits, desc="LightGBM Folds"):
    X_tr, X_val = X_scaled[train_idx], X_scaled[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150)]
    )
    
    oof_preds[valid_idx] = model.predict(X_val)
    test_preds += model.predict(X_test_scaled) / kf.n_splits
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[valid_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

full_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Final LightGBM CV RMSE: {full_rmse:.4f}")

# Post-Processing
bias = np.mean(np.expm1(y) - np.expm1(oof_preds))
test_preds = np.expm1(test_preds) + bias
test_preds = np.clip(test_preds, 1, 314)


# Save Submission
submission['Calories'] = test_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSubmission saved.")

