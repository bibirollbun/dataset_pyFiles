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


# ===============================
# ROAD ACCIDENT RISK PREDICTION
# Kaggle Playground Series S5E10
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)

print("âœ… Libraries loaded successfully!")



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Columns:", train.columns.tolist()[:15], "...")  # show some cols
train.head()



train.columns



# Drop ID if present
if 'id' in train.columns:
    train.drop(columns=['id'], inplace=True)
if 'id' in test.columns:
    test.drop(columns=['id'], inplace=True)

# Target variable
TARGET = 'accident_risk'
y = train[TARGET]
X = train.drop(columns=[TARGET])

# Handle categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test), columns=test.columns)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

print("âœ… Preprocessing completed!")



from lightgbm import early_stopping, log_evaluation

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'seed': SEED
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmses = []
    for train_idx, valid_idx in kf.split(X_scaled):
        X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse',
            callbacks=[early_stopping(100), log_evaluation(0)]
        )
        
        preds = model.predict(X_valid)
        rmses.append(mean_squared_error(y_valid, preds, squared=False))
        
    return np.mean(rmses)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

best_params = study.best_params
print("âœ… Best parameters found:\n", best_params)



from lightgbm import early_stopping, log_evaluation

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
preds_lgb = np.zeros(len(test_scaled))
cv_rmse = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled)):
    print(f"ðŸ”¹ Fold {fold+1}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[early_stopping(100), log_evaluation(0)]
    )

    preds = model.predict(X_valid)
    fold_rmse = mean_squared_error(y_valid, preds, squared=False)
    cv_rmse.append(fold_rmse)
    preds_lgb += model.predict(test_scaled) / kf.n_splits

print(f"âœ… LightGBM CV RMSE: {np.mean(cv_rmse):.5f}")



# --- Ensemble Phase ---
import xgboost as xgb
from catboost import CatBoostRegressor

# --- Train CatBoost ---
cat = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=1500,
    loss_function='RMSE',
    random_seed=SEED,
    verbose=False
)
cat.fit(X_scaled, y)

# --- Train XGBoost ---
xg = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=SEED
)
xg.fit(X_scaled, y)

# --- Get predictions from each model ---
preds_lgb = model.predict(test_scaled)  # From your previous trained LightGBM model
preds_cat = cat.predict(test_scaled)
preds_xgb = xg.predict(test_scaled)

# --- Weighted blending (tune weights if needed) ---
final_preds = (0.5 * preds_lgb) + (0.3 * preds_cat) + (0.2 * preds_xgb)

# --- Post-processing ---
final_preds = np.clip(final_preds, 0, 1)

# --- Create submission file ---
submission = pd.DataFrame({
    "id": test.index,
    "accident_risk": final_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved successfully!")
print("Preview:")
print(submission.head())



test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print(test.head())
print(test.columns)



submission = pd.DataFrame({
    "id": test['id'],           # Use the original IDs
    "accident_risk": final_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved successfully!")
print(submission.head())



import matplotlib.pyplot as plt
import lightgbm as lgb

lgb.plot_importance(model, max_num_features=20, figsize=(10,6))
plt.title("Top 20 Feature Importances - LightGBM")
plt.show()


