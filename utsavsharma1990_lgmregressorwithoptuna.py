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


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# ----------------------------
# 2. Load data
# ----------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


# Target variable
y = train["accident_risk"]
X = train.drop(columns=["id", "accident_risk"]).copy()


X_test = test.drop(columns=["id"]).copy()


# 2. Calculate the correlation matrix
# The .corr() method in pandas calculates pairwise correlation coefficients
correlation_matrix = train.select_dtypes(include=['float64', 'int64']).corr()

# 3. Create the heatmap using seaborn
# Define the plot size for better readability
plt.figure(figsize=(10, 8))

# Use sns.heatmap() to create the visualization
sns.heatmap(
    correlation_matrix,
    annot=True,          # Display the correlation values on the heatmap
    cmap='coolwarm',     # Choose a divergent colormap for better visual distinction
    vmin=-1,             # Set the minimum color value to -1
    vmax=1,              # Set the maximum color value to 1
    fmt=".2f",           # Format the annotations to 2 decimal places
    linewidths=.5        # Add lines between cells for clarity
)

# 4. Add titles and show the plot
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()


X_processed = X.copy()
X_test_processed = X_test.copy()


X_processed.head()


# --- 1. Simple Risk Score Feature Engineering ---
# Map categorical features to numerical values for risk score calculation
risk_mapping = {
    'road_type': {'rural': 0, 'urban': 1, 'highway': 2},
    'lighting': {'dim': 0, 'normal': 1, 'bright': 2}
}
X_processed['road_type_encoded'] = X_processed['road_type'].map(risk_mapping['road_type'])
X_processed['lighting_encoded'] = X_processed['lighting'].map(risk_mapping['lighting'])
X_test_processed['road_type_encoded'] = X_test_processed['road_type'].map(risk_mapping['road_type'])
X_test_processed['lighting_encoded'] = X_test_processed['lighting'].map(risk_mapping['lighting'])



# Create a risk score feature based on weighted sum of relevant features
# You can adjust the weights based on domain knowledge
# Example: Higher weight for 'Speed' and 'road_type'
X_processed['risk_score'] = (
    0.5 * X_processed['road_type_encoded'] +
    0.2 * X_processed['lighting_encoded'] +
    0.3 * X_processed['speed_limit']
)
X_test_processed['risk_score'] = (
    0.5 * X_test_processed['road_type_encoded'] +
    0.2 * X_test_processed['lighting_encoded'] +
    0.3 * X_test_processed['speed_limit']
)


X_processed["curvature_squared"] = X_processed["curvature"] ** 2
X_processed["speed_curvature_ratio"] = X_processed["speed_limit"] / (X_processed["curvature"] + 0.01)
X_processed["accident_density"] = X_processed["num_reported_accidents"] / (X_processed["num_lanes"] + 0.1)

X_test_processed["curvature_squared"] = X_test_processed["curvature"] ** 2
X_test_processed["speed_curvature_ratio"] = X_test_processed["speed_limit"] / (X_test_processed["curvature"] + 0.01)
X_test_processed["accident_density"] = X_test_processed["num_reported_accidents"] / (X_test_processed["num_lanes"] + 0.1)


# --- 2. Simple One-Hot Encoding ---
# Identify categorical features
categorical_cols = [
    'road_type', 'lighting', 'weather', 'time_of_day',
    'road_signs_present', 'public_road'
]


X_processed.head()


# Apply one-hot encoding using pandas get_dummies
X_encoded = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_processed, columns=categorical_cols, drop_first=True)

# Align columns to ensure the test set has the same columns as the training set
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Drop the intermediate encoded columns used only for the risk score
X_encoded = X_encoded.drop(columns=['road_type_encoded', 'lighting_encoded'])
X_test_encoded = X_test_encoded.drop(columns=['road_type_encoded', 'lighting_encoded'])




X_encoded.head()


# --- 2. Simple One-Hot Encoding ---
# Identify categorical features
categorical_features = [
    'road_type_rural','road_type_urban', 'lighting_dim','lighting_night', 'weather_foggy','weather_rainy', 
    'time_of_day_evening','time_of_day_morning',
    'road_signs_present_True', 'public_road_True'
]


# --- Configuration ---
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)


def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        # 'n_estimators': trial.suggest_int('n_estimators', 1,3),
        #'learning_rate': trial.suggest_float('learning_rate', 0.01,0.02),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.05, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.5),
        'random_state': 42,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_encoded):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_encoded.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X_encoded.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], preds))
        scores.append(rmse)
    return np.mean(scores)

def xgb_objective(trial):
    params = {
         'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
       #  'n_estimators': trial.suggest_int('n_estimators', 1,3),
        #'learning_rate': trial.suggest_float('learning_rate', 0.01,0.02),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        'random_state': 42,
        'enable_categorical': True,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_encoded):
        model = xgb.XGBRegressor(**params)
        model.fit(X_encoded.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X_encoded.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], preds))
        scores.append(rmse)
    return np.mean(scores)

def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        #'iterations': trial.suggest_int('iterations', 1,3),
        #'learning_rate': trial.suggest_float('learning_rate', 0.01,0.02),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_seed': 42,
        'verbose': 0,
        'cat_features': categorical_features
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_encoded):
        model = cb.CatBoostRegressor(**params)
        model.fit(X_encoded.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X_encoded.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], preds))
        scores.append(rmse)
    return np.mean(scores)


# Need to run optuna for LGBM
lgbm_study = optuna.create_study(direction='minimize')
lgbm_study.optimize(lgbm_objective, n_trials=50)
best_lgbm_params = lgbm_study.best_trial.params
# print(f"Best LGBM params: {best_lgbm_params}")

# Need to run optuna for XGBoost
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=50)
best_xgb_params = xgb_study.best_trial.params
# print(f"Best XGB params: {best_xgb_params}")

# Need to run optuna for CatBoost
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=50)
best_cat_params = cat_study.best_trial.params
# print(f"Best CatBoost params: {best_cat_params}")


# --- Step 2 & 3: Stacking Ensemble with Cross-Validation and OOF Predictions ---
print("\n===== Starting Stacking Ensemble with Tuned Models =====")
oof_preds_xgb = np.zeros(len(X_encoded))
oof_preds_cat = np.zeros(len(X_encoded))
oof_preds_lgbm = np.zeros(len(X_encoded))
test_preds_xgb = []
test_preds_cat = []
test_preds_lgbm = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_encoded)):
    print(f"\n===== Fold {fold + 1} =====")
    X_tr, y_tr = X_encoded.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X_encoded.iloc[val_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42, enable_categorical=True)
    xgb_model.fit(X_tr, y_tr)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb.append(xgb_model.predict(X_test_encoded))
    xgb_rmse_val = np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val)))
    print(f"XGB RMSE (Validation): {xgb_rmse_val:.4f}")

    # CatBoost
    cat_model = cb.CatBoostRegressor(**best_cat_params, verbose=0, random_seed=42, cat_features=categorical_features)
    cat_model.fit(X_tr, y_tr)
    oof_preds_cat[val_idx] = cat_model.predict(X_val)
    test_preds_cat.append(cat_model.predict(X_test_encoded))
    cat_rmse_val = np.sqrt(mean_squared_error(y_val, cat_model.predict(X_val)))
    print(f"CatBoost RMSE (Validation): {cat_rmse_val:.4f}")

    # LightGBM
    lgbm_model = lgb.LGBMRegressor(**best_lgbm_params, random_state=42)
    lgbm_model.fit(X_tr, y_tr)
    oof_preds_lgbm[val_idx] = lgbm_model.predict(X_val)
    test_preds_lgbm.append(lgbm_model.predict(X_test_encoded))
    lgbm_rmse_val = np.sqrt(mean_squared_error(y_val, lgbm_model.predict(X_val)))
    print(f"LGBM RMSE (Validation): {lgbm_rmse_val:.4f}")




# --- Step 4: XGBoost Meta-model training ---
meta_features = pd.DataFrame({
    'xgb_pred': oof_preds_xgb,
    'cat_pred': oof_preds_cat,
    'lgbm_pred': oof_preds_lgbm
})

from xgboost import XGBRegressor

# Use XGBRegressor as the meta-learner
meta_model = XGBRegressor(random_state=42, n_estimators=100)
#meta_model = XGBRegressor(random_state=42, n_estimators=2)
meta_model.fit(meta_features, y)


# --- Step 5: Final predictions on the test set ---
avg_test_preds_xgb = np.mean(test_preds_xgb, axis=0)
avg_test_preds_cat = np.mean(test_preds_cat, axis=0)
avg_test_preds_lgbm = np.mean(test_preds_lgbm, axis=0)
test_meta_features = pd.DataFrame({
    'xgb_pred': avg_test_preds_xgb,
    'cat_pred': avg_test_preds_cat,
    'lgbm_pred': avg_test_preds_lgbm
})
final_predictions = meta_model.predict(test_meta_features)

print("\n===== Ensemble Evaluation =====")
ensemble_rmse_oof = np.sqrt(mean_squared_error(y, meta_model.predict(meta_features)))
print(f"Ensemble RMSE (Overall Validation/OOF): {ensemble_rmse_oof:.4f}")

print("\nFinal Ensemble Predictions (Stacking):")
print(final_predictions[:10])


# ----------------------------
# 9. Submission
# ----------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "target": final_predictions
})
submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv saved successfully.")

