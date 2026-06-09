# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.describe()


train_df.info()


for c in  ['road_type', 'lighting', 'weather', 'time_of_day']:
    print(train_df[c].unique())


def data_process(df):
    df_new = df.copy()

    numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
    categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
    boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

    # Convert boolean features to int
    for col in boolean_features:
        df_new[col] = df_new[col].astype(int)

    # Ordinal encoding for categorical features
    encoder = OrdinalEncoder()
    df_new[categorical_features] = encoder.fit_transform(df_new[categorical_features])

    return df_new



train_process = data_process(train_df)
test_process = data_process(test_df)


train_process.sample()


X = train_process.drop(["accident_risk", "id"], axis=1)
y = train_process["accident_risk"]
X_test = test_process.drop(["id"], axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(
    learning_rate=0.16795682865614925,
    max_depth=4,
    subsample=0.9742208104810293,
    colsample_bytree=0.8874274285223323,
    n_estimators=1000,
    verbosity=0,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predict 
xgb_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

final_preds = xgb_model.predict(X_test) 

# Submission 
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": final_preds
})

# Save to CSV
submission.to_csv("Xgboost_submission.csv", index=False)
print("âœ… Final predictions saved to submission.csv")


lgb_model = lgb.LGBMRegressor(
    learning_rate=0.16284853116944661,
    num_leaves=109,
    max_depth=5,
    min_child_samples=63,
    feature_fraction=0.8894490458907941,
    bagging_fraction=0.887597096482173,
    bagging_freq=6,
    n_estimators=1000,
    random_state=422
)

lgb_model.fit(X_train, y_train)

# Predict
lgb_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_preds))
print(f"LightGBM RMSE: {lgb_rmse:.4f}")
final_preds = lgb_model.predict(X_test)  # or xgb_model.predict(X_test)

# Submission 
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": final_preds
})

# Save to CSV
submission.to_csv("LightGbm_submission.csv", index=False)
print("âœ… Final predictions saved to submission.csv")



# Predict on validation set
xgb_val_preds = xgb_model.predict(X_val)
lgb_val_preds = lgb_model.predict(X_val)

# Average predictions for validation set
ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2

# Evaluate ensemble performance
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")

# Predict on test set using both models
xgb_test_preds = xgb_model.predict(X_test)
lgb_test_preds = lgb_model.predict(X_test)

# Average predictions for final ensemble
ensemble_test_preds = (xgb_test_preds + lgb_test_preds) / 2

# Prepare submission
ensemble_submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": ensemble_test_preds
})

# Save to CSV
ensemble_submission.to_csv("Ensemble_submission.csv", index=False)
print("âœ… Final ensemble predictions saved to Ensemble_submission.csv")




# Store predictions
xgb_val_preds_list = []
xgb_test_preds_list = []

lgb_val_preds_list = []
lgb_test_preds_list = []

# 8 XGBoost models with GPU
for i in range(8):
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.16795682865614925,
        max_depth=4,
        subsample=0.9742208104810293,
        colsample_bytree=0.8874274285223323,
        n_estimators=1000,
        verbosity=0,
        objective='reg:squarederror',
        random_state=42 + i,
        tree_method='gpu_hist',         # GPU support
        predictor='gpu_predictor'       # Use GPU predictor
    )
    xgb_model.fit(X_train, y_train)
    xgb_val_preds_list.append(xgb_model.predict(X_val))
    xgb_test_preds_list.append(xgb_model.predict(X_test))

# 8 LightGBM models with GPU
for i in range(8):
    lgb_model = lgb.LGBMRegressor(
        learning_rate=0.16284853116944661,
        num_leaves=109,
        max_depth=5,
        min_child_samples=63,
        feature_fraction=0.8894490458907941,
        bagging_fraction=0.887597096482173,
        bagging_freq=6,
        n_estimators=1000,
        random_state=100 + i,
        device='gpu'                    # GPU support
    )
    lgb_model.fit(X_train, y_train)
    lgb_val_preds_list.append(lgb_model.predict(X_val))
    lgb_test_preds_list.append(lgb_model.predict(X_test))

# Average predictions
ensemble_val_preds = np.mean(xgb_val_preds_list + lgb_val_preds_list, axis=0)
ensemble_test_preds = np.mean(xgb_test_preds_list + lgb_test_preds_list, axis=0)

# Evaluate ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
print(f"ðŸš€ Ensemble of 8 XGB + 8 LGB with GPU RMSE: {ensemble_rmse:.4f}")

# Save predictions
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": ensemble_test_preds
})
submission.to_csv("Ensemble_8xgb_8lgb_gpu.csv", index=False)
print("âœ… Final GPU ensemble predictions saved to Ensemble_8xgb_8lgb_gpu.csv")




