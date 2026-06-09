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


# Beginner-friendly CatBoost Regression — Improved Version

import numpy as np
import pandas as pd
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# List input files for reference
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# =================== 1. Data Loading ===================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Keep test IDs for submission
test_id = test['id']
test = test.drop(columns=['id'])
train = train.drop(columns=['id'])

# =================== 2. Basic EDA & Preprocessing ===================
print("Missing values in train:\n", train.isnull().sum())
print("Missing values in test:\n", test.isnull().sum())

# Fill missing values if any (simple imputation shown, customize as needed)
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

target = "accident_risk"
categorical_cols = ["road_type", "weather", "lighting", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]

# Convert booleans to int
for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

# Ensure categorical columns are string
for col in categorical_cols:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

# =============== 3. (Optional) Feature Engineering ================
# Example: Time of day cyclical encoding (improves model for cyclical vars)
# Uncomment if "time_of_day" is hour integer/float
# train['time_of_day_sin'] = np.sin(2 * np.pi * train['time_of_day'].astype(float) / 24)
# train['time_of_day_cos'] = np.cos(2 * np.pi * train['time_of_day'].astype(float) / 24)
# test['time_of_day_sin'] = np.sin(2 * np.pi * test['time_of_day'].astype(float) / 24)
# test['time_of_day_cos'] = np.cos(2 * np.pi * test['time_of_day'].astype(float) / 24)

# =============== 4. Train/Validation Split ================
X = train.drop(columns=[target])
y = train[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use CatBoost's Pool for both training and validation sets
train_pool = Pool(X_train, label=y_train, cat_features=categorical_cols)
val_pool = Pool(X_val, label=y_val, cat_features=categorical_cols)
test_pool = Pool(test, cat_features=categorical_cols)

# =============== 5. Model Training with Early Stopping ================
final_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=6,
    bagging_temperature=0.3,
    loss_function='RMSE',
    random_seed=42,
    verbose=200
)

# Fit with validation for early stopping
final_model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=100,  # Stop when validation does not improve
    use_best_model=True
)

# =============== 6. Evaluation ================
val_preds = final_model.predict(val_pool)
val_preds = np.clip(val_preds, 0, 1)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.5f}")

# =============== 7. Feature Importance ================
importances = final_model.get_feature_importance()
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.4f}")

# =============== 8. Prediction and Submission ================
test_preds = np.clip(final_model.predict(test_pool), 0, 1)
submission = pd.DataFrame({
    'id': test_id,
    'accident_risk': test_preds
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created successfully and no conversion errors!")

# =================== Notes & Next Steps ===================
# - Consider more advanced feature engineering (statistical aggregates, group-based means, domain features)
# - Experiment with hyperparameters or use grid/random search for further improvements
# - Try bagging/ensembling with other models (e.g., LightGBM, XGBoost)
# - Use more detailed EDA to find outliers and distributions

# ----------------- End of Notebook -------------------








