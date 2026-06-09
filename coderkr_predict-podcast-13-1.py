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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna  # for optimizing blend weights

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Drop high-cardinality columns
drop_cols = ['Podcast_Name', 'Episode_Title']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# Separate features and target
X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to 'category' dtype
for col in categorical_cols:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# LightGBM dataset
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols)

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

# Train LightGBM
print("Training LightGBM...")
from lightgbm import early_stopping, log_evaluation

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=5000,
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(500)
    ]
)

# Train CatBoost
print("Training CatBoost...")
cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=100,
    cat_features=categorical_cols,
    verbose=500
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict on validation set
lgb_val_preds = lgb_model.predict(X_val)
cat_val_preds = cat_model.predict(X_val)

# Blend predictions (50% LGBM + 50% CatBoost)
val_preds = 0.5 * lgb_val_preds + 0.5 * cat_val_preds

# Validation RMSE
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test set
lgb_test_preds = lgb_model.predict(test)
cat_test_preds = cat_model.predict(test)

# Optimize Blend Weights using Optuna
def objective(trial):
    # Define the weight for blending
    lgb_weight = trial.suggest_float('lgb_weight', 0.1, 0.9)
    cat_weight = 1.0 - lgb_weight

    # Blend predictions
    blended_preds = lgb_weight * lgb_val_preds + cat_weight * cat_val_preds
    blended_rmse = mean_squared_error(y_val, blended_preds, squared=False)
    return blended_rmse

# Run Optuna to find best blend weight
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best blend weight
best_lgb_weight = study.best_params['lgb_weight']
best_cat_weight = 1.0 - best_lgb_weight

print(f"Optimized Blend Weights -> LGBM: {best_lgb_weight:.4f}, CatBoost: {best_cat_weight:.4f}")

# Final blended predictions for test set using optimized weights
final_preds = best_lgb_weight * lgb_test_preds + best_cat_weight * cat_test_preds

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Final optimized submission saved.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Drop high-cardinality columns
drop_cols = ['Podcast_Name', 'Episode_Title']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# Separate features and target
X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to 'category' dtype
for col in categorical_cols:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train CatBoost
print("Training CatBoost...")

cat_model = CatBoostRegressor(
    iterations=7500,          # Set iterations to 10000
    learning_rate=0.015,        # Set learning rate to 0.01
    depth=10,                   # Depth of the trees
    loss_function='RMSE',      # Objective function (RMSE)
    eval_metric='RMSE',        # Evaluation metric (RMSE)
    random_seed=42,            # Random seed for reproducibility
    early_stopping_rounds=100, # Early stopping to prevent overfitting
    cat_features=categorical_cols,  # Specify categorical features
    verbose=500                # Show progress every 500 iterations
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict on validation set
cat_val_preds = cat_model.predict(X_val)

# Validation RMSE
rmse = mean_squared_error(y_val, cat_val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test set
cat_test_preds = cat_model.predict(test)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': cat_test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Final CatBoost submission saved.")





