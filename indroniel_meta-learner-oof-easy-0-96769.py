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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from category_encoders import CatBoostEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from google.colab import drive


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Perform feature engineering
# Drop the id column as it's not useful for modeling
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Convert binary categorical columns to 0/1
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    train_df[col] = train_df[col].apply(lambda x: 1 if x == 'yes' else 0)
    test_df[col] = test_df[col].apply(lambda x: 1 if x == 'yes' else 0)

# Feature engineering for 'pdays'
train_df['pdays_contacted'] = train_df['pdays'].apply(lambda x: 1 if x != -1 else 0)
test_df['pdays_contacted'] = test_df['pdays'].apply(lambda x: 1 if x != -1 else 0)

# Replace -1 in 'pdays' with a large number before log transformation
train_df['pdays_log'] = train_df['pdays'].replace(-1, 99999)
test_df['pdays_log'] = test_df['pdays'].replace(-1, 99999)

# Apply log transformation
train_df['pdays_log'] = np.log1p(train_df['pdays_log'])
test_df['pdays_log'] = np.log1p(test_df['pdays_log'])

# Create a new feature as the ratio of campaign to duration, handling division by zero
train_df['campaign_duration_ratio'] = train_df['campaign'] / (train_df['duration'] + 1e-6)
test_df['campaign_duration_ratio'] = test_df['campaign'] / (test_df['duration'] + 1e-6)

# Separate features (X) and target (y)
X = train_df.drop('y', axis=1)
y = train_df['y']
X_test = test_df.copy()

# Get the list of numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Define the number of folds for StratifiedKFold
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Initialize OOF prediction arrays
oof_preds_lgbm = np.zeros(len(X))
oof_preds_catboost = np.zeros(len(X))

# Define base models
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'early_stopping_round': 100
}

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 100
}

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_FOLDS} ---")

    # Split the data into train and validation folds
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize and fit CatBoostEncoder on training fold to prevent leakage
    cbe = CatBoostEncoder(cols=categorical_cols, random_state=42)
    X_train_encoded = cbe.fit_transform(X_train_fold, y_train_fold)
    X_val_encoded = cbe.transform(X_val_fold)

    # Initialize and fit RobustScaler on training fold
    scaler = RobustScaler()
    X_train_scaled = X_train_encoded.copy()
    X_val_scaled = X_val_encoded.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_scaled[numerical_cols])
    X_val_scaled[numerical_cols] = scaler.transform(X_val_scaled[numerical_cols])

    # Train LightGBM Model
    print("Training LightGBM model...")
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train_scaled, y_train_fold, eval_set=[(X_val_scaled, y_val_fold)])

    # Make OOF predictions for LightGBM
    oof_preds_lgbm[val_idx] = lgbm_model.predict_proba(X_val_scaled)[:, 1]

    # Train CatBoost Model
    print("Training CatBoost model...")
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(X_train_scaled, y_train_fold, eval_set=(X_val_scaled, y_val_fold), early_stopping_rounds=100, verbose=0)

    # Make OOF predictions for CatBoost
    oof_preds_catboost[val_idx] = catboost_model.predict_proba(X_val_scaled)[:, 1]

print("\n--- Cross-validation complete ---")
print(f"LightGBM OOF AUC: {roc_auc_score(y, oof_preds_lgbm):.4f}")
print(f"CatBoost OOF AUC: {roc_auc_score(y, oof_preds_catboost):.4f}")

# Train the meta-learner on the OOF predictions
print("\n--- Training meta-learner ---")
X_meta = pd.DataFrame({
    'lgbm_oof': oof_preds_lgbm,
    'catboost_oof': oof_preds_catboost
})
meta_learner = LogisticRegression(solver='liblinear', random_state=42)
meta_learner.fit(X_meta, y)
print("Meta-learner trained successfully.")

# Prepare the test data for final predictions
final_cbe = CatBoostEncoder(cols=categorical_cols, random_state=42)
X_encoded_final = final_cbe.fit_transform(X, y)
test_encoded_final = final_cbe.transform(X_test.copy())

final_scaler = RobustScaler()
X_scaled_final = X_encoded_final.copy()
test_scaled_final = test_encoded_final.copy()

X_scaled_final[numerical_cols] = final_scaler.fit_transform(X_scaled_final[numerical_cols])
test_scaled_final[numerical_cols] = final_scaler.transform(test_scaled_final[numerical_cols])

# Train the base models on the entire dataset
final_lgbm = lgb.LGBMClassifier(**{k: v for k, v in lgbm_params.items() if k != 'early_stopping_round'})
final_lgbm.fit(X_scaled_final, y)

final_catboost = CatBoostClassifier(**catboost_params)
final_catboost.fit(X_scaled_final, y, verbose=0)

# Generate predictions on the test data
test_preds_lgbm = final_lgbm.predict_proba(test_scaled_final)[:, 1]
test_preds_catboost = final_catboost.predict_proba(test_scaled_final)[:, 1]

# Create a DataFrame of test predictions for the meta-learner
X_meta_test = pd.DataFrame({
    'lgbm_oof': test_preds_lgbm,
    'catboost_oof': test_preds_catboost
})

# Get the final predictions from the meta-learner
final_predictions = meta_learner.predict_proba(X_meta_test)[:, 1]

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': range(750000, 750000 + len(final_predictions)), 'y': final_predictions})

# Save the submission file to a CSV
submission_df.to_csv('submission.csv', index=False)
print("\nFinal predictions saved to submission.csv")

