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


# Setup and Imports
!pip install xgboost lightgbm -q

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


# Load Data
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
except FileNotFoundError:
    print("Dataset not found. Please add the competition data to your notebook.")
    # Create dummy dataframes to prevent errors
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

# Store test IDs for submission
if not test_df.empty:
    test_ids = test_df['id']
    # Drop id columns
    train_df = train_df.drop('id', axis=1)
    test_df = test_df.drop('id', axis=1)


# Feature Engineering
def create_features(df):
    df['Social_Interaction_Score'] = df['Social_event_attendance'] * df['Friends_circle_size']
    df['Alone_Comfort'] = df['Time_spent_Alone'] / (df['Friends_circle_size'] + 1) # Avoid division by zero
    df['Social_Media_to_Friends_Ratio'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    return df

if not train_df.empty:
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    print("✅ New features created successfully.")


# Preprocessing
if not train_df.empty:
    categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Personality' in categorical_features:
        categorical_features.remove('Personality')

    for col in categorical_features:
        le = LabelEncoder()
        combined_data = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined_data)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    # Encode the target variable separately
    target_le = LabelEncoder()
    train_df['Personality'] = target_le.fit_transform(train_df['Personality'])

    print("✅ Preprocessing complete.")


# Cross-Validation Training
if not train_df.empty:
    # Define features (X) and target (y)
    X = train_df.drop('Personality', axis=1)
    y = train_df['Personality']
    X_test = test_df

    # Cross-validation setup
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Placeholders for predictions
    oof_preds_lgb = np.zeros((len(train_df), 2))
    test_preds_lgb = np.zeros((len(test_df), 2))
    oof_preds_xgb = np.zeros((len(train_df), 2))
    test_preds_xgb = np.zeros((len(test_df), 2))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # --- LightGBM ---
        lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)
        test_preds_lgb += lgb_model.predict_proba(X_test) / N_SPLITS

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_estimators=500, use_label_encoder=False)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
        test_preds_xgb += xgb_model.predict_proba(X_test) / N_SPLITS

    # Evaluate OOF predictions
    acc_lgb = accuracy_score(y, np.argmax(oof_preds_lgb, axis=1))
    acc_xgb = accuracy_score(y, np.argmax(oof_preds_xgb, axis=1))
    print(f"\nLGBM OOF Accuracy: {acc_lgb:.5f}")
    print(f"XGB OOF Accuracy: {acc_xgb:.5f}")


# Ensembling and Submission
if not train_df.empty:
    # Simple averaging ensemble
    ensemble_preds_proba = (test_preds_lgb * 0.5) + (test_preds_xgb * 0.5)
    ensemble_preds_encoded = np.argmax(ensemble_preds_proba, axis=1)

    # Inverse transform to get original string labels
    final_predictions = target_le.inverse_transform(ensemble_preds_encoded)

    # Create the submission file
    submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
    submission_df.to_csv('submission.csv', index=False)

    print("\n✅ Submission file created successfully using an ensemble of LGBM and XGB!")
    print(submission_df.head())

