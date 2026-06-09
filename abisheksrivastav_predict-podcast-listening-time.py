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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Drop ID column
train_df.drop(columns=['id'], inplace=True)
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True)

# Handle missing values
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)

train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)

train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)
test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].mode()[0], inplace=True)

# Combine for consistent encoding
combined = pd.concat([train_df.drop(columns=['Listening_Time_minutes']), test_df], axis=0).reset_index(drop=True)

# Label Encoding for categorical features
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# Split back
train_encoded = combined[:len(train_df)]
test_encoded = combined[len(train_df):]

# Add target column back to train
train_encoded['Listening_Time_minutes'] = train_df['Listening_Time_minutes']

# Prepare data for modeling
X = train_encoded.drop(columns=['Listening_Time_minutes'])
y = train_encoded['Listening_Time_minutes']
X_test = test_encoded

# Stratified binning for continuous target
bins = np.floor(pd.qcut(y, q=10, labels=False, duplicates='drop'))

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
rmse_scores = []

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'random_state': 42
}

# Train
for fold, (train_idx, val_idx) in enumerate(kf.split(X, bins)):
    print(f"\nğŸ”� Fold {fold + 1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

    rmse = mean_squared_error(y_val, val_preds, squared=False)
    rmse_scores.append(rmse)
    print(f"âœ… Fold {fold + 1} RMSE: {rmse:.4f}")

# Overall RMSE
overall_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"\nğŸ�¯ Overall OOF RMSE: {overall_rmse:.4f}")

# Create submission
submission['Listening_Time_minutes'] = test_preds
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved!")


