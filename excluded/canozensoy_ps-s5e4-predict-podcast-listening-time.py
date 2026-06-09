# Kaggle Playground Series S5E4: Podcast Listening Time Prediction
# Simplified pipeline: Lightweight XGBoost model, target encoding, fast inference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from tqdm import tqdm
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

warnings.filterwarnings("ignore")


# 1. Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# 2. Fill missing values
for df in [train_df, test_df]:
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)


# 3. Feature engineering
for df in [train_df, test_df]:
    df['ads_per_minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['popularity_diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['title_length'] = df['Episode_Title'].apply(lambda x: len(str(x)))


# 4. Categorical columns to be target encoded
categorical_cols = ['Podcast_Name', 'Episode_Sentiment', 'Publication_Day']
rmv = ['Listening_Time_minutes']


# 5. Target encoding function
def target_encode(train, valid, test, target, cols):
    for col in tqdm(cols, desc="Target Encoding", leave=False):
        te = TargetEncoder(cols=[col])
        te.fit(train[col], train[target])
        train[f'TE_{col}'] = te.transform(train[col])[col].astype('float32')
        valid[f'TE_{col}'] = te.transform(valid[col])[col].astype('float32')
        test[f'TE_{col}'] = te.transform(test[col])[col].astype('float32')
    return train, valid, test


# 6. Lightweight XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'tree_method': 'hist',  # fast training
    'device': 'gpu',
    'verbosity': 1
}


# 7. K-Fold training
FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} ###")
    X_train = train_df.iloc[train_idx].copy()
    X_valid = train_df.iloc[valid_idx].copy()
    y_train = X_train[rmv[0]]
    y_valid = X_valid[rmv[0]]
    X_test = test_df.copy()

    X_train, X_valid, X_test = target_encode(X_train, X_valid, X_test, rmv[0], categorical_cols)

    drop_cols = rmv + categorical_cols
    te_features = [f'TE_{col}' for col in categorical_cols]
    used_features = [c for c in X_train.columns if c not in drop_cols and X_train[c].dtype in ['int64', 'float64']]
    final_features = used_features + te_features

    model = XGBRegressor(**xgb_params)
    model.fit(X_train[final_features], y_train,
              eval_set=[(X_valid[final_features], y_valid)],
              early_stopping_rounds=50, verbose=False)

    oof_preds[valid_idx] = model.predict(X_valid[final_features])
    test_preds += model.predict(X_test[final_features]) / FOLDS

    gc.collect()


# 8. Score and submission
rmse = np.sqrt(mean_squared_error(train_df[rmv[0]], oof_preds))
print(f"Validation RMSE: {rmse:.5f}")

sub['Listening_Time_minutes'] = test_preds
sub.to_csv("submission.csv", index=False)
print("Submission saved.")

