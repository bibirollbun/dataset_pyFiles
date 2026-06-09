# Kaggle Playground Series S5E4: Podcast Listening Time Prediction
# Enhanced pipeline: Outlier handling, advanced features, Optuna tuning with verbose output

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
import optuna

warnings.filterwarnings("ignore")


# 1. Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# 2. Fill missing values and handle outliers
for df in [train_df, test_df]:
    df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)
    df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
    df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
    
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].clip(upper=325)
    df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=10)
    df['Has_Guest'] = (df['Guest_Popularity_percentage'] > 0).astype(int)
    df['Episode_Title'] = df['Episode_Title'].str.replace("Episode ", "", regex=False).astype(int)


# 3. Feature engineering
for df in [train_df, test_df]:
    df['ads_per_minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['popularity_diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['title_length'] = df['Episode_Title'].apply(lambda x: len(str(x)))
    df['host_guest_avg_popularity'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2
    df['length_per_popularity'] = df['Episode_Length_minutes'] / (df['Host_Popularity_percentage'] + 1)
    df['long_episode'] = (df['Episode_Length_minutes'] > 120).astype(int)


# 4. Categorical columns to be target encoded
categorical_cols = ['Podcast_Name', 'Episode_Sentiment', 'Publication_Day', 'Genre', 'Publication_Time']
target_col = 'Listening_Time_minutes'


# 5. Target encoding function
def target_encode(train, valid, test, target, cols):
    for col in tqdm(cols, desc="Target Encoding", leave=False):
        te = TargetEncoder(cols=[col])
        te.fit(train[col], train[target])
        train[f'TE_{col}'] = te.transform(train[col])[col].astype('float32')
        valid[f'TE_{col}'] = te.transform(valid[col])[col].astype('float32')
        test[f'TE_{col}'] = te.transform(test[col])[col].astype('float32')
    return train, valid, test


# 6. Optuna objective
FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'gpu',
        'n_estimators': 500,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'max_depth': trial.suggest_int("max_depth", 4, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'gamma': trial.suggest_float("gamma", 0, 5),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 1.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0)
    }

    oof_preds = np.zeros(len(train_df))
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        X_train = train_df.iloc[train_idx].copy()
        X_valid = train_df.iloc[valid_idx].copy()
        y_train = X_train[target_col]
        y_valid = X_valid[target_col]

        X_train, X_valid, _ = target_encode(X_train, X_valid, test_df.copy(), target_col, categorical_cols)

        drop_cols = categorical_cols + [target_col]
        te_features = [f'TE_{col}' for col in categorical_cols]
        used_features = [c for c in X_train.columns if c not in drop_cols and X_train[c].dtype in ['int64', 'float64']]
        final_features = used_features + te_features

        model = XGBRegressor(**params)
        model.fit(X_train[final_features], y_train,
                  eval_set=[(X_valid[final_features], y_valid)],
                  early_stopping_rounds=50, verbose=0)

        oof_preds[valid_idx] = model.predict(X_valid[final_features])

    return mean_squared_error(train_df[target_col], oof_preds, squared=False)


# 7. Run Optuna optimization
study = optuna.create_study(direction="minimize", study_name="xgb_opt")
study.optimize(objective, n_trials=25, show_progress_bar=True)


# 8. Retrain using best parameters
best_params = study.best_params
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'gpu',
    'n_estimators': 500
})

print("Training final model with best params:", best_params)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} ###")
    X_train = train_df.iloc[train_idx].copy()
    X_valid = train_df.iloc[valid_idx].copy()
    y_train = X_train[target_col]
    y_valid = X_valid[target_col]
    X_test = test_df.copy()

    X_train, X_valid, X_test = target_encode(X_train, X_valid, X_test, target_col, categorical_cols)

    drop_cols = categorical_cols + [target_col]
    te_features = [f'TE_{col}' for col in categorical_cols]
    used_features = [c for c in X_train.columns if c not in drop_cols and X_train[c].dtype in ['int64', 'float64']]
    final_features = used_features + te_features

    model = XGBRegressor(**best_params)
    model.fit(X_train[final_features], y_train,
              eval_set=[(X_valid[final_features], y_valid)],
              early_stopping_rounds=50, verbose=500)

    oof_preds[valid_idx] = model.predict(X_valid[final_features])
    test_preds += model.predict(X_test[final_features]) / FOLDS

    gc.collect()


# 9. Score and submission
rmse = np.sqrt(mean_squared_error(train_df[target_col], oof_preds))
print(f"Validation RMSE: {rmse:.5f}")

sub['Listening_Time_minutes'] = test_preds
sub.to_csv("submission.csv", index=False)
print("Submission saved.")

