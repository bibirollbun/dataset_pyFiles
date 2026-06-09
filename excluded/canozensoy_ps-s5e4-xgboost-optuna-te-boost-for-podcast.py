# Optimized XGBoost with Target Encoding and Optuna

import numpy as np
import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from itertools import combinations
import optuna
from sklearn.metrics import mean_squared_error

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.copy_on_write = True


# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_original = pd.read_csv("/kaggle/input/original-podcast-dataset/podcast_dataset.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

df = pd.concat([df_train, df_original, df_test], axis=0, ignore_index=True)
df.drop(columns=['id'], inplace=True)
df = df.drop_duplicates()

df['Episode_Length_minutes'] = np.clip(df['Episode_Length_minutes'], 0, 120)
df['Host_Popularity_percentage'] = np.clip(df['Host_Popularity_percentage'], 20, 100)
df['Guest_Popularity_percentage'] = np.clip(df['Guest_Popularity_percentage'], 0, 100)
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Publication_Day'] = df['Publication_Day'].map(day_mapping)

time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True).astype(int)

le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col]) + 1

for col in ['Episode_Length_minutes']:
    df[col + '_sqrt'] = np.sqrt(df[col])
    df[col + '_squared'] = df[col] ** 2

for col in ['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title', 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads']:
    df[col + '_EP'] = df.groupby(col)['Episode_Length_minutes'].transform('mean')

def process_combinations_fast(df, columns_to_encode, pair_size, max_batch_size=2000):
    str_df = df[columns_to_encode].astype(str)
    le = LabelEncoder()
    if isinstance(pair_size, int):
        pair_size = [pair_size]
    for r in pair_size:
        combos_iter = combinations(columns_to_encode, r)
        for cols in combos_iter:
            new_name = '+'.join(cols)
            result = str_df[cols[0]].copy()
            for col in cols[1:]:
                result += str_df[col]
            df[new_name] = le.fit_transform(result) + 1
    return df

df = process_combinations_fast(
    df,
    ['Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage'],
    [2, 3, 5, 7]
)

df = df.astype('float32')


# Split Train/Test
df_train = df.iloc[:-len(df_test)]
df_test = df.iloc[-len(df_test):].reset_index(drop=True)
df_train = df_train[df_train['Listening_Time_minutes'].notnull()]
target = df_train.pop('Listening_Time_minutes')
df_test.drop(columns=['Listening_Time_minutes'], inplace=True)


# Optuna Hyperparameter Tuning
seed1 = 42
cv = KFold(n_splits=3, shuffle=True, random_state=seed1)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'device': 'cuda',
    'seed': seed1
}

def objective(trial):
    tuned_params = {
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
    }
    full_params = params.copy()
    full_params.update(tuned_params)
    fold_scores = []
    for train_idx, valid_idx in cv.split(df_train):
        X_train, y_train = df_train.iloc[train_idx], target.iloc[train_idx]
        X_valid, y_valid = df_train.iloc[valid_idx], target.iloc[valid_idx]

        encoder = TargetEncoder(n_folds=5, seed=seed1, stat="mean")
        for col in df_train.columns[:20]:
            X_train[col+'_te'] = encoder.fit_transform(X_train[[col]], y_train)
            X_valid[col+'_te'] = encoder.transform(X_valid[[col]])

        dtrain = xgb.DMatrix(X_train.filter(like='_te'), label=y_train)
        dvalid = xgb.DMatrix(X_valid.filter(like='_te'), label=y_valid)

        model = xgb.train(
            full_params,
            dtrain,
            num_boost_round=10000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        preds = model.predict(dvalid)
        fold_scores.append(mean_squared_error(y_valid, preds, squared=False))
    return np.mean(fold_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("âœ… Best RMSE:", study.best_value)
print("ðŸŽ¯ Best Parameters:", study.best_params)

params.update(study.best_params)


# Final model training
final_preds = np.zeros(df_test.shape[0])
kf = KFold(n_splits=7, shuffle=True, random_state=seed1)

for idx_train, idx_valid in kf.split(df_train):
    X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
    X_test = df_test[X_train.columns].copy()

    encoder = TargetEncoder(n_folds=5, seed=seed1, stat="mean")
    for col in df_train.columns[:20]:
        X_train[col+'_te'] = encoder.fit_transform(X_train[[col]], y_train)
        X_valid[col+'_te'] = encoder.transform(X_valid[[col]])
        X_test[col+'_te'] = encoder.transform(X_test[[col]])

    for col in df_train.columns[20:]:
        X_train[col] = encoder.fit_transform(X_train[[col]], y_train)
        X_valid[col] = encoder.transform(X_valid[[col]])
        X_test[col] = encoder.transform(X_test[[col]])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=30,
        verbose_eval=500
    )

    final_preds += model.predict(dtest)

final_preds /= 7


# Save submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission['Listening_Time_minutes'] = final_preds
submission.to_csv("submission.csv", index=False)
print("ðŸš€ Submission saved.")

