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


# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
pd.options.mode.copy_on_write = True

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


# Efficient Combination Processing
def process_combinations_fast(df, columns_to_encode, pair_size, max_batch_size=2000):
    str_df = df[columns_to_encode].astype(str)
    le = LabelEncoder()
    
    if isinstance(pair_size, int):
        pair_size = [pair_size]

    total_new_cols = 0
    for r in pair_size:
        print(f"\nProcessing {r}-combinations...")
        combos_iter = combinations(columns_to_encode, r)
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations to process: {n_combinations}")

        batch_cols = []
        batch_names = []

        with tqdm(total=n_combinations, desc=f"{r}-combinations") as pbar:
            while True:
                batch_cols.clear()
                batch_names.clear()

                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break

                if not batch_cols:
                    break

                for cols, new_name in zip(batch_cols, batch_names):
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += str_df[col]
                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)

                total_new_cols += len(batch_cols)

        print(f"Completed {r}-combinations. Total columns now: {len(df.columns)}")
    return df


# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_original = pd.read_csv("/kaggle/input/original-podcast-dataset/podcast_dataset.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



df = pd.concat([df_train, df_original, df_test], axis=0, ignore_index=True)
df.drop(columns=['id'], inplace=True)
df = df.drop_duplicates()

# Outlier removal
df['Episode_Length_minutes'] = np.clip(df['Episode_Length_minutes'], 0, 120)
df['Host_Popularity_percentage'] = np.clip(df['Host_Popularity_percentage'], 20, 100)
df['Guest_Popularity_percentage'] = np.clip(df['Guest_Popularity_percentage'], 0, 100)
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

# Encode categorical features
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


# Feature Engineering
for col in ['Episode_Length_minutes']:
    df[col + '_sqrt'] = np.sqrt(df[col])
    df[col + '_squared'] = df[col] ** 2

for col in tqdm(['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title',
                 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads']):
    df[col + '_EP'] = df.groupby(col)['Episode_Length_minutes'].transform('mean')

# Process Combinations
df = process_combinations_fast(df, 
    ['Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage', 
     'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage'], 
    [2, 3, 5, 7], 
    max_batch_size=1000
)

df = df.astype('float32')

# Split Train/Test
df_train = df.iloc[:-len(df_test)]
df_test = df.iloc[-len(df_test):].reset_index(drop=True)

df_train = df_train[df_train['Listening_Time_minutes'].notnull()]
target = df_train.pop('Listening_Time_minutes')
df_test.drop(columns=['Listening_Time_minutes'], inplace=True)

print(df_train.shape, df_test.shape)


# XGBoost Setup
seed1 = 42
cv = KFold(7, random_state=seed1, shuffle=True)
pred_test = np.zeros((250000,))

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': seed1,
    'max_depth': 18,
    'learning_rate': 0.035,
    'min_child_weight': 60,
    'reg_alpha': 4,
    'reg_lambda': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'colsample_bynode': 0.5,
    'device': "cuda"
}

def lr_decay(epoch):
    if epoch < 100:
        return 0.035
    else:
        return 0.015

callbacks = xgb.callback.LearningRateScheduler(lr_decay)

# Save learning curves
train_rmses, valid_rmses = [], []

for idx_train, idx_valid in cv.split(df_train):
    X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
    X_test = df_test[X_train.columns].copy()

    features = df_train.columns
    encoder1 = TargetEncoder(n_folds=5, seed=seed1, stat="mean")

    for col in tqdm(features[:20]):
        X_train[col+'_te1'] = encoder1.fit_transform(X_train[[col]], y_train)
        X_valid[col+'_te1'] = encoder1.transform(X_valid[[col]])
        X_test[col+'_te1'] = encoder1.transform(X_test[[col]])

    for col in tqdm(features[20:]):
        X_train[col] = encoder1.fit_transform(X_train[[col]], y_train)
        X_valid[col] = encoder1.transform(X_valid[[col]])
        X_test[col] = encoder1.transform(X_test[[col]])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)
    
 
    # Before model training
    evals_result = {}
    
    model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000000,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=30,
    verbose_eval=500,
    callbacks=[callbacks],
    evals_result=evals_result  # <<< Important
)


    train_rmse_list = evals_result['train']['rmse']
    valid_rmse_list = evals_result['validation']['rmse']

    best_iter = model.best_iteration


    train_rmses.append(train_rmse_list[best_iter])
    valid_rmses.append(valid_rmse_list[best_iter])




    pred_test += np.maximum(0, np.minimum(120, model.predict(dtest)))

    print("----------------------------------------------------------------")

pred_test /= 7


# Save Submission
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_sub['Listening_Time_minutes'] = pred_test
df_sub.to_csv('submission.csv', index=False)



