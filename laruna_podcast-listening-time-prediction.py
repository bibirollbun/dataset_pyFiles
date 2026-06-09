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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import psutil
from tqdm import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    return f"Memory Usage: {mem:.2f} MB"


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    train_df[col].fillna(train_df[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)

train_df.dropna(inplace=True)

train_df = train_df[train_df['Number_of_Ads'] < 10]

categorical_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')
    label_encoders[col] = le

train_df['Episode_Num'] = train_df['Episode_Title'].str.extract('Episode (\d+)').astype('category')
test_df['Episode_Num'] = test_df['Episode_Title'].str.extract('Episode (\d+)').astype('category')
train_df.drop(columns=['Episode_Title'], inplace=True)
test_df.drop(columns=['Episode_Title'], inplace=True)


from tqdm import tqdm
from itertools import combinations

columns_to_encode = [
    'Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage',
    'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day',
    'Publication_Time', 'Genre', 'Guest_Popularity_percentage'
]

# Just use pairwise to control memory
pair_size = 2
combinations_list = list(combinations(columns_to_encode, pair_size))
batch_size = 10  # smaller batch

for i in range(0, len(combinations_list), batch_size):
    batch = combinations_list[i:i+batch_size]
    for cols in tqdm(batch, desc="Encoding combinations"):
        new_col_name = '_'.join(cols)
        
        # Combine columns as strings
        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        
        # Skip if too many categories
        if train_df[new_col_name].nunique() < 100:
            train_df[new_col_name] = train_df[new_col_name].astype('category')
            test_df[new_col_name] = test_df[new_col_name].astype('category')
        else:
            train_df.drop(columns=[new_col_name], inplace=True)
            test_df.drop(columns=[new_col_name], inplace=True)

    gc.collect()
    print(f"Memory usage: {train_df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    print(f"Total columns now: {train_df.shape[1]}")



X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']

cv = KFold(5, random_state=42, shuffle=True)
y_pred = np.zeros(len(test_df))

encoded_columns = X.columns

for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train = X.iloc[idx_train].copy(), y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid].copy(), y.iloc[idx_valid]
    X_test = test_df[X.columns].copy()

    encoder = TargetEncoder()
    X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse',
        verbosity=-1,
        max_bin=1024,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(100)
        ]
    )

    y_pred += model.predict(X_test)
pred_lgbm = y_pred / 5


submission_lgbm = pd.DataFrame({
    'id': sample_submission['id'],
    'Listening_Time_minutes': pred_lgbm
})

submission_lgbm.to_csv('submission.csv', index=False)
submission_lgbm.head()

