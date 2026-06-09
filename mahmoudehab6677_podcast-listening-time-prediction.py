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
import gc
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
SEED = 42
np.random.seed(SEED)


# Fill missing values
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

# Optional: Remove outliers in Number_of_Ads
train = train[train['Number_of_Ads'] < 10]

# Drop any remaining missing values
train = train.dropna()


# Label Encoding for categorical features
categorical_features = [
    'Podcast_Name', 'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment'
]

label_encoders = {col: LabelEncoder() for col in categorical_features}

for col in categorical_features:
    train[col] = label_encoders[col].fit_transform(train[col])
    test[col] = label_encoders[col].transform(test[col])


# Extract Episode Number
train['Episode_Num'] = train['Episode_Title'].str.extract('(\d+)').astype(int)
test['Episode_Num'] = test['Episode_Title'].str.extract('(\d+)').astype(int)

# Drop Episode_Title
train.drop(columns=['Episode_Title'], inplace=True)
test.drop(columns=['Episode_Title'], inplace=True)
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']


fold_preds = []
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):  # Start counting from 1
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Initialize dictionary to store evaluation results
    evals_result = {}
    
    model = lgb.LGBMRegressor(
        n_estimators=4500,
        learning_rate=0.020,
        num_leaves=1024,
        max_depth=-1,
        subsample=0.7,
        colsample_bytree=0.7,
        max_bin=1024,
        objective='regression',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    # Now fold_num is properly defined
    print(f"Fold {fold_num} best iteration: {model.best_iteration_}")
    print(f"Fold {fold_num} best score: {model.best_score_}")
    
    preds = model.predict(test[X.columns])
    fold_preds.append(preds)
    
    gc.collect()


submission = pd.DataFrame({
    'id': sample_submission['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv('submission.csv', index=False)

submission.head()

