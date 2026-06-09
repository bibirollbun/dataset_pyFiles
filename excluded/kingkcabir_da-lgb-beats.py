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


beat_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
beats_1 = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
beats_2 = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

beats_1[:4]


lent = '*'*40
class get_summary:
    def __init__(self, x):
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame()
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
             #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
     
    #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        if not cols_with_missing.empty:
            return cols_with_missing.to_dict()
        else:
            return f"{'......No missing values detected......'}"
print(f"Training dataset:\n{get_summary(beats_1).data_set()}\n{lent}\nTest dataset:\n{get_summary(beats_2).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(beats_1).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(beats_2).total_missing()}")


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

for cols in beats_1.columns.drop('id'):
    plt.figure(figsize=(10, 5))
    sns.histplot(beats_1[cols], kde=True)
    plt.title(f'Distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')

    plt.show()


from sklearn.model_selection import train_test_split

X = beats_1.drop(['id', 'BeatsPerMinute'], axis=1)
y = beats_1.BeatsPerMinute

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


from sklearn.metrics import mean_squared_error
import lightgbm as lgb


params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "min_data_in_leaf": 20
}

lgb_model = lgb.LGBMRegressor(**params, random_state=12, n_jobs=-1)
lgb_model.fit(X_train, y_train)


preds_1 = lgb_model.predict(X_val)
RMSE = np.sqrt(mean_squared_error(y_val, preds_1))
print(f"RMSE: {RMSE:.3F}")


preds_2 = lgb_model.predict(beats_2.drop('id', axis=1))


submission = beat_sub
submission['BeatsPerMinute'] = preds_2
submission.to_csv('submission.csv', index=False)

