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


RAR_trn = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
RAR_tst = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


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
print(f"Training dataset:\n{get_summary(RAR_trn).data_set()}\n{lent}\nTest dataset:\n{get_summary(RAR_tst).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(RAR_trn).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(RAR_tst).total_missing()}")


import seaborn as sns
import matplotlib.pyplot as plt


data = RAR_trn
def plot(data, column):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=data[column], y=data['accident_risk'])
    plt.title(f"Distribution by {column}")
    plt.ylabel('accident_risk')
    plt.xlabel(column)
    return plt.show()


plot(data=data, column='road_type')


plot(data=data, column='lighting')


plot(data=data, column='weather')


plot(data=data, column='road_signs_present')


plot(data=data, column='public_road')


plot(data=data, column='time_of_day')


plot(data=data, column='holiday')


plot(data=data, column='school_season')


sns.lineplot(x=data['curvature'], y=data['accident_risk'])


features_trn = RAR_trn.drop(['id', 'num_reported_accidents', 'accident_risk'], axis=1)
features_tst = RAR_tst.drop(['id', 'num_reported_accidents'], axis=1)


X = pd.get_dummies(features_trn)
y = RAR_trn['accident_risk']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


from sklearn.metrics import mean_squared_error
import xgboost as xgb


params = {
    "objective": "binary:logistic",
    "metric": "rmse",
    "num_leaves": 30,
    "learning_rate": 0.1,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0
}

xgb_model = xgb.XGBRegressor(**params, random_state=10, n_jobs=5)
xgb_model.fit(X_train, y_train)


preds_trn = xgb_model.predict(X_val)
RMSE = np.sqrt(mean_squared_error(y_val, preds_trn))
print(f"RMSE: {RMSE:.3F}")


X_test = pd.get_dummies(features_tst)
preds_tst = xgb_model.predict(X_test)


submission = pd.DataFrame({'id': RAR_tst['id'], 'accident_risk': preds_tst})
submission.head(4)


submission.to_csv("submission.csv", index=False)

