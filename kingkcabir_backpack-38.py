# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import numpy as np 
import pandas as pd 

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#LOAD DATASET
subs = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_pack = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_pack = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
ext_trn = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
#DATASETS SUMMARY
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
        return cols_with_missing.to_dict()
print(f"Training dataset:\n{get_summary(train_pack).data_set()}\nTest dataset:\n{get_summary(test_pack).data_set()}")
print(f"columns with missing values train\n{get_summary(train_pack).total_missing()}\ncolumns with missing values test\n{get_summary(test_pack).total_missing()}")


#joining the train and the extra train dataframes
merged_train = pd.concat([train_pack, ext_trn], ignore_index=True)
merged_train.shape


def sales_by_category(data, column):
    plt.figure(figsize=(10,6))
    sns.countplot(data=data, x=column)
    plt.title(f"UNIT SOLD BY {column.upper()}")
    plt.show()
sales_by_category(data=train_pack, column='Size')


sales_by_category(data=train_pack, column='Brand')


sales_by_category(data=train_pack, column='Material')



def fill_missing_and_encode(data):
    encoder = OrdinalEncoder()
    for column in data.columns:
        if data[column].dtype == 'float64'or data[column].dtype == 'int':
            data[column] = data[column].fillna(data[column].mean())
        elif data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
            data[[column]] = encoder.fit_transform(data[[column]])
    return data.head(2)

fill_missing_and_encode(merged_train)


fill_missing_and_encode(test_pack)


X = merged_train.drop(['id', 'Price'], axis=1)
y = merged_train['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=12, test_size=0.30)


X_test = test_pack.drop('id', axis=1)


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 1.0)
    }

    model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=12)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, pred))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


best_params = study.best_params
xgb_model = xgb.XGBRegressor(**best_params, random_state=12)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_val)
score = np.sqrt(mean_squared_error(y_val, xgb_pred))
print(f"RMSE: {score:.4f}")


predict = xgb_model.predict(X_test)
submission = pd.DataFrame({'id': test_pack['id'], 'Price': predict})
print(submission.head(4))


submission.to_csv("submission.csv", index=False)

