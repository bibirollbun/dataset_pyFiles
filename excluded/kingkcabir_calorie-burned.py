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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


submit = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
cal_tr = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
cal_ts = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

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
            return f"{'No missing values detected'}"
print(f"Training dataset:\n{get_summary(cal_tr).data_set()}\nTest dataset:\n{get_summary(cal_ts).data_set()}")
print(f"columns with missing values train\n{get_summary(cal_tr).total_missing()}\ncolumns with missing values test\n{get_summary(cal_ts).total_missing()}")


cal_tr.describe().T


#visualizing th distribution of each columns
sns.set_style('darkgrid')
plot_cols = cal_tr.columns.drop('id')
_rows = len(plot_cols)
plt.figure(figsize=(15, 3 * _rows))

for r, column in enumerate(plot_cols, 1):
    plt.subplot(_rows, 2, r)
    if cal_tr[column].nunique() <= 10:
        sns.countplot(x=column, data=cal_tr)
    else:
        sns.histplot(x=cal_tr[column], kde=True, bins=10, color='k')
        
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 4))
sns.lineplot(cal_tr, x='Age', y='Calories')
sns.lineplot(cal_tr, x='Age', y='Body_Temp')
plt.title('Calories burned by Age and body temprature')
plt.show()


plt.figure(figsize=(9, 4))
sns.lineplot(cal_tr, x='Age', y='Heart_Rate')
plt.title('Heart_rate by age during workout')
plt.show()


plt.figure(figsize=(9, 4))
sns.lineplot(cal_tr, x='Heart_Rate', y='Calories')
plt.title('Calories burned by Heart_rate')
plt.show()


plt.figure(figsize=(9, 4))
sns.lineplot(cal_tr, x='Weight', y='Calories')
plt.title('Calories burned by weight')
plt.show()


plt.figure(figsize=(9, 4))
sns.lineplot(cal_tr, x='Duration', y='Calories')
plt.title('Calories burned by Duration of workout')
plt.show()


def feature_eng(df):
    encoder = LabelEncoder()
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Sex'] = encoder.fit_transform(df['Sex'])
    
    return df.head(3)

feature_eng(cal_tr)


feature_eng(cal_ts)


X = cal_tr.drop(['id', 'Calories'], axis=1)

y = np.log1p(cal_tr['Calories'])


X_test = np.log1p(cal_ts.drop('id', axis=1))


kf = KFold(n_splits=5, shuffle=True, random_state=30)
xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))
xgb_scores = []

xgb_params = {
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': False,
    'random_state': 30,
    'early_stopping_rounds': 100
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
    xgb_oof[val_idx] = model.predict(X.iloc[val_idx])
    xgb_preds += model.predict(X_test) / kf.n_splits
    fold_score = np.sqrt(mean_squared_log_error(np.expm1(y.iloc[val_idx]), np.expm1(xgb_oof[val_idx])))
    print(f"Fold {fold+1} - XGBoost RMSLE: {fold_score:.5f}")
    xgb_scores.append(fold_score)
print(f"\n XGBoost Mean RMSLE: {np.mean(xgb_scores):.5f}")


final_preds = 0.49 * np.expm1(xgb_preds) + 0.51 
submission = submit
submission['Calories'] = np.clip(final_preds, 1, 314)
submission.head(2)


submission.to_csv('submission.csv', index=False)

