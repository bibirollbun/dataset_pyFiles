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


payback_1 = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
payback_2 = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


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
print(f"Training dataset:\n{get_summary(payback_1).data_set()}\n{lent}\nTest dataset:\n{get_summary(payback_2).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(payback_1).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(payback_2).total_missing()}")


import seaborn as sns
import matplotlib.pyplot as plt


vals = payback_1.drop('id', axis=1).describe().T
vals = vals.drop('count', axis=1)
vals.plot(kind='bar', stacked=False, figsize=(10, 4))
plt.title('Descriptive Statistics for payback_1 Features')
plt.xlabel('Statistic')
plt.ylabel('Value')
plt.legend(title='Features', bbox_to_anchor=(1.10, 1), loc='upper left')
plt.tight_layout()
plt.show()

print('_'* 86)
payback_1.drop('id', axis=1).describe().T


def plot_chart(data, column1, column2):
    plt.figure(figsize=(4, 4))
    sns.barplot(data, x=column1, y=column2)
    plt.show


plot_chart(data=payback_1, column1='gender', column2='annual_income')
plot_chart(data=payback_1, column1='gender', column2='credit_score')
plot_chart(data=payback_1, column1='gender', column2='loan_amount')


plot_chart(data=payback_1, column1='marital_status', column2='annual_income')
plot_chart(data=payback_1, column1='marital_status', column2='credit_score')
plot_chart(data=payback_1, column1='marital_status', column2='loan_amount')


from sklearn.preprocessing import LabelEncoder


'''This function add more insightful columns to the dataframe'''
def feature_eng(payback_1):
    payback_1['monthly_income'] = payback_1['annual_income'] / 12
    payback_1['estimated_monthly_debt_payment'] = payback_1['monthly_income'] * payback_1['debt_to_income_ratio']
    payback_1['loan_to_income_ratio'] = payback_1['loan_amount'] / payback_1['annual_income']
    payback_1['int_to_income_ratio']= payback_1['loan_amount'] * payback_1['interest_rate'] / payback_1['annual_income']

    return payback_1.head(3)

feature_eng(payback_1)


#calling the feature_eng function on payback_2
feature_eng(payback_2)


'''This function encodes the categorical columns'''
enc = LabelEncoder()
def encod(payback_1):
    for column in payback_1.columns:
        if payback_1[column].dtype == 'object':
            payback_1[column] = enc.fit_transform(payback_1[column])
    return payback_1.head(2)


encod(payback_1)


encod(payback_2)


from sklearn.model_selection import train_test_split

X = payback_1.drop(['id', 'loan_paid_back'], axis=1)
X_test = payback_2.drop('id', axis=1)


y = payback_1.loan_paid_back
y


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=31)


from sklearn.metrics import roc_auc_score
import xgboost as xgb


params = {'n_estimators': 2000, 
          'max_depth': 9, 
          'learning_rate': 0.1, 
          'subsample': 0.7, 
          'colsample_bytree': 0.8, 
          'min_child_weight': 7, 
          'gamma': 0.5, 
          'reg_alpha': 0.9, 
          'reg_lambda': 1.6,
          'random_state': 31,
          'tree_method': 'hist'
         }
_model = xgb.XGBRegressor(**params)
_model.fit(X_train, y_train)

y_pred = _model.predict(X_val)
auc_score = roc_auc_score(y_val, y_pred)
print(f"ROC_CURVE: {auc_score:.4f}")


prediction = _model.predict(X_test)
submission = pd.DataFrame({'id': payback_2['id'],
                           'loan_paid_back': prediction})


submission.to_csv("submission.csv", index=False)

