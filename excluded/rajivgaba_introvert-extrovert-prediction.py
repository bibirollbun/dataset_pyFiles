# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
# for dirname, _, filenames in os.walk(playground_series_s5e7_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


playground_series_s5e7_path = "/kaggle/input/playground-series-s5e7/"


train_file = os.path.join(playground_series_s5e7_path, 'train.csv')
test_file = os.path.join(playground_series_s5e7_path, 'test.csv')


train_data = pd.read_csv(train_file)


train_data.info()


train_data.head()


train_data.isnull().mean()


num_cols = train_data.select_dtypes(include=['int64','float64']).columns.to_list()
cat_cols = train_data.select_dtypes(include=['object']).columns.to_list()
num_cols.remove('id')
cat_cols.remove('Personality')



num_cols, cat_cols


import matplotlib.pyplot as plt
import seaborn as sns


import warnings

warnings.filterwarnings('ignore')


plt.figure()
sns.set_style('whitegrid')
sns.set_context('notebook')
for col in num_cols:
  sns.histplot(train_data[col], kde=True, bins=5, alpha=0.5, binwidth=0.6, color='pink')
  plt.show()


train_data.isnull().sum()


# Impute missing values for numeric columns

from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
num_imputer.fit(train_data[num_cols])
train_data[num_cols] = num_imputer.transform(train_data[num_cols])


# impute missing values for categorical columns

imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(train_data[cat_cols])
train_data[cat_cols] = imputer.transform(train_data[cat_cols])


# Label encoding for categorical columns

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in cat_cols:
  train_data[col] = label_encoder.fit_transform(train_data[col])


train_data['Personality'] = train_data['Personality'].map({'Extrovert': 0, 'Introvert': 1})


train_data['Personality'].value_counts()


# Split the data in train and test sets

from sklearn.model_selection import train_test_split
X = train_data.drop(['id', 'Personality'], axis=1)
y = train_data['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logistic_regression_model = LogisticRegression()
svc_model = SVC()

import xgboost as xgb
xgb_model = xgb.XGBClassifier()


from catboost import CatBoostClassifier
cb_model = CatBoostClassifier()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logistic_regression_scores = []
svc_scores = []
xgb_scores = []
cb_scores = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    logistic_regression_model.fit(X_train_fold, y_train_fold)
    lr_pred = logistic_regression_model.predict(X_val_fold)
    logistic_regression_scores.append(accuracy_score(y_val_fold, lr_pred))

    '''
    svc_model.fit(X_train_fold, y_train_fold)
    svc_pred = svc_model.predict(X_val_fold)
    svc_scores.append(accuracy_score(y_val_fold, svc_pred))

    xgb_model.fit(X_train_fold, y_train_fold)
    xgb_pred = xgb_model.predict(X_val_fold)
    xgb_scores.append(accuracy_score(y_val_fold, xgb_pred))
    
    cb_model.fit(X_train_fold, y_train_fold)
    cb_pred = cb_model.predict(X_val_fold)
    cb_scores.append(accuracy_score(y_val_fold, cb_pred))
    '''


print(sorted(cb_scores))
print(sorted(svc_scores))
print(sorted(xgb_scores))
print(sorted(logistic_regression_scores))


avg_logistic_regression_accuracy = np.mean(logistic_regression_scores)
avg_svc_accuracy = np.mean(svc_scores)
avg_xgb_accuracy = np.mean(xgb_scores)
avg_cb_accuracy = np.mean(cb_scores)

print(f"Average accuracy for Logistic Regression: {avg_logistic_regression_accuracy:.4f}")
print(f"Average accuracy for SVC: {avg_svc_accuracy:.4f}")
print(f"Average accuracy for XGB: {avg_xgb_accuracy: .4f}")
print(f"Average accuracy for Catboost: {avg_cb_accuracy: .4f}")


# Load test data and make predictions

test_data = pd.read_csv(test_file)
test_data.head()



pp_test_data = test_data.copy()
pp_test_data.drop('id', axis=1, inplace=True)

pp_test_data[cat_cols] = imputer.transform(pp_test_data[cat_cols])

pp_test_data[num_cols] = num_imputer.transform(pp_test_data[num_cols])

for col in cat_cols:
  pp_test_data[col] = label_encoder.transform(pp_test_data[col])


# predictions = xgb_model.predict(pp_test_data)
# predictions = cb_model.predict(pp_test_data)
predictions = logistic_regression_model.predict(pp_test_data)

mapping = {0: 'Extrovert', 1: 'Introvert'}

predictions = np.vectorize(mapping.get)(predictions)

predictions


results_df = pd.DataFrame({
    'id' : test_data['id'],
    'Personality': predictions
})


results_df.head()


results_df.to_csv('submission.csv', index=False)

