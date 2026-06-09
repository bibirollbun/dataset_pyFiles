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


df_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


df_train.head()


df_train.describe()


df_train = df_train.drop(columns=['ID_code'])


corr_matrix = df_train.corr() 
corr_matrix['target']


mean_corr = corr_matrix['target'].mean()
print(f'Средняя корреляция: {mean_corr=}')

good_features = corr_matrix['target'][abs(corr_matrix['target']) > mean_corr]
good_features


cols_to_index = good_features.index.values[1:]
cols_to_index


train_features = df_train[cols_to_index]
train_target = df_train['target']


from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(train_features)
X = scaler.transform(train_features)
y = train_target.values


positives = y[y == 1].shape[0]
negatives = y[y == 0].shape[0]

print(f'Положительный таргет: {positives}')
print(f'Отрицательный таргет: {negatives}')


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1) 

model = HistGradientBoostingClassifier(random_state=7, class_weight='balanced')
sigmoid = CalibratedClassifierCV(model, cv=5, method='sigmoid')
sigmoid.fit(X_train, y_train)
val_scores = sigmoid.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, val_scores)
print(f'ROC-AUC={roc_auc}')


from imblearn.over_sampling import SMOTE

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1) 
smote = SMOTE(random_state=7)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model_smote = HistGradientBoostingClassifier(random_state=7)
sigmoid_smote = CalibratedClassifierCV(model_smote, cv=5, method='sigmoid')
sigmoid_smote.fit(X_train_res, y_train_res)
val_scores = sigmoid_smote.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, val_scores)
print(f'ROC-AUC={roc_auc}')


test_data = df_test[cols_to_index]


X_test = scaler.transform(test_data)
y_test = sigmoid.predict_proba(X_test)[:, 1]


submission_df = pd.DataFrame()
submission_df['target'] = y_test
submission_df['ID_code'] = [f'test_{i}' for i in range(y_test.shape[0])]

submission_df.head()


submission_df.to_csv('submission.csv', index=False, float_format='%.6f')

