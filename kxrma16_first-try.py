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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_train.head()


df_test.head()


df_train.info()


df_train.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = df_train.corr()
# corr_matrix = corr_matrix.fillna(0)
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt="0.5f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


print(corr_matrix.isna().sum().sum())


X_train = df_train.drop(columns=['rainfall'])
y_train = df_train['rainfall']
X_test = df_test


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)
lasso.fit(X_train_scaled, y_train)
selected_features = X_train.columns[lasso.coef_.flatten()!= 0]
print("Selected features: ", selected_features)


X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print("New X_train shape:" , X_train_selected.shape)
print("New X_test shape:" , X_test_selected.shape)


!pip install xgboost


import xgboost as xgb
from sklearn.metrics import roc_auc_score


xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric = 'logloss', use_label_encoder=False)

xgb_model.fit(X_train_selected, y_train)
y_train_pred = xgb_model.predict_proba(X_train_selected)[:,1]
train_auc = roc_auc_score(y_train,y_train_pred)
print("Training AUC Score:" , train_auc)


y_test_pred = xgb_model.predict_proba(X_test_selected)[:,1]


df_submission['rainfall'] = y_test_pred
df_submission.to_csv('submission_xgb.csv', index=False)
df_submission.head()

