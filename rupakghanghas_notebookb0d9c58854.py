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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample= pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


# train.info()
# train.describe()


numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

categorical_cols = df.select_dtypes(include=[object]).columns.tolist()


numerical_cols.remove('loan_status')


scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
test[numerical_cols] = scaler.fit_transform(test[numerical_cols])
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
    test[col] = label_encoder.fit_transform(test[col])


y = df['loan_status']
if not np.array_equal(np.unique(y), [0, 1]):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

X = df.drop('loan_status', axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

print("XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
print("XGBoost AUC-ROC:", roc_auc_score(y_val, y_pred_proba_xgb))


cat_model = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=10, random_state=42, verbose=0)
cat_model.fit(X_train, y_train)

y_pred_cat = cat_model.predict(X_val)
y_pred_proba_cat = cat_model.predict_proba(X_val)[:, 1]

print("CatBoost Accuracy:", accuracy_score(y_val, y_pred_cat))
print("CatBoost AUC-ROC:", roc_auc_score(y_val, y_pred_proba_cat))


test['id']


test_predictions = xgb_model.predict_proba(test)


test_predictions


submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')
submission["loan_status"] = test_predictions

submission.to_csv('submission.csv', index=False)
submission.head()

