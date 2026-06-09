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


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
train.head()


test.isna().sum()


train.info()


train['employment_status'].value_counts()


train.describe(include='all')


train.info()


num_cols = train.select_dtypes(include=['int64', "float64"]).columns.tolist()
print("numeric col: ",num_cols)


cat_cols = train.select_dtypes(include=['object']).columns.tolist()
print("cat col: ",cat_cols)


from sklearn.preprocessing import LabelEncoder
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    le_dict[col] = le

train.head()
print(le_dict)


for col, le in le_dict.items():
    test[col] = le.transform(test[col])

test.head()


train.head()


X = train.drop(columns=['id', 'loan_paid_back'])
y = train["loan_paid_back"]


# Split into train/test
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


import xgboost as xgb
from sklearn.metrics import roc_auc_score



model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)


model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=50)


# Predict probabilities for ROC-AUC
y_pred_proba = model.predict_proba(X_val)[:,1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_val, y_pred_proba)
print("Validation ROC-AUC:", auc)


test_data = test.drop(columns = ['id'])


predictions = model.predict(test_data)


submission = pd.DataFrame({
    "id": test['id'],
    "loan_paid_back": predictions
})


submission.to_csv("submission.csv", index=False)


sub = pd.read_csv("/kaggle/working/submission.csv")
sub.head()

