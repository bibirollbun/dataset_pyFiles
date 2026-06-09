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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')


train = train.drop(columns=['id', 'CustomerId', 'Surname'])
test_ids = test['id']
test = test.drop(columns=['id', 'CustomerId', 'Surname'])


le_gender = LabelEncoder()
le_geo = LabelEncoder()
for df in [train, test]:
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Geography'] = le_geo.fit_transform(df['Geography'])

X = train.drop('Exited', axis=1)
y = train['Exited']
test, _ = test.align(X, join='right', axis=1, fill_value=0)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_scaled = scaler.transform(test)


model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)


val_probs = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_probs)
print("Validation ROC AUC Score:", roc_auc)


val_preds = (val_probs > 0.5).astype(int)
print(confusion_matrix(y_val, val_preds))
print(classification_report(y_val, val_preds))


test_probs = model.predict_proba(test_scaled)[:, 1]
submission = pd.DataFrame({'id': test_ids, 'Exited': test_probs})
submission.to_csv('submission.csv', index=False)





