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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')


value_counts = train['Status'].value_counts()
to_drop = value_counts[value_counts < 2].index
train = train[~train['Status'].isin(to_drop)]


X = train.drop(columns=['id', 'Status'])
y = train['Status']
X_test = test.drop(columns=['id'])


cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = X[col].fillna('unknown')
    X_test[col] = X_test[col].fillna('unknown')

    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


num_cols = X.select_dtypes(include=np.number).columns
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())


target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=150, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)


ensemble.fit(X_train_scaled, y_train)


y_val_proba = ensemble.predict_proba(X_val_scaled)
logloss = log_loss(y_val, y_val_proba)
print(f" Multi-class LogLoss (Validation): {logloss:.5f}")


y_test_proba = ensemble.predict_proba(X_test_scaled)


class_labels = target_le.inverse_transform(ensemble.classes_)
proba_df = pd.DataFrame(y_test_proba, columns=[f'Status_{label}' for label in class_labels])
submission = pd.DataFrame({'id': test['id']})
submission = pd.concat([submission, proba_df], axis=1)
submission.to_csv("submission.csv", index=False)
print("ğŸ“� submission.csv tayyor!")





