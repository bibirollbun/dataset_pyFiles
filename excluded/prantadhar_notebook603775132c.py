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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1️⃣ Load data
df_train = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\train.csv")
df_test  = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\test.csv")

# 2️⃣ Separate target and test ids
X_train = df_train.drop(columns=['loan_paid_back'])
y_train = df_train['loan_paid_back']

if 'id' in df_test.columns:
    test_ids = df_test['id']
    df_test = df_test.drop(columns=['id'])
else:
    test_ids = df_test.index

# 3️⃣ One-hot encode categorical columns
combined = pd.concat([X_train, df_test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, drop_first=False)

X_train_enc = combined.iloc[:len(X_train)].reset_index(drop=True)
df_test_enc  = combined.iloc[len(X_train):].reset_index(drop=True)

# 4️⃣ Train-validation split to check accuracy
X_tr, X_val, y_tr, y_val = train_test_split(X_train_enc, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=50, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# 5️⃣ Train on full train set and predict test
model.fit(X_train_enc, y_train)
pred = model.predict(df_test_enc)

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': pred})
submission.to_csv(r"D:\ml 15 days\kaggle compitition2\submission.csv", index=False)
print("Submission saved!")




