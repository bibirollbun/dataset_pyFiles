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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


train.info()


train.shape


test.shape


test.info()


train['Personality'].value_counts()


train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


num_cols = train.select_dtypes(include=["float64","int"]).columns
num_cols


cat_cols = train.select_dtypes(exclude=["float64","int"]).columns
cat_cols


train[num_cols] = train[num_cols].fillna(train[num_cols].mean())
train[cat_cols] = train[cat_cols].fillna(train[cat_cols].mode().iloc[0])

cols = ['Stage_fear', 'Drained_after_socializing']
test[cols] = test[cols].fillna(test[cols].mode().iloc[0])
test[num_cols] = test[num_cols].fillna(test[num_cols].mean())




train.isnull().sum()
test.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


for col in cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])




target_le = LabelEncoder()
train['Personality'] = target_le.fit_transform(train['Personality'])


X = train.drop(columns=['Personality'])
y = train['Personality']


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Initialize
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((test.shape[0], len(np.unique(y))))
accuracy_scores = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    accuracy_scores.append(acc)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    # Predict on test set (probabilities)
    test_preds += model.predict_proba(test) / skf.n_splits

# Final test predictions as most probable class
final_preds = np.argmax(test_preds, axis=1)


final_preds_labels = target_le.inverse_transform(final_preds)
submission['Personality'] = final_preds_labels
submission.to_csv("submission.csv", index=False)
submission.head()

