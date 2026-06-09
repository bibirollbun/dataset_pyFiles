# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('../input/aviakompaniya/train_dataset.csv',index_col=['id'])
test = pd.read_csv('../input/aviakompaniya/test_dataset.csv',index_col=['id'])
sample_sub = pd.read_csv('../input/aviakompaniya/sample_submission.csv')


train.head()


def age_group(df):
    labels = ['Child', 'Young', 'Adult', 'Senior']
    return pd.qcut(df['Age'], q=4, labels=labels)

train['Age group'] = age_group(train)
test['Age group'] = age_group(test)


train.info()


test.info()


train.columns


from sklearn.model_selection import train_test_split

y_col = 'satisfaction'

X = train.drop(columns=[y_col])
y = train[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split

cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age group']

train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(X_test, label=y_test, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    cat_features=cat_features,
    verbose=100
)

cv_results = cv(
    Pool(X, label=y, cat_features=cat_features),
    model.get_params(),
    fold_count=5,
    plot=False
)


model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predict
preds = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)

# Output results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")


pred_test = model.predict(test)

sample_sub['satisfaction'] = pred_test
sample_sub.to_csv('submission.csv', index=False)




