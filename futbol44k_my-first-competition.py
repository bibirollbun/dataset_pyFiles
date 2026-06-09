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


import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
train.shape


test = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')
test


train.head(5)


for col in train.columns:
    fig, ax = plt.subplots(figsize=(9, 3))
    fig = sns.histplot(data=train, x=col, hue='smoking', bins=100)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['smoking']),
    train['smoking'], test_size=0.2, random_state=2)


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test


clf = LogisticRegression(random_state=55).fit(X_train, y_train)


tr_pred = clf.predict_proba(X_train)
ts_pred = clf.predict_proba(X_test)


roc_auc_score(y_train, tr_pred[:, 1])


roc_auc_score(y_test, ts_pred[:, 1])


from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier


best_k = -1
best_score = 0.5
scores = []
ks = []

for k in range(80, 121, 5):
    alg = KNeighborsClassifier(n_neighbors=k)
    alg.fit(X_train, y_train)
    ts_pred = alg.predict_proba(X_test)

    if roc_auc_score(y_test, ts_pred[:, 1]) > best_score:
        best_k = k
        best_score = roc_auc_score(y_test, ts_pred[:, 1])

    scores.append(roc_auc_score(y_test, ts_pred[:, 1]))
    ks.append(k)
    print(ks[-1], scores[-1])


plt.plot(ks, scores)


print("Bets score: ", best_score)
print("Bets k: ", best_k)


boost = CatBoostClassifier(loss_function='Logloss', iterations=2000,
                           learning_rate=0.2,
                           eval_metric='AUC', random_state = 44)


boost.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model = True,
        plot = True)


roc_auc_score(y_test, boost.predict_proba(X_test)[:, 1])


train = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')


train.head()


scaler = StandardScaler()
y_train = train['smoking']
X_train = scaler.fit_transform(train.drop(columns=['smoking']))
X_test = scaler.transform(test)
X_test


clf = LogisticRegression(random_state=55).fit(X_train, y_train)
tr_pred_logreg = clf.predict_proba(X_train)
ts_pred_logreg = clf.predict_proba(X_test)


roc_auc_score(y_train, tr_pred_logreg[:, 1])


alg = KNeighborsClassifier(n_neighbors=97)
alg.fit(X_train, y_train)
ts_pred_knn = alg.predict_proba(X_test)


boost2 = CatBoostClassifier(loss_function='Logloss', iterations=2000,
                           learning_rate=0.2,
                           eval_metric='AUC', random_state = 44)


boost2.fit(X_train, y_train, eval_set=(X_train, y_train), use_best_model = True)


test_pred_boost = boost2.predict_proba(X_test)
train_pred_boost = boost2.predict_proba(X_train)


test_pred_boost


ts_pred_logreg


ts_pred_knn


final = 0.5 * test_pred_boost[:, 1] + 0.25 * ts_pred_knn[:, 1] + 0.25 * ts_pred_logreg[:, 1]
test['pred'] = final
test


test[['id', 'pred']].to_csv('first_sub.csv', index=False)




