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
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print(train.info())
print(train.describe())


sns.countplot(x='rainfall', data=train)
plt.title("Rainfall Distribution")
plt.show()


train.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


corr = train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

xgb.fit(X_train, y_train)

y_pred_proba = xgb.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"Validation AUC: {auc_score:.4f}")

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}
grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, scoring='roc_auc', cv=3)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best AUC:", grid.best_score_)


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='best')
plt.show()


test_ids = test['id']
X_test = test.drop(['id'], axis=1)
test_pred_proba = xgb.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_pred_proba
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

