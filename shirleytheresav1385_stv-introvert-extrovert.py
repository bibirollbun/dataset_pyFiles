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


# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(train.shape, test.shape)
train.head()


print(train.shape)
print(train.dtypes)
train.head()
train.describe()
train['Personality'].value_counts()




X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)


y = y.map({'Extrovert': 1, 'Introvert': 0})


for col in ['Stage_fear', 'Drained_after_socializing']:
    X[col] = X[col].map({'Yes': 1, 'No': 0})
    X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})


X.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)



# ðŸ”· Encode categorical columns BEFORE imputation
X['Stage_fear'] = X['Stage_fear'].map({'Yes': 1, 'No': 0})
X['Drained_after_socializing'] = X['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

X_test['Stage_fear'] = X_test['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test['Drained_after_socializing'] = X_test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# ðŸ”· Check datatypes
print(X.dtypes)  # should show int or float for all




print(X.columns)
print(len(X.columns))



X = X.dropna(axis=1, how='all')
X_test = X_test.dropna(axis=1, how='all')

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)


X = X.fillna(X.median())
X_test = X_test.fillna(X.median())





# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


clf.fit(X_scaled, y)


test_preds = clf.predict(X_test_scaled)
test_preds_labels = pd.Series(test_preds).map({1: 'Extrovert', 0: 'Introvert'})


submission = sample_submission.copy()
submission['Personality'] = test_preds_labels
submission.to_csv('submission.csv', index=False)

print("submission.csv saved!")
submission.tail()




