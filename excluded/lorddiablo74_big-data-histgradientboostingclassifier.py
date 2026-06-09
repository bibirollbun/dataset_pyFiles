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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier


sns.set()


train_raw = pd.read_csv('/kaggle/input/mma-865-2025-w/kiva_train.csv')
test_raw = pd.read_csv('/kaggle/input/mma-865-2025-w/kiva_test.csv')
train_raw.shape, test_raw.shape


train_raw.head().T


test_raw.head().T


train_raw.info()


test_raw.info()


train_raw.describe()


test_raw.describe()


sns.countplot(x='status', data=train_raw);


sns.countplot(x='country', data=train_raw, hue='status');


sns.countplot(x='gender', data=train_raw, hue='status');


sns.countplot(x='nonpayment', data=train_raw, hue='status');


sns.countplot(x='sector', data=train_raw, hue='status');
plt.xticks(rotation=45);


sns.FacetGrid(train_raw, hue='status', height=5).map(sns.distplot, 'loan_amount').add_legend();


submission = pd.DataFrame()
submission['id']= test_raw['id']
submission.shape


df = pd.concat([train_raw.drop(['status', 'id', 'en'], axis=1), test_raw.drop(['id', 'en'], axis=1)], axis=0)
df.shape


df.isnull().sum()


df = pd.get_dummies(df, drop_first=True)
df.shape


X = df[:train_raw.shape[0]]
test = df[train_raw.shape[0]:]
y = train_raw['status']
X.shape, test.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.2)


model = GradientBoostingClassifier()
model.fit(X, y)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)


selector = SelectFromModel(GradientBoostingClassifier())
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]
print(selected_features)


pca = PCA(n_components=2)
pca_data = pca.fit_transform(df)
pca_data.shape


pca_df = pd.DataFrame()
pca_df['pca_1'] = pca_data[:train_raw.shape[0], 0]
pca_df['pca_2'] = pca_data[:train_raw.shape[0], 1]
pca_df['status'] = train_raw['status']


sns.lmplot(data=pca_df, x='pca_1', y='pca_2', hue='status', fit_reg=False);


model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


predict = model.predict_proba(test)[:,1]
submission['status'] = predict
submission.to_csv('submission.csv', index=False)
submission.head()




