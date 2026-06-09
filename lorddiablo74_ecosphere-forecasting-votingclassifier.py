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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier


train_raw = pd.read_csv('/kaggle/input/ecosphere-forecasting/train.csv')
test_raw = pd.read_csv('/kaggle/input/ecosphere-forecasting/test.csv')
train_raw.shape, test_raw.shape


train_raw.head()


test_raw.head()


train_raw['Air Quality'].value_counts()


train_raw.describe().T


test_raw.describe().T


X = train_raw.drop(['Id', 'Air Quality'], axis=1)
y = train_raw['Air Quality']
X.shape, y.shape


y = y.map({'Good': 0, 'Moderate': 1, 'Poor': 2, 'Hazardous': 3})


submission = pd.DataFrame()
submission['Id'] = test_raw['Id']


scaller = RobustScaler()
X = scaller.fit_transform(X)
test_raw = scaller.transform(test_raw.drop('Id', axis=1))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


catboost_params = {
    'learning_rate': 0.03, 
    'l2_leaf_reg': 0.001, 
    'iterations': 600, 
    'depth': 4, 
    'bagging_temperature': 0.5555555555555556,
    'verbose': False
}
extratrees_params = {
    'n_estimators': 400, 
    'min_samples_split': 10, 
    'min_samples_leaf': 1, 
    'max_features': 'sqrt', 
    'max_depth': 20, 
    'bootstrap': False
}


models = [
    CatBoostClassifier(**catboost_params),
    ExtraTreesClassifier(**extratrees_params)
]


estimators = [(model.__class__.__name__, model) for model in models]


voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)
accuracy = voting_clf.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')


ConfusionMatrixDisplay.from_estimator(voting_clf, X_test, y_test);


predict = voting_clf.predict(test_raw)


submission['Air_Quality_Level'] = predict
submission['Air_Quality_Level'] = submission['Air_Quality_Level'].map({0: 'Good', 1: 'Moderate', 2: 'Poor', 3: 'Hazardous'})
submission.to_csv('submission.csv', index=False)
submission.head()




