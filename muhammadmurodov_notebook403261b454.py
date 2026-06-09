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


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
train.sample(5)


test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
test.sample(5)


print(train.shape)
test.shape


print(train.info(), '\n\n')
train.describe()


train.isnull().sum()


plt.figure(figsize=(12, 8))

numeric_df = train.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('Correlation')
plt.show()


train.head()


y=train['Exited']
train.drop(['Exited', 'CustomerId', 'Surname'], axis = 1, inplace=True)
train.head()


test_ids=test['CustomerId']
test.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


test.sample()


train['Gender'] = train['Gender'].map({'Female':0, 'Male':1})
test['Gender'] = test['Gender'].map({'Female':0, 'Male':1})


train = pd.get_dummies(train, columns=['Geography'], drop_first=True)
test = pd.get_dummies(test, columns=['Geography'], drop_first=True)


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

xgb_model = XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'  
)

xgb_cv = cross_val_score(xgb_model, train, y, cv=5, scoring='roc_auc')
print("Mean ROC AUC:", xgb_cv.mean())



xgb_cv


xgb_model.fit(train, y)


xgb_probs = xgb_model.predict_proba(test)[:, 1]


file = pd.DataFrame({
    'id':test_ids, 'Exited': xgb_probs 
})
file.to_csv('submission.csv', index=False)

