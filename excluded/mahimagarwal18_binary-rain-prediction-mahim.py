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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')


train_df.head()


train_df.info() # no missing values in train data
print(train_df.isnull().sum())


test_df.head()


test_df.info()
print(test_df.isnull().sum())


X = train_df.drop(columns='rainfall')
y = train_df['rainfall']


# Splitting data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# handling missing values and normalizing values
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


my_pipeline = Pipeline([
    ('imputer', SimpleImputer()),  
    ('scaler', StandardScaler())
])

X_train_new = my_pipeline.fit_transform(X_train)
X_val_new = my_pipeline.fit_transform(X_val)


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()

lr_model.fit(X_train_new, y_train)
y_pred = lr_model.predict_proba(X_val_new)[:,1]


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_val, y_pred)
print(roc_auc)


from sklearn.metrics import accuracy_score

y_predict = lr_model.predict(X_val_new)
acc = accuracy_score(y_val, y_predict)

print(acc)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()

rf_model.fit(X_train_new, y_train)
y_pred = rf_model.predict_proba(X_val_new)[:,1]

roc_auc = roc_auc_score(y_val, y_pred)
print(roc_auc)


from sklearn.metrics import accuracy_score

y_predict = rf_model.predict(X_val_new)
acc = accuracy_score(y_val, y_predict)

print(acc)


X_test = my_pipeline.fit_transform(test_df)

predictions = rf_model.predict_proba(X_test)[:,1]


submission = pd.DataFrame({'id': test_df.index, 'rainfall': predictions})
submission.to_csv('submission.csv', index=False)


submission.head()




