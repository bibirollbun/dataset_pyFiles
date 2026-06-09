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


import numpy as np # linear algebra
import warnings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


train=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
train.head()


test=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
test.head()


train.head()


train.info()


train=train.drop_duplicates()
train.info()


train.isnull().sum()


fig, axes = plt.subplots(2,2, figsize=(15,8))

sns.countplot(x='Gender', hue='Exited', palette='viridis', data=train, ax=axes[0,0])
sns.countplot(x='Geography', hue='Exited', palette='viridis', data=train, ax=axes[0,1])
sns.countplot(x='Tenure', hue='Exited', palette='viridis', data=train, ax=axes[1,0])
sns.countplot(x='IsActiveMember', hue='Exited', palette='viridis', data=train, ax=axes[1,1])

plt.show()


cat_cols = ['Geography', 'Gender']
encoder=OneHotEncoder(sparse_output=False)
cat_encoded=encoder.fit_transform(train[cat_cols])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols), index=train.index)

train_num = train.drop(columns=cat_cols)
train_encoded = pd.concat([train_num, cat_encoded_df], axis=1)

train_encoded=train_encoded.drop(columns=['id','Surname','CustomerId'])
train_encoded.head()


train_encoded.corrwith(train_encoded.Exited)


y=train['Exited']
X=train.drop(columns=['Exited','id','Surname','CustomerId'])


X.head()


X_cat=['Geography','Gender']
X_num=X.drop(columns=['Geography','Gender'])



num_attribs=list(X_num)
cat_attribs=list(X_cat)

full_pipeline=ColumnTransformer([
          ('cat', OneHotEncoder(), cat_attribs),
          ('num', StandardScaler(), num_attribs )             
])


X_transformed=full_pipeline.fit_transform(X)
X_transformed


X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


LR_classifier=LogisticRegression()
LR_model=LR_classifier.fit(X_train, y_train)


y_pred=LR_model.predict_proba(X_test)[:, 1]


# Model evaluation
print("R O C:", metrics.roc_auc_score(y_test,y_pred))


model_RF=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
RF_model=model_RF.fit(X_train, y_train)
y_pred=RF_model.predict_proba(X_test)[:, 1]


# Model evaluation
print("R O C:", metrics.roc_auc_score(y_test,y_pred))


model_GB = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42)
model_GB.fit(X_train, y_train)
y_pred = model_GB.predict_proba(X_test)[:, 1]


# Model evaluation
print("R O C:", metrics.roc_auc_score(y_test,y_pred))


knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
y_prob = knn.predict_proba(X_test)[:, 1]


param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5,scoring='roc_auc')
knn_gscv.fit(X_train, y_train)

knn_gscv.best_params_, knn_gscv.best_score_


# Model evaluation
print("R O C:", metrics.roc_auc_score(y_test,y_prob))


XGB_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
XGB_model.fit(X_train, y_train)
y_proba=XGB_model.predict_proba(X_test)[:, 1]


# Model evaluation
print("R O C:", metrics.roc_auc_score(y_test,y_proba))


test_data=test.drop(columns=['id','Surname','CustomerId'])
test_transformed=full_pipeline.transform(test_data)
y_test_pred=RF_model.predict_proba(test_transformed)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],  
    'Exited': y_test_pred
})

submission.to_csv('submission.csv', index=False)

