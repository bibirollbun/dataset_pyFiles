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


d=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


d.head()


d.info()


d.describe()


d.dtypes


d.shape


d.isnull().sum()


d.columns


d.index


d['y'].value_counts()


d1=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


d1.head()


d1.info()


d1.index


d1.dtypes


d1.isnull().sum()


d1['poutcome'].value_counts()


X=d.drop(['id','y'],axis=1)
y=d['y']
X_val=d1.drop(['id'],axis=1)


from sklearn.preprocessing import LabelEncoder,StandardScaler
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
l= {}
for i in cat_cols:
    le = LabelEncoder()
    le1= pd.concat([X[i].astype(str), X_val[i].astype(str)]).unique()
    le.fit(le1)
    X[i] = le.transform(X[i].astype(str))
    X_val[i] = le.transform(X_val[i].astype(str))
    l[i] = le


from sklearn.model_selection import train_test_split,RandomizedSearchCV


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape


X_test.shape


import xgboost as xg


x= xg.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}


r= RandomizedSearchCV(
    estimator=x,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


r.fit(X_train,y_train)


best_model = r.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, y_pred_proba)


auc_score


test_pred_proba= best_model.predict_proba(X_val)[:, 1]
test_pred_proba = np.clip(test_pred_proba, 0, 1)


submission = pd.DataFrame({
    'id':d1['id'],
    'y': test_pred_proba
})
submission.to_csv('submission.csv',index=False)


submission.head()


submission.shape


submission['y'].value_counts()

