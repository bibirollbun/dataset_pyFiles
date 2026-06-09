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
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier 
from sklearn.metrics import roc_auc_score


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', header=0)
train.head()


train.shape


cols_check=train.select_dtypes(object)
cols_check.columns


# Numerical & Categorical splits

num_cols=[]
cat_cols=[]
for col in train.columns:
    if col in ('id','loan_paid_back') :
        continue
    
    try:
        pd.to_numeric(train[col])
        num_cols.append(col)
    except(ValueError,TypeError):
        cat_cols.append(col)


print("Num_cols - ",num_cols)
print("Cat_cols - ",cat_cols)


#pipeline

numeric_transformer=Pipeline(steps=[

("imputer",SimpleImputer(strategy='median')),
("scaler",StandardScaler())
    
])

categorical_transformer=Pipeline(steps=[
("imputer",SimpleImputer(strategy='most_frequent')),
("encoder",OneHotEncoder())

    
])

preprocessor=ColumnTransformer(
    transformers=[
        ("num",numeric_transformer,num_cols),
        ("cat",categorical_transformer,cat_cols)
        
    ]
    
)


param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    # "model__min_child_weight": [1, 3, 5],
    # "model__gamma": [0, 0.1, 0.3],
    # "model__subsample": [0.6, 0.8, 1.0],
    # "model__colsample_bytree": [0.6, 0.8, 1.0],
    # "model__reg_lambda": [0.1, 1, 10],
    # "model__reg_alpha": [0, 0.1, 1]
}


model=XGBClassifier(
    eval_metric='logloss'
)


pipeline=Pipeline(steps=[
    ('preprocess',preprocessor),
    ('model',model)
])

grid_search=GridSearchCV(
estimator=pipeline,
param_grid=param_grid,
scoring='roc_auc'
)




X = train.drop(columns=["loan_paid_back", "id"], errors="ignore")
y = train["loan_paid_back"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


grid_search.fit(X_train,y_train)


print("Best Params", grid_search.best_params_)
print("Best Score", grid_search.best_score_)


best_piepline=grid_search.best_estimator_


y_pred_test=grid_search.predict(X_test)


roc_auc_score(y_test,y_pred_test)


test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test.drop(columns='id',inplace=True)
test.head()



test_orig = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test_ids = test_orig['id']

test_data = test_orig.drop(columns='id')

y_test_pred_proba = best_piepline.predict_proba(test_data)[:, 1] 

df_submit = pd.DataFrame({
    "id": test_ids,
    "target": y_test_pred_proba  
})


df_submit.to_csv("submission.csv", index=False)



df_submit.head()







