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


train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


from sklearn.model_selection import train_test_split

train=train.drop("id",axis=1)

y=train["y"]
x=train.drop("y",axis=1)


train.isnull().sum()


x.info()


y.info()


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


cat_cols=list(x.select_dtypes(include="object").columns)

cat_transformers=Pipeline(steps=[
    ("onehotecode",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    
    ("cat", cat_transformers, cat_cols)
    
], remainder="passthrough") 

model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestClassifier(n_estimators=200,max_depth=4,class_weight='balanced'))
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)


print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


from xgboost import XGBClassifier

scale_pos_weight = 131795 / 18205


model1=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss'))
])

model1.fit(x_train,y_train)


print(model1.score(x_train,y_train))
print(model1.score(x_test,y_test))


y.value_counts(normalize=True)


from sklearn.metrics import classification_report

y_pred=model1.predict(x_test)

classification_report(y_test, y_pred)



model1=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",XGBClassifier(learning_rate= 0.1, max_depth= 7,n_estimators= 200,scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss'))
])

model1.fit(x_train,y_train)

print(model1.score(x_train,y_train))
print(model1.score(x_test,y_test))


from sklearn.model_selection import GridSearchCV


param_grid = {
    'model__max_depth': [3, 5, 7],          
    'model__learning_rate': [0.01, 0.1],   
    'model__n_estimators': [100, 200]}      


grid_search = GridSearchCV(model1, param_grid=param_grid, cv=3, scoring='f1', verbose=1)

grid_search.fit(x_train, y_train)


print("En iyi parametreler:", grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print(classification_report(y_test, y_pred))


print("En iyi parametreler:", grid_search.best_params_)
print("En iyi F1 skoru:", grid_search.best_score_)


submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
submission.head()


y_pred1 = best_model.predict(test)       
submission["y"] = y_pred1
submission.to_csv("submission.csv", index=False)

