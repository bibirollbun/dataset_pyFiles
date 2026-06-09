# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data= pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')

train_data.fillna(train_data.median(),inplace=True)
test_data.fillna(test_data.median(),inplace=True)


train_data.head()


test_data.head()


x= train_data.drop(columns=['target'])
y=train_data['target']



test_ids= test_data['id']
x_test=test_data.drop(columns=['id'])


x_train,x_val,y_train,y_val =train_test_split(x,y,test_size= 0.2,random_state=42)



scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)



base_models=[
    ('xgb',XGBRegressor(random_state=42)),
     ('lightgbm',LGBMRegressor(random_state=42)),
     ('CatBoost',CatBoostRegressor(random_state=42,verbose=0)),
]
   


stacking_model=StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(random_state=42)
)


def objective(trial):
    params={
        'xgb__n_estimators':trial.suggest_int('xgb__n_estimators',100,500),
        'xgb__learning_rate':trial.suggest_float('xgb__learning_rate',0.01,0.2),
        
        'lightgbm__n_estimators':trial.suggest_int('lightgbm__n_estimators',100,500),
        'lightgbm__learning_rate':trial.suggest_float('lightgbm__learning_rate',0.01,0.2),
        'catboost__iterations':trial.suggest_int('catboost__iterations',500,1000),
        'catboost__learning_rate':trial.suggest_int('catboost__learning_rate',0.01,0.2)
        
    }

    stacking_model.set_params(**params)
    scores = cross_val_score(stacking_model,x_train,y_train,cv=3,scoring='r2')
    return np.mean(scores)


study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=50,n_jobs=-1)


best_params = study.best_params
print(f"best paramters: {best_params}")


stacking_model.set_params(**best_params)
stacking_model.fit(x_train,y_train)


val_r2 =r2_score(y_val,stacking_model.predict(x_val))
print(f"r2 :{val_r2}")






test_predictions=stacking_model.predict(x_test)


submission = pd.DataFrame({'id': test_ids,'target': test_predictions})
submission.to_csv('submission.csv',index=False)



print("Submission File created")




