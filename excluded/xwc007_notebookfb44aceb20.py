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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,r2_score


train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train=train.drop(columns=['id'])


le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])
test['Sex']=le.fit_transform(test['Sex'])


y = train['Calories'] 
X=train.drop(['Calories'],axis=1)


X = X.copy()
y_log = np.log1p(y)
numeric_cols = X.select_dtypes(include=[np.number]).columns
positive_cols = [col for col in numeric_cols if (X[col] > 0).all()]
X[positive_cols] = X[positive_cols].apply(np.log1p)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)


xgb_params = {
    "n_estimators": 1000,          
    "learning_rate": 0.03,         
    "max_depth": 5,                
    "min_child_weight": 2,         
    "subsample": 0.8,              
    "colsample_bytree": 0.8,       
    "gamma": 0.2,                  
    "reg_alpha": 0.1,              
    "reg_lambda": 1.2,             
    "random_state": 42,            
    "tree_method": "hist",         
}


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
model = XGBRegressor()
model.fit(X_train, y_train_log)
y_pred = model.predict(X_test)
rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred))
rmse_log


numeric_cols = X.select_dtypes(include=[np.number]).columns
positive_cols = [col for col in numeric_cols if (X[col] > 0).all()]
X[positive_cols] = X[positive_cols].apply(np.log1p)
test_cleaned = test.drop(columns=["id"])
test_cleaned[positive_cols] = test_cleaned[positive_cols].apply(np.log1p)


prediction=model.predict(test_cleaned)
prediction=np.expm1(prediction)


submission = pd.DataFrame({
    "id": test["id"],          
    "Calories": prediction     
})
submission.to_csv("submission.csv", index=False)


submission


submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
 

