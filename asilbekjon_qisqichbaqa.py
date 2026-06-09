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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


test=pd.read_csv('/kaggle/input/regressiontask/test.csv')
train=pd.read_csv('/kaggle/input/regressiontask/train.csv')
train.drop('id',axis=1,inplace=True)
test_id=test.id
test.drop('id',axis=1,inplace=True)


test.info()


test.describe()


categorical=['Sex']
numeric=['Length','Diameter','Height','Weight','Shucked Weight','Viscera Weight','Viscera Weight']


# numeric
num_pipeline = Pipeline([
    ('scaler',StandardScaler())
])

# categorical
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
])
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numeric),
    ('cat', OneHotEncoder(), categorical)
])


X=train.drop('Age',axis=1)
y=train['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 1: Transform with full_pipeline first
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)


RF=RandomForestRegressor(max_depth=9)
RF.fit(X_train_prepared, y_train)
yhat2= RF.predict(X_test_prepared)
mae2=mean_absolute_error(y_test,yhat2)
print("Random forest regression mae:",mae2)


grid={'max_depth':np.arange(1,10)}
knn_gscv=GridSearchCV(RF, grid, cv=5)
knn_gscv.fit(X_train_prepared,y_train)
knn_gscv.best_params_


DT=DecisionTreeRegressor()
DT.fit(X_train_prepared, y_train)
yhat3= DT.predict(X_test_prepared)
mae3=mean_absolute_error(y_test,yhat3)
print("Decision tree regression mae:",mae3)


XGB=XGBRegressor(random_state=42,max_depth=3)
XGB.fit(X_train_prepared, y_train)
yhat4= XGB.predict(X_test_prepared)
mae4=mean_absolute_error(y_test,yhat4)
print("XGBR regression mae:",mae4)


CBR=CatBoostRegressor(iterations=500,learning_rate=0.05,verbose=0,max_depth=5)
CBR.fit(X_train_prepared, y_train)
yhat5= CBR.predict(X_test_prepared)
mae5=mean_absolute_error(y_test,yhat5)
print("Cat boost regression mae:",mae5)


grid={'max_depth':np.arange(1,10)}
knn_gscv=GridSearchCV(CBR, grid, cv=5)
knn_gscv.fit(X_train_prepared,y_train)
knn_gscv.best_params_


GBR=GradientBoostingRegressor()
GBR.fit(X_train_prepared, y_train)
yhat6= GBR.predict(X_test_prepared)
mae6=mean_absolute_error(y_test,yhat6)
print("Gradient boosting regression mae:",mae6)


from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
svr.fit(X_train_prepared, y_train)
yhat_svr = svr.predict(X_test_prepared)
mae_svr = mean_absolute_error(y_test, yhat_svr)
print("SVR MAE:", mae_svr)


poly_features = PolynomialFeatures(degree=3, include_bias=False)
full_pipeline.fit(X_train)
X_train_prepared = full_pipeline.transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
X_train_poly = poly_features.fit_transform(X_train_prepared)
X_test_poly = poly_features.transform(X_test_prepared)
full_pipeline.fit(X_train)
LR = linear_model.LinearRegression()
LR.fit(X_train_poly, y_train)
yhat = LR.predict(X_test_poly)
mae=mean_absolute_error(y_test,yhat)
print("Poly linear regression mae:",mae)


#test predict
test_prepared=full_pipeline.transform(test)
test_predict = svr.predict(test_prepared)


len(test_id) == len(test_predict)  # should be True


submission = pd.DataFrame({'id': test_id, 'predicted_yield': test_predict})
submission.to_csv('submission_qisqichbaqa.csv', index=False)

