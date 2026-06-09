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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


train=pd.read_csv('/kaggle/input/yovvoyi-ko-k-maymunjon-hosildorligini-aniqlash/train.csv')
train.drop('id',axis=1,inplace=True)


train.info()


train.describe()


train.isnull().sum()


%matplotlib inline
train.hist(figsize=(20,15))
plt.show()


train.corrwith(train['yield']).abs().sort_values(ascending=False)


X=train.drop('yield',axis=1)
y=train['yield'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_prepared=scaler.fit_transform(X_train)
X_test_prepared=scaler.transform(X_test)


LR = linear_model.LinearRegression()
LR.fit(X_train_prepared, y_train)
yhat = LR.predict(X_test_prepared)

mae=mean_absolute_error(y_test,yhat)
print("Linear regression mae:",mae)


poly_features_1 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features_1.fit_transform(X_train_prepared)
X_test_poly = poly_features_1.transform(X_test_prepared)
LR = linear_model.LinearRegression()
LR.fit(X_train_poly, y_train)
yhat1 = LR.predict(X_test_poly)
mae1=mean_absolute_error(y_test,yhat1)
print("Poly linear regression mae:",mae1)


RF=RandomForestRegressor(min_samples_leaf=4,min_samples_split=14,max_depth=9)
RF.fit(X_train_prepared,y_train)
yhat2=RF.predict(X_test_prepared)
mae2=mean_absolute_error(y_test,yhat2)
print("Random forest regression mae:",mae2)


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


GBR=GradientBoostingRegressor()
GBR.fit(X_train_prepared, y_train)
yhat6= GBR.predict(X_test_prepared)
mae6=mean_absolute_error(y_test,yhat6)
print("Gradient boosting regression mae:",mae6)


ridge=Ridge(alpha=10)
ridge.fit(X_train_prepared,y_train)
yhat7=ridge.predict(X_test_prepared)
mae7=mean_absolute_error(y_test,yhat7)
print("Ridge regression mae:",mae7)


from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
ridge_cv.fit(X_train_prepared, y_train)

best_alpha = ridge_cv.alpha_
print("Best alpha:", best_alpha)



from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=150, epsilon=0.1)
svr.fit(X_train_prepared, y_train)
yhat_svr = svr.predict(X_test_prepared)
mae_svr = mean_absolute_error(y_test, yhat_svr)
print("SVR MAE:", mae_svr)



from sklearn.neighbors import KNeighborsRegressor()
knn = KNeighborsRegressor(n_neighbors=15, p=1, weights='distance')
knn.fit(X_train_prepared, y_train)
yhat_knn = knn.predict(X_test_prepared)
mae_knn = mean_absolute_error(y_test, yhat_knn)
print("KNN Regressor MAE:", mae_knn)


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

param_grid = {
    'n_neighbors': range(3, 21),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_prepared, y_train)

print("Best params:", grid_search.best_params_)



test=pd.read_csv('/kaggle/input/yovvoyi-ko-k-maymunjon-hosildorligini-aniqlash/test.csv')



test_id=test.id
test.drop('id',axis=1,inplace=True)


test_prepared=scaler.transform(test)


test_predict = RF.predict(test_prepared)


len(test_id) == len(test_predict)  


submission = pd.DataFrame({'id': test_id, 'yield': test_predict})
submission.to_csv('submission_maymunjon.csv', index=False)

