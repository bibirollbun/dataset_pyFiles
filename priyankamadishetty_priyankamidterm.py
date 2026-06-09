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


train_data = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')

print("Train and test data loaded Successfully")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
print("Imported Libraries Successfully")


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("Imported Libraries Successfully")


def regression_(x,y):
    L=LinearRegression()
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR=ExtraTreeRegressor()
    GBR=GradientBoostingRegressor()
    XGBC=XGBRegressor()
    dt=DecisionTreeRegressor()
    kn=KNeighborsRegressor()
    #I separate the data as train and test. (20% test)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=73)
    #I put my models and model names in a list.
    algos=[L,R,Lass,E,ETR,GBR,XGBC,dt,kn]
    algo_names=['Linear','Ridge','Lasso','ElasticNet','Extra Tree','Gradient Boosting','XGradientBoosting','DecisionTree','KNeighbors']
    #I create empty lists for error rates and success rates, which I will fill in later.
    r2Score = []
    rmse = []
    mae = []
    result=pd.DataFrame(columns=['R2_score','RMSE','MAE'],index=algo_names)
    #I run the regression algorithms for each model and find the results of r2_score, mean_absolute_error and mean_squared_error.
    #And I add these results I found to the empty lists I created above with the append () function.
    for algo in algos:
        p=algo.fit(x_train,y_train).predict(x_test)
        r2Score.append(r2_score(y_test,p))
        rmse.append(mean_squared_error(y_test,p)**.5)
        mae.append(mean_absolute_error(y_test,p))
    result['R2_score']=r2Score
    result.RMSE=rmse
    result.MAE=mae
    return result.sort_values('R2_score',ascending=False)



features = train_data.select_dtypes(include=['int64', 'float64']).columns.drop(['MedHouseVal', 'id'])
X = train_data[features]
y = train_data['MedHouseVal']
X_test = test_data[features]
print("selected features successfully")


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)
print("Completed Data Preprocessing")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
print("scaled successfully")


from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to beast cancer data
pca.fit(X_test)
pca.fit(X)
X_test_pca = pca.transform(X_test)
X_pca = pca.transform(X)
print("Applied PCA")


regression_(X_pca,y)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=73)
GBR=GradientBoostingRegressor()
p=GBR.fit(X_train,y_train).predict(X_val)
r2Score= r2_score(y_val,p)
r2Score


GBR.fit(X, y)

# Predictions
predictions = GBR.predict(X_test)

submission = pd.DataFrame({'id': test_data['id'], 'MedHouseVal': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")

