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


train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')
print("Train")
print(train.head())
print("Test")
print(test.head())


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

print("Imported Libraries")


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

print("Imported Libraries")


from sklearn.model_selection import train_test_split

def regression(x,y):
    L=LinearRegression()
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR=ExtraTreesRegressor()
    GBR=GradientBoostingRegressor()
    XGB=XGBRegressor()
    DTR=DecisionTreeRegressor()
    KNR=KNeighborsRegressor()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

    algos=[L,R,Lass,E,ETR,GBR,XGB,DTR,KNR]
    algo_names=['Linear','Ridge','Lasso','ElasticNet','Extra Tree','Gradient Boosting','XGradientBoosting','DecisionTree','KNeighbors']

    r2score = []
    rmse = []
    mae = []

    result=pd.DataFrame(columns=['R2 Score','RMSE','MAE'],index=algo_names)

    for algo in algos:
        reg = algo.fit(x_train,y_train).predict(x_test)
        r2score.append(r2_score(y_test,reg))
        rmse.append(mean_squared_error(y_test,reg)**.5)
        mae.append(mean_absolute_error(y_test,reg))

    result['R2 Score'] = r2score
    result['RMSE'] = rmse
    result['MAE'] = mae
    
    return result.sort_values('R2 Score',ascending=False)


features = train.select_dtypes(include=['int64', 'float64']).columns.drop(['id','MedHouseVal'])
X = train[features]
y = train['MedHouseVal']
X_test = test[features]

print(X.head())
print(y.head())
print(X_test.head())


from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X = imp.fit_transform(X)
X_test = imp.transform(X_test)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
pca.fit(X_test)

X_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)


table = regression(X_pca, y)
print(table)
print("XGradientBoosting had the second highest R2 score meaning it was the second best generalized model. Gradient Boosting was the best in terms of R2 score.")
print("XGradientBoosting had the lowest Mean Absolute Error value. The MAE treats all errors as equally weighted.")
print("XGradientBoosting had the second lowest Root Mean Square Error value. The RMSE treats larger errors as 'worse' by squaring each error before summation.")
print("Given the details above, I believe XGRadientBoosting to be the best performing algorithm overall")


submission_preds = XGBRegressor().fit(X,y).predict(X_test)
df = pd.DataFrame({"id": test['id'],
                   "MedHouseVal": submission_preds})

df.to_csv("submission.csv", index=False)

