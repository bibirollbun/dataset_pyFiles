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


train = pd.read_csv("/kaggle/input/playground-series-s3e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e1/test.csv")



from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split #To make code appear neat
#I will add this needed import for future steps here


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def regression_(x, y):
    L = LinearRegression()
    R = Ridge()
    Lass = Lasso()
    E = ElasticNet()
    ETR = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    XGBC = XGBRegressor()
    dt = DecisionTreeRegressor()
    kn = KNeighborsRegressor() 
    svm = SVR()

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # List models and model names
    algos = [L, R, Lass, E, ETR, GBR, XGBC, dt, kn, svm]
    algo_names = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Extra Tree', 'Gradient Boosting', 
                  'XGradient Boosting', 'Decision Tree', 'KNeighbors', 'SVM'] 

    # Result storage
    r2Score = []
    rmse = []
    mae = []

    result = pd.DataFrame(columns=['R2_score', 'RMSE', 'MAE'], index=algo_names)

    # Train each model and evaluate
    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        r2Score.append(r2_score(y_test, p))
        rmse.append(mean_squared_error(y_test, p, squared=False))
        mae.append(mean_absolute_error(y_test, p))

    # Store results
    result['R2_score'] = r2Score
    result['RMSE'] = rmse
    result['MAE'] = mae

    # Return sorted results
    return result.sort_values('R2_score', ascending=False)




features = train.select_dtypes(include = ['int64', 'float64']).columns.drop(['MedHouseVal', 'id'])
X = train[features]
y = train['MedHouseVal']
X_test = test[features]


# Impute values that are missing
# Import library
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)



# Import library
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# Import library
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
# fitting the PCA model to data
pca.fit(X_test)
pca.fit(X)

# transform data 
X_test_pca = pca.transform(X_test)
X_pca = pca.transform(X)
print("Original shape: {}".format(str(X_test.shape)))
print("Reduced shape: {}".format(str(X_test_pca.shape)))


result = regression_(X_pca, y)
print(result)


# The best performing model is Gradient Boosting
# Train model
model = GradientBoostingRegressor()
model.fit(X_pca, y)

# create predicitions
predictions = model.predict(X_test_pca)


# Create submission file
submission = pd.DataFrame({'id': test['id'], 'MedHouseVal': predictions})
submission.to_csv('submission.csv', index = False)
print("Submission file 'submission.csv' created.")

