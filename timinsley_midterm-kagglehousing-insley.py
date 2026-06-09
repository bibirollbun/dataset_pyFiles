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


# installing mglearn 
!pip install mglearn


# Installing scikit for xgboost 
!pip install scikit-learn xgboost


# calling the test and train data to build the model on 
train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')
# outputting the shape of the train dataset 
#outputting the shape of the test dataset 
print("shape of train dataset:", train.shape)
print("shape of test dataset:", test.shape)


# importing the needed libraries for each model to make predictions. 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# importing the libraries for metrics 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


#importing train test split to be called in later code  
from sklearn.model_selection import train_test_split


def regression_(x, y):
    # initialize the models 
    L=LinearRegression()   
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR = ExtraTreeRegressor() 
    GBR=GradientBoostingRegressor()
    XGBC = XGBRegressor()
    dt=DecisionTreeRegressor()
    kn=KNeighborsRegressor()
    
    
# separate the data and split the data with a 20% test
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

# model names in a list 
    
    algos = [L, R, Lass, E, ETR, GBR, XGBC, dt, kn]
    algo_name = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'ExtraTree', 'GradientBoosting', 'XGBRegressor', 'DecisionTree', 'KNeighbors']

#Empty lists for success and errors. These will be filled in later 

    r2Score = []
    rmse = []
    mae = []

# dataframes for the results table 
    
    result = pd.DataFrame(columns=['R2_score', 'RMSE', 'MAE'], index = algo_name)

# Using the append function to add results to the empty lists created above 
    
    for algo in algos:
        p = algo.fit(x_train,y_train).predict(x_test)
        r2Score.append(r2_score(y_test,p))
        rmse.append(mean_squared_error(y_test,p)**.5)
        mae.append(mean_absolute_error(y_test,p))
         
# fill the columns in the table with lists 
    
    result['R2_score'] = r2Score
    result['RMSE'] = rmse
    result['MAE'] = mae


# sort the result table by r2_score and return the value 
    
    return result.sort_values('R2_score',ascending = False)



features = train.select_dtypes(include = ['int64', 'float64']).columns.drop(['id', 'MedHouseVal'])
X = train[features] # feature matrix for training
y = train['MedHouseVal'] # target variable House Value 
X_test = test[features]


# Processing the data 
# Impute the missing data values 
imputer = SimpleImputer()
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)
y = train['MedHouseVal']


# Scale the data for the feature matrix 
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
# convert data back to the data frame for column operations 
X = pd.DataFrame(X, columns=features)
X_test = pd.DataFrame(X_test, columns=features)


# will be keeping first 2 components of the pca  
pca = PCA(n_components = 2)
# fit PCA model to the Housing price data set. 
pca.fit(X_test)
pca.fit(X)
# Transform the data onto the first two principle components 
Xtest_pca = pca.transform(X_test)
X_pca = pca.transform(X)
# print out the shape of the data 
print("Original shape: {}".format(str(X_test.shape)))
print("Reduced shape: {}".format(str(Xtest_pca.shape)))


# calling the regression models listed above 
# iterating the data through each model 
# outputting each r_2, RMSE, MAE for each model into a readable columner table 
result_df = regression_(X, y)

# Display the model performance
print(result_df)


# training the model with the best perfomance 
model = XGBRegressor() 
# fit model with pca
model.fit(X_pca, y)
# make predicitions with the above model 
predictions = model.predict(Xtest_pca)
# submit the model and create a csv file
submission = pd.DataFrame({'id' : test['id'], 'MedHouseVal' : predictions})
submission.to_csv('submission.csv', index=False)


print("Submission file 'submission.csv' created.")

