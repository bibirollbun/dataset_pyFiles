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


# Load the train and test data
train_data = pd.read_csv("/kaggle/input/playground-series-s3e1/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s3e1/test.csv")


# Assuimg ExtraTree, GradientBoosing and XGB are regressors for the imports.
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Function tests multiple models at once to find which preforms best
from sklearn.model_selection import train_test_split
def regression_(x,y):

    #I include the models imported
    L=LinearRegression()
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR=ExtraTreeRegressor()
    GBR=GradientBoostingRegressor()
    XGBR=XGBRegressor()
    dt=DecisionTreeRegressor()
    kn=KNeighborsRegressor()
    svm=SVR()
    
    #I separate the data as train and test.
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=4)
    
    #I put my models and model names in a list.
    algos=[L,R,Lass,E,ETR,GBR,XGBR,dt,kn,svm]
    algo_names=['Linear','Ridge','Lasso','ElasticNet','Extra Tree','Gradient Boosting','XGradientBoosting','DecisionTree','KNeighbors', 'SVM']
    
    #I create empty lists for error rates and success rates, which I will fill in later.
    r2Score = []
    rmse = []
    mae = []
    
    #I am creating a dataframe as I want to see all results as a table.
    #Its columns will be 'R2_score', 'RMSE', 'MAE'. The indexes will take from the string I created for the model names.
    result=pd.DataFrame(columns=['R2_score','RMSE','MAE'],index=algo_names)
    
    #I run the regression algorithms for each model and find the results of r2_score, mean_absolute_error and mean_squared_error.
    #And I add these results I found to the empty lists I created above with the append () function.
    for algo in algos:
        p=algo.fit(x_train,y_train).predict(x_test)
        r2Score.append(r2_score(y_test,p))
        rmse.append(mean_squared_error(y_test,p)**.5)
        mae.append(mean_absolute_error(y_test,p))
        
    #I fill the columns in the table with these lists.
    result['R2_score']=r2Score
    result.RMSE=rmse
    result.MAE=mae
    
    #I sort my result table by r2_score value and return it.
    return result.sort_values('R2_score',ascending=False)


# Selecting features, use only numerical features for simplicity,
# and explicitly exclude the 'id' column as it's not a predictive feature.
features = train_data.select_dtypes(include=['int64', 'float64']).columns.drop(['MedHouseVal', 'id'])
X = train_data[features]
y = train_data['MedHouseVal']
X_test = test_data[features]
print("Selected Features")


# Data preprocessing
# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)
print("Finished Data Preprocessing")


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
print("Applied Feature Scaling")


from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to Housing Dataset
pca.fit(X_test)
pca.fit(X)

# transform data onto the first two principal components
X_test_pca = pca.transform(X_test)
X_pca = pca.transform(X)
print("Original shape: {}".format(str(X_test.shape)))
print("Reduced shape: {}".format(str(X_test_pca.shape)))

print("Applied PCA")


# Gradient Boosting has the best score, so I will use it.
print(regression_(X_pca, y))
print("Called Regression")


model = LinearRegression()
model.fit(X_pca, y)

# Predictions
predictions = model.predict(X_test_pca)
print("Made Predictions")


# Create submission file
submission = pd.DataFrame({'id': test_data['id'], 'MedHouseVal': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")

