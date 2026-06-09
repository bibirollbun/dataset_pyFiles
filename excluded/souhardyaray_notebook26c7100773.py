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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import os,sys
import warnings

from sklearn.model_selection import train_test_split,KFold,GroupKFold,RepeatedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class Config:
    train_data_path="/kaggle/input/playground-series-s5e5/train.csv"
    test_data_path="/kaggle/input/playground-series-s5e5/test.csv"
    target='Calories'
    warnings_ignore=warnings.filterwarnings('ignore')
    seed=42


Config.warnings_ignore


train_df=pd.read_csv(Config.train_data_path)
test_df=pd.read_csv(Config.test_data_path)


len(test_df)


train_df['Sex']=train_df['Sex'].map({"male":1,"female":0})
test_df['Sex']=test_df['Sex'].map({"male":1,"female":0})


def FE(train, test):
     
    train['duration_by_heart_rate'] = train['Duration'] / train['Heart_Rate']
    test['duration_by_heart_rate'] = test['Duration'] / test['Heart_Rate'] 
    
    train['heart_rate_by_temp'] = train['Heart_Rate'] / train['Body_Temp']
    test['heart_rate_by_temp'] = test['Heart_Rate'] / test['Body_Temp']

    
    train['duration_by_temp'] = train['Duration'] / train['Body_Temp']
    test['duration_by_temp'] = test['Duration'] / test['Body_Temp']

    train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
    test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)
    
    train['Weight_Height_Ratio'] = train['Weight'] / train['Height']
    test['Weight_Height_Ratio'] = test['Weight'] / test['Height']
  
    train['Weight_Duration'] = train['Weight'] * train['Duration']
    test['Weight_Duration'] = test['Weight'] * test['Duration']
    
    train['Age_Duration'] = train['Age'] * train['Duration']
    test['Age_Duration'] = test['Age'] * test['Duration']
    
    train['Heart_Temp'] = train['Heart_Rate'] * train['Body_Temp']
    test['Heart_Temp'] = test['Heart_Rate'] * test['Body_Temp']

    return train, test

train_df, test_df = FE(train_df, test_df)


X = train_df.drop(columns=['Calories'])
y = train_df['Calories']


X_train = X.drop(columns=['id'])
y_train = y.drop(columns=['id'])


X_test = test_df.copy()
X_test = X_test.drop(columns = ['id'])


import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def pca_regression(X_train, y_train, n_components=None):
    """
    Perform PCA Regression (Principal Component Regression)
    
    Parameters:
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,)
        Target values
    n_components : int, float or 'mle', default=None
        Number of components to keep. If None, all components are kept.
        
    Returns:
    pipeline : sklearn Pipeline object
        Fitted PCA regression pipeline
    best_n_components : int
        Optimal number of components (if n_components=None)
    """
    
    # Create a pipeline that first scales the data, applies PCA, then fits a linear regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA()),               # PCA transformation
        ('regression', LinearRegression())  # Linear regression
    ])
    
    if n_components is not None:
        # If number of components is specified, use it directly
        pipeline.set_params(pca__n_components=n_components)
        pipeline.fit(X_train, y_train)
        return pipeline, n_components
    else:
        # If number of components is not specified, perform grid search to find optimal number
        param_grid = {
            'pca__n_components': np.arange(1, min(X_train.shape[0], X_train.shape[1]))
        }
        
        # Using negative MSE as scoring (higher is better)
        grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        
        best_n_components = grid_search.best_params_['pca__n_components']
        best_pipeline = grid_search.best_estimator_
        
        return best_pipeline, best_n_components

# Example usage:
# Assuming X_train and y_train are your training data and target variable
# model, n_components = pca_regression(X_train, y_train)
# To use a specific number of components: model, _ = pca_regression(X_train, y_train, n_components=5)


model, _ = pca_regression(X_train, y_train)


y_pred = model.predict(X_test) 


model.fit(X_train,y_train)


sub_df = pd.DataFrame({"id":test_df['id'],"Calories":y_pred})


sub_df.to_csv("submission.csv",index=False)


sub_df.sort_values('Calories', ascending=True)







