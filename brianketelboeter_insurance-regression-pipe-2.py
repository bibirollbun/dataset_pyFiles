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


# Regression tools
import sklearn.linear_model as LM
import statsmodels.api as sm
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import LassoCV

# Graphing tools
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
# imputation and pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
# print(train.head())
# print(test.head())


def CV_insurance(df,alph):
    """perform 10-fold Cross Validation on dataframe
    returns 10-fold CV estimate 
    (10-fold sum over RMSE on predictive sets)
    """
    se = 0
    # transformation constants
    const_vals = [13]
    mean_vals = list(range(0,20))
    
    tmean_vals =[('num',SimpleImputer(),mean_vals)]
    tconst =[('num',SimpleImputer(strategy = 'constant',fill_value = 0),const_vals)]
    # define transformer
    trans1 = ColumnTransformer(transformers = tmean_vals, remainder = 'passthrough')
    trans2 = ColumnTransformer(transformers = tconst, remainder = 'passthrough')
    # kfold splits
    kfold = KFold(n_splits = 10)
    for train, test in kfold.split(df):
        (Xtrain,Xtest,ytrain,ytest) = (df.loc[train].drop('Premium Amount',axis=1),
                                       df.loc[test].drop('Premium Amount',axis=1),
                                       df.loc[train]['Premium Amount'],df.loc[test]['Premium Amount'])
        # model = LM.LinearRegression()
        model = Ridge(alpha= alph)
        # define pipeline
        pipeline = Pipeline(steps = [('const',trans2),('mean',trans1),('m',model)])
        pipeline.fit(Xtrain,ytrain)
        yhat = pipeline.predict(Xtest)
        # scoring
        se+=np.mean((yhat-ytest)**2)
    return (se**(1/2))
       
        


    
    


#LassoCV used as model here 
from sklearn.metrics import mean_squared_error as MSE
# Define dataframe
train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = LabelEncoder().fit_transform(train[col])
X,y = train.drop('Premium Amount', axis=1), train['Premium Amount']
const_vals = [13]
mean_vals = list(range(0,20))
tmean_vals =[('num',SimpleImputer(),mean_vals)]
tconst =[('num',SimpleImputer(strategy = 'constant',fill_value = 0),const_vals)]
# transformations
trans1 = ColumnTransformer(transformers = tmean_vals, remainder = 'passthrough')
trans2 = ColumnTransformer(transformers = tconst, remainder = 'passthrough')
# pipeline
# cv = RepeatedKFold(n_splits = 10, n_repeats = 3)
model = LassoCV()
pipeline = Pipeline(steps = [('const',trans2),('mean',trans1),('m',model)])
# fit pipeline
results = pipeline.fit(X,y)
results


pipeline.get_params()


for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = LabelEncoder().fit_transform(test[col])
yhat = pipeline.predict(test)


submission_df = pd.DataFrame({'id':test.id,'Premium Amount':yhat})
submission_df.to_csv('submission.csv',index=False)
print("Submission file created.")


submission_df




