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


import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test=pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
data


data.isna().sum()


def transform_data(data):
    data['sale_date']= pd.to_datetime(data['sale_date'])
    data['subdivision']=data['subdivision'].fillna('Missing')
    data['submarket'] = data['submarket'].fillna('Missing')
    obj = data.select_dtypes(include=['object']).columns
    for i in obj:
        data[i]= data[i].astype('category')
    data['sale_nbr']= data['sale_nbr'].fillna(0)
    data['sale_nbr']= data['sale_nbr'].astype('int8')


#simple feature engineering 
def feature_engineer(X):
    min_sale_year=X['sale_date'].dt.year.min()
    X['sale_since_first_year']= X['sale_date'].dt.year - min_sale_year
    X['diff'] = X['imp_val']-X['land_val']
    X['house_age_at_sale'] = X['sale_date'].dt.year- X['year_built']


transform_data(data)
feature_engineer(data)
y = data['sale_price']
X= data.select_dtypes(include=['int64','float64','category','int8','datetime64[ns]','int32'])
X.drop("sale_price",axis=1,inplace=True)
X.info()


X[['present_use']]


from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from catboost import CatBoostRegressor


CATBOOST_PARAMS = {
    'iterations': 2000, #number of trees to make
    'learning_rate': 0.05, #Smaller value makes training slower but more robust and accurate. Often paired with high n_estimators
    'subsample': 0.8, #precentage of data to use in each tree 80% here to prevent overfitting
    'random_seed': 10,
    'verbose': False,  # suppress output
    'early_stopping_rounds': 100, #This parameter is used to prevent overfitting and speed up training by stopping the model early when it stops improving.
    'thread_count': -1,#use all threads
}



categorical_features = X.select_dtypes(include=['category'])
categorical_features = list(categorical_features.columns)
categorical_features


#Upper model
model_upper = CatBoostRegressor(**CATBOOST_PARAMS, loss_function= 'Quantile:alpha=0.95',cat_features=categorical_features )


#LOWER MODEL
model_lower= CatBoostRegressor(**CATBOOST_PARAMS,loss_function="Quantile:alpha=0.05",cat_features= categorical_features)



model_lower.fit(X,y)
model_upper.fit(X,y)


transform_data(test)
feature_engineer(test)
lower = model_lower.predict(test)
upper = model_upper.predict(test)



lower_importance= model_lower.get_feature_importance()
upper_importance= model_upper.get_feature_importance()

feature_names= X.columns
importance= pd.DataFrame({
    'feature_name':feature_names,
    'lower_model':lower_importance,
    'upper_model':upper_importance
})
importance= importance.sort_values(by=['lower_model','upper_model'],ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(y='feature_name',x='lower_model',data=importance)


old_score=0
def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage
        
    return np.mean(score)


score = winkler_score(y,lower,upper)
if old_score==0:
    print('First Time')
    print(score)
    old_score=score
elif old_score>score:
    print("Better")
    print(old_score," ",score)
    old_score=score
else:
    print("worse")
    print(old_score,' ',score)
   


sub = pd.DataFrame({
        'id':test.id,
        'pi_lower':lower,
        'pi_upper':upper
    })
sub
sub.to_csv('submission.csv',index=False)

