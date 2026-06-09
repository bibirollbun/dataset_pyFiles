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


#import basic python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer


#Import data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

#check data
print(train)

#show data dimension
print(train.shape)

#show data type
print(train.dtypes) 
#All are of the object type except num_sold
#Then consider splitting the date column and extracting information for the year, month, and week


#Distribution information of sticker sales (extreme value, quantile value)
train['num_sold'].describe()


train['country'].value_counts()
#The 'country' column contains data for six countries


train['store'].value_counts()
#The 'store' column contains data information for the three sticker stores


train['date'].value_counts()
# It runs from January 1, 2010 to December 31, 2016 and contains 90 data messages per day


null_values = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
print(null_values)
# Check the number of null values,'num_sold' has 8871 null_values

null_percent = pd.DataFrame(train.isnull().sum().sort_values(ascending=False) /
len(train)*100,columns=['%'])
print(null_percent)
# View the proportion of null values. The ratio is not high, and then the vacancy value will be considered to be deleted


# Data visualization (for num_sold)
plt.figure(figsize=(10, 6))
sns.histplot(data=train,x='num_sold',stat='density',kde=True,color='red',bins=30)
plt.title(f'Distribution of num_sold')
plt.show()
# As you can see, most num_sold numbers are distributed in the 0-2000 range


# Show average num_sold grouped by country
plt.figure(figsize=(10, 6))
country_sell = train.groupby('country')['num_sold'].mean()
text_values = [f"{value:.0f}" for value in country_sell.values]
plt.bar(country_sell.index, country_sell.values,color='red')
for i in range(len(country_sell)):
    plt.text(country_sell.index[i], country_sell.values[i], text_values[i], ha='center',va='bottom')
plt.ylim([0,1600])
plt.xlabel("country")
plt.ylabel("Average num_sold")
plt.show()
#Norway has the highest sticker average num_sold among these countries, while Singapore, Canada and France are in the second tier, with little difference in their average num_sold


#Show average num_sold grouped by store 
plt.figure(figsize=(8, 6))
store_sell = train.groupby('store')['num_sold'].mean()
text_values = [f"{value:.0f}" for value in store_sell.values]
plt.bar(store_sell.index, store_sell.values,color='red')
for i in range(len(store_sell)):
    plt.text(store_sell.index[i], store_sell.values[i], text_values[i], ha='center',va='bottom')
plt.ylim([0,1200])
plt.xlabel("store")
plt.ylabel("Average num_sold")
plt.show()
#As you can see, Premium Sticker Mart Stores have the highest average num_sold


#Show average num_sold grouped by product
plt.figure(figsize=(10, 6))
product_sell = train.groupby('product')['num_sold'].mean()
text_values = [f"{value:.0f}" for value in product_sell.values]
plt.bar(product_sell.index, product_sell.values,color='red')
for i in range(len(product_sell)):
    plt.text(product_sell.index[i], product_sell.values[i], text_values[i], ha='center',
va='bottom')
plt.ylim([0,1600])
plt.xlabel("product")
plt.ylabel("average num_sold")
plt.show()
#sticker products of kaggle series have the highest average sales


# Define the data processing function and custom evaluation function (MAPE)
# Perform data feature processing (Here a function is defined to transform the date and split the information it contains)
def feature_processing(df):
    # Convert the date column, extract information from it
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # In the preliminary data exploration, 'country', 'store', 'product' are known to be string 'object' objects, and there are many duplicate values, convert them to categorical 'category' objects
    for col in ['country', 'store', 'product']:
        df[col] = df[col].astype('category')
    return df


# Define the function to remove missing values
def na_process(df):
    df = df.dropna()
    return df


# Define the MAPE evaluation metric and the MAPE-based score scorer (for subsequent CV parameter selection)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 10
# (The smaller the MAPE value, the better, so greater_is_better is selected as False)
custom_mape_scorer = make_scorer(mape, greater_is_better=False)

train = feature_processing(train)
train = na_process(train)
X = train.drop(columns=['id', 'date', 'num_sold'])
X = pd.get_dummies(X, drop_first=True)
y = np.log1p(train['num_sold'])

test = feature_processing(test)
X_test = test.drop(columns=['id', 'date'])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# CatBoost regression modeling
param_grid_catb = {
    'iterations': [300, 500, 800,1000],  # Different values for the number of trees
    'learning_rate': [0.05, 0.1, 0.2],  # Different values for the learning rate
    'depth': [4, 6, 8, 10]  # Different values for the tree depth
}
# Create a CatBoostRegressor model instance
model_catb = CatBoostRegressor(
    loss_function='MAPE',
    cat_features=[],
    random_seed=42,
    verbose=False
)  # cat_features indicates which features are categorical features. The model will perform One-hot encoding or more efficient processing on these categorical features during modeling
# Create a GridSearchCV object, specifying the model, parameter grid, number of cross-validation folds, etc.
kf_catb = KFold(n_splits=4, shuffle=True, random_state=42)
grid_search_catb = GridSearchCV(estimator=model_catb, param_grid=param_grid_catb, scoring=custom_mape_scorer, cv=kf_catb)
''' 
The grid parameters given in the catboost official documentation are as follows:
grid_search=model_catb.grid_search(
    param_grid_catb,
    X,
    y,
    cv=4,
    partition_random_seed=42,
    search_by_train_test_split=True,
    shuffle=True,
    train_size=0.8,
    verbose=True,
    plot=True
)
### https://catboost.ai/docs/en/concepts/python-reference_catboostregressor ###
### Usage provided on the official Catboost website ###

'''  

grid_search_catb.fit(X, y) 
print("Best parameter combination:", grid_search_catb.best_params_)
print("MAPE corresponding to the best parameter combination:", -grid_search_catb.best_score_)
results_catb = pd.DataFrame(grid_search_catb.cv_results_)
results_catb = results_catb.sort_values(by='mean_test_score', ascending=True).reset_index(drop=True)
print(results_catb)
results_catb.to_csv('grid_search_results_catb.csv', index=False)

best_params_catb = grid_search_catb.best_params_
best_model_catb = CatBoostRegressor(
    **best_params_catb,
    loss_function='MAPE',
    cat_features=[],
    random_seed=42,
    verbose=False
)
best_model_catb.fit(X, y)

y_pred_catb = best_model_catb.predict(X_test)
exp_y_pred_catb = np.expm1(y_pred_catb)
test['predicted_num_sold'] = exp_y_pred_catb
test[['id', 'predicted_num_sold']].to_csv('submission_catb.csv', index=False)

## Public Score:0.12235


# XGBoost regression modeling
param_grid_xgb = {
    'n_estimators': [300, 500, 800,1000],  # default 100
    'learning_rate':[0.05, 0.1, 0.3],  # default 0.3
    'max_depth': [4,6,8,10],  # default 3
    'gamma': [0.05,0.1,0.2],  # Regularization parameter that can range from 0 to infinity. A higher value indicates a higher strength of regularization and a lower likelihood of overfitting (but if it's too large, it may lead to underfitting)
    #"colsample_bytree":1 # Use a fraction of the total features or predictors to build the tree (default 1)
    #"subsample":1 # Use a proportion of the total number of training samples to build the tree (default 1)
    #"min_child_weight":1 # Minimum number of samples required for a child node (default 1). The min_child_weight parameter aims to regularize by limiting the depth of the tree. A higher value of this parameter reduces the likelihood of overfitting on the training data.
    # Details please refer to:  https://neptune.ai/blog/xgboost-vs-lightgbm
}
model_xgb = XGBRegressor(**param_grid_xgb, min_child_weight=10)
kf_xgb = KFold(n_splits=4, shuffle=True, random_state=42)
grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid_xgb, scoring=custom_mape_scorer, cv=kf_xgb)

grid_search_xgb.fit(X, y)
print("Best parameter combination:", grid_search_xgb.best_params_)
#pd.DataFrame(grid_search_xgb.cv_results_)[["mean_test_score","param_n_estimators","param_learning_rate","param_max_depth","param_gamma"]].sort_values(by="mean_test_score").reset_index(drop=True)
print("MAPE corresponding to the best parameter combination:", -grid_search_xgb.best_score_)
results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
results_xgb = results_xgb.sort_values(by='mean_test_score', ascending=True).reset_index(drop=True)
print(results_xgb)
results_xgb.to_csv('grid_search_results_xgb.csv', index=False)

best_params_xgb = grid_search_xgb.best_params_
best_model_xgb = XGBRegressor(
    **best_params_xgb,
    min_child_weight=10)
best_model_xgb.fit(X, y)

y_pred_xgb = best_model_xgb.predict(X_test)
exp_y_pred_xgb = np.expm1(y_pred_xgb)
test['predicted_num_sold'] = exp_y_pred_xgb
test[['id', 'predicted_num_sold']].to_csv('submission_xgb.csv', index=False)
## Public Score:0.12295


# LightGBM modeling
param_grid_lgbm={
    "learning_rate":[0.05,0.1,0.2],
    "n_estimators":[300,500,800,1000],
    "max_depth":[4,6,8,10]
}
model_lgbm=LGBMRegressor(
    objective='regression',
    metric='mape',
    random_state=42,
    n_jobs=4,
    verbosity=-1
    #n_estimators=600,
    #learning_rate=0.05,
    #max_depth=-1,
    #num_leaves=31,
    #min_child_samples=10,
)
kf_lgbm = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_lgbm= GridSearchCV(estimator=model_lgbm,param_grid=param_grid_lgbm,cv=kf_lgbm,scoring=custom_mape_scorer)
grid_search_lgbm.fit(X, y)
print("Best parameter combination:", grid_search_lgbm.best_params_)
print("MAPE corresponding to the best parameter combination:", -grid_search_lgbm.best_score_)
results_lgbm = pd.DataFrame(grid_search_lgbm.cv_results_)
results_lgbm = results_lgbm.sort_values(by='mean_test_score', ascending=True).reset_index(drop=True)
print(results_lgbm)
results_lgbm.to_csv('grid_search_results_lgbm.csv', index=False)

# Get the best hyperparameters from the grid search
best_params_lgbm = grid_search_lgbm.best_params_
# Initialize a LightGBM regressor model with the best hyperparameters
best_model_lgbm = LGBMRegressor(
    **best_params_lgbm,
    objective='regression',
    metric='mape',
    random_state=42,
    n_jobs=4,
    verbosity=-1
    # n_estimators=600,
    # learning_rate=0.05,
    # max_depth=-1,
    # num_leaves=31,
    # min_child_samples=10,
)

# Train the model with the best hyperparameters
best_model_lgbm.fit(X, y)

y_pred_lgbm = best_model_lgbm.predict(X_test)
exp_y_pred_lgbm = np.expm1(y_pred_lgbm)
test['predicted_num_sold'] = exp_y_pred_lgbm
test[['id', 'predicted_num_sold']].to_csv('submission_lgbm.csv', index=False)
#Public Score:0.11737

