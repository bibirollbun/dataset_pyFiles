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
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')

train.head()


# Getting the shape of the dataset
train.shape


# Checking the datatypes
train.info()


# Checking for missing, unique, and duplicated data

def check(data):
    l=[]
    columns=data.columns
    for i in columns:
        dtypes = data[i].dtypes
        nunique = data[i].nunique()
        sum_null=data[i].isnull().sum()
        l.append([i, dtypes, nunique, sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['column', 'dtypes', 'unique', 'sum_null']
    return df_check

check(train)


# Looking at our values

columns=train.columns
for col in columns:
    print(f"Unique values in '{col}'':")
    print("*" * 50)
    print(train[col].unique())
    print()


# Checking our summary statistics

train.describe().T


# Creating the hist_boxplot function

def hist_box(data, col):
    f,(ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (0.15, 0.85)}, figsize = (12,6))
    sns.boxplot(data=data, x=col, ax=ax_box, showmeans=True)
    sns.histplot(data=data, x=col, kde=True, ax=ax_hist)
    plt.show()


for col in train.columns:
    print(col)
    hist_box(train, col)


# Checking for correlations between features

plt.figure(figsize=(20,10))
sns.heatmap(train.corr(), annot=True, fmt='.1f', cmap='viridis')


from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor



train_features = train.drop(['FloodProbability'], axis=1)
train_target = train['FloodProbability']


# Add an intercept term

train_features = sm.add_constant(train_features)

# Building the model on the OLS algorithm

ols_model = sm.OLS(train_target,train_features)

# Fitting the model

ols_result = ols_model.fit()

print(ols_result.summary())


# Evaluating the Variance Inflation Factor

vif_series = pd.Series(
    [variance_inflation_factor(train_features.values, i) for i in range(train_features.shape[1])],
    index=train_features.columns,
    dtype=float
)

print("VIF Scores: \n\n{}\n".format(vif_series))


# Checking for mean of residuals
residual = ols_result.resid
residual.mean()


# Checking normality of error

sns.histplot(residual, kde=True)


# checking the linearity of variables

fitted = ols_result.fittedvalues

sns.residplot(x=fitted, y=residual, color='red')
plt.xlabel('Fitted Values')
plt.ylabel('Residual')
plt.title("Residual Plot")
plt.show()


# Log Transformation of target variable

train_target_log = np.log(train_target)


ols_model_log = sm.OLS(train_target_log, train_features)

ols_result_log = ols_model_log.fit()

print(ols_result_log.summary())


# checking the linearity of variables

fitted = ols_result_log.fittedvalues

sns.residplot(x=fitted, y=residual, color='red')
plt.xlabel('Fitted Values')
plt.ylabel('Residual')
plt.title("Residual Plot")
plt.show()


# Square Root Transformation of target variable

train_target_sqrt = np.sqrt(train_target)


ols_model_sqrt = sm.OLS(train_target_sqrt, train_features)

ols_result_sqrt = ols_model_sqrt.fit()

# checking the linearity of variables

fitted = ols_result_sqrt.fittedvalues

sns.residplot(x=fitted, y=residual, color='red')
plt.xlabel('Fitted Values')
plt.ylabel('Residual')
plt.title("Residual Plot")
plt.show()



# Using the Goldfeld-Quandt Test for homoscedasticity

from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

name = ['F Statistics', 'p-value']
test = sms.het_goldfeldquandt(train_target_sqrt, train_features)
lzip(name, test)


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from lightgbm import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree


# Splitting the data
X = train_features
y = train_target

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


dt= DecisionTreeRegressor(max_depth=5, min_samples_split = 10, min_samples_leaf = 5, random_state=42)
dt.fit(X_train,y_train)


# Predicting with our model
y_pred = dt.predict(X_test)

# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(f"RMSE: {rmse: .2f}")
print(f"R2 Score: {r2: .4f}")


# visualizing the tree

plt.figure(figsize = (20,10))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True)
plt.show()


# Tuning the tree using GridSearch

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3,5,10,15],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,5,10]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)

print("Best params:", grid_search.best_params_)


# Building our Tuned Tree
dt_tuned= DecisionTreeRegressor(max_depth=15, min_samples_split = 2, min_samples_leaf = 10, random_state=42)
dt_tuned.fit(X_train,y_train)


# Predicting with our model
y_pred = dt_tuned.predict(X_test)

# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(f"RMSE: {rmse: .2f}")
print(f"R2 Score: {r2: .4f}")


# Light GBM model construction

lgbm_params = {
    'boosting_type': 'gbdt', 
    'n_estimators':1500, 
    'learning_rate' :  0.012,    
    'num_leaves' : 250, 
    'subsample_for_bin': 165700, 
    'min_child_samples': 114, 
    'reg_alpha': 2.075e-06, 
    'reg_lambda': 3.839e-07, 
    'colsample_bytree': 0.9634,
    'subsample': 0.9592, 
    'max_depth': 10,
    'random_state':0,
    'verbosity':-1}

lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train,y_train)


# Predicting with the LGBM model

y_pred = lgbm_model.predict(X_test)


# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(f"RMSE: {rmse: .2f}")
print(f"R2 Score: {r2: .4f}")



# Evaluating our feature importances

feature_importances = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": lgbm_model.feature_importances_
})

# Sorting our features
feature_importances = feature_importances.sort_values(by="Importance", ascending = False)

# Plotting the importances
plt.figure(figsize=(20,10))
sns.barplot(x="Importance", y="Feature", data=feature_importances, palette = 'colorblind')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("LGBM Feature Importances")
plt.show()


df_test.shape
df_test = sm.add_constant(df_test)


df_test.head()


y_sub =  lgbm_model.predict(df_test)


print (y_sub)


# converting to a dataframe

submission =  pd.DataFrame({'id': df_test['id'], 'predictions': y_sub})

submission.to_csv('submission.csv', index=False)


submission.head()

