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

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import warnings 
warnings.filterwarnings('ignore')
# Disable LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger('lightgbm').setLevel(logging.INFO)
logging.getLogger('lightgbm').setLevel(logging.ERROR)


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/training_extra.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e2/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("data shape :",data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.isna().sum().sort_values(ascending=False)


data = data.dropna()
#data.shape
#train_data = pd.concat([train_data, data], ignore_index=True)
train_data = train_data.drop("id", axis=1)
train_data = train_data.drop_duplicates()
print("shape of the data :",train_data.shape)


train_data.head(2)


#train_data = train_data.drop('id', axis = 1)
num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['Price']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


len(cat_cols), len(cat_cols_test)


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object']).columns] = train_data.select_dtypes(include=['object']).apply(lambda x: x.fillna("missing"))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object']).columns] = test_data.select_dtypes(include=['object']).apply(lambda x: x.fillna('missing'))


import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def anova_correlation(train_data, cat_cols, target_col):
    results = {}
    for col in cat_cols:
        groups = [train_data[target_col][train_data[col] == category] for category in train_data[col].unique()]
        f_stat, p_value = f_oneway(*groups)
        results[col] = f_stat  # Higher F-statistic means stronger relation
    return results

# Identify categorical columns
#categorical_cols = ['cat_col1', 'cat_col2']  # Replace with your categorical column names
anova_results = anova_correlation(train_data, cat_cols, 'Price')

# Display sorted results
sorted(anova_results.items(), key=lambda x: x[1], reverse=True)

#Interpretation: Higher F-values mean stronger correlation between the categorical column and the numerical target.


from scipy.stats import pearsonr

def mean_encoding_correlation(train_data, cat_cols, target_col):
    correlations = {}
    for col in cat_cols:
        mean_encoded = train_data.groupby(col)[target_col].transform('mean')
        correlation, _ = pearsonr(mean_encoded, train_data[target_col])
        correlations[col] = correlation
    return correlations

mean_corr_results = mean_encoding_correlation(train_data, cat_cols, 'Price')

# Display sorted results
sorted(mean_corr_results.items(), key=lambda x: abs(x[1]), reverse=True)

#Interpretation:
#1. A high absolute correlation value (close to 1 or -1) indicates a strong relationship.
#2. Values closer to 0 suggest little or no correlation.


train_data['Compartments'] = train_data['Compartments'].astype('object')
test_data['Compartments'] = test_data['Compartments'].astype('object')

train_data['Weight Capacity (kg)'] = train_data['Weight Capacity (kg)'].astype('object')
test_data['Weight Capacity (kg)'] = test_data['Weight Capacity (kg)'].astype('object')


X = train_data.drop(['Price'], axis=1)
y = train_data['Price']
test = test_data.drop(['id'],axis=1)


parameters2 = {'iterations': 900, 'depth': 5, 'learning_rate': 0.05515348558143167, 'l2_leaf_reg': 0.19078568306082053, 'border_count': 177, 'random_strength': 2.484310649640042, 'bagging_temperature': 8.33889442063459}
parameters3 = {'iterations': 600, 'depth': 6, 'learning_rate': 0.17296991523031371, 'l2_leaf_reg': 0.0017458887986447312, 'border_count': 118, 'random_strength': 7.959386361317887, 'bagging_temperature': 5.650368573511559}
#value: 38.64634062502156.


from catboost import CatBoostRegressor

# Ensure all categorical features are strings
for col in X.columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
cat_features = X_train.columns.tolist()  # All columns are categorical

# Initialize CatBoost Regressor
model = CatBoostRegressor(**parameters3, loss_function="RMSE", cat_features=cat_features, verbose=0)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True)

# Predictions
y_pred = model.predict(X_test)
cat_pred = model.predict(test)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

sub = pd.DataFrame({'id': test_data.id, 'Price': cat_pred})
sub.to_csv("submission.csv", index=False)
print(sub.head())

