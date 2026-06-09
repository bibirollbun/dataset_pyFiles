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


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


train_file = "/kaggle/input/playground-series-s5e5/train.csv"
test_file = "/kaggle/input/playground-series-s5e5/test.csv"


train_data = pd.read_csv(train_file)
train_data.info()


train_data.head()


# Segregate columns

num_cols = train_data.select_dtypes(['int64','float64']).columns.to_list()

target = 'Calories'

cat_cols = train_data.select_dtypes('object').columns.to_list()

num_cols.remove(target)
num_cols.remove('id')


num_cols


# EDA

plt.figure()
sns.set_style('darkgrid')
sns.countplot(data=train_data, x='Sex', width = 0.4)
plt.tight_layout()
plt.show()


# EDA - Univariate Analysis 

plt.figure()
fig, axes = plt.subplots(3, 2, figsize=[15,10])
sns.set_style('darkgrid')
for i, col in enumerate(num_cols):
    sns.histplot(data=train_data, x=col, bins=30, kde=True, ax=axes[i//2,i%2], color='pink')
    axes[i//2,i%2].set_title(col)
    plt.tight_layout()
plt.show()

    


plt.figure()
fig, axes = plt.subplots(3, 2, figsize=[15,10])
sns.set_style('darkgrid')
# sns.color_palette("dark", 8)
for i, col in enumerate(num_cols):
    sns.boxplot(data=train_data, x='Sex', y=col, ax=axes[i//2,i%2])
    axes[i//2,i%2].set_title(col)
    plt.tight_layout()
plt.show()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Weight', x='Duration', hue='Calories')
plt.show()


train_data.head()


plt.figure(figsize=[8,8])
sns.heatmap(data=train_data[num_cols].corr(), cmap='coolwarm', annot=True, fmt='0.2f')
plt.show()


train_data.head()


train_data.describe()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Heart_Rate', x='Duration', hue='Calories')
plt.show()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Body_Temp', x='Duration', hue='Calories')
plt.show()


train_data.info()


import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



le = LabelEncoder()


train_data['Sex'] = le.fit_transform(train_data['Sex'])


train_data.info()


train_data.drop(columns=['id'], inplace=True)


train_data[num_cols].describe()


for col in num_cols:
    plt.figure(figsize=[7,4])
    sns.boxplot(data=train_data, y=col)
    plt.show()


train_data[num_cols].head()


df_train, df_test = train_test_split(train_data, test_size=0.20, random_state=101)


y_train = df_train.pop('Calories')
X_train = df_train


y_test = df_test.pop('Calories')
X_test = df_test


scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])


X_test[num_cols] = scaler.transform(X_test[num_cols])


df_train.head()


df_test.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# model building 

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    # 'Random Forest': RandomForestRegressor(n_estimators=100),
    'XGBoost' : XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'LightGBM' : LGBMRegressor(),
    'Catboost' : CatBoostRegressor(verbose=0)
}


metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0.1, a_max=None)
    msle = mean_squared_log_error(y_test, y_pred)
    rmsle = np.sqrt(msle)
    print(f"model:{name}, rmsle:{rmsle}")
    metrics[name] = rmsle


# hyperparameters tuning

from sklearn.model_selection import RandomizedSearchCV 

# hpt_model = XGBRegressor(random_state=42)
hpt_model = CatBoostRegressor(verbose=0)

'''
XGB_params = {
        'n_estimators' : [100, 200, 500, 700, 800, 900],
        'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25], 
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 5],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 12]
        }


cb_params = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'iterations': [100, 200, 500],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.1, 0.2, 0.5, 1.0],
    'border_count': [32, 64, 128]
}
'''

cb_params = {'learning_rate': [0.1], 
             'l2_leaf_reg': [9], 
             'iterations': [500], 
             'depth': [10], 
             'border_count': [64], 
             'bagging_temperature': [0.1]
             }
folds = 3

param_comb = 100

grid_search = RandomizedSearchCV(
    estimator=hpt_model ,
    param_distributions=cb_params,
    scoring='neg_mean_squared_error', 
    n_jobs=-1,
    n_iter=param_comb, 
    cv=5, 
    verbose=3, 
    random_state=42
)

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
y_pred = np.clip(y_pred, a_min=0.1, a_max=None)
msle = mean_squared_log_error(y_test, y_pred)
rmsle = np.sqrt(msle)
print(f"Catboost rmsle:{rmsle}")


# treatment of test data

test_data = pd.read_csv(test_file)
df = test_data.copy()
df['Sex'] = le.fit_transform(df['Sex'])
df.drop(columns=['id'], inplace=True)


df[num_cols] = scaler.transform(df[num_cols])


# Predictions on test data

predictions = grid_search.predict(df)


results_df = pd.DataFrame(
    {
        'id' : test_data['id'],
        'Calories' : predictions
    }
)


# create submission file

results_df.to_csv('calorie_expend_submission.csv', index=False)

