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


train_file = "/kaggle/input/playground-series-s5e4/train.csv"
test_file = "/kaggle/input/playground-series-s5e4/test.csv"
submit_file = "/kaggle/input/playground-series-s5e4/sample_submission.csv"


!pip install mambular -q


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
import math
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from catboost import CatBoostRegressor, Pool

from mambular.models import MambularRegressor


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


train_data.head()


train_data['Guest_Popularity_percentage'] = train_data['Guest_Popularity_percentage'].fillna(train_data['Guest_Popularity_percentage'].median())
train_data['Episode_Length_minutes'] = train_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].median())
mode_val = train_data['Number_of_Ads'].mode()[0]
train_data['Number_of_Ads'] = train_data['Number_of_Ads'].fillna(mode_val)
train_data['Host_Popularity_percentage'] = train_data['Host_Popularity_percentage'].apply(lambda x: 100 if x > 100 else x)
train_data['Guest_Popularity_percentage'] = train_data['Guest_Popularity_percentage'].apply(lambda x: 100 if x > 100 else x)
train_data = train_data[train_data['Number_of_Ads'] <  12]



cat_cols = train_data.select_dtypes(include='object').columns.to_list()
num_cols = train_data.select_dtypes(include=['int64','float64']).columns.to_list()
num_cols.remove('id')
cat_cols, num_cols



train_data[num_cols].describe()


for feature in num_cols:
    plt.figure(figsize=(12, 5))

    # Histogram with KDE (Kernel Density Estimate)
    plt.subplot(1, 2, 1)
    sns.histplot(train_data[feature], kde=True, bins=30, color='pink')
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Box plot to identify outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_data[feature], color='pink')
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()



box_cols = ['Episode_Sentiment', 'Publication_Time', 'Publication_Day', 'Genre' ]
median_values = train_data.groupby(box_cols)['Listening_Time_minutes'].median().sort_values()
sorted_categories = median_values.index

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten() # flatten axes for easier iteration

for i, col in enumerate(box_cols):
  sns.boxplot(data=train_data, x=col, y="Listening_Time_minutes", ax=axes[i], palette='Paired')
  axes[i].set_title(col)  # Set title for each subplot
plt.tight_layout()
plt.show()


plt.figure(figsize=[15,4])
sns.boxplot(data=train_data, x='Number_of_Ads', y='Listening_Time_minutes', palette='coolwarm')
plt.show()


train_data[num_cols].head()


for x in num_cols:
    plt.figure()
    sns.set_style('darkgrid')
    sns.scatterplot(data=train_data, x=x, y='Listening_Time_minutes', alpha=0.5, color='royalblue')
    plt.tight_layout()
    plt.show()


train_data['Episode_Number'] = train_data['Episode_Title'].apply(lambda x: x[-2:])
train_data['Episode_Number'].head()


plt.figure(figsize=[15,6])
sns.scatterplot(data=train_data, x='Episode_Number', y= 'Listening_Time_minutes')
plt.tight_layout()
plt.show()


train_data[cat_cols].head()


from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


le = LabelEncoder()

train_data['Genre'] = le.fit_transform(train_data['Genre'])

train_data['Publication_Day'] = le.fit_transform(train_data['Publication_Day'])

train_data['Publication_Time'] = le.fit_transform(train_data['Publication_Time'])

train_data['Episode_Sentiment'] = le.fit_transform(train_data['Episode_Sentiment'])


train_data.head()


from sklearn.model_selection import train_test_split


df_train, df_test = train_test_split(train_data, test_size=0.25, random_state=100)

df_train.shape, df_test.shape

# df_train.drop(columns=['id','Podcast_Name','Episode_Title','Episode_Number'], axis=1, inplace=True)
# df_test.drop(columns=['id','Podcast_Name','Episode_Title','Episode_Number'],  axis=1, inplace=True)

df_train.drop(columns=['id','Podcast_Name','Episode_Title'], axis=1, inplace=True)
df_test.drop(columns=['id','Podcast_Name','Episode_Title'],  axis=1, inplace=True)

y_train = df_train.pop('Listening_Time_minutes')
X_train = df_train

y_test = df_test.pop('Listening_Time_minutes')
X_test = df_test


df_train.head()


# scale_cols = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
scale_cols = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Episode_Number']

# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])

X_test[scale_cols] = scaler.transform(X_test[scale_cols])


X_train.shape, X_test.shape, y_train.shape, y_test.shape



target = "Listening_Time_minutes"

# Train a Gradient Boosted Trees model
model = MambularRegressor()




X_train.drop('Number_of_Ads', axis=1, inplace=True)


X_test.drop('Number_of_Ads', axis=1, inplace=True)


model.fit(X_train, y_train, max_epochs=150, lr=0.0001)


y_preds = model.predict(X_test)
y_preds


r2score = r2_score(y_preds,y_test)
mse = mean_squared_error(y_test, y_preds)
rmse = math.sqrt(mse)
print(f"r2 score is {r2score} and RMSE is {rmse}")





























'''


xgb_reg = xgb.XGBRegressor(n_jobs = -1)
xgb_reg.get_params()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)

r2score = r2_score(y_pred,y_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(f"r2 score is {r2score} and RMSE is {rmse}")

'''


'''
# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

param_distributions = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    },
    'XGBoost': {
        'n_estimators' : [100, 200, 500, 750],
        'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25],
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 12]
        }
}

'''


'''
for name, model in models.items():
    folds=3
    param_comb = 100
    
    if name in param_distributions:
        search = RandomizedSearchCV(model, param_distributions[name], n_iter=10, random_state=100, n_jobs=-1, cv=3, verbose=3)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"algo:{name} with {best_model}")
    else:
        model.fit(X_train, y_train)
        best_model = model
        print(f"algo:{name} with {best_model}")
    
    predictions = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    print(f'{name} MSE: {mse}')
'''


'''
model = XGBRegressor() 


%%time 

params = {
        'n_estimators' : [700],
        'learning_rate' : [0.05],
        'min_child_weight': [7],
        'gamma': [1.5],
        'subsample': [0.8],
        'colsample_bytree': [1.0],
        'max_depth': [12]
        }
xgb_hpt = RandomizedSearchCV(model, params, n_iter=10, n_jobs=-1, cv=3, verbose=3, random_state=100)
xgb_hpt.fit(X_train, y_train)
y_pred_hpt = xgb_hpt.predict(X_test)

r2score = r2_score(y_pred_hpt,y_test)
mse = mean_squared_error(y_test, y_pred_hpt)
rmse = math.sqrt(mse)
print(params)
print(f"r2 score is {r2score} and RMSE is {rmse}")

'''





'''
# Initialize CatBoostRegressor
model_cb = CatBoostRegressor(iterations=1000, 
                          learning_rate=0.05, 
                          depth=10,
                          eval_metric='RMSE',
                          random_seed = 100,
                          verbose=3,
                          loss_function='RMSE')



model_cb.fit(X_train, y_train, verbose=100)

# Make predictions
y_pred_cb = model_cb.predict(X_test)

r2score = r2_score(y_pred_cb,y_test)
mse = mean_squared_error(y_test, y_pred_cb)
rmse = math.sqrt(mse)
print(f"r2 score is {r2score} and RMSE is {rmse}")

'''


# Treatment of test data for final predictions

test_data_orig = test_data.copy()

test_data['Guest_Popularity_percentage'] = test_data['Guest_Popularity_percentage'].fillna(test_data['Guest_Popularity_percentage'].median())
test_data['Episode_Length_minutes'] = test_data['Episode_Length_minutes'].fillna(test_data['Episode_Length_minutes'].median())
mode_val = test_data['Number_of_Ads'].mode()[0]
test_data['Number_of_Ads'] = test_data['Number_of_Ads'].fillna(mode_val)

test_data['Host_Popularity_percentage'] = test_data['Host_Popularity_percentage'].apply(lambda x: 100 if x > 100 else x)
test_data['Guest_Popularity_percentage'] = test_data['Guest_Popularity_percentage'].apply(lambda x: 100 if x > 100 else x)
# test_data = test_data[test_data['Number_of_Ads'] <  12]

test_data['Episode_Number'] = test_data['Episode_Title'].apply(lambda x: x[-2:])

test_data['Genre'] = le.fit_transform(test_data['Genre'])

test_data['Publication_Day'] = le.fit_transform(test_data['Publication_Day'])

test_data['Publication_Time'] = le.fit_transform(test_data['Publication_Time'])

test_data['Episode_Sentiment'] = le.fit_transform(test_data['Episode_Sentiment'])

test_data.drop(columns=['id','Podcast_Name','Episode_Title'],  axis=1, inplace=True)

test_data[scale_cols] = scaler.transform(test_data[scale_cols])


# predict the output on test data

# test_pred = xgb_reg.predict(test_data) # prediction using XGB regular model
# test_pred = xgb_hpt.predict(test_data) # prediction using XGB with hyper paramter tuning
# test_pred = model_cb.predict(test_data)

test_pred = model.predict(test_data)


# make a dataframe for final output 

results_df = pd.DataFrame({
    'id': test_data_orig['id'],
    'Listening_Time_minutes' : test_pred.round(3)
})

results_df.head()


# conversion to csv file 

results_df.to_csv("rgb_prediction_result.csv", index=False)

