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
import datetime
from datetime import datetime
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

train['expected_day'] = (train.index.values) % 365 + 1

train['day'] = train['expected_day']
train.drop('expected_day', axis=1, inplace=True)

train_1 = train[0:365]
train_2 = train[365:730]
train_3 = train[730:1095]
train_4 = train[1095:1460]
train_5 = train[1460:1825]
train_6 = train[1825:2190]

train_1['day_of_year'] = train_1['day']
train_1

from datetime import datetime

train_1['date'] = train_1['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['day_of_year'] = train_2['day']
train_3['day_of_year'] = train_3['day']
train_4['day_of_year'] = train_4['day']
train_5['day_of_year'] = train_5['day']
train_6['day_of_year'] = train_6['day']
train_2['date'] = train_2['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_3['date'] = train_3['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_4['date'] = train_4['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_5['date'] = train_5['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_6['date'] = train_6['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['date'] = train_2['date'].apply(lambda x: x.replace(year=x.year + 1))
train_3['date'] = train_3['date'].apply(lambda x: x.replace(year=x.year + 2))
train_4['date'] = train_4['date'].apply(lambda x: x.replace(year=x.year + 3))
train_5['date'] = train_5['date'].apply(lambda x: x.replace(year=x.year + 4))
train_6['date'] = train_6['date'].apply(lambda x: x.replace(year=x.year + 5))

train_data = pd.concat([train_1, train_2, train_3, train_4, train_5, train_6])

train_data.set_index('date', inplace=True)
train_data.index = pd.to_datetime(train_data.index)

target = train_data.rainfall


test_1 = test[0:365]
test_2 = test[365:730]

test_1['date'] = test_1['day'].apply(lambda x: datetime.strptime(str(x), '%j'))
test_2['date'] = test_2['day'].apply(lambda x: datetime.strptime(str(x), '%j'))

test_1['date'] = test_1['date'].apply(lambda x: x.replace(year=x.year + 6))
test_2['date'] = test_2['date'].apply(lambda x: x.replace(year=x.year + 7))

test_data = pd.concat([test_1, test_2])

test_data.set_index('date', inplace=True)

test_data.index = pd.to_datetime(test_data.index)

test_data['day_of_year'] = test_data['day']


def saturation_vapor_pressure(T):
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def feature_cooking(df):


    # LCL高さ（Lifting Condensation Level Height）
    df['LCL_height'] = (125 * (df['temparature'] - df['dewpoint'])) / (df['temparature'] - 17.78)

    # 湿球温度（Wet Bulb Temperature）
    df['wet_bulb_temperature'] = df['temparature'] - ((df['temparature'] - df['dewpoint']) * (1 - df['humidity']/100))

    # 季節性指標（日付の正弦と余弦）
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # 温度範囲
    df['temperature_range'] = df['maxtemp'] - df['mintemp']

    # 温度-露点差
    df['temperature_dewpoint_diff'] = df['temparature'] - df['dewpoint']

    # 比湿（Specific Humidity）
    df['e_s'] = saturation_vapor_pressure(df['temparature'])
    df['specific_humidity'] = (0.622 * (df['humidity']/100 * df['e_s'])) / (df['pressure'] - (1 - 0.622) * (df['humidity']/100 * df['e_s']))

    # 風向成分（正弦と余弦）
    df['wind_direction_sin'] = np.sin(df['winddirection'] * np.pi / 180)
    df['wind_direction_cos'] = np.cos(df['winddirection'] * np.pi / 180)

    # 風速の2乗
    df['windspeed_squared'] = df['windspeed'] ** 2

    return df


feature_cooking(train_data)
feature_cooking(test_data)


def cook(df):
    for c in ['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']:
        for gap in [1]:
            df[c+f"_shift{gap}"]=df[c].shift(gap)
            df[c+f"_diff{gap}"]=df[c].diff(gap)
    return df
cook(train_data)
cook(test_data)


train_data.fillna(train_data.median(), inplace = True)
test_data.fillna(test_data.median(), inplace = True)

train_data


from sklearn.model_selection import train_test_split

X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,shuffle = False, random_state=0)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# Define the parameter grid for LogisticRegression
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'] # Solvers compatible with L1 penalty
}

# Initialize LogisticRegression
logreg_model = LogisticRegression(random_state=42, max_iter=1000)


# Perform GridSearchCV
grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Train the best model on the whole training data
best_logreg_model = grid_search.best_estimator_
best_logreg_model.fit(X, y)

# Make predictions on the validation set
y_pred_prob = best_logreg_model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

# Evaluate the model
roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f"ROC AUC on validation set: {roc_auc}")


# Predict probabilities on the test data using the best model
predictions = best_logreg_model.predict_proba(test_data)[:, 1]


sub['rainfall'] = predictions
# sub.to_csv('submission.csv', index = False)
sub


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost Pool objects
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# Define the parameter grid for grid search
param_grid = {
    'iterations': [100, 200],  # Example values, adjust as needed
    'learning_rate': [0.01, 0.1],  # Example values, adjust as needed
    'depth': [4, 6, 8],  # Example values, adjust as needed
}

# Initialize the CatBoost classifier
model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=100)


# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc') # using roc_auc as evaluation metric
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(train_pool, eval_set=val_pool)

# Make predictions on the test set
# predictions = best_model.predict(test_data)

# Make predictions on the validation set
y_pred_prob = best_model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

# Evaluate the model
roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f"ROC AUC on validation set: {roc_auc}")


# Make predictions on the test set
predictions = best_model.predict(test_data)



sub['rainfall'] = predictions
# sub.to_csv('submission.csv', index = False)
sub



# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'] # Solver for logistic regression
}

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter

# Perform GridSearchCV with roc_auc scoring
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc') # using roc_auc as evaluation metric
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_prob = best_model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

# Evaluate the model
roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f"ROC AUC on validation set: {roc_auc}")

# Predict probabilities on the test data using the best model
predictions = best_logreg_model.predict_proba(test_data)[:, 1]



sub['rainfall'] = predictions
sub.to_csv('submission.csv', index = False)
sub

