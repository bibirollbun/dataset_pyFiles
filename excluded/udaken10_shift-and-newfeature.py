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
train_data


train_data.set_index('date', inplace=True)
train_data.index = pd.to_datetime(train_data.index)
train_data.head()

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
test_data


def saturation_vapor_pressure(T):
    return 6.112 * np.exp(17.67 * T / (T + 243.5))



def feature_cooking_1(df):


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

    df["saturation_index"] = df["humidity"] / (df["dewpoint"] + 1e-6)
    df["pressure_change"] = df["pressure"].diff().fillna(0)
    df["cloud_sun_ratio"] = df["cloud"] / (df["sunshine"] + 1e-6)
    df['3days_moving_ave_cloud'] = df['cloud'].rolling(3).mean()
    df['6das_moving_ave_cloud'] = df['cloud'].rolling(6).mean()

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)
    df['3days_moving_ave'] = df['humidity'].rolling(3).mean()
    df['6days_moving_ave'] = df['humidity'].rolling(6).mean()
    df['windspeed*humidity'] = df['windspeed'] * df['humidity']
    df['3days_winspe*humidty'] = df['windspeed*humidity'].rolling(3).mean()
    df['3days_moving_ave_windspeed'] = df['windspeed'].rolling(3).mean()
    df['6days_moving_ave_windspeed'] = df['windspeed'].rolling(6).mean()

    return df


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


feature_cooking_1(train_data)
feature_cooking_1(test_data)


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


from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Define features (X) and target (y)
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=6, shuffle=False,
                      # random_state=42
                      )

# Initialize lists to store predictions and scores
predictions = []
scores = []

# Iterate through folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize and train CatBoostClassifier
    model = CatBoostClassifier(
        iterations=10000,  # Adjust as needed
        learning_rate=0.1, # Adjust as needed
        eval_metric='AUC', # Evaluation metric
        loss_function='Logloss', # Loss function for binary classification
        random_seed=42, # for reproducibility
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=100)

    # Predict probabilities for the validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    scores.append(roc_auc)
    print(f"Fold {fold + 1} ROC AUC: {roc_auc}")

    #Predict on test data
    predictions.append(model.predict_proba(test_data)[:,1])

print(f"Mean ROC AUC across folds: {np.mean(scores)}")

# Average the predictions from each fold
final_predictions = np.mean(predictions, axis=0)

sub['rainfall'] = final_predictions
sub.to_csv('submission.csv', index = False)
sub


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV

# Define features (X) and target (y)
X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost Pool objects
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# Define the parameter grid for grid search
param_grid = {
    'iterations': [100, 200, 500, 1000, 1500],  # Example values, adjust as needed
    'learning_rate': [0.01,0.05, 0.1],  # Example values, adjust as needed
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
predictions = best_model.predict(test_data)


sub['rainfall'] = predictions
# sub.to_csv('submission.csv', index = False)
sub




