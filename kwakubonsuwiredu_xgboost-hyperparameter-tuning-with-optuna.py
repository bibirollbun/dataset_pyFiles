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


!pip install xgboost
!pip install optuna
import pandas as pd
import numpy as np
from supplemental_english import *


###########################################################  FUNCTIONS  ################################################################################
# Symmetrical Mean Absolute Percentage Error
def SMAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    return np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100


# Get letters in plate
def letters(plate):
    return_list = []
    for char in plate:
        if char.isalpha():
            return_list.append(char)
    return ''.join(return_list)


# Get numbers in plate
def numbers(plate):
    return_list = []
    for char in plate:
        if char.isdigit():
            return_list.append(char)
    return ''.join(return_list[:3])


# Get region code
def reg_code(plate):
    return_list = []
    for x in plate[::-1]:
        if x.isdigit():
            return_list.append(x)
        elif not x.isdigit():
            break
    return ''.join(return_list[::-1])


# Get region name
def reg_name(region_code):
    search_list = list(REGION_CODES.values())
    ret_list = list(REGION_CODES.keys())
    for index in range(len(search_list)):
        if region_code in search_list[index]:
            return ret_list[index]


# Describe number plate
def describe_plate(letters: str, number: int, region: str, gov_codes: dict):
    """
    Classify a Russian number plate using already extracted components.

    Args:
        letters (str): The 3-letter string from the plate (e.g., 'PTT').
        number (int): The 3-digit number (e.g., 700).
        region (str): Region code as a string (e.g., '790').
        gov_codes (dict): The GOVERNMENT_CODES dictionary.

    Returns:
        dict: {
            "description": str,
            "forbidden_to_buy": int,
            "road_advantage": int,
            "significance_level": int
        }
    """
    # Normalize region (remove leading zeros)
    region = str(int(region))

    number = int(number)

    for (code_letters, (start, end), code_region), (desc, forbidden, advantage, significance) in gov_codes.items():
        if code_letters == letters and start <= number <= end and code_region == region:
            return {
                "description": desc,
                "forbidden_to_buy": forbidden,
                "road_advantage": advantage,
                "significance_level": significance
            }

    return {
        "description": "Regular plate",
        "forbidden_to_buy": 0,
        "road_advantage": 0,
        "significance_level": 0
    }




data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')


########################################################################################################################################################


data['numbers'] = [numbers(x) for x in data['plate']]
data['letters'] = [letters(x) for x in data['plate']]
data['region_code'] = [reg_code(x) for x in data['plate']]
data['region_name'] = [reg_name(x) for x in data['region_code']]
data['description'] = [describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['description']
                       for plate in data['plate']]
data['forbidden_to_buy'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['forbidden_to_buy'] for plate in
    data['plate']]
data['road_advantage'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['road_advantage'] for plate in
    data['plate']]
data['significance'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['significance_level'] for plate in
    data['plate']]

# Creating a numerical column for region_name
region_mapping = {region: index for index, region in enumerate(REGION_CODES)}
# Creating a mapping for the alphabets
all = 'abcdefghijklmnopqrstuvwxyz'
alphabet = {all: index for index, all in enumerate(all)}

# Assing numbers to region names
data['region_name_num'] = [region_mapping[x] for x in data['region_name']]

# Assing numbers to letters of plate
data['letter1'] = [alphabet[x[0].lower()] for x in data['letters']]
data['letter2'] = [alphabet[x[1].lower()] for x in data['letters']]
data['letter3'] = [alphabet[x[2].lower()] for x in data['letters']]

# Split numerical column of number plate
data['number1'] = [x[0] for x in data['numbers']]
data['number2'] = [x[1] for x in data['numbers']]
data['number3'] = [x[2] for x in data['numbers']]

# Change date column to datetime
data['date'] = pd.to_datetime(data['date'])

data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

description_dict = {all: index for index, all in enumerate(list(data.description.unique()))}
data['desc_int'] = [description_dict[x] for x in data['description']]

print(data[data['region_code'] == '50'], '\n', data.columns)

features = ['letter1', 'letter2', 'letter3', 'number1', 'number2', 'number3', 'region_name_num', 'year', 'month',
            'day', 'significance', 'road_advantage', 'forbidden_to_buy', 'desc_int']  # X
target = ['price']  # y


X = data[features].astype(int)
y = np.ravel(data[['price']])


####################################### MODEL #################################################
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



### HYPER PARAMETER TUNING WITH OPTUNA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import optuna

def observation(trial, X_train, y_train):
    booster = trial.suggest_categorical('booster',['gbtree', 'gblinear'])
    if booster == 'gbtree':
        eta = trial.suggest_float('eta', 0.1, 1)
        gamma = trial.suggest_float('gamma', 0, 1)
        min_child_weight = trial.suggest_float('min_child_weight', 0, 1)
        grow_policy = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        tree_method = trial.suggest_categorical('tree_method', ['auto','exact','approx'])
        objective = trial.suggest_categorical('objective',["reg:squarederror","reg:squaredlogerror","reg:pseudohubererror","reg:absoluteerror","reg:gamma","reg:tweedie",])
        eval_metric = trial.suggest_categorical('eval_metric', ['rmse', 'mae', 'mape', 'logloss'])
        
        model = xgb.XGBRegressor(eta = eta, gamma = gamma, min_child_weight = min_child_weight, grow_policy=grow_policy, tree_method=tree_method, objective = objective, eval_metric = eval_metric)
        score = cross_val_score(model, X_train, y_train, cv = 5, scoring=make_scorer(SMAPE, greater_is_better=False)).mean()
        return abs(score)

    else:
        eta = trial.suggest_float('eta', 0.01, 1)
        objective = trial.suggest_categorical('objective',["reg:squarederror","reg:squaredlogerror","reg:pseudohubererror","reg:absoluteerror","reg:gamma","reg:tweedie",])
        eval_metric = trial.suggest_categorical('eval_metric', ['rmse', 'mae', 'mape', 'logloss'])
        alpha = trial.suggest_float('alpha',0,10)
        
        model = xgb.XGBRegressor(eta = eta, objective = objective,alpha=alpha, eval_metric=eval_metric)
        score = cross_val_score(model, X_train, y_train, cv = 5, scoring=make_scorer(SMAPE, greater_is_better=False)).mean()

        return abs(score)

study = optuna.create_study()
study.optimize(lambda trial: observation(trial, X_train, y_train), n_trials=50)

print('Best Prarmerters: ',study.best_params,'\nBest Study: ',study.best_value)


test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


########################################################################################################################################################


test['numbers'] = [numbers(x) for x in test['plate']]
test['letters'] = [letters(x) for x in test['plate']]
test['region_code'] = [reg_code(x) for x in test['plate']]
test['region_name'] = [reg_name(x) for x in test['region_code']]
test['description'] = [describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['description']
                       for plate in test['plate']]
test['forbidden_to_buy'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['forbidden_to_buy'] for plate in
    test['plate']]
test['road_advantage'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['road_advantage'] for plate in
    test['plate']]
test['significance'] = [
    describe_plate(letters(plate), numbers(plate), reg_code(plate), GOVERNMENT_CODES)['significance_level'] for plate in
    test['plate']]

# Creating a numerical column for region_name
region_mapping = {region: index for index, region in enumerate(REGION_CODES)}
# Creating a mapping for the alphabets
all = 'abcdefghijklmnopqrstuvwxyz'
alphabet = {all: index for index, all in enumerate(all)}

# Assing numbers to region names
test['region_name_num'] = [region_mapping[x] for x in test['region_name']]

# Assing numbers to letters of plate
test['letter1'] = [alphabet[x[0].lower()] for x in test['letters']]
test['letter2'] = [alphabet[x[1].lower()] for x in test['letters']]
test['letter3'] = [alphabet[x[2].lower()] for x in test['letters']]

# Split numerical column of number plate
test['number1'] = [x[0] for x in test['numbers']]
test['number2'] = [x[1] for x in test['numbers']]
test['number3'] = [x[2] for x in test['numbers']]

# Change date column to datetime
test['date'] = pd.to_datetime(test['date'])

test['day'] = test['date'].dt.day
test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year

description_dict = {all: index for index, all in enumerate(list(test.description.unique()))}
test['desc_int'] = [description_dict[x] for x in test['description']]

print(test[test['region_code'] == '50'], '\n', test.columns)

features = ['letter1', 'letter2', 'letter3', 'number1', 'number2', 'number3', 'region_name_num', 'year', 'month',
            'day', 'significance', 'road_advantage', 'forbidden_to_buy', 'desc_int']  # X

test[features]


model = xgb.XGBRegressor(**study.best_params)
model.fit(X,y)



test[features] = test[features].astype(int)
prediction = model.predict(test[features])
submission = pd.DataFrame()
submission['id'] = test['id']
submission['price'] = prediction

submission.to_csv('submission.csv', index=False)

