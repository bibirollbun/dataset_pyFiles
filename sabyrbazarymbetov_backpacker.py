import numpy as np
import pandas as pd

from catboost import Pool, CatBoostClassifier, CatBoostRegressor

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import mean_squared_error

import random

from scipy.stats import zscore

from eda_utility_library import categorize_columns, plot_pie_charts, violin_plots, missing_data_summary

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


ss = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


RMV = ['Price', 'id']
categorized_columns = categorize_columns(train, rmv=RMV)

FEATURES = list(set(train.columns) - set(RMV))



for col_type in categorized_columns.keys():
    if len(categorized_columns[col_type]) > 0:
        print(col_type)


DISCRETE = categorized_columns['discrete']
CONTINUOUS = categorized_columns['continuous']
CATEGORICAL = categorized_columns['categorical']


train = pd.concat([train, train_extra], axis=0)


def remove_duplicates(df):
    # Display the count of duplicate groups
    duplicate_groups = df.groupby(FEATURES).size().reset_index(name='Count')
    duplicate_groups = duplicate_groups[duplicate_groups['Count'] > 1]  # Keep only real duplicates
    
    # Merge with original data to see full duplicate details
    duplicate_entries = df.merge(duplicate_groups, on=FEATURES, how='inner')

    df = df[~df['id'].isin(duplicate_entries['id'])]
    return df

train = remove_duplicates(train)
test = remove_duplicates(test)


train['z_score'] = zscore(train['Price'])
train = train[train['z_score'].abs() < 3]  # Keep only values within 3 std deviations
train = train.drop(columns=['z_score'])  # Remove extra column


CATEGORICAL


# Impute Missing Values
train[CATEGORICAL] = train[CATEGORICAL].fillna('NAN')
test[CATEGORICAL] = test[CATEGORICAL].fillna('NAN')


# One Hot Encoding
ohe = OneHotEncoder(handle_unknown='error', sparse_output=False)

dummy = ohe.fit(train[CATEGORICAL])
OHE_COLUMNS = list(ohe.get_feature_names_out())

train[OHE_COLUMNS] = ohe.transform(train[CATEGORICAL])
test[OHE_COLUMNS] = ohe.transform(test[CATEGORICAL])


train.head()


DISCRETE


train['Compartments'].isna().sum()


CONTINUOUS


train['Weight Capacity (kg)'].isna().sum()


imp_mean = SimpleImputer(strategy='median')

col = 'Weight Capacity (kg)'
train[col] = imp_mean.fit_transform(train[col].values.reshape(-1, 1))
test[col] = imp_mean.transform(test[col].values.reshape(-1, 1))


FEATURES = OHE_COLUMNS + CONTINUOUS + DISCRETE

X_train, X_val, y_train, y_val = train_test_split(train[FEATURES], train['Price'], random_state=3, shuffle=True)


# Param Grid
# param_grid = {
#     'iterations': [500],
#     'learning_rate': [0.1, 0.5, 1.0],
#     'depth': [2, 3, 4],
#     'random_strength': [0, 0.5, 1],
#     'bagging_temperature': [0, 0.1, 0.5, 0.9],
# }
param_grid = {
    'iterations': [500],
    'learning_rate': [0.4, 0.5, 0.6],
    'depth': [3, 4, 5],
    'random_strength': [0.4, 0.5, 0.6],
    'bagging_temperature': [0]
}

best_model = None
best_score = float('inf')

for params in ParameterGrid(param_grid):
    model = CatBoostRegressor(**params, loss_function='RMSE')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
    
    val_score = model.best_score_['validation']['RMSE']
    
    if val_score < best_score:
        best_score = val_score
        best_model = model

print("Best parameters:", best_model.get_params())



preds = best_model.predict(X_val)

mse = mean_squared_error(preds, y_val)
rmse = mse**0.5
print(f'RMSE: {rmse}')


preds = best_model.predict(test[FEATURES])
ss['Price'] = preds

ss.to_csv('submission.csv', index=False)

