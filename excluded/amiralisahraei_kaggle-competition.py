!pip install optuna catboost


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xg
from lightgbm import LGBMRegressor
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_data.info()


# fill the missing values with the mean of the column
train_data['num_sold'] = train_data.groupby(['store', 'product','country'])['num_sold'].transform(lambda x: x.fillna(x.median()))
train_data['num_sold'] = train_data['num_sold'].fillna(train_data['num_sold'].mean())


def complete_feature(df):

    '''transform date feature and create new datetime features'''

    df = df.copy()

    # #change type of column
    df['date'] = pd.to_datetime(df['date'])

    #create new features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['per_month'] = df['date'].dt.to_period('M')
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday = 0, Sunday = 6
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].apply(lambda x: x in [5, 6])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)

    df['Season'] = df['date'].dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
    df['day_sin2'] = np.sin(4 * np.pi * df['day'] / 365.0)
    df['day_cos2'] = np.cos(4 * np.pi * df['day'] / 365.0)
    df['day_sin3'] = np.sin(6 * np.pi * df['day'] / 365.0)
    df['day_cos3'] = np.cos(6 * np.pi * df['day'] / 365.0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['month_sin2'] = np.sin(4 * np.pi * df['month'] / 12.0)
    df['month_cos2'] = np.cos(4 * np.pi * df['month'] / 12.0)
    df['sin_year'] = np.sin(2*np.pi*df['year']/365)
    df['cos_year'] = np.cos(2*np.pi*df['year']/365)
    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['day_country'] = df['day'].astype(str) + "_" + df['country']

    df['continents'] = df['country'].map({'Finland':1, 'Norway':1, 'Italy':1, 'Canada':2, 'Kenya':3, 'Singapore':4})
    df['parts'] = df['country'].map({'Finland':1, 'Norway':1, 'Italy':2, 'Canada':1, 'Kenya':3, 'Singapore':2})

    #drop columns
    # df.drop('date', axis=1, inplace=True)

    return df


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()

holidays = cal.holidays(start=train_data['date'].min(), end=train_data['date'].max())
train_data['is_holiday'] = train_data['date'].isin(holidays).astype(int)

holidays = cal.holidays(start=test_data['date'].min(), end=test_data['date'].max())
test_data['is_holiday'] = test_data['date'].isin(holidays).astype(int)


train_data = complete_feature(train_data)
test_data = complete_feature(test_data)


train_data.head()


train_data = train_data.drop('date', axis=1)


test_data = test_data.drop('date', axis=1)


test_data.head()


# Select all categorical columns
categorical = train_data.select_dtypes(include=['object', 'category']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each categorical column in the train and test data
for col in categorical:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])


train_data.head()


# find the outliers
sns.boxplot(train_data['num_sold'])
plt.show()


# remove the outliers

# Calculate Q1, Q3, and IQR
Q1 = train_data['num_sold'].quantile(0.25)
Q3 = train_data['num_sold'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3

# Filter the DataFrame to remove outliers
train_data = train_data[train_data['num_sold'] <= upper_bound]


sns.boxplot(train_data['num_sold'])
plt.show()


train_data.head()


train_data.head()


# Normalize the data

# num_sold_scaler = StandardScaler()
# train_data['num_sold'] = num_sold_scaler.fit_transform(train_data['num_sold'].values.reshape(-1, 1))

# scaler = StandardScaler()
# numerical_cols = ['year', 'month', 'day', 'day_of_week', 'quarter']
# train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
# test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])



train_data.head()


test_data.head()


input_features = [
    'country', 'store', 'product', 'is_holiday',
    'year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'day_of_year',
    'day_sin', 'day_cos', 'day_sin2', 'day_cos2', 'day_sin3', 'day_cos3',
    'month_sin', 'month_cos', 'month_sin2', 'month_cos2',
    'is_month_start', 'is_month_end', 'is_year_start', 'is_year_end',
    'month_country', 'day_country', 'sin_year', 'cos_year',
    'day_of_week_sin', 'day_of_week_cos', 'group', 'continents', 'Season'
]


X = train_data[input_features]
y = train_data['num_sold']
y = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Initialize and train the MLP model
mlp = MLPRegressor(max_iter=500, random_state=42, tol=0.1, hidden_layer_sizes=(256, 128, 64, 32))
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate MSE and MAPE
mse_value = mean_squared_error(y_test, y_pred)
mape_value = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse_value:.4f}')
print(f'Mean Absolute Percentage Error: {mape_value:.4f}')


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),  # Adjusted upper limit to 500
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Allow deeper trees
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),  # Lower min LR
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),  # Explore more subsampling
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),  # More variation
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # New parameter
        'gamma': trial.suggest_float('gamma', 0, 5),  # Regularization to avoid overfitting
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10),  # L2 Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10)  # L1 Regularization
    }

    model = xg.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape_value = mean_absolute_percentage_error(y_test, y_pred)  # Convert to percentage
    return mape_value


# Optimize with more trials
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Increase to 100 trials
best_params = study.best_params


best_params = {
    'n_estimators': 450,
    'max_depth': 12,
    'learning_rate': 0.023521702954537385,
    'subsample': 0.5869029071622219,
    'colsample_bytree': 0.9575936372384953,
    'min_child_weight': 1,
    'gamma': 4.10175128260193,
    'reg_lambda': 7.877217000376005,
    'reg_alpha': 1.9502474077769807}

# Initialize and train the XGBoost model
final_xgb_model = xg.XGBRegressor(**best_params)
final_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = final_xgb_model.predict(X_test)

# Evaluate MSE and MAPE
mse_value = mean_squared_error(y_test, y_pred)
mape_value = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse_value:.4f}')
print(f'Mean Absolute Percentage Error: {mape_value:.4f}')


lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
lgbm.fit(X_train, y_train)

# Make predictions
y_pred = lgbm.predict(X_test)

# Evaluate MSE and MAPE
mse_value = mean_squared_error(y_test, y_pred)
mape_value = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse_value:.4f}')
print(f'Mean Absolute Percentage Error: {mape_value:.4f}')


random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions
y_pred = random_forest.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mape_value = mean_absolute_percentage_error(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Percentage Error: {mape_value:.4f}')


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000, step=250),  # Number of boosting iterations
        'depth': trial.suggest_int('depth', 3, 16),  # Tree depth
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),  # Learning rate
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10),  # L2 regularization
        'border_count': trial.suggest_int('border_count', 32, 255),  # Number of bins for numeric features
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),  # Strength of Bayesian bagging
        'random_strength': trial.suggest_float('random_strength', 0, 10),  # Random noise level
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),  # Tree growth strategy
    }

    # Train CatBoost model with the selected hyperparameters
    model = CatBoostRegressor(**params, loss_function='MAPE', random_state=42, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=200)

    # Predict and evaluate MAPE
    y_pred = model.predict(X_test)
    mape_value = mean_absolute_percentage_error(y_test, y_pred)  # Convert to percentage
    return mape_value  # Optuna will try to minimize this


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
params = study.best_params


params = {
    'iterations': 1500,
    'depth': 14,
    'learning_rate': 0.1297620535868649,
    'l2_leaf_reg': 7.157558632036221,
    'border_count': 167,
    'bagging_temperature': 2.254474405608795,
    'random_strength': 2.4835213299953276,
    'grow_policy': 'SymmetricTree'
}


model = CatBoostRegressor(**params)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=200)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate MAPE and MSE
mape_value = mean_absolute_percentage_error(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)

# Print the evaluation metrics
print(f"MAPE: {mape_value:.4f}")
print(f"MSE: {mse_value:.4f}")


# Step 1: Define individual regression models
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = CatBoostRegressor(**params)

# Step 2: Create the VotingRegressor
voting_regressor = VotingRegressor(estimators=[
    ('rf', model1),
    ('cb', model2),
])

# Step 3: Train the VotingRegressor on the training data
voting_regressor.fit(X_train, y_train)

# Step 4: Predict on the test data
y_pred = voting_regressor.predict(X_test)

# Evaluate MAPE and MSE
mape_value = mean_absolute_percentage_error(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)

# Print the evaluation metrics
print(f"MAPE: {mape_value:.4f}")
print(f"MSE: {mse_value:.4f}")


test_prediction = model.predict(test_data[input_features])


test_prediction


np.expm1(test_prediction)


test_prediction.shape


# make a cv file includes the prediction and the id
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': test_prediction})
submission.to_csv('sample_submission.csv', index=False)




