# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# Load your dataset# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


test.head()


train.isnull().sum()


test.isnull().sum()


train.shape


test.shape


train.describe()


test.describe()


train.info()


test.info()


# Feature engineering: Remove 'id' and 'day' columns, handle missing values
X = train.drop(columns=['rainfall', 'id', 'day'])
y = train['rainfall']
X_test = test.drop(columns=['id', 'day'])


# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost Model Training


# Initialize XGBoost model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10, subsample=0.8)


# Convert data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


# Specify parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 10,
    'learning_rate': 0.01,
    'subsample': 0.8
}


# Train the model with early stopping
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50)



# Predictions
test_dmatrix = xgb.DMatrix(X_test)
test_predictions = model.predict(test_dmatrix)


# Create the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_predictions
})

submission.to_csv('submission1.csv', index=False)

