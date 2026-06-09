import pandas as pd

# ✅ Use the correct dataset folder path
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# Show the first few rows
train.head()




# Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# Target column
target = 'Lap_Time_Seconds'
y = train[target]
y_log = np.log1p(y)  # Log-transform target to reduce RMSE sensitivity

# Columns for Frequency Encoding (instead of dropping)
freq_cols = ['rider_name', 'team_name', 'bike_name', 'shortname', 'circuit_name', 'rider', 'team', 'bike', 'Session', 'weather', 'track']

# Frequency Encoding function
def frequency_encoding(df_train, df_test, cols):
    for col in cols:
        freq = df_train[col].value_counts() / len(df_train)
        df_train[col + "_freq"] = df_train[col].map(freq)
        df_test[col + "_freq"] = df_test[col].map(freq)
    return df_train, df_test

train, test = frequency_encoding(train, test, freq_cols)

# Drop original high-cardinality categorical columns + 'Unique ID'
cols_to_drop = ['Unique ID'] + freq_cols
X = train.drop(columns=[target] + cols_to_drop, errors='ignore')
X_test = test.drop(columns=cols_to_drop, errors='ignore')

# One-hot encoding remaining categorical columns
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align test with train columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Improved XGBoost hyperparameters (further tunable)
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method='hist'
)

# Train
model.fit(X_train, y_train)

# Predict on validation (inverse transform)
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)
y_val_actual = np.expm1(y_val)

# RMSE
rmse = mean_squared_error(y_val_actual, val_preds, squared=False)
print("Validation RMSE:", rmse)

# Predict on test set
test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log)

# Final submission
submission = sample.copy()
submission['Lap_Time_Seconds'] = test_preds
submission.to_csv('solution.csv', index=False)

print("Total predictions:", submission.shape[0])
submission.head(10)

