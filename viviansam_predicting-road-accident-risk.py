import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import library
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df.info()


df.head()


# Drop the 'ID' column
df = df.drop(columns=['id'])


# Check unique value of categorical variables
# List of categorical variables
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

# Check unique values for each categorical variable
for feature in categorical_features:
    unique_features = df[feature].unique()
    print(f"Unique values for {feature}: {unique_features}")


# Check the range of numeric variables
# List of numeric variables
numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
 
# Check min max for each numeric variable
for feature in numeric_features:
    min_feature = df[feature].min()
    max_feature = df[feature].max()
    print(f"{feature}: Min: {min_feature}, Max: {max_feature}")


# One-hot encoding for categorical variables
col_onehot = ['road_type', 'lighting', 'weather', 'time_of_day']

# perform one-hot encoding
df = pd.get_dummies(df, columns=col_onehot)
df.info()


# Data splitting
X = df.drop(columns=['accident_risk']) # features
y = df['accident_risk'] # target variable

# Split into at 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# LR
# Initialize model
lin_reg = LinearRegression()

# Train model
lin_reg.fit(X_train, y_train)

# Predict on test set
y_pred = lin_reg.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# RF
# Initialize model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# XGB
# Initialize model
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

# Train model
xgb.fit(X_train, y_train)

# Predict on test set
y_pred = xgb.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' column from df_test
df_test = df_test.drop(columns=['id'])


# perform one-hot encoding
df_test = pd.get_dummies(df_test, columns=col_onehot)


# Predict
y_test = xgb.predict(df_test)


# Create a DataFrame with 'ID' and 'accident_risk' columns
output = pd.DataFrame({'id': id_test, 'accident_risk': y_test})
output.head()


output.to_csv('submission.csv', index=False)

