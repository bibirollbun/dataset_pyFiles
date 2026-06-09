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
import numpy as np
#lets start it by reading CSV and concatinate
X = pd.read_csv('/kaggle/input/brain-image-and-neurodegenerative-disorders/trainImage.csv')
y = pd.read_csv('/kaggle/input/brain-image-and-neurodegenerative-disorders/trainIndex.csv')

# Combine into a single DataFrame
df = pd.concat([X, y], axis=1)



df = df.drop(columns=['colId'])
print(df.columns)


#rename index to target
df.rename(columns={'Index':'target'},inplace=True)
print(df.columns)


print(df.shape)


#lets try linear regression model first
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

x = df.drop('target', axis = 1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
print(x_train.shape)
print(x_test.shape)


model_lr = LinearRegression()
model_lr.fit(x_train,y_train)

y_pred = model_lr.predict(x_test)


# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


#very poor results from Linear regression, lets tru random forest
from sklearn.ensemble import RandomForestRegressor
# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


#neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

X = df.drop("target", axis=1)
y = df["target"]

# Normalize pixel values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build neural network
model = Sequential()
model.add(Dense(256, input_dim=x.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression output

# Compile model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Predict and evaluate
y_pred = model.predict(X_val).flatten()
r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


#LightGBM
import lightgbm as lgb

# Assuming df is your combined DataFrame with target column
x = df.drop('target', axis=1)
y = df['target']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = lgb.LGBMRegressor()
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns


residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.show()

# Residuals vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()


#tuning with grid search
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.1],
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}

model = lgb.LGBMRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='r2', cv=3, verbose=2)

grid_search.fit(x_train, y_train)
print("Best Params:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)


from sklearn.model_selection import RandomizedSearchCV


param_dist = {
    'num_leaves': np.arange(20, 150, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 10, 20, 30]
}

model = lgb.LGBMRegressor()
rand_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                 n_iter=20, scoring='r2', cv=5, verbose=2, random_state=42)

rand_search.fit(x_train, y_train)
print("Best Params:", rand_search.best_params_)
print("Best R² Score:", rand_search.best_score_)


print(lgb.__version__)


from sklearn.metrics import r2_score, mean_squared_error

# Get the best model directly from the search
best_model = rand_search.best_estimator_

# Fit the model again on full training data (optionally, merge x_train and x_val)
best_model.fit(x_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(x_test)
print("Final R² on test:", r2_score(y_test, y_pred))
print("MSE on test:", mean_squared_error(y_test, y_pred))


# Predict and evaluate for submission file
# Load training data
X = pd.read_csv('/kaggle/input/brain-image-and-neurodegenerative-disorders/trainImage.csv')
y = pd.read_csv('/kaggle/input/brain-image-and-neurodegenerative-disorders/trainIndex.csv')

# Drop colId if present
if 'colId' in X.columns:
    X = X.drop(columns=['colId'])

if 'colId' in y.columns:
    y = y.drop(columns=['colId'])

# Retrain best model on full data
best_model.fit(X, y)

# Load test data
test_df = pd.read_csv('/kaggle/input/brain-image-and-neurodegenerative-disorders/testImage.csv')
test_ids = test_df['colId']
X_test_final = test_df.drop(columns=['colId'])

# Predict on test data
test_predictions = best_model.predict(X_test_final)

# Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'Expected': test_predictions
})

submission.to_csv('submission.csv', index=False)







