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
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ## 1. import libray and dataset

df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')



# Drop rows with missing values in 'num_sold'
df.dropna(subset=['num_sold'], inplace=True)

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Extract year, month, and day into separate columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['weekofyear'] = df['date'].dt.isocalendar().week

# Drop the original 'date' column
df = df.drop('date', axis=1)

# Initialize LabelEncoder
le = LabelEncoder()

# List of columns to label encode
categorical_cols = ['store', 'country','product']  # Add other categorical columns as needed

# Apply Label Encoding to specified columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df=df[['id','year', 'month', 'day', 'dayofweek', 'weekofyear', 'country', 'store', 'product', 'num_sold' ]]

# Split the data into features (X) and target (y)
X = df.drop(['num_sold', 'id'], axis=1)  # Exclude 'num_sold' and 'id'
y = df['num_sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
models = {
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MSE': mse, 'R-squared': r2}

# Create a DataFrame for results
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model', 'MSE', 'R-squared']

# Print the results
print(results_df)

# Determine the best model based on MSE (lower is better)
best_model = results_df.loc[results_df['MSE'].idxmin()]
print(f'\nBest model based on MSE:\n{best_model}')




# prompt: decision tree modle fit and precit the df_test = pd.read_csv('/content/test.csv') 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Load the test dataset
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Preprocess the test data (mimicking the training data preprocessing)
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['weekofyear'] = df_test['date'].dt.isocalendar().week
df_test = df_test.drop('date', axis=1)

le = LabelEncoder()
categorical_cols = ['store', 'country', 'product']
for col in categorical_cols:
    df_test[col] = le.fit_transform(df_test[col])

df_test = df_test[['id', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'country', 'store', 'product']]

# Use the best model (DecisionTreeRegressor in this case)
# Assuming 'X_train', 'y_train' from your training code
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(df_test.drop('id', axis = 1))

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': df_test['id'], 'num_sold': predictions})

# Save predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)




