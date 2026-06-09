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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load data
train_path = "/mnt/data/playground_series_s5e2/train.csv"
test_path = "/mnt/data/playground_series_s5e2/test.csv"
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Exploratory Data Analysis
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())


# Visualizing missing values
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df['Price'], bins=50, kde=True)
plt.title("Distribution of Backpack Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# Preprocessing
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_features = ['Compartments', 'Weight Capacity (kg)']

preprocessor = ColumnTransformer([
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]), categorical_features),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ]), numerical_features)
], remainder='passthrough')

# Prepare data
X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']


# Fit the preprocessor to avoid feature mismatch
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# Train simple model with encoding and missing value handling
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)


# Evaluate model
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Linear Regression RMSE: {rmse}')

# Make predictions on test set
X_test = test_df.drop(columns=['id'])
X_test = preprocessor.transform(X_test)
test_predictions = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({'id': test_df['id'], 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

