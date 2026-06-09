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
# Replace with your dataset's path (check the printed filenames)
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')



test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df.head()


test.head()


df.isna().sum()


test.isna().sum()


# Calculate the mean of the 'num_sold' column
mean_num_sold = df['num_sold'].mean()

# Fill null values with the mean
df['num_sold'] = df['num_sold'].fillna(mean_num_sold)


df.isna().sum()


# Preprocess both datasets identically
def preprocess_data(df):
    # Convert 'date' to datetime and extract features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.drop(columns=['date'])
    
    # One-hot encode categorical columns
    categorical_cols = ['country', 'store', 'product']
    df = pd.get_dummies(df, columns=categorical_cols)
    return df



# Preprocess training and test data
df = preprocess_data(df)
test = preprocess_data(test)


df.head()


# Split training data into features (X_train) and target (y_train)
X_train = df.drop(columns=['num_sold'])
y_train = df['num_sold']


from sklearn.ensemble import RandomForestRegressor
# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Predict on the test data
test_predictions = model.predict(test)


# Save predictions to a CSV file (include the test data's original ID if available)
output_df = pd.DataFrame({
    'id': test['id'],  # Assuming 'id' exists in the test data
    'num_sold': test_predictions
})
output_df.to_csv('/kaggle/working//test_predictions2.csv', index=False)







