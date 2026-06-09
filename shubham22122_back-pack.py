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


## Backpack Prediction Challenge - Data Exploration & Model Training

# Importing Required Libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



# Loading Datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Merging Additional Training Data
df_train = pd.concat([df_train, df_train_extra], ignore_index=True)


# Display Dataset Information
df_train.info()


# Function to Handle Missing Values
def fill_na(df):
    df['Brand'] = df['Brand'].fillna('Unknown')
    df['Material'] = df['Material'].fillna('Unknown')
    df['Size'] = df['Size'].fillna('Unknown')
    df['Laptop Compartment'] = df['Laptop Compartment'].fillna('Unknown')
    df['Waterproof'] = df['Waterproof'].fillna('Unknown')
    df['Style'] = df['Style'].fillna('Unknown')
    df['Color'] = df['Color'].fillna('Unknown')
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())
    return df


# Filling Missing Values
df_train = fill_na(df_train)
df_test = fill_na(df_test)


# Checking for Remaining Null Values
print("Train Dataset Null Values:")
print(df_train.isnull().sum())
print("\nTest Dataset Null Values:")
print(df_test.isnull().sum())



# Function to Encode Categorical Variables
def label_encode(df):
    label_encoder = LabelEncoder()
    df['Brand'] = label_encoder.fit_transform(df['Brand'])
    df['Material'] = label_encoder.fit_transform(df['Material'])
    df['Size'] = label_encoder.fit_transform(df['Size'])
    df['Laptop Compartment'] = label_encoder.fit_transform(df['Laptop Compartment'])
    df['Waterproof'] = label_encoder.fit_transform(df['Waterproof'])
    df['Style'] = label_encoder.fit_transform(df['Style'])
    df['Color'] = label_encoder.fit_transform(df['Color'])
    return df


# Encoding Categorical Features
df_train = label_encode(df_train)
df_test = label_encode(df_test)



# Splitting Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Price', axis=1), df_train['Price'], test_size=0.2, random_state=42)



# Initializing and Training the Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)



# Making Predictions
y_pred = model.predict(x_test)
test_pred = model.predict(df_test)



# Calculating RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")


# Creating Submission File
submission = pd.DataFrame({'id': df_test.index, 'Price': test_pred})
submission.to_csv('submission.csv', index=False)



# Displaying Submission Output
print("Submission Preview:")
print(submission.head())

