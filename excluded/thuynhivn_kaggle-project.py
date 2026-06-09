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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, classification_report

# Loading the data
df = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/train.csv')
test_data = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/test.csv')
df.head()


print("\nInformation:")
print(df.info())

print('\nSummary')
print(df.describe())

print('\nMissing Values')
print(df.isnull().sum())


# Datetime format
df['orderDate'] = pd.to_datetime(df['orderDate'], errors='coerce')
df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], errors='coerce')
df['creationDate'] = pd.to_datetime(df['creationDate'], errors='coerce')
df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')



df['deliveryTime'] = (df['deliveryDate'] - df['orderDate']).dt.days

df['pre_age'] = (df['creationDate']-df['dateOfBirth']).dt.days / 365

df['price_to_age_ratio'] = df['price'] / df['pre_age']

df['time_since_last_purchase'] = (df['orderDate'] - df.groupby('customerID')['orderDate'].shift(1)).dt.days

df['order_quarter'] = df['orderDate'].dt.quarter

df['itemID_size'] = df.groupby('itemID')['itemID'].transform('size')

df['total_purchases'] = df.groupby('customerID')['price'].transform('sum')

df['total_spent'] = df.groupby('customerID')['price'].transform('sum')

df['purchase_frequency'] = df.groupby('customerID')['orderDate'].transform('count')

df.head()


# Encode categorical variables using one-hot encoding
categorical_columns = ['size', 'color', 'salutation', 'state'] 

# Ensure only valid columns are encoded
existing_categorical_columns = [col for col in categorical_columns if col in df.columns]

# One-hot encode the categorical columns
df = pd.get_dummies(df, columns=existing_categorical_columns, drop_first=True)

# Define features (X) and target (y) without dropping any other columns
X = df.drop('returnShipment', axis=1)
y = df['returnShipment']


# Handle potential issues with invalid dates
df['deliveryTime'] = df['deliveryTime'].fillna(1)
df['pre_age'] = df['pre_age'].fillna(df['pre_age'].median())
df['price_to_age_ratio'] = df['price_to_age_ratio'].fillna(1)
df['time_since_last_purchase'] = df['time_since_last_purchase'].fillna(0)



from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values for both training and validation sets
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
y_train = y_train.fillna(0)

# Drop unnecessary columns if they exist
columns_to_drop = ['creationDate', 'dateOfBirth','orderDate', 'deliveryDate','color','salutation','state',
                   'order_quarter','time_since_last_purchase','total_purchases','total_spent','pre_age','itemID_size'
                   'price_to_age_ratio']
                   
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_val = X_val.drop(columns=columns_to_drop, errors='ignore')

# Ensure categorical variables are properly encoded
X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)

# Re-align columns in X_train and X_val
X_train, X_val = X_train.align(X_val, join='inner', axis=1)

# Check shapes and consistency
print(X_train.shape, X_val.shape, y_train.shape)

# Train Random Forest
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)


predictions = model.predict_proba(X_val)
# Extract probabilities of class 1
positive_class_probs = predictions[:, 1]
print(positive_class_probs)


# Ensure test_data is preprocessed similarly to X_train
test_data = pd.get_dummies(test_data, drop_first=True)

# Align test_data columns with X_train
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

# Predict probabilities for the positive class (class 1)
test_data['returnShipment'] = model.predict_proba(test_data)[:, 1]




