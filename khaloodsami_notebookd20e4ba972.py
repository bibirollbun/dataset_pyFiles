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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')



print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("Missing Values in Train Data:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])


# Fill numeric missing values with median
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('isFraud', errors='ignore')

num_imputer = SimpleImputer(strategy='median')
train_df[num_features] = num_imputer.fit_transform(train_df[num_features])
test_df[num_features] = num_imputer.transform(test_df[num_features])


cat_features = train_df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_features] = cat_imputer.fit_transform(train_df[cat_features])
test_df[cat_features] = cat_imputer.transform(test_df[cat_features])


from sklearn.preprocessing import OrdinalEncoder

# Use OrdinalEncoder to handle unseen labels
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[cat_features] = encoder.fit_transform(train_df[cat_features])
test_df[cat_features] = encoder.transform(test_df[cat_features])



# Step 3: Feature Scaling
scaler = StandardScaler()
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('isFraud', errors='ignore')

train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols.intersection(test_df.columns)] = scaler.transform(test_df[numeric_cols.intersection(test_df.columns)])



X = train_df.drop(columns=['isFraud'])  # Target variable: isFraud
y = train_df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



plt.figure(figsize=(10, 6))
sns.countplot(x=y_train)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_df['TransactionAmt'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()



print("Final Train Shape:", X_train.shape)
print("Final Test Shape:", X_test.shape)
print("Preprocessing Completed Successfully!")

