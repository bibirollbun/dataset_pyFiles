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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")



print(train.isnull().sum())
print(test.isnull().sum())


print(train.describe())
print(train.describe(include='all'))



# Fill missing values with mode using apply
for col in ['Laptop Compartment', 'Waterproof']:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])



def clean_and_transform(df):
    # Fill missing categorical values
    df.fillna({
        'Brand': 'Unknown',
        'Material': 'Unknown',
        'Size': 'Unknown',
        'Style': 'Unknown',
        'Color': 'Unknown'
    }, inplace=True)
    
    # Fill missing numerical values
    df.fillna({
        
        'Weight Capacity (kg)': df['Weight Capacity (kg)'].median(),
        
    }, inplace=True)
    
    # Encode binary categorical columns
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    
    return df


# Clean train and test datasets
train = clean_and_transform(train)
test = clean_and_transform(test)

categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    
    # Handle unseen categories in test data
    test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    test[col] = le.transform(test[col])

# Check results
print(train.info())
print(test.info())



print(train.describe())
print(test.describe())


X_train = train.drop(columns=['id', 'Price'])  # Features (excluding target and ID)
y_train = train['Price']  # Target variable (Price)


X_test = test.drop(columns=['id'])  # Features (test set)



model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
predictions = model.predict(X_test)


# Create the submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'Price': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

