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
train_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


# sample_submission_csv
training_extra.shape, sample_submission_csv.shape , test_csv.shape, train_csv.shape 


import pandas as pd
X_test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
X_test.shape
display(X_test)


import pandas as pd
X_train=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
X_train.shape
display(X_train)


df = pd.DataFrame(X_train)
df 


df.shape


df.head(5)


df.iloc[2000:2020]


df.info() 


all_id = df['id'] 


# df.drop(columns=['id'], inplace=True) 


df 


df.isnull().sum()


# Fill categorical columns with mode (most frequent value)
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median(), inplace=True)


df.isnull().sum()


df.dtypes


categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']



# Ensure numeric columns are of correct type
df['Compartments'] = df['Compartments'].astype(int)  # Convert to int if no decimals
df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].astype(float)
df['Price'] = df['Price'].astype(float)


df.dtypes


from sklearn.preprocessing import LabelEncoder 


# Apply Label Encoding to categorical columns
label_encoders = {}  
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  
    label_encoders[col] = le 


# Ensure numeric columns have the correct data type
df['Compartments'] = df['Compartments'].astype(int)  # Convert to int if no decimals
df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].astype(float)
df['Price'] = df['Price'].astype(float) 


# Display the updated data types
df.dtypes


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Separate features and target variable
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable


# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


test_ids = X_test['id']


# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


y_pred.shape


y_pred 


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")


# Save predictions to a CSV file with ID
submission = pd.DataFrame({'id': test_ids, 'Predicted_Price': y_pred})
submission.to_csv("submission.csv", index=False)


result = pd.read_csv('submission.csv')
result 


# y_pred.shape
X_test.shape 







