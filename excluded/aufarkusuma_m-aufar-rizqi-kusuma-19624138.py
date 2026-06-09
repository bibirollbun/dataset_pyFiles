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


# Read training and test data
train = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/train.csv')
test = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')
sample_submission = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')


from sklearn.impute import SimpleImputer

# Combine data to one dataset
df = pd.concat([train, test])

# Drop unnecessary feature
df = df.drop(
    ['name',
    'description', 
    'neighborhood_overview', 
    'host_id', 
    'host_name', 
    'host_since', 
    'host_location', 
    'host_about', 
    'host_listings_count',
    'bathrooms_text', 
    'availability_30',
    'availability_60',
    'availability_90',
    'number_of_reviews_ltm', 
    'number_of_reviews_l30d',
    'first_review',
    'last_review', ],
    axis=1)


"""
Note : 
1. host_acceptance_rate, host_response_rate --> tranform percentage to 0 to 1 value
2. host_is_superhost, host_has_profile_pic , host_identity_verified, has_availability   --> change to 0 and 1 (from boolean to numeric)
3. host_neighbourhood --> can be used, can also be not (try both cases)
4. host_verifications, amenities --> count how many amenities/verification that can be used (JSON file ex: ['email', 'phone'])
5. property_type, room_type  --> transform this category into number 

"""

# Divide data base on numerical and categorical data
numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include = ['object']).columns.tolist()

# Imputer for numerical value
numeric_imputer = SimpleImputer(strategy = 'mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Imputer for categorical value
categorical_imputer = SimpleImputer(strategy = 'most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])


import json 
from sklearn.preprocessing import OrdinalEncoder

# Transformation for percentage value
cols_percentage = ['host_acceptance_rate', 'host_response_rate']
for col in cols_percentage:
    df[col] = df[col].str.rstrip('%').astype(float) / 100

# Transformation for JSON file
cols_json = ['amenities']
for col in cols_json :
    df[col] = df[col].apply(lambda x : len(json.loads(x)))

# Transformation for data that has category
encoder = OrdinalEncoder()
cols_categ = ['host_is_superhost', 'host_has_profile_pic' , 'host_identity_verified', 'has_availability', 'property_type', 'room_type', 'host_verifications', 'neighbourhood', 'neighbourhood_cleansed','city', 'host_neighbourhood', 'host_response_time']
df[cols_categ] = encoder.fit_transform(df[cols_categ])


# Merge many columns
df = pd.concat([df[numeric_cols], df[categorical_cols]], axis=1)

# divide again df into train and test
train_final = df.iloc[:len(train)]
test_final = df.iloc[len(train):]


from sklearn.model_selection import train_test_split

# Seperate target variable
y = train_final.price
X = train_final.drop(columns=['price'])

# Divide data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



df.dtypes


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Define models
xgb_model = XGBRegressor(
    n_estimators = 1700,
    learning_rate = 0.1,
    random_state = 42, 
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
y_pred_train = xgb_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f'RMSE on training set: {rmse_train:.2f}')

# Predict on validation set
y_pred_val = xgb_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'RMSE on validation set: {rmse_val:.2f}')


test_final = test_final.drop(columns=['price'], axis=1)


submission_file = 'submission.csv'
test_ids = test.id
price = xgb_model.predict(test_final)
submission = pd.DataFrame({'id' : test_ids, 'price': price})
submission.to_csv(submission_file, index=False)
submission.head()

