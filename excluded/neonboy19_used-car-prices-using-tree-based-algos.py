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

train_data = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')


print(f"Train data : {train_data.shape}")
print(f"Test data : {test_data.shape}")



train_data.info()


train_data.head()


train_data.head()


train_data.isnull().sum()


# def handleNull(df):
    
#     for col in df.columns:

#         if df[col].isnull().sum() / len(df) >= 0.5:

#             print(f"{col} has alot of null values")
#             df.drop(columns = col , axis = 1 , inplace = True) 
            
#         elif df[col].isnull().any() and df[col].dtype == 'object':

#             mode_value = df[col].mode()[0]
#             print(f"replacing the NaN values of {col} with Mode : {mode_value}")
#             df[col].fillna(mode_value , inplace = True)

#         else:

#             mean_value = df[col].mean()
#             print(f"replacing the NaN values of {col} with Mean : {mean_value}")
#             df[col].fillna(mean_value , inplace = True)


# handleNull(train_data)

            

    


def cat_col_nulls(df):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in cat_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            print(f"replacing the NaN values of {col} with Mode : {mode_value}")
            df[col].fillna(mode_value , inplace = True)


def num_val_nulls(df):

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = df.drop(columns = cat_cols)
    
    for col in df.columns:
    
        if df[col].isnull().any():
            
            mean_value = df[col].mean()
            print(f"replacing the NaN values of {col} with Mean : {mean_value}")
            df[col].fillna(mean_value , inplace = True)


cat_col_nulls(train_data)
cat_col_nulls(test_data)


train_data.isnull().sum()


cat_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']
train_data[cat_cols].nunique()


train_data.drop(columns = ['id'] , inplace = True)
test_data.drop(columns = ['id'] , inplace = True)



train_data['years_old'] = 2025 - train_data['model_year']
test_data['years_old'] = 2025 - test_data['model_year']



for col in train_data.columns:

    if train_data[col].nunique() > 1000:

        print(f"All unique values of {col} : {train_data[col].unique()}")


import re

#making a function for this so it is easier 
def extract_value(pattern, engine_str, cast_type=float):
    match = re.search(pattern, engine_str)
    return cast_type(match.group(1)) if match else None

def extract_horsepower(engine_str):
    return extract_value(r'(\d+)\.?\d*HP', engine_str, int)

def extract_engine_size(engine_str):
    return extract_value(r'(\d+\.?\d*)L', engine_str, float)

def extract_cylinder_count(engine_str):
    return extract_value(r'(\d+) Cylinder', engine_str, int)





train_data['hp'] = train_data['engine'].apply(extract_horsepower)
train_data['size'] = train_data['engine'].apply(extract_engine_size)
train_data['cyl_count'] = train_data['engine'].apply(extract_cylinder_count)

test_data['hp'] = test_data['engine'].apply(extract_horsepower)
test_data['size'] = test_data['engine'].apply(extract_engine_size)
test_data['cyl_count'] = test_data['engine'].apply(extract_cylinder_count)


train_data.info()


train_data.isnull().sum()


num_val_nulls(train_data)
num_val_nulls(test_data)


train_data.isnull().sum()


train_data['fuel_type'].unique()


train_data['fuel_type'].replace(['â€“', 'not supported'], 'Unknown', inplace=True)
test_data['fuel_type'].replace(['â€“', 'not supported'], 'Unknown', inplace=True)



print(train_data['fuel_type'].unique())


train_data['transmission'].unique()



import re

def extract_transmission_type(transmission_str):
    transmission_types = {
        'Automatic': ['Automatic', 'A/T'],
        'Manual': ['Manual', 'M/T'],
        'CVT': ['CVT']
    }
    for key, keywords in transmission_types.items():
        if any(keyword in transmission_str for keyword in keywords):
            return key
    return 'Other'

def extract_speeds(transmission_str):
    return int(match.group(1)) if (match := re.search(r'(\d+)-Speed', transmission_str)) else None



train_data['trans_type'] = train_data['transmission'].apply(extract_transmission_type)
train_data['num_speeds'] = train_data['transmission'].apply(extract_speeds)

test_data['trans_type'] = test_data['transmission'].apply(extract_transmission_type)
test_data['num_speeds'] = test_data['transmission'].apply(extract_speeds)


train_data.drop('transmission', axis=1, inplace=True)
test_data.drop('transmission', axis=1, inplace=True)



train_data.info()


train_data.isnull().sum()


num_val_nulls(train_data)
num_val_nulls(test_data)


train_data.drop(columns = ['clean_title'] , axis = 1 , inplace = True)
test_data.drop(columns = ['clean_title'] , axis = 1 , inplace = True)



train_data.nunique()


from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()

train_data['accident'] = le.fit_transform(train_data['accident'])
test_data['accident'] = le.fit_transform(test_data['accident'])



train_data['accident']


cat_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']


cat_cols





for col in cat_cols:


    print(f"{col} has {train_data[col].unique()} values")


for col in cat_cols:

    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])
    


x = train_data.drop(columns = ['price'])
y = train_data['price']


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(x_train, y_train)  # No need for scaling
y_pred_rf = rf.predict(x_val)
rmse_rf = mean_squared_error(y_val, y_pred_rf, squared=False)
print(f'Random Forest RMSE: {rmse_rf:.2f}')

# 2. GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=123)
gbr.fit(x_train, y_train)  # No need for scaling
y_pred_gbr = gbr.predict(x_val)
rmse_gbr = mean_squared_error(y_val, y_pred_gbr, squared=False)
print(f'Gradient Boosting RMSE: {rmse_gbr:.2f}')

# 3. XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=123)
xgb.fit(x_train, y_train)  # No need for scaling
y_pred_xgb = xgb.predict(x_val)
rmse_xgb = mean_squared_error(y_val, y_pred_xgb, squared=False)
print(f'XGBoost RMSE: {rmse_xgb:.2f}')


test_predictions = gbr.predict(test_data)

submission = pd.DataFrame({
    'id': df_sample['id'],
    'price': test_predictions
})

print(submission.head())


submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")

