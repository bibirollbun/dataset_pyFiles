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


# Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb



# Load train, test, and sample submission files
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")  # If needed

# Display first few rows
train_data.head()



train_data.head()


train_data.describe()


train_data.info()


train_data.columns


# lets underatnd the uniqu columns in the data 

cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']

for col in cols:
    print('value counts for {col}:\n')
    print(train_data[col].value_counts())
    print("--"*50)


categorical_columns = ['Brand', 'Material', 'Size','Laptop Compartment',
       'Waterproof', 'Style', 'Color']

numerical_columns = ['Compartments','Weight Capacity (kg)']


categorical_columns


numerical_columns


for col in numerical_columns:
    print(f"Missing values in {col}: {train_data[col].isnull().sum()}")



for col in categorical_columns:
    print(f"Missing values in {col}: {train_data[col].isnull().sum()}")



for col in numerical_columns:
    plt.figure(figsize=(8,4))
    sns.histplot(train_data[col],bins=30,kde=True)
    plt.title('Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.show()


for col in numerical_columns:
    skewness =train_data[col].skew()
    print(f'Skewness of {col}:{skewness}')


import scipy.stats as stats

for col in numerical_columns:
    plt.figure(figsize=(6,6))
    stats.probplot(train_data[col].dropna(),dist='norm',plot=plt)
    plt.title(f'QQ plot for {col}')
    plt.show()



test_data.head()


## since both the distribution are approx normal I'm using mean here 

mean_weight = train_data['Weight Capacity (kg)'].mean()
train_data['Weight Capacity (kg)'].fillna(mean_weight,inplace=True)
test_data['Weight Capacity (kg)'].fillna(mean_weight,inplace=True)


for col in categorical_columns:
    mode_value =train_data[col].mode()[0]
    train_data[col].fillna(mode_value,inplace=True)
    test_data[col].fillna(mode_value,inplace=True)
print('missing values handled in both train and test datasets')


categorical_columns


## now lets encode the categorical valriables 

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

encoded_train=encoder.fit_transform(train_data[categorical_columns])
encoded_test = encoder.transform(test_data[categorical_columns])



# now converting to dataframe 
encoded_train_df = pd.DataFrame(encoded_train,columns=encoder.get_feature_names_out(categorical_columns))
encoded_test_df = pd.DataFrame(encoded_test,columns=encoder.get_feature_names_out(categorical_columns))

train_data=train_data.drop(columns=categorical_columns).reset_index(drop=True)
test_data=test_data.drop(columns =categorical_columns).reset_index(drop=True)

train_data = pd.concat([train_data,encoded_train_df],axis=1)
test_data=pd.concat([test_data,encoded_test_df],axis=1)

print('encoding on categorical data is done ')


# Now scaling the numerical features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_data[numerical_columns]=scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns]=scaler.transform(test_data[numerical_columns])

print('NUmerical features are scaled successfully ')


print("Missing values in train:\n", train_data.isnull().sum())
print("\nMissing values in test:\n", test_data.isnull().sum())

print("\nColumns in train:", train_data.columns)
print("\nColumns in test:", test_data.columns)



import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train_data[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


from sklearn.model_selection import train_test_split

X= train_data.drop(columns=['id','Price'])
y= train_data['Price']




X


y


X_train , X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=10)

print('Split done on training set ')


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

lr_model = LinearRegression()

lr_model.fit(X_train,y_train)

y_pred =lr_model.predict(X_val)

mae = mean_absolute_error(y_val,y_pred)
rmse = mean_squared_error(y_val,y_pred,squared=False)
print(f"ðŸ”¹ Linear Regression - MAE: {mae:.4f}, RMSE: {rmse:.4f}")



# for better peroformance lets use Xgboost 

import xgboost as xgb

xgb_model =xgb.XGBRegressor(n_estimators =10, learning_rate=0.1,random_state=10)

xgb_model.fit(X_train,y_train)




xgb_model.fit(X_train,y_train)
y_pred_xgb = xgb_model.predict(X_val)


mae_xgb = mean_absolute_error(y_val,y_pred_xgb)
rmse_xgb = mean_squared_error(y_val,y_pred_xgb,squared=False)


print('Mae:',mae_xgb)
print('RMSE:',rmse_xgb)



from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
}

# Grid search
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Predict
y_pred_best = best_xgb.predict(X_val)

# Evaluate
mae_best = mean_absolute_error(y_val, y_pred_best)
rmse_best = mean_squared_error(y_val, y_pred_best, squared=False)

print(f"âœ… Tuned XGBoost - MAE: {mae_best:.4f}, RMSE: {rmse_best:.4f}")



# Ensure test data does not have 'Price'
X_test = test_data.drop(columns=['id', 'Price'], errors='ignore')

# Predict using the trained model
test_data['Price'] = best_xgb.predict(X_test)

print("âœ… Predictions on test data completed successfully!")



# Create submission file
submission = test_data[['id', 'Price']]

# Save as CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file 'submission.csv' is ready! ðŸš€")





