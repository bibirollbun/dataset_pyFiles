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


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


train= pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train.head(6)


train.info()


train.describe()


train.isnull().sum()


train_df=train.dropna(axis=0, how='any')
test_df= test.dropna(axis=0, how='any')


train_df.isnull().sum()


train_df['date'] = pd.to_datetime(train_df['date'])
train_df.set_index('date', inplace=True)
test_df['date'] = pd.to_datetime(test_df['date'])
test_df.set_index('date', inplace=True)


train_df


plt.plot(train_df.index, train_df['num_sold'])
plt.title('Stickers Sold over Time', fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Stickers Sold', fontsize=15)


train_df['country'].value_counts().plot(kind='pie', figsize=(6,6))


train_df.pivot_table(values = 'num_sold', index = 'store', aggfunc='sum').plot(kind='bar')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder
# Create a label encoder
label_encoder = LabelEncoder()

train_encoded = train_df.copy()
test_encoded = test_df.copy()
train_encoded['store'] = label_encoder.fit_transform(train_df['store'])
train_encoded['country'] = label_encoder.fit_transform(train_df['country'])
train_encoded['product'] = label_encoder.fit_transform(train_df['product'])
test_encoded['store'] = label_encoder.fit_transform(test_df['store'])
test_encoded['country'] = label_encoder.fit_transform(test_df['country'])
test_encoded['product'] = label_encoder.fit_transform(test_df['product'])


from sklearn.model_selection import train_test_split
# Separate features (X) and target variable (y)
X = train_encoded.drop(['id','num_sold'], axis=1)
y = train_encoded['num_sold']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xg 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fit model
model = xg.XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10, seed = 123) 
model.fit(X_train, y_train)

pred = model.predict(X_test) 
  
# RMSE Computation 
rmse = np.sqrt(mean_squared_error(y_test, pred)) 
print("RMSE : % f" %(rmse)) 


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Creating an XGBoost regressor
model_xgb = xgb.XGBRegressor()

#Training the model on the training data
model_xgb.fit(X_train, y_train)

#Making predictions on the test set
predictions = model_xgb.predict(X_test)

# Calculate the mean squared error and R-squared score
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


test_encoded=test_encoded.drop('id',axis=1)
future_pred = model_xgb.predict(test_encoded)
future_pred


sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub['num_sold'] = future_pred
sub.to_csv('submission.csv', index=False)


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Define categorical and numerical features
categorical_features = ['store', 'country', 'product']


# Target variable
target = 'num_sold'  

# Split the data into training and testing sets
X = train_df.drop(['num_sold','id'],axis=1)
y = train_df['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model_catb = CatBoostRegressor(  iterations=1000,   learning_rate=0.05,  
   # depth=6,                  
   # cat_features=categorical_features,  
   # loss_function='RMSE',     
   # verbose=100               
# Initialize and train the CatBoost model
model_catb = CatBoostRegressor(cat_features=categorical_features)

# Fit the model
model_catb.fit(X_train, y_train)
# Make predictions
y_pred = model_catb.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Feature Importance
feature_importances = model_catb.get_feature_importance(prettified=True)
print("\nFeature Importances:")
print(feature_importances)



test_df=test_df.drop('id',axis=1)
future_pred_catb = model_catb.predict(test_df)
future_pred_catb


sub_catb = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub_catb['num_sold'] = future_pred_catb
sub_catb.to_csv('submission.csv', index=False)

