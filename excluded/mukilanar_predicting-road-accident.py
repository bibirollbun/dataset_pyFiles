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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.head()


test.head()


sub.head()


train["accident_risk"].describe()


train.isnull().sum()


import matplotlib.pyplot as plt
# Create subplots
fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # 1 row, 5 columns

axs[0].hist(train['accident_risk'], bins=20, color='blue', edgecolor='black')
axs[0].set_title('accident_risk')

axs[1].hist(train['num_reported_accidents'], bins=5, color='blue', edgecolor='black')
axs[1].set_title('num_reported_accidents')

axs[2].hist(train['curvature'], bins=20, color='blue', edgecolor='black')
axs[2].set_title('curvature')

axs[3].hist(train['speed_limit'], bins=5, color='blue', edgecolor='black')
axs[3].set_title('speed_limit')

axs[4].hist(train['num_lanes'], bins=5, color='blue', edgecolor='black')
axs[4].set_title('num_lanes')
#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()


# Prepare Features and target

# Create the feature matrix X (input data) from the training set. df_train is your full training DataFrame.
# You are removing:
# 'id': likely a unique identifier (not useful for modeling)
# 'accident_risk': the target variable (you don’t want it as a feature)
# Result: X contains only the independent variables (features).
X = train.drop(columns=['id','accident_risk'], axis=1)

# You're creating the target variable y. This is what you want the model to predict. 
# Result: y contains the labels for training the model.
y = train['accident_risk']

# Preparing the test set features. Dropping ID again, as it is not necessary
X_test = test.drop(columns=['id'], axis=1)


# Define feature types
categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numeric = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


#Preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
preprocessor = ColumnTransformer([
    ('categ', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('bool', 'passthrough', boolean),
    ('num', 'passthrough', numeric)
])


# Define Model
# RMSE was 0.05642824635297133 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=.1,
                    max_depth=5,
                    random_state=42))
])


# RMSE is RMSE:  0.05923440973080326
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# # Pipeline
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators = 300, random_state = 42))
# ])


from sklearn.model_selection import train_test_split
# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_val)


# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print("RMSE: ", rmse)


y_pred2 = model.predict(X_test)
sub['accident_risk'] = y_pred2
sub.to_csv('submission.csv', index=False)
print(sub.head())

