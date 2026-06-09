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


#Reading the data
df = pd.read_csv('/kaggle/input/kaggle-2-1/train.csv')
test_data = pd.read_csv('/kaggle/input/kaggle-2-1/test.csv')


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os


print("Basic Information:")
print(df.info())


print("\nSummary Statistics:")
print(df.describe())


print("\nUnique Values in Each Column:")
print(df.nunique())


print("\nMissing Values:")
print(df.isnull().sum())


# Create new variable in train_data

# Lot area per living
df['LOT_TO_LVG_RATIO'] = df['LND_SQFOOT'] / df['TOT_LVG_AREA']

# Average distance to the interest
df['AVG_DIST'] = df[['RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'HWY_DIST']].mean(axis=1)

# Proximity to the water
df['WATER_PROX'] = df[['OCEAN_DIST', 'WATER_DIST']].min(axis=1)

# Age related
df['AGE_BIN'] = pd.cut(df['age'], bins=[0, 30, 60, 100], labels=['Young', 'Mid', 'Old'])

# Temporal
df['IS_SPRING'] = df['month_sold'].isin([3, 4, 5]).astype(int)
df['IS_SUMMER'] = df['month_sold'].isin([6, 7, 8]).astype(int)
df['IS_FALL'] = df['month_sold'].isin([9, 10, 11]).astype(int)
df['IS_WINTER'] = df['month_sold'].isin([12, 1, 2]).astype(int)

df.head()


# test_data new features
# Lot area per living
test_data['LOT_TO_LVG_RATIO'] = test_data['LND_SQFOOT'] / test_data['TOT_LVG_AREA']

# Average distance to the interest
test_data['AVG_DIST'] = test_data[['RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'HWY_DIST']].mean(axis=1)

# Proximity to the water
test_data['WATER_PROX'] = test_data[['OCEAN_DIST', 'WATER_DIST']].min(axis=1)

# Age related
test_data['AGE_BIN'] = pd.cut(test_data['age'], bins=[0, 30, 60, 100], labels=['Young', 'Mid', 'Old'])

# Temporal
test_data['IS_SPRING'] = test_data['month_sold'].isin([3, 4, 5]).astype(int)
test_data['IS_SUMMER'] = test_data['month_sold'].isin([6, 7, 8]).astype(int)
test_data['IS_FALL'] = test_data['month_sold'].isin([9, 10, 11]).astype(int)
test_data['IS_WINTER'] = test_data['month_sold'].isin([12, 1, 2]).astype(int)

test_data.head()


# Define the mapping
age_bin_mapping = {'Young': 1, 'Mid': 2, 'Old': 3}

# Apply the mapping
test_data['AGE_BIN_NUM'] = test_data['AGE_BIN'].map(age_bin_mapping)
df['AGE_BIN_NUM'] = df['AGE_BIN'].map(age_bin_mapping)

df.head()


train_data = df[df['SALE_PRC'].notna()] 


features = ['TOT_LVG_AREA', 'LND_SQFOOT', 'RAIL_DIST', 'OCEAN_DIST', 'SPEC_FEAT_VAL']

X_train = train_data[features]
y_train = train_data['SALE_PRC']
X_test = test_data[features] 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


test_predictions = model.predict(X_test_scaled)


# Extract features from test_data
test_features = test_data[['TOT_LVG_AREA', 'LND_SQFOOT', 'RAIL_DIST', 'OCEAN_DIST', 'SPEC_FEAT_VAL']]

# Scale the test features
test_features_scaled = scaler.transform(test_features)

# Generate predictions
test_predictions = model.predict(test_features_scaled)

# Add predictions to the test dataset
test_data['SALE_PRC'] = test_predictions

# Create the submission DataFrame with 'id' and 'SALE_PRC'
submission = test_data[['id', 'SALE_PRC']]

# Save the submission file
submission.to_csv('submission5.csv', index=False)
print("Submission file created successfully as 'submission.csv'")




