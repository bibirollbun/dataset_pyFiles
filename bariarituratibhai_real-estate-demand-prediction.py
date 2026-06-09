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



# Predict new_house_transaction_amount

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

#  Load train/test
train = pd.read_csv("/kaggle/input/train-dataset/train.csv")
test = pd.read_csv("/kaggle/input/test-datasets/test.csv")

#  Extract month from 'id'
train['month_str'] = train['id'].apply(lambda x: x.split('_')[0])
train['month'] = pd.to_datetime(train['month_str'], format='%Y %b', errors='coerce')

test['month_str'] = test['id'].apply(lambda x: x.split('_')[0])
test['month'] = pd.to_datetime(test['month_str'], format='%Y %b', errors='coerce')

#  Create target
train['new_house_transaction_amount'] = train['area_new_house_transactions'] * train['price_new_house_transactions']

#  Encode sector
le = LabelEncoder()
train['sector_encoded'] = le.fit_transform(train['id'].apply(lambda x: x.split('_')[1]))
test['sector_encoded'] = le.transform(test['id'].apply(lambda x: x.split('_')[1]))

#⃣ Time features
for df in [train, test]:
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month

#  Combine train and test to create lag features
full = pd.concat([train, test], sort=False).sort_values(['sector_encoded','month']).reset_index(drop=True)

# Lag features
for lag in [1,2,3]:
    full[f'lag_amount_{lag}'] = full.groupby('sector_encoded')['new_house_transaction_amount'].shift(lag)

# Rolling mean
full['rolling_3_amount'] = full.groupby('sector_encoded')['new_house_transaction_amount'].shift(1).rolling(3).mean()

# Fill NaNs in lag/rolling for test set with 0
full[['lag_amount_1','lag_amount_2','lag_amount_3','rolling_3_amount']] = full[['lag_amount_1','lag_amount_2','lag_amount_3','rolling_3_amount']].fillna(0)

# Split back
train_features = full[full['new_house_transaction_amount'].notna()]
test_features = full[full['new_house_transaction_amount'].isna()]

#  Features and target
features = ['sector_encoded','year','month_num','lag_amount_1','lag_amount_2','lag_amount_3','rolling_3_amount']
target = 'new_house_transaction_amount'

X_train = train_features[features]
y_train = train_features[target]
X_test = test_features[features]

#  Train XGBoost
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

#  Predict
y_test_pred = model.predict(X_test)
test_features['new_house_transaction_amount'] = y_test_pred

#  Prepare submission
submission = test_features[['id','new_house_transaction_amount']]
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")
print(submission.head())




# EDA + Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Load datasets
train = pd.read_csv("/kaggle/input/train-dataset/train.csv")
test = pd.read_csv("/kaggle/input/test-datasets/test.csv")

#  Basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:\n", train.columns)
print("\nMissing values in train:\n", train.isna().sum())

#  Extract month and sector
train['month_str'] = train['id'].apply(lambda x: x.split('_')[0])
train['sector'] = train['id'].apply(lambda x: x.split('_')[1])
train['month'] = pd.to_datetime(train['month_str'], format='%Y %b', errors='coerce')

test['month_str'] = test['id'].apply(lambda x: x.split('_')[0])
test['sector'] = test['id'].apply(lambda x: x.split('_')[1])
test['month'] = pd.to_datetime(test['month_str'], format='%Y %b', errors='coerce')

#  Create target
train['new_house_transaction_amount'] = train['area_new_house_transactions'] * train['price_new_house_transactions']

#  Summary statistics
print("\nTrain target stats:")
print(train['new_house_transaction_amount'].describe())

#  Distribution of new_house_transaction_amount
plt.figure(figsize=(10,6))
sns.histplot(train['new_house_transaction_amount'], bins=50, kde=True)
plt.title("Distribution of New House Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

#  Sector-wise transaction amount
plt.figure(figsize=(12,6))
sns.boxplot(x='sector', y='new_house_transaction_amount', data=train)
plt.title("Transaction Amount by Sector")
plt.xlabel("Sector")
plt.ylabel("Transaction Amount")
plt.xticks(rotation=45)
plt.show()

# Time series trend (total transaction amount per month)
monthly_amount = train.groupby('month')['new_house_transaction_amount'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='new_house_transaction_amount', data=monthly_amount, marker='o')
plt.title("Monthly Trend of Total Transaction Amount")
plt.xlabel("Month")
plt.ylabel("Total Transaction Amount")
plt.show()

#  Area vs price scatter
plt.figure(figsize=(10,6))
sns.scatterplot(x='area_new_house_transactions', y='price_new_house_transactions', hue='sector', data=train)
plt.title("Area vs Price by Sector")
plt.xlabel("Area of New House Transactions")
plt.ylabel("Price of New House Transactions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
corr = train[['area_new_house_transactions','price_new_house_transactions','new_house_transaction_amount']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation between Features and Target")
plt.show()

#  Count of records per sector
plt.figure(figsize=(10,5))
sns.countplot(x='sector', data=train)
plt.title("Number of Records per Sector")
plt.xlabel("Sector")
plt.ylabel("Count")
plt.show()


