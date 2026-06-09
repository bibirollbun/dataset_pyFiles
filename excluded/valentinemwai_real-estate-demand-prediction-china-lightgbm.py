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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


import pandas as pd

# 1️⃣ Load all datasets
pre_owned = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv")
pre_owned_nearby = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv")
land = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv")
land_nearby = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv")
new_house = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
new_house_nearby = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv")
sector_poi = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv")
city_search = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv")
city_indexes = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv")

# 2️⃣ Merge all sector-month level data first
df = pre_owned.merge(pre_owned_nearby, on=["month", "sector"], how="left")
df = df.merge(land, on=["month", "sector"], how="left")
df = df.merge(land_nearby, on=["month", "sector"], how="left")
df = df.merge(new_house, on=["month", "sector"], how="left")
df = df.merge(new_house_nearby, on=["month", "sector"], how="left")

# 3️⃣ Add static sector-level POI data (join only on sector)
df = df.merge(sector_poi, on="sector", how="left")
#df = df.merge(city_search, on="month", how="left")

# 5️⃣ Final check
print(df.shape)
print(df.head())


city_indexes.head()


# 1️⃣ Ensure month is datetime and extract year
df["month"] = pd.to_datetime(df["month"])
df["year"] = df["month"].dt.year

# 2️⃣ Keep only one row per year in city_indexes
city_indexes_unique = city_indexes.drop_duplicates(subset=["city_indicator_data_year"])

# 3️⃣ Merge on year
df = df.merge(
    city_indexes_unique,
    left_on="year",
    right_on="city_indicator_data_year",
    how="left"
)

# 4️⃣ Remove extra merge key if not needed
df.drop(columns=["city_indicator_data_year"], inplace=True)

print(df.shape)
print(df.head())



df.describe()


missing_info = df.isnull().sum().reset_index()
missing_info.columns = ["column", "missing_count"]
missing_info["missing_percentage"] = (missing_info["missing_count"] / len(df)) * 100
missing_info = missing_info[missing_info["missing_count"] > 0]  # Only columns with missing values
missing_info.sort_values(by="missing_count", ascending=False, inplace=True)

print(missing_info)




# Drop columns with more than 70% missing values
threshold = len(df) * 0.7
df_new = df.dropna(axis=1, thresh=threshold)

print(df_new.shape)  # to check new dimensions



df_new.isnull().sum()


train_df=df_new.dropna()


train_df.isnull().sum()


train_df.shape


df_grouped = df.groupby("month", as_index=False)["amount_new_house_transactions"].sum()

plt.figure(figsize=(12, 6))
plt.plot(df_grouped["month"], df_grouped["amount_new_house_transactions"], marker='o')
plt.title("Amount of New House Transactions Over Time", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Total Amount")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df["amount_new_house_transactions"], bins=30, edgecolor='black', alpha=0.7, kde=True)
plt.title("Distribution of New House Transaction Amounts", fontsize=14)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# Aggregate total per sector
sector_totals = (
    df.groupby("sector", as_index=False)["amount_new_house_transactions"]
      .sum()
      .sort_values(by="amount_new_house_transactions", ascending=False)
      .head(10)
)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(sector_totals["sector"], sector_totals["amount_new_house_transactions"], color='skyblue')
plt.title("Top 10 Sectors by New House Transaction Amount", fontsize=14)
plt.xlabel("Sector")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.5)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_df = train_df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Display correlations with target variable only
target_corr = corr_matrix["amount_new_house_transactions"].sort_values(ascending=False)
print("Correlation with amount_new_house_transaction:")
print(target_corr)

# Plot full correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Top correlations with target variable
top_features = target_corr.index[:15]  # Top 15 correlated features
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Top Correlated Features with Target", fontsize=14)
plt.show()



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


top_features = target_corr.index[:15]
top_features


import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define target and features
target = "amount_new_house_transactions"
top_features = target_corr.index[:30].tolist()   # ensure it's a list
features = [col for col in top_features if col != target]

for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].astype('category')

# Handle missing values in selected features and target
train_df = train_df[features + [target]].dropna()

train = train_df.iloc[:-1152]   # everything except last 1152
test  = train_df.iloc[-1152:]   # last 1152 rows

# Split data
X = train[features]
y = train[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

# Predictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Remove any NaNs just in case
mask = ~np.isnan(y_pred)
y_test_clean = y_test.iloc[mask]
y_pred_clean = y_pred[mask]

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
r2 = r2_score(y_test_clean, y_pred_clean)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")



mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")


test_df= pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
test_df.head()


test_new  = test[features]
#y_test = test[target]
test_pred = model.predict(test_new)
test_df["new_house_transaction_amount"] = test_pred


submission = test_df[["id","new_house_transaction_amount"]]
submission.to_csv("submission.csv",index=False)
print("✅ Submission saved: submission.csv")
print(submission.head())

