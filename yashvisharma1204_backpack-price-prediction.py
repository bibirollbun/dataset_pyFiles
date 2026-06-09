# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# ğŸ“Œ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Loading the train dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")  # Update the path to your train.csv file


# Loading the test dataset
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv") # Update the path to your test.csv file


test_data.head()


df.head()


test_data.info()


# This tells us the number of rows, columns, and data types.  
df.info()  


# Checking for missing values in each column  
df.isnull().sum()  


# Checking for missing values in each column  
test_data.isnull().sum()  


# ğŸ“Š 2. Check Missing Values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cmap="coolwarm", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


combined_data = pd.concat([df, test_data], ignore_index=True)


# Handle missing values in categorical columns
categorical_cols = combined_data.select_dtypes(include=[object]).columns
for col in categorical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])


# Handle missing values in numerical columns using SimpleImputer
numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.difference(['Price'])
combined_data[numeric_cols] = combined_data[numeric_cols].fillna(combined_data[numeric_cols].mean())


# Boxplot to check for outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Price"])
plt.title("Boxplot of Price to Detect Outliers")
plt.show()


# Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=50, kde=True, color="blue")
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()


# Pairplot to check relationships
selected_features = ["Price", "Compartments", "Weight Capacity (kg)"]
sns.pairplot(df[selected_features])
plt.show()


# Count plots for categorical variables
categorical_features = ["Brand", "Material", "Size", "Style"]
for feature in categorical_features:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {feature}")
    plt.show()


# Price vs Features Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x="Material", y="Price", data=df)
plt.xticks(rotation=45)
plt.title("Price Distribution by Material")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Brand", y="Price", data=df, estimator=np.mean)
plt.xticks(rotation=45)
plt.title("Average Price by Brand")
plt.show()


# Encoding categorical columns using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])


# Split back into train and test
train_data = combined_data.iloc[:len(df)]
test_data = combined_data.iloc[len(df):].drop(columns=['Price'], errors='ignore')


X = train_data.drop(columns=['Price'])
y = train_data['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ğŸ“Š Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pd.concat([X_train, y_train], axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Feature Engineering
train_data["Price_per_kg"] = train_data["Price"] / train_data["Weight Capacity (kg)"]


# Verifying that the encoding is consistent for both train and test data
train_data.head()


test_data.head()


# ğŸ“Œ Train Models
## âœ… 1ï¸�âƒ£ Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_val)


# ğŸ“Œ 2ï¸�âƒ£ Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_val)


# ğŸ“Œ 3ï¸�âƒ£ XGBoost
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_val)


def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"ğŸ“ˆ {model_name} - RMSE: {rmse:.4f}, RÂ² Score: {r2:.4f}")


evaluate_model(y_val, y_pred_lr, "Linear Regression")
evaluate_model(y_val, y_pred_rf, "Random Forest")
evaluate_model(y_val, y_pred_xgb, "XGBoost")


# ğŸ“� Predictions on Test Data
test_predictions_lr = lin_reg.predict(test_data)
test_predictions_rf = rf_reg.predict(test_data)
test_predictions_xgb = xgb_reg.predict(test_data)


# Create Submission File
submission = pd.DataFrame({'id': test_data.index, 'Price': test_predictions_lr})  # Use Linear regression predictions
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission file saved: submission.csv")


submission

