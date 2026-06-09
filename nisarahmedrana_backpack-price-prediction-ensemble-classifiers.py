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


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extras=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


import matplotlib.pyplot as plt
import seaborn as sns


train.head(10)


print("Dataset Shape:", train.shape)


print("Dataset Shape:", test.shape)


print("Dataset Info:")
train.info()


# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())


# Option 2: Impute missing categorical values with the mode
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


# Option 3: Impute missing numerical values with the mean or median
train['Weight Capacity (kg)']=train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median()) # Median is often preferred for skewed data
test['Weight Capacity (kg)']=test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median()) # Median is often preferred for skewed data


# Numerical Features Analysis
numerical_features = ['Compartments', 'Weight Capacity (kg)', 'Price']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(train[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=train[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

    print(f"Skewness of {feature}:", train[feature].skew())  # Check for skewness
    print(f"Kurtosis of {feature}:", train[feature].kurt()) # Check for kurtosis


# Categorical Features Analysis
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for feature in categorical_features:
    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
    sns.countplot(x=train[feature], order=train[feature].value_counts().index[:10]) # Show top 10 for large categories
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

    print(f"Value Counts for {feature}:\n", train[feature].value_counts()) # Print value counts


# Correlation Matrix for numerical features
correlation_matrix = train[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# Handling missing values by filling with 'Unknown'
train.fillna("Unknown", inplace=True)
test.fillna('Unknown', inplace=True)


# Encoding categorical features for train and test data
label_encoders = {}
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
for col in categorical_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


# Convert 'Unknown' values to NaN
train["Weight Capacity (kg)"] = pd.to_numeric(train["Weight Capacity (kg)"], errors='coerce')
test["Weight Capacity (kg)"] = pd.to_numeric(test["Weight Capacity (kg)"], errors='coerce')

# Fill NaN values with a default (e.g., mean or median)
train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median(), inplace=True)
test["Weight Capacity (kg)"].fillna(test["Weight Capacity (kg)"].median(), inplace=True)

# Convert to float32
train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].astype(np.float32)
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].astype(np.float32)


# Splitting data
X = train.drop(columns=["Price"])
y = train["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


import xgboost as xgb
# Training
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Predictions
y_pred_xgboost = model.predict(X_test)


from sklearn.ensemble import RandomForestRegressor

# Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)


from sklearn.ensemble import GradientBoostingRegressor

# Training
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# Predictions
y_pred_gbr = gbr_model.predict(X_test)


from sklearn.ensemble import AdaBoostRegressor

# Training
ada_model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
ada_model.fit(X_train, y_train)

# Predictions
y_pred_ada = ada_model.predict(X_test)


import lightgbm as lgb

# Training
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgb_model.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)


from catboost import CatBoostRegressor

# Training
cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0)
cat_model.fit(X_train, y_train)

# Predictions
y_pred_cat = cat_model.predict(X_test)


from sklearn.ensemble import ExtraTreesRegressor

# Training
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)
# Predictions
y_pred_et = et_model.predict(X_test)


print("RMSE - XGBoost:", np.sqrt(mean_squared_error(y_test, y_pred_xgboost)))
print("RMSE - Random Forest:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("RMSE - Gradient Boosting:", np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print("RMSE - AdaBoost:", np.sqrt(mean_squared_error(y_test, y_pred_ada)))
print("RMSE - LightGBM:", np.sqrt(mean_squared_error(y_test, y_pred_lgb)))
print("RMSE - CatBoost:", np.sqrt(mean_squared_error(y_test, y_pred_cat)))
print("RMSE - Extra Trees:", np.sqrt(mean_squared_error(y_test, y_pred_et)))


# Splitting data
X_train = train.drop(columns=["Price"])
y_train = train["Price"]
X_test = test


from sklearn.ensemble import ExtraTreesRegressor

# Training
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)
# Predictions
y_pred = et_model.predict(X_test)


# Creating the DataFrame with 'id' and 'price'
submission = pd.DataFrame({
    "id": X_test['id'],  # Assuming the test dataset retains original indexing
    "price": y_pred
})

# Displaying the first few rows
print(submission.head())



submission.to_csv("Submission_02.csv", index=False)

