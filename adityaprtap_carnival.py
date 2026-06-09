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

data = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
df = pd.DataFrame(data)
df2 = pd.DataFrame(data)
print(data.head())


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Add target column
top_features = [
    'Health Score', 'Annual Income', 'Credit Score', 
    'Age', 'Vehicle Age', 'Insurance Duration', 'Premium Amount'
]

# Keep only selected columns
df = df[top_features].copy()

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

# Separate features (X) and target (y)
X = df.drop('Premium Amount', axis=1)
y = df['Premium Amount']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train XGBoost regressor
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



import pandas as pd

# Load the test dataset
test_df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/test.csv")

# Keep only the columns the model was trained on
top_features = [
    'Health Score', 'Annual Income', 'Credit Score', 
    'Age', 'Vehicle Age', 'Insurance Duration'
]

# Subset test data
X_test_final = test_df[top_features].copy()

# Fill missing values with median (same logic as training)
X_test_final = X_test_final.fillna(df[top_features].median(numeric_only=True))

# Predict Premium Amount
predictions = model.predict(X_test_final)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Premium Amount': predictions
})

# Save to CSV in Kaggle working directory
output_path = "/kaggle/working/premium_predictions.csv"
submission.to_csv(output_path, index=False)

print(f"✅ Predictions saved successfully to: {output_path}")
print(submission.head())



print(df.info())


import pandas as pd

df = df.drop(columns=['id', 'Occupation', 'Location', 'Previous Claims', 'Policy Start Date'])
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Annual Income'].fillna(df['Annual Income'].mean(), inplace=True)
print(df[['Age', 'Annual Income']].isnull().sum())
print(df.info())



import pandas as pd
import numpy as np

# Fill Marital Status
mask = df['Marital Status'].isna()
df.loc[mask, 'Marital Status'] = np.where(
    (df.loc[mask, 'Age'] < 25) | (df.loc[mask, 'Number of Children'] == 0),
    'Single',
    'Married'
)

# Compute mean number of children for married people (excluding zeros)
mean_children = df.loc[
    (df['Marital Status'] == 'Married') & (df['Number of Children'] > 0),
    'Number of Children'
].mean()

# Fill Number of Children properly
mask_children = df['Number of Children'].isna()
df.loc[mask_children, 'Number of Children'] = df.loc[mask_children, 'Marital Status'].apply(
    lambda x: 0 if x == 'Single' else mean_children
)

# Drop Customer Feedback
df = df.drop(columns=['Customer Feedback'])

# Fill remaining numeric columns with their mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())



print(df.info())


import pandas as pd

continuous_cols = [
    'Age', 'Annual Income', 'Health Score', 
    'Vehicle Age', 'Credit Score', 'Insurance Duration'
]

categorical_cols = [
    'Marital Status', 'Education Level', 'Policy Type',
    'Smoking Status', 'Exercise Frequency', 'Property Type'
]

target_col = 'Premium Amount'

df_encoded = pd.get_dummies(df[continuous_cols + categorical_cols], drop_first=True)
y = df[target_col]



correlations = df_encoded.corrwith(y).sort_values(ascending=False)
print(correlations)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X = df_encoded
y = y  # Premium Amount

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)



import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", rmse_xgb)



import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print("LightGBM RMSE:", rmse_lgb)



import pandas as pd
import matplotlib.pyplot as plt

importances = model.feature_importances_  # use your trained RandomForest/XGB/LGBM
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feat_imp[:20].plot(kind='barh')  # top 20 features
plt.title('Top Feature Importances')
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Features and target
X = df_encoded
y = y  # Premium Amount

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial RandomForest
model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and calculate RMSE
y_pred = model.predict(X_test)
rmse_initial = np.sqrt(mean_squared_error(y_test, y_pred))
print("Initial RMSE:", rmse_initial)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot top 20 features
plt.figure(figsize=(10,6))
importances[:20].plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.show()

# Drop features with very low importance (e.g., <0.001)
low_importance_features = importances[importances < 0.001].index
X_train_reduced = X_train.drop(columns=low_importance_features)
X_test_reduced = X_test.drop(columns=low_importance_features)

# Retrain model with reduced features
model_reduced = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
model_reduced.fit(X_train_reduced, y_train)

# Predict and calculate new RMSE
y_pred_reduced = model_reduced.predict(X_test_reduced)
rmse_reduced = np.sqrt(mean_squared_error(y_test, y_pred_reduced))
print("RMSE after dropping low-importance features:", rmse_reduced)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Select top features
# ==============================
top_features = [
    'Health Score', 'Annual Income', 'Credit Score', 
    'Age', 'Vehicle Age', 'Insurance Duration'
]

X_selected = X[top_features]
y_selected = y  # Premium Amount

# ==============================
# 2️⃣ Split data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_selected, test_size=0.2, random_state=42
)

# ==============================
# 3️⃣ Train model
# ==============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ==============================
# 4️⃣ Predict and evaluate
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained using only top features\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# ==============================
# 5️⃣ Plot feature importances
# ==============================
importances = pd.Series(model.feature_importances_, index=top_features)
importances.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importances (Top Features Only)")
plt.xlabel("Importance")
plt.show()



import pandas as pd

# keep only selected features
top_features = [
    'Health Score', 'Annual Income', 'Credit Score', 
    'Age', 'Vehicle Age', 'Insurance Duration'
]

df = df[top_features].copy()

# Check missing values before filling
print("Missing values before filling:")
print(df.isna().sum())

# Fill missing values
# We'll use median for numeric data (robust against outliers)
df = df.fillna(df.median(numeric_only=True))

# Verify filling worked
print("\nMissing values after filling:")
print(df.isna().sum())

# Optional: show summary
print("\nDataFrame shape after cleaning:", df.shape)


