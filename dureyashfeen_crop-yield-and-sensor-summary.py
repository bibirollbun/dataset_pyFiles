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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


df_tr =  pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv')
df_ts = pd.read_csv("/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv")
sample_sub = pd.read_csv("/kaggle/input/crop-yield-prediction-challenge/sample_submission.csv")


df_tr.head()


df_tr.tail()


df_tr.info()


df_tr.dtypes


df_tr.isnull().sum()


df_tr.nunique()


df_tr.describe()


df_tr['soil_moisture'].min
df_tr['avg_temperature'].min


df_tr['avg_temperature'].max
df_tr['avg_temperature'].max


df_tr["soil_moisture"].describe()


df_tr['avg_temperature'].describe()


df_encoded = pd.get_dummies(df_tr, columns=['avg_temperature', ])
df_encoded = pd.get_dummies(df_tr, columns=['soil_moisture', ])


df_tr['avg_temperature'].nunique


np.std(df_tr['avg_temperature'])


np.std(df_tr['soil_moisture'])


# --- Correlation Heatmap (only for numeric columns) ---
plt.figure(figsize=(10,6))
sns.heatmap(df_tr.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# --- Pairplot for Key Numeric Features (if not too many columns) ---
numeric_cols = df_tr.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) <= 6:
    sns.pairplot(df[numeric_cols])
    plt.show()

# --- Distribution Plots for Each Numeric Feature ---
df_tr[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()


# Select numeric columns only
numeric_cols = df_tr.select_dtypes(include=['float64', 'int64']).columns

# Box plots for all numeric features
plt.figure(figsize=(15, len(numeric_cols) * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(x=df_tr[col], color='skyblue')
    plt.title(f'Box Plot - {col}')
plt.tight_layout()
plt.show()


def cap_outliers(df_tr, column):
    Q1 = df_tr[column].quantile(0.25)
    Q3 = df_tr[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_tr[column] = df_tr[column].clip(lower, upper)
    return df_tr

for col in numeric_cols:
    df_tr = cap_outliers(df_tr, col)


plt.figure(figsize=(15, len(numeric_cols) * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(x=df_tr[col], color='lightgreen')
    plt.title(f'Box Plot After Outlier Handling - {col}')
plt.tight_layout()
plt.show()


# 4. Feature Selection
# Assuming 'yield' is the target column
target = 'yield_tpha'
X = df_tr.drop(columns=[target])
y = df_tr[target]

# Handle categorical features using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Training (LightGBM)
params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': -1,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

model = LGBMRegressor(**params)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# 10. Feature Importance Visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature', palette='viridis')
plt.title("Top 15 Important Features for Crop Yield Prediction", fontsize=14)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()



# 2. Keep a copy of ID column for submission
# Adjust 'id' if your dataset uses a different identifier
submission = pd.DataFrame()
submission['id'] = df_ts['id']

# 3. Apply same preprocessing steps as training
# Fill missing values
for col in df_ts.columns:
    if df_ts[col].dtype in ['float64', 'int64']:
        df_ts[col].fillna(df_ts[col].median(), inplace=True)
    else:
        df_ts[col].fillna(df_ts[col].mode()[0], inplace=True)

# 4. Drop non-feature columns (like 'id') if present
X_test_final = df_ts.drop(columns=['id'], errors='ignore')

# Apply same encoding
X_test_final = pd.get_dummies(X_test_final, drop_first=True)

# Align features to match training columns
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# Apply same scaling
X_test_final = scaler.transform(X_test_final)

# 5. Generate predictions
submission['yield'] = model.predict(X_test_final)

# 6. Save submission file
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully: submission.csv")

# Optional: display first few rows
submission.head()




