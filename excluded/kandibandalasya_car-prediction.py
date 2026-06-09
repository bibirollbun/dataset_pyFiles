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


import os

# List all datasets attached to your notebook
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print("   ", filename)



# ============================================================
# ğŸš— Car Price Prediction | Kaggle Playground S4E9
# âœ… Clean Version with Visualizations & No Warnings
# ============================================================

# -----------------------------
# 1ï¸�âƒ£ Imports
# -----------------------------
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# -----------------------------
# 2ï¸�âƒ£ Locate Dataset Folder Automatically
# -----------------------------
data_dir = None
for dirname, _, filenames in os.walk('/kaggle/input'):
    if 'train.csv' in filenames and 'test.csv' in filenames:
        data_dir = dirname
        print("âœ… Found dataset directory:", data_dir)
        break

if data_dir is None:
    raise FileNotFoundError("â�Œ Could not find train.csv and test.csv in /kaggle/input")

# -----------------------------
# 3ï¸�âƒ£ Load Data
# -----------------------------
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("âœ… Data Loaded:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

# -----------------------------
# 4ï¸�âƒ£ Basic Cleaning
# -----------------------------
train = train.fillna("Unknown")
test = test.fillna("Unknown")

# Replace infinite values to prevent seaborn/pandas warnings
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

X = train.drop(['id', 'price'], axis=1)
y = train['price']
X_test = test.drop(['id'], axis=1)

# -----------------------------
# 5ï¸�âƒ£ Exploratory Data Analysis (EDA)
# -----------------------------

# --- Price Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(y, bins=50, kde=True, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# --- Top 10 Most Common Car Models ---
if 'model' in X.columns:
    plt.figure(figsize=(10,5))
    top_models = X['model'].value_counts().head(10)
    sns.barplot(x=top_models.index, y=top_models.values, palette="viridis")
    plt.title("Top 10 Most Common Car Models")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# --- Average Price by Brand (if available) ---
if 'brand' in X.columns:
    plt.figure(figsize=(10,5))
    brand_price = train.groupby('brand')['price'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=brand_price.index, y=brand_price.values, palette="coolwarm")
    plt.title("Average Price by Brand (Top 10)")
    plt.xlabel("Brand")
    plt.ylabel("Average Price")
    plt.xticks(rotation=45)
    plt.show()

# --- Correlation Heatmap (Numerical Features Only) ---
plt.figure(figsize=(10,6))
numeric_cols = train.select_dtypes(include=[np.number]).columns
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 6ï¸�âƒ£ Safe Label Encoding
# -----------------------------
cat_cols = X.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    le.fit(list(X[col].astype(str).values) + list(X_test[col].astype(str).values))
    X[col] = le.transform(list(X[col].astype(str).values))
    X_test[col] = le.transform(list(X_test[col].astype(str).values))

print(f"âœ… Encoded {len(cat_cols)} categorical features safely.")

# -----------------------------
# 7ï¸�âƒ£ Split Data
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8ï¸�âƒ£ Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# 9ï¸�âƒ£ Validate Model
# -----------------------------
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"âœ… Validation RMSE: {rmse:.2f}")

# --- Plot Predicted vs Actual Prices ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.show()

# -----------------------------
# ğŸ”Ÿ Feature Importance Visualization
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="crest")
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.show()

# -----------------------------
# 11ï¸�âƒ£ Predict on Test Data & Save Submission
# -----------------------------
test_preds = model.predict(X_test)
submission = sample_submission.copy()
submission['price'] = test_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("ğŸ�‰ Submission file saved successfully to: /kaggle/working/submission.csv")


