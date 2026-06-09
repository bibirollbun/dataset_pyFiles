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


# ============================================================
# ğŸ�� Playground Series S4E9 - Car Price Prediction
# Author: Varun (GPT-5 assisted)
# ============================================================

# === 1. Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# === 2. Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")

print("âœ… Data Loaded")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())

# === 3. Basic EDA (Exploratory Data Analysis) ===

# --- Distribution of target variable ---
plt.figure(figsize=(8, 5))
sns.histplot(train["price"], kde=True, bins=50, color="royalblue")
plt.title("Distribution of Car Prices", fontsize=14)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# --- Average price by brand (Top 15 brands) ---
plt.figure(figsize=(10, 6))
top_brands = train.groupby("brand")["price"].mean().sort_values(ascending=False).head(15)
sns.barplot(x=top_brands.values, y=top_brands.index, palette="viridis")
plt.title("Top 15 Brands by Average Price", fontsize=14)
plt.xlabel("Average Price")
plt.ylabel("Brand")
plt.show()

# --- Correlation of numeric features with price ---
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(8, 5))
sns.heatmap(train[numeric_cols].corr()[["price"]].sort_values(by="price", ascending=False), annot=True, cmap="coolwarm")
plt.title("Correlation with Price", fontsize=14)
plt.show()

# === 4. Data Preprocessing ===
target = "price"
X = train.drop(columns=[target, "id"])
y = train[target]
test_id = test["id"]
X_test = test.drop(columns=["id"])

# Fill missing values
X = X.fillna("Unknown")
X_test = X_test.fillna("Unknown")

# === 5. Feature Engineering ===
def extract_engine_features(df):
    df["horsepower"] = df["engine"].str.extract(r"(\d+\.?\d*)HP").astype(float)
    df["engine_size"] = df["engine"].str.extract(r"(\d+\.\d+)L").astype(float)
    df["num_cylinders"] = df["engine"].str.extract(r"(\d+) Cylinder").astype(float)
    df.drop(columns=["engine"], inplace=True)
    return df

X = extract_engine_features(X)
X_test = extract_engine_features(X_test)

# Replace missing numeric values with mean
for col in ["horsepower", "engine_size", "num_cylinders"]:
    X[col].fillna(X[col].mean(), inplace=True)
    X_test[col].fillna(X[col].mean(), inplace=True)

# === 6. Separate categorical and numerical columns ===
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# === 7. Preprocessing + Model Pipeline ===
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

model = XGBRegressor(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# === 8. Train / Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 9. Train the Model ===
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)

# === 10. Evaluation Metrics ===
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)
print(f"âœ… Validation RMSE: {rmse:.4f}")
print(f"âœ… RÂ² Score: {r2:.4f}")

# --- Predicted vs Actual Plot ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title("Predicted vs Actual Prices", fontsize=14)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# --- Residual Distribution ---
residuals = y_val - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True, color="salmon")
plt.title("Residual Distribution (Actual - Predicted)", fontsize=14)
plt.show()

# === 11. Train on Full Data & Predict on Test Set ===
pipe.fit(X, y)
test_preds = pipe.predict(X_test)

# === 12. Feature Importance (Approximation) ===
# To show top categorical or numeric feature influence
xgb_model = pipe.named_steps["model"]
if hasattr(xgb_model, "feature_importances_"):
    importances = xgb_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(range(20), importances[:20])
    plt.title("Top 20 Feature Importances (approx.)", fontsize=14)
    plt.show()

# === 13. Prepare Submission ===
submission = pd.DataFrame({
    "id": test_id,
    "price": test_preds
})
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv saved successfully!")
submission.head()


