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
import pandas as pd

# Check what Kaggle dataset folders are available
print("Available folders:", os.listdir("/kaggle/input"))



import pandas as pd

# Use the correct dataset path
dataset_path = "/kaggle/input/playground-series-s5e5"

# Load the data files
train = pd.read_csv(f"{dataset_path}/train.csv")
test = pd.read_csv(f"{dataset_path}/test.csv")
sample_sub = pd.read_csv(f"{dataset_path}/sample_submission.csv")

# Show some basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Columns:", train.columns.tolist())
train.head()






from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

target = "Calories"
features = [col for col in train.columns if col not in [target, "id"]]

# --- Fix Sex column to make sure it's text-based ---
train["Sex"] = train["Sex"].astype(str).str.lower()
test["Sex"] = test["Sex"].astype(str).str.lower()

# Encode categorical feature
le = LabelEncoder()
le.fit(pd.concat([train["Sex"], test["Sex"]], axis=0))
train["Sex"] = le.transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

# Prepare features and target
X = train[features].copy()
y = train[target]

# Split train-validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Validate
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
r2 = r2_score(y_val, val_preds)

print(f"ğŸ“Š Validation RMSE: {rmse:.4f}")
print(f"ğŸ“ˆ RÂ² Score: {r2:.4f}")

# Predict on test set
X_test = test[features].copy()
preds = model.predict(X_test)

# Save submission
sample_sub["Calories"] = preds
sample_sub.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv created successfully! You can now submit it on Kaggle.")
display(sample_sub.head())



# ====================================================
# ğŸ“¦ 1ï¸�âƒ£ Import Libraries
# ====================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================
# ğŸ“‚ 2ï¸�âƒ£ Load Data
# ====================================================
base_path = "/kaggle/input"
folders = os.listdir(base_path)
print("Available folders:", folders)

dataset_path = f"{base_path}/playground-series-s5e5"
print(f"âœ… Using dataset path: {dataset_path}")

train = pd.read_csv(f"{dataset_path}/train.csv")
test = pd.read_csv(f"{dataset_path}/test.csv")
sample_sub = pd.read_csv(f"{dataset_path}/sample_submission.csv")

# Replace infinities with NaN just to be clean
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
display(train.head())

# ====================================================
# ğŸ“Š 3ï¸�âƒ£ Exploratory Data Analysis (EDA)
# ====================================================

# Missing values heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values in Training Data")
plt.show()

# Distribution of Calories
plt.figure(figsize=(7, 4))
sns.histplot(train["Calories"], kde=True, bins=40, color="orange")
plt.title("Distribution of Calories")
plt.xlabel("Calories")
plt.ylabel("Count")
plt.show()

# Calories by Sex
plt.figure(figsize=(6, 4))
sns.boxplot(data=train, x="Sex", y="Calories", palette="Set2")
plt.title("Calories Burned by Sex")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

# ====================================================
# âš™ï¸� 4ï¸�âƒ£ Prepare Data
# ====================================================
target = "Calories"
features = [col for col in train.columns if col not in [target, "id"]]

train["Sex"] = train["Sex"].astype(str).str.lower()
test["Sex"] = test["Sex"].astype(str).str.lower()

# Encode categorical variable
le = LabelEncoder()
le.fit(pd.concat([train["Sex"], test["Sex"]], axis=0))
train["Sex"] = le.transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

X = train[features].copy()
y = train[target]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================================================
# ğŸ¤– 5ï¸�âƒ£ Train Model
# ====================================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Validation predictions
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
r2 = r2_score(y_val, val_preds)

print(f"ğŸ“Š Validation RMSE: {rmse:.4f}")
print(f"ğŸ“ˆ RÂ² Score: {r2:.4f}")

# ====================================================
# ğŸ“ˆ 6ï¸�âƒ£ Model Performance Visualization
# ====================================================

# Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_preds, alpha=0.3, color="purple")
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.show()

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
feat_imp.plot(kind="barh", color="skyblue")
plt.title("Feature Importance - Random Forest")
plt.show()

# ====================================================
# ğŸ�� 7ï¸�âƒ£ Generate Submission
# ====================================================
X_test = test[features].copy()
preds = model.predict(X_test)

sample_sub["Calories"] = preds
sample_sub.to_csv("submission.csv", index=False)

print("\nâœ… submission.csv created successfully! You can now submit it on Kaggle.")
display(sample_sub.head())


