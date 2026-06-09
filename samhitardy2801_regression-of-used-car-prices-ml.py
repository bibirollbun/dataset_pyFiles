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


# ==============================
# Step 1: Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================
# Step 2: Load Datasets
# ==============================
train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')

print("âœ… Files loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)
print("\nğŸ“‹ Columns in train:", list(train.columns))

# ==============================
# Step 3: Handle Missing / Infinite Values
# ==============================
train = train.replace([np.inf, -np.inf], np.nan).fillna('missing')
test = test.replace([np.inf, -np.inf], np.nan).fillna('missing')

# ==============================
# Step 4: Identify Target Column Automatically
# ==============================
possible_targets = ['price', 'selling_price', 'target']
target = None
for col in possible_targets:
    if col in train.columns:
        target = col
        break

if target is None:
    raise ValueError("â�Œ Couldn't find a target column (like 'price'). Please check your dataset columns!")

print(f"\nğŸ�¯ Target column detected: {target}")

# ==============================
# Step 5: Encode Categorical Columns
# ==============================
label_encoders = {}
for col in train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

print("\nğŸ”¤ Label encoding complete!")

# ==============================
# Step 6: Basic EDA Visualizations
# ==============================
plt.figure(figsize=(8,4))
sns.histplot(train[target], bins=40, kde=True, color='teal')
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, cmap='YlGnBu', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

if target in corr.columns:
    top_corr = corr[target].sort_values(ascending=False)[1:11]
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='magma')
    plt.title("Top 10 Features Correlated with Price")
    plt.xlabel("Correlation Value")
    plt.show()

# ==============================
# Step 7: Train-Test Split
# ==============================
X = train.drop(target, axis=1)
y = train[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nğŸ“Š Data split into training and validation sets!")

# ==============================
# Step 8: Model Training
# ==============================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("\nğŸ¤– Random Forest model trained successfully!")

# ==============================
# Step 9: Model Evaluation
# ==============================
val_preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, val_preds)
print(f"ğŸ“ˆ Validation Mean Absolute Error (MAE): {mae:.4f}")

# ==============================
# Step 10: Feature Importance
# ==============================
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,5))
importances.sort_values(ascending=False)[:15].plot(kind='bar', color='orange')
plt.title("Top 15 Important Features (Random Forest)")
plt.ylabel("Importance Score")
plt.show()

# ==============================
# Step 11: Make Predictions on Test Data
# ==============================
test_preds = model.predict(test)

# ==============================
# Step 12: Create and Save Submission File
# ==============================
submission = sample_submission.copy()
submission[submission.columns[-1]] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission file created successfully!")
print("ğŸ“� Saved as: /kaggle/working/submission.csv")
print("\nğŸ§¾ Example rows:")
print(submission.head())


