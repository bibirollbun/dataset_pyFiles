# AgriYield 2025 - Starter Notebook

# 1. Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure visualization
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 2. Load the Data
train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())

# 3. Exploratory Data Analysis (EDA)

# Summary statistics
print(train.describe())

# ✅ FIX: Correlation heatmap using only numeric columns
numeric_cols = train.select_dtypes(include=[np.number])
corr = numeric_cols.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Yield distribution
sns.histplot(train['yield'], kde=True, bins=30)
plt.title('Distribution of Yield')
plt.xlabel("Yield (kg/ha)")
plt.show()

# 4. Prepare Data for Modeling

# Drop non-numeric ID column
X = train.drop(['field_id', 'yield'], axis=1)
y = train['yield']
X_test = test.drop(['field_id'], axis=1)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Baseline Model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation predictions and RMSE
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")

# 6. Predict on Test Set

test_preds = model.predict(X_test)
submission['yield'] = test_preds

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")





