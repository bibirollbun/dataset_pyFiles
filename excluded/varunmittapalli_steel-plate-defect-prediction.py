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


# ================================================================
# ğŸ§  Steel Plate Defect Prediction (Playground Series S4E3)
# Final Version - Includes 'Other_Faults'
# Author: Varun Mittapalli
# ================================================================

# ============ 1. Setup ============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

RANDOM_STATE = 42
plt.style.use("dark_background")

# ============ 2. Load Data ============
train = pd.read_csv("/kaggle/input/playground-series-s4e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e3/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s4e3/sample_submission.csv")

print("âœ… Data Loaded Successfully!")
print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")

# ============ 3. Identify Target Columns ============
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
id_col = 'id'

# ============ 4. EDA ============
print("\nğŸ”� Defect label distribution:")
display(train[target_cols].sum())

# Correlation Heatmap
num_cols = train.select_dtypes(include=np.number).columns.drop(target_cols, errors='ignore')

plt.figure(figsize=(10,6))
sns.heatmap(train[num_cols].corr(), cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

# Histogram of a sample numeric column
plt.figure(figsize=(8,5))
sns.histplot(train[num_cols[0]], bins=30, kde=True)
plt.title(f"Distribution of {num_cols[0]}")
plt.show()

# Countplot for defect types
plt.figure(figsize=(8,5))
train[target_cols].sum().sort_values().plot(kind='bar', color='skyblue')
plt.title("Count of Each Defect Type")
plt.ylabel("Count")
plt.show()

# ============ 5. Prepare Data ============
X = train.drop(columns=target_cols + [id_col])
y = train[target_cols]
X_test = test.drop(columns=[id_col])

# --- Align columns between train and test ---
missing_cols_in_test = [col for col in X.columns if col not in X_test.columns]
for col in missing_cols_in_test:
    X_test[col] = 0

extra_cols_in_test = [col for col in X_test.columns if col not in X.columns]
if extra_cols_in_test:
    X_test = X_test.drop(columns=extra_cols_in_test)

X_test = X_test[X.columns]

print(f"âœ… Columns aligned successfully: Train={X.shape}, Test={X_test.shape}")

# ============ 6. Scale Features ============
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ============ 7. Model ============
if HAS_LGB:
    base_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

model = MultiOutputClassifier(base_model)

# ============ 8. Train & Evaluate ============
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

score = f1_score(y_valid, y_pred, average='micro')
print(f"âœ… Validation F1 Score: {score:.4f}")

# ============ 9. Feature Importance ============
if hasattr(model.estimators_[0], 'feature_importances_'):
    imp = model.estimators_[0].feature_importances_
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10,5))
    sns.barplot(data=imp_df, x='Importance', y='Feature', color='orange')
    plt.title("Top 15 Important Features (Pastry defect)")
    plt.show()

# ============ 10. Predict & Submit ============
final_preds = model.predict(X_test_scaled)
submission = pd.DataFrame(final_preds, columns=target_cols)
submission.insert(0, 'id', test[id_col])

submission.to_csv("submission.csv", index=False)
print("âœ… Final submission file created successfully!")

submission.head()


