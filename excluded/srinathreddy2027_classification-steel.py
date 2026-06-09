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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# Original Code (unchanged)
# -------------------------------
train_data = pd.read_csv("/kaggle/input/playground-series-s4e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e3/test.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
feature_cols = [col for col in train_data.columns if col not in (['id'] + target_cols)]

X = train_data[feature_cols]
y = train_data[target_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_data[feature_cols])

predictions = pd.DataFrame()
predictions["id"] = test_data["id"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

for col in target_cols:
    print(f"\nTraining model for: {col}")
    model.fit(X_scaled, y[col])
    preds = model.predict_proba(X_test_scaled)[:, 1]  # probability of defect
    predictions[col] = preds

predictions.to_csv("submission_steel.csv", index=False)
print("\n✅ Submission file 'submission_steel.csv' created successfully!")
print(predictions.head())

# -------------------------------
# Visualization Additions
# -------------------------------

# 1️⃣ Distribution of Each Defect (Target Variable)
plt.figure(figsize=(10,5))
y.sum().sort_values(ascending=False).plot(kind='bar', color='steelblue')
plt.title("Count of Each Defect Type in Training Data", fontsize=14)
plt.xlabel("Defect Type")
plt.ylabel("Count")
plt.show()

# 2️⃣ Correlation Heatmap of Numeric Features
plt.figure(figsize=(12,6))
corr = train_data[feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# 3️⃣ Feature Importance (Average across all targets)
importance_matrix = np.zeros(len(feature_cols))
for col in target_cols:
    model.fit(X_scaled, y[col])
    importance_matrix += model.feature_importances_

importance_matrix /= len(target_cols)
importances = pd.Series(importance_matrix, index=feature_cols).sort_values(ascending=False)
top_features = importances.head(15)

plt.figure(figsize=(10,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Most Important Features (Average across all defects)", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# 4️⃣ Pairwise Relationship (Optional Small Sample)
sample = train_data.sample(300, random_state=42)
sns.pairplot(sample[target_cols + feature_cols[:2]], diag_kind='kde')
plt.suptitle("Pairwise Relationships Between Features and Defects (Sample)", y=1.02)
plt.show()

# 5️⃣ Correlation between Defect Labels
plt.figure(figsize=(7,5))
sns.heatmap(y.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Between Defect Types", fontsize=14)
plt.show()


