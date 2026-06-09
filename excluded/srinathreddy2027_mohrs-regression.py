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
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# Original Code (unchanged)
# -------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s3e25/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e25/test.csv")

X = train.drop(columns=["Hardness"])   # Features
y = train["Hardness"]      

if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in test.columns:
    test_id = test["id"]
    test = test.drop(columns=["id"])
else:
    test_id = pd.Series(range(len(test)))

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

preds = model.predict(test)

submission = pd.DataFrame({
    "id": test_id,
    "Hardness": preds
})

submission.to_csv("submission_mohs.csv", index=False)
print("✅ submission_mohs.csv generated successfully!")

# -------------------------------
# Visualization Additions
# -------------------------------

# 1️⃣ Distribution of Target Variable
plt.figure(figsize=(8,5))
sns.histplot(y, kde=True, bins=30, color="skyblue")
plt.title("Distribution of Hardness Values", fontsize=14)
plt.xlabel("Hardness")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Correlation Heatmap (numeric features only)
plt.figure(figsize=(10,6))
corr = train.select_dtypes(include=np.number).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# 3️⃣ Top 15 Feature Importances from RandomForest
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Most Important Features for Hardness Prediction", fontsize=14)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.show()

# 4️⃣ Actual vs Predicted (Train Data Sample)
sample_idx = np.random.choice(len(X), size=min(300, len(X)), replace=False)
sample_preds = model.predict(X.iloc[sample_idx])

plt.figure(figsize=(6,6))
sns.scatterplot(x=y.iloc[sample_idx], y=sample_preds, color="darkgreen", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Actual vs Predicted Hardness (Train Sample)", fontsize=14)
plt.xlabel("Actual Hardness")
plt.ylabel("Predicted Hardness")
plt.show()

# 5️⃣ Feature Importance Distribution
plt.figure(figsize=(8,4))
sns.histplot(importances, bins=20, color="teal", kde=True)
plt.title("Distribution of Feature Importance Values", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Count")
plt.show()


