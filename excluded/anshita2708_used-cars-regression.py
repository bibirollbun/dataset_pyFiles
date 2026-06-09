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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# Original Code (unchanged)
# -------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")

X = train.drop(columns=["price"])
y = train["price"]

cat_cols = X.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

X.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X, y)

preds = model.predict(test)

submission = pd.DataFrame({
    "id": test["id"],
    "price": preds
})

submission.to_csv("submission_car.csv", index=False)
print("✅ submission_car.csv created successfully!")

# -------------------------------
# Visualization Additions
# -------------------------------

# 1️⃣ Distribution of Target Variable
plt.figure(figsize=(8,5))
sns.histplot(y, kde=True, bins=30, color="royalblue")
plt.title("Distribution of Car Prices", fontsize=14)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Correlation Heatmap (Numeric features only)
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
plt.title("Top 15 Important Features", fontsize=14)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.show()

# 4️⃣ Scatter Plot — True Price vs Prediction (using small sample)
sample_idx = np.random.choice(len(X), size=300, replace=False)
sample_preds = model.predict(X.iloc[sample_idx])

plt.figure(figsize=(6,6))
sns.scatterplot(x=y.iloc[sample_idx], y=sample_preds, color="darkorange", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Actual vs Predicted Price (Train Sample)", fontsize=14)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()


