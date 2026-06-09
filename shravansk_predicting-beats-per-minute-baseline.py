# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

#test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
print("Shape of dataset:", df.shape)
df.head()




# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("✅ No missing values found.")

# Duplicates
print("Duplicate rows:", df.duplicated().sum())



target = "BeatsPerMinute"

plt.figure(figsize=(8,5))
sns.histplot(df[target], kde=True, bins=30)
plt.title(f"Distribution of {target}")
plt.show()


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
X = df.drop(columns=['id', target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))


# ========================
# 11. Feature Importance
# ========================
plt.figure(figsize=(10,6))
plt.barh(X.columns, xgb.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
X_test_final = test_df.drop(columns=['id'])
y_pred_test = xgb.predict(X_test_final)
X_test_final


output = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_pred_test
})

output.to_csv("test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")
print(output.head())



Things




