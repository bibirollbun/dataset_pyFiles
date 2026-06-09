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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder


# ğŸ“Œ Load and prepare data â€“ use Calories dataset example
# NOTE: Replace with your actual train_df
train = train_df.copy()

X = train.drop(columns=['id', 'Calories'], errors='ignore')
y = train['Calories']

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical column
le = LabelEncoder()
X_train['Sex'] = le.fit_transform(X_train['Sex'])
X_val['Sex'] = le.transform(X_val['Sex'])

# ğŸ”§ Train baseline regressor (Gradient Boosting)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# ğŸ“� Compute RMSE and RMSLE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
rmsle = np.sqrt(mean_squared_log_error(y_val, np.maximum(0, y_pred)))

print(f"RMSE: {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")

# ğŸ”� Visual Comparison â€“ True vs Predicted (Linear vs Log)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred, alpha=0.3, color='blue')
plt.plot([0, 350], [0, 350], color='red', linestyle='--')
plt.title('True vs Predicted (Linear Scale)')
plt.xlabel('True Calories')
plt.ylabel('Predicted Calories')

plt.subplot(1, 2, 2)
#plt.scatter(np.log1p(y_val), np.log1p(y_pred), alpha=0.3, color='green')
plt.scatter(np.log1p(y_val), np.log1p(np.maximum(0, y_pred)), alpha=0.3, color='green')
plt.plot([0, 6], [0, 6], color='red', linestyle='--')
plt.title('True vs Predicted (Log1p Scale)')
plt.xlabel('log1p(True)')
plt.ylabel('log1p(Predicted)')

plt.tight_layout()
plt.show()


import seaborn as sns

plt.figure(figsize=(12, 5))

# Orijinal daÄŸÄ±lÄ±m
plt.subplot(1, 2, 1)
sns.histplot(train_df["Calories"], bins=50, kde=True, color='tomato')
plt.title("Original Target Distribution (Calories)")
plt.xlabel("Calories")
plt.ylabel("Frequency")

# log1p dÃ¶nÃ¼ÅŸÃ¼m sonrasÄ±
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df["Calories"]), bins=50, kde=True, color='seagreen')
plt.title("log1p(Calories) Distribution")
plt.xlabel("log1p(Calories)")
plt.ylabel("Frequency")

plt.suptitle("Effect of log1p Transformation on Target Variable", fontsize=14)
plt.tight_layout()
plt.show()

