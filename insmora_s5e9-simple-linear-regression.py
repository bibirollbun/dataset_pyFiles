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


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv",index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv",index_col="id")


train.head()


train.describe()


import matplotlib.pyplot as plt
# histograms for each column
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
train.hist(ax=axes, edgecolor='black', grid=False)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#  features and target
X = train.drop(columns=["BeatsPerMinute"])
y = train["BeatsPerMinute"]

# TRAIN AND VALIDATION SET
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# standarization (mean=0, std=1)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled   = feature_scaler.transform(X_val)
test_scaled  = feature_scaler.transform(test)


#scaling target
target_scaler = StandardScaler()  
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled   = target_scaler.transform(y_val.values.reshape(-1, 1))



import matplotlib.pyplot as plt

features = X.columns
n = len(features)
cols = 3  # number of columns in plot grid
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

for idx, feature in enumerate(features):
    i, j = divmod(idx, cols)
    ax = axes[i][j]
    ax.hist(X[feature], bins=30, alpha=0.5, label='Original', density=True, color='blue', edgecolor='black')
    ax.hist(X_train_scaled[:, idx], bins=30, alpha=0.5, label='Scaled', density=True, color='orange', edgecolor='black')
    ax.set_title(feature)
    ax.legend()
    ax.set_ylabel('Density')

# Remove any unused subplots
for rem in range(i * cols + j + 1, rows * cols):
    fig.delaxes(axes.flatten()[rem])

plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)

# predict val set
y_val_pred_scaled = lr_model.predict(X_val_scaled)

# undo the scaling
y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled)

# evaluation
mse = mean_squared_error(y_val, y_val_pred)
rmse = mse**0.5
r2 = r2_score(y_val, y_val_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")



# -------------------
# SUBMISSION
#--------------------
y_test_pred_scaled = lr_model.predict(test_scaled)

# undo scaling
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)

# y_test_pred is now a NumPy array with predicted Bpm
print(y_test_pred[:10])  # show first 10 predictions



pred_df = pd.DataFrame({
    "id": test.index,
    "BeatsPerMinute": y_test_pred.flatten() 
})

# Save to CSV
pred_df.to_csv("submission.csv", index=False)
pred_df



