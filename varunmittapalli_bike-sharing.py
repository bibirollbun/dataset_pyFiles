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


# =====================================================================
# ğŸš´â€�â™‚ï¸� Bike Sharing Demand - Kaggle Competition (Late Submission)
# Author: Varun Mittapalli
# =====================================================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Configure plot style
plt.style.use("dark_background")
sns.set_palette("mako")
plt.rcParams["figure.figsize"] = (10, 5)

# Step 2: Load Dataset
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

print("âœ… Data Loaded Successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nColumns:", train.columns.tolist())

# Step 3: Initial Data Check
display(train.head())
display(train.describe())

# Step 4: Feature Engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

# Step 5: Exploratory Data Analysis (EDA)
# -----------------------------------------------------

# Plot 1: Count by Hour
plt.figure(figsize=(10,5))
sns.barplot(x="hour", y="count", data=train, palette="mako")
plt.title("â�° Average Demand by Hour")
plt.show()

# Plot 2: Count by Month
plt.figure(figsize=(10,5))
sns.barplot(x="month", y="count", data=train, palette="viridis")
plt.title("ğŸ“… Average Demand by Month")
plt.show()

# Plot 3: Count by Day of Week
plt.figure(figsize=(10,5))
sns.barplot(x="dayofweek", y="count", data=train, palette="cool")
plt.title("ğŸ—“ï¸� Average Demand by Day of Week (0=Mon, 6=Sun)")
plt.show()

# Plot 4: Weather vs Demand
plt.figure(figsize=(8,5))
sns.boxplot(x="weather", y="count", data=train)
plt.title("ğŸŒ¦ï¸� Weather vs Demand")
plt.show()

# Plot 5: Temperature vs Demand
plt.figure(figsize=(8,5))
sns.scatterplot(x="temp", y="count", data=train, hue="season", alpha=0.7)
plt.title("ğŸŒ¡ï¸� Temperature vs Demand (Colored by Season)")
plt.show()

# Step 6: Prepare Training Data
y = train["count"]
train = train.drop(["count", "casual", "registered", "datetime"], axis=1)
test_ids = test["datetime"]
test = test.drop(["datetime"], axis=1)

# Step 7: Split Data for Validation
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42)

# Step 8: Model Training (Random Forest)
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred = model.predict(X_val)
y_pred = np.where(y_pred < 0, 0, y_pred)

rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f"âœ… Validation RMSLE: {rmsle:.4f}")

# Step 10: Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=train.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("ğŸ�¯ Feature Importance")
plt.show()

# Step 11: Predict on Test Set
test_pred = model.predict(test)
test_pred = np.where(test_pred < 0, 0, test_pred)

# Step 12: Create Submission File
submission = pd.DataFrame({
    "datetime": test_ids,
    "count": test_pred
})
submission.to_csv("submission.csv", index=False)

print("\nğŸš€ submission.csv created successfully! You can now upload it to Kaggle.")
submission.head()


