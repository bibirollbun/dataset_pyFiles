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


# ğŸš² Bike Sharing Demand â€” Kaggle Notebook with Visualization
# âœ… Single cell, ready to run in Kaggle
# Competition: https://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
sample_submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

print("âœ… Files loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Step 2: Feature engineering
for df in [train, test]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek

# Step 3: Data visualization

# 3.1 Count over years
plt.figure(figsize=(6,4))
sns.barplot(x='year', y='count', data=train, palette='viridis')
plt.title("Average Bike Rentals per Year")
plt.show()

# 3.2 Monthly trend
plt.figure(figsize=(8,4))
sns.barplot(x='month', y='count', data=train, palette='magma')
plt.title("Average Bike Rentals per Month")
plt.show()

# 3.3 Hourly pattern
plt.figure(figsize=(10,4))
sns.barplot(x='hour', y='count', data=train, palette='coolwarm')
plt.title("Average Bike Rentals per Hour")
plt.show()

# 3.4 Effect of weather
plt.figure(figsize=(6,4))
sns.barplot(x='weather', y='count', data=train, palette='cubehelix')
plt.title("Effect of Weather on Bike Rentals")
plt.show()

# 3.5 Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Drop columns not needed
drop_cols = ['datetime', 'casual', 'registered']
X = train.drop(drop_cols + ['count'], axis=1)
y = np.log1p(train['count'])  # log-transform to reduce skew
X_test = test.drop(['datetime'], axis=1)

# Step 5: Split train-validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Step 7: Validate
y_pred = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred)))
print(f"ğŸ“Š Validation RMSLE: {rmsle:.4f}")

# Step 8: Feature Importance Visualization
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,5))
importances.sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance Score")
plt.show()

# Step 9: Predict on test set
test_preds = np.expm1(model.predict(X_test))  # reverse log1p

# Step 10: Create submission
submission = sample_submission.copy()
submission['count'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission file created successfully!")
print("ğŸ“� Path: /kaggle/working/submission.csv")


