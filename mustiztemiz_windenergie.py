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


df_train = pd.read_csv("/kaggle/input/ensimag-mmis-2024/train.csv")
df_test = pd.read_csv("/kaggle/input/ensimag-mmis-2024/test.csv")


df_train.head()


df_test.head()


df_train.info()


df_test.info()


df_train.describe()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Convert date to datetime
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

# Extract time-based features
df_train['hour'] = df_train['date'].dt.hour
df_test['hour'] = df_test['date'].dt.hour

# Select features
features = ['u10', 'v10', 'u100', 'v100', 'hour']
X = df_train[features]
y = df_train['production']

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Make predictions on test set
X_test = df_test[features]
predictions = rf_model.predict(X_test)

# Save predictions to CSV
df_test['production'] = predictions
output_df = df_test[['date', 'production']]
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

# Show feature importance and sample predictions
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)
print("Sample Predictions:")
print(df_test[['date', 'production']].head())

# Calculate R2 score on a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_predictions = rf_model.predict(X_val)
r2 = r2_score(y_val, val_predictions)
print(f"Model R2 Score on validation set: {r2:.4f}")




