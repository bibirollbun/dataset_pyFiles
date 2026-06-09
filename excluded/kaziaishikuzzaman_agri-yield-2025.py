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
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score



train_df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission_df = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
display(train_df.head())


# Check for missing values
print("Missing values in training data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())


# Prepare features and target
X = train_df.drop(columns=['field_id', 'yield'])
y = train_df['yield']
X_test = test_df.drop(columns=['field_id'])


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Initialize and train Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)


# Evaluate model on validation set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse:.2f}')


# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
print(f'Cross-validation RMSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')


# Make predictions on test set
test_predictions = model.predict(X_test_scaled)


# Ensure predictions are non-negative (since yield can't be negative)
test_predictions = np.maximum(test_predictions, 0)


# Prepare submission file
submission_df['yield'] = test_predictions
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved to /kaggle/working/submission.csv")





