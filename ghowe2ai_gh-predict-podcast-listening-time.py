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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np  # For calculating RMSE

# Load your train and test datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Define target and features
target_column = "Listening_Time_minutes"
feature_columns = ["id", "Podcast_Name", "Episode_Title", "Guest_Popularity_percentage", "Episode_Length_minutes"]

# Prepare training data
y_train = train_data[target_column]
X_train = pd.get_dummies(train_data[feature_columns])

# Prepare test data
X_test = pd.get_dummies(test_data[feature_columns])

# Ensure columns match for both train and test datasets
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train the regression model
model = HistGradientBoostingRegressor(max_depth=5, random_state=1)
model.fit(X_train, y_train)

# Predict for the test data
test_data["Listening_Time_minutes"] = model.predict(X_test)

# Evaluate the model on the training data using RMSE
y_train_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Calculate RMSE
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Truncate Listening_Time_minutes to 2 decimal places
test_data["Listening_Time_minutes"] = test_data["Listening_Time_minutes"].round(2)

# Save the predictions to submission.csv in the required format
submission = test_data[["id", "Listening_Time_minutes"]]
submission.to_csv("submission.csv", index=False)
print("submission.csv file saved with predicted Listening_Time_minutes truncated to 2 decimal places!")



