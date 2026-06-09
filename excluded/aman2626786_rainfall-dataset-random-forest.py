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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# Select features and target column (assuming last column is the target)
target_column = train_data.columns[-1]  # Assuming the last column is the target


X = train_data.drop(columns=["id", target_column])
y = train_data[target_column]


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Fill missing values with the median
X_train.fillna(X_train.median(), inplace=True)
X_val.fillna(X_val.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data.drop(columns=["id"]))


# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)


# Make predictions on the test dataset
test_predictions_rf = rf_model.predict(X_test_scaled)



# Create submission file
submission_rf = pd.DataFrame({"id": test_data["id"], target_column: test_predictions_rf})
submission_rf.to_csv("predictions_rf.csv", index=False)




