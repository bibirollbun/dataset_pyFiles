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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load training data
train_path = "/kaggle/input/messy-food-waste-prediction-dataset/train.csv"  
train_df = pd.read_csv(train_path)

# Preprocessing for training data
# ---------------------------------
# Drop non-feature columns
train_df.drop(columns=["date", "ID"], inplace=True)  # Explicitly remove both ID and date

# Normalize categorical features
train_df["staff_experience"] = train_df["staff_experience"].str.lower().str.strip()
train_df["waste_category"] = train_df["waste_category"].str.lower().str.strip()

# Handle missing values
staff_exp_mode = train_df["staff_experience"].mode()[0]
train_df["staff_experience"].fillna(staff_exp_mode, inplace=True)

# Label Encoding
label_encoders = {}
categorical_cols = ["staff_experience", "waste_category"]

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Prepare features and target
X = train_df.drop(columns=["food_waste_kg"])
y = train_df["food_waste_kg"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model training
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    early_stopping_rounds=10,
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluation
y_pred = xgb_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Preprocessing for test data
# ----------------------------
test_df = pd.read_csv("/kaggle/input/messy-food-waste-prediction-dataset/test.csv")

# Preserve ID column for submission
submission_ids = test_df["ID"]

# Drop same columns as training data
test_features = test_df.drop(columns=["ID", "date"])  # Match training features

# Normalize categorical features
test_features["staff_experience"] = test_features["staff_experience"].str.lower().str.strip()
test_features["waste_category"] = test_features["waste_category"].str.lower().str.strip()

# Handle missing values using training data's mode
test_features["staff_experience"].fillna(staff_exp_mode, inplace=True)

# Handle unseen categories
for col in categorical_cols:
    test_features[col] = np.where(
        test_features[col].isin(label_encoders[col].classes_),
        test_features[col],
        train_df[col].mode()[0]
    )
    test_features[col] = label_encoders[col].transform(test_features[col])

# Standardize using training scaler
test_scaled = scaler.transform(test_features)

# Generate predictions
test_predictions = xgb_model.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({
    "ID": submission_ids,
    "food_waste_kg": test_predictions
})

# Save submission
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
print(submission.head())


submission







