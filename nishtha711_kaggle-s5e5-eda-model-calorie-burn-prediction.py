# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Preview data
print(train_df.head())


# Drop ID column if exists
train_df.drop(columns=['id'], inplace=True, errors='ignore')
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True, errors='ignore')

# Check column names
print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)

# Try to identify actual column name for gender-related data
gender_col = [col for col in train_df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
if gender_col:
    gender_col = gender_col[0]
    train_df[gender_col] = train_df[gender_col].map({'male': 0, 'female': 1})
    test_df[gender_col] = test_df[gender_col].map({'male': 0, 'female': 1})
else:
    print("Gender-related column not found. Please verify column names.")


# Split into features and target
X = train_df.drop(columns=['Calories'])
y = train_df['Calories']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate model
val_preds = model.predict(X_val)
mse = mean_squared_error(y_val, val_preds)
print(f"Validation MSE: {mse:.2f}")


# Predict on test set
test_preds = model.predict(test_df)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")

