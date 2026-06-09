# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import scikit-learn modules for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load the training dataset
train_df = pd.read_csv("/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv")

print("âœ… Training Dataset Loaded!")
print("Shape:", train_df.shape)
train_df.head()


# List of features for model training
# Ensure these match the column names in the provided dataset
features = [
    "soil_ph", "soil_moisture", "avg_temperature", "total_rainfall",
    "fertilizer_amount", "pesticide_usage", "sunlight_hours",
    "nitrogen_content", "phosphorus_content", "potassium_content",
    "irrigation_frequency"
]

# The target variable is yield_tpha (tons per hectare)
target = "yield_tpha"

# Separate features and target
x_train = train_df[features].values
y_train = train_df[target].values

print("Feature Matrix Shape:", x_train.shape)
print("Target Vector Shape:", y_train.shape)

train_df[features].head()
train_df[[target]].head()


# Split data into training and validation sets (optional)
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train, y_train, test_size=0.2, random_state=42
# )

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# x_val_scaled = scaler.transform(x_val)

pd.DataFrame(x_train_scaled, columns=features).head()


# Initialize a Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(x_train_scaled, y_train)

print("âœ… Model Training Complete!")


"""
# Uncomment this block if you created a validation split above.

y_val_pred = model.predict(x_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("\nValidation Performance:")
print(f"  RMSE: {val_rmse:.4f}")
"""


# Load the test dataset
test_df = pd.read_csv("/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv")

print("âœ… Test Dataset Loaded!")
print("Shape:", test_df.shape)
test_df.head()

# Extract and scale the test features
x_submission = test_df[features].values
x_submission_scaled = scaler.transform(x_submission)

# Predict yield for the test data
y_submission_pred = model.predict(x_submission_scaled)

# Preview predictions
pd.DataFrame(y_submission_pred, columns=[target]).head()


# Create submission DataFrame
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "yield_tpha": y_submission_pred
})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv created successfully!")
submission_df.head()

