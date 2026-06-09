import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import scikit-learn modules for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load the training dataset ("train.csv")
train_df = pd.read_csv("/kaggle/input/steel-quality-challenge/train.csv")
print("Training Dataset Shape:", train_df.shape)
train_df.head()


# List of features we will use for modeling.
# Make sure these match the column names in your dataset.
features = [
    "plate_thickness", "plate_length", "min_luminosity", "defect_area",
    "brightness_index", "edge_index", "square_index", "total_luminosity",
    "cooling_rate", "processing_time", "temperature"
]

# The target variable is quality_score (scaled between 0 and 1)
target = "quality_score"


# Create the feature matrix X and target vector y from training data
x_train = train_df[features].values
y_train = train_df[target].values


train_df[features].head() # x_train


train_df[[target]].head() # y_train


# Split the data into training and validation sets (80% training, 20% test)
# x_train, x_val, y_train, y_val = train_test_split(
#    x_train, y_train, test_size=0.2, random_state=42
#)

# Scale the features so that each feature has mean=0 and std=1
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# x_val_scaled = scaler.transform(x_val)

pd.DataFrame(x_train_scaled, columns=features).head()


# We use RandomForestRegressor for its ease of use and good performance out-of-box.

# Creating the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Trainig the Model
model.fit(x_train_scaled, y_train)


'''
# Predict on validation set
y_val_pred = model.predict(x_test_scaled)

# Calculate evaluation metrics
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("\nValidation Performance:")
print(f"  RMSE: {val_rmse:.4f}")

'''


# Load the test dataset
test_df = pd.read_csv("/kaggle/input/steel-quality-challenge/test.csv")
print("Test Dataset Shape:", test_df.shape)
test_df.head()


# Extract and scale the features from test data
x_submission = test_df[features].values
x_submission_scaled = scaler.transform(x_submission)

pd.DataFrame(x_submission_scaled, columns=features) #.head()


# Predict the quality score for test set
y_submission_pred = model.predict(x_submission_scaled)

pd.DataFrame(y_submission_pred, columns=[target]) # .head()


# Extract IDs for submission
ids = test_df["id"]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    "id": ids,
    "quality_score": y_submission_pred
})

submission_df


# Save the DataFrame as a CSV file
submission_df.to_csv("submission.csv", index=False)
print("submission.csv has been created!")

