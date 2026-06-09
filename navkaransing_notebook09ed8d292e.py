import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
train_data = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv")
test_data = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv")
sample_submission = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/sample_submission.csv")

# Separate features and target
X = train_data.drop(columns=["target"])
y = train_data["target"]

# Drop the 'id' column from the test set if it exists
X_test = test_data.drop(columns=["id"], errors="ignore")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,  # Number of trees
    max_depth=None,    # Maximum depth of the tree (None means no limit)
    min_samples_split=2,  # Minimum samples to split a node
    min_samples_leaf=1,   # Minimum samples per leaf
    random_state=42,
    n_jobs=-1           # Use all available CPU cores
)
rf_model.fit(X_train, y_train)

# Validate the model
y_val_pred = rf_model.predict(X_val)
validation_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
validation_r2 = r2_score(y_val, y_val_pred)

print(f"Validation RMSE: {validation_rmse}")
print(f"Validation R²: {validation_r2}")

# Predict on the test set
y_test_pred = rf_model.predict(X_test)

# Save predictions to submission file
submission = sample_submission.copy()
submission["target"] = y_test_pred
submission.to_csv("random_forest_submission.csv", index=False)

print("Submission file saved as 'random_forest_submission.csv'")


