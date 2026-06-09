import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# Load pre-split data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Preprocess: Encode 'Sex' (e.g., 0/1)
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})


display(train_data)


display(test_data)


# Define features (X) and target (y)
X_train = train_data.drop(["id", "Calories"], axis=1)
y_train = train_data["Calories"]
X_test = test_data.drop(["id"], axis=1)

# Train model
model = GradientBoostingRegressor(random_state=42) 
model.fit(X_train, y_train)


# Generate predictions
predictions = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_data["id"],  # Preserve original IDs
    "Calories": predictions.round(2)  # Round to 2 decimal places
})

display(submission)


# Save to CSV (without index)
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")

