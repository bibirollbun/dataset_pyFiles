import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.info()


train_df


test_df.info()


target = train_df['accident_risk']
test_ids = test_df['id']

# Drop 'id' and 'accident_risk' from the training set
train_df = train_df.drop(columns=['id', 'accident_risk'])
test_df = test_df.drop(columns=['id'])


# Concatenate train and test data for consistent encoding
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Identify boolean and object columns
bool_cols = combined_df.select_dtypes(include=['bool']).columns
object_cols = combined_df.select_dtypes(include=['object']).columns

# Convert boolean columns to integer 0/1
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)

# One-Hot Encoding for object type columns
# This handles 'road_type', 'lighting', 'weather', and 'time_of_day'
combined_df = pd.get_dummies(combined_df, columns=object_cols, drop_first=True)

# Separate the combined data back into training and testing sets
X_train = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]
y_train = target


# Using a modest set of hyper-parameters for efficiency on a large dataset
rfr = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Limiting depth to speed up training and reduce overfitting
    min_samples_split=20,  # Minimum samples required to split an internal node
    random_state=42,       # For reproducibility
    n_jobs=-1,              # Use all avail    
    #min_samples_leaf=8,     # Additional regularization
    max_features=0.5      # Feature subsampling
    
)

print("Starting model training (RandomForestRegressor)...")
rfr.fit(X_train, y_train)
print("Model training complete.")


# Generate predictions on the test set
predictions = rfr.predict(X_test)

# Ensure predictions are between 0 and 1
predictions = np.clip(predictions, 0, 1)

# Create the submission 
submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created.")
print(submission_df.head())

