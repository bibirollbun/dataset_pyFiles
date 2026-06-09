# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')

# Preview the data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# Drop 'id' column only if it exists
if 'id' in train.columns:
    train.drop('id', axis=1, inplace=True)

if 'id' in test.columns:
    test.drop('id', axis=1, inplace=True)

# Define the target column
target = 'NObeyesdad'

# Separate features and target
X = train.drop(target, axis=1)
y = train[target]

# Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)
test_encoded = pd.get_dummies(test, drop_first=True)

# Align test data columns with training data
missing_cols = set(X_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

test_encoded = test_encoded[X_encoded.columns]






# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier with tuned hyperparameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)


# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))



# Load the test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")

# Encode categorical features
test_encoded = pd.get_dummies(test_df)

# Align test columns with training columns
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Make predictions using the trained model
test_predictions = model.predict(test_encoded)

# Load sample submission file
submission = pd.read_csv("/kaggle/input/playground-series-s4e2/sample_submission.csv")

# Insert predictions
submission['NObeyesdad'] = test_predictions

# Save submission file
submission.to_csv("submission.csv", index=False)


