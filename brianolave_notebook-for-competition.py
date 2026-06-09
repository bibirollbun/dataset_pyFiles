import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss

# Load the datasets
df = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/train.csv')
test_data = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/test.csv')

# Store test IDs before processing
test_ids = test_data['id']


# Data Overview
print(df.info())
print(df.describe())
print(df.isnull().sum())


# Convert date columns to datetime format
df['orderDate'] = pd.to_datetime(df['orderDate'], errors='coerce')
df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], errors='coerce')
df['creationDate'] = pd.to_datetime(df['creationDate'], errors='coerce')
df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')

# Feature Engineering
df['deliveryTime'] = (df['deliveryDate'] - df['orderDate']).dt.days
df['pre_age'] = (df['creationDate'] - df['dateOfBirth']).dt.days / 365
df['price_to_age_ratio'] = df['price'] / df['pre_age']
df['time_since_last_purchase'] = (df['orderDate'] - df.groupby('customerID')['orderDate'].shift(1)).dt.days
df['order_quarter'] = df['orderDate'].dt.quarter
df['itemID_size'] = df.groupby('itemID')['itemID'].transform('size')
df['total_purchases'] = df.groupby('customerID')['price'].transform('sum')
df['total_spent'] = df.groupby('customerID')['price'].transform('sum')
df['purchase_frequency'] = df.groupby('customerID')['orderDate'].transform('count')


# Encode Categorical Variables
categorical_columns = ['size', 'color', 'salutation', 'state']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Handle missing values
df.fillna(0, inplace=True)

# Splitting Data
X = df.drop(['returnShipment', 'orderDate', 'deliveryDate', 'creationDate', 'dateOfBirth'], axis=1, errors='ignore')
y = df['returnShipment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Random Forest Model
# model = RandomForestClassifier(random_state=42, n_estimators=100)
# model.fit(X_train, y_train)


# Train Gradient Boosting Model
model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)


# Compute Brier Score
val_predictions = model.predict_proba(X_val)[:, 1]
brier_score = brier_score_loss(y_val, val_predictions)
print(f"Brier Score: {brier_score}")

# Preprocess Test Data
test_data = test_data.drop(columns=['orderDate', 'deliveryDate', 'creationDate', 'dateOfBirth'], errors='ignore')
test_data = pd.get_dummies(test_data, drop_first=True)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)
test_data.fillna(0, inplace=True)


# Ensure test_data retains the correct number of rows
if len(test_data) != 96219:
    print(f"Warning: test_data has {len(test_data)} rows instead of 96,219!")

# Generate Predictions
test_predictions = model.predict_proba(test_data)[:, 1]

# Prepare submission file
submission = pd.DataFrame({
    'id': test_ids,  # Restore original ID column
    'returnShipment': test_predictions
})

# Ensure submission file has the correct number of rows
assert len(submission) == 96219, "Submission file does not have 96,219 rows!"

# Save the submission file with proper formatting
submission.to_csv('submission.csv', index=False, float_format='%.6f', encoding='utf-8')
print("submission.csv has been created successfully with the correct number of rows and format.")

