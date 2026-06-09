!pip install lightgbm==3.3.5



# Import necessary libraries
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping

# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')  # Replace with the actual path if needed
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')   # Replace with the actual path if needed

# Preprocessing
# Convert 'date' to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# One-hot encode categorical features
categorical_features = ['country', 'store', 'product']
train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)

# Extract features from the date column
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['dayofweek'] = train['date'].dt.dayofweek

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek

# Define features and target
features = [col for col in train.columns if col not in ['date', 'num_sold', 'id']]
target = 'num_sold'

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Set model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
model = lgb.train(params,
                train_data,
                num_boost_round=100000,
                valid_sets=[val_data],
                callbacks=[early_stopping(stopping_rounds=200)],  # Pass early stopping as a callback
                verbose_eval=100)

# Make predictions on the test set
predictions = model.predict(test[features])

# Create submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'num_sold': predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")




