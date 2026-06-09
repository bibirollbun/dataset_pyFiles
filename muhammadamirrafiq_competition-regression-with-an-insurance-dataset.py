import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load the Training Data
train_data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')

# Define categorical columns
categorical_columns = [
    'Education Level', 'Exercise Frequency', 
    'Occupation', 'Location', 'Policy Type', 'Smoking Status',
    'Property Type', 'Customer Feedback', 'Gender',
    'Marital Status'
]

# Preprocessing Function
def preprocess_data(df, label_encoders=None, is_train=True, latest_date=None):
    # Convert 'Policy Start Date' to datetime
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    
    if is_train:
        latest_date = df['Policy Start Date'].max()
    
    # Assign quarter intervals
    def assign_quarter_interval(policy_date, latest):
        if pd.isnull(policy_date):
            return -1  # Assign -1 for missing dates
        diff_years = latest.year - policy_date.year
        diff_months = latest.month - policy_date.month
        total_diff_months = diff_years * 12 + diff_months
        interval = total_diff_months // 3  # Integer division for quarterly intervals
        return interval
    
    df['Policy_Start_Interval'] = df['Policy Start Date'].apply(lambda x: assign_quarter_interval(x, latest_date))
    df['Policy_Start_Interval'] = df['Policy_Start_Interval'].astype(int)
    
    # Drop the original 'Policy Start Date' column
    df = df.drop('Policy Start Date', axis=1)
    
    # Encode categorical columns
    if is_train:
        label_encoders = {}
        for column in categorical_columns:
            # Handle missing values by filling with 'Missing'
            if df[column].isnull().any():
                df[column] = df[column].fillna('Missing')
            
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            label_encoders[column] = encoder
        return df, label_encoders, latest_date
    else:
        for column in categorical_columns:
            # Handle missing values by filling with 'Missing'
            if df[column].isnull().any():
                df[column] = df[column].fillna('Missing')
            
            # Use the existing encoder
            if label_encoders and column in label_encoders:
                encoder = label_encoders[column]
                # Handle unseen labels by assigning a new value
                df[column] = df[column].map(lambda s: '<unknown>' if s not in encoder.classes_ else s)
                # Update classes_ to include '<unknown>' if needed
                if '<unknown>' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, '<unknown>')
                df[column] = encoder.transform(df[column])
            else:
                raise ValueError(f'Label encoder for {column} not provided.')
        return df

# 2. Preprocess Training Data
train_data, label_encoders, latest_date = preprocess_data(train_data, is_train=True)

# Separate Features and Target
X_train_full = train_data.drop(['id', 'Premium Amount'], axis=1)
Y_train_full = train_data['Premium Amount']

# 3. Load and Preprocess Test Data
test_data = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')

# **Preserve the 'id' column before preprocessing**
test_ids = test_data['id'].copy()

# Preprocess Test Data using the same label encoders and latest_date from training
test_data_processed = preprocess_data(test_data, label_encoders=label_encoders, is_train=False, latest_date=latest_date)

# **Ensure 'id' is not included in the features for prediction**
# Drop 'id' from test_data_processed to create the feature set
X_test = test_data_processed.drop(['id'], axis=1, errors='ignore')  # Use 'errors="ignore"' in case 'id' was already dropped

# 4. Train the Model on the Full Training Data
# Log-transform the target variable
Y_train_full_log = np.log1p(Y_train_full)

# Initialize the XGBoost Regressor with early_stopping_rounds set in the constructor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,  # Moved here as per warning
    eval_metric='rmse',
    verbosity=1
)

# Since you want to train on the entire dataset, reserve a small portion for early stopping
# Here, we'll use a simple split for early stopping
X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
    X_train_full, Y_train_full_log, test_size=0.2, random_state=42
)

# Fit the model with early stopping
model.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val_part, y_val_part)],
    verbose=100
)

# Optional: Retrain on the full dataset without early stopping using the best iteration
# Uncomment the following lines if you prefer to train on all data after finding optimal rounds
optimal_n_estimators = model.best_iteration
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=optimal_n_estimators,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='rmse',
    verbosity=1
)
model.fit(
    X_train_full, Y_train_full_log,
    verbose=100
)

# 5. Make Predictions on the Test Data
preds_log_test = model.predict(X_test)
preds_test = np.expm1(preds_log_test)  # Reverse the log transformation

# 6. Prepare the Submission File
# **Use the preserved 'id's stored in test_ids**
submission = pd.DataFrame({
    'id': test_ids,  # Use the preserved 'id's
    'Premium Amount': preds_test
})

# Verify the submission format
print(submission.head())

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")








