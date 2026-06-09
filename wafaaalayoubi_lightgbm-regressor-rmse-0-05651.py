# --- Core Libraries ---
import numpy as np
import pandas as pd
import warnings

# --- Machine Learning Libraries ---
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# --- Tweak Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# --- Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
except FileNotFoundError:
    print("Please adjust file paths for local execution.")


print("Data loaded successfully!")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# --- Display a sample of the data to confirm ---
print("\n--- Training Data Head ---")
display(train_df.head())

print("\n--- Test Data Head ---")
display(test_df.head())


# --- Store test IDs for submission ---
test_ids = test_df['id']

# --- Drop unnecessary columns ---
# Drop 'id' because it's an identifier.
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

print("Dropped 'id' and 'num_reported_accidents' columns.")
print(f"New train shape: {train_df.shape}")
print(f"New test shape: {test_df.shape}\n")


# --- Define feature types based on EDA ---
# Note: We are now treating 'num_lanes' as categorical
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'num_lanes']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numerical_features = ['curvature', 'speed_limit']
target = 'accident_risk'


# --- Convert boolean features to integers (0 or 1) ---
for col in boolean_features:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

print("Converted boolean features to integers.")


# --- Apply One-Hot Encoding to categorical features ---
# Using pd.get_dummies is a straightforward way to do this
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=False)
test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=False)

# Ensure both dataframes have the same columns after one-hot encoding
train_labels = train_df[target]
train_ids, test_ids_align = train_df.align(test_df, join='inner', axis=1, copy=False) # Important for consistency

print("Applied One-Hot Encoding.\n")


# --- Separate features (X) and target (y) ---
X = train_df.drop(columns=[target])
y = train_df[target]
X_test = test_df


# --- Display the final processed data ---
print(f"Final training features shape: {X.shape}")
print(f"Final test features shape: {X_test.shape}")
print("\n--- Processed Training Data Head ---")
display(X.head())


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Split the data into a training set (80%) and a validation set (20%) ---
# We use shuffle=False to simulate a time-series split, which is a robust practice.
# Using a fixed random_state ensures the split is the same every time we run the code.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# --- 2. Initialize the LightGBM Regressor ---
lgbm = lgb.LGBMRegressor(
    objective='regression_l1',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    seed=42,
    n_jobs=-1,
    verbose=-1
)


# --- 3. Train the model ONLY on the training portion ---
print("\nTraining model on 80% of the data...")
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         eval_metric='rmse',
         callbacks=[lgb.early_stopping(100, verbose=False)])

print("Model training complete!")


# --- 4. Make predictions on the validation set (the 20% holdout) ---
print("Making predictions on the validation set...")
val_predictions = lgbm.predict(X_val)


# --- 5. Calculate and display the RMSE ---
rmse = mean_squared_error(y_val, val_predictions, squared=False)

print("\n==============================================")
print(f"         VALIDATION RMSE: {rmse:.5f}         ")
print("==============================================")


# --- 1. Initialize the Final Model ---
# We use the exact same parameters we validated earlier for consistency.
final_lgbm = lgb.LGBMRegressor(
    objective='regression_l1',
    metric='rmse',
    n_estimators=1000, # We could use the 'best_iteration' from early stopping, but 1000 is a solid number for a baseline.
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    seed=42,
    n_jobs=-1,
    verbose=-1
)


# --- 2. Train on the FULL Training Dataset ---
print("Training the final model on 100% of the data...")
final_lgbm.fit(X, y)
print("Final model training complete!")


# --- 3. Make Predictions on the Test Data ---
print("Making final predictions on the test data...")
final_predictions = final_lgbm.predict(X_test)


# --- 4. Create the Submission File ---
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': final_predictions})

# Good practice: Clip predictions to the valid [0, 1] range.
submission_df['accident_risk'] = submission_df['accident_risk'].clip(0, 1)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("\n--- Submission File Head ---")
display(submission_df.head())




