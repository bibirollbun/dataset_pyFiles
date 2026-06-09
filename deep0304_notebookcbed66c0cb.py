import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# --- 1. Data Preprocessing and Feature Engineering ---

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    # Use fit_transform on the training data
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Feature Engineering
# Handle potential NaN in delay columns before creating the new feature
train_data['Arrival Delay in Minutes'].fillna(train_data['Arrival Delay in Minutes'].mean(), inplace=True)
train_data['Total_Delay'] = train_data['Departure Delay in Minutes'] + train_data['Arrival Delay in Minutes']

# Create a composite score for inflight experience
inflight_service_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
train_data['Inflight_Experience'] = train_data[inflight_service_cols].mean(axis=1)


# Define features and target variable
# Drop original columns that were used to create new features to avoid redundancy
X = train_data.drop(columns=[
    'Unnamed: 0', 'id', 'satisfaction', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
])
y = train_data['satisfaction']


# --- 2. Corrected Splitting and Imputation Workflow ---

# Split the data into training and validation sets BEFORE imputation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values with SimpleImputer
# Fit the imputer ONLY on the training data to prevent data leakage
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# Transform the validation data using the imputer fitted on the training data
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)


# --- 3. Train a LightGBM Classifier ---

# Initialize the model with good starting hyperparameters
model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    n_estimators=1000,          # More trees
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.8,       # Feature subsampling
    subsample=0.8               # Data subsampling
)

# Train the model with early stopping to prevent overfitting
# Early stopping will monitor the validation set and stop training if performance doesn't improve
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='accuracy',
          callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if accuracy doesn't improve for 100 rounds

# --- 4. Validate the New Model ---
y_pred = model.predict(X_val)
print(f"New Model Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")


# --- 5. Preprocess Test Data and Make Predictions ---

# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Preprocess the test dataset exactly like the training data
# Encode categorical variables
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Apply the same feature engineering
solution['Total_Delay'] = solution['Departure Delay in Minutes'] + solution['Arrival Delay in Minutes']
solution['Inflight_Experience'] = solution[inflight_service_cols].mean(axis=1)

# Select features for prediction, ensuring the columns match the training data
X_test = solution.drop(columns=['Unnamed: 0', 'id', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], errors='ignore')

# Ensure test columns are in the same order as training columns
X_test = X_test[X_train.columns]

# Handle missing values using the imputer fitted on the training data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Make predictions
solution['satisfaction'] = model.predict(X_test)

# Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' created successfully.")
solution[['ID', 'satisfaction']].head()

