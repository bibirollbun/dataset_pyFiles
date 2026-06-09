import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


# Set random seed for reproducibility
np.random.seed(42)


# Load data
train_file = '/kaggle/input/playground-series-s5e7/train.csv'
test_file = '/kaggle/input/playground-series-s5e7/test.csv'
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)



# Print columns and head for debugging
print("Train dataset columns:", train_df.columns.tolist())
print("Train dataset head:\n", train_df.head())
print("Test dataset columns:", test_df.columns.tolist())
print("Test dataset head:\n", test_df.head())


# Separate features and target
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Encode target
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id'].values  # Preserve test IDs

# Handle categorical variables
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'No': 0, 'Yes': 1})
    X_test[col] = X_test[col].map({'No': 0, 'Yes': 1})
    
    # Impute missing values in categorical columns with mode
    imputer = SimpleImputer(strategy='most_frequent')
    X[col] = imputer.fit_transform(X[[col]]).ravel()
    X_test[col] = imputer.transform(X_test[[col]]).ravel()

# Handle numerical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train LightGBM with provided best parameters
best_params = {
    'learning_rate': 0.023533042331948473,
    'max_depth': 18,
    'min_child_samples': 24,
    'n_estimators': 289,
    'num_leaves': 81,
    'subsample': 0.8925879806965068,
    'random_state': 42,
    'verbose': -1
}
best_model = LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)

# Validate best model
val_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy with Best Model: {accuracy:.4f}')


# Predict on test set
test_pred = best_model.predict(X_test)

# Reverse encode predictions to original labels
reverse_mapping = {0: 'Introvert', 1: 'Extrovert'}
test_pred_labels = [reverse_mapping[pred] for pred in test_pred]


# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_pred_labels
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


# Preview submission
print("\nSubmission preview:\n", submission.head())

