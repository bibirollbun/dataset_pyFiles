import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer # Import SimpleImputer


# --- 1. Load Data ---
# (Keep your data loading code here, ensuring DATA_PATH is correct and files load)
DATA_PATH = "/kaggle/input/playground-series-s5e7/" # This is the standard path for this competition

try:
    train_df = pd.read_csv(DATA_PATH + "train.csv")
    test_df = pd.read_csv(DATA_PATH + "test.csv")
    sample_submission_df = pd.read_csv(DATA_PATH + "sample_submission.csv")
    print("Data loaded successfully!")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError:
    print(f"Error: One or more data files not found in {DATA_PATH}. Please check the path and file names.")
    import sys
    sys.exit("Data files not found. Exiting.")



# --- 2. Separate target variable and prepare IDs ---
X = train_df.drop("Personality", axis=1)
y = train_df["Personality"]

test_ids = test_df['id']
X = X.drop('id', axis=1)
test_df_processed = test_df.drop('id', axis=1)


# --- 3. Identify Categorical and Numerical Columns ---
# Iterate through columns to find non-numeric types
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical Features identified: {list(categorical_features)}")
print(f"Numerical Features identified: {list(numerical_features)}")



# --- 4. Preprocessing Pipeline ---

# Create numerical transformer (impute then passthrough)
# For numerical features, we'll impute missing values with the mean
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('passthrough', 'passthrough') # Keep numerical columns as they are after imputation
])

# Create categorical transformer (impute most frequent then one-hot encode)
# For categorical features, we'll impute missing values with the most frequent value
# It's important to impute *before* one-hot encoding, as OneHotEncoder doesn't handle NaNs directly.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Encode the target variable 'Personality' (Introvert/Extrovert) into numerical format
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nTarget classes mapping: {list(le.classes_)}")



# --- 5. Create a full pipeline (preprocessing + model) ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])


# --- 6. Model Training ---

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("\nTraining model...")
# Fit the pipeline (it will first preprocess X_train, including imputation, then train the classifier)
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- 7. Evaluation (on validation set) ---
y_val_pred = model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")




# --- 8. Prediction on Test Data ---
print("\nMaking predictions on the test set...")
test_predictions_encoded = model_pipeline.predict(test_df_processed)

# Convert numerical predictions back to original labels
test_predictions_labels = le.inverse_transform(test_predictions_encoded)
print("Predictions made.")




# --- 9. Create Submission File ---
submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions_labels})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

