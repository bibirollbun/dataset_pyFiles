# ==============================================================================
#  1. SETUP AND IMPORT LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Machine Learning Model
import lightgbm as lgb

# Set some visual styles for plots
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

print("âœ… Libraries imported successfully!")


# ==============================================================================
#  2. LOAD THE DATA
# ==============================================================================
try:
    # Define file paths (assuming this is a new competition)
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_submission_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'

    # Load data into pandas DataFrames
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(sample_submission_path)

    print("âœ… Data loaded successfully!")
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

except FileNotFoundError as e:
    print(f"â�Œ ERROR: {e}")


# ==============================================================================
#  3. EXPLORATORY DATA ANALYSIS (EDA) - CORRECTED
# ==============================================================================
print("\n--- Training Data Info ---")
train_df.info()

# CORRECTED: Visualize the actual target variable 'Personality'
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_df, palette=['#4e79a7', '#f28e2b'])
plt.title('Distribution of Personality Types (Target Variable)')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()

# Keep the test 'id' for the submission file
test_ids = test_df['id']
# Drop the 'id' column from training and test sets as it's not a feature
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# ==============================================================================
#  4. FEATURE ENGINEERING & PREPROCESSING - CORRECTED
# ==============================================================================
# Define features (X) and target (y)
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# **NEW**: Encode the target variable ('Introvert'/'Extrovert') into numbers (0/1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numerical features from the new dataset
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nCategorical Features: {categorical_features}")
print(f"Numerical Features: {numerical_features}")

# **NEW & CRITICAL**: Create preprocessing pipelines to handle missing data and scaling

# Pipeline for numerical features:
# 1. Impute missing values with the median
# 2. Scale features to a standard range
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features:
# 1. Impute missing values with the most frequent value
# 2. One-Hot Encode the categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine these pipelines into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\nâœ… Preprocessing pipeline created with imputation and scaling.")


# ==============================================================================
#  5. MODEL TRAINING - CORRECTED
# ==============================================================================
# Define the model. 'binary' objective is correct for two personality types.
lgbm = lgb.LGBMClassifier(random_state=42, objective='binary')

# Create the full model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm)
])

print("\nâ�³ Training the LightGBM model...")
# Train the model on the feature data (X) and the *encoded* target (y_encoded)
model_pipeline.fit(X, y_encoded)

print("âœ… Model training complete!")


# ==============================================================================
#  6. PREDICTION AND SUBMISSION FILE GENERATION - CORRECTED
# ==============================================================================
print("\nMaking predictions on the test data...")

# Generate predictions (these will be 0s and 1s)
encoded_predictions = model_pipeline.predict(test_df)

# **NEW**: Convert the numerical predictions back to original labels ('Introvert'/'Extrovert')
final_predictions = label_encoder.inverse_transform(encoded_predictions)

print("âœ… Predictions generated and decoded.")

# Create the submission file
submission_output = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})

# Save the submission file
submission_output.to_csv('submission.csv', index=False)

print("\nâœ… Submission file 'submission.csv' created successfully!")
print("--- First 5 Rows of Submission File ---")
display(submission_output.head())

print("\nğŸš€ Notebook execution finished! ğŸš€")

