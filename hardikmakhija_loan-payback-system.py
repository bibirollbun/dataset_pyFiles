import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- 1. DATA LOADING (Kaggle Path) ---
# This path must match the folder name of your competition data source.
KAGGLE_INPUT_PATH = '/kaggle/input/playground-series-s5e11/'

try:
    df_train = pd.read_csv(KAGGLE_INPUT_PATH + 'train.csv')
    df_test = pd.read_csv(KAGGLE_INPUT_PATH + 'test.csv')
    df_submission_template = pd.read_csv(KAGGLE_INPUT_PATH + 'sample_submission.csv')
    print("Files loaded successfully.")
except FileNotFoundError:
    print("Error: Files not found at the standard Kaggle path. Please verify the directory name.")
    raise

# Define columns
ID_COLUMN_NAME = df_test.columns[0]
PREDICTION_COLUMN_NAME = df_submission_template.columns[1]
TARGET = 'loan_paid_back'

# --- 2. DATA PREPARATION ---
X_train = df_train.drop([ID_COLUMN_NAME, TARGET], axis=1)
y_train = df_train[TARGET]
X_test = df_test.drop(ID_COLUMN_NAME, axis=1)

numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

# --- 3. PREPROCESSING PIPELINE ---

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 4. OPTIMIZED RANDOM FOREST MODEL PIPELINE ---

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150,        # Reduced number of trees
        max_depth=15,             # Limit depth for speed
        min_samples_leaf=5,      # Enforce shallower leaves
        random_state=42,
        n_jobs=-1,               # Use all available cores
        verbose=0                # Turn off progress messages
    ))
])

# --- 5. TRAINING AND PREDICTION ---
print("\nStarting Optimized Random Forest model training...")
model_pipeline.fit(X_train, y_train)
print("Random Forest model training complete.")

# Generate predictions (probability of loan_paid_back)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# --- 6. SUBMISSION FILE CREATION ---
test_ids = df_test[ID_COLUMN_NAME]

df_submission = pd.DataFrame({
    ID_COLUMN_NAME: test_ids,
    PREDICTION_COLUMN_NAME: y_pred_proba
})

# Save the file as submission.csv
output_filename = 'submission.csv'
df_submission.to_csv(output_filename, index=False)

# --- 7. FINAL CONFIRMATION ---
print(f"\n✅ SUCCESS: Final submission file '{output_filename}' created.")
print(f"Model used: Optimized RandomForestClassifier.")
print(f"Rows in file: {len(df_submission)}")
print("\nFirst 5 rows of submission.csv:")
print(df_submission.head())

