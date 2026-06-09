#install scikit learn 1.5.2 as this version supports root_mean_squared_log_error
!pip uninstall scikit-learn -y
!pip install -q scikit-learn==1.5.2


import sklearn
sklearn.__version__


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


train_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/train.csv')
test_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/test.csv')


# Save 'id' column for submission
test_ids = test_df['id']

# Define the target column
target_column = 'Premium Amount'


# Select categorical and numerical columns (initial)
categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

# Print out column information
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


# Split train data into features and target
X = train_df.drop(columns=[target_column, "id"])
y = train_df[target_column]


categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns


# Preprocessing pipeline for numerical features
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Preprocessing pipeline for categorical features
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # Encode categorical features
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_columns),
        ('cat', cat_pipeline, categorical_columns)
    ]
)

# Preprocess train and test data
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test_df.drop(columns=['id']))


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)


# --- 1. Linear Regression ---
print("--- Training Linear Regression ---")
# Instantiate the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)
print("Linear Regression training complete.")

# Predict on the validation set
y_pred_lr = lr_model.predict(X_val)
print("-" * 30)


def calculate_rmsle(y_true, y_pred):
    """Calculates RMSLE, clipping negative predictions to 0."""
    if np.any(y_true < 0):
        print(f"Warning: Negative true values found. RMSLE calculation may fail or be invalid.")

    y_pred_clipped = np.maximum(y_pred, 0)

    msle = mean_squared_log_error(y_true, y_pred_clipped)
    rmsle = np.sqrt(msle)
    return rmsle

print("--- Evaluating Models using RMSLE ---")

# Evaluate Linear Regression
rmsle_lr = calculate_rmsle(y_val, y_pred_lr)
print(f"Linear Regression RMSLE: {rmsle_lr:.4f}")


sample_submission = pd.read_csv("/kaggle/input/big-oai-final-course-1/sample_submission.csv")
sample_submission.head()


# Predict
y_pred = lr_model.predict(test_processed)


submission_df = pd.DataFrame()

if 'id' in test_df.columns:
    submission_df['id'] = test_df['id'].values
else:
    submission_df['id'] = sample_submission['id'].values

target_column_name = sample_submission.columns[1] # Assumes target is the second column
submission_df[target_column_name] = y_pred

# Save
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Submission file '{submission_filename}' created.")
print(submission_df.head())

