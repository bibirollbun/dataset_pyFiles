# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

# Combine train and extra
train_full = pd.concat([train, extra], axis=0).reset_index(drop=True)

# Drop id column
train_full.drop(columns=['id'], inplace=True, errors='ignore')
test_ids = test['id']
test.drop(columns=['id'], inplace=True, errors='ignore')

# Separate features and target
X = train_full.drop('Price', axis=1)
y = train_full['Price']
X_test = test.copy()

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Ensemble Model
voting = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
])

# Full Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', voting)
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate RMSE on validation set
y_valid_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test set
preds = pipeline.predict(X_test)

# Save Submission
submission['Price'] = preds
submission.to_csv('submission.csv', index=False)
print("Submission.csv is ready for upload!")

