import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# Enable GPU acceleration
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU


# Load datasets
train_path = '/kaggle/input/playground-series-s4e12/train.csv'
test_path = '/kaggle/input/playground-series-s4e12/test.csv'


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# Preprocessing and Modeling
X = train_df.drop(columns=['id', 'Premium Amount', 'Policy Start Date'])
y = train_df['Premium Amount']
test_ids = test_df['id']


# Define preprocessing
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])



# GPU-optimized XGBoost model
xgb_params = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'verbosity': 1
}

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(**xgb_params))
])




# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model with early stopping
model.fit(X_train, y_train,)



# Validate model
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"\nValidation RMSE: {rmse:.2f}")


# Generate predictions
test_preds = model.predict(test_df.drop(columns=['id', 'Policy Start Date']))


# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': test_preds
})


submission.to_csv('submission.csv', index=False)
print("Submission file created!")




