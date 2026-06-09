import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import time  # Import the time module
import optuna  # Import Optuna
import lightgbm as lgb
import catboost as cb


import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_df.head().style.background_gradient(cmap='plasma')


test_df.head().style.background_gradient(cmap='plasma')


submission_df.head().style.background_gradient(cmap='plasma')


train_df.describe().style.background_gradient(cmap='tab20c')


test_df.describe().style.background_gradient(cmap='tab20c')


print("------- Train Data Info --------")
train_df.info()
print("------- Test Data Info --------")
test_df.info()


#Check for missing values
print("\n--- Missing values in Train Data ---")
train_df.isnull().sum()


print("\n--- Missing values in Test Data ---")
test_df.isnull().sum()


# Impute missing values using SimpleImputer
numerical_cols = ['Compartments', 'Weight Capacity (kg)']
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


# Numerical imputation with median
num_imputer = SimpleImputer(strategy='median')
train_df[numerical_cols] = num_imputer.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = num_imputer.transform(test_df[numerical_cols])


# Categorical imputation with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])


# Verify no more missing values
print("\n--- Missing values in Train Data (after imputation) ---")
train_df.isnull().sum()


print("\n--- Missing values in Test Data (after imputation) ---")
test_df.isnull().sum()


# Impute missing 'Brand' in test_df using the most frequent value from train_df
most_frequent_brand = train_df['Brand'].mode()[0]
test_df['Brand'].fillna(most_frequent_brand, inplace=True)


# Encoding Categorical Features

# Identify categorical and numerical features
categorical_features = train_df.select_dtypes(include='object').columns.tolist()
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('Price')  # Remove target variable from numerical features
numerical_features.remove('id')

# Check if 'id' is in categorical_features before removing
if 'id' in categorical_features:
    categorical_features.remove('id')  

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing numerical values with mean
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Data Splitting

# Separate features (X) and target (y) from the training data
X = train_df.drop(['Price', 'id'], axis=1)
y = train_df['Price']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Fit and transform the training data using the preprocessor
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the validation data using the fitted preprocessor
X_val_processed = preprocessor.transform(X_val)

# Transform the test data using the fitted preprocessor
test_df_processed = preprocessor.transform(test_df.drop('id', axis=1))



# --- Model Training and Hyperparameter Tuning (CatBoost) ---

# Define the parameter grid - REDUCED for faster results
param_grid_cb = {
    'iterations': [200], 
    'learning_rate': [0.03], 
    'depth': [4],  
    'l2_leaf_reg': [1],  # L2 regularization
    'border_count': [20], # try different border count
    'random_strength': [0.5] # try different random strength
}


# Initialize CatBoost Regressor
cb_model = cb.CatBoostRegressor(random_state=42, verbose=0)  # verbose=0 to suppress output

# Initialize GridSearchCV
grid_search_cb = GridSearchCV(estimator=cb_model, param_grid=param_grid_cb, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

# Measure the start time
start_time_cb = time.time()

# Fit the grid search to the processed training data
grid_search_cb.fit(X_train_processed, y_train)

# Measure the end time
end_time_cb = time.time()

# Calculate the elapsed time
elapsed_time_cb = end_time_cb - start_time_cb
print(f"GridSearchCV (CatBoost) took {elapsed_time_cb:.2f} seconds")  # Print the elapsed time

# Get the best parameters
best_params_cb = grid_search_cb.best_params_
print(f"Best Parameters (CatBoost): {best_params_cb}")

# Get the best model
best_cb = grid_search_cb.best_estimator_


# --- Model Evaluation ---

# Predict on the validation set (CatBoost)
val_predictions = best_cb.predict(X_val_processed)

# Calculate RMSE on the validation set (CatBoost)
rmse_cb = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'\nRMSE on the Validation Set (CatBoost): {rmse_cb}')


# Make predictions on the processed test data (CatBoost)
test_predictions = best_cb.predict(test_df_processed)

# Create a submission DataFrame with the 'id' from test_df and the predictions
submission_df['Price'] = test_predictions

# Round the price predictions
submission_df['Price'] = submission_df['Price'].round(2)

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Display the head of the submission file
print('\nSubmission File Head:')
submission_df.head()


# --- Visualization ---

# 1: Distribution of Predicted Price

plt.figure(figsize=(10, 6))
sns.histplot(test_predictions, kde=True, color='mediumseagreen')
plt.title('Distribution of Predicted Price', fontsize=16)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# 2: Predicted Price vs. Weight Capacity (Test Data)

plt.figure(figsize=(12, 7))
sns.scatterplot(x=test_df['Weight Capacity (kg)'], y=test_predictions, color='darkkhaki', alpha=0.7)
plt.title('Predicted Price vs. Weight Capacity (Test Data)', fontsize=16)
plt.xlabel('Weight Capacity (kg)', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.show()

# 3: Distribution of Price: Train vs Validation vs Test

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.histplot(y_train, kde=True, color='royalblue')
plt.title('Train Data Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(y_val, kde=True, color='coral')
plt.title('Validation Data Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(test_predictions, kde=True, color='mediumseagreen')
plt.title('Test Data Predicted Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 4: Actual vs. Predicted Price on Validation Data
plt.figure(figsize=(10,6))
plt.scatter(y_val, val_predictions, alpha=0.5, color='rebeccapurple')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Price on Validation Data")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2) # Ideal prediction line
plt.show()




