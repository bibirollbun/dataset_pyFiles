# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import necessary libraries
import pandas as pd

# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

# Display the first few rows of each dataset
print("Training Data Sample:")
print(train_data.head())
print("\nTest Data Sample:")
print(test_data.head())
print("\nExtra Training Data Sample:")
print(train_extra_data.head())



# Check data types and missing values in each dataset
print("Training Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())
print("\nExtra Training Data Info:")
print(train_extra_data.info())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())
print("\nMissing Values in Extra Training Data:")
print(train_extra_data.isnull().sum())



# Handling Missing Values

# For categorical columns, fill missing values with 'Unknown'
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                       'Waterproof', 'Style', 'Color']

for col in categorical_columns:
    train_data[col].fillna('Unknown', inplace=True)
    test_data[col].fillna('Unknown', inplace=True)
    train_extra_data[col].fillna('Unknown', inplace=True)

# For numerical columns, fill missing values with the mean
numerical_columns = ['Weight Capacity (kg)']

for col in numerical_columns:
    train_data[col].fillna(train_data[col].mean(), inplace=True)
    test_data[col].fillna(test_data[col].mean(), inplace=True)
    train_extra_data[col].fillna(train_extra_data[col].mean(), inplace=True)

# Verify that there are no more missing values
print("Missing Values in Training Data After Handling:")
print(train_data.isnull().sum())
print("\nMissing Values in Test Data After Handling:")
print(test_data.isnull().sum())
print("\nMissing Values in Extra Training Data After Handling:")
print(train_extra_data.isnull().sum())



# Basic Statistics for Numerical Columns

# Describe numerical columns in the training data
print("Training Data - Numerical Columns Statistics:")
print(train_data.describe())

# Describe numerical columns in the test data
print("\nTest Data - Numerical Columns Statistics:")
print(test_data.describe())

# Describe numerical columns in the extra training data
print("\nExtra Training Data - Numerical Columns Statistics:")
print(train_extra_data.describe())



# Explore unique values in categorical columns for the training data
print("Unique Values in Categorical Columns (Training Data):")
for col in categorical_columns:
    print(f"\n{col}:")
    print(train_data[col].value_counts())
    
# Explore unique values in categorical columns for the test data
print("\nUnique Values in Categorical Columns (Test Data):")
for col in categorical_columns:
    print(f"\n{col}:")
    print(test_data[col].value_counts())



# Import necessary library for encoding
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode binary categorical variables
binary_columns = ['Laptop Compartment', 'Waterproof']
for col in binary_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])
    train_extra_data[col] = label_encoder.transform(train_extra_data[col])

# One-hot encode nominal categorical variables
train_data = pd.get_dummies(train_data, columns=['Brand', 'Material', 'Size', 'Style', 'Color'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Brand', 'Material', 'Size', 'Style', 'Color'], drop_first=True)
train_extra_data = pd.get_dummies(train_extra_data, columns=['Brand', 'Material', 'Size', 'Style', 'Color'], drop_first=True)

# Display the new shape of the datasets
print("Training Data Shape After Encoding:", train_data.shape)
print("Test Data Shape After Encoding:", test_data.shape)
print("Extra Training Data Shape After Encoding:", train_extra_data.shape)



# Get the list of columns in training and test data
train_cols = set(train_data.columns)
test_cols = set(test_data.columns)

# Find the missing columns
missing_cols_train = list(test_cols - train_cols)
missing_cols_test = list(train_cols - test_cols)

print("Missing Columns in Training Data:", missing_cols_train)
print("Missing Columns in Test Data:", missing_cols_test)

# Add the missing columns in respective datasets and fill with 0
for col in missing_cols_train:
    train_data[col] = 0
for col in missing_cols_test:
    test_data[col] = 0

# Verify the shapes of the datasets
print("\nTraining Data Shape After Fix:", train_data.shape)
print("Test Data Shape After Fix:", test_data.shape)
print("Extra Training Data Shape After Encoding:", train_extra_data.shape)



from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Identify numerical columns (including the one-hot encoded columns)
numerical_cols = ['Compartments', 'Weight Capacity (kg)'] + [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64'] and col not in ['Price', 'id']]

# Fit and transform the training data
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# Transform the test data
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

# Transform the extra training data
train_extra_data[numerical_cols] = scaler.transform(train_extra_data[numerical_cols])

# Display the first few rows of the scaled training data
print("Scaled Training Data Sample:")
print(train_data.head())



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the data
X = train_data.drop(['Price', 'id'], axis=1)  # Drop 'Price' (target) and 'id'
y = train_data['Price']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Root Mean Squared Error (RMSE) on the Validation Set:", rmse)



from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = rf_model.predict(X_val)

# Evaluate the model using RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Root Mean Squared Error (RMSE) on the Validation Set (Random Forest):", rf_rmse)



# Prepare the test data
X_test = test_data.drop(['id', 'Price'], axis=1)

# Make predictions on the test data
test_pred = model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'id': test_data['id'], 'Price': test_pred})

print(submission.head())


# Save the submission file
#submission.to_csv('submission_lg.csv', index=False)

print("Submission file created successfully!")



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the data
X = train_data.drop(['Price', 'id'], axis=1)  # Drop 'Price' (target) and 'id'
y = train_data['Price']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100,  # Number of boosting rounds
                             learning_rate=0.1,   # Step size shrinkage
                             max_depth=3,         # Maximum depth of a tree
                             random_state=42)      # Random seed for reproducibility

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = xgb_model.predict(X_val)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Root Mean Squared Error (RMSE) on the Validation Set (XGBoost):", rmse)



from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np

# Prepare the data
X = train_data.drop(['Price', 'id'], axis=1)
y = train_data['Price']

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1)

# Perform Grid Search
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# Get the best model
best_xgb_model = grid_search.best_estimator_



# Prepare the test data
X_test = test_data.drop(['id', 'Price'], axis=1)

# Get the feature names used during training
train_features = list(X_train.columns)

# Ensure that the test data has the same columns in the same order as the training data
X_test = X_test[train_features]

# Make predictions on the test data using the best model
test_pred = best_xgb_model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'id': test_data['id'], 'Price': test_pred})

# Save the submission file
#submission.to_csv('submission_xgboost.csv', index=False)

print("Submission file 'submission_xgboost.csv' created successfully!")



from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def create_polynomial_features(df, numerical_cols, degree=2):
    """
    Creates polynomial features from specified numerical columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numerical_cols (list): List of numerical column names to create polynomial features from.
        degree (int): Degree of the polynomial features.

    Returns:
        pd.DataFrame: DataFrame with added polynomial features.
    """

    # Select the numerical columns
    numerical_data = df[numerical_cols]

    # Initialize PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)

    # Fit and transform the data
    poly_features = poly.fit_transform(numerical_data)

    # Create column names
    feature_names = poly.get_feature_names_out(numerical_cols)

    # Create a DataFrame from the polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

    # Concatenate with original DataFrame
    df = pd.concat([df, poly_df], axis=1)

    return df



# Select numerical columns to create polynomial features from
numerical_cols = ['Compartments', 'Weight Capacity (kg)']

# Create polynomial features for training data
train_data = create_polynomial_features(train_data, numerical_cols, degree=2)

# Create polynomial features for test data
test_data = create_polynomial_features(test_data, numerical_cols, degree=2)

# Create polynomial features for extra training data
train_extra_data = create_polynomial_features(train_extra_data, numerical_cols, degree=2)

print("Training Data Shape After Polynomial Feature Creation:", train_data.shape)
print("Test Data Shape After Polynomial Feature Creation:", test_data.shape)
print("Extra Training Data Shape After Polynomial Feature Creation:", train_extra_data.shape)



from sklearn.feature_selection import SelectKBest, f_regression

def select_top_features(X, y, k=20):
    """
    Selects the top k features from a DataFrame using SelectKBest.

    Args:
        X (pd.DataFrame): Input DataFrame of features.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.

    Returns:
        tuple: (DataFrame with selected features, list of selected feature names)
    """

    # Initialize SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k)

    # Fit SelectKBest on the training data
    selector.fit(X, y)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X.columns[selected_indices].tolist()

    # Transform the data to include only the selected features
    X_selected = X[selected_features]

    return X_selected, selected_features



# Prepare the data
X = train_data.drop(['Price', 'id'], axis=1)
y = train_data['Price']

# Select top 20 features
X_selected, selected_features = select_top_features(X, y, k=20)

# Update the training data
train_data_selected = train_data[['Price', 'id'] + selected_features]

# Update the test data
test_data_selected = test_data[['id'] + selected_features]

# Update the extra training data
train_extra_data_selected = train_extra_data[['Price', 'id'] + selected_features]

print("Training Data Shape After Feature Selection:", train_data_selected.shape)
print("Test Data Shape After Feature Selection:", test_data_selected.shape)
print("Extra Training Data Shape After Feature Selection:", train_extra_data_selected.shape)



# Check for duplicate feature names
duplicate_features = train_data_selected.columns[train_data_selected.columns.duplicated()].unique()
print("Duplicate Features:", duplicate_features)


# Function to rename duplicate columns
def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# Rename duplicate features in the training and test data
train_data_selected = rename_duplicate_columns(train_data_selected)
test_data_selected = rename_duplicate_columns(test_data_selected)

# Check again for duplicates
duplicate_features_after_rename = train_data_selected.columns[train_data_selected.columns.duplicated()].unique()
print("Duplicate Features After Rename:", duplicate_features_after_rename)



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
X = train_data_selected.drop(['Price', 'id'], axis=1)
y = train_data_selected['Price']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# Train the model with early stopping
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round=100,
                      valid_sets=[lgb_train, lgb_val],  # Evaluate on training and validation sets
                      callbacks=[lgb.early_stopping(stopping_rounds=10)]) # Stop if no improvement after 10 rounds

# Make predictions on the validation set
y_pred_val = lgb_model.predict(X_val)

# Evaluate the model using RMSE
rmse_val = mean_squared_error(y_val, y_pred_val) ** 0.5
print("Root Mean Squared Error (RMSE) on the Validation Set:", rmse_val)

# Prepare the test data
X_test = test_data_selected.drop(['id'], axis=1)

# Make predictions on the test data
test_pred = lgb_model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'id': test_data_selected['id'], 'Price': test_pred})

# Save the submission file
#submission.to_csv('submission_lightgbm.csv', index=False)

print("Submission file 'submission_lightgbm.csv' created successfully!")



import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the data
X = train_data_selected.drop(['Price', 'id'], axis=1)
y = train_data_selected['Price']

# Step 1: Summary Statistics
print("Summary Statistics:")
print(train_data_selected.describe())

# Step 2: Distribution of Target Variable
plt.figure(figsize=(10, 5))
sns.histplot(train_data_selected['Price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Step 3: Check for Missing Values
missing_values = train_data_selected.isnull().sum()
print("Missing Values:")
print(missing_values[missing_values > 0])

# Step 4: Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = train_data_selected.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Visualizations for Important Features
# Example: Scatter plot between 'Weight Capacity (kg)' and 'Price'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data_selected, x='Weight Capacity (kg)', y='Price')
plt.title('Weight Capacity vs Price')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Price')
plt.show()



# Get feature importances from the trained LightGBM model
feature_importances = lgb_model.feature_importance(importance_type='split')
feature_names = X_train.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))  # Top 20 features
plt.title('Top 20 Feature Importances from LightGBM Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



import numpy as np

def remove_outliers_iqr(df, column, threshold=1.5):
    """
    Removes outliers from a specified column of a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the column to remove outliers from.
        threshold (float): Threshold to determine outliers based on IQR.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers

# Remove outliers from the 'Price' column in the training data
train_data_no_outliers = remove_outliers_iqr(train_data_selected, 'Price', threshold=1.5)

print("Original Training Data Shape:", train_data_selected.shape)
print("Training Data Shape After Outlier Removal:", train_data_no_outliers.shape)



train_data_no_outliers = remove_outliers_iqr(train_data_selected, 'Price', threshold=1.0)

print("Original Training Data Shape:", train_data_selected.shape)
print("Training Data Shape After Outlier Removal:", train_data_no_outliers.shape)



# Apply outlier removal to the extra training data
train_extra_data_no_outliers = remove_outliers_iqr(train_extra_data_selected, 'Price', threshold=1.0)

print("Original Extra Training Data Shape:", train_extra_data_selected.shape)
print("Extra Training Data Shape After Outlier Removal:", train_extra_data_no_outliers.shape)



# Check column names
print("Train Data Columns:", train_data_no_outliers.columns.tolist())
print("Extra Train Data Columns:", train_extra_data_no_outliers.columns.tolist())

# Check data types
print("Train Data Data Types:\n", train_data_no_outliers.dtypes)
print("Extra Train Data Data Types:\n", train_extra_data_no_outliers.dtypes)



def fix_duplicate_weight_capacity_columns(df):
    """
    Renames duplicate 'Weight Capacity (kg)' columns in a DataFrame.
    """
    cols = df.columns.tolist()
    weight_capacity_count = 0
    new_cols = []
    for col in cols:
        if col == 'Weight Capacity (kg)' and weight_capacity_count > 0:
            new_col = f'Weight Capacity (kg)_{weight_capacity_count}'
            weight_capacity_count += 1
            new_cols.append(new_col)
        elif col == 'Weight Capacity (kg)':
            weight_capacity_count += 1
            new_cols.append(col)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

train_extra_data_no_outliers = fix_duplicate_weight_capacity_columns(train_extra_data_no_outliers)



# Get the column names from train_data_no_outliers
desired_columns = train_data_no_outliers.columns.tolist()

# Add missing columns to train_extra_data_no_outliers
for col in desired_columns:
    if col not in train_extra_data_no_outliers.columns:
        train_extra_data_no_outliers[col] = False  # Or 0, depending on the expected data type

# Reorder the columns in train_extra_data_no_outliers to match train_data_no_outliers
train_extra_data_no_outliers = train_extra_data_no_outliers[desired_columns]



# Reset the index for both DataFrames before concatenation
train_data_no_outliers = train_data_no_outliers.reset_index(drop=True)
train_extra_data_no_outliers = train_extra_data_no_outliers.reset_index(drop=True)

# Concatenate the original and extra training data
combined_train_data = pd.concat([train_data_no_outliers, train_extra_data_no_outliers], axis=0)

print("Combined Training Data Shape:", combined_train_data.shape)



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
X = combined_train_data.drop(['Price', 'id'], axis=1)
y = combined_train_data['Price']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# Train the model with early stopping
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round=100,
                      valid_sets=[lgb_train, lgb_val],  # Evaluate on training and validation sets
                      callbacks=[lgb.early_stopping(stopping_rounds=10)]) # Stop if no improvement after 10 rounds

# Make predictions on the validation set
y_pred_val = lgb_model.predict(X_val)

# Evaluate the model using RMSE
rmse_val = mean_squared_error(y_val, y_pred_val) ** 0.5
print("Root Mean Squared Error (RMSE) on the Validation Set:", rmse_val)

# Prepare the test data
X_test = test_data_selected.drop(['id'], axis=1)

# Make predictions on the test data
test_pred = lgb_model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'id': test_data_selected['id'], 'Price': test_pred})

# Save the submission file
#submission.to_csv('submission_lightgbm.csv', index=False)

print("Submission file 'submission_lightgbm.csv' created successfully!")



from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

# Define the parameter distribution
param_dist = {
    'objective': ['regression'],
    'metric': ['rmse'],
    'boosting_type': ['gbdt'],
    'num_leaves': [20, 31, 40, 50],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'bagging_freq': [1, 5],
    'verbose': [-1]
}

# Create the LightGBM regressor
lgbm = lgb.LGBMRegressor()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_dist,
                                   n_iter=20, cv=3, scoring='neg_root_mean_squared_error',
                                   verbose=1, n_jobs=-1, random_state=42)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", random_search.best_params_)

# Get the best model
best_lgbm = random_search.best_estimator_



# Evaluate the best model on the validation set
y_pred_val = best_lgbm.predict(X_val)
rmse_val = mean_squared_error(y_val, y_pred_val) ** 0.5
print("Root Mean Squared Error (RMSE) on the Validation Set with Best Model:", rmse_val)

# Prepare the test data
X_test = test_data_selected.drop(['id'], axis=1)

# Make predictions on the test data
test_pred = best_lgbm.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'id': test_data_selected['id'], 'Price': test_pred})

# Save the submission file
# submission.to_csv('submission_lightgbm_tuned.csv', index=False)

print("Submission file 'submission_lightgbm_tuned.csv' created successfully!")



from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
# Define categorical features
categorical_features = ['Waterproof', 'Brand_Jansport', 'Brand_Under Armour',
                        'Brand_Unknown', 'Material_Leather', 'Material_Nylon',
                        'Material_Polyester', 'Size_Medium', 'Size_Unknown',
                        'Style_Unknown', 'Color_Blue', 'Color_Gray',
                        'Color_Green', 'Color_Pink', 'Color_Red',
                        'Color_Unknown']

# Convert categorical features to string
for col in categorical_features:
    combined_train_data[col] = combined_train_data[col].astype(str)

# Prepare data for CatBoost
X_catboost = combined_train_data.drop(['Price', 'id'], axis=1)
y_catboost = combined_train_data['Price']

# Split the data into training and validation sets
X_train_catboost, X_val_catboost, y_train_catboost, y_val_catboost = train_test_split(X_catboost, y_catboost, test_size=0.2, random_state=42)

# Create the CatBoost regressor
catboost_model = CatBoostRegressor(iterations=1000, 
                                    learning_rate=0.1, 
                                    depth=6, 
                                    cat_features=categorical_features,
                                    verbose=100)

# Fit the model on the training data
catboost_model.fit(X_train_catboost, y_train_catboost)

# Make predictions on the validation set
y_pred_val_catboost = catboost_model.predict(X_val_catboost)

# Evaluate the model using RMSE
rmse_val_catboost = mean_squared_error(y_val_catboost, y_pred_val_catboost) ** 0.5
print("Root Mean Squared Error (RMSE) on the Validation Set with CatBoost:", rmse_val_catboost)

# Prepare the test data
X_test_catboost = test_data_selected.drop(['id'], axis=1)

# Convert test categorical features to string as well
for col in categorical_features:
    X_test_catboost[col] = X_test_catboost[col].astype(str)

# Make predictions on the test data
test_pred_catboost = catboost_model.predict(X_test_catboost)



# Create a submission file
submission_catboost = pd.DataFrame({'id': test_data_selected['id'], 'Price': test_pred_catboost})

# Save the submission file
submission_catboost.to_csv('submission_catboost.csv', index=False)

print("Submission file 'submission_catboost.csv' created successfully!")





