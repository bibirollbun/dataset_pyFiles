# Import necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Load the competition dataset
# This cell is designed to work both locally and on Kaggle

import os

# Check if we're running on Kaggle
IN_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if IN_KAGGLE:
    # On Kaggle, data is available in the input directory
    COMPETITION_NAME = 'playground-series-s5e2'
    DATA_PATH = f'/kaggle/input/{COMPETITION_NAME}/'
else:
    # For local execution, use the current directory
    DATA_PATH = './'

# Load the training and testing data
train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')
sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# Print dataset shapes to verify loading
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Examine the structure of the datasets

# Display the first few rows of the training data
print("First 5 rows of the training data:")
display(train_df.head())

# Check for missing values
print("\nMissing values in training data:")
display(train_df.isnull().sum())

print("\nMissing values in test data:")
display(test_df.isnull().sum())

# Display basic statistics for the training data
print("\nBasic statistics for numerical columns in training data:")
display(train_df.describe())

# Display information about the data types
print("\nData types in training data:")
display(train_df.dtypes)


# Create a function to plot the distribution of categorical variables
def plot_categorical_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create a function to plot the distribution of numerical variables
def plot_numerical_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # Add boxplot to show outliers
    plt.figure(figsize=(10, 3))
    sns.boxplot(data=df, x=column)
    plt.title(f"Boxplot of {title}")
    plt.tight_layout()
    plt.show()

# Plot distribution of categorical variables
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    plot_categorical_distribution(train_df, col, f"Distribution of {col}")

# Plot distribution of numerical variables
for col in ['Compartments', 'Weight Capacity (kg)', 'Price']:
    plot_numerical_distribution(train_df, col, f"Distribution of {col}")


# Analyze relationships between categorical variables and price
# Create box plots to visualize how price varies across different categories

plt.figure(figsize=(15, 10))

# Create a 2x4 grid of subplots for categorical variables
plt.subplot(2, 4, 1)
sns.boxplot(x='Brand', y='Price', data=train_df)
plt.title('Price by Brand')
plt.xticks(rotation=45)

plt.subplot(2, 4, 2)
sns.boxplot(x='Material', y='Price', data=train_df)
plt.title('Price by Material')
plt.xticks(rotation=45)

plt.subplot(2, 4, 3)
sns.boxplot(x='Size', y='Price', data=train_df)
plt.title('Price by Size')

plt.subplot(2, 4, 4)
sns.boxplot(x='Laptop Compartment', y='Price', data=train_df)
plt.title('Price by Laptop Compartment')

plt.subplot(2, 4, 5)
sns.boxplot(x='Waterproof', y='Price', data=train_df)
plt.title('Price by Waterproof')

plt.subplot(2, 4, 6)
sns.boxplot(x='Style', y='Price', data=train_df)
plt.title('Price by Style')
plt.xticks(rotation=45)

plt.subplot(2, 4, 7)
sns.boxplot(x='Color', y='Price', data=train_df)
plt.title('Price by Color')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Analyze relationships between numerical variables and price
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Compartments', y='Price', data=train_df, alpha=0.5)
plt.title('Price vs Compartments')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Weight Capacity (kg)', y='Price', data=train_df, alpha=0.5)
plt.title('Price vs Weight Capacity')

plt.tight_layout()
plt.show()

# Calculate correlation between numerical variables
correlation_matrix = train_df[['Compartments', 'Weight Capacity (kg)', 'Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# Handle missing values in the dataset
print("Missing values in train dataset:")
print(train_df.isnull().sum())

# Fill missing values
# For categorical variables, fill with the most frequent value
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    if train_df[col].isnull().sum() > 0:
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
        
# For numerical variables, fill with the median
for col in ['Compartments', 'Weight Capacity (kg)']:
    if train_df[col].isnull().sum() > 0:
        train_df[col] = train_df[col].fillna(train_df[col].median())

# Check if all missing values are handled
print("\nMissing values after imputation:")
print(train_df.isnull().sum())


# Handle missing values in the test dataset
print("Missing values in test dataset:")
print(test_df.isnull().sum())

# Fill missing values in test dataset using the same strategy as for train dataset
# For categorical variables, fill with the most frequent value
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    if test_df[col].isnull().sum() > 0:
        # Use the mode from the training data for consistency
        test_df[col] = test_df[col].fillna(train_df[col].mode()[0])
        
# For numerical variables, fill with the median
for col in ['Compartments', 'Weight Capacity (kg)']:
    if test_df[col].isnull().sum() > 0:
        # Use the median from the training data for consistency
        test_df[col] = test_df[col].fillna(train_df[col].median())

# Check if all missing values are handled
print("\nMissing values in test dataset after imputation:")
print(test_df.isnull().sum())

# Save the preprocessed datasets for later use
train_df_processed = train_df.copy()
test_df_processed = test_df.copy()


# Feature Engineering

# Create a copy of the processed datasets to avoid modifying the originals
train_features = train_df_processed.copy()
test_features = test_df_processed.copy()

# Encode categorical variables
# We'll use Label Encoding for simplicity, but One-Hot Encoding could also be used
label_encoders = {}
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for col in categorical_cols:
    le = LabelEncoder()
    train_features[col] = le.fit_transform(train_features[col])
    test_features[col] = le.transform(test_features[col])
    label_encoders[col] = le
    
    # Print mapping for reference
    print(f"Encoding for {col}:")
    for i, category in enumerate(le.classes_):
        print(f"  {category} -> {i}")

# Create interaction features
# 1. Combine Size and Weight Capacity
train_features['Size_Weight'] = train_features['Size'] * train_features['Weight Capacity (kg)']
test_features['Size_Weight'] = test_features['Size'] * test_features['Weight Capacity (kg)']

# 2. Combine Material and Waterproof
train_features['Material_Waterproof'] = train_features['Material'] * train_features['Waterproof']
test_features['Material_Waterproof'] = test_features['Material'] * test_features['Waterproof']

# 3. Combine Brand and Style
train_features['Brand_Style'] = train_features['Brand'] * train_features['Style']
test_features['Brand_Style'] = test_features['Brand'] * test_features['Style']

# 4. Combine Compartments and Laptop Compartment
train_features['Compartments_Laptop'] = train_features['Compartments'] * train_features['Laptop Compartment']
test_features['Compartments_Laptop'] = test_features['Compartments'] * test_features['Laptop Compartment']

# Display the first few rows of the engineered features
print("\nFirst 5 rows of the engineered training features:")
display(train_features.head())

# Check the shape of the engineered features
print(f"\nTraining features shape: {train_features.shape}")
print(f"Testing features shape: {test_features.shape}")


# Model Building: Baseline Models

# Prepare the data for modeling
X = train_features.drop(['id', 'Price'], axis=1)
y = train_features['Price']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Train a baseline Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the validation set
lr_val_pred = lr_model.predict(X_val)

# Evaluate the model
lr_val_rmse = np.sqrt(mean_squared_error(y_val, lr_val_pred))
lr_val_r2 = r2_score(y_val, lr_val_pred)

print("\nLinear Regression Model Performance:")
print(f"RMSE: {lr_val_rmse:.4f}")
print(f"R²: {lr_val_r2:.4f}")

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
rf_val_pred = rf_model.predict(X_val)

# Evaluate the model
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
rf_val_r2 = r2_score(y_val, rf_val_pred)

print("\nRandom Forest Model Performance:")
print(f"RMSE: {rf_val_rmse:.4f}")
print(f"R²: {rf_val_r2:.4f}")

# Train a Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on the validation set
gb_val_pred = gb_model.predict(X_val)

# Evaluate the model
gb_val_rmse = np.sqrt(mean_squared_error(y_val, gb_val_pred))
gb_val_r2 = r2_score(y_val, gb_val_pred)

print("\nGradient Boosting Model Performance:")
print(f"RMSE: {gb_val_rmse:.4f}")
print(f"R²: {gb_val_r2:.4f}")


# Make predictions on the test set using the Gradient Boosting model
# (which had the best performance among our models)

# Prepare the test data
X_test = test_features.drop(['id'], axis=1)

# Make predictions
test_predictions = gb_model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': test_predictions
})

# Display the first few rows of the submission file
print("First 5 rows of the submission file:")
display(submission.head())

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")

# Display basic statistics of the predictions
print("\nPrediction statistics:")
print(f"Min price: ${submission['Price'].min():.2f}")
print(f"Max price: ${submission['Price'].max():.2f}")
print(f"Mean price: ${submission['Price'].mean():.2f}")
print(f"Median price: ${submission['Price'].median():.2f}")


# Explore feature interactions
# Let's see if combinations of features have a stronger relationship with price

# 1. Interaction between Brand and Material
plt.figure(figsize=(15, 8))
sns.boxplot(x='Brand', y='Price', hue='Material', data=train_df)
plt.title('Price by Brand and Material')
plt.xticks(rotation=45)
plt.legend(title='Material', loc='upper right')
plt.tight_layout()
plt.show()

# 2. Interaction between Style and Size
plt.figure(figsize=(12, 6))
sns.boxplot(x='Style', y='Price', hue='Size', data=train_df)
plt.title('Price by Style and Size')
plt.legend(title='Size', loc='upper right')
plt.tight_layout()
plt.show()

# 3. Interaction between Laptop Compartment and Waterproof
plt.figure(figsize=(10, 6))
sns.boxplot(x='Laptop Compartment', y='Price', hue='Waterproof', data=train_df)
plt.title('Price by Laptop Compartment and Waterproof')
plt.legend(title='Waterproof', loc='upper right')
plt.tight_layout()
plt.show()


# Prepare data for modeling
# First, let's handle missing values in both train and test datasets

# Create a copy of the dataframes to avoid modifying the originals
train_model_df = train_df.copy()
test_model_df = test_df.copy()

# Check for missing values in test dataset
print("Missing values in test dataset:")
print(test_model_df.isnull().sum())

# Fill missing values in test dataset
# For categorical variables, fill with the most frequent value from training data
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    if col in test_model_df.columns and test_model_df[col].isnull().sum() > 0:
        test_model_df[col] = test_model_df[col].fillna(train_df[col].mode()[0])
        
# For numerical variables, fill with the median from training data
for col in ['Compartments', 'Weight Capacity (kg)']:
    if col in test_model_df.columns and test_model_df[col].isnull().sum() > 0:
        test_model_df[col] = test_model_df[col].fillna(train_df[col].median())

# Now encode categorical variables using Label Encoding
label_encoders = {}
for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    le = LabelEncoder()
    train_model_df[col] = le.fit_transform(train_model_df[col])
    
    # Apply the same transformation to the test set
    if col in test_model_df.columns:
        test_model_df[col] = le.transform(test_model_df[col])
    
    label_encoders[col] = le

# Feature Engineering: Create interaction features
train_model_df['Brand_Material'] = train_model_df['Brand'] * train_model_df['Material']
train_model_df['Style_Size'] = train_model_df['Style'] * train_model_df['Size']
train_model_df['Laptop_Waterproof'] = train_model_df['Laptop Compartment'] * train_model_df['Waterproof']

# Apply the same feature engineering to the test set
test_model_df['Brand_Material'] = test_model_df['Brand'] * test_model_df['Material']
test_model_df['Style_Size'] = test_model_df['Style'] * test_model_df['Size']
test_model_df['Laptop_Waterproof'] = test_model_df['Laptop Compartment'] * test_model_df['Waterproof']

# Display the first few rows of the transformed data
print("\nTransformed training data:")
print(train_model_df.head())

# Split the data into features and target
X = train_model_df.drop(['id', 'Price'], axis=1)
y = train_model_df['Price']

# Display the feature names
print("\nFeatures for modeling:")
print(X.columns.tolist())


# Train and evaluate machine learning models with a simpler approach
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to evaluate with fewer estimators for faster training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    
    # Store results
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    
    # If it's a tree-based model, get feature importances
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances for {name}:")
        print(feature_importances.head(10))

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

# Identify the best model
best_model_name = results_df['R²'].idxmax()
print(f"\nBest Model: {best_model_name} with R² = {results_df.loc[best_model_name, 'R²']:.4f}")


# Make predictions on the test set using the best model (Gradient Boosting)
# First, ensure the test data has the same features as the training data
test_features = test_model_df.drop('id', axis=1)

# Get the best model
best_model = models['Gradient Boosting']

# Make predictions
test_predictions = best_model.predict(test_features)

# Create a submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': test_predictions
})

# Display the first few rows of the submission file
print("Submission Preview:")
print(submission.head())

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")

# Create a histogram of the predicted prices
plt.figure(figsize=(10, 6))
plt.hist(test_predictions, bins=30, alpha=0.7)
plt.title('Distribution of Predicted Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()

# Compare the distribution of predicted prices with the training prices
plt.figure(figsize=(12, 6))
plt.hist(train_df['Price'], bins=30, alpha=0.7, label='Training Prices')
plt.hist(test_predictions, bins=30, alpha=0.7, label='Predicted Prices')
plt.title('Comparison of Training and Predicted Price Distributions')
plt.xlabel('Price')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

