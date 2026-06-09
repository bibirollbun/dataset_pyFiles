import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Display first few rows of training data
print("First few rows of training data:")
print(train.head())

# Basic information about the datasets
print("\nTraining data shape:", train.shape)
print("Extra training data shape:", train_extra.shape)
print("Test data shape:", test.shape)

# Check for missing values
print("\nMissing values in training data:")
print(train.isnull().sum())

# Display data types and basic statistics
print("\nData types:")
print(train.dtypes)

print("\nSummary statistics:")
print(train.describe())

# Display unique values in categorical columns
print("\nUnique values in categorical columns:")
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} unique values:")
    print(train[col].value_counts())


# Function to display missing value percentages
def missing_values_analysis(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_table = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    }).sort_values(by='Missing Values', ascending=False)
    return missing_table[missing_table['Missing Values'] > 0]

# Display missing values analysis
print("Missing Values Analysis:")
print(missing_values_analysis(train))

# Handle missing values
def preprocess_data(df):
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Fill missing numerical values with median
    numerical_cols = df.select_dtypes(include=['float64']).columns
    for col in numerical_cols:
        df_processed[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col].fillna(df[col].mode()[0], inplace=True)
    
    return df_processed

# Process the data
train_processed = preprocess_data(train)

# Verify no missing values remain
print("\nMissing values after processing:")
print(train_processed.isnull().sum())


plt.figure(figsize=(15, 6))

# Price Distribution
plt.subplot(1, 2, 1)
sns.histplot(data=train_processed, x='Price', bins=50, kde=True)
plt.title('Distribution of Backpack Prices')
plt.axvline(train_processed['Price'].mean(), color='red', linestyle='--', label=f"Mean: {train_processed['Price'].mean():.2f}")
plt.axvline(train_processed['Price'].median(), color='green', linestyle='--', label=f"Median: {train_processed['Price'].median():.2f}")
plt.legend()

# Price Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(y=train_processed['Price'])
plt.title('Price Distribution Box Plot')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 6))

# Price by Brand and Size
plt.subplot(1, 1, 1)
sns.boxplot(data=train_processed, x='Brand', y='Price', hue='Size')
plt.title('Price Distribution by Brand and Size')
plt.xticks(rotation=45)
plt.legend(title='Size', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Average price by Brand
brand_price = train_processed.groupby('Brand')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
brand_price.plot(kind='bar')
plt.title('Average Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))

# Price by Material and Waterproof
plt.subplot(2, 1, 1)
sns.boxplot(data=train_processed, x='Material', y='Price', hue='Waterproof')
plt.title('Price Distribution by Material and Waterproof')
plt.xticks(rotation=45)

# Price by Style and Laptop Compartment
plt.subplot(2, 1, 2)
sns.boxplot(data=train_processed, x='Style', y='Price', hue='Laptop Compartment')
plt.title('Price Distribution by Style and Laptop Compartment')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Correlation Analysis
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']
plt.figure(figsize=(10, 8))
sns.heatmap(train_processed[numerical_cols].corr(), 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Scatter plots with regression lines
plt.figure(figsize=(15, 5))

# Weight Capacity vs Price
plt.subplot(1, 2, 1)
sns.regplot(data=train_processed, 
            x='Weight Capacity (kg)', 
            y='Price', 
            scatter_kws={'alpha':0.5}, 
            line_kws={'color': 'red'})
plt.title('Weight Capacity vs Price')

# Compartments vs Price
plt.subplot(1, 2, 2)
sns.regplot(data=train_processed, 
            x='Compartments', 
            y='Price', 
            scatter_kws={'alpha':0.5}, 
            line_kws={'color': 'red'})
plt.title('Compartments vs Price')

plt.tight_layout()
plt.show()


# Feature Engineering
def create_features(df):
    # Create a copy of the dataframe
    df_features = df.copy()
    
    # Create price ranges
    df_features['price_range'] = pd.qcut(df_features['Price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Create interaction features
    df_features['capacity_per_compartment'] = df_features['Weight Capacity (kg)'] / df_features['Compartments']
    
    # Create binary features
    df_features['is_premium'] = df_features['Price'] > df_features['Price'].median()
    
    # Feature encoding
    categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Waterproof', 'Laptop Compartment']
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    
    return df_features

# Create features for training data
train_features = create_features(train_processed)

# Print new features info
print("Features created. New shape:", train_features.shape)
print("\nNew features preview:")
print(train_features.head())


# Import necessary modules for modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np

# Prepare features and target
X = train_features.drop(['Price', 'price_range', 'id', 'is_premium'], axis=1)
y = train_features['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to test
models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    results[name] = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R2': train_r2,
        'Test R2': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    }

# Display results
results_df = pd.DataFrame(results).round(4)
print("\nModel Performance Comparison:")
print(results_df)

# Plot feature importance for the best model
best_model = models['XGBoost']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Most Important Features')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))

# Training set predictions
plt.subplot(1, 2, 1)
plt.scatter(y_train, best_model.predict(X_train_scaled), alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Set: Actual vs Predicted')

# Test set predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, best_model.predict(X_test_scaled), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Test Set: Actual vs Predicted')

plt.tight_layout()
plt.show()


def create_features(df, is_training=False):
    # Create a copy of the dataframe
    df_features = df.copy()
    
    # Create price ranges only for training data
    if is_training and 'Price' in df_features.columns:
        df_features['price_range'] = pd.qcut(df_features['Price'], q=5, 
                                           labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Create interaction features
    df_features['capacity_per_compartment'] = df_features['Weight Capacity (kg)'] / df_features['Compartments']
    
    # Feature encoding
    categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Waterproof', 'Laptop Compartment']
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    
    return df_features

# Load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_processed = preprocess_data(test)
test_features = create_features(test_processed, is_training=False)

# Align test features with training features
test_features_aligned = test_features[X.columns]
X_test_scaled = scaler.transform(test_features_aligned)

# Make predictions using the trained model
test_predictions = model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Price': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print("\nFirst few rows of predictions:")
print(submission.head())


from google.colab import files  # if using Google Colab
# or
from IPython.display import FileLink  # if using Jupyter notebook

# First create the submission file
submission.to_csv('submission.csv', index=False)

# If using Google Colab:
files.download('submission.csv')

# If using Jupyter notebook:
FileLink('submission.csv')

