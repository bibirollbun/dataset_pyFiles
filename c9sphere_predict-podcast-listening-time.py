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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# Load the datasets
print("Loading datasets...")
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Display the first few rows of the training dataset
train.head()


# Display the first few rows of the test dataset
test.head()


# Display the first few rows of the sample submission
sample_submission.head()


# Check the data types and missing values in the training dataset
print("Train dataset info:")
train.info()


# Check the data types and missing values in the test dataset
print("Test dataset info:")
test.info()


# Calculate the percentage of missing values in each column
print("Missing values in train dataset (percentage):")
train_missing = (train.isnull().sum() / len(train)) * 100
print(train_missing[train_missing > 0])

print("\nMissing values in test dataset (percentage):")
test_missing = (test.isnull().sum() / len(test)) * 100
print(test_missing[test_missing > 0])


# Get summary statistics for numeric columns in the training dataset
train.describe()


# Explore categorical features
categorical_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in categorical_features:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(f"Top 5 most common values:")
    print(train[col].value_counts().head())


# Plot the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50)
plt.title('Distribution of Listening Time')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Count')
plt.show()

# Print summary statistics for the target variable
print("Summary statistics for Listening_Time_minutes:")
print(train['Listening_Time_minutes'].describe())


# Calculate correlations between numeric features and the target variable
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
corr = train[numeric_cols].corr()['Listening_Time_minutes'].sort_values(ascending=False)
print("Correlation with target (numeric features only):")
print(corr)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Features)')
plt.tight_layout()
plt.show()


# Calculate average listening time by categorical features
print("Average Listening Time by Genre:")
print(train.groupby('Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False))

print("\nAverage Listening Time by Publication Day:")
print(train.groupby('Publication_Day')['Listening_Time_minutes'].mean().sort_values(ascending=False))

print("\nAverage Listening Time by Publication Time:")
print(train.groupby('Publication_Time')['Listening_Time_minutes'].mean().sort_values(ascending=False))

print("\nAverage Listening Time by Episode Sentiment:")
print(train.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean().sort_values(ascending=False))


# Plot listening time by genre
plt.figure(figsize=(14, 6))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train.sample(10000))
plt.title('Listening Time by Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Plot listening time by publication day
plt.figure(figsize=(14, 6))
sns.boxplot(x='Publication_Day', y='Listening_Time_minutes', data=train.sample(10000))
plt.title('Listening Time by Publication Day')
plt.tight_layout()
plt.show()


# Plot listening time by episode sentiment
plt.figure(figsize=(14, 6))
sns.boxplot(x='Episode_Sentiment', y='Listening_Time_minutes', data=train.sample(10000))
plt.title('Listening Time by Episode Sentiment')
plt.tight_layout()
plt.show()


# Plot scatter plots for numeric features vs target
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train.sample(5000))
plt.title('Listening Time vs Episode Length')

plt.subplot(2, 2, 2)
sns.scatterplot(x='Host_Popularity_percentage', y='Listening_Time_minutes', data=train.sample(5000))
plt.title('Listening Time vs Host Popularity')

plt.subplot(2, 2, 3)
sns.scatterplot(x='Guest_Popularity_percentage', y='Listening_Time_minutes', data=train.sample(5000))
plt.title('Listening Time vs Guest Popularity')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Number_of_Ads', y='Listening_Time_minutes', data=train.sample(5000))
plt.title('Listening Time vs Number of Ads')

plt.tight_layout()
plt.show()


# Prepare data for modeling
print("Preparing data...")

# Take a sample of the training data to speed up processing
train_sample = train.sample(frac=0.1, random_state=42)
print(f"Using {len(train_sample)} samples out of {len(train)} total records")

# Based on our analysis, Episode_Title has too many unique values and might cause memory issues
# We'll drop it and id as they're not useful predictors
X = train_sample.drop(['Listening_Time_minutes', 'Episode_Title', 'id'], axis=1)
y = train_sample['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")


# Create preprocessing pipelines
print("Creating preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Train a Linear Regression model
print("Training Linear Regression model...")
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

lr_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_lr = lr_model.predict(X_val)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr = r2_score(y_val, y_pred_lr)

print(f"Linear Regression - RMSE: {rmse_lr:.4f}, R2: {r2_lr:.4f}")


# Train a Random Forest model
print("Training Random Forest model...")
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")


# Train a Gradient Boosting model
print("Training Gradient Boosting model...")
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

gb_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_gb = gb_model.predict(X_val)
rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
r2_gb = r2_score(y_val, y_pred_gb)

print(f"Gradient Boosting - RMSE: {rmse_gb:.4f}, R2: {r2_gb:.4f}")


# Compare model performance
models = {
    'Linear Regression': {'RMSE': rmse_lr, 'R2': r2_lr},
    'Random Forest': {'RMSE': rmse_rf, 'R2': r2_rf},
    'Gradient Boosting': {'RMSE': rmse_gb, 'R2': r2_gb}
}

# Create a DataFrame for comparison
model_comparison = pd.DataFrame({
    'RMSE': [models[model]['RMSE'] for model in models],
    'R2': [models[model]['R2'] for model in models]
}, index=models.keys())

print("Model Comparison:")
print(model_comparison)

# Plot model comparison
plt.figure(figsize=(10, 6))
model_comparison['RMSE'].plot(kind='bar')
plt.title('Model Comparison - RMSE (lower is better)')
plt.ylabel('RMSE')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Find the best model
best_model_name = model_comparison['RMSE'].idxmin()
best_rmse = model_comparison.loc[best_model_name, 'RMSE']
best_r2 = model_comparison.loc[best_model_name, 'R2']

print(f"Best model: {best_model_name}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best R2: {best_r2:.4f}")

# Select the best model
if best_model_name == 'Linear Regression':
    best_model = lr_model
    y_pred = y_pred_lr
elif best_model_name == 'Random Forest':
    best_model = rf_model
    y_pred = y_pred_rf
else:  # Gradient Boosting
    best_model = gb_model
    y_pred = y_pred_gb


# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.3)
plt.plot([0, 120], [0, 120], 'r--')
plt.xlabel('Actual Listening Time')
plt.ylabel('Predicted Listening Time')
plt.title('Actual vs Predicted Listening Time')
plt.show()


# Plot residuals
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Listening Time')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Residual Distribution')
plt.show()


# Extract feature importance if the best model is tree-based
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("Extracting feature importances...")
    
    # Get feature names after preprocessing
    cat_features = best_model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_features)
    
    # Get feature importances
    importances = best_model.named_steps['model'].feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 important features:")
    print(feature_importance.head(10))


# Prepare test data the same way as training data
print("Preparing test data...")
test_data = test.drop(['Episode_Title', 'id'], axis=1)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = best_model.predict(test_data)

# Create submission file
print("Creating submission file...")
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': test_predictions
})

# Display the first few rows of the submission file
submission.head()

