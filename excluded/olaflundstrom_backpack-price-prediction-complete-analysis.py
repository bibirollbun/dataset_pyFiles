import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set styling for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

print("Training set shape:", train.shape)
print("Test set shape:", test.shape)


# Display basic information about the training data
print("\nTraining Data Info:")
print(train.info())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train.isnull().sum())

# Display summary statistics
print("\nNumerical Features Summary:")
print(train.describe())


plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='Price', bins=50)
plt.title('Distribution of Backpack Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

print("\nPrice Statistics:")
print(train['Price'].describe())


# Analyze categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(train[col].value_counts().head())
    
    plt.figure(figsize=(10, 5))
    train[col].value_counts().head(10).plot(kind='bar')
    plt.title(f'Top 10 Most Common Values in {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Calculate correlations for numerical features
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = train[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()


def preprocess_data(train_df, test_df):
    """
    Preprocess both training and test data consistently.
    Returns processed features and target variable.
    """
    # Create copies to avoid modifying original data
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Separate target variable
    y_train = train_processed['Price']
    
    # Remove id and target columns
    X_train = train_processed.drop(['id', 'Price'], axis=1)
    X_test = test_processed.drop(['id'], axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Handle numerical features
    num_imputer = SimpleImputer(strategy='median')
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
    
    # Handle categorical features
    label_encoders = {}
    for col in categorical_cols:
        # Fill missing values with 'missing'
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')
        
        # Encode categorical variables
        label_encoders[col] = LabelEncoder()
        X_train[col] = label_encoders[col].fit_transform(X_train[col])
        X_test[col] = label_encoders[col].transform(X_test[col])
    
    return X_train, X_test, y_train

# Preprocess the data
X_train, X_test, y_train = preprocess_data(train, test)


# Initialize model with optimized parameters
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("\nCross-validation RMSE scores:", -cv_scores)
print("Average RMSE:", -cv_scores.mean())


# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


# Generate predictions for test set
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Price': predictions
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")

# Display sample predictions
print("\nSample predictions:")
print(submission.head())

