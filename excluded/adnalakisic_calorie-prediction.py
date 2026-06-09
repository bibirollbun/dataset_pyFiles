# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# Data Loading

print("Step 1: Loading Data...")
# Replace with your actual file paths
train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Custom RMSLE evaluation function
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error
    """
    # Ensure values are positive (add 1 to handle zeros)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Make sure predictions are not negative
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate RMSLE
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

# Function to evaluate models
def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation and RMSLE
    """
    # Define custom RMSLE scorer
    def rmsle_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return -rmsle(y, y_pred)  # Negative because sklearn maximizes scores
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=rmsle_scorer)
    return -np.mean(scores)  # Convert back to positive RMSLE


# EDA

print("\nStep 2: Exploratory Data Analysis...")

# Display basic information
print("\nTraining Data Shape:", train_data.shape)
print("\nTraining Data Sample:")
print(train_data.head())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(train_data.describe())

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Calories'], kde=True)
plt.title('Distribution of Calories')
plt.savefig('calories_distribution.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = train_data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Relationships between numerical features and target
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(18, 12))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=feature, y='Calories', data=train_data)
    plt.title(f'{feature} vs calories')
plt.tight_layout()
plt.savefig('features_vs_calories.png')
plt.close()

# Calories by sex
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Calories', data=train_data)
plt.title('Calories Distribution by Sex')
plt.savefig('calories_by_sex.png')
plt.close()


# Feature Engineering
print("\nStep 3: Feature Engineering...")

# Function to create features for both train and test sets
def create_features(df):
    # Make a copy of the dataframe to avoid changing the original
    df_new = df.copy()
    
    # Convert sex to numeric
    df_new['Sex_numeric'] = df_new['Sex'].map({'male': 0, 'female': 1})
    
    # Calculate BMI
    df_new['BMI'] = df_new['Weight'] / ((df_new['Height']/100) ** 2)
    
    # Create intensity features
    df_new['Intensity'] = df_new['Heart_Rate'] * df_new['Duration']
    
    # Heart rate zones (rough approximation)
    df_new['HR_Zone'] = pd.cut(df_new['Heart_Rate'], 
                              bins=[0, 100, 120, 140, 160, 200],
                              labels=[1, 2, 3, 4, 5])
    
    # Temperature effect
    df_new['Temp_Factor'] = df_new['Body_Temp'] / 37.0  # Normalized to normal body temp
    
    # Weight-duration interaction
    df_new['Weight_Duration'] = df_new['Weight'] * df_new['Duration']
    
    # Age groups
    df_new['Age_Group'] = pd.cut(df_new['Age'], 
                               bins=[0, 20, 30, 40, 50, 60, 100],
                               labels=[1, 2, 3, 4, 5, 6])
    
    return df_new

# Apply feature engineering
train_data_fe = create_features(train_data)
test_data_fe = create_features(test_data)

print("\nNew Features Created. Training Data Shape:", train_data_fe.shape)
print("Sample with New Features:")
print(train_data_fe.head())



# Model Preparation, Training and Evaluation
print("\nStep 4: Model Preparation...")

# Separate features and target
X = train_data_fe.drop(['id', 'Calories', 'Sex'], axis=1)  # Remove 'Sex' since we have Sex_numeric
y = train_data_fe['Calories']
test_ids = test_data_fe['id']
X_test = test_data_fe.drop(['id', 'Sex'], axis=1)  # Drop 'Sex' here too

# Ensure all categorical columns are encoded numerically
categorical_cols = X.select_dtypes(include='category').columns.tolist()

if categorical_cols:
    print(f"Encoding categorical columns: {categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align train/test to have same columns
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features Shape:", X_train.shape)
print("Validation Features Shape:", X_val.shape)


print("\nStep 5: Model Training and Evaluation...")

# Models to try
models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Store results
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate RMSLE
    train_rmsle = rmsle(y_train, train_preds)
    val_rmsle = rmsle(y_val, val_preds)
    
    print(f"{name} - Training RMSLE: {train_rmsle:.4f}, Validation RMSLE: {val_rmsle:.4f}")
    
    # Also perform cross-validation
    cv_rmsle = evaluate_model(model, X, y, cv=5)
    print(f"{name} - Cross-Validation RMSLE: {cv_rmsle:.4f}")
    
    results[name] = {
        'model': model,
        'train_rmsle': train_rmsle,
        'val_rmsle': val_rmsle,
        'cv_rmsle': cv_rmsle
    }

# Find the best model based on validation RMSLE
best_model_name = min(results, key=lambda x: results[x]['val_rmsle'])
print(f"\nBest Model: {best_model_name} with Validation RMSLE: {results[best_model_name]['val_rmsle']:.4f}")



# Final Training and Submission
print("\nStep 6: Training Final Model on Full Data and Creating Submission...")

# Retrain the best model on the full training data
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Predict on test set
final_predictions = final_model.predict(X_test)

# Clip predictions to ensure no negative calorie values
final_predictions = np.clip(final_predictions, 0, None)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': final_predictions
})

# Save to CSV
submission.to_csv("submission.csv", index=False)


