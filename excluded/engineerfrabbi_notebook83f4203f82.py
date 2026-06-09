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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import joblib
import os
import warnings
import gc  # Garbage collector for memory management
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('submission', exist_ok=True)
os.makedirs('advanced_models', exist_ok=True)


# Function to calculate RMSLE (Root Mean Squared Logarithmic Error)
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error
    Note: Handles negative predictions by clipping them to a small positive value
    """
    # Ensure predictions are positive (required for log)
    y_pred = np.maximum(y_pred, 1e-5)
    y_true = np.maximum(y_true, 1e-5)
    
    # Calculate RMSLE
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


# Load the data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Submission shape: {submission.shape}")


# Display basic information
print("\nTrain data info:")
train.info()


# Check for missing values
print("\nChecking for missing values:")
print(train.isnull().sum())


# Basic statistics
print("\nBasic statistics:")
train.describe().T


# Visualize target variable distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], kde=True)
plt.title('Distribution of Calories', fontsize=14)
plt.xlabel('Calories', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('plots/calories_distribution.png')
plt.show()


# Log transformation of target variable
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['Calories']), kde=True, color='orange')
plt.title('Distribution of Log(Calories+1)', fontsize=14)
plt.xlabel('Log(Calories+1)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('plots/log_calories_distribution.png')
plt.show()


# Correlation analysis
print("\nCalculating feature correlations...")
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
correlation = train[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=14)
plt.savefig('plots/correlation_matrix.png')
plt.show()


# Feature correlation with target
target_corr = correlation['Calories'].sort_values(ascending=False)
print("\nFeature correlation with target (Calories):")
print(target_corr)


# Calories by Sex
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories by Sex', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Calories', fontsize=12)
plt.savefig('plots/calories_by_sex.png')
plt.show()


# Key feature relationships with target
key_features = ['Duration', 'Heart_Rate', 'Body_Temp']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(key_features):
    sns.regplot(x=feature, y='Calories', data=train.sample(10000), ax=axes[i], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[i].set_title(f'{feature} vs Calories', fontsize=12)
    
plt.tight_layout()
plt.savefig('plots/key_features_vs_calories.png')
plt.show()


# Create a function for advanced feature engineering to apply to both train and test
def advanced_feature_engineering(df):
    """
    Apply advanced feature engineering to the dataset
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Basic features from original engineering
    df_new['BMI'] = df_new['Weight'] / ((df_new['Height']/100) ** 2)
    df_new['Sex_male'] = (df_new['Sex'] == 'male').astype(int)
    
    # Age groups with more granularity
    df_new['Age_Group'] = pd.cut(df_new['Age'], 
                              bins=[19, 25, 30, 35, 40, 45, 50, 55, 60, 80], 
                              labels=list(range(9)))
    df_new['Age_Group'] = df_new['Age_Group'].astype(int)
    
    # BMI categories (underweight, normal, overweight, obese)
    df_new['BMI_Category'] = pd.cut(df_new['BMI'], 
                                 bins=[0, 18.5, 25, 30, 100], 
                                 labels=[0, 1, 2, 3])
    df_new['BMI_Category'] = df_new['BMI_Category'].astype(int)
    
    # Advanced interaction features
    df_new['Duration_Heart'] = df_new['Duration'] * df_new['Heart_Rate']
    df_new['Weight_Heart'] = df_new['Weight'] * df_new['Heart_Rate']
    df_new['BMI_Duration'] = df_new['BMI'] * df_new['Duration']
    df_new['Heart_Temp'] = df_new['Heart_Rate'] * df_new['Body_Temp']
    df_new['Duration_Temp'] = df_new['Duration'] * df_new['Body_Temp']
    df_new['Age_Heart'] = df_new['Age'] * df_new['Heart_Rate']
    df_new['Weight_Duration'] = df_new['Weight'] * df_new['Duration']
    
    # Polynomial features for key variables
    df_new['Duration_Squared'] = df_new['Duration'] ** 2
    df_new['Heart_Rate_Squared'] = df_new['Heart_Rate'] ** 2
    df_new['Body_Temp_Squared'] = df_new['Body_Temp'] ** 2
    
    # Ratios and normalized features
    df_new['Heart_Rate_by_Age'] = df_new['Heart_Rate'] / df_new['Age']
    df_new['Duration_by_Weight'] = df_new['Duration'] / df_new['Weight']
    df_new['Heart_Rate_by_Weight'] = df_new['Heart_Rate'] / df_new['Weight']
    
    # Logarithmic transformations for skewed features
    df_new['Log_Duration'] = np.log1p(df_new['Duration'])
    df_new['Log_Heart_Rate'] = np.log1p(df_new['Heart_Rate'])
    df_new['Log_Weight'] = np.log1p(df_new['Weight'])
    
    # Trigonometric transformations to capture cyclical patterns
    df_new['Sin_Heart_Rate'] = np.sin(df_new['Heart_Rate'] / 200 * np.pi)
    df_new['Cos_Heart_Rate'] = np.cos(df_new['Heart_Rate'] / 200 * np.pi)
    
    # Interaction between categorical and continuous features
    df_new['Sex_Duration'] = df_new['Sex_male'] * df_new['Duration']
    df_new['Sex_Heart'] = df_new['Sex_male'] * df_new['Heart_Rate']
    df_new['Sex_BMI'] = df_new['Sex_male'] * df_new['BMI']
    df_new['Age_Group_Heart'] = df_new['Age_Group'] * df_new['Heart_Rate']
    
    # Statistical aggregations
    df_new['Feature_Sum'] = df_new['Duration'] + df_new['Heart_Rate'] + df_new['Body_Temp']
    df_new['Feature_Mean'] = (df_new['Duration'] + df_new['Heart_Rate'] + df_new['Body_Temp']) / 3
    
    # Drop original Sex column as we've encoded it
    df_new.drop('Sex', axis=1, inplace=True)
    
    return df_new

# Apply advanced feature engineering
print("Applying advanced feature engineering...")
train_fe = advanced_feature_engineering(train)
test_fe = advanced_feature_engineering(test)

print("\nEngineered train data shape:", train_fe.shape)
print("Engineered test data shape:", test_fe.shape)

print("\nNew features added:")
new_features = set(train_fe.columns) - set(train.columns) | {'Sex_male'}
print(len(new_features), "new features")
print(new_features)


# Split features and target
X = train_fe.drop(['Calories', 'id'], axis=1)
y = train_fe['Calories']
X_test = test_fe.drop(['id'], axis=1)

# Log transform the target (since RMSLE is the evaluation metric)
y_log = np.log1p(y)


# Check for outliers in the target variable
print("\nChecking for outliers in target variable...")
z_scores = stats.zscore(y)
outliers = (abs(z_scores) > 3)
print(f"Number of outliers detected: {np.sum(outliers)}")

# Plot outliers
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, alpha=0.5)
plt.scatter(np.where(outliers)[0], y[outliers], color='red', alpha=0.5)
plt.title('Target Variable with Outliers Highlighted', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Calories', fontsize=12)
plt.savefig('advanced_models/outliers.png')
plt.show()


# Create a version of the dataset with outliers removed for comparison
X_no_outliers = X[~outliers]
y_no_outliers = y[~outliers]
y_log_no_outliers = y_log[~outliers]

print(f"Dataset shape after outlier removal: {X_no_outliers.shape}")


# Train-validation split with stratification on binned target
y_bins = pd.qcut(y, q=10, labels=False)
X_train, X_val, y_train, y_val, y_log_train, y_log_val, y_bins_train, y_bins_val = train_test_split(
    X, y, y_log, y_bins, test_size=0.2, random_state=42, stratify=y_bins)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


# Standardize features
print("Standardizing features...")
preprocessor = StandardScaler()

# Fit preprocessor on training data
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)


# Best CatBoost parameters from optimization
best_catboost_params = {
    'iterations': 2998,
    'learning_rate': 0.016933195141294204,
    'depth': 10,
    'l2_leaf_reg': 1.02482828209293e-08,
    'border_count': 255,
    'bagging_temperature': 4.773325130759612,
    'random_strength': 0.1321531285154764,
    'verbose': False,
    'random_seed': 42
}

print("Training optimized CatBoost model...")
cb_model = cb.CatBoostRegressor(**best_catboost_params)
cb_model.fit(X_train_scaled, y_log_train, eval_set=(X_val_scaled, y_log_val), early_stopping_rounds=100, verbose=False)


# Evaluate CatBoost model
y_pred_cb_log = cb_model.predict(X_val_scaled)
y_pred_cb = np.expm1(y_pred_cb_log)
rmsle_cb = rmsle(y_val, y_pred_cb)
r2_cb = r2_score(y_val, y_pred_cb)

print(f"CatBoost - RMSLE: {rmsle_cb:.6f}, R²: {r2_cb:.6f}")


# Perform cross-validation on the optimized model
print("\nPerforming cross-validation on optimized CatBoost model...")

# Function for cross-validation with RMSLE
def rmsle_cv_score(model, X, y_log, cv=5):
    """Calculate cross-validation RMSLE score"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmsle_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        # Use .iloc for positional indexing with pandas Series
        y_log_train_cv, y_val_cv = y_log.iloc[train_idx], np.expm1(y_log.iloc[val_idx])
        
        # Train model
        model.fit(X_train_cv, y_log_train_cv, verbose=False)
        
        # Predict and calculate RMSLE
        y_pred_log = model.predict(X_val_cv)
        y_pred = np.expm1(y_pred_log)
        score = rmsle(y_val_cv, y_pred)
        rmsle_scores.append(score)
    
    return rmsle_scores

# Sample size for cross-validation (to manage memory constraints)
SAMPLE_SIZE = 200000
np.random.seed(42)
idx = np.random.choice(len(X_train_scaled), SAMPLE_SIZE, replace=False)
X_train_sample = X_train_scaled[idx]
y_log_train_sample = y_log_train.iloc[idx]

print(f"Running 5-fold cross-validation on CatBoost with {SAMPLE_SIZE} samples...")
cv_scores = rmsle_cv_score(cb_model, X_train_sample, y_log_train_sample, cv=5)

print(f"Cross-validation RMSLE scores: {cv_scores}")
print(f"Mean CV RMSLE: {np.mean(cv_scores):.6f}")
print(f"Standard deviation: {np.std(cv_scores):.6f}")


# Feature importance for CatBoost model
feature_importance = cb_model.get_feature_importance()

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title(f'Top 15 Feature Importance - Optimized CatBoost', fontsize=14)
plt.tight_layout()
plt.savefig('advanced_models/catboost_feature_importance.png')
plt.show()


# Plot actual vs predicted values for CatBoost model
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_cb, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Calories', fontsize=12)
plt.ylabel('Predicted Calories', fontsize=12)
plt.title(f'Actual vs Predicted Calories - Optimized CatBoost', fontsize=14)
plt.tight_layout()
plt.savefig('advanced_models/catboost_actual_vs_predicted.png')
plt.show()


# Generate predictions for test set
print("Generating predictions for test set...")
test_pred_cb_log = cb_model.predict(X_test_scaled)
test_pred_cb = np.expm1(test_pred_cb_log)

# Create submission dataframe
submission_final = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred_cb
})

# Verify submission format
print("\nSubmission format verification:")
print(f"Shape: {submission_final.shape}")
print("First 5 rows:")
print(submission_final.head())


# Check for invalid values (negative or NaN)
invalid_count = np.sum(np.isnan(test_pred_cb) | (test_pred_cb < 0))
if invalid_count > 0:
    print(f"WARNING: Found {invalid_count} invalid predictions (NaN or negative).")
    # Replace invalid values with minimum valid value (1.0)
    submission_final['Calories'] = np.maximum(submission_final['Calories'].fillna(1.0), 1.0)
    print("Invalid values have been replaced with 1.0")
else:
    print("All predictions are valid (no NaN or negative values).")


# Check prediction distribution
print("\nPrediction statistics:")
print(submission_final['Calories'].describe())


# Plot prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(submission_final['Calories'], kde=True)
plt.title('Distribution of Predicted Calories', fontsize=14)
plt.xlabel('Calories', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('advanced_models/prediction_distribution.png')
plt.show()


# Compare with training data distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train['Calories'], kde=True, color='blue')
plt.title('Training Data Calories', fontsize=12)
plt.xlabel('Calories', fontsize=10)

plt.subplot(1, 2, 2)
sns.histplot(submission_final['Calories'], kde=True, color='green')
plt.title('Predicted Calories', fontsize=12)
plt.xlabel('Calories', fontsize=10)

plt.tight_layout()
plt.savefig('advanced_models/train_vs_prediction_distribution.png')
plt.show()


# Save submission file
submission_path = 'advanced_models/advanced_submission.csv'
submission_final.to_csv(submission_path, index=False)
print(f"\nFinal submission file saved to {submission_path}")


# Compare with sample submission format
if submission_final.shape == submission.shape and all(submission_final.columns == submission.columns):
    print("Submission format matches the sample submission format.")
else:
    print("WARNING: Submission format doesn't match the sample submission format!")
    print(f"Sample shape: {submission.shape}, Our shape: {submission_final.shape}")
    print(f"Sample columns: {submission.columns}, Our columns: {submission_final.columns}")

