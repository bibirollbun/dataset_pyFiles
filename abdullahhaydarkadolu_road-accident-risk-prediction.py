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


# Data manipulation libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# Load datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Display basic information
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Display first few rows of train data
train_data.head()


# Display first few rows of test data
test_data.head()


# Check data types and missing values
print("Train data info:")
train_data.info()

print("\nTest data info:")
test_data.info()


# Statistical summary of train data
train_data.describe()


# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_data['accident_risk'], kde=True)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()


# Correlation matrix for numerical features
numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
correlation_matrix = train_data[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Analyze categorical features
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

# Create a figure with subplots for each categorical feature
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    # Calculate mean accident risk for each category
    category_risk = train_data.groupby(feature)['accident_risk'].mean().sort_values(ascending=False)
    
    # Plot bar chart
    sns.barplot(x=category_risk.index, y=category_risk.values, ax=axes[i])
    axes[i].set_title(f'Average Accident Risk by {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Average Accident Risk')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Analyze relationship between numerical features and target
numerical_features_no_target = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features_no_target):
    sns.scatterplot(x=train_data[feature], y=train_data['accident_risk'], alpha=0.5, ax=axes[i])
    axes[i].set_title(f'Accident Risk vs {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Accident Risk')

plt.tight_layout()
plt.show()


# Box plots for categorical features vs accident risk
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.boxplot(x=feature, y='accident_risk', data=train_data, ax=axes[i])
    axes[i].set_title(f'Accident Risk by {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Accident Risk')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Separate features and target
X = train_data.drop(['id', 'accident_risk'], axis=1)
y = train_data['accident_risk']
test_X = test_data.drop(['id'], axis=1)

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")


# Feature engineering

# Create a copy of the data for feature engineering
X_fe = X.copy()
test_X_fe = test_X.copy()

# 1. Create interaction features
X_fe['speed_curve_interaction'] = X_fe['speed_limit'] * X_fe['curvature']
test_X_fe['speed_curve_interaction'] = test_X_fe['speed_limit'] * test_X_fe['curvature']

# 2. Create binary features for high-risk conditions
X_fe['is_night'] = (X_fe['lighting'] == 'night').astype(int)
test_X_fe['is_night'] = (test_X_fe['lighting'] == 'night').astype(int)

X_fe['is_bad_weather'] = X_fe['weather'].isin(['rainy', 'foggy']).astype(int)
test_X_fe['is_bad_weather'] = test_X_fe['weather'].isin(['rainy', 'foggy']).astype(int)

# 3. Create feature for lane density (higher values might indicate more complex traffic patterns)
X_fe['lane_density'] = X_fe['num_lanes'] / 4  # Normalize by max lanes
test_X_fe['lane_density'] = test_X_fe['num_lanes'] / 4

# 4. Create risk factor feature combining multiple risk elements
X_fe['combined_risk_factor'] = (X_fe['speed_limit'] / 70) * (X_fe['curvature'] + 0.1) * (1 + 0.5 * X_fe['is_night']) * (1 + 0.3 * X_fe['is_bad_weather'])
test_X_fe['combined_risk_factor'] = (test_X_fe['speed_limit'] / 70) * (test_X_fe['curvature'] + 0.1) * (1 + 0.5 * test_X_fe['is_night']) * (1 + 0.3 * test_X_fe['is_bad_weather'])

# Update categorical and numerical columns
categorical_cols = X_fe.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("New features added:")
print([col for col in X_fe.columns if col not in X.columns])


# Create preprocessing pipeline
# Categorical features: one-hot encoding
# Numerical features: standard scaling

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_fe, y, test_size=0.2, random_state=42)


# Define a function to evaluate models
def evaluate_model(model, X_train, X_val, y_train, y_val):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Print results
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    return model, val_rmse


# Create and evaluate different models
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]),
    
    'Ridge Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ]),
    
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(n_estimators=100, random_state=42))
    ]),
    
    'LightGBM': Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(n_estimators=100, random_state=42))
    ])
}

# Dictionary to store model results
model_results = {}

# Evaluate each model
for name, model in models.items():
    print(f"\n{'-'*50}\nEvaluating {name}\n{'-'*50}")
    trained_model, val_rmse = evaluate_model(model, X_train, X_val, y_train, y_val)
    model_results[name] = {'model': trained_model, 'val_rmse': val_rmse}


# Compare model performances
model_names = list(model_results.keys())
rmse_values = [model_results[name]['val_rmse'] for name in model_names]

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, rmse_values)
plt.title('Model Comparison - Validation RMSE')
plt.xlabel('Model')
plt.ylabel('RMSE (lower is better)')
plt.xticks(rotation=45)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Find the best performing model
best_model_name = min(model_results, key=lambda x: model_results[x]['val_rmse'])
best_model = model_results[best_model_name]['model']
print(f"Best model: {best_model_name} with validation RMSE: {model_results[best_model_name]['val_rmse']:.4f}")


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

if best_model_name == 'XGBoost':
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': np.linspace(0.01, 0.3, 10),
        'model__subsample': np.linspace(0.7, 1.0, 4),
        'model__colsample_bytree': np.linspace(0.7, 1.0, 4)
    }
else:
    param_dist = {'model__n_estimators': [100, 200, 300]}  # Örnek olarak sadeleştir

random_search = RandomizedSearchCV(
    estimator=best_model,
    param_distributions=param_dist,
    n_iter=5,  # sadece 20 farklı kombinasyonu dene
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_fe, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best RMSE: {-random_search.best_score_:.4f}")

optimized_model = random_search.best_estimator_



# Extract feature importance if the model supports it
if hasattr(optimized_model.named_steps['model'], 'feature_importances_'):
    # Get feature names from preprocessor
    preprocessor = optimized_model.named_steps['preprocessor']
    feature_names = []
    
    # Get numerical feature names
    if numerical_cols:
        feature_names.extend(numerical_cols)
    
    # Get one-hot encoded feature names
    if categorical_cols:
        ohe = preprocessor.named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    # Get feature importances
    importances = optimized_model.named_steps['model'].feature_importances_
    
    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("Top 10 most important features:")
    print(feature_importance_df.head(10))


# Make predictions on test data
test_predictions = optimized_model.predict(test_X_fe)

# Ensure predictions are within the valid range [0, 1]
test_predictions = np.clip(test_predictions, 0, 1)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_data['id'],
    'accident_risk': test_predictions
})

# Display first few rows of submission
submission.head()


# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


# Analyze prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(submission['accident_risk'], kde=True)
plt.title('Distribution of Predicted Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()

# Compare with training data distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_data['accident_risk'], kde=True)
plt.title('Training Data Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(submission['accident_risk'], kde=True)
plt.title('Predicted Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

