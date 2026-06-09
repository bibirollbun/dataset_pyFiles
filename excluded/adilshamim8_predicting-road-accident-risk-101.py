# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Visualization settings
plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Get a glimpse of the training data
train.head()


# Summary statistics of the training data
train.describe(include='all')


# Check data types and missing values
print("Train dataset info:")
train.info()

print("\nTest dataset info:")
test.info()


# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(train['accident_risk'], kde=True)
plt.title('Distribution of Accident Risk', fontsize=16)
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()

# Statistical summary of target variable
print(train['accident_risk'].describe())


# Check for outliers in the target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['accident_risk'])
plt.title('Boxplot of Accident Risk', fontsize=16)
plt.show()


# Identifying categorical variables
categorical_features = [col for col in train.columns if train[col].dtype == 'object']
boolean_features = [col for col in train.columns if train[col].dtype == 'bool']
numeric_features = [col for col in train.columns if (train[col].dtype != 'object' and train[col].dtype != 'bool' and col != 'accident_risk' and col != 'id')]

print(f"Categorical features: {categorical_features}")
print(f"Boolean features: {boolean_features}")
print(f"Numeric features: {numeric_features}")


# Analyze categorical variables
for cat_feature in categorical_features:
    plt.figure(figsize=(12, 6))
    
    # Distribution of categories
    plt.subplot(1, 2, 1)
    sns.countplot(y=cat_feature, data=train, order=train[cat_feature].value_counts().index)
    plt.title(f'Distribution of {cat_feature}')
    plt.xlabel('Count')
    
    # Average accident risk by category
    plt.subplot(1, 2, 2)
    category_risk = train.groupby(cat_feature)['accident_risk'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='accident_risk', y=cat_feature, data=category_risk)
    plt.title(f'Average Accident Risk by {cat_feature}')
    plt.xlabel('Average Accident Risk')
    
    plt.tight_layout()
    plt.show()
    
    # Print unique values
    print(f"\nUnique values of {cat_feature}: {train[cat_feature].nunique()}")
    print(train[cat_feature].value_counts())


# Analyze boolean variables
for bool_feature in boolean_features:
    plt.figure(figsize=(12, 6))
    
    # Distribution of boolean values
    plt.subplot(1, 2, 1)
    sns.countplot(x=bool_feature, data=train)
    plt.title(f'Distribution of {bool_feature}')
    
    # Average accident risk by boolean value
    plt.subplot(1, 2, 2)
    sns.barplot(x=bool_feature, y='accident_risk', data=train)
    plt.title(f'Average Accident Risk by {bool_feature}')
    
    plt.tight_layout()
    plt.show()
    
    # Print value counts
    print(f"\n{bool_feature} value counts:")
    print(train[bool_feature].value_counts())
    print(f"Average accident risk when {bool_feature}=True: {train[train[bool_feature]==True]['accident_risk'].mean():.4f}")
    print(f"Average accident risk when {bool_feature}=False: {train[train[bool_feature]==False]['accident_risk'].mean():.4f}")


# Analyze numeric variables
for num_feature in numeric_features:
    plt.figure(figsize=(12, 6))
    
    # Distribution of numeric feature
    plt.subplot(1, 2, 1)
    sns.histplot(train[num_feature], kde=True)
    plt.title(f'Distribution of {num_feature}')
    
    # Relationship with target variable
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=num_feature, y='accident_risk', data=train, alpha=0.5)
    plt.title(f'{num_feature} vs. Accident Risk')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{num_feature} summary statistics:")
    print(train[num_feature].describe())
    
    # Calculate correlation with target
    corr = train[[num_feature, 'accident_risk']].corr().iloc[0, 1]
    print(f"Correlation with accident_risk: {corr:.4f}")


# Create a correlation heatmap for numeric features
numeric_cols = numeric_features + ['accident_risk']
plt.figure(figsize=(12, 10))
correlation_matrix = train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


# Analyze interactions between important features

# Speed limit and curvature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='speed_limit', y='curvature', hue='accident_risk', data=train, palette='viridis', alpha=0.6)
plt.title('Speed Limit vs. Curvature (colored by Accident Risk)')
plt.show()

# Road type and time of day
plt.figure(figsize=(14, 8))
pivot_data = train.pivot_table(index='road_type', columns='time_of_day', values='accident_risk', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Average Accident Risk by Road Type and Time of Day')
plt.tight_layout()
plt.show()

# Weather and lighting
plt.figure(figsize=(14, 8))
pivot_data = train.pivot_table(index='weather', columns='lighting', values='accident_risk', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Average Accident Risk by Weather and Lighting')
plt.tight_layout()
plt.show()


# Number of lanes and road type
plt.figure(figsize=(14, 8))
sns.boxplot(x='road_type', y='accident_risk', hue='num_lanes', data=train)
plt.title('Accident Risk by Road Type and Number of Lanes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Check feature distributions between train and test sets
for cat_feature in categorical_features:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    train[cat_feature].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'{cat_feature} Distribution in Train')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    test[cat_feature].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'{cat_feature} Distribution in Test')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


# Check numeric features distribution
for num_feature in numeric_features:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(train[num_feature], kde=True)
    plt.title(f'{num_feature} Distribution in Train')
    
    plt.subplot(1, 2, 2)
    sns.histplot(test[num_feature], kde=True)
    plt.title(f'{num_feature} Distribution in Test')
    
    plt.tight_layout()
    plt.show()


# Combine train and test for consistent preprocessing
all_data = pd.concat([train.drop('accident_risk', axis=1), test], axis=0, ignore_index=True)
print(f"Combined data shape: {all_data.shape}")

# Keep track of train and test IDs
train_idx = train['id']
test_idx = test['id']


# One-hot encode categorical features
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, drop_first=False)
print(f"Shape after one-hot encoding: {all_data_encoded.shape}")


# Create interaction features

# Interaction between speed limit and curvature
all_data_encoded['speed_curvature'] = all_data_encoded['speed_limit'] * all_data_encoded['curvature']

# Interaction between lanes and speed limit
all_data_encoded['lanes_speed'] = all_data_encoded['num_lanes'] * all_data_encoded['speed_limit']

# Risk index based on num_reported_accidents and speed
all_data_encoded['accident_speed_risk'] = all_data_encoded['num_reported_accidents'] * all_data_encoded['speed_limit'] / 100

# Curvature severity (squared curvature)
all_data_encoded['curvature_squared'] = all_data_encoded['curvature'] ** 2

# Create a "high risk" feature by combining multiple factors
for weather_type in ['Rain', 'Snow', 'Fog']:
    if f'weather_{weather_type}' in all_data_encoded.columns:
        for lighting in ['Dark_No_Lights', 'Dusk', 'Dawn']:
            if f'lighting_{lighting}' in all_data_encoded.columns:
                # High risk is bad weather + poor lighting
                col_name = f'high_risk_{weather_type}_{lighting}'
                all_data_encoded[col_name] = all_data_encoded[f'weather_{weather_type}'] & all_data_encoded[f'lighting_{lighting}']


# Check the shape after feature engineering
print(f"Data shape after feature engineering: {all_data_encoded.shape}")

# List the new features created
new_features = ['speed_curvature', 'lanes_speed', 'accident_speed_risk', 'curvature_squared']
high_risk_features = [col for col in all_data_encoded.columns if 'high_risk' in col]
new_features.extend(high_risk_features)

print(f"New features created: {len(new_features)}")
print(new_features)


# Split back into train and test
train_encoded = all_data_encoded[all_data_encoded['id'].isin(train_idx)]
test_encoded = all_data_encoded[all_data_encoded['id'].isin(test_idx)]

print(f"Train encoded shape: {train_encoded.shape}")
print(f"Test encoded shape: {test_encoded.shape}")


# Add target back to train_encoded
train_encoded = train_encoded.merge(train[['id', 'accident_risk']], on='id', how='left')


# Prepare X and y for modeling
X = train_encoded.drop(['id', 'accident_risk'], axis=1)
y = train_encoded['accident_risk']

# Prepare test data
X_test = test_encoded.drop(['id'], axis=1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")


# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")


# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_val, y_val):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    return train_rmse, val_rmse


# Define baseline models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
}

# Results dictionary
results = {}

# Evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    train_rmse, val_rmse = evaluate_model(model, X_train, y_train, X_val, y_val)
    results[name] = {'Train RMSE': train_rmse, 'Validation RMSE': val_rmse}
    print(f"{name} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
    print('-'*50)


# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('Validation RMSE')

# Plot results
plt.figure(figsize=(12, 8))
results_df.plot(kind='bar')
plt.title('Model Comparison - RMSE', fontsize=16)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

print(results_df)


# Based on baseline results, select the best model and tune it
# For this example, let's assume XGBoost performed the best

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Due to computational constraints, we'll use a randomized search
# In practice, you might want to use a more exhaustive search
from sklearn.model_selection import RandomizedSearchCV

# Initialize the model
xgb_model = xgb.XGBRegressor(random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings sampled
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)


# Get the best parameters
print("Best parameters:", random_search.best_params_)
print("Best RMSE score:", -random_search.best_score_)

# Evaluate best model on validation set
best_model = random_search.best_estimator_
val_preds = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE with best model: {val_rmse:.4f}")


# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 20 Feature Importances', fontsize=16)
plt.tight_layout()
plt.show()


# Let's create a blend of top performing models
# Based on earlier results, let's use XGBoost, LightGBM, and CatBoost

# Train the final models on the entire training data

# XGBoost with optimized parameters
final_xgb = xgb.XGBRegressor(**random_search.best_params_, random_state=42)
final_xgb.fit(X, y)

# LightGBM model
final_lgb = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
final_lgb.fit(X, y)

# CatBoost model
final_cb = cb.CatBoostRegressor(
    n_estimators=300,
    learning_rate=0.05,
    depth=5,
    random_state=42,
    verbose=0
)
final_cb.fit(X, y)


# Make predictions with each model
xgb_preds = final_xgb.predict(X_test)
lgb_preds = final_lgb.predict(X_test)
cb_preds = final_cb.predict(X_test)

# Blend predictions with weighted average
# Weights can be tuned based on validation performance
final_preds = 0.5 * xgb_preds + 0.3 * lgb_preds + 0.2 * cb_preds

# Ensure predictions are within [0, 1] range
final_preds = np.clip(final_preds, 0, 1)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_preds
})

# Check the submission dataframe
print(submission.head())

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


# Check submission statistics
print(f"Submission statistics:")
print(f"Min: {submission['accident_risk'].min()}")
print(f"Max: {submission['accident_risk'].max()}")
print(f"Mean: {submission['accident_risk'].mean()}")
print(f"Std: {submission['accident_risk'].std()}")

# Plot submission distribution
plt.figure(figsize=(10, 6))
sns.histplot(submission['accident_risk'], kde=True)
plt.title('Distribution of Predicted Accident Risk', fontsize=16)
plt.axvline(x=train['accident_risk'].mean(), color='r', linestyle='--', label=f'Train Mean: {train["accident_risk"].mean():.4f}')
plt.axvline(x=submission['accident_risk'].mean(), color='g', linestyle='--', label=f'Prediction Mean: {submission["accident_risk"].mean():.4f}')
plt.legend()
plt.show()

