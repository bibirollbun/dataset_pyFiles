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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
display(train.info(), train.head(), train.describe().T)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

# Let's first check for any missing values in the original data
print("Missing values in train data:")
print(train.isnull().sum())
print("\nMissing values in test data:")
print(test.isnull().sum())

# Function to create features with careful handling of potential NaNs
def create_features(df):
    # Make a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Create cyclical features for day of year to capture seasonality
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day']/365)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day']/365)
    
    # Create temperature range feature
    df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
    
    # Create temperature and humidity interaction
    df_new['temp_humidity'] = df_new['temparature'] * df_new['humidity']
    df_new['dewpoint_diff'] = df_new['temparature'] - df_new['dewpoint']
    
    # Create pressure gradient features (can be useful for weather prediction)
    df_new['pressure_low'] = (df_new['pressure'] < df_new['pressure'].mean()).astype(int)
    
    # Wind features
    df_new['wind_chill'] = df_new['temparature'] - (0.5 * df_new['windspeed'])
    
    # Create categorical features for wind direction
    # Handle potential issues with wind direction
    # Make sure winddirection is within 0-360 range
    df_new['winddirection'] = df_new['winddirection'].clip(0, 360)
    
    # Convert continuous wind direction to cardinal directions
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    df_new['wind_direction_cat'] = pd.cut(df_new['winddirection'], 
                                          bins=bins, 
                                          labels=labels, 
                                          include_lowest=True)
    
    # One-hot encode wind direction
    wind_dummies = pd.get_dummies(df_new['wind_direction_cat'], prefix='wind_dir')
    df_new = pd.concat([df_new, wind_dummies], axis=1)
    
    # Create season based on day
    df_new['season'] = pd.cut(df_new['day'], 
                              bins=[0, 91, 182, 273, 366], 
                              labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    season_dummies = pd.get_dummies(df_new['season'], prefix='season')
    df_new = pd.concat([df_new, season_dummies], axis=1)
    
    # Drop original categorical columns that we've encoded
    df_new = df_new.drop(['wind_direction_cat', 'season'], axis=1)
    
    return df_new

# Feature correlation analysis
def plot_feature_correlations(df):
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return corr

# Transform the data
train_featured = create_features(train)
test_featured = create_features(test)

# Visualize correlations
corr_matrix = plot_feature_correlations(train_featured)
high_corr_with_target = corr_matrix['rainfall'].sort_values(ascending=False)
print("Features most correlated with rainfall:")
print(high_corr_with_target)

# Check for missing values after feature engineering
print("\nMissing values in train data after feature engineering:")
print(train_featured.isnull().sum().sum())
print("\nMissing values in test data after feature engineering:")
print(test_featured.isnull().sum().sum())

# If there are missing values, let's see which columns they're in
if test_featured.isnull().sum().sum() > 0:
    print("\nColumns with missing values in test data:")
    print(test_featured.isnull().sum()[test_featured.isnull().sum() > 0])

# Prepare data for modeling
X = train_featured.drop(['id', 'rainfall'], axis=1)
y = train_featured['rainfall']
X_test = test_featured.drop(['id'], axis=1)

# Handle missing values using imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling after imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_imputed)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Dictionary to store validation scores
val_scores = {}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predict probabilities on validation set
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    val_scores[name] = val_auc
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    print(f"{name} Validation AUC: {val_auc:.4f}")
    print(f"{name} Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {val_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()

# Find the best model
best_model_name = max(val_scores, key=val_scores.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with validation AUC: {val_scores[best_model_name]:.4f}")

# Feature importance for the best model (if available)
if hasattr(best_model, 'feature_importances_'):
    # We need the column names but we've applied transformations
    # Use original column names from X dataframe
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 most important features:")
    print(feature_importances.head(10))

# Let's focus on tuning the best model
if best_model_name in ['LightGBM', 'XGBoost', 'Random Forest']:
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'LightGBM':
        # Define simplified best parameters for LightGBM
        best_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'max_depth': 5,
            'min_child_samples': 30
        }
        tuned_model = lgb.LGBMClassifier(random_state=42, **best_params)
        
    elif best_model_name == 'XGBoost':
        # Define simplified best parameters for XGBoost
        best_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        tuned_model = xgb.XGBClassifier(random_state=42, **best_params)
        
    elif best_model_name == 'Random Forest':
        # Define simplified best parameters for Random Forest
        best_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
        tuned_model = RandomForestClassifier(random_state=42, **best_params)
    
    # Train the tuned model
    tuned_model.fit(X_train_scaled, y_train)
    
    # Evaluate the tuned model
    y_val_pred_tuned = tuned_model.predict_proba(X_val_scaled)[:, 1]
    val_auc_tuned = roc_auc_score(y_val, y_val_pred_tuned)
    
    print(f"Tuned {best_model_name} Validation AUC: {val_auc_tuned:.4f}")
    
    # If tuned model is better, use it for predictions
    if val_auc_tuned > val_scores[best_model_name]:
        best_model = tuned_model
        print("Using tuned model for final predictions.")
    else:
        print("Original model performs better than tuned model.")

# Generate predictions on the test set
test_preds = best_model.predict_proba(X_test_scaled)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")


submission.head()




