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


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats
import joblib


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
display(df_train.head())
df_train.info()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print('Dataset shape:', df_test.shape)
display(df_test.head())
df_test.info()


y_train = df_train['accident_risk']
X_train = df_train.drop(['id' , 'accident_risk'], axis=1)
df_test_IDdel = df_test.drop('id', axis=1)
X_test = df_test_IDdel.copy()


X_train['is_train'] = 1
X_test['is_train'] = 0
X_total = pd.concat([X_train, X_test], axis = 0)


X_total.isna().sum().sort_values(ascending=False)


def cap_outliers(train_df, column):
    # Calculate Q1, Q3, and IQR from train only
    Q1 = train_df[column].quantile(0.25)
    Q3 = train_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the train data
    train_df[column] = train_df[column].clip(lower=lower_bound, upper=upper_bound)

    print(f"{column}: Capped outliers in train and test using bounds ({lower_bound:.2f}, {upper_bound:.2f})")

# Apply to all numeric columns
numeric_columns = X_total.select_dtypes(include=[np.number]).columns

for col in numeric_columns:
    cap_outliers(X_total, col)


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Get categorical columns
categorical_columns = X_total.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns.tolist())

# Initialize OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit on X_train and transform
X_total_encoded = X_total.copy()
X_total_encoded_cat = ohe.fit_transform(X_total[categorical_columns])
X_total_encoded_cat = pd.DataFrame(X_total_encoded_cat, 
                                   columns=ohe.get_feature_names_out(categorical_columns),
                                   index=X_total.index)

# Drop original categorical columns and add encoded columns
X_total_encoded = X_total_encoded.drop(columns=categorical_columns)
X_total_encoded = pd.concat([X_total_encoded, X_total_encoded_cat], axis=1)

# Show results
print(f"\nOriginal X_train shape: {X_total.shape}, Encoded X_train shape: {X_total_encoded.shape}")
display(X_total_encoded.head())



import seaborn as sns
import matplotlib.pyplot as plt

# Get numerical columns (excluding 'id' since it was dropped)
numerical_cols = X_total_encoded.select_dtypes(include=['float64', 'int64']).columns

# Set up the figure size
plt.figure(figsize=(15, len(numerical_cols)*4))

# Create subplots for each numerical variable
for i, col in enumerate(numerical_cols, 1):
    # Create a subplot for histograms
    plt.subplot(len(numerical_cols), 2, 2*i-1)
    sns.histplot(data=X_total_encoded, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    
    # Create a subplot for box plots
    plt.subplot(len(numerical_cols), 2, 2*i)
    sns.boxplot(data=X_total_encoded, x=col)
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)

# Adjust layout
plt.tight_layout()
plt.show()

# Display summary statistics
print("\nSummary Statistics for Continuous Variables:")
display(X_total_encoded[numerical_cols].describe())


# Initialize the StandardScaler
scaler = StandardScaler()

# Create a copy of the dataframe
X_total_scaled = X_total_encoded.copy()

# Exclude 'is_train' from scaling
cols_to_scale = [col for col in numerical_cols if col != 'is_train']

# Scale only the continuous variables (excluding 'is_train')
X_total_scaled[cols_to_scale] = scaler.fit_transform(X_total_scaled[cols_to_scale])

# Display the summary statistics of scaled variables
print("Summary statistics of scaled continuous variables:")
display(X_total_scaled[cols_to_scale].describe())

# Compare original vs scaled data for first few rows
print("\nComparison of original vs scaled data for first few rows:")
comparison = pd.DataFrame()
for col in cols_to_scale:
    comparison[f'{col}_original'] = X_total_encoded[col]
    comparison[f'{col}_scaled'] = X_total_scaled[col]
display(comparison.head())


# Feature Engineering

# 1. Create interaction features between road characteristics
X_total_scaled['lanes_speed_interaction'] = X_total_scaled['num_lanes'] * X_total_scaled['speed_limit']
X_total_scaled['curvature_speed_interaction'] = X_total_scaled['curvature'] * X_total_scaled['speed_limit']

# 2. Create risk density feature (accidents per lane)
X_total_scaled['accidents_per_lane'] = X_total_scaled['num_reported_accidents'] / X_total_scaled['num_lanes']

# 3. Create binary features for high-risk conditions
X_total_scaled['is_high_speed'] = (X_total_encoded['speed_limit'] >= 60).astype(int)
X_total_scaled['is_high_curvature'] = (X_total_encoded['curvature'] >= X_total_encoded['curvature'].quantile(0.75)).astype(int)

# 4. Create time-based risk features
X_total_scaled['is_rush_hour'] = ((X_total_encoded['time_of_day_morning'] == 1) | 
                            (X_total_encoded['time_of_day_evening'] == 1)).astype(int)

# 5. Create combined road condition features
X_total_scaled['high_risk_combination'] = ((X_total_scaled['is_high_speed'] == 1) & 
                                    (X_total_scaled['is_high_curvature'] == 1)).astype(int)

# 6. Weather and visibility risk
X_total_scaled['poor_visibility_conditions'] = ((X_total_encoded['weather_foggy'] == 1) | 
                                         (X_total_encoded['lighting_night'] == 1) |
                                         (X_total_encoded['lighting_dim'] == 1)).astype(int)

# Display the new features
print("Newly created features:")
new_features = ['lanes_speed_interaction', 'curvature_speed_interaction', 'accidents_per_lane',
                'is_high_speed', 'is_high_curvature', 'is_rush_hour', 
                'high_risk_combination', 'poor_visibility_conditions']
print("\nSample of new features:")
display(X_total_scaled[new_features].head())


bool_columns = X_total_scaled.select_dtypes(include=['bool']).columns
X_total_scaled[bool_columns] = X_total_scaled[bool_columns].astype('int64')


# Separating X_total_scaled back into train and test sets
X_train_processed = X_total_scaled[X_total_scaled['is_train'] == 1].drop('is_train', axis=1)
X_test_processed = X_total_scaled[X_total_scaled['is_train'] == 0].drop('is_train', axis=1)


# Import necessary libraries for modeling and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_processed, y_train)

# Make predictions on training set
y_train_pred = lr_model.predict(X_train_processed)

# Calculate training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Metrics:")
print(f"R² Score: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE: {train_mae:.4f}")

# Make predictions on test set
y_test_pred = lr_model.predict(X_test_processed)

# Display the first few predictions
print("\nFirst few predictions on test set:")
print(y_test_pred[:5])


# Analyze feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'Coefficient': lr_model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

# Display top 15 most important features
print("Top 15 Most Important Features:")
display(feature_importance.head(15))

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_train - y_train_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()


from sklearn.model_selection import cross_val_score, KFold

# Create KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross validation
cv_scores = cross_val_score(lr_model, X_train_processed, y_train, cv=kf, 
                           scoring='r2',
                           n_jobs=-1)

# Print results
print("Cross Validation Scores:", cv_scores)
print("Average R² Score: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std() * 2))

# Get RMSE scores
rmse_scores = np.sqrt(-cross_val_score(lr_model, X_train_processed, y_train, 
                                     scoring='neg_mean_squared_error',
                                     cv=kf, n_jobs=-1))

print("\nRMSE Scores:", rmse_scores)
print("Average RMSE: {:.4f} (+/- {:.4f})".format(rmse_scores.mean(), rmse_scores.std() * 2))


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import numpy as np

import matplotlib.pyplot as plt

# Define parameter grids for each model
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
elastic_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Initialize models
ridge = Ridge(random_state=42)
lasso = Lasso(random_state=42)
elastic = ElasticNet(random_state=42)

# Create GridSearchCV objects
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
elastic_cv = GridSearchCV(elastic, elastic_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit models
print("Fitting Ridge...")
ridge_cv.fit(X_train_processed, y_train)
print("Fitting Lasso...")
lasso_cv.fit(X_train_processed, y_train)
print("Fitting ElasticNet...")
elastic_cv.fit(X_train_processed, y_train)

# Get best parameters and scores
models = {'Ridge': ridge_cv, 'Lasso': lasso_cv, 'ElasticNet': elastic_cv}
for name, model in models.items():
    print(f"\n{name} Results:")
    print(f"Best parameters: {model.best_params_}")
    print(f"Best RMSE: {np.sqrt(-model.best_score_):.4f}")
    
    # Make predictions and calculate metrics
    y_train_pred = model.predict(X_train_processed)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test_processed)
    print(f"First few test predictions: {y_test_pred[:5]}")

    # Plot actual vs predicted for training data
    plt.figure(figsize=(10, 5))
    
    # Actual vs Predicted plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} - Actual vs Predicted')
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{name} - Residual Plot')
    
    plt.tight_layout()
    plt.show()


# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

# Initialize the model with default parameters
gb_model = GradientBoostingRegressor(random_state=42)

# ----- Cross-Validation -----
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# R2 Cross-validation
cv_r2_scores = cross_val_score(gb_model, X_train_processed, y_train, cv=cv, scoring='r2')

# RMSE Cross-validation (note: scoring expects higher=better, so we use negative MSE)
cv_rmse_scores = np.sqrt(-cross_val_score(
    gb_model, X_train_processed, y_train, cv=cv, scoring='neg_mean_squared_error'
))

# MAE Cross-validation
cv_mae_scores = -cross_val_score(
    gb_model, X_train_processed, y_train, cv=cv, scoring='neg_mean_absolute_error'
)

print("Cross-Validation Results (3-fold):")
print(f"Average R2: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"Average RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
print(f"Average MAE: {cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}")


# ----- Original Model Training -----
# Fit the model
gb_model.fit(X_train_processed, y_train)

# Make predictions
y_train_pred_gb = gb_model.predict(X_train_processed)
y_test_pred_gb = gb_model.predict(X_test_processed)

# Calculate metrics
train_mse_gb = mean_squared_error(y_train, y_train_pred_gb)
train_rmse_gb = np.sqrt(train_mse_gb)
train_mae_gb = mean_absolute_error(y_train, y_train_pred_gb)
train_r2_gb = r2_score(y_train, y_train_pred_gb)

print("Training Metrics for Gradient Boosting Model:")
print(f"MSE: {train_mse_gb:.4f}")
print(f"RMSE: {train_rmse_gb:.4f}")
print(f"MAE: {train_mae_gb:.4f}")
print(f"R2 Score: {train_r2_gb:.4f}")

# Feature importance
feature_importance_gb = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': gb_model.feature_importances_
})
feature_importance_gb = feature_importance_gb.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_gb.head(10))



from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# Initialize models with default parameters
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)
cat_model = CatBoostRegressor(
    random_state=42,
    verbose=False,
    allow_writing_files=False
)

# Create KFold cross-validator
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Dictionary to store results
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model
}

# Perform cross-validation for each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # R² scores
    r2_scores = cross_val_score(model, X_train_processed, y_train, 
                               cv=kf, scoring='r2', n_jobs=-1)
    
    # RMSE scores
    rmse_scores = np.sqrt(-cross_val_score(model, X_train_processed, y_train,
                                         scoring='neg_mean_squared_error',
                                         cv=kf, n_jobs=-1))
    
    # MAE scores
    mae_scores = -cross_val_score(model, X_train_processed, y_train,
                                scoring='neg_mean_absolute_error',
                                cv=kf, n_jobs=-1)
    
    print(f"{name} Cross-Validation Results:")
    print(f"R² Score: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    print(f"RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    print(f"MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")

    # Fit the model on full training data
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_processed)
    y_test_pred = model.predict(X_test_processed)
    
    # Calculate metrics on training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"\nTraining Metrics:")
    print(f"R² Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    
    # Display feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train_processed.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features for {name}:")
        print(feature_importance.head(10))


from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression

# Define base models
base_models = {
    'xgb': xgb_model,
    'lgb': lgb_model,
    'cat': cat_model,
    'gb': gb_model
}

# Function to get out-of-fold predictions
def get_oof_predictions(models, X, y, n_splits=5):
    # Initialize the matrix to store predictions
    oof_preds = np.zeros((len(models), X.shape[0]))
    
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # For each model
    for i, (name, model) in enumerate(models.items()):
        print(f"Getting OOF predictions for {name}...")
        
        # Initialize array to store this model's predictions
        oof_pred = np.zeros(X.shape[0])
        
        # For each fold
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            oof_pred[val_idx] = model.predict(X_val_fold)
        
        # Store predictions
        oof_preds[i] = oof_pred
    
    return oof_preds.T

# Get out-of-fold predictions for training data
print("Getting out-of-fold predictions...")
oof_preds = get_oof_predictions(base_models, X_train_processed, y_train)

# Train meta model
print("\nTraining meta model...")
meta_model = LinearRegression()
meta_model.fit(oof_preds, y_train)

# Get predictions from all base models on test set
print("\nGetting base model predictions on test set...")
test_preds = np.zeros((len(base_models), X_test_processed.shape[0]))
for i, (name, model) in enumerate(base_models.items()):
    print(f"Getting predictions for {name}...")
    model.fit(X_train_processed, y_train)
    test_preds[i] = model.predict(X_test_processed)

# Make final predictions using meta model
final_preds = meta_model.predict(test_preds.T)

# Print meta-model coefficients
print("\nMeta-model coefficients:")
for name, coef in zip(base_models.keys(), meta_model.coef_):
    print(f"{name}: {coef:.4f}")

print(f"Meta-model intercept: {meta_model.intercept_:.4f}")

# Calculate OOF R² score
oof_r2 = r2_score(y_train, meta_model.predict(oof_preds))
print(f"\nOut-of-fold R² score: {oof_r2:.4f}")

# First few predictions
print("\nFirst few final predictions:")
print(final_preds[:5])


oof_rmse = np.sqrt(mean_squared_error (y_train, meta_model.predict(oof_preds)))
print(f"\nOut-of-fold RMSE score: {oof_rmse:.4f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


sub['accident_risk'] = test_preds.T
sub.to_csv('/kaggle/working/submission.csv',index=False)
sub.head()

