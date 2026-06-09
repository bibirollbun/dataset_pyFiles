import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("/kaggle/input/agriyield-2025/train.csv")
data


!pip install xgboost
!pip install catboost
!pip install lightgbm


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold



# Advanced boosting models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Install with: pip install catboost")

import warnings
warnings.filterwarnings('ignore')


# Load datasets
train_df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
      
# Display data info
print("\n" + "="*80)
print("DATASET ANALYSIS")
print("="*80)

print("\nğŸ“ˆ Training Data Overview:")
print(train_df.head())
print(f"\nTraining Data Info:")
print(train_df.info())
print(f"\nğŸ“Š Training Data Statistics:")
print(train_df.describe())

print(f"\nğŸ�¯ Target Variable (yield) Distribution:")
if 'yield' in train_df.columns:
    print(f"Mean: {train_df['yield'].mean():.2f}")
    print(f"Std: {train_df['yield'].std():.2f}")
    print(f"Min: {train_df['yield'].min():.2f}")
    print(f"Max: {train_df['yield'].max():.2f}")
print(f"\nğŸ“Š Training Data Statistics:")
print(train_df.describe())

print(f"\nğŸ�¯ Target Variable (yield) Distribution:")
if 'yield' in train_df.columns:
    print(f"Mean: {train_df['yield'].mean():.2f}")
    print(f"Std: {train_df['yield'].std():.2f}")
    print(f"Min: {train_df['yield'].min():.2f}")
    print(f"Max: {train_df['yield'].max():.2f}")

print(f"\nğŸ”� Test Data Overview:")
print(test_df.head())

# Identify feature columns (exclude id and target)
feature_cols = [col for col in train_df.columns if col not in ['field_id', 'yield']]
target_col = 'yield'

print(f"\nğŸ�¯ Target Column: {target_col}")
print(f"ğŸ“Š Feature Columns ({len(feature_cols)}): {feature_cols}")

# Prepare data for modeling
X = train_df[feature_cols]
y = train_df[target_col]
X_test_final = test_df[feature_cols]

# Check for missing values
print(f"\nğŸ”� Missing Values Check:")
print(f"Training features missing: {X.isnull().sum().sum()}")
print(f"Training target missing: {y.isnull().sum()}")
print(f"Test features missing: {X_test_final.isnull().sum().sum()}")

# Handle missing values if any
if X.isnull().sum().sum() > 0 or X_test_final.isnull().sum().sum() > 0:
    print("âš ï¸� Missing values detected. Filling with median...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test_final = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_final_scaled = scaler.transform(X_test_final)

print(f"\nğŸ“Š Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test_final.shape[0]} samples")


print(X_train_scaled.shape)
print(X_val_scaled.shape)
print(X_test_final.shape)
print(y_train.shape)


import tensorflow as tf

if tf.test.gpu_device_name():
    print("GPU Found")
    !nvidia-smi
else:
    print("GPU not found")



# Define models and their hyperparameter grids
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    
    'Ridge Regression': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        }
    },
    
    'Lasso Regression': {
        'model': Lasso(random_state=42),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        }
    },
    
    'ElasticNet': {
        'model': ElasticNet(random_state=42),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }
    },
    
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    
    'Extra Trees': {
        'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
    },
    
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.2]
        }
    },
    
    'K-Nearest Neighbors': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    }
}

# Add advanced boosting models if available
if XGBOOST_AVAILABLE:
    models['XGBoost'] = {
        'model': xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.1, 1.2]
        }
    }

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = {
        'model': lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
    }

if CATBOOST_AVAILABLE:
    models['CatBoost'] = {
        'model': cb.CatBoostRegressor(random_state=42, verbose=False),
        'params': {
            'iterations': [100, 200, 300],
            'depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bylevel': [0.8, 0.9, 1.0]
        }
    }

print("\n" + "="*80)
print("AVAILABLE MODELS:")
print("="*80)
print("ğŸ“Š Standard Models: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree")
print("ğŸŒ³ Ensemble Models: Random Forest, Gradient Boosting, Extra Trees, AdaBoost")
print("ğŸ”§ Other Models: SVR, KNN, Neural Network")

advanced_models = []
if XGBOOST_AVAILABLE:
    advanced_models.append("XGBoost")
if LIGHTGBM_AVAILABLE:
    advanced_models.append("LightGBM")
if CATBOOST_AVAILABLE:
    advanced_models.append("CatBoost")

if advanced_models:
    print(f"ğŸš€ Advanced Boosting: {', '.join(advanced_models)}")
else:
    print("âš ï¸�  Advanced Boosting: Not available (install xgboost, lightgbm, catboost)")

print(f"\nTotal Models Available: {len(models)}")

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Store results
results = []
trained_models = {}

print("\n" + "="*80)
print("MODEL TRAINING AND EVALUATION WITH HYPERPARAMETER TUNING")
print("="*80)

# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate each model
for name, config in models.items():
    print(f"\nğŸ”„ Training {name}...")
    
    try:
        model = config['model']
        param_grid = config['params']
        
        if param_grid:  # If there are parameters to tune
            # Use GridSearchCV with cross-validation
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"âœ… Best parameters: {best_params}")
        else:
            # No hyperparameters to tune
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = "No hyperparameters"
        
        # Store the trained model
        trained_models[name] = best_model
        
        # Make predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_val_pred = best_model.predict(X_val_scaled)
        
        # Calculate metrics
        train_rmse = rmse(y_train, y_train_pred)
        val_rmse = rmse(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Cross-validation RMSE (on full training data)
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                  cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        # Store results
        results.append({
            'Model': name,
            'Best_Params': str(best_params),
            'Train_RMSE': train_rmse,
            'Val_RMSE': val_rmse,
            'CV_RMSE': cv_rmse,
            'CV_Std': cv_std,
            'Train_R2': train_r2,
            'Val_R2': val_r2,
            'Trained_Model': best_model
        })
        
        print(f"ğŸ“Š Train RMSE: {train_rmse:.2f}")
        print(f"ğŸ“Š Validation RMSE: {val_rmse:.2f}")
        print(f"ğŸ“Š CV RMSE: {cv_rmse:.2f} Â± {cv_std:.2f}")
        print(f"ğŸ“Š Validation RÂ²: {val_r2:.4f}")
        
    except Exception as e:
        print(f"â�Œ Error training {name}: {str(e)}")
        continue

# Create results DataFrame and sort by Validation RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Val_RMSE').reset_index(drop=True)

print("\n" + "="*120)
print("FINAL RESULTS SUMMARY (Sorted by Validation RMSE)")
print("="*120)

# Display results table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
display_df = results_df.drop('Trained_Model', axis=1).round(4)
print(display_df.to_string(index=False))

# Best model analysis
if not results_df.empty:
    best_model_name = results_df.iloc[0]['Model']
    best_model = results_df.iloc[0]['Trained_Model']
    best_val_rmse = results_df.iloc[0]['Val_RMSE']
    
    print(f"\nğŸ�† BEST MODEL: {best_model_name}")
    print(f"ğŸ“Š Best Validation RMSE: {best_val_rmse:.4f}")
    print(f"ğŸ”§ Best Parameters: {results_df.iloc[0]['Best_Params']}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nğŸ“ˆ Feature Importance for {best_model_name}:")
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(feature_importance.to_string(index=False))
    
    elif hasattr(best_model, 'coef_'):
        print(f"\nğŸ“ˆ Feature Coefficients for {best_model_name}:")
        feature_coef = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': best_model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        print(feature_coef.to_string(index=False))

    print("\n" + "="*80)
    print("GENERATING KAGGLE SUBMISSION")
    print("="*80)
    
    # Generate predictions for test set using the best model
    test_predictions = best_model.predict(X_test_final_scaled)
    
    # Create submission file
    submission = sample_submission.copy()
    submission['yield'] = test_predictions
    
    # Save submission
    submission_filename = f'submission_{best_model_name.lower().replace(" ", "_")}.csv'
    submission.to_csv(submission_filename, index=False)






# Evaluate all trained models on the test set if true values are available
test_results = []

for model_name, model in trained_models.items():
    try:
        # Predict on test set
        y_test_pred = model.predict(X_test_final_scaled)
        
        # Calculate metrics if y_test_final is available
        test_rmse = rmse(y_test_final, y_test_pred)
        test_r2 = r2_score(y_test_final, y_test_pred)
        
        test_results.append({
            'Model': model_name,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        })
    except Exception as e:
        print(f"Error evaluating {model_name} on test set: {str(e)}")
        continue

# Convert to DataFrame and sort by Test RMSE
test_results_df = pd.DataFrame(test_results)
test_results_df = test_results_df.sort_values('Test_RMSE').reset_index(drop=True)

print("\n" + "="*100)
print("BEST MODEL BASED ON TEST DATA (X_test_final)")
print("="*100)
print(test_results_df.to_string(index=False))


# Select the best model based on test RMSE
best_test_model_name = test_results_df.iloc[0]['Model']
best_test_model = trained_models[best_test_model_name]

# Predict for submission
final_predictions = best_test_model.predict(X_test_final_scaled)
submission = sample_submission.copy()
submission['yield'] = final_predictions

# Save submission
submission_filename = f'submission_best_on_test_{best_test_model_name.lower().replace(" ", "_")}.csv'
submission.to_csv(submission_filename, index=False)

print(f"\nğŸ�† Submission file saved as: {submission_filename}")


