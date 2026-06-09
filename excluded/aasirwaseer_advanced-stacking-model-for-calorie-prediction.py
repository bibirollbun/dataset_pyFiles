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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("Loading and preparing data...")
# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Enhanced feature engineering with feature selection
def preprocess_data(df, is_train=True, feature_selector=None):
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    result['Sex'] = le.fit_transform(result['Sex'])
    
    # BASIC FEATURES
    # Physical characteristics
    result['BMI'] = result['Weight'] / ((result['Height']/100) ** 2)
    result['Weight_Height_Ratio'] = result['Weight'] / result['Height']
    result['Age_Group'] = pd.cut(result['Age'], bins=[19, 30, 40, 50, 60, 80], 
                              labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Exercise intensity metrics
    result['HR_Max'] = 220 - result['Age']
    result['HR_Intensity'] = result['Heart_Rate'] / result['HR_Max']
    result['HR_Reserve'] = result['HR_Max'] - result['Heart_Rate']  # Heart rate reserve
    result['HR_Reserve_Used'] = (result['Heart_Rate'] - 60) / (result['HR_Max'] - 60)  # % of reserve used
    
    # Core interaction terms from previous successful model
    result['Duration_Heart'] = result['Duration'] * result['Heart_Rate']
    result['Duration_Temp'] = result['Duration'] * result['Body_Temp']
    result['Duration_Heart_Temp'] = result['Duration'] * result['Heart_Rate'] * result['Body_Temp']
    result['Sex_Duration'] = result['Sex'] * result['Duration']
    result['Age_Heart'] = result['Age'] * result['Heart_Rate']
    
    # Power transformations of key features
    result['Duration_Squared'] = result['Duration'] ** 2
    result['Duration_Cubed'] = result['Duration'] ** 3
    result['Heart_Rate_Squared'] = result['Heart_Rate'] ** 2
    result['Duration_Heart_Squared'] = result['Duration_Heart'] ** 2
    
    # Log transformations
    result['Log_Duration'] = np.log1p(result['Duration'])
    result['Log_Heart_Rate'] = np.log1p(result['Heart_Rate'])
    result['Log_Duration_Heart'] = np.log1p(result['Duration_Heart'])
    
    # Physiological composite features
    result['Efficiency_Factor'] = result['Duration_Heart'] / (result['BMI'] + 1)
    result['HR_Efficiency'] = result['Heart_Rate'] / (result['Age'] + 20)
    result['BMI_Heart'] = result['BMI'] * result['Heart_Rate']
    result['Weight_Duration'] = result['Weight'] * result['Duration']
    result['Age_Duration'] = result['Age'] * result['Duration']
    result['Heart_Temp'] = result['Heart_Rate'] * result['Body_Temp']
    
    # Specialized ratios and differences
    result['Duration_Per_kg'] = result['Duration'] / result['Weight']
    result['Heart_Per_Temp'] = result['Heart_Rate'] / result['Body_Temp']
    
    # NEW ADVANCED FEATURES
    
    # Exercise physiology metrics
    result['EPOC_Estimate'] = result['Duration'] * (result['HR_Intensity'] ** 2)  # Excess post-exercise oxygen consumption estimate
    result['Training_Impulse'] = result['Duration'] * result['HR_Intensity'] * (1 + result['HR_Intensity'])  # Modified TRIMP formula
    result['Work_Estimate'] = result['Duration'] * result['Heart_Rate'] * result['Weight']  # Crude work estimate
    
    # Advanced metabolic indicators
    result['Metabolic_Factor'] = result['Duration'] * result['HR_Intensity'] * (result['Weight'] / result['Height'])
    result['Heat_Production'] = result['Duration_Heart_Temp'] / result['Weight']  # Heat produced relative to body size
    
    # Sex-specific interactions
    result['Sex_Weight_Interaction'] = result['Sex'] * result['Weight']
    result['Sex_BMI_Duration'] = result['Sex'] * result['BMI'] * result['Duration']
    result['Sex_Heart_Efficiency'] = result['Sex'] * result['HR_Efficiency']
    
    # Age-specific interactions
    result['Age_BMI_Heart'] = result['Age'] * result['BMI'] * result['Heart_Rate']
    result['Age_Duration_Heart'] = result['Age'] * result['Duration_Heart']
    
    # Temperature interactions
    result['Temp_BMI'] = result['Body_Temp'] * result['BMI']
    result['Temp_Duration_Weight'] = result['Body_Temp'] * result['Duration'] * result['Weight']
    
    # Define feature columns and target
    if is_train:
        X = result.drop(['id', 'Calories'], axis=1)
        y = result['Calories']
        
        # Apply feature selection if we're in stage 2
        if feature_selector is not None:
            selected_features = X.columns[feature_selector.get_support()]
            X = X[selected_features]
            return X, y, selected_features
        return X, y
    else:
        X_test = result.drop(['id'], axis=1)
        if feature_selector is not None:
            selected_features = feature_selector.get_support()
            X_test = X_test.iloc[:, selected_features]
        return X_test

# Calculate RMSLE
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Preprocess the train data for initial feature selection
X, y = preprocess_data(train_df, is_train=True)

# Create bins for target stratification in cross-validation
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Define categorical features for tree-based models
cat_features = ['Sex', 'Age_Group']

print("Performing feature selection using Lasso...")
# Feature selection with Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_selector = SelectFromModel(
    Lasso(alpha=0.005, random_state=42),
    max_features=40  # Keep top 40 features
)
feature_selector.fit(X_scaled, y)

# Get selected features and update datasets
X, y, selected_features = preprocess_data(train_df, is_train=True, feature_selector=feature_selector)
X_test = preprocess_data(test_df, is_train=False, feature_selector=feature_selector)

print(f"Selected {len(selected_features)} features for modeling: {selected_features}")

# Function to get OOF predictions for stacking
def get_oof_predictions(model_class, model_params, X, y, test_X, cv_strategy, cat_features=None):
    n_folds = cv_strategy.n_splits
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_X))
    
    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y_bins)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Special handling for CatBoost
        if model_class.__name__ == 'CatBoostRegressor':
            if cat_features and any(feat in X.columns for feat in cat_features):
                valid_cat_features = [feat for feat in cat_features if feat in X.columns]
                train_pool = Pool(X_train, y_train, cat_features=valid_cat_features)
                val_pool = Pool(X_val, y_val, cat_features=valid_cat_features)
                test_pool = Pool(test_X, cat_features=valid_cat_features)
                
                model = model_class(**model_params)
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True, verbose=100)
                
                oof_preds[val_idx] = model.predict(val_pool)
                test_preds += model.predict(test_pool) / n_folds
            else:
                # If no valid categorical features
                model = model_class(**model_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, use_best_model=True, verbose=100)
                
                oof_preds[val_idx] = model.predict(X_val)
                test_preds += model.predict(test_X) / n_folds
        else:
            # For other models
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(test_X) / n_folds
    
    # Ensure positive predictions
    oof_preds = np.maximum(oof_preds, 0.1)
    test_preds = np.maximum(test_preds, 0.1)
    
    # Calculate validation RMSLE
    score = rmsle(oof_preds, y)
    print(f"{model_class.__name__} - OOF RMSLE: {score:.6f}")
    
    return oof_preds, test_preds, score

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define base models for level 1
base_models = [
    {
        'name': 'CatBoost',
        'class': CatBoostRegressor,
        'params': {
            'iterations': 3000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 4.0,
            'random_strength': 0.6,
            'bagging_temperature': 0.1,
            'grow_policy': 'SymmetricTree',
            'min_data_in_leaf': 10,
            'random_seed': 42,
            'verbose': 100
        }
    },
    {
        'name': 'LightGBM',
        'class': LGBMRegressor,
        'params': {
            'n_estimators': 3000,
            'learning_rate': 0.03,
            'num_leaves': 128,
            'max_depth': 12,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'XGBoost',
        'class': XGBRegressor,
        'params': {
            'n_estimators': 3000,
            'learning_rate': 0.03,
            'max_depth': 9,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
    }
]

# Train base models and get out-of-fold predictions
print("\nTraining Level 1 models...")
oof_predictions = {}
test_predictions = {}
scores = {}

for model_dict in base_models:
    print(f"\nTraining {model_dict['name']}...")
    oof_preds, test_preds, score = get_oof_predictions(
        model_dict['class'],
        model_dict['params'],
        X, y, X_test, cv_strategy,
        cat_features=cat_features
    )
    
    oof_predictions[model_dict['name']] = oof_preds
    test_predictions[model_dict['name']] = test_preds
    scores[model_dict['name']] = score

# Create a meta-features dataset for level 2
meta_train = pd.DataFrame(oof_predictions)
meta_test = pd.DataFrame(test_predictions)

# Add selected original features to meta-features
top_original_features = ['Duration_Heart_Temp', 'Duration_Heart', 'HR_Intensity', 
                         'Sex_Duration', 'Duration', 'Heart_Rate']
valid_top_features = [f for f in top_original_features if f in X.columns]

if valid_top_features:
    for feat in valid_top_features:
        meta_train[feat] = X[feat].values
        meta_test[feat] = X_test[feat].values

print("\nTraining Level 2 (meta) model...")
# Train a meta-model on the out-of-fold predictions
meta_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bagging_temperature=0.1,
    verbose=100,
    random_seed=42
)

# Get meta-model predictions with cross-validation
meta_oof_preds = np.zeros(len(meta_train))
meta_test_preds = np.zeros(len(meta_test))

for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(meta_train, y_bins)):
    print(f"Training meta-model fold {fold+1}/5")
    X_train, X_val = meta_train.iloc[train_idx], meta_train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    meta_model.fit(X_train, y_train, verbose=100)
    
    meta_oof_preds[val_idx] = meta_model.predict(X_val)
    meta_test_preds += meta_model.predict(meta_test) / 5

# Ensure positive predictions
meta_oof_preds = np.maximum(meta_oof_preds, 0.1)
meta_test_preds = np.maximum(meta_test_preds, 0.1)

# Calculate final validation RMSLE
meta_score = rmsle(meta_oof_preds, y)
print(f"\nMeta-model OOF RMSLE: {meta_score:.6f}")

# Compare with base models
print("\nModel performance comparison:")
for model_name, score in scores.items():
    print(f"{model_name}: {score:.6f}")
print(f"Stacked model: {meta_score:.6f}")

# Calculate weighted average of all models (backup approach)
weights = {
    'CatBoost': 0.4,
    'LightGBM': 0.3,
    'XGBoost': 0.3,
}

weighted_test_preds = sum(test_predictions[model] * weights[model] for model in weights)

# Blend meta-model with weighted ensemble for extra robustness
final_preds = 0.7 * meta_test_preds + 0.3 * weighted_test_preds

# Create submission files
print("\nCreating submission files...")
meta_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': meta_test_preds
})
meta_submission.to_csv('stacked_meta_submission.csv', index=False)

weighted_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': weighted_test_preds
})
weighted_submission.to_csv('weighted_ensemble_submission.csv', index=False)

blended_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_preds
})
blended_submission.to_csv('blended_stacked_submission.csv', index=False)

print("All submission files created.")

# Visualize the predictions from different models
plt.figure(figsize=(12, 8))
for model_name, preds in test_predictions.items():
    plt.hist(preds, alpha=0.3, bins=50, label=model_name)
plt.hist(meta_test_preds, alpha=0.5, bins=50, label='Meta-model')
plt.hist(final_preds, alpha=0.7, bins=50, label='Final blend')
plt.legend()
plt.title('Distribution of Predictions Across Models')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.savefig('model_predictions_comparison.png')
plt.close()

# Create feature importance visualization for the base CatBoost model
if 'CatBoost' in base_models[0]['name']:
    # Refit a model on the full data for feature importance
    final_model = CatBoostRegressor(**base_models[0]['params'])
    
    # Handle categorical features properly
    if cat_features and any(feat in X.columns for feat in cat_features):
        valid_cat_features = [feat for feat in cat_features if feat in X.columns]
        final_pool = Pool(X, y, cat_features=valid_cat_features)
        final_model.fit(final_pool, verbose=100)
    else:
        final_model.fit(X, y, verbose=100)
    
    # Get feature importance
    feature_importance = final_model.get_feature_importance()
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("\nTop 20 important features:")
    print(importance_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_stacked.png')
    plt.close()

print("\nStacking model complete! You now have three submission options:")
print("1. stacked_meta_submission.csv - Pure meta-model predictions")
print("2. weighted_ensemble_submission.csv - Weighted average of base models")
print("3. blended_stacked_submission.csv - Blend of meta-model and weighted ensemble (recommended)")

