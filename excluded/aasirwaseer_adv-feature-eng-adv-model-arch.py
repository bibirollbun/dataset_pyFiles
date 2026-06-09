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
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

print("Loading and preparing data...")
# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Calculate RMSLE
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Feature importance analysis function
def analyze_feature_importance(X, y):
    print("Analyzing feature importance...")
    importance_dict = {}
    
    # Method 1: Mutual information (non-linear relationship measure)
    mi_scores = mutual_info_regression(X, y, random_state=SEED)
    mi_importance = pd.DataFrame({'Feature': X.columns, 'MI_Importance': mi_scores})
    mi_importance = mi_importance.sort_values('MI_Importance', ascending=False)
    
    # Method 2: Quick RandomForest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({'Feature': X.columns, 'RF_Importance': rf.feature_importances_})
    rf_importance = rf_importance.sort_values('RF_Importance', ascending=False)
    
    # Combine and return top features
    for idx, row in mi_importance.iterrows():
        importance_dict[row['Feature']] = importance_dict.get(row['Feature'], 0) + row['MI_Importance']
    
    for idx, row in rf_importance.iterrows():
        importance_dict[row['Feature']] = importance_dict.get(row['Feature'], 0) + row['RF_Importance']
    
    combined_importance = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Combined_Importance': list(importance_dict.values())
    }).sort_values('Combined_Importance', ascending=False)
    
    print("Top 20 important features:")
    print(combined_importance.head(20))
    return combined_importance

# Feature engineering function
def preprocess_data(df, is_train=True, top_features=None):
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
                               labels=range(5)).astype(int)
    
    # Exercise intensity metrics
    result['HR_Max'] = 220 - result['Age']
    result['HR_Intensity'] = result['Heart_Rate'] / result['HR_Max']
    result['HR_Reserve'] = result['HR_Max'] - result['Heart_Rate']
    
    # Core interaction terms
    result['Duration_Heart'] = result['Duration'] * result['Heart_Rate']
    result['Duration_Temp'] = result['Duration'] * result['Body_Temp']
    result['Duration_Heart_Temp'] = result['Duration'] * result['Heart_Rate'] * result['Body_Temp']
    result['Sex_Duration'] = result['Sex'] * result['Duration']
    result['Age_Heart'] = result['Age'] * result['Heart_Rate']
    
    # Power transformations
    result['Duration_Squared'] = result['Duration'] ** 2
    result['Heart_Rate_Squared'] = result['Heart_Rate'] ** 2
    
    # Physiological composite features
    result['Efficiency_Factor'] = result['Duration_Heart'] / (result['BMI'] + 1)
    result['BMI_Heart'] = result['BMI'] * result['Heart_Rate']
    result['Weight_Duration'] = result['Weight'] * result['Duration']
    result['Age_Duration'] = result['Age'] * result['Duration']
    result['Heart_Temp'] = result['Heart_Rate'] * result['Body_Temp']
    
    # Advanced exercise physiology metrics
    result['TRIMP'] = result['Duration'] * result['HR_Intensity'] * (0.64 * np.exp(1.92 * result['HR_Intensity']))
    result['Work_Estimate'] = result['Duration'] * result['Heart_Rate'] * result['Weight']
    
    # Lactate threshold and aerobic/anaerobic estimates
    result['Lactate_Threshold_HR'] = result['HR_Max'] * 0.85
    result['Anaerobic_Factor'] = np.maximum(0, (result['Heart_Rate'] - result['Lactate_Threshold_HR'])) * result['Duration'] / 60
    result['Aerobic_Factor'] = np.minimum(result['Heart_Rate'], result['Lactate_Threshold_HR']) * result['Duration'] / 60
    
    # Energy expenditure models
    result['MET_Estimate'] = 1 + (result['HR_Intensity'] * 9)
    result['Energy_From_HR'] = result['MET_Estimate'] * result['Weight'] * result['Duration'] / 60 * 5
    
    # Define feature columns and target
    if is_train:
        X = result.drop(['id', 'Calories'], axis=1)
        y = result['Calories']
        
        if top_features is not None:
            # Use predetermined top features
            X = X[top_features]
        return X, y
    else:
        X_test = result.drop(['id'], axis=1)
        if top_features is not None:
            # Use predetermined top features
            X_test = X_test[[col for col in top_features if col in X_test.columns]]
        return X_test

# Preprocess the train data for initial analysis
print("Preprocessing training data...")
X, y = preprocess_data(train_df, is_train=True)

# Analyze feature importance
importance_df = analyze_feature_importance(X, y)
top_features = importance_df['Feature'].head(40).tolist()

# Update datasets with selected features
print("Updating datasets with selected features...")
X = X[top_features]
X_test = preprocess_data(test_df, is_train=False, top_features=top_features)

# Create bins for target stratification in cross-validation
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Define categorical features for tree-based models
cat_features = ['Sex', 'Age_Group']

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
                valid_cat_features = [i for i, feat in enumerate(X.columns) if feat in cat_features]
                train_pool = Pool(X_train, y_train, cat_features=valid_cat_features)
                val_pool = Pool(X_val, y_val, cat_features=valid_cat_features)
                test_pool = Pool(test_X, cat_features=valid_cat_features)
                
                model = model_class(**model_params)
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50,
                          use_best_model=True, verbose=100)
                
                oof_preds[val_idx] = model.predict(val_pool)
                test_preds += model.predict(test_pool) / n_folds
            else:
                # If no valid categorical features
                model = model_class(**model_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50, use_best_model=True, verbose=100)
                
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

# Define base models
print("\nTraining models...")
base_models = [
    {
        'name': 'CatBoost',
        'class': CatBoostRegressor,
        'params': {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'random_strength': 0.6,
            'bagging_temperature': 0.1,
            'grow_policy': 'SymmetricTree',
            'min_data_in_leaf': 10,
            'random_seed': SEED,
            'verbose': 100
        }
    },
    {
        'name': 'LightGBM',
        'class': LGBMRegressor,
        'params': {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'num_leaves': 128,
            'max_depth': 12,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'verbose': -1
        }
    },
    {
        'name': 'XGBoost',
        'class': XGBRegressor,
        'params': {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 9,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'verbosity': 0
        }
    }
]

# Train base models and get out-of-fold predictions
oof_predictions = {}
test_predictions = {}
scores = {}

for model_dict in base_models:
    print(f"\nTraining {model_dict['name']}...")
    oof_preds, test_preds, score = get_oof_predictions(
        model_dict['class'],
        model_dict['params'],
        X, y, X_test, cv,
        cat_features=cat_features
    )
    
    oof_predictions[model_dict['name']] = oof_preds
    test_predictions[model_dict['name']] = test_preds
    scores[model_dict['name']] = score

# Create a meta-features dataset for stacking
print("\nCreating meta-features for stacking...")
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

print("\nTraining meta model...")
# Train meta-model
meta_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3.0,
    verbose=100,
    random_seed=SEED
)


# Get meta-model predictions with cross-validation
meta_oof_preds = np.zeros(len(meta_train))
meta_test_preds = np.zeros(len(meta_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(meta_train, y_bins)):
    print(f"Training meta-model fold {fold+1}/5")
    X_train, X_val = meta_train.iloc[train_idx], meta_train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    meta_model.fit(X_train, y_train)
    
    meta_oof_preds[val_idx] = meta_model.predict(X_val)
    meta_test_preds += meta_model.predict(meta_test) / 5

# Ensure positive predictions
meta_test_preds = np.maximum(meta_test_preds, 0.1)

# Calculate final validation RMSLE
meta_score = rmsle(meta_oof_preds, y)
print(f"Meta model OOF RMSLE: {meta_score:.6f}")

# Create weighted ensemble
print("\nCreating weighted ensemble...")
# Rank-based weighted ensemble (give more weight to models that performed better)
model_ranks = {model: rank for rank, (model, _) in enumerate(sorted(scores.items(), key=lambda x: x[1]))}

# Calculate weights inversely proportional to rank
base_weights = {model: 1/(rank+1) for model, rank in model_ranks.items()}

# Normalize weights
base_weight_sum = sum(base_weights.values())
base_weights = {model: weight/base_weight_sum for model, weight in base_weights.items()}

# Calculate rank-weighted predictions
rank_weighted_preds = sum(test_predictions[model] * weight for model, weight in base_weights.items())

# Final blend - combine meta model with weighted ensemble
final_blend = 0.7 * meta_test_preds + 0.3 * rank_weighted_preds

# Ensure predictions are positive
final_blend = np.maximum(final_blend, 0.1)

# Create submission files
print("\nCreating submission files...")
meta_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': meta_test_preds
})
meta_submission.to_csv('meta_submission.csv', index=False)

weighted_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': rank_weighted_preds
})
weighted_submission.to_csv('weighted_submission.csv', index=False)

final_submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_blend
})
final_submission.to_csv('final_submission.csv', index=False)

# Visualize model performance comparison
print("\nVisualizing model performance...")
plt.figure(figsize=(12, 8))

# Sort scores for better visualization
base_scores = sorted([(model, score) for model, score in scores.items()], key=lambda x: x[1])

# Plot base model scores
model_names = [model for model, _ in base_scores] + ['Meta Model']
model_scores = [score for _, score in base_scores] + [meta_score]
plt.barh(model_names, model_scores)
plt.xlabel('RMSLE')
plt.title('Model Performance')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_performance_comparison.png')

# Create prediction distribution visualization
plt.figure(figsize=(12, 8))

# Plot selected model predictions
for model_name in test_predictions:
    plt.hist(test_predictions[model_name], alpha=0.3, bins=50, label=model_name)
plt.hist(meta_test_preds, alpha=0.3, bins=50, label='Meta Model')
plt.hist(final_blend, alpha=0.5, bins=50, label='Final Blend')
plt.legend()
plt.title('Distribution of Predictions Across Models')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.savefig('prediction_distributions.png')

print("\nModel training complete! You now have several submission options:")
print("1. meta_submission.csv - Meta-model predictions")
print("2. weighted_submission.csv - Rank-weighted ensemble")
print("3. final_submission.csv - Final optimized blend (recommended)")

# Feature importance visualization
plt.figure(figsize=(14, 10))
importance_df_plot = importance_df.head(20).sort_values('Combined_Importance')
plt.barh(importance_df_plot['Feature'], importance_df_plot['Combined_Importance'])
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nAll visualizations and submission files created.")


