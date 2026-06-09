import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor




df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)
display(df_train.head())


# Drop 'id' column (not useful for modeling)
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


target_col = 'Calories'
print(df_train[target_col].describe())
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(df_train[target_col], bins=30, color='skyblue', edgecolor='black')
plt.title('Calories Distribution')
plt.subplot(1,2,2)
plt.boxplot(df_train[target_col], vert=False)
plt.title('Boxplot of Calories')
plt.show()


numeric_cols = df_train.select_dtypes(include=[np.number]).drop(columns=[target_col]).columns.tolist()
df_train[numeric_cols].hist(figsize=(12,8), bins=30)
plt.suptitle('Numeric Feature Distributions')
plt.show()


corrs = df_train.corr(numeric_only=True)[target_col].sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.heatmap(df_train[numeric_cols + [target_col]].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


sns.countplot(x='Sex', data=df_train)
plt.title('Sex Distribution')
plt.show()


print('Missing Values in Train Dataset')
df_train.isna().sum()


print('Missing Values in Test Dataset')
df_test.isna().sum()


def feature_engineering(df, handle_outliers=True):
    """Comprehensive feature engineering function combining all transformations"""
    # Store original columns to determine which ones were added
    original_columns = df.columns.tolist()
    
    # Define numerical features for interactions
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    
    # --- ENCODE CATEGORICAL ---
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # --- BASIC BODY METRICS ---
    # BMI calculation
    height_m = df['Height'] / 100
    df['BMI'] = df['Weight'] / (height_m ** 2)
    
    # BMR (Basal Metabolic Rate)
    df['BMR'] = np.where(
        df['Sex'] == 1,
        88.362 + (13.397 * df['Weight']) + (4.799 * df['Height']) - (5.677 * df['Age']),
        447.593 + (9.247 * df['Weight']) + (3.098 * df['Height']) - (4.330 * df['Age'])
    )
    
    # Body Surface Area
    df['BSA'] = 0.007184 * (df['Height'] ** 0.725) * (df['Weight'] ** 0.425)
    
    # --- HEART RATE FEATURES ---
    # Max heart rate approximation
    df['Max_HR'] = 220 - df['Age']
    df['HR_Zone'] = (df['Heart_Rate'] / df['Max_HR'] * 10).astype(int)
    df['HR_Reserve'] = df['Heart_Rate'] - (220 - df['Age'])
    df['Normalized_HR'] = df['Heart_Rate'] / df['Age']
    
    # --- INTENSITY/EFFORT METRICS ---
    # MET - Metabolic Equivalent of Task (approximation)
    met = 3.5 * df['Weight'] / 200
    df['MET'] = met
    
    # Calorie estimation based on MET and Duration
    df['Estimated_Calories'] = met * df['Duration']
    df['Calories_per_minute'] = df['Estimated_Calories'] / df['Duration']
    
    # --- INTERACTION FEATURES ---
    # Duration and Heart Rate combinations (strong predictors)
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']
    df['Duration_HR_by_Age'] = df['Duration_HR'] / df['Age']
    df['HeartRate_by_Weight'] = df['Heart_Rate'] / df['Weight']
    df['BodyTemp_by_HeartRate'] = df['Body_Temp'] / df['Heart_Rate']
    df['Weight_Height_Ratio'] = df['Weight'] / height_m
    
    # --- POLYNOMIAL FEATURES ---
    # Square features for highly correlated variables
    df['Duration_squared'] = df['Duration'] ** 2
    df['Heart_Rate_squared'] = df['Heart_Rate'] ** 2
    
    # --- CROSS-TERM INTERACTIONS ---
    # Create pairwise interactions between numerical features
    print(f"Creating cross-term interactions between {len(numerical_features)} numerical features...")
    interaction_count = 0
    
    # Nested loop to create interaction terms
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            # Skip combinations we already created manually
            if (feature1 == 'Duration' and feature2 == 'Heart_Rate') or \
               (feature1 == 'Body_Temp' and feature2 == 'Heart_Rate') or \
               (feature1 == 'Heart_Rate' and feature2 == 'Weight'):
                continue
            
            cross_term_name = f"{feature1}_x_{feature2}"
            df[cross_term_name] = df[feature1] * df[feature2]
            interaction_count += 1
    
    print(f"Added {interaction_count} cross-term interaction features")
    
    # --- OUTLIER HANDLING ---
    if handle_outliers:
        outlier_columns = ['Duration', 'Heart_Rate', 'Duration_HR', 'Duration_squared', 'Heart_Rate_squared']
        for col in outlier_columns:
            q1, q3 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q1, q3)
    
    # Return new feature names
    new_features = [col for col in df.columns if col not in original_columns and col != 'Calories']
    return df, new_features

# Apply feature engineering to train and test sets
print("Applying feature engineering...")
df_train, engineered_features = feature_engineering(df_train)
df_test, _ = feature_engineering(df_test)

print(f"Added {len(engineered_features)} engineered features:")
print(", ".join(engineered_features))

# Visualize top correlations with target
plt.figure(figsize=(12, 8))
corrs = df_train.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
top_corrs = corrs.head(15)
sns.barplot(x=top_corrs.values, y=top_corrs.index)
plt.title('Top 15 Features Correlation with Calories')
plt.tight_layout()
plt.show()


X = df_train.drop(columns=[target_col])
y = np.log1p(df_train[target_col])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, y_train.shape, X_val.shape, y_val.shape


import time
from sklearn.model_selection import KFold

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def ensemble_cross_validation(X, y, X_test, models_config, n_folds=5):
    """Cross-validation with multiple models and ensemble prediction"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Arrays to store predictions
    oof_preds = {name: np.zeros(len(X)) for name in models_config.keys()}
    test_preds = {name: np.zeros(len(X_test)) for name in models_config.keys()}
    
    # DataFrame to store RMSE scores for each fold and model
    scores_df = pd.DataFrame(columns=['fold', 'model', 'rmse', 'time'])
    
    # Start CV loop
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n{'='*10} Fold {fold+1}/{n_folds} {'='*10}")
        
        # Split data for this fold
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        fold_start_time = time.time()
        
        # Train each model in the ensemble
        for name, model_config in models_config.items():
            print(f"Training {name}...")
            model_start_time = time.time()
            
            # Instantiate model with config
            model = model_config['class'](**model_config['params'])
            
            # Fit model
            if name == 'xgb' or name == 'lgbm':  # These support eval_set
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                )
            else:  # Other models without eval_set
                model.fit(X_train, y_train)
            
            # Make predictions
            oof_preds[name][valid_idx] = model.predict(X_valid)
            test_preds[name] += model.predict(X_test) / n_folds
            
            # Evaluate this model on this fold
            execution_time = time.time() - model_start_time
            fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[name][valid_idx]))
            print(f"{name} - Fold {fold+1} RMSE: {fold_rmse:.6f} (took {execution_time:.1f}s)")
            
            # Store results in DataFrame
            scores_df = pd.concat([scores_df, pd.DataFrame({
                'fold': [fold + 1],
                'model': [name],
                'rmse': [fold_rmse],
                'time': [execution_time]
            })])
        
        # Create ensemble prediction for this fold
        fold_ensemble = np.zeros(len(valid_idx))
        for name in models_config.keys():
            fold_ensemble += oof_preds[name][valid_idx] / len(models_config)
            
        # Evaluate ensemble on this fold
        ensemble_fold_rmse = np.sqrt(mean_squared_error(y_valid, fold_ensemble))
        ensemble_time = time.time() - fold_start_time
        print(f"Ensemble - Fold {fold+1} RMSE: {ensemble_fold_rmse:.6f}")
        print(f"Fold {fold+1} completed in {ensemble_time:.1f}s")
        
        # Store ensemble results
        scores_df = pd.concat([scores_df, pd.DataFrame({
            'fold': [fold + 1],
            'model': ['ensemble'],
            'rmse': [ensemble_fold_rmse],
            'time': [ensemble_time]
        })])
    
    # Calculate final metrics for each model
    print("\n" + "="*50)
    print("Final Results:")
    
    # Create a DataFrame for final results
    final_scores = []
    
    for name in models_config.keys():
        final_rmse = np.sqrt(mean_squared_error(y, oof_preds[name]))
        print(f"{name} - Final CV RMSE: {final_rmse:.6f}")
        final_scores.append({'model': name, 'final_rmse': final_rmse})
    
    # Create final ensemble prediction (average of all models)
    oof_ensemble = np.zeros(len(X))
    test_ensemble = np.zeros(len(X_test))
    
    for name in models_config.keys():
        oof_ensemble += oof_preds[name] / len(models_config)
        test_ensemble += test_preds[name] / len(models_config)
        
    ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
    print(f"Ensemble - Final CV RMSE: {ensemble_rmse:.6f}")
    final_scores.append({'model': 'ensemble', 'final_rmse': ensemble_rmse})
    
    # Create final scores DataFrame
    final_scores_df = pd.DataFrame(final_scores)
    
    return {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'oof_ensemble': oof_ensemble,
        'test_ensemble': test_ensemble,
        'rmse': ensemble_rmse,
        'scores_df': scores_df,
        'final_scores_df': final_scores_df
    }

xgb_params = {
    "max_depth": 10, 
    "colsample_bytree": 0.7, 
    "subsample": 0.9, 
    "n_estimators": 2000, 
    "learning_rate": 0.02,
    "gamma": 0.01, 
    "max_delta_step": 2, 
    "early_stopping_rounds": 100, 
    "eval_metric": 'rmse',
    "enable_categorical": True, 
    "random_state": 42
}

lgbm_params = { 
    "max_depth": 10, 
    "colsample_bytree": 0.7, 
    "subsample": 0.9, 
    "n_estimators": 2000, 
    "learning_rate": 0.02,
    "gamma": 0.01, 
    "max_delta_step": 2, 
    "early_stopping_rounds": 100, 
    "eval_metric": 'rmse',
    "enable_categorical": True, 
    "random_state": 42,
    "verbose":-1
}

# Configure models for ensemble
models_config = {
    'xgb': {
        'class': XGBRegressor,
        'params': xgb_params
    },
    'lgbm': {
        'class': LGBMRegressor,
        'params': lgbm_params
    },
    'cbdt': {
        'class': CatBoostRegressor,
        'params': {
            "verbose":100, 
            "random_seed":42, 
            "cat_features":['Sex'], 
            "early_stopping_rounds":100
        }
    }
}

# Run ensemble cross-validation
cv_results = ensemble_cross_validation(X, y, df_test, models_config, n_folds=5)


# Convert log predictions back to original scale
ensemble_preds = np.expm1(cv_results['test_ensemble'])
ensemble_oof = np.expm1(cv_results['oof_ensemble'])
y_true = np.expm1(y)

# Calculate RMSLE (which is our competition metric)
validation_rmsle = rmsle(y_true, ensemble_oof)
print(f"Ensemble Validation RMSLE: {validation_rmsle:.6f}")

# Compare with individual models
rmsle_scores = []
for name in models_config.keys():
    model_oof = np.expm1(cv_results['oof_preds'][name])
    model_rmsle = rmsle(y_true, model_oof)
    print(f"{name} Validation RMSLE: {model_rmsle:.6f}")
    rmsle_scores.append({'model': name, 'rmsle': model_rmsle})

# Add ensemble RMSLE
rmsle_scores.append({'model': 'ensemble', 'rmsle': validation_rmsle})

# Create DataFrame of RMSLE scores
rmsle_df = pd.DataFrame(rmsle_scores).sort_values('rmsle')
display(rmsle_df)


# Function to create weighted ensemble predictions based on model performance
def create_weighted_ensemble(oof_preds, test_preds, y_true, weights=None):
    """Creates weighted ensemble predictions in log space for optimal RMSLE performance"""
    # If no weights provided, optimize weights based on OOF performance
    if weights is None:
        # Calculate error for each model (lower is better)
        errors = {}
        for name, preds in oof_preds.items():
            errors[name] = np.sqrt(mean_squared_error(y_true, preds))
        
        # Convert errors to weights (better models get higher weight)
        # Use inverse of error as weight, then normalize
        raw_weights = {name: 1/err for name, err in errors.items()}
        total = sum(raw_weights.values())
        weights = {name: w/total for name, w in raw_weights.items()}
        
        print("\nOptimized model weights based on performance:")
        for name, weight in weights.items():
            print(f"{name}: {weight:.4f}")
    
    # Create weighted ensemble predictions (still in log space)
    oof_ensemble = np.zeros(len(y_true))
    test_ensemble = np.zeros(len(next(iter(test_preds.values()))))
    
    for name in oof_preds.keys():
        oof_ensemble += weights[name] * oof_preds[name]
        test_ensemble += weights[name] * test_preds[name]
    
    # Evaluate ensemble
    ensemble_rmse = np.sqrt(mean_squared_error(y_true, oof_ensemble))
    print(f"Weighted Ensemble RMSE: {ensemble_rmse:.6f}")
    
    return oof_ensemble, test_ensemble, weights

# Create weighted ensemble from CV results
weighted_oof, weighted_test, optimal_weights = create_weighted_ensemble(
    oof_preds=cv_results['oof_preds'],
    test_preds=cv_results['test_preds'],
    y_true=y
)

# Compare weighted vs unweighted ensemble performance
unweighted_rmse = cv_results['rmse']
weighted_rmse = np.sqrt(mean_squared_error(y, weighted_oof))

print(f"\nPerformance comparison:")
print(f"Equal weights ensemble RMSE: {unweighted_rmse:.6f}")
print(f"Optimized weights ensemble RMSE: {weighted_rmse:.6f}")
print(f"Improvement: {(unweighted_rmse - weighted_rmse):.6f} ({(unweighted_rmse - weighted_rmse)/unweighted_rmse*100:.2f}%)")

# Convert optimized ensemble predictions back to original scale
weighted_ensemble_preds = np.expm1(weighted_test)
weighted_ensemble_oof = np.expm1(weighted_oof)

# Calculate RMSLE for weighted ensemble
weighted_validation_rmsle = rmsle(np.expm1(y), weighted_ensemble_oof)
print(f"\nWeighted Ensemble Validation RMSLE: {weighted_validation_rmsle:.6f}")
print(f"Original Ensemble Validation RMSLE: {validation_rmsle:.6f}")
print(f"RMSLE Improvement: {(validation_rmsle - weighted_validation_rmsle):.6f} ({(validation_rmsle - weighted_validation_rmsle)/validation_rmsle*100:.2f}%)")


# Display detailed per-fold performance
print("\n=== Per-fold Model Performance ===")
display(cv_results['scores_df'])

# Display final CV scores
print("\n=== Final CV Performance ===")
display(cv_results['final_scores_df'].sort_values('final_rmse'))

# Visualize RMSE across folds
plt.figure(figsize=(12, 6))
sns.boxplot(data=cv_results['scores_df'], x='model', y='rmse')
plt.title('RMSE Distribution Across Folds')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize average RMSE by model
avg_scores = cv_results['scores_df'].groupby('model')['rmse'].mean().reset_index()
plt.figure(figsize=(10, 5))
ax = sns.barplot(data=avg_scores.sort_values('rmse'), x='model', y='rmse')
plt.title('Average RMSE by Model')
plt.xlabel('Model')
plt.ylabel('Average RMSE')

# Add value labels on bars
for i, v in enumerate(avg_scores.sort_values('rmse')['rmse']):
    ax.text(i, v + 0.001, f'{v:.6f}', ha='center')
    
plt.tight_layout()
plt.show()


# Create a comparison DataFrame of all methods
all_methods = []

# Add individual models
for name in models_config.keys():
    model_oof = np.expm1(cv_results['oof_preds'][name])
    model_rmsle = rmsle(y_true, model_oof)
    all_methods.append({
        'method': name, 
        'rmsle': model_rmsle,
        'type': 'individual'
    })

# Add ensemble methods
all_methods.append({'method': 'equal_weights_ensemble', 'rmsle': validation_rmsle, 'type': 'ensemble'})
all_methods.append({'method': 'weighted_ensemble', 'rmsle': weighted_validation_rmsle, 'type': 'ensemble'})

# Create DataFrame and display sorted by performance
methods_df = pd.DataFrame(all_methods).sort_values('rmsle')
display(methods_df)

# Visualize comparison
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=methods_df, x='method', y='rmsle', hue='type')
plt.title('RMSLE by Method')
plt.xlabel('Method')
plt.ylabel('RMSLE (lower is better)')
plt.xticks(rotation=45)

# Add value labels on bars
for i, v in enumerate(methods_df['rmsle']):
    ax.text(i, v + 0.001, f'{v:.6f}', ha='center')

plt.tight_layout()
plt.show()

# Visualize model contributions to the ensemble
plt.figure(figsize=(10, 5))
plt.bar(optimal_weights.keys(), optimal_weights.values())
plt.title('Model Weights in Optimized Ensemble')
plt.ylabel('Weight')
plt.ylim(0, max(optimal_weights.values()) * 1.1)

# Add value labels on bars
for i, (model, weight) in enumerate(optimal_weights.items()):
    plt.text(i, weight + 0.01, f'{weight:.4f}', ha='center')

plt.tight_layout()
plt.show()


# Load sample submission file
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Create weighted ensemble submission (best approach)
weighted_submission = submission.copy()
weighted_submission['Calories'] = weighted_ensemble_preds
weighted_submission.to_csv('submission.csv', index=False)
print('Weighted ensemble submission created!')

