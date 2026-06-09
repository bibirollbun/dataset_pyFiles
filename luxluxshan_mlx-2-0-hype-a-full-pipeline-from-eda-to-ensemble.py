# mlX 2.0 Music Popularity Prediction - Complete Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

# Load data
train_df = pd.read_csv('/kaggle/input/mlx-2-0-regression/train.csv')
test_df = pd.read_csv('/kaggle/input/mlx-2-0-regression/test.csv')
sample_submission = pd.read_csv('/kaggle/input/mlx-2-0-regression/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("\nFirst few rows:")
print(train_df.head())

# Basic info about the dataset
print("\nDataset Info:")
print(train_df.info())
print("\nTarget variable statistics:")
print(train_df['target'].describe())

# Check for missing values
print("\nMissing values in train:")
print(train_df.isnull().sum())
print("\nMissing values in test:")
print(test_df.isnull().sum())


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

# Target distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train_df['target'], bins=50, alpha=0.7)
plt.title('Target Distribution')
plt.xlabel('Popularity Score')

plt.subplot(1, 2, 2)
plt.boxplot(train_df['target'])
plt.title('Target Boxplot')
plt.ylabel('Popularity Score')
plt.tight_layout()
plt.show()

# Correlation with target
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_cols:
    numeric_cols.remove('target')
if 'id' in numeric_cols:
    numeric_cols.remove('id')

correlations = train_df[numeric_cols + ['target']].corr()['target'].sort_values(ascending=False)
print("\nTop correlations with target:")
print(correlations.head(10))
print("\nBottom correlations with target:")
print(correlations.tail(10))

# Correlation heatmap for top features
top_features = correlations.abs().head(15).index.tolist()
plt.figure(figsize=(12, 10))
sns.heatmap(train_df[top_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Top Features')
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

# Target distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train_df['target'], bins=50, alpha=0.7)
plt.title('Target Distribution')
plt.xlabel('Popularity Score')

plt.subplot(1, 2, 2)
plt.boxplot(train_df['target'])
plt.title('Target Boxplot')
plt.ylabel('Popularity Score')
plt.tight_layout()
plt.show()

# Correlation with target
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_cols:
    numeric_cols.remove('target')
if 'id' in numeric_cols:
    numeric_cols.remove('id')

correlations = train_df[numeric_cols + ['target']].corr()['target'].sort_values(ascending=False)
print("\nTop correlations with target:")
print(correlations.head(10))
print("\nBottom correlations with target:")
print(correlations.tail(10))

# Correlation heatmap for top features
top_features = correlations.abs().head(15).index.tolist()
plt.figure(figsize=(12, 10))
sns.heatmap(train_df[top_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Top Features')
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def safe_numeric_operation(df, cols, operation='mean'):
    """Safely perform numeric operations on columns, handling mixed data types"""
    if len(cols) == 0:
        return None
    
    # Ensure all columns are numeric
    numeric_df = pd.DataFrame()
    for col in cols:
        if col in df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if numeric_df.empty:
        return None
    
    if operation == 'mean':
        return numeric_df.mean(axis=1)
    elif operation == 'std':
        return numeric_df.std(axis=1).fillna(0)
    elif operation == 'max':
        return numeric_df.max(axis=1)
    elif operation == 'min':
        return numeric_df.min(axis=1)
    elif operation == 'sum':
        return numeric_df.sum(axis=1)
    else:
        return numeric_df.mean(axis=1)

# SAFER VERSION - SIMPLIFIED FEATURE ENGINEERING

def simple_feature_engineering(df):
    """Simplified feature engineering that's safer with mixed data types"""
    df = df.copy()
    
    print("Starting simple feature engineering...")
    print(f"Initial shape: {df.shape}")
    
    # Store original columns
    original_cols = df.columns.tolist()
    
    # ========== SAFE DATA TYPE HANDLING ==========
    # Identify definitely categorical columns to keep as-is
    categorical_cols = ['id', 'track_identifier', 'creator_collective']
    
    # Convert numeric-looking columns safely
    for col in df.columns:
        if col not in categorical_cols and col != 'target':
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if not converted.isna().all():  # If at least some values converted
                        df[col] = converted.fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
                        print(f"Converted {col} to numeric")
                except:
                    print(f"Keeping {col} as categorical")
    
    # ========== SIMPLE AGGREGATIONS ==========
    # Group similar features and create simple aggregations
    feature_groups = {
        'emotional_charge': [col for col in df.columns if 'emotional_charge' in col],
        'emotional_resonance': [col for col in df.columns if 'emotional_resonance' in col],
        'groove_efficiency': [col for col in df.columns if 'groove_efficiency' in col],
        'beat_frequency': [col for col in df.columns if 'beat_frequency' in col],
        'rhythmic_cohesion': [col for col in df.columns if 'rhythmic_cohesion' in col],
        'duration_ms': [col for col in df.columns if 'duration_ms' in col],
        'tonal_mode': [col for col in df.columns if 'tonal_mode' in col],
        'harmonic_scale': [col for col in df.columns if 'harmonic_scale' in col],
        'organic_texture': [col for col in df.columns if 'organic_texture' in col],
        'organic_immersion': [col for col in df.columns if 'organic_immersion' in col],
        'instrumental_density': [col for col in df.columns if 'instrumental_density' in col],
        'vocal_presence': [col for col in df.columns if 'vocal_presence' in col],
        'performance_authenticity': [col for col in df.columns if 'performance_authenticity' in col],
        'intensity_index': [col for col in df.columns if 'intensity_index' in col],
        'composition_label': [col for col in df.columns if 'composition_label' in col],
        'time_signature': [col for col in df.columns if 'time_signature' in col]
    }
    
    # Create aggregated features safely
    for group_name, cols in feature_groups.items():
        if len(cols) >= 2:
            # Ensure all columns in group are numeric
            numeric_cols = []
            for col in cols:
                if col in df.columns:
                    try:
                        # Convert to numeric if not already
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        numeric_cols.append(col)
                    except:
                        continue
            
            if len(numeric_cols) >= 2:
                try:
                    # Simple aggregations
                    df[f'{group_name}_avg'] = df[numeric_cols].mean(axis=1)
                    df[f'{group_name}_max'] = df[numeric_cols].max(axis=1)
                    df[f'{group_name}_sum'] = df[numeric_cols].sum(axis=1)
                    
                    # Add standard deviation if possible
                    std_vals = df[numeric_cols].std(axis=1)
                    df[f'{group_name}_std'] = std_vals.fillna(0)
                    
                    print(f"Created aggregations for {group_name}")
                except Exception as e:
                    print(f"Warning: Could not create aggregations for {group_name}: {e}")
    
    # ========== SIMPLE METADATA FEATURES ==========
    # Album features
    if 'album_component_count' in df.columns:
        try:
            df['is_single'] = (df['album_component_count'] == 1).astype(int)
            df['is_album'] = (df['album_component_count'] > 6).astype(int)
            print("Created album features")
        except:
            print("Warning: Could not create album features")
    
    # Artist features
    if 'artist_count' in df.columns:
        try:
            df['is_solo'] = (df['artist_count'] == 1).astype(int)
            df['is_collaboration'] = (df['artist_count'] > 1).astype(int)
            print("Created artist features")
        except:
            print("Warning: Could not create artist features")
    
    # Weekday features
    if 'weekday_of_release' in df.columns:
        try:
            df['weekday_of_release'] = pd.to_numeric(df['weekday_of_release'], errors='coerce').fillna(0)
            df['is_weekend'] = (df['weekday_of_release'].isin([5, 6])).astype(int)
            df['is_friday'] = (df['weekday_of_release'] == 4).astype(int)
            print("Created weekday features")
        except:
            print("Warning: Could not create weekday features")
    
    # ========== SIMPLE RATIOS ==========
    # Create a few safe interaction features
    try:
        if 'emotional_charge_avg' in df.columns and 'emotional_resonance_avg' in df.columns:
            df['emotional_interaction'] = df['emotional_charge_avg'] * df['emotional_resonance_avg']
            
        if 'groove_efficiency_avg' in df.columns and 'beat_frequency_avg' in df.columns:
            df['rhythm_interaction'] = df['groove_efficiency_avg'] * df['beat_frequency_avg']
            
        if 'vocal_presence_avg' in df.columns and 'instrumental_density_avg' in df.columns:
            df['vocal_instrumental_ratio'] = df['vocal_presence_avg'] / (df['instrumental_density_avg'] + 0.001)
            
        print("Created interaction features")
    except Exception as e:
        print(f"Warning: Could not create interaction features: {e}")
    
    print(f"Final shape: {df.shape}")
    print(f"Added {df.shape[1] - len(original_cols)} new features")
    
    return df

# Replace the feature engineering call
print("Using simplified feature engineering...")
train_engineered = simple_feature_engineering(train_df)
test_engineered = simple_feature_engineering(test_df)

print("Features after engineering:", train_engineered.shape[1])


# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================

def preprocess_data(train, test, target_col='target'):
    # Separate features and target
    if target_col in train.columns:
        X_train = train.drop([target_col, 'id'], axis=1, errors='ignore')
        y_train = train[target_col]
    else:
        X_train = train.drop(['id'], axis=1, errors='ignore')
        y_train = None
    
    X_test = test.drop(['id'], axis=1, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        print(f"Processing categorical column: {col}")
        
        # Target encoding for high cardinality, one-hot for low cardinality
        if X_train[col].nunique() > 10 and y_train is not None:
            # Target encoding with regularization
            overall_mean = y_train.mean()
            
            # Calculate means for each category
            category_means = train.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
            category_means.columns = [col, 'mean', 'count']
            
            # Regularization: blend with overall mean based on count
            min_samples = 5
            alpha = 10
            category_means['regularized_mean'] = (
                (category_means['count'] * category_means['mean'] + alpha * overall_mean) / 
                (category_means['count'] + alpha)
            )
            
            # Create mapping
            encoding_map = dict(zip(category_means[col], category_means['regularized_mean']))
            
            X_train[col + '_encoded'] = X_train[col].map(encoding_map).fillna(overall_mean)
            X_test[col + '_encoded'] = X_test[col].map(encoding_map).fillna(overall_mean)
            
        else:
            # One-hot encoding for low cardinality
            X_train_dummies = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
            X_test_dummies = pd.get_dummies(X_test[col], prefix=col, drop_first=True)
            
            # Align columns between train and test
            all_cols = set(X_train_dummies.columns) | set(X_test_dummies.columns)
            for dummy_col in all_cols:
                if dummy_col not in X_train_dummies.columns:
                    X_train_dummies[dummy_col] = 0
                if dummy_col not in X_test_dummies.columns:
                    X_test_dummies[dummy_col] = 0
            
            # Reorder columns to match
            X_train_dummies = X_train_dummies[sorted(all_cols)]
            X_test_dummies = X_test_dummies[sorted(all_cols)]
            
            X_train = pd.concat([X_train, X_train_dummies], axis=1)
            X_test = pd.concat([X_test, X_test_dummies], axis=1)
        
        # Drop original categorical column
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
    
    # Handle missing values
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    # Fill missing values with median for numerical columns
    for col in numeric_cols:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # Remove any infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ensure test set has same columns as train set
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    
    extra_cols = set(X_test.columns) - set(X_train.columns)
    for col in extra_cols:
        X_test = X_test.drop(col, axis=1)
    
    # Reorder test columns to match train
    X_test = X_test[X_train.columns]
    
    print(f"Final feature count: {len(X_train.columns)}")
    print(f"Features with high variance: {(X_train.var() > 0.01).sum()}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, scaler

# Preprocess data
X_train, X_test, y_train, scaler = preprocess_data(train_engineered, test_engineered)

print("Final feature shape - Train:", X_train.shape)
print("Final feature shape - Test:", X_test.shape)


# ============================================================================
# STEP 5: MODEL SELECTION AND TRAINING
# ============================================================================

# Split data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=None
)

# Define models
models = {
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
    'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse')
}

# Train and evaluate models
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Fit model
    model.fit(X_train_split, y_train_split)
    
    # Predict on validation set
    val_pred = model.predict(X_val_split)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
    results[name] = rmse
    
    print(f"{name} Validation RMSE: {rmse:.4f}")
    
    # Store predictions for ensemble
    predictions[name] = model.predict(X_test)

# Display results
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:15}: {rmse:.4f}")


# ============================================================================
# STEP 6: HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================

# Find best model
best_model_name = min(results.keys(), key=lambda x: results[x])
print(f"\nBest model: {best_model_name}")

# Hyperparameter tuning for best models (top 2)
top_models = sorted(results.items(), key=lambda x: x[1])[:2]

tuned_models = {}
for model_name, _ in top_models:
    print(f"\nTuning {model_name}...")
    
    if model_name == 'LightGBM':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
        }
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
        }
        model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        
    elif model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [100, 200],
        }
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    else:
        continue
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_split, y_train_split)
    
    # Best model
    best_model = grid_search.best_estimator_
    tuned_models[model_name] = best_model
    
    # Validation score
    val_pred = best_model.predict(X_val_split)
    rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Tuned {model_name} Validation RMSE: {rmse:.4f}")


# ============================================================================
# STEP 7: ENSEMBLE METHODS
# ============================================================================

# Simple ensemble - average of top models
ensemble_pred = np.zeros(len(X_test))
ensemble_weights = []

# Use top 3 models for ensemble
top_3_models = sorted(results.items(), key=lambda x: x[1])[:3]

for model_name, rmse in top_3_models:
    weight = 1 / rmse  # Inverse of RMSE as weight
    ensemble_weights.append(weight)
    
    if model_name in tuned_models:
        pred = tuned_models[model_name].predict(X_test)
    else:
        pred = predictions[model_name]
    
    ensemble_pred += weight * pred

# Normalize weights
ensemble_pred /= sum(ensemble_weights)

print(f"\nEnsemble weights: {dict(zip([name for name, _ in top_3_models], ensemble_weights))}")


# ============================================================================
# STEP 8: FINAL PREDICTIONS AND SUBMISSION
# ============================================================================

# Train final model on full training data
final_model_name = min(results.keys(), key=lambda x: results[x])

if final_model_name in tuned_models:
    final_model = tuned_models[final_model_name]
else:
    final_model = models[final_model_name]

# Retrain on full data
final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)

# Create submissions
submissions = {
    'single_model': final_predictions,
    'ensemble': ensemble_pred
}

for method, preds in submissions.items():
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': preds
    })
    
    # Ensure predictions are within valid range
    submission['target'] = submission['target'].clip(0, 100)
    
    submission.to_csv(f'submission_{method}.csv', index=False)
    print(f"\n{method.title()} submission saved!")
    print(f"Prediction range: {preds.min():.2f} to {preds.max():.2f}")
    print(f"Mean prediction: {preds.mean():.2f}")

print("\n" + "="*50)
print("FEATURE IMPORTANCE (from best model)")
print("="*50)

if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title(f'Top 15 Features - {final_model_name}')
    plt.tight_layout()
    plt.show()

print("\nProcess completed! Check submission files.")

