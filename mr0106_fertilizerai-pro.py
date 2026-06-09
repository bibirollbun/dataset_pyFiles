# -*- coding: utf-8 -*-
"""
Optimal Fertilizer Recommendation System - Grandmaster Edition
Competition: Kaggle Playground Series S5E6 (Season 5, Episode 6)
Technique: Advanced Ensemble with Optimized Feature Engineering
Target Metric: MAP@3 (Mean Average Precision at 3)
"""

# =============================================
# INITIAL SETUP & CONFIGURATION
# =============================================
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
from time import time
from tqdm.notebook import tqdm
from sklearn.base import clone

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.5f}'.format)
np.random.seed(42)

# Model and Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
import optuna
from optuna.samplers import TPESampler

# Visualization
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300

# GPU Configuration
USE_GPU = False
try:
    from numba import cuda
    USE_GPU = cuda.is_available()
except:
    pass


# =============================================
# DATA LOADING & OPTIMIZATION
# =============================================
print("\nğŸ”� [1/4] Loading and optimizing datasets...")

# Memory-efficient data loading
dtypes = {
    'Temperature': 'float32', 'Humidity': 'float32', 
    'Moisture': 'float32', 'Nitrogen': 'float32',
    'Potassium': 'float32', 'Phosphorous': 'float32',
    'Soil Type': 'category', 'Crop Type': 'category'
}

def load_data():
    try:
        train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", dtype=dtypes)
        test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", dtype=dtypes)
        submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
        
        # Fix column names
        train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
        test.rename(columns={'Temparature': 'Temperature'}, inplace=True)

        # Try to load original data if available
        try:
            original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
            original.rename(columns={'Temparature': 'Temperature'}, inplace=True)
            train = pd.concat([train, original], axis=0, ignore_index=True)
            print("âœ… Original data successfully loaded and merged")
        except FileNotFoundError:
            print("âš ï¸� Original data not found, using only competition data")
        
        return train, test, submission
    
    except Exception as e:
        print(f"â�Œ Error loading data: {str(e)}")
        raise

train, test, submission = load_data()

# Advanced memory optimization
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type.name == 'category':
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage reduced to {end_mem:.2f} MB (decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# =============================================
# ADVANCED FEATURE ENGINEERING
# =============================================
print("\nğŸ”§ [2/4] Performing advanced feature engineering...")

def create_features(df):
    """Create domain-specific agricultural features with interactions"""
    
    # Nutrient ratios with Laplace smoothing
    df['N/P_ratio'] = (df['Nitrogen'] + 1) / (df['Phosphorous'] + 1)
    df['N/K_ratio'] = (df['Nitrogen'] + 1) / (df['Potassium'] + 1)
    df['P/K_ratio'] = (df['Phosphorous'] + 1) / (df['Potassium'] + 1)
    df['NP_balance'] = df['Nitrogen'] - df['Phosphorous']
    
    # Environmental interaction features
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
    df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
    df['Environmental_Stress'] = ((df['Temperature'] > 30) & (df['Humidity'] < 40)).astype(int)
    
    # Nutrient aggregates and scores
    nutrients = ['Nitrogen', 'Phosphorous', 'Potassium']
    df['Nutrient_Sum'] = df[nutrients].sum(axis=1)
    df['Nutrient_Imbalance'] = df['Nitrogen'] - df['Phosphorous'] - df['Potassium']
    df['NPK_Score'] = df['Nitrogen']*0.5 + df['Phosphorous']*0.3 + df['Potassium']*0.2
    df['Nutrient_Variance'] = df[nutrients].var(axis=1)
    
    # Soil-Crop interactions
    df['Soil_Crop_Combo'] = df['Soil Type'].astype(str) + "_" + df['Crop Type'].astype(str)
    
    # Growing conditions
    df['Growing_Condition'] = np.where(
        (df['Temperature'] > 25) & (df['Humidity'] > 60),
        'Optimal', 
        'Suboptimal'
    )
    
    return df

# Apply feature engineering
try:
    train = create_features(train)
    test = create_features(test)
    print("âœ… Feature engineering completed successfully")
    print(f"Total features after engineering: {train.shape[1]}")
except Exception as e:
    print(f"â�Œ Error in feature engineering: {str(e)}")
    raise


# =============================================
# DATA ENCODING & PREPARATION
# =============================================
print("\nğŸ”  [3/4] Encoding and preparing data...")

# Target encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# Ensure categorical types
cat_cols = ['Soil Type', 'Crop Type', 'Soil_Crop_Combo', 'Growing_Condition']
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Target encoding for high-cardinality features
for col in ['Soil_Crop_Combo']:
    te = TargetEncoder()
    train[f'{col}_target'] = te.fit_transform(train[col], train['Fertilizer Name'])
    test[f'{col}_target'] = te.transform(test[col])

# Frequency encoding
for col in cat_cols:
    freq_encoding = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq_encoding)
    test[f'{col}_freq'] = test[col].map(freq_encoding)

# Ordinal encoding
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = ordinal_encoder.fit_transform(train[cat_cols])
test[cat_cols] = ordinal_encoder.transform(test[cat_cols])

# Prepare final datasets
drop_cols = ['id', 'Fertilizer Name']
X = train.drop(drop_cols, axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

print(f"\nFinal training shape: {X.shape}")
print(f"Features used:\n{list(X.columns)}")


# =============================================
# MODEL TRAINING & ENSEMBLING
# =============================================
print("\nğŸ�‹ï¸� [4/4] Training optimized ensemble models...")

# ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù�Ø¦ÙˆÙŠØ© Ø¥Ù„Ù‰ Ø±Ù‚Ù…ÙŠØ© Ù‚Ø¨Ù„ Ø§Ø³ØªØ®Ø¯Ø§Ù… XGBoost
cat_cols_freq = [col for col in X.columns if '_freq' in col]
for col in cat_cols_freq:
    X[col] = X[col].astype('float32')
    X_test[col] = X_test[col].astype('float32')

# Enhanced MAP@3 metric calculation
def mapk(actual, predicted, k=3):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted)[:,:k]
    score = 0.0
    for i in range(len(actual)):
        if actual[i] in predicted[i]:
            rank = np.where(predicted[i] == actual[i])[0][0]
            score += 1.0 / (rank + 1)
    return score / len(actual)

# Cross-validation setup
N_SPLITS = 7
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(le.classes_)))
test_preds = np.zeros((X_test.shape[0], len(le.classes_)))
oof_true = []
oof_top3_preds = []

fold_loglosses = []
fold_map3s = []

# Optimized model configurations
models = {
    'xgb': XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.4,
        objective='multi:softprob',
        num_class=len(le.classes_),
        tree_method='gpu_hist' if USE_GPU else 'hist',
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
        enable_categorical=False  # ØªÙ… ØªØ¹Ø·ÙŠÙ„ Ø§Ù„Ø¯Ø¹Ù… Ø§Ù„Ù�Ø¦ÙˆÙŠ Ù„ØªÙ�Ø§Ø¯ÙŠ Ø§Ù„Ù…Ø´Ø§ÙƒÙ„
    ),
    'lgb': LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        colsample_bytree=0.8,
        subsample=0.8,
        objective='multiclass',
        n_jobs=-1,
        random_state=42
    ),
    'cat': CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function='MultiClass',
        random_seed=42,
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=0
    ),
    'rf': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
}

# Feature importance storage
feature_importances = pd.DataFrame(index=X.columns)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*40}")
    print(f"Fold {fold}/{N_SPLITS}")
    print(f"{'='*40}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    fold_preds = []
    weights = []
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        start_time = time()
        
        current_model = clone(model)
        
        try:
            if name == 'xgb':
                current_model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=100
                )
            elif name == 'cat':
                current_model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=0
                )
            elif name == 'lgb':
                current_model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[
                        lgb.early_stopping(100, verbose=0),
                        lgb.log_evaluation(100)
                    ]
                )
            else:  # RandomForest
                current_model.fit(X_train, y_train)
            
            # Get validation predictions
            val_preds = current_model.predict_proba(X_valid)
            fold_preds.append(val_preds)
            
            # Calculate model weight based on fold performance
            fold_score = mapk(y_valid, np.argsort(-val_preds, axis=1)[:, :3])
            weights.append(fold_score)
            print(f"{name.upper()} Fold {fold} MAP@3: {fold_score:.5f}")
            
            # Store feature importances
            if hasattr(current_model, 'feature_importances_'):
                feature_importances[f'fold_{fold}_{name}'] = current_model.feature_importances_
            
            # Generate test predictions (weighted by fold MAP@3)
            X_test_aligned = X_test[X_train.columns]
            test_preds += current_model.predict_proba(X_test_aligned) * fold_score
            
            print(f"Training completed in {time()-start_time:.2f} seconds")
        
        except Exception as e:
            print(f"â�Œ Error training {name}: {str(e)}")
            continue
    
    # Weighted ensemble predictions
    if len(fold_preds) > 0:
        weights = np.array(weights) / sum(weights)
        val_preds_ensemble = np.average(fold_preds, axis=0, weights=weights)
        oof_preds[valid_idx] = val_preds_ensemble
        
        # Calculate metrics
        fold_logloss = log_loss(y_valid, val_preds_ensemble)
        fold_loglosses.append(fold_logloss)
        
        top3_preds = np.argsort(-val_preds_ensemble, axis=1)[:, :3]
        fold_map3 = mapk(y_valid, top3_preds, k=3)
        fold_map3s.append(fold_map3)
        
        print(f"\nFold {fold} Metrics:")
        print(f"- Log Loss: {fold_logloss:.5f}")
        print(f"- MAP@3: {fold_map3:.5f}")
        
        oof_true.extend(y_valid)
        oof_top3_preds.extend(top3_preds)

# Final weighted test predictions
if len(fold_preds) > 0:
    test_preds /= sum(weights) * N_SPLITS

    # Overall metrics
    map3_score = mapk(oof_true, oof_top3_preds, k=3)
    mean_logloss = np.mean(fold_loglosses)
    std_logloss = np.std(fold_loglosses)

    print(f"\n{'='*40}")
    print("Cross-Validation Results:")
    print(f"- Mean MAP@3: {map3_score:.5f}")
    print(f"- Mean Log Loss: {mean_logloss:.5f} (Â±{std_logloss:.5f})")
    print(f"{'='*40}")
else:
    print("\nâ�Œ All models failed during training")


# =============================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\nğŸ“Š Analyzing feature importance...")

def analyze_feature_importance(feature_importances, X, n_splits=7, top_n=20):
    """
    Analyze and visualize feature importance with robust error handling
    
    Parameters:
    - feature_importances: DataFrame containing raw importance scores
    - X: Original feature DataFrame (for column reference)
    - n_splits: Number of CV splits used
    - top_n: Number of top features to display
    """
    
    # 1. Data Validation
    if feature_importances.empty:
        print("âš ï¸� No feature importance data available")
        return
    
    # 2. Normalize importance scores within each model/fold
    for fold in range(1, n_splits+1):
        for model_type in ['xgb', 'lgb', 'rf']:
            col_name = f'fold_{fold}_{model_type}'
            if col_name in feature_importances.columns:
                # Normalize to [0,1] range per model/fold
                col = feature_importances[col_name]
                feature_importances[col_name] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    # 3. Calculate weighted average importance
    model_weights = {'xgb': 0.4, 'lgb': 0.4, 'rf': 0.2}  # Adjust based on model performance
    for model_type, weight in model_weights.items():
        model_cols = [c for c in feature_importances.columns if f'mean_{model_type}' in c]
        if model_cols:
            feature_importances[f'weighted_{model_type}'] = feature_importances[model_cols].mean(axis=1) * weight
    
    if 'mean_importance' in feature_importances.columns:
        feature_importances.drop('mean_importance', axis=1, inplace=True)
    
    weighted_cols = [c for c in feature_importances.columns if 'weighted_' in c]
    if weighted_cols:
        feature_importances['mean_importance'] = feature_importances[weighted_cols].sum(axis=1)
    else:
        feature_importances['mean_importance'] = feature_importances[[c for c in feature_importances.columns if 'mean_' in c]].mean(axis=1)

    # 4. Prepare visualization data
    try:
        top_features = (feature_importances['mean_importance']
                       .sort_values(ascending=False)
                       .head(top_n)
                       .reset_index())
        top_features.columns = ['Feature', 'Importance']
        
        # 5. Create visualization
        plt.figure(figsize=(12, max(6, top_n//3)))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=top_features,
            palette='viridis',
            orient='h'
        )
        plt.title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.xlabel('Normalized Importance Score')
        plt.ylabel('')
        plt.tight_layout()
        
        # Save high-quality version
        plt.savefig(
            'feature_importance.png',
            dpi=300,
            bbox_inches='tight',
            transparent=True
        )
        plt.show()
        
        # 6. Print top features
        print(f"\nTop {top_n} Features:")
        print(top_features.to_string(index=False))
        
    except Exception as e:
        print(f"âš ï¸� Error creating visualization: {str(e)}")
        if 'mean_importance' in feature_importances.columns:
            print("\nTop features (raw output):")
            print(feature_importances['mean_importance'].sort_values(ascending=False).head(top_n))

# Execute analysis
analyze_feature_importance(feature_importances, X, n_splits=N_SPLITS, top_n=20)


# =============================================
# SUBMISSION GENERATION
# =============================================
print("\nğŸ�¯ Generating competition submission...")

def create_submission(preds, le, submission_df):
    """
    Create competition submission file with comprehensive validation
    and proper type handling.
    
    Parameters:
    - preds: numpy array of predicted probabilities
    - le: LabelEncoder used for target variable
    - submission_df: DataFrame with 'id' column
    
    Returns:
    - submission_df: DataFrame ready for submission
    """
    try:
        # 1. Get top 3 predictions
        top3 = np.argsort(-preds, axis=1)[:, :3]
        
        # 2. Convert numeric labels to original fertilizer names
        labels = le.inverse_transform(top3.ravel()).reshape(top3.shape)
        
        # 3. Ensure proper string type
        if labels.dtype.kind not in ['U', 'S', 'O']:  # Unicode, String, Object
            labels = labels.astype(str)
        
        # 4. Validate shapes match
        if labels.shape[0] != submission_df.shape[0]:
            raise ValueError(f"Shape mismatch: {labels.shape[0]} predictions vs {submission_df.shape[0]} submission rows")
        
        # 5. Create space-separated strings
        submission_df['Fertilizer Name'] = [' '.join(row) for row in labels]
        
        # 6. Validate no None/NaN values
        if submission_df['Fertilizer Name'].isnull().any():
            raise ValueError("Submission contains null values in predictions")
            
        # 7. Save to CSV
        submission_df.to_csv('submission.csv', index=False)
        
        return submission_df
    
    except Exception as e:
        print(f"â�Œ Error creating submission: {str(e)}")
        # Fallback: Save raw predictions if formatting fails
        submission_df['Fertilizer Name'] = [' '.join(map(str, row)) for row in top3]
        submission_df.to_csv('submission_fallback.csv', index=False)
        print("âš ï¸� Created fallback submission file with raw predictions")
        return submission_df

# Generate submission with validation
submission = create_submission(test_preds, le, submission.copy())

if submission is not None:
    print("âœ… Submission file created successfully!")
    print(f"\nSample submission:\n{submission.head(3)}")
    print(f"\nUnique predictions: {submission['Fertilizer Name'].nunique()}")
    print("\nğŸ�† Model is ready to win the competition!")
else:
    print("â�Œ Failed to create submission file")

# Clean up memory
del train, test, X, y, X_test
gc.collect()

