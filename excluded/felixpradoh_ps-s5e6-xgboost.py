# Core data manipulation and analysis
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning - Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# Model persistence and metadata
import joblib
import json
from datetime import datetime

# Utilities
import time
from collections import Counter

# Configuration
np.random.seed(513)



def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  # only the first correct prediction counts
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Define file paths
data_path = '/kaggle/input/playground-series-s5e6/'
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')
sample_submission_path = os.path.join(data_path, 'sample_submission.csv')

# Load datasets
print("ğŸ“‚ Loading datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# print("âœ… Data loaded successfully:")
# print(f"  â€¢ Training set: {train_df.shape}")
# print(f"  â€¢ Test set: {test_df.shape}")
# print(f"  â€¢ Sample submission: {sample_submission.shape}")

# Separate features and target variable
target_column = 'Fertilizer Name'

# Split training data
X_raw = train_df.drop(columns=[target_column])
y_raw = train_df[target_column]
X_test_raw = test_df.copy()

print("âœ… Data separation completed:")
print(f"  â€¢ Training features: {X_raw.shape}")
print(f"  â€¢ Training target: {y_raw.shape}")
print(f"  â€¢ Test features: {X_test_raw.shape}")
print(f"  â€¢ Target classes: {y_raw.nunique()}")


def create_features(df):
    """
    Create engineered features based on agricultural domain knowledge
    
    Args:
        df: DataFrame with agricultural features
        
    Returns:
        DataFrame with additional engineered features
    """
    df_eng = df.copy()
    
    # NPK Ratios (crucial for agricultural decisions)
    df_eng['N_P_ratio'] = df_eng['Nitrogen'] / (df_eng['Phosphorous'] + 0.001)
    df_eng['N_K_ratio'] = df_eng['Nitrogen'] / (df_eng['Potassium'] + 0.001)
    df_eng['P_K_ratio'] = df_eng['Phosphorous'] / (df_eng['Potassium'] + 0.001)
    
    # Total NPK and NPK Balance
    df_eng['Total_NPK'] = df_eng['Nitrogen'] + df_eng['Phosphorous'] + df_eng['Potassium']
    npk_mean = df_eng[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)
    df_eng['NPK_Balance'] = df_eng[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1) / (npk_mean + 0.001)
    
    # Environmental indices
    df_eng['Temp_Hum_index'] = df_eng['Temparature'] * df_eng['Humidity'] / 100
    df_eng['Moist_Balance'] = df_eng['Moisture'] - df_eng['Humidity']
    df_eng['Environ_Stress'] = np.sqrt((df_eng['Temparature'] - 25)**2 + (df_eng['Humidity'] - 65)**2)
    df_eng['Temp_Moist_inter'] = df_eng['Temparature'] * df_eng['Moisture'] / 100
    
    # Dominant nutrient
    npk_cols = ['Nitrogen', 'Phosphorous', 'Potassium']
    df_eng['Dominant_NPK'] = df_eng[npk_cols].idxmax(axis=1)
    
    # Categorical binning
    df_eng['Temp_Cat'] = pd.cut(df_eng['Temparature'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['Hum_Cat'] = pd.cut(df_eng['Humidity'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['N_Level'] = pd.cut(df_eng['Nitrogen'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['K_Level'] = pd.cut(df_eng['Potassium'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['P_Level'] = pd.cut(df_eng['Phosphorous'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Soil-Crop interaction
    df_eng['Soil_Crop_Combo'] = df_eng['Soil Type'].astype(str) + '_' + df_eng['Crop Type'].astype(str)
    
    return df_eng

# Apply feature engineering
print("ğŸ”§ Applying feature engineering...")
X_train_featured = create_features(X_raw)
X_test_featured = create_features(X_test_raw)

# Display new feature names
original_features = set(X_raw.columns)
new_features = [col for col in X_train_featured.columns if col not in original_features]

print(f"âœ… Feature engineering completed: {X_raw.shape[1]} â†’ {X_train_featured.shape[1]} features (+{X_train_featured.shape[1] - X_raw.shape[1]})")


def encode_categorical_features(X_train, X_test, y_train):
    """
    Encode categorical features using LabelEncoder
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_train: Training target
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded, y_encoded, encoders_dict)
    """
    
    # Initialize encoders dictionary
    encoders = {}
    
    # Create copies to avoid modifying originals
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"ğŸ”¢ Encoding categorical features...")
    print(f"Categorical columns found: {categorical_cols}")
    
    # Encode categorical features
    for col in categorical_cols:
        print(f"  â€¢ Encoding: {col}")
        
        # Create encoder
        encoder = LabelEncoder()
        
        # Fit on combined training and test data to ensure consistency
        combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        encoder.fit(combined_values)
        
        # Transform both datasets
        X_train_enc[col] = encoder.transform(X_train[col].astype(str))
        X_test_enc[col] = encoder.transform(X_test[col].astype(str))
        
        # Store encoder
        encoders[col] = encoder
            
    # Encode target variable
    print(f"\nğŸ�¯ Encoding target variable: {target_column}")
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_train)
    encoders['target'] = target_encoder
    
    print(f"  â€¢ Target classes: {len(target_encoder.classes_)}")
    # print(f"  â€¢ Class mapping preview: {dict(zip(target_encoder.classes_[:5], range(5)))}")
    
    return X_train_enc, X_test_enc, y_encoded, encoders

# Apply encoding
X_train_encoded, X_test_encoded, y_encoded, label_encoders = encode_categorical_features(
    X_train_featured, X_test_featured, y_raw
)

print(f"\nâœ… Encoding completed:")
print(f"  â€¢ Training features: {X_train_encoded.shape}")
print(f"  â€¢ Test features: {X_test_encoded.shape}")
print(f"  â€¢ Encoded target: {y_encoded.shape}")
print(f"  â€¢ Encoders stored: {len(label_encoders)}")


# =============================================================================
# FEATURE SELECTION FOR THE MODEL
# =============================================================================

# Feature selection
features_to_use = [
    # Original features
    'Temparature',
    'Humidity', 
    'Moisture',
    'Nitrogen',
    'Potassium', 
    'Phosphorous',
    
    # Engineered features
    # 'N_P_ratio',
    # 'N_K_ratio',
    # 'P_K_ratio',
    # 'Total_NPK',
    # 'NPK_Balance',
    # 'Temp_Hum_index',
    # 'Moist_Balance',
    # 'Environ_Stress',
    # 'Temp_Moist_inter',
    # 'Temp_Cat',
    # 'Hum_Cat',
    # 'N_Level',
    # 'K_Level',
    # 'P_Level',
    
    # Combinations
    # 'Soil_Crop_Combo',
    # 'Dominant_NPK',
    
    # Categorical features
    'Soil Type',
    'Crop Type',
]

# Validate features
available_features = [f for f in features_to_use if f in X_train_encoded.columns]
missing_features = [f for f in features_to_use if f not in X_train_encoded.columns]

features_to_use = available_features

if missing_features:
    print(f"Missing features: {missing_features}")

print(f"âœ… Selected features ({len(features_to_use)}): {features_to_use}")

# Create final datasets
X_final = X_train_encoded[features_to_use].copy()
X_test_final = X_test_encoded[features_to_use].copy()

# print(f"Training: {X_final.shape}, Test: {X_test_final.shape}, Target: {y_encoded.shape}")


# =============================================================================
# STRATIFIED 10-FOLD CROSS-VALIDATION CONFIGURATION
# =============================================================================

# Cross-validation parameters
N_SPLITS = 10  # 10-fold cross-validation for robust evaluation
RANDOM_STATE = 42
SHUFFLE = True

# Initialize StratifiedKFold to maintain class distribution
skf = StratifiedKFold(
    n_splits=N_SPLITS, 
    shuffle=SHUFFLE, 
    random_state=RANDOM_STATE
)

print(f"ğŸ”„ CROSS-VALIDATION CONFIGURATION:")
print(f"  â€¢ Number of folds: {N_SPLITS}")
print(f"  â€¢ Strategy: Stratified (maintains class proportions)")
print(f"  â€¢ Shuffle: {SHUFFLE}")
print(f"  â€¢ Random state: {RANDOM_STATE}")

# Analyze class distribution for stratification
print(f"\nğŸ“Š Class distribution analysis:")
unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
print(f"  â€¢ Total classes: {len(unique_classes)}")
print(f"  â€¢ Total samples: {len(y_encoded)}")
print(f"  â€¢ Samples per fold: ~{len(y_encoded) // N_SPLITS}")

# Check minimum class size for stratification
min_class_count = min(class_counts)
print(f"  â€¢ Minimum class size: {min_class_count}")
if min_class_count < N_SPLITS:
    print(f"  âš ï¸� Warning: Smallest class has {min_class_count} samples, less than {N_SPLITS} folds")
    print(f"    Some folds may not contain all classes")
else:
    print(f"  âœ… All classes have sufficient samples for {N_SPLITS}-fold CV")

# Preview fold splits
print(f"\nğŸ”� Fold size preview:")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_final, y_encoded)):
    if fold_idx < 3:  # Show first 3 folds
        print(f"  Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
    elif fold_idx == 3:
        print("  ...")
    elif fold_idx == N_SPLITS - 1:  # Show last fold
        print(f"  Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        break


# =============================================================================
# XGBOOST HYPERPARAMETER CONFIGURATION
#
#ğŸ�¯ EXPERIMENTATION ENCOURAGED!
# These parameters provide a solid baseline, but feel free to experiment!
# Try different learning rates, depths, or regularization for better scores.
# =============================================================================

# Calculate class weights for imbalanced dataset
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

print("âš–ï¸� Class weight calculation:")
print(f"  â€¢ Balanced class weights computed for {len(class_weight_dict)} classes")
print(f"  â€¢ Weight range: {min(class_weights):.3f} - {max(class_weights):.3f}")

# XGBoost hyperparameters (optimized for multi-class classification)
xgb_params = {
    # Multi-class objective
    'objective': 'multi:softprob',
    'num_class': len(label_encoders['target'].classes_),
    'eval_metric': 'mlogloss',
    
    # Tree structure
    # 'max_depth': 5,
    # 'min_child_weight': 1,
    # 'subsample': 1,
    # 'colsample_bytree': 1,
    
    # Learning parameters
    # 'learning_rate': 0.05,
    # 'n_estimators': 3000,  # High number with early stopping
    
    # Regularization
    # 'reg_alpha': 1.0,  # L1 regularization
    # 'reg_lambda': 1.0,  # L2 regularization
    # 'gamma': 0.00,      # Minimum split loss
    # 'max_delta_step': 1,  # Maximum delta step for tree weights
    
    # Performance
    # 'random_state': RANDOM_STATE,
    # 'n_jobs': -1,
    # 'verbosity': 0,
    
    # 'device': 'cpu',
    # 'tree_method': 'hist', 

    # GPU acceleration (comment out if no GPU available)
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    
    # Early stopping will be handled separately
    # 'early_stopping_rounds': 300


    # 'num_class': len(np.unique(y)), 
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'max_bin': 128,
    'colsample_bytree': 0.3, 
    'colsample_bylevel': 1,  
    'colsample_bynode': 1,  
    # 'tree_method': 'hist',  
    'random_state': 42,
    'eval_metric': 'mlogloss',
    # 'device': "cuda",
    'enable_categorical':True,
    'n_estimators':10000,
    'early_stopping_rounds':50,
}

# Early stopping configuration
es = 300
eval_metric = 'mlogloss'

print(f"\nğŸš€ XGBOOST CONFIGURATION:")
print(f"  â€¢ Objective: {xgb_params['objective']}")
print(f"  â€¢ Number of classes: {xgb_params['num_class']}")
print(f"  â€¢ Max depth: {xgb_params['max_depth']}")
print(f"  â€¢ Learning rate: {xgb_params['learning_rate']}")
print(f"  â€¢ Max estimators: {xgb_params['n_estimators']}")
print(f"  â€¢ Early stopping: {xgb_params['early_stopping_rounds']} rounds")
print(f"  â€¢ Evaluation metric: {xgb_params['eval_metric']}")
# print(f"  â€¢ Regularization: L1={xgb_params['reg_alpha']}, L2={xgb_params['reg_lambda']}")
print(f"  â€¢ Tree method: {xgb_params.get('tree_method', 'hist')}")
print(f"  â€¢ GPU ID: {xgb_params.get('gpu_id', 'N/A')}")
print(f"  â€¢ Class balancing: Enabled")



# =============================================================================
# 10-FOLD CROSS-VALIDATION TRAINING
# =============================================================================

def train_xgboost_cv(X, y, features, cv_splitter, params):
    """
    Train XGBoost models using cross-validation
    
    Args:
        X: Feature matrix
        y: Target vector (encoded)
        features: List of feature names to use
        cv_splitter: Cross-validation splitter (StratifiedKFold)
        params: XGBoost parameters
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Dict with trained models, predictions, and metrics
    """
    
    # Initialize storage
    models = {}
    oof_predictions = np.zeros((len(X), params['num_class']))  # Out-of-fold predictions
    cv_scores = []
    feature_importance_list = []
    
    print(f"ğŸ�‹ï¸� Starting {N_SPLITS}-Fold Cross-Validation Training...")
    print(f"â�° Training started at: {time.strftime('%H:%M:%S')}")
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        
        fold_start_time = time.time()
        print(f"\nğŸ“� FOLD {fold_idx + 1}/{N_SPLITS}")
        print(f"  â€¢ Train samples: {len(train_idx)}")
        print(f"  â€¢ Validation samples: {len(val_idx)}")
        
        # Split data
        X_train_fold = X.iloc[train_idx][features]
        X_val_fold = X.iloc[val_idx][features]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Calculate sample weights for this fold
        fold_class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_fold),
            y=y_train_fold
        )
        fold_class_weight_dict = dict(zip(np.unique(y_train_fold), fold_class_weights))
        sample_weights = np.array([fold_class_weight_dict.get(label, 1.0) for label in y_train_fold])
        
        # Initialize model
        model = XGBClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train_fold, y_train_fold,
            sample_weight=sample_weights,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Predict validation set
        val_pred_proba = model.predict_proba(X_val_fold)
        val_pred_classes = model.predict(X_val_fold)
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] = val_pred_proba
        
        # Calculate fold metrics
        fold_accuracy = accuracy_score(y_val_fold, val_pred_classes)
        
        # Calculate MAP@3 for this fold and get top 3 predictions for each sample
        val_top3_indices = np.argsort(val_pred_proba, axis=1)[:, -3:][:, ::-1]
        
        # Convert to lists for mapk function
        actual_list = y_val_fold.tolist() if hasattr(y_val_fold, 'tolist') else list(y_val_fold)
        predicted_list = val_top3_indices.tolist()
        
        # Calculate MAP@3 using the correct format
        fold_map3 = mapk(actual_list, predicted_list, k=3)
        
        # Store results
        cv_scores.append({
            'fold': fold_idx + 1,
            'accuracy': fold_accuracy,
            'map3': fold_map3,
            'best_iteration': model.best_iteration,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'training_time': time.time() - fold_start_time
        })
        
        # Store model and feature importance
        models[f'fold_{fold_idx + 1}'] = model
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_,
                'fold': fold_idx + 1
            })
            feature_importance_list.append(importance_df)
        
        fold_time = time.time() - fold_start_time
        print(f"  âœ… Fold completed in {fold_time:.1f}s")
        print(f"  ğŸ“Š Accuracy: {fold_accuracy:.4f} | MAP@3: {fold_map3:.4f}")
        print(f"  ğŸ”„ Best iteration: {model.best_iteration}")
    
    # Calculate overall metrics
    oof_pred_classes = np.argmax(oof_predictions, axis=1)
    overall_accuracy = accuracy_score(y, oof_pred_classes)
    
    # Calculate overall MAP@3 and get top 3 predictions for each sample
    oof_top3_indices = np.argsort(oof_predictions, axis=1)[:, -3:][:, ::-1]
    
    # Convert to lists for mapk function
    actual_list = y.tolist() if hasattr(y, 'tolist') else list(y)
    predicted_list = oof_top3_indices.tolist()
    
    # Calculate MAP@3 using the correct format
    overall_map3 = mapk(actual_list, predicted_list, k=3)
    
    # Combine feature importance across folds
    if feature_importance_list:
        feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
        feature_importance_summary = feature_importance_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        feature_importance_summary = feature_importance_summary.sort_values('mean', ascending=False)
    else:
        feature_importance_summary = None
    
    return {
        'models': models,
        'oof_predictions': oof_predictions,
        'cv_scores': cv_scores,
        'overall_accuracy': overall_accuracy,
        'overall_map3': overall_map3,
        'feature_importance': feature_importance_summary
    }

# Execute cross-validation training
print("ğŸš€ Starting model training...")
start_time = time.time()

training_results = train_xgboost_cv(
    X=X_final,
    y=y_encoded,
    features=features_to_use,
    cv_splitter=skf,
    params=xgb_params,
)

total_time = time.time() - start_time

print(f"\nğŸ�‰ CROSS-VALIDATION TRAINING COMPLETED!")
print(f"â�° Training finished at: {time.strftime('%H:%M:%S')}")
print(f"â�±ï¸� Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"ğŸ“Š Overall Accuracy: {training_results['overall_accuracy']:.4f}")
print(f"ğŸ“Š Overall MAP@3: {training_results['overall_map3']:.4f}")


# =============================================================================
# CROSS-VALIDATION RESULTS EVALUATION
# =============================================================================

print("ğŸ“Š CROSS-VALIDATION RESULTS")
print("=" * 60)

# Extract results from training
cv_results_df = pd.DataFrame(training_results['cv_scores'])

# Calculate statistics
accuracy_mean = cv_results_df['accuracy'].mean()
accuracy_std = cv_results_df['accuracy'].std()
map3_mean = cv_results_df['map3'].mean()
map3_std = cv_results_df['map3'].std()

print(f"ğŸ�¯ FINAL METRICS:")
print(f"  ğŸ“ˆ Cross-Validation Accuracy: {accuracy_mean:.4f} Â± {accuracy_std:.4f}")
print(f"  ğŸ“ˆ Cross-Validation MAP@3:    {map3_mean:.4f} Â± {map3_std:.4f}")
print(f"  ğŸ“ˆ Out-of-Fold Accuracy:      {training_results['overall_accuracy']:.4f}")
print(f"  ğŸ“ˆ Out-of-Fold MAP@3:         {training_results['overall_map3']:.4f}")

# Stability evaluation
accuracy_cv = accuracy_std / accuracy_mean if accuracy_mean > 0 else 0
map3_cv = map3_std / map3_mean if map3_mean > 0 else 0

print(f"\nğŸ”� STABILITY ANALYSIS:")
print(f"  ğŸ“Š Coefficient of variation (Accuracy): {accuracy_cv:.3f}")
print(f"  ğŸ“Š Coefficient of variation (MAP@3):    {map3_cv:.3f}")
print(f"  {'âœ… Stable model' if accuracy_cv < 0.05 else 'âš ï¸� Variable model'} (Accuracy CV < 0.05)")
print(f"  {'âœ… Stable model' if map3_cv < 0.05 else 'âš ï¸� Variable model'} (MAP@3 CV < 0.05)")

# Training time analysis
avg_fold_time = cv_results_df['training_time'].mean()
print(f"\nâ�±ï¸� TRAINING TIMES:")
print(f"  ğŸ“Š Average time per fold: {avg_fold_time:.1f}s")
print(f"  ğŸ“Š Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

# Detailed results by fold
print(f"\nğŸ“‹ DETAILED RESULTS BY FOLD:")
print("Fold  Accuracy   MAP@3    Best_Iter  Time(s)")
print("-" * 50)
for _, row in cv_results_df.iterrows():
    print(f"{row['fold']:2.0f}    {row['accuracy']:.4f}   {row['map3']:.4f}     {row['best_iteration']:4.0f}   {row['training_time']:6.1f}")

print("-" * 50)
print(f"Mean  {accuracy_mean:.4f}   {map3_mean:.4f}     {cv_results_df['best_iteration'].mean():4.0f}   {avg_fold_time:6.1f}")

# Feature importance analysis
if training_results['feature_importance'] is not None:
    print(f"\nğŸ”� TOP 10 MOST IMPORTANT FEATURES:")
    print("Rank  Feature               Importance")
    print("-" * 40)
    for i, (_, row) in enumerate(training_results['feature_importance'].head(10).iterrows()):
        print(f"{i+1:2d}.   {row['feature']:20} {row['mean']:8.4f}")


# =============================================================================
# FILE SAVING CONFIGURATION
# =============================================================================

# Configure model name based on MAP@3
overall_map3 = training_results['overall_map3']
model_name = f"XGB_10CV_MAP@3-{overall_map3:.5f}".replace('.', '')
model_dir = f"/kaggle/working/XGB/{N_SPLITS}CV/{model_name}"

# Create directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

print(f"ğŸ“� MODEL DIRECTORY:")
print(f"  {model_dir}")

# File name configuration - KAGGLE COMPETITION RECOMMENDED AND ESSENTIALS ONLY
base_filename = model_name
files_to_create = {
    'hparams': f"{base_filename}_hparams.json",                 # âœ… RECOMMENDED - Hyperparameters for reproducibility
    'metrics': f"{base_filename}_metrics.json",                 # âœ… RECOMMENDED - Performance metrics and config
    'submission': f"{base_filename}_submission.csv",            # âœ… ESSENTIAL - Competition submission file
    'submission_info': f"{base_filename}_submission_info.json"  # âœ… RECOMMENDED - Submission metadata
}

print(f"\nğŸ“� FILES TO CREATE:")
for file_type, filename in files_to_create.items():
    print(f"  {file_type:15}: {filename}")




# =============================================================================
# SAVE HYPERPARAMETERS AND METRICS - COMPETITION
# =============================================================================

# Extract metrics from training results
cv_results_df = pd.DataFrame(training_results['cv_scores'])
map3_mean = cv_results_df['map3'].mean()
map3_std = cv_results_df['map3'].std()
accuracy_mean = cv_results_df['accuracy'].mean()
accuracy_std = cv_results_df['accuracy'].std()

# 1. HYPERPARAMETERS DATA
hparams_data = {
    "model_type": "XGBClassifier",
    "model_abbreviation": "XGB",
    "cv_strategy": f"{N_SPLITS}-Fold Stratified Cross Validation",
    "optimization_method": "Manual hyperparameter tuning",
    "ensemble_method": "Average of fold predictions",
    
    # Fixed hyperparameters used
    "hyperparameters": xgb_params,
    
    # General configuration
    "features_selected": features_to_use,
    "num_features": len(features_to_use),
    "class_weights_used": True,
    "random_state": RANDOM_STATE,
    "cv_splits": N_SPLITS,
    "total_models": len(training_results['models']),
    "early_stopping_rounds": es
}

# 2. METRICS DATA  
metrics_data = {
    "model_type": "XGBClassifier",
    "model_abbreviation": "XGB",
    "tier": "10_FOLD_CV",
    "target_variable": "Fertilizer Name",
    "cv_strategy": f"{N_SPLITS}-Fold Stratified Cross Validation",
    "optimization_method": "Manual hyperparameter tuning",
    
    # Main performance metrics
    "map3_score_cv_mean": float(map3_mean),
    "map3_score_cv_std": float(map3_std),
    "map3_score_oof": float(training_results['overall_map3']),
    "accuracy_cv_mean": float(accuracy_mean),
    "accuracy_cv_std": float(accuracy_std),
    "accuracy_oof": float(training_results['overall_accuracy']),
    
    # Model configuration
    "num_classes": len(label_encoders['target'].classes_),
    "features_used": len(features_to_use),
    "features_list": features_to_use,
    "cv_folds": N_SPLITS,
    "total_models_trained": len(training_results['models']),
    
    # Detailed fold results
    "fold_results": training_results['cv_scores'],
    
    # Stability statistics
    "accuracy_cv_coefficient": float(accuracy_std / accuracy_mean) if accuracy_mean > 0 else 0.0,
    "map3_cv_coefficient": float(map3_std / map3_mean) if map3_mean > 0 else 0.0,
    
    # Training performance
    "training_time_total": float(total_time),
    "training_time_per_fold_avg": float(cv_results_df['training_time'].mean()),
    
    # Hyperparameters used
    "hyperparameters": xgb_params,
    
    # Feature importance summary (lightweight)
    "top_features": training_results['feature_importance'].head(10).to_dict('records') if training_results['feature_importance'] is not None else None,
    
    # Metadata
    "timestamp": datetime.now().isoformat(),
    "kaggle_competition": "playground-series-s5e6",
    "ensemble_method": "Average of 10-fold CV models",
    "models_saved": False,  # Models used in-memory only for predictions
    "memory_optimized": True
}

# Save both files
hparams_file = os.path.join(model_dir, files_to_create['hparams'])
metrics_file = os.path.join(model_dir, files_to_create['metrics'])

with open(hparams_file, 'w') as f:
    json.dump(hparams_data, f, indent=2)
    
with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2)

print(f"âœ… Competition files saved:")
print(f"  ğŸ“„ Hyperparameters: {files_to_create['hparams']}")
print(f"  ğŸ“Š Metrics: {files_to_create['metrics']}")
print(f"  ğŸ’¾ Size: Lightweight (~20-35 KB total)")
print(f"  ğŸš€ Competition ready!")


# =============================================================================
# GENERATE TEST PREDICTIONS AND CREATE KAGGLE SUBMISSION
# =============================================================================

print(f"ğŸ”® Generating test predictions using {len(training_results['models'])}-model ensemble...")

# Generate ensemble predictions using all trained models
test_predictions_all = []
for fold_name, model in training_results['models'].items():
    pred_proba = model.predict_proba(X_test_final)
    test_predictions_all.append(pred_proba)

# Average predictions across all folds
test_predictions_ensemble = np.mean(test_predictions_all, axis=0)

# Get top 3 predictions for each sample (MAP@3 format)
test_top3_indices = np.argsort(test_predictions_ensemble, axis=1)[:, -3:][:, ::-1]

# Convert prediction indices to fertilizer names
test_top3_names = []
for i in range(len(test_top3_indices)):
    top3_for_sample = []
    for j in range(3):
        class_idx = test_top3_indices[i, j]
        class_name = label_encoders['target'].inverse_transform([class_idx])[0]
        top3_for_sample.append(class_name)
    test_top3_names.append(top3_for_sample)

# Create submission format (space-separated top 3 fertilizers)
submission_predictions = [' '.join(top3_names) for top3_names in test_top3_names]

# Create final submission DataFrame
submission = pd.DataFrame({
    'id': sample_submission['id'].copy(),
    'Fertilizer Name': submission_predictions
})

# Save submission file
submission_file = os.path.join(model_dir, files_to_create['submission'])
submission.to_csv(submission_file, index=False)

# Create submission metadata
submission_info = {
    "model_type": "XGBClassifier",
    "cv_strategy": f"{N_SPLITS}-Fold Stratified Cross Validation",
    "ensemble_method": "Average of 10-fold CV models",
    "map3_score_cv": f"{map3_mean:.5f} Â± {map3_std:.5f}",
    "map3_score_oof": float(training_results['overall_map3']),
    "submission_file": files_to_create['submission'],
    "num_predictions": len(submission),
    "features_used": len(features_to_use),
    "hyperparameters": xgb_params,
    "timestamp": datetime.now().isoformat(),
    "kaggle_competition": "playground-series-s5e6"
}

# Save submission metadata
submission_info_file = os.path.join(model_dir, files_to_create['submission_info'])
with open(submission_info_file, 'w') as f:
    json.dump(submission_info, f, indent=2)

print(f"âœ… KAGGLE SUBMISSION READY")
print(f"  ğŸ“„ File: {files_to_create['submission']}")
print(f"  ğŸ“Š Samples: {len(submission):,}")
print(f"  ğŸ“ˆ MAP@3 (CV): {map3_mean:.5f} Â± {map3_std:.5f}")
print(f"  ğŸ“ˆ MAP@3 (OOF): {training_results['overall_map3']:.5f}")
print(f"  â�±ï¸� Training: {total_time/60:.1f} minutes")
print(f"  ğŸš€ Ready for competition upload!")






