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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("IMPROVED DISTRACTED DRIVING RISK DETECTION MODEL")
print("="*70)

# Load the data
train_df = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_train.csv')
test_df = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_test.csv')
unlabeled_df = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_full_unlabeled_data.csv')

print(f"\nDataset Shapes:")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(f"Unlabeled data: {unlabeled_df.shape}")

# Drop label_source column if it exists
train_df_numeric = train_df.drop(columns=['label_source'], errors='ignore')
test_df_numeric = test_df.drop(columns=['label_source'], errors='ignore')
unlabeled_df_numeric = unlabeled_df.drop(columns=['label_source'], errors='ignore')

print(f"\nTarget distribution in training data:")
print(train_df['risk_level'].value_counts().sort_index())


# Enhanced Feature Engineering
def create_enhanced_features(df):
    """Create comprehensive engineered features"""
    df_new = df.copy()
    
    # Speed-related features
    df_new['speed_violation_ratio'] = df['speed'] / (df['design_speed'] + 0.001)
    df_new['speed_excess'] = np.maximum(0, df['speed'] - df['design_speed'])
    df_new['is_speeding'] = (df['speed'] > df['design_speed']).astype(int)
    df_new['speed_squared'] = df['speed'] ** 2
    df_new['speed_cubed'] = df['speed'] ** 3
    
    # Acceleration features
    df_new['hard_braking'] = (df['acceleration'] < -2.5).astype(int)
    df_new['moderate_braking'] = ((df['acceleration'] < -1.0) & (df['acceleration'] >= -2.5)).astype(int)
    df_new['rapid_acceleration'] = (df['acceleration'] > 2.0).astype(int)
    df_new['acceleration_squared'] = df['acceleration'] ** 2
    df_new['abs_acceleration'] = np.abs(df['acceleration'])
    
    # Engine metrics
    df_new['engine_stress'] = df['rpm'] * df['engine_load_value'] / 100
    df_new['throttle_load_ratio'] = df['throttle_position'] / (df['engine_load_value'] + 1)
    df_new['throttle_load_diff'] = df['throttle_position'] - df['engine_load_value']
    df_new['engine_efficiency'] = df['speed'] / (df['rpm'] + 1) * 1000
    df_new['high_rpm'] = (df['rpm'] > 3000).astype(int)
    df_new['very_high_rpm'] = (df['rpm'] > 3500).astype(int)
    df_new['low_rpm_high_load'] = ((df['rpm'] < 2000) & (df['engine_load_value'] > 50)).astype(int)
    
    # Engine temperature
    df_new['engine_temp_normal'] = ((df['engine_temperature'] >= 85) & (df['engine_temperature'] <= 105)).astype(int)
    df_new['engine_temp_high'] = (df['engine_temperature'] > 105).astype(int)
    df_new['engine_temp_low'] = (df['engine_temperature'] < 85).astype(int)
    
    # Driver biometrics
    hr_mean = df['heart_rate'].mean()
    hr_std = df['heart_rate'].std()
    df_new['heart_rate_zscore'] = (df['heart_rate'] - hr_mean) / (hr_std + 0.001)
    df_new['elevated_heart_rate'] = (df['heart_rate'] > 100).astype(int)
    df_new['very_high_heart_rate'] = (df['heart_rate'] > 110).astype(int)
    df_new['heart_rate_squared'] = df['heart_rate'] ** 2
    
    # Weather and visibility
    df_new['weather_risk'] = df['current_weather'] * (1 / (df['visibility'] + 0.1)) * (df['precipitation'] + 1)
    df_new['low_visibility'] = (df['visibility'] < 5).astype(int)
    df_new['very_low_visibility'] = (df['visibility'] < 3.5).astype(int)
    df_new['heavy_precipitation'] = (df['precipitation'] > 10).astype(int)
    df_new['moderate_precipitation'] = ((df['precipitation'] > 5) & (df['precipitation'] <= 10)).astype(int)
    df_new['bad_weather'] = (df['current_weather'] > 15).astype(int)
    df_new['weather_visibility_ratio'] = df['current_weather'] / (df['visibility'] + 0.1)
    df_new['visibility_precipitation_interaction'] = df['visibility'] * df['precipitation']
    
    # Location risk
    df_new['location_risk'] = df['accidents_onsite'] + df['accidents_time']
    df_new['dangerous_location'] = (df['accidents_onsite'] > 50).astype(int)
    df_new['very_dangerous_location'] = (df['accidents_onsite'] > 100).astype(int)
    df_new['recent_accidents'] = (df['accidents_time'] > 3).astype(int)
    df_new['accidents_squared'] = df['accidents_onsite'] ** 2
    
    # Time-based features
    df_new['is_rush_hour'] = df['observation_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_new['is_night'] = ((df['observation_hour'] >= 20) | (df['observation_hour'] <= 5)).astype(int)
    df_new['is_evening'] = df['observation_hour'].isin([17, 18, 19, 20]).astype(int)
    df_new['is_late_night'] = ((df['observation_hour'] >= 22) | (df['observation_hour'] <= 4)).astype(int)
    
    # Combined risk scores
    df_new['aggressive_driving_score'] = (
        df_new['rapid_acceleration'] * 2 + 
        df_new['hard_braking'] * 2 + 
        df_new['is_speeding'] * 1.5 +
        df_new['very_high_rpm']
    )
    
    df_new['environmental_risk_score'] = (
        df_new['low_visibility'] * 2 +
        df_new['heavy_precipitation'] * 1.5 +
        df_new['bad_weather'] * 1.5
    )
    
    df_new['total_risk_score'] = (
        df_new['speed_violation_ratio'] + 
        df_new['weather_risk'] + 
        df_new['location_risk'] / 10 +
        df_new['aggressive_driving_score'] +
        df_new['environmental_risk_score']
    )
    
    # Interaction features
    df_new['speed_weather_interaction'] = df['speed'] * df['current_weather']
    df_new['speed_visibility_interaction'] = df['speed'] / (df['visibility'] + 0.1)
    df_new['speed_acceleration_product'] = df['speed'] * df_new['abs_acceleration']
    df_new['speed_heart_rate_interaction'] = df['speed'] * df['heart_rate'] / 100
    df_new['rpm_throttle_interaction'] = df['rpm'] * df['throttle_position'] / 100
    df_new['location_weather_interaction'] = df['accidents_onsite'] * df['current_weather'] / 10
    df_new['speed_location_interaction'] = df['speed'] * df['accidents_onsite'] / 100
    
    # Binned features
    df_new['speed_bin'] = pd.cut(df['speed'], bins=[-0.001, 20, 40, 60, 80, 100, 200], labels=False, include_lowest=True)
    df_new['speed_bin'] = df_new['speed_bin'].fillna(0).astype(int)
    
    df_new['heart_rate_bin'] = pd.cut(df['heart_rate'], bins=[0, 70, 85, 100, 115, 200], labels=False, include_lowest=True)
    df_new['heart_rate_bin'] = df_new['heart_rate_bin'].fillna(0).astype(int)
    
    df_new['rpm_bin'] = pd.cut(df['rpm'], bins=[-0.001, 1500, 2500, 3500, 5000], labels=False, include_lowest=True)
    df_new['rpm_bin'] = df_new['rpm_bin'].fillna(0).astype(int)
    
    df_new['visibility_bin'] = pd.cut(df['visibility'], bins=[0, 3.5, 6, 10, 20], labels=False, include_lowest=True)
    df_new['visibility_bin'] = df_new['visibility_bin'].fillna(0).astype(int)
    
    # Polynomial features for key metrics
    df_new['speed_rpm_ratio'] = df['speed'] / (df['rpm'] + 1) * 1000
    df_new['power_estimate'] = df['rpm'] * df['throttle_position'] * df['speed'] / 10000
    
    # Risk flags
    df_new['multiple_risk_factors'] = (
        df_new['is_speeding'] + 
        df_new['low_visibility'] + 
        df_new['dangerous_location'] + 
        df_new['elevated_heart_rate'] +
        df_new['hard_braking']
    )
    
    df_new['extreme_conditions'] = (
        (df_new['very_low_visibility'] == 1) | 
        (df_new['very_dangerous_location'] == 1) |
        (df_new['very_high_heart_rate'] == 1)
    ).astype(int)
    
    return df_new


# Apply feature engineering
print("\nApplying enhanced feature engineering...")
train_features = create_enhanced_features(train_df_numeric)
test_features = create_enhanced_features(test_df_numeric)
unlabeled_features = create_enhanced_features(unlabeled_df_numeric)

print(f"Number of features after engineering: {len([col for col in train_features.columns if col != 'risk_level'])}")

# Prepare data
feature_cols = [col for col in train_features.columns if col != 'risk_level']
X = train_features[feature_cols]
y = train_features['risk_level']
X_test = test_features[feature_cols]
X_unlabeled = unlabeled_features[feature_cols]

# Handle infinity and NaN values
for df in [X, X_test, X_unlabeled]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Convert to DataFrames
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
X_unlabeled_scaled_df = pd.DataFrame(X_unlabeled_scaled, columns=feature_cols)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_df, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")


# Define models with optimized parameters
def get_models():
    models = {}
    
    # XGBoost
    models['xgb'] = XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        objective='multi:softprob',
        num_class=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss',
        gamma=0.15,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=0.2
    )
    
    # LightGBM
    models['lgb'] = LGBMClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        metric='multi_logloss',
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=0.2,
        verbosity=-1
    )
    
    # CatBoost
    models['cat'] = CatBoostClassifier(
        iterations=800,
        depth=7,
        learning_rate=0.03,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_state=RANDOM_STATE,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        verbose=0,
        l2_leaf_reg=3
    )
    
    return models


# Train models
print("\n" + "="*70)
print("TRAINING ENSEMBLE MODELS")
print("="*70)

# Adjust labels for XGBoost and LightGBM (0-indexed)
y_train_adjusted = y_train - 1
y_val_adjusted = y_val - 1

models = get_models()
trained_models = {}
val_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name.upper()}...")
    
    if name == 'xgb':
        model.fit(
            X_train, y_train_adjusted,
            eval_set=[(X_val, y_val_adjusted)],
            verbose=False
        )
        val_pred = model.predict(X_val)
        val_pred_adjusted = val_pred + 1
    elif name == 'lgb':
        model.fit(
            X_train, y_train_adjusted,
            eval_set=[(X_val, y_val_adjusted)]
        )
        val_pred = model.predict(X_val)
        val_pred_adjusted = val_pred + 1
    else:  # CatBoost uses 1-indexed
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_pred_adjusted = np.array(model.predict(X_val).flatten(), dtype=int)
    
    val_predictions[name] = val_pred_adjusted
    trained_models[name] = model
    
    accuracy = accuracy_score(y_val, val_pred_adjusted)
    print(f"{name.upper()} Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Distribution: {np.bincount(val_pred_adjusted, minlength=5)[1:]}")


# Ensemble predictions - Weighted Voting
print("\n" + "="*70)
print("ENSEMBLE PREDICTIONS")
print("="*70)

# Calculate weights based on validation accuracy
weights = {}
for name in models.keys():
    accuracy = accuracy_score(y_val, val_predictions[name])
    weights[name] = accuracy ** 2

# Normalize weights
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\nModel Weights: {weights}")

# Weighted ensemble for validation
ensemble_val_proba = np.zeros((len(X_val), 4))
for name, model in trained_models.items():
    proba = model.predict_proba(X_val)
    ensemble_val_proba += weights[name] * proba

ensemble_val_pred = np.argmax(ensemble_val_proba, axis=1) + 1
ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)

print(f"\nEnsemble Validation Accuracy: {ensemble_val_accuracy:.4f}")
print(f"Ensemble Validation Distribution: {np.bincount(ensemble_val_pred, minlength=5)[1:]}")


# Pseudo-labeling using unlabeled data
print("\n" + "="*70)
print("PSEUDO-LABELING")
print("="*70)

# Get predictions on unlabeled data
pseudo_proba = np.zeros((len(X_unlabeled_scaled_df), 4))
for name, model in trained_models.items():
    proba = model.predict_proba(X_unlabeled_scaled_df)
    pseudo_proba += weights[name] * proba

pseudo_confidence = np.max(pseudo_proba, axis=1)
pseudo_labels = np.argmax(pseudo_proba, axis=1) + 1

# Select high-confidence predictions (top 30%)
confidence_threshold = np.percentile(pseudo_confidence, 70)
high_confidence_mask = pseudo_confidence >= confidence_threshold

print(f"Confidence threshold: {confidence_threshold:.4f}")
print(f"High-confidence samples: {high_confidence_mask.sum()} / {len(X_unlabeled_scaled_df)}")

# Add high-confidence pseudo-labeled data to training
X_pseudo = X_unlabeled_scaled_df[high_confidence_mask]
y_pseudo = pseudo_labels[high_confidence_mask]

X_augmented = pd.concat([X_scaled_df, X_pseudo], ignore_index=True)
y_augmented = np.concatenate([y.values, y_pseudo])

print(f"Augmented training size: {len(X_augmented)}")
print(f"Pseudo-label distribution: {np.bincount(y_pseudo, minlength=5)[1:]}")


# Retrain models on augmented data with cross-validation
print("\n" + "="*70)
print("RETRAINING ON AUGMENTED DATA")
print("="*70)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

final_models = {}
cv_scores = {name: [] for name in models.keys()}

for name in models.keys():
    print(f"\nCross-validating {name.upper()}...")
    
    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_augmented, y_augmented), 1):
        X_fold_train, X_fold_val = X_augmented.iloc[train_idx], X_augmented.iloc[val_idx]
        y_fold_train, y_fold_val = y_augmented[train_idx], y_augmented[val_idx]
        
        model = get_models()[name]
        
        if name == 'lgb':
            model.fit(X_fold_train, y_fold_train - 1)
            fold_pred = model.predict(X_fold_val) + 1
        elif name == 'xgb':
            model.fit(X_fold_train, y_fold_train - 1, verbose=False)
            fold_pred = model.predict(X_fold_val) + 1
        else:
            model.fit(X_fold_train, y_fold_train, verbose=False)
            fold_pred = np.array(model.predict(X_fold_val).flatten(), dtype=int)
        
        fold_acc = accuracy_score(y_fold_val, fold_pred)
        cv_scores[name].append(fold_acc)
        fold_models.append(model)
    
    final_models[name] = fold_models
    mean_cv = np.mean(cv_scores[name])
    std_cv = np.std(cv_scores[name])
    print(f"{name.upper()} CV Accuracy: {mean_cv:.4f} (+/- {std_cv:.4f})")


# Final predictions on test set
print("\n" + "="*70)
print("GENERATING FINAL PREDICTIONS")
print("="*70)

test_predictions_all = {}

for name, fold_models in final_models.items():
    fold_probas = []
    for model in fold_models:
        proba = model.predict_proba(X_test_scaled_df)
        fold_probas.append(proba)
    
    # Average predictions across folds
    avg_proba = np.mean(fold_probas, axis=0)
    test_predictions_all[name] = avg_proba

# Weighted ensemble
final_test_proba = np.zeros((len(X_test_scaled_df), 4))
for name, proba in test_predictions_all.items():
    final_test_proba += weights[name] * proba

final_test_pred = np.argmax(final_test_proba, axis=1) + 1

# Create submission
submission = pd.DataFrame({
    'id': range(len(final_test_pred)),
    'risk_level': final_test_pred
})

print("\n" + "="*70)
print("SUBMISSION SUMMARY")
print("="*70)
print(f"Submission shape: {submission.shape}")
print(f"\nPredicted class distribution:")
print(submission['risk_level'].value_counts().sort_index())
print(f"\nPredicted class distribution (%):")
print((submission['risk_level'].value_counts(normalize=True).sort_index() * 100).round(2))

# Confidence analysis
max_proba = np.max(final_test_proba, axis=1)
print(f"\nMean prediction confidence: {max_proba.mean():.4f}")
print(f"Median prediction confidence: {np.median(max_proba):.4f}")

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved as 'submission.csv'")

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)
for name in models.keys():
    mean_cv = np.mean(cv_scores[name])
    std_cv = np.std(cv_scores[name])
    print(f"{name.upper()} Mean CV: {mean_cv:.4f} (+/- {std_cv:.4f})")

print(f"\nEnsemble Validation Accuracy: {ensemble_val_accuracy:.4f}")
print(f"Total features used: {len(feature_cols)}")
print(f"Training samples (original): {len(train_df)}")
print(f"Training samples (augmented): {len(X_augmented)}")
print(f"Test samples: {len(X_test)}")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)

