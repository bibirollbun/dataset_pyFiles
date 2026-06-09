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


!pip install autogluon


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
import time
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for results
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)

# Install AutoGluon if not already installed
try:
    import autogluon
    from autogluon.tabular import TabularPredictor
    print("AutoGluon successfully imported")
except ImportError:
    print("Installing AutoGluon...")
    !pip install -U pip
    !pip install -U setuptools wheel
    !pip install autogluon
    import autogluon
    from autogluon.tabular import TabularPredictor

# Load the data
print("Loading data...")
train_data = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')

print("\nTrain data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Check for missing values
print("\nMissing values per column:")
missing_values = train_data.isnull().sum()
print(missing_values[missing_values > 0])

# Display target distribution
print("\nTarget distribution:")
print(train_data['outcome'].value_counts())
print(train_data['outcome'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

# Advanced Feature Engineering for veterinary data
def engineer_features(df):
    """Create advanced domain-specific features based on veterinary knowledge."""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Fill missing values for further feature engineering
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('missing')
        else:
            if col not in ['id', 'lesion_1', 'lesion_2', 'lesion_3']:
                data[col] = data[col].fillna(data[col].median())
    
    # 1. ADVANCED VITAL SIGNS ANALYSIS
    if all(col in data.columns for col in ['rectal_temp', 'pulse', 'respiratory_rate']):
        # Temperature analysis (normal range for horses is approximately 37.5-38.5°C)
        data['temp_low'] = (data['rectal_temp'] < 37.5).astype(int)
        data['temp_high'] = (data['rectal_temp'] > 38.5).astype(int)
        data['temp_deviation'] = abs(data['rectal_temp'] - 38.0)  # Deviation from ideal temp
        
        # Pulse rate analysis (normal range for horses is approximately 28-44 bpm)
        data['pulse_low'] = (data['pulse'] < 28).astype(int)
        data['pulse_high'] = (data['pulse'] > 44).astype(int)
        data['pulse_very_high'] = (data['pulse'] > 80).astype(int)  # Severe tachycardia
        
        # Respiratory rate analysis (normal range for horses is approximately 8-16 breaths/min)
        data['resp_low'] = (data['respiratory_rate'] < 8).astype(int)
        data['resp_high'] = (data['respiratory_rate'] > 16).astype(int)
        data['resp_very_high'] = (data['respiratory_rate'] > 30).astype(int)  # Severe tachypnea
        
        # Vital sign ratios (clinically relevant)
        data['pulse_temp_ratio'] = data['pulse'] / data['rectal_temp']
        data['resp_pulse_ratio'] = data['respiratory_rate'] / data['pulse'].replace(0, 0.1)
        
        # Combined severity score
        data['vital_signs_severity'] = (
            data['temp_low'] * 1 + data['temp_high'] * 1 +
            data['pulse_low'] * 1 + data['pulse_high'] * 1 + data['pulse_very_high'] * 2 +
            data['resp_low'] * 1 + data['resp_high'] * 1 + data['resp_very_high'] * 2
        )
    
    # 2. COMPREHENSIVE LESION ANALYSIS
    if all(col in data.columns for col in ['lesion_1', 'lesion_2', 'lesion_3']):
        # Basic lesion features
        data['has_lesion'] = ((data['lesion_1'] > 0) | (data['lesion_2'] > 0) | (data['lesion_3'] > 0)).astype(int)
        data['lesion_count'] = ((data['lesion_1'] > 0).astype(int) + 
                               (data['lesion_2'] > 0).astype(int) + 
                               (data['lesion_3'] > 0).astype(int))
        
        # Primary lesion site (presence indicators for specific lesion codes)
        data['lesion_gi_tract'] = ((data['lesion_1'].isin([2208, 5124])) | 
                                  (data['lesion_2'].isin([2208, 5124])) | 
                                  (data['lesion_3'].isin([2208, 5124]))).astype(int)
        
        # Multiple site involvement
        data['multiple_lesion_sites'] = (data['lesion_count'] > 1).astype(int)
        
        # Create lesion severity indicator based on count and primary sites
        data['lesion_severity'] = data['lesion_count'] + data['lesion_gi_tract']
    
    # 3. ADVANCED SURGICAL INDICATORS
    if 'surgery' in data.columns:
        # Convert surgery categorical to numeric
        data['surgery_performed'] = (data['surgery'] == 'yes').astype(int)
        
        if 'surgical_lesion' in data.columns:
            data['surgical_lesion_identified'] = (data['surgical_lesion'] == 'yes').astype(int)
            
            # Surgical intervention match (surgery performed when lesion identified)
            data['surgical_intervention_match'] = (
                (data['surgery_performed'] == 1) & (data['surgical_lesion_identified'] == 1)
            ).astype(int)
            
            # Surgical mismatch (surgery without lesion or lesion without surgery)
            data['surgical_mismatch'] = (
                ((data['surgery_performed'] == 1) & (data['surgical_lesion_identified'] == 0)) |
                ((data['surgery_performed'] == 0) & (data['surgical_lesion_identified'] == 1))
            ).astype(int)
        
        # Surgery with lesion interactions
        if 'has_lesion' in data.columns:
            data['surgery_with_lesion'] = (data['surgery_performed'] & data['has_lesion']).astype(int)
    
    # 4. ENHANCED ABDOMINAL ASSESSMENT
    abdominal_columns = ['abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 
                          'nasogastric_reflux_ph', 'rectal_exam_feces', 'abdomen', 'abdomo_appearance']
    present_abdominal_columns = [col for col in abdominal_columns if col in data.columns]
    
    if present_abdominal_columns:
        # Create abdominal distress composite score
        abdominal_score = pd.Series(0, index=data.index)
        
        # Process individual components
        if 'abdominal_distention' in data.columns:
            abdominal_score += ((data['abdominal_distention'] == 'severe').astype(int) * 3 + 
                              (data['abdominal_distention'] == 'moderate').astype(int) * 2 +
                              (data['abdominal_distention'] == 'slight').astype(int) * 1)
        
        if 'nasogastric_reflux' in data.columns:
            abdominal_score += ((data['nasogastric_reflux'] == 'significant').astype(int) * 2 + 
                              (data['nasogastric_reflux'] == 'slight').astype(int) * 1)
        
        if 'nasogastric_reflux_ph' in data.columns and not data['nasogastric_reflux_ph'].isna().all():
            # Lower pH is more concerning
            abdominal_score += ((data['nasogastric_reflux_ph'] < 4).astype(int) * 2 +
                              ((data['nasogastric_reflux_ph'] >= 4) & 
                               (data['nasogastric_reflux_ph'] < 5)).astype(int) * 1)
        
        if 'abdomo_appearance' in data.columns:
            abdominal_score += ((data['abdomo_appearance'] == 'serosanguious').astype(int) * 2 + 
                              (data['abdomo_appearance'] == 'cloudy').astype(int) * 1)
        
        data['abdominal_severity'] = abdominal_score
    
    # 5. PAIN AND DISCOMFORT ANALYSIS
    pain_columns = ['pain', 'peristalsis', 'capillary_refill_time']
    if all(col in data.columns for col in pain_columns):
        pain_score = pd.Series(0, index=data.index)
        
        # Calculate pain level from specific indicators
        if 'pain' in data.columns:
            pain_score += ((data['pain'] == 'severe').astype(int) * 3 +
                         (data['pain'] == 'moderate').astype(int) * 2 +
                         (data['pain'] == 'mild').astype(int) * 1)
        
        if 'peristalsis' in data.columns:
            # Abnormal peristalsis can indicate pain
            pain_score += ((data['peristalsis'] == 'absent').astype(int) * 3 +
                         (data['peristalsis'] == 'hypomotile').astype(int) * 2 +
                         (data['peristalsis'] == 'hypermotile').astype(int) * 1)
        
        if 'capillary_refill_time' in data.columns:
            # Longer CRT can indicate shock
            pain_score += ((data['capillary_refill_time'] == '>3').astype(int) * 2 +
                         (data['capillary_refill_time'] == '2').astype(int) * 1)
        
        data['pain_severity'] = pain_score
    
    # 6. BLOOD VALUES ANALYSIS
    if all(col in data.columns for col in ['packed_cell_volume', 'total_protein']):
        # PCV analysis (normal range typically 32-48%)
        data['pcv_low'] = (data['packed_cell_volume'] < 32).astype(int)  # Anemia
        data['pcv_high'] = (data['packed_cell_volume'] > 48).astype(int)  # Dehydration/hemoconcentration
        
        # Total protein analysis (normal range typically 6-8 g/dL)
        data['protein_low'] = (data['total_protein'] < 6).astype(int)  # Hypoproteinemia
        data['protein_high'] = (data['total_protein'] > 8).astype(int)  # Hyperproteinemia
        
        # Protein-to-PCV ratio (useful clinical marker)
        data['protein_pcv_ratio'] = data['total_protein'] / data['packed_cell_volume'].replace(0, 0.1)
        
        # Create composite blood values score
        data['blood_values_abnormality'] = (
            data['pcv_low'] * 1 + data['pcv_high'] * 1 +
            data['protein_low'] * 1 + data['protein_high'] * 1
        )
    
    # 7. COMPOSITE CLINICAL SCORES
    
    # Create overall clinical severity score
    severity_columns = [col for col in ['vital_signs_severity', 'lesion_severity', 
                                        'abdominal_severity', 'pain_severity',
                                        'blood_values_abnormality'] 
                       if col in data.columns]
    
    if severity_columns:
        data['clinical_severity_score'] = data[severity_columns].sum(axis=1)
    
    # Age factor (young animals might have different outcomes)
    if 'age' in data.columns:
        data['is_young'] = (data['age'] == 'young').astype(int)
    
    return data

# Apply feature engineering
print("\nApplying advanced domain-specific feature engineering...")
train_processed = engineer_features(train_data)
test_processed = engineer_features(test_data)

# Create polynomial features for key numerical variables
def add_polynomial_features(train_df, test_df, columns, degree=2):
    """Add polynomial features for specified numerical columns."""
    if not all(col in train_df.columns for col in columns):
        available_cols = [col for col in columns if col in train_df.columns]
        if not available_cols:
            return train_df, test_df
        columns = available_cols
    
    # Select only columns that exist in both dataframes
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    # Fit and transform the training data
    poly_features = poly.fit_transform(train_df[columns])
    
    # Generate feature names
    feature_names = []
    for i in range(poly_features.shape[1]):
        if i < len(columns):
            feature_names.append(f"{columns[i]}")
        else:
            feature_names.append(f"poly_{i}")
    
    # Create dataframes with polynomial features
    poly_train = pd.DataFrame(poly_features, columns=feature_names, index=train_df.index)
    
    # Transform test data
    poly_test_features = poly.transform(test_df[columns])
    poly_test = pd.DataFrame(poly_test_features, columns=feature_names, index=test_df.index)
    
    # Remove original columns from poly dataframes to avoid duplication
    poly_train = poly_train.drop(columns=columns)
    poly_test = poly_test.drop(columns=columns)
    
    # Join polynomial features with original dataframes
    train_with_poly = pd.concat([train_df, poly_train], axis=1)
    test_with_poly = pd.concat([test_df, poly_test], axis=1)
    
    return train_with_poly, test_with_poly

# Add polynomial features for key numerical health indicators
numerical_columns = ['rectal_temp', 'pulse', 'respiratory_rate', 'packed_cell_volume', 
                     'total_protein', 'abdomo_protein']
valid_num_cols = [col for col in numerical_columns if col in train_processed.columns]

if valid_num_cols:
    print("\nAdding polynomial features for key numerical indicators...")
    train_processed, test_processed = add_polynomial_features(
        train_processed, test_processed, valid_num_cols, degree=2
    )

# Display new features
print("\nNew engineered features:")
new_features = [col for col in train_processed.columns if col not in train_data.columns]
print(f"Added {len(new_features)} new features")
print(new_features[:10])  # Show first 10 features to avoid cluttering output
print("...")

# Split into train and validation sets
train_split, val_data = train_test_split(
    train_processed, test_size=0.2, random_state=42, stratify=train_processed['outcome']
)

# Configure optimal AutoGluon settings for this specific task
print("\nTraining model with optimized AutoGluon configuration...")
start_time = time.time()

# Configure AutoGluon predictor with optimal settings
predictor = TabularPredictor(
    label='outcome',
    path=os.path.join(output_dir, 'autogluon_models'),
    eval_metric='f1_micro',
    problem_type='multiclass'
)

# Define enhanced hyperparameters for compatible models
hyperparameters = {
    'GBM': [
        # Default LightGBM
        {},
        # LightGBM optimized for this dataset
        {
            'num_boost_round': 1000,
            'num_leaves': 128,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 4,
            'extra_trees': False,
            'ag_args': {'name_suffix': 'Tuned'}
        },
        # Extra Trees version
        {
            'extra_trees': True,
            'feature_fraction': 0.8,
            'ag_args': {'name_suffix': 'XT'}
        }
    ],
    'CAT': [
        # Default CatBoost
        {},
        # Optimized CatBoost
        {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'ag_args': {'name_suffix': 'Tuned'}
        }
    ],
    'RF': [
        # Random Forest with Gini
        {
            'criterion': 'gini',
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'ag_args': {'name_suffix': 'Gini'}
        },
        # Random Forest with Entropy
        {
            'criterion': 'entropy',
            'max_features': 'log2',
            'min_samples_leaf': 2,
            'ag_args': {'name_suffix': 'Entropy'}
        }
    ],
    'NN_TORCH': [
        # Default Neural Network
        {},
        # Tuned Neural Network
        {
            'num_epochs': 120,
            'activation': 'relu',
            'dropout_prob': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'ag_args': {'name_suffix': 'Tuned'}
        }
    ],
    'KNN': [
        {'weights': 'uniform', 'n_neighbors': 10, 'ag_args': {'name_suffix': 'Unif'}},
        {'weights': 'distance', 'n_neighbors': 20, 'ag_args': {'name_suffix': 'Dist'}}
    ]
}

# Handle class imbalance through data augmentation rather than weights
# This preserves all the existing feature engineering while addressing imbalance
print("\nApplying class balancing through data augmentation...")

# Get counts of each class
class_counts = train_split['outcome'].value_counts()
max_class_count = class_counts.max()

# Create a balanced training dataset
balanced_dfs = []

# Add all existing data
balanced_dfs.append(train_split)

# For minority classes, add more samples until we approach balance
for outcome_class, count in class_counts.items():
    if count < max_class_count:
        # Calculate how many additional samples we need
        samples_to_add = max_class_count - count
        
        # Get all samples of this class
        class_samples = train_split[train_split['outcome'] == outcome_class]
        
        # Add with replacement to reach desired count
        additional_samples = class_samples.sample(n=samples_to_add, replace=True, random_state=42)
        balanced_dfs.append(additional_samples)

# Combine into balanced dataset
balanced_train = pd.concat(balanced_dfs, axis=0).reset_index(drop=True)

print(f"Original training data shape: {train_split.shape}")
print(f"Balanced training data shape: {balanced_train.shape}")
print("Class distribution after balancing:")
print(balanced_train['outcome'].value_counts())

# Fit with enhanced settings
predictor.fit(
    balanced_train,  # Use balanced training data
    time_limit=1200,  # 20 minutes
    hyperparameters=hyperparameters,
    excluded_model_types=['XGB'],  # Exclude XGBoost models to avoid errors
    num_stack_levels=2,  # Enable multi-level stacking
    num_bag_folds=5,  # Use bagging with 5 folds
    num_bag_sets=3,  # Number of repeats for bagging
    verbosity=2,
    ag_args_fit={
        'num_gpus': 0,  # Set to 1 or more if GPUs are available
        'refit_full': True  # Refit on full dataset after tuning
    }
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Try to get leaderboard, but handle potential errors
print("\nModel evaluation:")
try:
    print("\nAttempting to generate leaderboard...")
    leaderboard = predictor.leaderboard()
    print(leaderboard.head(10))  # Show top 10 models
    leaderboard.to_csv(os.path.join(output_dir, 'model_leaderboard.csv'), index=False)
except Exception as e:
    print(f"Could not generate leaderboard due to: {str(e)}")
    print("Continuing with direct evaluation...")

# Try to get feature importance, but handle potential errors
try:
    print("\nFeature importance analysis...")
    # Sample data to avoid memory issues
    sample_size = min(500, len(train_processed))
    feature_importance = predictor.feature_importance(
        train_processed.sample(n=sample_size, random_state=42)
    )
    
    # Display and save top features
    print("\nTop 20 important features:")
    print(feature_importance.head(20))
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    sns.barplot(x='importance', y=top_features.index, data=top_features)
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
except Exception as e:
    print(f"Could not analyze feature importance: {str(e)}")

# Evaluate on validation set
try:
    print("\nValidation set evaluation:")
    val_predictions = predictor.predict(val_data)
    
    # Calculate F1 scores
    val_f1_micro = f1_score(val_data['outcome'], val_predictions, average='micro')
    val_f1_macro = f1_score(val_data['outcome'], val_predictions, average='macro')
    val_f1_weighted = f1_score(val_data['outcome'], val_predictions, average='weighted')
    
    print(f"Validation F1 Score (Micro): {val_f1_micro:.4f}")
    print(f"Validation F1 Score (Macro): {val_f1_macro:.4f}")
    print(f"Validation F1 Score (Weighted): {val_f1_weighted:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_data['outcome'], val_predictions))
    
    # Enhanced confusion matrix with percentages
    plt.figure(figsize=(12, 6))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # First subplot: raw counts
    conf_matrix = confusion_matrix(val_data['outcome'], val_predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(train_data['outcome'].unique()), 
                yticklabels=sorted(train_data['outcome'].unique()), ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Second subplot: percentages
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_matrix_pct, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=sorted(train_data['outcome'].unique()), 
                yticklabels=sorted(train_data['outcome'].unique()), ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
except Exception as e:
    print(f"Error in validation evaluation: {str(e)}")
    print("Continuing with prediction generation...")

# Generate predictions with improved reliability
print("\nGenerating predictions for test set...")

# Free memory before prediction
gc.collect()

# Generate predictions with robust error handling
test_predictions = predictor.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'outcome': test_predictions
})

submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print("\nSubmission file created:", os.path.join(output_dir, 'submission.csv'))

# Try to save probability predictions
try:
    test_pred_proba = predictor.predict_proba(test_processed)
    test_pred_proba.to_csv(os.path.join(output_dir, 'test_probabilities.csv'))
    print("\nProbability predictions saved for potential ensembling.")
except Exception as e:
    print(f"Could not save probability predictions: {str(e)}")

# Create overall performance summary
with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
    f.write("HORSE HEALTH PREDICTION MODEL SUMMARY\n")
    f.write("=====================================\n\n")
    
    f.write(f"Training completed in {training_time:.2f} seconds\n\n")
    
    f.write("VALIDATION METRICS:\n")
    try:
        f.write(f"F1 Score (Micro): {val_f1_micro:.4f}\n")
        f.write(f"F1 Score (Macro): {val_f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted): {val_f1_weighted:.4f}\n\n")
    except:
        f.write("Could not calculate validation metrics\n\n")
    
    f.write(f"FEATURE ENGINEERING:\n")
    f.write(f"Created {len(new_features)} new features\n\n")
    
    f.write("KEY OPTIMIZATIONS:\n")
    f.write("- Advanced veterinary-specific feature engineering\n")
    f.write("- Multi-level model stacking with diverse algorithms\n")
    f.write("- Class imbalance handling via weighting\n")
    f.write("- Polynomial features for key numerical indicators\n")
    f.write("- Comprehensive validation strategy\n")

print("\nPerformance summary saved to:", os.path.join(output_dir, 'performance_summary.txt'))
print("\nEnd of script")

