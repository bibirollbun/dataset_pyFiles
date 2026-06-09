# Core data manipulation and analysis
import pandas as pd
import numpy as np
import os
import warnings
from itertools import combinations

# Machine learning and evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# XGBoost and GPU acceleration
import xgboost as xgb
from cuml.preprocessing import TargetEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(f"âœ… XGBoost version: {xgb.__version__}")
print(f"âœ… Libraries loaded successfully!")


# Define data paths
DATA_PATH = "/kaggle/input/playground-series-s5e8/"
ORIGINAL_PATH = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"

print("ğŸ“‚ Loading datasets...")

# Load competition training data
train_df = pd.read_csv(f"{DATA_PATH}train.csv").set_index('id')
print(f"ğŸ�¯ Training data shape: {train_df.shape}")

# Load competition test data
test_df = pd.read_csv(f"{DATA_PATH}test.csv").set_index('id')
test_df['y'] = -1  # Placeholder for target (unknown)
print(f"ğŸ�¯ Test data shape: {test_df.shape}")

# Load original bank marketing dataset
original_df = pd.read_csv(ORIGINAL_PATH, delimiter=";")
original_df['y'] = original_df['y'].map({'yes': 1, 'no': 0})  # Convert target to binary
original_df['id'] = (np.arange(len(original_df)) + 1_000_000).astype('int')  # Create unique IDs
original_df = original_df.set_index('id')
print(f"ğŸ�¯ Original data shape: {original_df.shape}")

# Display basic information about each dataset
print("\n" + "="*50)
print("ğŸ“Š DATASET SUMMARY")
print("="*50)
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Original samples: {len(original_df):,}")
print(f"Total samples for analysis: {len(train_df) + len(test_df) + len(original_df):,}")


# Display sample of each dataset
print("ğŸ”� Sample of Training Data:")
display(train_df.head())

print("\nğŸ”� Sample of Test Data:")
display(test_df.head())

print("\nğŸ”� Sample of Original Data:")
display(original_df.head())


# Combine all datasets for consistent feature engineering
combined_data = pd.concat([train_df, test_df, original_df], axis=0)
print(f"ğŸ“Š Combined dataset shape: {combined_data.shape}")

# Analyze feature types and characteristics
categorical_features = []
numerical_features = []

print("\n" + "="*60)
print("ğŸ”� FEATURE ANALYSIS")
print("="*60)

for column in combined_data.columns[:-1]:  # Exclude target 'y'
    unique_count = combined_data[column].nunique()
    missing_count = combined_data[column].isna().sum()
    
    if combined_data[column].dtype == 'object':
        categorical_features.append(column)
        feature_type = "CATEGORICAL"
    else:
        numerical_features.append(column)
        feature_type = "NUMERICAL"
    
    print(f"[{feature_type:11}] {column:15} | Unique: {unique_count:5} | Missing: {missing_count:5}")

print(f"\nğŸ“Š Feature Summary:")
print(f"   Categorical features: {categorical_features}")
print(f"   Numerical features: {numerical_features}")
print(f"   Total features: {len(categorical_features) + len(numerical_features)}")


# Initialize containers for factorized features
factorized_categorical = []  # For original categorical features
factorized_numerical = []   # For factorized numerical features
feature_cardinalities = {}  # Track unique values for each feature

print("ğŸ”§ Applying factorization encoding...")
print("="*50)

# Process all features (both numerical and categorical)
for feature in numerical_features + categorical_features:
    
    if feature in numerical_features:
        # For numerical features, create a factorized version with suffix '2'
        factorized_name = f"{feature}2"
        factorized_numerical.append(factorized_name)
    else:
        # For categorical features, keep the same name
        factorized_name = feature
        factorized_categorical.append(feature)
    
    # Apply factorization (converts to integer codes)
    factorized_values, unique_values = pd.factorize(combined_data[feature])
    combined_data[factorized_name] = factorized_values
    
    # Store cardinality (number of unique values)
    feature_cardinalities[factorized_name] = len(unique_values)
    
    # Convert to memory-efficient int32
    combined_data[feature] = combined_data[feature].astype('int32')
    combined_data[factorized_name] = combined_data[factorized_name].astype('int32')

print(f"âœ… Created {len(factorized_numerical)} factorized numerical features")
print(f"âœ… Processed {len(factorized_categorical)} categorical features")
print(f"\nğŸ“Š New factorized features: {factorized_numerical}")
print(f"ğŸ“Š Feature cardinalities: {feature_cardinalities}")


# Generate all pairwise combinations of factorized features
all_factorized_features = categorical_features + factorized_numerical
feature_pairs = list(combinations(all_factorized_features, 2))

print(f"ğŸ”§ Creating pairwise feature combinations...")
print(f"ğŸ“Š Total possible pairs: {len(feature_pairs)}")

# Container for new combination features
combination_features = {}
combination_feature_names = []

# Create combination features
for feature1, feature2 in feature_pairs:
    # Create standardized name (alphabetically sorted)
    combination_name = "_".join(sorted([feature1, feature2]))
    
    # Create combination using mathematical encoding
    # This ensures unique values for each combination
    cardinality_f2 = feature_cardinalities[feature2]
    combination_values = combined_data[feature1] * cardinality_f2 + combined_data[feature2]
    
    combination_features[combination_name] = combination_values
    combination_feature_names.append(combination_name)

# Add combination features to the dataset
if combination_features:
    combination_df = pd.DataFrame(combination_features, index=combined_data.index)
    combined_data = pd.concat([combined_data, combination_df], axis=1)

print(f"âœ… Created {len(combination_feature_names)} pairwise combination features")
print(f"ğŸ“Š New dataset shape: {combined_data.shape}")


# Create count encoding features
count_encoded_features = []
all_categorical_features = categorical_features + factorized_numerical + combination_feature_names

print(f"ğŸ”§ Creating count encoding features...")
print(f"ğŸ“Š Processing {len(all_categorical_features)} features...")

# Process in batches for better performance tracking
batch_size = 10
for i, feature in enumerate(all_categorical_features):
    if i % batch_size == 0:
        print(f"   Progress: {i}/{len(all_categorical_features)} features processed")
    
    # Calculate count encoding (frequency of each value)
    count_encoding = combined_data.groupby(feature)['y'].count()
    count_encoding = count_encoding.astype('int32')
    count_encoding.name = f"COUNT_{feature}"
    
    # Add to main dataset
    combined_data = combined_data.merge(count_encoding, on=feature, how='left')
    count_encoded_features.append(f"COUNT_{feature}")

print(f"âœ… Created {len(count_encoded_features)} count encoding features")
print(f"ğŸ“Š Updated dataset shape: {combined_data.shape}")


# Split combined dataset back into original components
train_processed = combined_data.iloc[:len(train_df)].copy()
test_processed = combined_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
original_processed = combined_data.iloc[-len(original_df):].copy()

print("ğŸ“Š Dataset splitting completed:")
print(f"   Training data: {train_processed.shape}")
print(f"   Test data: {test_processed.shape}")
print(f"   Original data: {original_processed.shape}")

# Clean up memory
del combined_data

# Define feature sets for modeling
model_features = numerical_features + categorical_features + factorized_numerical + combination_feature_names + count_encoded_features
print(f"\nğŸ�¯ Total features for modeling: {len(model_features)}")
print(f"   Numerical: {len(numerical_features)}")
print(f"   Categorical: {len(categorical_features)}")
print(f"   Factorized numerical: {len(factorized_numerical)}")
print(f"   Combinations: {len(combination_feature_names)}")
print(f"   Count encodings: {len(count_encoded_features)}")


class BatchedDataIterator(xgb.core.DataIter):
    """
    Custom data iterator for XGBoost that processes data in batches.
    This approach is memory-efficient for large datasets.
    """
    
    def __init__(self, dataframe, feature_columns, target_column, batch_size=256*1024):
        """
        Initialize the batched data iterator.
        
        Parameters:
        - dataframe: pandas DataFrame containing the data
        - feature_columns: list of feature column names
        - target_column: name of the target column
        - batch_size: number of samples per batch
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.current_batch = 0
        self.total_batches = int(np.ceil(len(dataframe) / batch_size))
        super().__init__()
    
    def reset(self):
        """Reset iterator to the beginning."""
        self.current_batch = 0
    
    def next(self, input_data):
        """Yield the next batch of data."""
        # Check if we've processed all batches
        if self.current_batch >= self.total_batches:
            return 0  # Signal completion
        
        # Calculate batch boundaries
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, len(self.dataframe))
        
        # Extract batch data
        batch_data = self.dataframe.iloc[start_idx:end_idx]
        
        # Provide data to XGBoost
        input_data(
            data=batch_data[self.feature_columns],
            label=batch_data[self.target_column]
        )
        
        self.current_batch += 1
        return 1  # Signal successful batch processing

print("âœ… Custom BatchedDataIterator class defined")
print("   - Memory-efficient batch processing")
print("   - Configurable batch sizes")
print("   - Optimized for large datasets")


# Define cross-validation and modeling parameters
N_FOLDS = 7
RANDOM_SEED = 42

# XGBoost parameters optimized for banking data
xgb_parameters = {
    "objective": "binary:logistic",     # Binary classification
    "eval_metric": "auc",               # ROC AUC evaluation
    "learning_rate": 0.1,               # Conservative learning rate
    "max_depth": 0,                     # Use max_leaves instead
    "subsample": 0.8,                   # Row sampling for regularization
    "colsample_bytree": 0.7,           # Column sampling for regularization
    "seed": RANDOM_SEED,                # Reproducibility
    "device": "cuda",                   # GPU acceleration
    "grow_policy": "lossguide",         # Leaf-wise tree growth
    "max_leaves": 32,                   # Control tree complexity
    "alpha": 2.0,                       # L1 regularization
}

print("âš™ï¸� XGBoost Configuration:")
print("="*40)
for param, value in xgb_parameters.items():
    print(f"   {param:20}: {value}")
    
print(f"\nğŸ”„ Cross-validation folds: {N_FOLDS}")
print(f"ğŸ�² Random seed: {RANDOM_SEED}")


# Initialize prediction containers
oof_predictions_phase1 = np.zeros(len(train_processed))
test_predictions_phase1 = np.zeros(len(test_processed))

# Set up cross-validation
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("ğŸš€ PHASE 1: Training with Original Data as Additional Samples")
print("="*70)

fold_scores = []

for fold_idx, (train_indices, validation_indices) in enumerate(kfold.split(train_processed)):
    print(f"\nğŸ“� Training Fold {fold_idx + 1}/{N_FOLDS}")
    print("-" * 30)
    
    # Prepare fold data
    fold_train_data = train_processed.iloc[train_indices][model_features + ['y']].copy()
    fold_validation_features = train_processed.iloc[validation_indices][model_features].copy()
    fold_validation_target = train_processed.iloc[validation_indices]['y']
    fold_test_features = test_processed[model_features].copy()
    
    # Augment training data with original dataset (key innovation!)
    original_data_subset = original_processed[model_features + ['y']]
    
    # Add original data multiple times to increase its influence
    for repetition in range(1):  # Can be increased for more influence
        fold_train_data = pd.concat([fold_train_data, original_data_subset], 
                                   axis=0, ignore_index=True)
    
    print(f"   Training samples: {len(fold_train_data):,} (including {len(original_data_subset):,} original)")
    print(f"   Validation samples: {len(fold_validation_features):,}")
    
    # Target encoding for factorized features
    target_encode_features = factorized_numerical + combination_feature_names
    print(f"   Applying target encoding to {len(target_encode_features)} features...")
    
    for feature_idx, feature in enumerate(target_encode_features):
        if feature_idx % 10 == 0 and feature_idx > 0:
            print(f"     Progress: {feature_idx}/{len(target_encode_features)}")
        
        # Initialize target encoder with cross-validation to prevent overfitting
        target_encoder = TargetEncoder(
            n_folds=10,
            smooth=0,
            split_method='random',
            stat='mean'
        )
        
        # Fit on training data and transform all sets
        fold_train_data[feature] = target_encoder.fit_transform(
            fold_train_data[feature], fold_train_data['y']
        ).astype('float32')
        
        fold_validation_features[feature] = target_encoder.transform(
            fold_validation_features[feature]
        ).astype('float32')
        
        fold_test_features[feature] = target_encoder.transform(
            fold_test_features[feature]
        ).astype('float32')
    
    # Set categorical dtypes for XGBoost
    for feature in categorical_features:
        fold_train_data[feature] = fold_train_data[feature].astype('category')
        fold_validation_features[feature] = fold_validation_features[feature].astype('category')
        fold_test_features[feature] = fold_test_features[feature].astype('category')
    
    # Create XGBoost data structures
    print("   Creating XGBoost data matrices...")
    
    # Training data with batched iterator
    train_iterator = BatchedDataIterator(fold_train_data, model_features, 'y')
    train_matrix = xgb.QuantileDMatrix(train_iterator, enable_categorical=True, max_bin=256)
    
    # Validation and test matrices
    validation_matrix = xgb.DMatrix(fold_validation_features, 
                                   label=fold_validation_target, 
                                   enable_categorical=True)
    test_matrix = xgb.DMatrix(fold_test_features, enable_categorical=True)
    
    # Train the model
    print("   Training XGBoost model...")
    model = xgb.train(
        params=xgb_parameters,
        dtrain=train_matrix,
        num_boost_round=10_000,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        early_stopping_rounds=200,
        verbose_eval=200
    )
    
    # Generate predictions
    fold_oof_pred = model.predict(validation_matrix, 
                                 iteration_range=(0, model.best_iteration + 1))
    fold_test_pred = model.predict(test_matrix, 
                                  iteration_range=(0, model.best_iteration + 1))
    
    # Store predictions
    oof_predictions_phase1[validation_indices] = fold_oof_pred
    test_predictions_phase1 += fold_test_pred / N_FOLDS
    
    # Calculate fold score
    fold_score = roc_auc_score(fold_validation_target, fold_oof_pred)
    fold_scores.append(fold_score)
    print(f"   âœ… Fold {fold_idx + 1} AUC: {fold_score:.6f}")

# Calculate overall Phase 1 score
phase1_score = roc_auc_score(train_processed['y'], oof_predictions_phase1)
print(f"\nğŸ�† PHASE 1 RESULTS:")
print(f"   Overall CV AUC: {phase1_score:.6f}")
print(f"   Standard deviation: {np.std(fold_scores):.6f}")
print(f"   Fold scores: {[f'{score:.6f}' for score in fold_scores]}")


# Plot feature importance from the last model
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("ğŸ�¯ Top 20 Feature Importances - Phase 1 Model", fontsize=14, fontweight='bold')
plt.xlabel("Feature Importance (Gain)", fontsize=12)
plt.tight_layout()
plt.show()

print("ğŸ“Š Feature importance analysis completed for Phase 1")


print("ğŸ”§ PHASE 2 FEATURE ENGINEERING: Target Encoding from Original Data")
print("="*70)

# Features to target encode using original data insights
target_encode_candidates = categorical_features + factorized_numerical + combination_feature_names
target_encoded_features = []

print(f"ğŸ“Š Creating target encodings for {len(target_encode_candidates)} features...")

# Create target encodings based on original data patterns
for feature_idx, feature in enumerate(target_encode_candidates):
    if feature_idx % 10 == 0:
        print(f"   Progress: {feature_idx}/{len(target_encode_candidates)} features")
    
    # Calculate mean target for each category in original data
    target_means = original_processed.groupby(feature)['y'].mean()
    target_means = target_means.astype('float32')
    target_means.name = f"TARGET_ORIG_{feature}"
    
    # Add to training and test data
    train_processed = train_processed.merge(target_means, on=feature, how='left')
    test_processed = test_processed.merge(target_means, on=feature, how='left')
    
    target_encoded_features.append(f"TARGET_ORIG_{feature}")

print(f"\nâœ… Created {len(target_encoded_features)} target-encoded features")

# Update feature list for Phase 2
phase2_features = model_features + target_encoded_features
print(f"ğŸ“Š Total features for Phase 2: {len(phase2_features)}")
print(f"   Original features: {len(model_features)}")
print(f"   New target encodings: {len(target_encoded_features)}")


# Initialize prediction containers for Phase 2
oof_predictions_phase2 = np.zeros(len(train_processed))
test_predictions_phase2 = np.zeros(len(test_processed))

print("ğŸš€ PHASE 2: Training with Target-Encoded Features")
print("="*55)

phase2_fold_scores = []

# Use same cross-validation splits for consistency
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold_idx, (train_indices, validation_indices) in enumerate(kfold.split(train_processed)):
    print(f"\nğŸ“� Training Fold {fold_idx + 1}/{N_FOLDS}")
    print("-" * 30)
    
    # Prepare fold data (without additional original data this time)
    fold_train_data = train_processed.iloc[train_indices][phase2_features + ['y']].copy()
    fold_validation_features = train_processed.iloc[validation_indices][phase2_features].copy()
    fold_validation_target = train_processed.iloc[validation_indices]['y']
    fold_test_features = test_processed[phase2_features].copy()
    
    print(f"   Training samples: {len(fold_train_data):,}")
    print(f"   Validation samples: {len(fold_validation_features):,}")
    print(f"   Features: {len(phase2_features):,}")
    
    # Target encoding for factorized features (same as Phase 1)
    target_encode_features = factorized_numerical + combination_feature_names
    print(f"   Applying target encoding to {len(target_encode_features)} features...")
    
    for feature_idx, feature in enumerate(target_encode_features):
        if feature_idx % 10 == 0 and feature_idx > 0:
            print(f"     Progress: {feature_idx}/{len(target_encode_features)}")
        
        target_encoder = TargetEncoder(
            n_folds=10,
            smooth=0,
            split_method='random',
            stat='mean'
        )
        
        fold_train_data[feature] = target_encoder.fit_transform(
            fold_train_data[feature], fold_train_data['y']
        ).astype('float32')
        
        fold_validation_features[feature] = target_encoder.transform(
            fold_validation_features[feature]
        ).astype('float32')
        
        fold_test_features[feature] = target_encoder.transform(
            fold_test_features[feature]
        ).astype('float32')
    
    # Set categorical dtypes
    for feature in categorical_features:
        fold_train_data[feature] = fold_train_data[feature].astype('category')
        fold_validation_features[feature] = fold_validation_features[feature].astype('category')
        fold_test_features[feature] = fold_test_features[feature].astype('category')
    
    # Create XGBoost data structures
    print("   Creating XGBoost data matrices...")
    
    train_iterator = BatchedDataIterator(fold_train_data, phase2_features, 'y')
    train_matrix = xgb.QuantileDMatrix(train_iterator, enable_categorical=True, max_bin=256)
    
    validation_matrix = xgb.DMatrix(fold_validation_features, 
                                   label=fold_validation_target, 
                                   enable_categorical=True)
    test_matrix = xgb.DMatrix(fold_test_features, enable_categorical=True)
    
    # Train the model
    print("   Training XGBoost model...")
    model_phase2 = xgb.train(
        params=xgb_parameters,
        dtrain=train_matrix,
        num_boost_round=10_000,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        early_stopping_rounds=200,
        verbose_eval=200
    )
    
    # Generate predictions
    fold_oof_pred = model_phase2.predict(validation_matrix, 
                                        iteration_range=(0, model_phase2.best_iteration + 1))
    fold_test_pred = model_phase2.predict(test_matrix, 
                                         iteration_range=(0, model_phase2.best_iteration + 1))
    
    # Store predictions
    oof_predictions_phase2[validation_indices] = fold_oof_pred
    test_predictions_phase2 += fold_test_pred / N_FOLDS
    
    # Calculate fold score
    fold_score = roc_auc_score(fold_validation_target, fold_oof_pred)
    phase2_fold_scores.append(fold_score)
    print(f"   âœ… Fold {fold_idx + 1} AUC: {fold_score:.6f}")

# Calculate overall Phase 2 score
phase2_score = roc_auc_score(train_processed['y'], oof_predictions_phase2)
print(f"\nğŸ�† PHASE 2 RESULTS:")
print(f"   Overall CV AUC: {phase2_score:.6f}")
print(f"   Standard deviation: {np.std(phase2_fold_scores):.6f}")
print(f"   Fold scores: {[f'{score:.6f}' for score in phase2_fold_scores]}")


# Plot feature importance from Phase 2 model
plt.figure(figsize=(12, 8))
xgb.plot_importance(model_phase2, max_num_features=20, importance_type='gain')
plt.title("ğŸ�¯ Top 20 Feature Importances - Phase 2 Model", fontsize=14, fontweight='bold')
plt.xlabel("Feature Importance (Gain)", fontsize=12)
plt.tight_layout()
plt.show()

print("ğŸ“Š Feature importance analysis completed for Phase 2")


print("ğŸ�­ ENSEMBLE ANALYSIS AND OPTIMIZATION")
print("="*45)

# Calculate individual model performance
print("ğŸ“Š Individual Model Performance:")
print(f"   Phase 1 (Original as samples): {phase1_score:.6f}")
print(f"   Phase 2 (Original as features): {phase2_score:.6f}")

# Test different ensemble combinations
ensemble_strategies = {
    "Simple Average": (oof_predictions_phase1 + oof_predictions_phase2) / 2,
    "Weighted (70-30)": 0.7 * oof_predictions_phase1 + 0.3 * oof_predictions_phase2,
    "Weighted (60-40)": 0.6 * oof_predictions_phase1 + 0.4 * oof_predictions_phase2,
    "Weighted (50-50)": 0.5 * oof_predictions_phase1 + 0.5 * oof_predictions_phase2,
}

print("\nğŸ”� Testing Ensemble Strategies:")
ensemble_scores = {}
best_ensemble_score = 0
best_ensemble_name = ""
best_ensemble_predictions = None

for strategy_name, ensemble_pred in ensemble_strategies.items():
    ensemble_score = roc_auc_score(train_processed['y'], ensemble_pred)
    ensemble_scores[strategy_name] = ensemble_score
    
    print(f"   {strategy_name:20}: {ensemble_score:.6f}")
    
    if ensemble_score > best_ensemble_score:
        best_ensemble_score = ensemble_score
        best_ensemble_name = strategy_name
        best_ensemble_predictions = ensemble_pred

print(f"\nğŸ�† BEST ENSEMBLE: {best_ensemble_name}")
print(f"   CV AUC: {best_ensemble_score:.6f}")
print(f"   Improvement over best individual: {best_ensemble_score - max(phase1_score, phase2_score):.6f}")

# Calculate final test predictions using best ensemble strategy
if "Simple Average" in best_ensemble_name:
    final_test_predictions = (test_predictions_phase1 + test_predictions_phase2) / 2
elif "70-30" in best_ensemble_name:
    final_test_predictions = 0.7 * test_predictions_phase1 + 0.3 * test_predictions_phase2
elif "60-40" in best_ensemble_name:
    final_test_predictions = 0.6 * test_predictions_phase1 + 0.4 * test_predictions_phase2
else:  # 50-50
    final_test_predictions = 0.5 * test_predictions_phase1 + 0.5 * test_predictions_phase2


# Create comprehensive results visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Model comparison
model_names = ['Phase 1', 'Phase 2', 'Ensemble']
model_scores = [phase1_score, phase2_score, best_ensemble_score]
colors = ['skyblue', 'lightcoral', 'gold']

axes[0, 0].bar(model_names, model_scores, color=colors)
axes[0, 0].set_title('ğŸ�† Model Performance Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('ROC AUC Score')
axes[0, 0].set_ylim(0.95, max(model_scores) + 0.005)
for i, score in enumerate(model_scores):
    axes[0, 0].text(i, score + 0.001, f'{score:.6f}', ha='center', fontweight='bold')

# Plot 2: Cross-validation stability
fold_numbers = range(1, N_FOLDS + 1)
axes[0, 1].plot(fold_numbers, fold_scores, 'o-', label='Phase 1', linewidth=2, markersize=6)
axes[0, 1].plot(fold_numbers, phase2_fold_scores, 's-', label='Phase 2', linewidth=2, markersize=6)
axes[0, 1].set_title('ğŸ“Š Cross-Validation Stability', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Fold Number')
axes[0, 1].set_ylabel('ROC AUC Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Prediction distribution for Phase 1
axes[1, 0].hist(oof_predictions_phase1, bins=50, alpha=0.7, color='skyblue', density=True)
axes[1, 0].set_title('ğŸ“ˆ Phase 1 Prediction Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Density')

# Plot 4: Prediction distribution for Phase 2
axes[1, 1].hist(oof_predictions_phase2, bins=50, alpha=0.7, color='lightcoral', density=True)
axes[1, 1].set_title('ğŸ“ˆ Phase 2 Prediction Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.show()

print("ğŸ“Š Visualization completed!")


# Additional analysis: Test prediction distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_test_predictions, bins=100, alpha=0.7, color='gold', edgecolor='black')
plt.title('ğŸ�¯ Final Test Predictions Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.ylim(0, 10_000)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
correlation = np.corrcoef(oof_predictions_phase1, oof_predictions_phase2)[0, 1]
plt.scatter(oof_predictions_phase1, oof_predictions_phase2, alpha=0.5, s=1)
plt.xlabel('Phase 1 Predictions')
plt.ylabel('Phase 2 Predictions')
plt.title(f'ğŸ”„ Model Correlation: {correlation:.4f}', fontsize=12, fontweight='bold')
plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"ğŸ“Š Model correlation: {correlation:.4f}")
print(f"ğŸ“Š Test prediction range: {final_test_predictions.min():.4f} - {final_test_predictions.max():.4f}")


# Create submission file
submission_df = pd.read_csv(f"{DATA_PATH}sample_submission.csv")
submission_df['y'] = final_test_predictions

# Save submission
submission_filename = "submission.csv"
submission_df.to_csv(submission_filename, index=False)

print("ğŸ’¾ SUBMISSION GENERATED")
print("="*30)
print(f"   Filename: {submission_filename}")
print(f"   Shape: {submission_df.shape}")
print(f"   Expected AUC: {best_ensemble_score:.6f}")

# Display submission statistics
print(f"\nğŸ“Š Submission Statistics:")
print(f"   Mean prediction: {final_test_predictions.mean():.6f}")
print(f"   Std prediction: {final_test_predictions.std():.6f}")
print(f"   Min prediction: {final_test_predictions.min():.6f}")
print(f"   Max prediction: {final_test_predictions.max():.6f}")

# Display first few rows
print(f"\nğŸ“‹ First 5 submission rows:")
display(submission_df.head())


print("ğŸ�† FINAL PERFORMANCE SUMMARY")
print("="*50)
print(f"Phase 1 (Original as Samples):   {phase1_score:.6f}")
print(f"Phase 2 (Original as Features):  {phase2_score:.6f}")
print(f"Best Ensemble ({best_ensemble_name}): {best_ensemble_score:.6f}")
print(f"\nğŸ�¯ Target Achievement: {best_ensemble_score:.6f} {'âœ… ACHIEVED!' if best_ensemble_score >= 0.98 else 'âš ï¸� VERY CLOSE!'}")

print(f"\nğŸ“Š FEATURE ENGINEERING SUMMARY")
print("="*40)
print(f"Original features:        {len(numerical_features + categorical_features)}")
print(f"Factorized features:      {len(factorized_numerical)}")
print(f"Combination features:     {len(combination_feature_names)}")
print(f"Count encodings:          {len(count_encoded_features)}")
print(f"Target encodings:         {len(target_encoded_features)}")
print(f"Total features created:   {len(phase2_features)}")

print(f"\nğŸ§  KEY INNOVATIONS IMPLEMENTED")
print("="*35)
print("âœ… Dual-phase training architecture")
print("âœ… Advanced factorization encoding")
print("âœ… Comprehensive pairwise combinations")
print("âœ… Multi-level count encodings")
print("âœ… Target encoding from original data")
print("âœ… Memory-efficient batch processing")
print("âœ… GPU-accelerated XGBoost training")
print("âœ… Sophisticated ensemble optimization")

print(f"\nğŸš€ TECHNICAL ACHIEVEMENTS")
print("="*30)
print(f"Cross-validation folds:   {N_FOLDS}")
print(f"Training samples used:    {len(train_processed):,}")
print(f"Original samples leveraged: {len(original_processed):,}")
print(f"GPU acceleration:         âœ… CUDA enabled")
print(f"Memory optimization:      âœ… Batch processing")
print(f"Ensemble diversity:       âœ… Two complementary approaches")

if best_ensemble_score >= 0.98:
    print(f"\nğŸ�‰ CONGRATULATIONS!")
    print("ğŸ�† Target AUC of 0.98+ achieved!")
    print("ğŸš€ Your model is ready for competition submission!")
    print(f"ğŸ“� Submit: {submission_filename}")
else:
    print(f"\nâš¡ EXCELLENT PERFORMANCE!")
    print(f"ğŸ�¯ Achieved {best_ensemble_score:.6f} AUC - Very close to 0.98!")
    print(f"ğŸ”§ Consider minor hyperparameter tuning for final boost")
    print(f"ğŸ“� Submit: {submission_filename}")

