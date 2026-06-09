import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import welch
import pickle
import gc

warnings.filterwarnings('ignore')

# Configuration
TEST_PATH = "/kaggle/working/"
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH = "/kaggle/input/preprocessing/preprocessed/eeg"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train.csv")
MODEL_OUTPUT_PATH = os.path.join(TEST_PATH, "models_svm_improved_v2")
FEATURE_CACHE_PATH = os.path.join(TEST_PATH, "feature_cache")

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FEATURE_CACHE_PATH, exist_ok=True)

CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
N_CLASSES = len(CLASSES)
BATCH_SIZE = 300  # Reduced batch size for better memory management
N_FEATURES = 500  # Reduced feature count for better generalization

print("Starting Enhanced SVM EEG Classification with Advanced Feature Engineering")

# KL Divergence function
def kl_divergence_numpy(y_true, y_pred_proba, epsilon=1e-7):
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0)
    kl_div = np.sum(y_true * np.log(y_true / y_pred_proba), axis=1)
    return np.mean(kl_div)

# Enhanced feature extraction with more informative features
def extract_enhanced_features(eeg_data):
    """Extract comprehensive features from EEG data including frequency domain"""
    features = []
    
    # Ensure data is 2D (channels x time)
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    n_channels, n_samples = eeg_data.shape
    
    # 1. Time domain features for each channel
    for ch in range(n_channels):
        channel_data = eeg_data[ch, :]
        
        # Basic statistical features
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.var(channel_data),
            np.median(channel_data),
            np.min(channel_data),
            np.max(channel_data),
            np.ptp(channel_data),  # Peak-to-peak
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.percentile(channel_data, 10),
            np.percentile(channel_data, 90),
        ])
        
        # Advanced statistical measures
        features.extend([
            skew(channel_data),
            kurtosis(channel_data),
            np.sqrt(np.mean(channel_data ** 2)),  # RMS
            np.mean(np.abs(channel_data)),  # Mean absolute value
            np.sum(np.abs(np.diff(channel_data))),  # Total variation
            entropy(np.abs(channel_data) + 1e-10),
        ])
        
        # Signal energy features
        features.extend([
            np.sum(channel_data ** 2),  # Total energy
            np.mean(channel_data ** 2),  # Mean power
            np.std(channel_data ** 2),   # Power variation
        ])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
        features.append(zero_crossings / len(channel_data))
        
        # Spectral features (frequency domain)
        try:
            # Use Welch's method for power spectral density
            freqs, psd = welch(channel_data, fs=200, nperseg=min(256, len(channel_data)//4))
            
            # Frequency band powers
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])    # Delta
            theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])      # Theta
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])     # Alpha
            beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])     # Beta
            gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 50)])    # Gamma
            
            total_power = np.sum(psd)
            
            if total_power > 1e-10:
                features.extend([
                    delta_power / total_power,
                    theta_power / total_power,
                    alpha_power / total_power,
                    beta_power / total_power,
                    gamma_power / total_power,
                ])
            else:
                features.extend([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal distribution if no power
            
            # Spectral centroid and bandwidth
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-10))
            
            features.extend([spectral_centroid, spectral_bandwidth])
            
        except Exception:
            # If spectral analysis fails, add zeros
            features.extend([0.2, 0.2, 0.2, 0.2, 0.2, 0, 0])
    
    # 2. Cross-channel features (connectivity measures)
    if n_channels > 1:
        # Global statistics
        features.extend([
            np.mean(eeg_data),
            np.std(eeg_data),
            np.var(eeg_data),
            np.median(eeg_data),
            np.min(eeg_data),
            np.max(eeg_data),
        ])
        
        # Correlation-based connectivity
        try:
            corr_matrix = np.corrcoef(eeg_data)
            # Remove diagonal elements and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            correlations = corr_matrix[mask]
            
            features.extend([
                np.mean(correlations),
                np.std(correlations),
                np.max(correlations),
                np.min(correlations),
                np.median(correlations),
            ])
        except Exception:
            features.extend([0, 0, 0, 0, 0])
        
        # Channel variance ratios
        channel_vars = np.var(eeg_data, axis=1)
        if np.sum(channel_vars) > 1e-10:
            var_ratios = channel_vars / np.sum(channel_vars)
            features.extend([
                np.max(var_ratios),
                np.min(var_ratios),
                np.std(var_ratios),
            ])
        else:
            features.extend([1/n_channels, 1/n_channels, 0])
    else:
        # Add zeros for single channel
        features.extend([0] * 14)
    
    return np.array(features, dtype=np.float32)

def process_eeg_file_enhanced(eeg_id):
    """Enhanced EEG file processing with better error handling"""
    eeg_path = os.path.join(PREPROCESSED_PATH, f"{eeg_id}.npy")
    try:
        eeg_data = np.load(eeg_path).astype(np.float32)
        
        # Handle NaN/Inf values more robustly
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            for ch in range(eeg_data.shape[0]):
                channel = eeg_data[ch, :]
                mask = np.isnan(channel) | np.isinf(channel)
                if np.any(mask):
                    # Use median of valid values for imputation
                    valid_values = channel[~mask]
                    if len(valid_values) > 0:
                        channel[mask] = np.median(valid_values)
                    else:
                        channel[mask] = 0.0
        
        # Robust standardization (channel-wise z-score with outlier handling)
        for ch in range(eeg_data.shape[0]):
            channel = eeg_data[ch, :]
            # Use robust statistics
            median_val = np.median(channel)
            mad = np.median(np.abs(channel - median_val))  # Median Absolute Deviation
            if mad > 1e-7:
                eeg_data[ch, :] = (channel - median_val) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
            else:
                eeg_data[ch, :] = channel - median_val
        
        # Extract enhanced features
        features = extract_enhanced_features(eeg_data)
        
        # Additional validation
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"Warning: Invalid features detected for {eeg_id}, replacing with zeros")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    except Exception as e:
        print(f"Error loading EEG data for {eeg_id}: {e}")
        return None

class EnhancedBatchDataGenerator:
    """Enhanced generator with better memory management"""
    def __init__(self, eeg_ids, labels, batch_size=BATCH_SIZE):
        self.eeg_ids = eeg_ids
        self.labels = labels
        self.batch_size = batch_size
        self.failed_count = 0
        
    def __iter__(self):
        for i in range(0, len(self.eeg_ids), self.batch_size):
            batch_ids = self.eeg_ids[i:i+self.batch_size]
            batch_labels = self.labels[i:i+self.batch_size]
            
            batch_features = []
            batch_y = []
            
            for eeg_id, label in zip(batch_ids, batch_labels):
                features = process_eeg_file_enhanced(eeg_id)
                if features is not None:
                    batch_features.append(features)
                    batch_y.append(label)
                else:
                    self.failed_count += 1
            
            if batch_features:
                yield np.array(batch_features, dtype=np.float32), np.array(batch_y)
            
            # Force garbage collection
            gc.collect()

# Load and preprocess training labels
try:
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(train_df)} annotations with {len(train_df['eeg_id'].unique())} unique EEG IDs")
except Exception as e:
    print(f"Error loading training labels: {e}")
    exit()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
train_df['label'] = label_encoder.transform(train_df['expert_consensus'])

# Print class distribution
print("Class distribution:")
class_dist = train_df['expert_consensus'].value_counts()
for cls, count in class_dist.items():
    print(f"  {cls}: {count} ({count/len(train_df)*100:.1f}%)")

# Load data with success filtering
print("Loading data...")
success_file_path = os.path.join(os.path.dirname(PREPROCESSED_PATH), "success.csv")
if os.path.exists(success_file_path):
    try:
        success_df = pd.read_csv(success_file_path)
        success_ids = set(success_df['eeg_id'].astype(str).tolist())
        print(f"Found success file with {len(success_ids)} successful preprocessing entries")
    except Exception as e:
        print(f"Error loading success file: {e}")
        success_ids = set(train_df['eeg_id'].astype(str).tolist())
else:
    print("Success file not found, using all available EEG IDs")
    success_ids = set(train_df['eeg_id'].astype(str).tolist())

# Filter samples based on success_ids
valid_samples = train_df[train_df['eeg_id'].astype(str).isin(success_ids)]
eeg_ids = valid_samples['eeg_id'].astype(str).tolist()
labels = valid_samples['label'].values
print(f"Processing {len(eeg_ids)} valid samples")

# Stratified train-validation split
train_ids, val_ids, train_labels, val_labels = train_test_split(
    eeg_ids, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

# Process training data in batches
print("Processing training data in batches...")
train_generator = EnhancedBatchDataGenerator(train_ids, train_labels)

all_train_features = []
all_train_labels = []

for batch_X, batch_y in tqdm(train_generator, desc="Processing training batches"):
    all_train_features.append(batch_X)
    all_train_labels.append(batch_y)

# Concatenate all batches
X_train = np.vstack(all_train_features)
y_train = np.hstack(all_train_labels)
print(f"Training feature matrix shape: {X_train.shape}")
print(f"Failed training samples: {train_generator.failed_count}")

# Clear intermediate data
del all_train_features, all_train_labels
gc.collect()

# Process validation data
print("Processing validation data in batches...")
val_generator = EnhancedBatchDataGenerator(val_ids, val_labels)

all_val_features = []
all_val_labels = []

for batch_X, batch_y in tqdm(val_generator, desc="Processing validation batches"):
    all_val_features.append(batch_X)
    all_val_labels.append(batch_y)

X_val = np.vstack(all_val_features)
y_val = np.hstack(all_val_labels)
print(f"Validation feature matrix shape: {X_val.shape}")
print(f"Failed validation samples: {val_generator.failed_count}")

# Clear intermediate data
del all_val_features, all_val_labels
gc.collect()

# Enhanced Preprocessing Pipeline
print("Enhanced preprocessing pipeline...")

# 1. Remove features with zero variance
print("Removing zero-variance features...")
variance_selector = VarianceThreshold(threshold=1e-6)
X_train_var = variance_selector.fit_transform(X_train)
X_val_var = variance_selector.transform(X_val)
print(f"Features after variance filtering: {X_train_var.shape[1]}")

# 2. Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_var)
X_val_imputed = imputer.transform(X_val_var)

# 3. Robust scaling (less sensitive to outliers)
print("Applying robust scaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)

# 4. Feature selection using multiple methods
print(f"Selecting top {N_FEATURES} features using hybrid approach...")

# Method 1: Statistical selection
stat_selector = SelectKBest(f_classif, k=min(1000, X_train_scaled.shape[1]))
X_train_stat = stat_selector.fit_transform(X_train_scaled, y_train)
X_val_stat = stat_selector.transform(X_val_scaled)

# Method 2: Recursive feature elimination with Random Forest
if X_train_stat.shape[1] > N_FEATURES:
    print("Applying recursive feature elimination...")
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe_selector = RFE(rf_selector, n_features_to_select=N_FEATURES, step=0.1)
    X_train_selected = rfe_selector.fit_transform(X_train_stat, y_train)
    X_val_selected = rfe_selector.transform(X_val_stat)
else:
    X_train_selected = X_train_stat
    X_val_selected = X_val_stat
    rfe_selector = None

print(f"Final selected features: {X_train_selected.shape[1]}")

# Clear intermediate arrays
del X_train, X_val, X_train_var, X_val_var, X_train_imputed, X_val_imputed
del X_train_scaled, X_val_scaled, X_train_stat, X_val_stat
gc.collect()

# Calculate balanced class weights
print("Calculating class weights...")
class_counts = pd.Series(y_train).value_counts().sort_index().values
class_weights = len(y_train) / (N_CLASSES * class_counts)
class_weight_dict = {i: class_weights[i] for i in range(N_CLASSES)}

print("Class weights:", {CLASSES[i]: f"{class_weights[i]:.3f}" for i in range(N_CLASSES)})

# Enhanced SVM with hyperparameter tuning
print("Training enhanced SVM with hyperparameter optimization...")

# Define parameter grid for GridSearch
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

# Use StratifiedKFold for cross-validation
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Base SVM
base_svm = SVC(
    class_weight=class_weight_dict,
    random_state=42,
    probability=True,  # Enable probability estimates
    max_iter=2000
)

# Grid search with cross-validation
print("Performing grid search...")
grid_search = GridSearchCV(
    base_svm,
    param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_selected, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.6f}")

# Get the best model
best_svm = grid_search.best_estimator_

# Make predictions
print("Making predictions...")
train_pred = best_svm.predict(X_train_selected)
val_pred = best_svm.predict(X_val_selected)

train_pred_proba = best_svm.predict_proba(X_train_selected)
val_pred_proba = best_svm.predict_proba(X_val_selected)

# Calculate metrics
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

# Calculate KL divergence
train_true_one_hot = np.eye(N_CLASSES)[y_train.astype(int)]
val_true_one_hot = np.eye(N_CLASSES)[y_val.astype(int)]

train_kl = kl_divergence_numpy(train_true_one_hot, train_pred_proba)
val_kl = kl_divergence_numpy(val_true_one_hot, val_pred_proba)

print(f"\nEnhanced SVM Results:")
print(f"Train Accuracy: {train_acc:.6f}, Train KL Divergence: {train_kl:.6f}")
print(f"Val Accuracy: {val_acc:.6f}, Val KL Divergence: {val_kl:.6f}")

# Detailed classification report
print("\nDetailed Classification Report (Validation):")
print(classification_report(y_val, val_pred, target_names=CLASSES))

# Class-wise accuracy
print("\nClass-wise Validation Accuracy:")
cm = confusion_matrix(y_val, val_pred)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, (cls, acc) in enumerate(zip(CLASSES, class_accuracies)):
    print(f"  {cls}: {acc:.4f} ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]})")

# Print prediction distributions
print("\nPrediction distribution:")
train_pred_dist = pd.Series(train_pred).value_counts().sort_index()
val_pred_dist = pd.Series(val_pred).value_counts().sort_index()
actual_dist = pd.Series(y_val).value_counts().sort_index()

print("Class distributions:")
for i, cls in enumerate(CLASSES):
    train_count = train_pred_dist.get(i, 0)
    val_count = val_pred_dist.get(i, 0)
    actual_count = actual_dist.get(i, 0)
    print(f"  {cls}: Train_pred={train_count}, Val_pred={val_count}, Val_actual={actual_count}")

# Enhanced Confusion Matrix
plt.figure(figsize=(12, 10))
cm_normalized = confusion_matrix(y_val, val_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Normalized Confusion Matrix (Validation Set) - Enhanced SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'confusion_matrix_normalized.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Feature importance (for RBF kernel, we can't get direct feature importance)
if hasattr(best_svm, 'coef_') and best_svm.coef_ is not None:
    try:
        feature_importance = np.abs(best_svm.coef_).mean(axis=0)
        
        plt.figure(figsize=(12, 8))
        sorted_idx = np.argsort(feature_importance)[-20:]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.title('Top 20 Feature Importances (Linear SVM)')
        plt.xlabel('Importance')
        plt.ylabel('Feature Index')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved!")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

# Save complete model pipeline
model_components = {
    'svm_model': best_svm,
    'variance_selector': variance_selector,
    'imputer': imputer,
    'scaler': scaler,
    'stat_selector': stat_selector,
    'rfe_selector': rfe_selector,
    'label_encoder': label_encoder,
    'grid_search_results': {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
}

model_path = os.path.join(MODEL_OUTPUT_PATH, "enhanced_svm_pipeline.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_components, f)
print(f"Complete model pipeline saved to: {model_path}")

# Enhanced submission file creation
def create_enhanced_submission():
    """Create submission with the complete preprocessing pipeline"""
    test_eeg_path = os.path.join(BASE_PATH, "test_eegs")
    if not os.path.exists(test_eeg_path):
        print("Warning: Test EEG path not found. Creating dummy submission.")
        submission_df = pd.DataFrame({'eeg_id': ['dummy_1', 'dummy_2']})
        for cls in CLASSES:
            submission_df[cls] = 1.0 / N_CLASSES
        submission_path = os.path.join(TEST_PATH, "submission_enhanced_svm.csv")
        submission_df.to_csv(submission_path, index=False)
        return submission_df
    
    test_files = [f.replace(".parquet", "") for f in os.listdir(test_eeg_path) 
                  if f.endswith(".parquet")]
    
    if len(test_files) == 0:
        print("No test files found")
        return None

    submission_df = pd.DataFrame({'eeg_id': test_files})
    for cls in CLASSES:
        submission_df[cls] = 0.0

    print(f"Processing {len(test_files)} test files...")
    predictions_made = 0
    failed_predictions = 0
    
    # Process in batches
    for i in tqdm(range(0, len(test_files), BATCH_SIZE), desc="Test predictions"):
        batch_files = test_files[i:i+BATCH_SIZE]
        batch_features = []
        valid_files = []
        
        for eeg_id in batch_files:
            features = process_eeg_file_enhanced(eeg_id)
            if features is not None:
                batch_features.append(features)
                valid_files.append(eeg_id)
            else:
                # Uniform probability for failed files
                for cls in CLASSES:
                    submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = 1.0 / N_CLASSES
                failed_predictions += 1
        
        if batch_features:
            try:
                batch_X = np.array(batch_features, dtype=np.float32)
                
                # Apply complete preprocessing pipeline
                batch_X = variance_selector.transform(batch_X)
                batch_X = imputer.transform(batch_X)
                batch_X = scaler.transform(batch_X)
                batch_X = stat_selector.transform(batch_X)
                if rfe_selector is not None:
                    batch_X = rfe_selector.transform(batch_X)
                
                batch_probs = best_svm.predict_proba(batch_X)
                
                # Update submission
                for j, eeg_id in enumerate(valid_files):
                    for k, cls in enumerate(CLASSES):
                        submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = batch_probs[j, k]
                    predictions_made += 1
                
            except Exception as e:
                print(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
                for eeg_id in valid_files:
                    for cls in CLASSES:
                        submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = 1.0 / N_CLASSES
                    failed_predictions += 1
        
        gc.collect()

    # Normalize probabilities
    prob_cols = CLASSES
    row_sums = submission_df[prob_cols].sum(axis=1)
    for cls in CLASSES:
        submission_df[cls] = submission_df[cls] / row_sums

    submission_path = os.path.join(TEST_PATH, "submission_enhanced_svm.csv")
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Enhanced submission saved to: {submission_path}")
    print(f"Successful predictions: {predictions_made}/{len(test_files)}")
    print(f"Failed predictions: {failed_predictions}/{len(test_files)}")
    
    return submission_df

# Create submission
print("\n" + "="*60)
print("CREATING ENHANCED SUBMISSION FILE")
print("="*60)

try:
    submission_df = create_enhanced_submission()
    if submission_df is not None:
        print("âœ“ Enhanced submission file created successfully!")
        print(f"Sample entries:\n{submission_df.head()}")
except Exception as e:
    print(f"Error creating submission: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "="*60)
print("ENHANCED SVM TRAINING SUMMARY")
print("="*60)
print(f"Model: {grid_search.best_params_['kernel'].upper()} SVM with optimized hyperparameters")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Training samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Final feature dimensions: {X_train_selected.shape[1]}")
print(f"Cross-validation score: {grid_search.best_score_:.6f}")
print(f"Final train accuracy: {train_acc:.6f}")
print(f"Final validation accuracy: {val_acc:.6f}")
print(f"Final validation KL divergence: {val_kl:.6f}")
print("="*60)

# Save training configuration
config = {
    'model_type': 'Enhanced SVM with GridSearch',
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'classes': CLASSES,
    'n_classes': N_CLASSES,
    'batch_size': BATCH_SIZE,
    'n_features_selected': X_train_selected.shape[1],
    'train_accuracy': float(train_acc),
    'val_accuracy': float(val_acc),
    'val_kl_divergence': float(val_kl),
    'n_train_samples': len(train_ids),
    'n_val_samples': len(val_ids),
    'class_weights': class_weight_dict,
    'preprocessing_steps': [
        'variance_threshold',
        'median_imputation', 
        'robust_scaling',
        'statistical_feature_selection',
        'recursive_feature_elimination'
    ]
}

config_path = os.path.join(MODEL_OUTPUT_PATH, "enhanced_training_config.pkl")
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"Enhanced training configuration saved to: {config_path}")

# Additional diagnostic plots
print("\nGenerating diagnostic plots...")

# 1. Class probability distributions
plt.figure(figsize=(15, 10))
for i, cls in enumerate(CLASSES):
    plt.subplot(2, 3, i+1)
    class_probs = val_pred_proba[:, i]
    plt.hist(class_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'{cls} - Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'class_probability_distributions.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# 2. Prediction confidence analysis
confidence_scores = np.max(val_pred_proba, axis=1)
correct_predictions = (val_pred == y_val)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
         label='Correct', color='green', edgecolor='black')
plt.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
         label='Incorrect', color='red', edgecolor='black')
plt.xlabel('Maximum Probability (Confidence)')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
confidence_bins = np.linspace(0, 1, 11)
bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
bin_accuracies = []

for i in range(len(confidence_bins)-1):
    mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i+1])
    if np.sum(mask) > 0:
        bin_acc = np.mean(correct_predictions[mask])
        bin_accuracies.append(bin_acc)
    else:
        bin_accuracies.append(0)

plt.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
plt.xlabel('Confidence Score')
plt.ylabel('Accuracy')
plt.title('Calibration Plot')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'confidence_analysis.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# 3. Per-class performance analysis
plt.figure(figsize=(12, 8))

# Precision, Recall, F1-Score per class
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_val, val_pred)

x = np.arange(len(CLASSES))
width = 0.25

plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
plt.bar(x, recall, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics')
plt.xticks(x, CLASSES, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'per_class_performance.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# 4. Learning curve analysis (if we had training history)
print("\nPerformance Summary by Class:")
print("-" * 50)
for i, cls in enumerate(CLASSES):
    print(f"{cls:8s}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, "
          f"F1={f1[i]:.3f}, Support={support[i]}")

print(f"\nOverall Metrics:")
print(f"Weighted Avg Precision: {np.average(precision, weights=support):.4f}")
print(f"Weighted Avg Recall: {np.average(recall, weights=support):.4f}")
print(f"Weighted Avg F1-Score: {np.average(f1, weights=support):.4f}")

# Memory cleanup
print("\nPerforming final memory cleanup...")
if 'X_train_selected' in locals():
    del X_train_selected
if 'X_val_selected' in locals():
    del X_val_selected
if 'train_pred_proba' in locals():
    del train_pred_proba
if 'val_pred_proba' in locals():
    del val_pred_proba

gc.collect()

# Performance improvement suggestions
print("\n" + "="*60)
print("PERFORMANCE IMPROVEMENT SUGGESTIONS")
print("="*60)

if val_acc < 0.5:
    print("âš ï¸�  Low accuracy detected. Consider:")
    print("   â€¢ Increasing feature engineering complexity")
    print("   â€¢ Using ensemble methods (Random Forest, XGBoost)")
    print("   â€¢ Collecting more training data")
    print("   â€¢ Trying deep learning approaches")
elif val_acc < 0.7:
    print("ğŸ“ˆ Moderate performance. Potential improvements:")
    print("   â€¢ Fine-tune hyperparameters further") 
    print("   â€¢ Add domain-specific EEG features")
    print("   â€¢ Use time-series specific methods")
    print("   â€¢ Consider neural networks")
else:
    print("âœ… Good performance achieved!")

if val_kl > 2.0:
    print("âš ï¸�  High KL divergence suggests overconfident predictions")
    print("   â€¢ Use probability calibration")
    print("   â€¢ Add label smoothing")
    print("   â€¢ Ensemble multiple models")

print("\nğŸ�¯ Key Improvements Made in This Version:")
print("   âœ“ Enhanced feature engineering with frequency domain analysis")
print("   âœ“ Robust preprocessing pipeline with outlier handling")
print("   âœ“ Hyperparameter optimization via GridSearch")
print("   âœ“ Balanced class weighting")
print("   âœ“ Multi-stage feature selection")
print("   âœ“ Comprehensive evaluation metrics")
print("   âœ“ Better memory management")

print(f"\nğŸ�� Enhanced SVM training completed successfully!")
print(f"Final validation accuracy: {val_acc:.4f} (vs {0.2:.4f} baseline)")
print(f"Improvement: {((val_acc - 0.2) / 0.2 * 100):+.1f}% over baseline")

# Save final results summary
results_summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'model_type': 'Enhanced SVM with GridSearch',
    'hyperparameters': grid_search.best_params_,
    'cv_score': float(grid_search.best_score_),
    'train_accuracy': float(train_acc),
    'validation_accuracy': float(val_acc),
    'validation_kl_divergence': float(val_kl),
    'per_class_metrics': {
        'precision': precision.tolist(),
        'recall': recall.tolist(), 
        'f1_score': f1.tolist(),
        'support': support.tolist()
    },
    'class_names': CLASSES,
    'feature_count': X_train_selected.shape[1] if 'X_train_selected' in locals() else N_FEATURES,
    'training_samples': len(train_ids),
    'validation_samples': len(val_ids)
}

results_path = os.path.join(MODEL_OUTPUT_PATH, "results_summary.json")
import json
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"Results summary saved to: {results_path}")

print("\n" + "="*60)

