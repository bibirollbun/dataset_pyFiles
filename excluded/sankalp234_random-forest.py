import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy.stats import entropy
import pickle
import gc

warnings.filterwarnings('ignore')

# Configuration
TEST_PATH = "/kaggle/working/"
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH = "/kaggle/input/preprocessing/preprocessed/eeg"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train.csv")
MODEL_OUTPUT_PATH = os.path.join(TEST_PATH, "models_rf")
FEATURE_CACHE_PATH = os.path.join(TEST_PATH, "feature_cache")

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FEATURE_CACHE_PATH, exist_ok=True)

CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
N_CLASSES = len(CLASSES)
BATCH_SIZE = 500  # Process data in batches

print("Starting Memory-Efficient Random Forest EEG Classification")

# KL Divergence function
def kl_divergence_numpy(y_true, y_pred_proba, epsilon=1e-7):
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0)
    kl_div = np.sum(y_true * np.log(y_true / y_pred_proba), axis=1)
    return np.mean(kl_div)

# Feature extraction function
def extract_features(eeg_data):
    """Extract statistical and frequency domain features from EEG data"""
    features = []
    
    # Time domain features for each channel
    for ch in range(eeg_data.shape[0]):  # 19 channels
        channel_data = eeg_data[ch, :]
        
        # Statistical features
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.var(channel_data),
            np.median(channel_data),
            np.min(channel_data),
            np.max(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.sum(np.abs(channel_data)),
            np.sum(channel_data ** 2),
        ])
        
        # Additional statistical measures
        features.extend([
            np.abs(np.mean(channel_data)),
            np.sqrt(np.mean(channel_data ** 2)),
            entropy(np.abs(channel_data) + 1e-10),
        ])
    
    # Cross-channel features
    features.extend([
        np.mean(eeg_data),
        np.std(eeg_data),
        np.var(eeg_data),
        np.corrcoef(eeg_data).mean(),
    ])
    
    return np.array(features)

def process_eeg_file(eeg_id):
    """Process a single EEG file and return features"""
    eeg_path = os.path.join(PREPROCESSED_PATH, f"{eeg_id}.npy")
    try:
        eeg_data = np.load(eeg_path).astype(np.float32)
        
        # Handle NaN/Inf values
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            for ch in range(eeg_data.shape[0]):
                channel = eeg_data[ch, :]
                mask = np.isnan(channel) | np.isinf(channel)
                if np.any(mask):
                    channel[mask] = np.nanmean(channel)
        
        # Standardize each channel
        eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-7)
        
        # Extract features
        features = extract_features(eeg_data)
        return features
        
    except Exception as e:
        print(f"Error loading EEG data for {eeg_id}: {e}")
        return None

class BatchDataGenerator:
    """Generator that yields batches of features and labels"""
    def __init__(self, eeg_ids, labels, batch_size=BATCH_SIZE):
        self.eeg_ids = eeg_ids
        self.labels = labels
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(0, len(self.eeg_ids), self.batch_size):
            batch_ids = self.eeg_ids[i:i+self.batch_size]
            batch_labels = self.labels[i:i+self.batch_size]
            
            batch_features = []
            batch_y = []
            
            for eeg_id, label in zip(batch_ids, batch_labels):
                features = process_eeg_file(eeg_id)
                if features is not None:
                    batch_features.append(features)
                    batch_y.append(label)
            
            if batch_features:
                yield np.array(batch_features), np.array(batch_y)

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

# Load data
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

# Train-validation split
train_ids, val_ids, train_labels, val_labels = train_test_split(
    eeg_ids, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

# METHOD: Use Random Forest with batch processing and imputation
print("\nMethod: Using Random Forest with batch processing and imputation...")

# Process training data in batches
print("Processing training data in batches...")
train_generator = BatchDataGenerator(train_ids, train_labels)

# Collect all training data
all_train_features = []
all_train_labels = []

for batch_X, batch_y in tqdm(train_generator, desc="Processing training batches"):
    all_train_features.append(batch_X)
    all_train_labels.append(batch_y)
    gc.collect()

# Concatenate all batches
X_train = np.vstack(all_train_features)
y_train = np.hstack(all_train_labels)
print(f"Training feature matrix shape: {X_train.shape}")

# Clear intermediate data
del all_train_features, all_train_labels
gc.collect()

# Process validation data
print("Processing validation data in batches...")
val_generator = BatchDataGenerator(val_ids, val_labels)

all_val_features = []
all_val_labels = []

for batch_X, batch_y in tqdm(val_generator, desc="Processing validation batches"):
    all_val_features.append(batch_X)
    all_val_labels.append(batch_y)
    gc.collect()

X_val = np.vstack(all_val_features)
y_val = np.hstack(all_val_labels)
print(f"Validation feature matrix shape: {X_val.shape}")

# Clear intermediate data
del all_val_features, all_val_labels
gc.collect()

# Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')  # Replace NaN with mean of each column
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Verify no NaN values remain
if np.any(np.isnan(X_train)) or np.any(np.isnan(X_val)):
    print("Warning: NaN values still present after imputation!")
else:
    print("Imputation successful: No NaN values in training or validation data.")

# Calculate class weights
class_counts = pd.Series(y_train).value_counts().sort_index().values
class_weights = len(y_train) / (N_CLASSES * class_counts)
class_weight_dict = {i: class_weights[i] for i in range(N_CLASSES)}

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
train_pred_proba = rf_model.predict_proba(X_train)
val_pred_proba = rf_model.predict_proba(X_val)

train_pred = np.argmax(train_pred_proba, axis=1)
val_pred = np.argmax(val_pred_proba, axis=1)

# Calculate metrics
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

# Calculate KL divergence
train_true_one_hot = np.eye(N_CLASSES)[y_train.astype(int)]
val_true_one_hot = np.eye(N_CLASSES)[y_val.astype(int)]

train_kl = kl_divergence_numpy(train_true_one_hot, train_pred_proba)
val_kl = kl_divergence_numpy(val_true_one_hot, val_pred_proba)

print(f"\nFinal Results:")
print(f"Train Accuracy: {train_acc:.6f}, Train KL Divergence: {train_kl:.6f}")
print(f"Val Accuracy: {val_acc:.6f}, Val KL Divergence: {val_kl:.6f}")

# Feature importance plot
feature_importance = rf_model.feature_importances_
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.title('Top 20 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (Validation Set) - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Save model
model_path = os.path.join(MODEL_OUTPUT_PATH, "random_forest_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Model saved to: {model_path}")

# Save imputer for future use
imputer_path = os.path.join(MODEL_OUTPUT_PATH, "imputer.pkl")
with open(imputer_path, 'wb') as f:
    pickle.dump(imputer, f)
print(f"Imputer saved to: {imputer_path}")

# Memory-efficient submission file creation
def create_prediction_file_efficient():
    """Create submission file with memory-efficient batch processing"""
    test_eeg_path = os.path.join(BASE_PATH, "test_eegs")
    if not os.path.exists(test_eeg_path):
        print("Warning: Test EEG path not found at", test_eeg_path)
        print("Creating dummy submission")
        submission_df = pd.DataFrame({'eeg_id': ['dummy_1', 'dummy_2']})
        for cls in CLASSES:
            submission_df[cls] = 1.0 / N_CLASSES
        submission_path = os.path.join(TEST_PATH, "submission_rf.csv")
        submission_df.to_csv(submission_path, index=False)
        print("Dummy submission file saved to:", submission_path)
        return submission_df
    
    test_files = [f.replace(".parquet", "") for f in os.listdir(test_eeg_path) if f.endswith(".parquet")]
    if len(test_files) == 0:
        print("No test files found in", test_eeg_path)
        return None

    submission_df = pd.DataFrame({'eeg_id': test_files})
    for cls in CLASSES:
        submission_df[cls] = 0.0

    print(f"Generating predictions for {len(test_files)} test files in batches...")
    predictions_made = 0
    failed_predictions = 0
    
    # Process test files in batches
    total_batches = (len(test_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(test_files), BATCH_SIZE), total=total_batches, desc="Processing test batches"):
        batch_files = test_files[i:i+BATCH_SIZE]
        batch_features = []
        valid_files = []
        
        # Extract features for current batch
        for eeg_id in batch_files:
            features = process_eeg_file(eeg_id)
            if features is not None:
                batch_features.append(features)
                valid_files.append(eeg_id)
            else:
                # Set uniform probabilities for failed files
                uniform_prob = 1.0 / N_CLASSES
                for cls in CLASSES:
                    submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = uniform_prob
                failed_predictions += 1
        
        # Make predictions for valid files in the batch
        if batch_features:
            try:
                batch_X = np.array(batch_features)
                batch_X = imputer.transform(batch_X)  # Apply imputation to test data
                batch_probs = rf_model.predict_proba(batch_X)
                
                # Update submission dataframe
                for j, eeg_id in enumerate(valid_files):
                    for k, cls in enumerate(CLASSES):
                        submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = batch_probs[j, k]
                    predictions_made += 1
                
                # Clean up batch memory
                del batch_X, batch_probs
                
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
                # Set uniform probabilities for failed batch
                uniform_prob = 1.0 / N_CLASSES
                for eeg_id in valid_files:
                    for cls in CLASSES:
                        submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = uniform_prob
                    failed_predictions += 1
        
        # Force garbage collection after each batch
        gc.collect()

    # Verify all rows have valid probabilities
    for cls in CLASSES:
        zero_mask = submission_df[cls] == 0.0
        if zero_mask.sum() > 0:
            print(f"Warning: {zero_mask.sum()} entries have zero probability for class {cls}")
            submission_df.loc[zero_mask, cls] = 1.0 / N_CLASSES

    # Normalize probabilities to ensure they sum to 1
    prob_cols = [cls for cls in CLASSES]
    row_sums = submission_df[prob_cols].sum(axis=1)
    for cls in CLASSES:
        submission_df[cls] = submission_df[cls] / row_sums

    submission_path = os.path.join(TEST_PATH, "submission_rf.csv")
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to: {submission_path}")
    print(f"Successfully made predictions for: {predictions_made}/{len(test_files)} test files")
    print(f"Failed predictions (using uniform distribution): {failed_predictions}/{len(test_files)}")
    
    # Display sample of submission
    print("\nSample submission entries:")
    print(submission_df.head())
    
    # Verify submission format
    print(f"\nSubmission shape: {submission_df.shape}")
    print(f"Required columns: {['eeg_id'] + CLASSES}")
    print(f"Actual columns: {list(submission_df.columns)}")
    
    return submission_df

def validate_submission(submission_df):
    """Validate the submission file format and content"""
    if submission_df is None:
        return False
    
    # Check required columns
    required_cols = ['eeg_id'] + CLASSES
    if not all(col in submission_df.columns for col in required_cols):
        print("Error: Missing required columns in submission")
        return False
    
    # Check for NaN values
    if submission_df.isnull().any().any():
        print("Error: NaN values found in submission")
        return False
    
    # Check probability constraints
    prob_cols = CLASSES
    row_sums = submission_df[prob_cols].sum(axis=1)
    
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print("Warning: Probabilities don't sum to 1.0 for all rows")
        print(f"Row sum range: {row_sums.min():.6f} to {row_sums.max():.6f}")
    
    # Check for negative probabilities
    if (submission_df[prob_cols] < 0).any().any():
        print("Error: Negative probabilities found")
        return False
    
    print("Submission validation passed!")
    return True

# Create and validate submission file
print("\n" + "="*50)
print("CREATING SUBMISSION FILE")
print("="*50)

try:
    submission_df = create_prediction_file_efficient()
    
    if submission_df is not None:
        is_valid = validate_submission(submission_df)
        if is_valid:
            print("âœ“ Submission file created and validated successfully!")
        else:
            print("âœ— Submission file validation failed!")
    else:
        print("âœ— Failed to create submission file!")
        
except Exception as e:
    print(f"Error creating submission file: {e}")
    import traceback
    traceback.print_exc()

# Print final summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Model: Random Forest (Memory-Efficient)")
print(f"Training samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Feature dimensions: {X_train.shape[1] if 'X_train' in locals() else 'N/A'}")
print(f"Final train accuracy: {train_acc:.6f}")
print(f"Final validation accuracy: {val_acc:.6f}")
print(f"Final validation KL divergence: {val_kl:.6f}")
print("="*50)

# Save label encoder for future use
label_encoder_path = os.path.join(MODEL_OUTPUT_PATH, "label_encoder.pkl")
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to: {label_encoder_path}")

# Save training configuration
config = {
    'classes': CLASSES,
    'n_classes': N_CLASSES,
    'batch_size': BATCH_SIZE,
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': class_weight_dict,
        'random_state': 42,
        'n_jobs': -1
    },
    'train_accuracy': float(train_acc),
    'val_accuracy': float(val_acc),
    'val_kl_divergence': float(val_kl),
    'n_train_samples': len(train_ids),
    'n_val_samples': len(val_ids)
}

config_path = os.path.join(MODEL_OUTPUT_PATH, "training_config.pkl")
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"Training configuration saved to: {config_path}")

# Final cleanup
print("\nCleaning up memory...")
if 'X_train' in locals():
    del X_train
if 'y_train' in locals():
    del y_train
if 'X_val' in locals():
    del X_val
if 'y_val' in locals():
    del y_val
if 'train_pred_proba' in locals():
    del train_pred_proba
if 'val_pred_proba' in locals():
    del val_pred_proba

gc.collect()
print("Memory cleanup completed!")
print("\nðŸŽ‰ Memory-Efficient Random Forest training completed successfully! ðŸŽ‰")

