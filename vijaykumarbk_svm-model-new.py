import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import multiprocessing
import pickle
import warnings
import random
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configuration
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH = "/kaggle/input/how-to-make-spectrogram-from-eeg/EEG_Spectrograms/"  # Path to preprocessed spectrogram .npy files
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train.csv")
MODEL_OUTPUT_PATH = os.path.join("/kaggle/working/", "models")
FEATURE_CACHE_PATH = os.path.join("/kaggle/working/", "feature_cache")
EXPECTED_CHANNELS = 128  # Adjusted to match Chris Deotte’s 4 montages (LL, LP, RR, RP)
FEATURES_PER_CHANNEL = 9  # Mean, var, skew, kurtosis, 4 frequency bands, spectral centroid

# Create output directories
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FEATURE_CACHE_PATH, exist_ok=True)

# Define classes
CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
N_CLASSES = len(CLASSES)
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

# Metrics functions
def kl_divergence(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0)
    kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl_div)

# Check GPU availability
def check_gpu_availability():
    try:
        test_model = XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        test_model.fit(np.zeros((2, 2)), [0, 1])
        return 'gpu_hist'
    except:
        print("GPU not available, falling back to CPU histogram")
        return 'hist'

# Check if files exist
if not os.path.exists(TRAIN_LABELS_PATH):
    print(f"Error: Training labels file not found at {TRAIN_LABELS_PATH}")
    print("Available files in BASE_PATH:", os.listdir(BASE_PATH) if os.path.exists(BASE_PATH) else "BASE_PATH does not exist")
    exit()

# Load and preprocess training labels
try:
    df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(df)} annotations with {len(df['eeg_id'].unique())} unique EEG IDs")
except Exception as e:
    print(f"Error loading training labels: {e}")
    exit()

# Non-overlapping EEG ID processing
train = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
train.columns = ['spec_id', 'min']

tmp = df.groupby('eeg_id')[['spectrogram_label_offset_seconds']].agg('max')
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlap eeg_id shape:', train.shape)
print(train.head())

# Encode hard labels
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
train['label'] = label_encoder.transform(train['target'])
print(f"Computed labels for {len(train)} non-overlapping EEG IDs")

# Display original class distribution
print("Original class distribution:")
print(train['target'].value_counts(normalize=True))

# Data augmentation function for spectrograms
def augment_spectrogram_data(spec_data, prob=0.5):
    augmented = spec_data.copy()
    for channel in range(spec_data.shape[0]):  # Loop over channels
        if random.random() < prob:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1 * np.std(augmented[channel, :, :]), augmented[channel, :, :].shape)
            augmented[channel, :, :] += noise
        if random.random() < prob:
            # Time shift (random shift within 10% of time axis)
            shift = int(random.uniform(-0.1, 0.1) * augmented.shape[2])
            augmented[channel, :, :] = np.roll(augmented[channel, :, :], shift, axis=1)
        if random.random() < prob:
            # Scale amplitude (random scaling between 0.9 and 1.1)
            scale = random.uniform(0.9, 1.1)
            augmented[channel, :, :] *= scale
    return augmented

# Feature loading and caching for spectrograms
def load_and_cache_features(eeg_id, augment=False):
    cache_file = os.path.join(FEATURE_CACHE_PATH, f"{eeg_id}_aug.npy" if augment else f"{eeg_id}.npy")
    if os.path.exists(cache_file):
        try:
            features = np.load(cache_file)
            if features.shape[0] == EXPECTED_CHANNELS * FEATURES_PER_CHANNEL and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                return features
            else:
                print(f"Warning: Cached features for {eeg_id} have incorrect shape {features.shape} or contain NaN/Inf")
        except Exception as e:
            print(f"Error loading cached features for {eeg_id}: {e}")
    
    try:
        spec_path = os.path.join(PREPROCESSED_PATH, f"{eeg_id}.npy")
        if not os.path.exists(spec_path):
            print(f"Spectrogram file not found for {eeg_id} at {spec_path}")
            return None
        spec_data = np.load(spec_path).astype(np.float32)  # Expected shape: (channels, freq, time)
        
        if augment:
            spec_data = augment_spectrogram_data(spec_data, prob=0.5)
        
        # Check shape and print for debugging
        #print(f"Spectrogram {eeg_id} shape: {spec_data.shape}")
        if spec_data.shape[0] != EXPECTED_CHANNELS:
            print(f"Error: Spectrogram data for {eeg_id} has {spec_data.shape[0]} channels, expected {EXPECTED_CHANNELS}")
            return None
        
        if np.any(np.isnan(spec_data)) or np.any(np.isinf(spec_data)):
            print(f"Warning: NaN or Inf values in spectrogram data for {eeg_id}")
            return None
        
        features = []
        # Assume frequency bins cover 0-100 Hz, adjust if different
        freqs = np.linspace(0, 50, spec_data.shape[1])  # Frequency axis (axis 1)
        for channel in range(min(spec_data.shape[0], EXPECTED_CHANNELS)):  # Limit to expected channels
            signal = spec_data[channel, :, :].flatten()  # Flatten each channel for stats
            # Compute statistical features
            mean_val = np.mean(signal)
            var_val = np.var(signal)
            skew_val = skew(signal)
            kurt_val = kurtosis(signal)
            if np.any(np.isnan([mean_val, var_val, skew_val, kurt_val])):
                print(f"Warning: NaN in statistical features for spectrogram {eeg_id}, channel {channel}")
                return None
            features.extend([mean_val, var_val, skew_val, kurt_val])
            # Compute frequency band powers (frequency axis is axis 1)
            freq_bands = [(0, 4), (4, 8), (8, 13), (13, 30)]  # Delta, Theta, Alpha, Beta
            for low, high in freq_bands:
                band_indices = (freqs >= low) & (freqs < high)
                band_power = np.mean(spec_data[channel, band_indices, :])
                if np.isnan(band_power) or np.isinf(band_power):
                    print(f"Warning: NaN/Inf in band power for spectrogram {eeg_id}, channel {channel}, band ({low},{high})")
                    return None
                features.append(band_power)
            # Compute spectral centroid (weighted mean of frequencies)
            spec_channel = np.mean(spec_data[channel, :, :], axis=1)  # Average over time
            spec_sum = np.sum(spec_channel)
            if spec_sum == 0 or np.isnan(spec_sum) or np.isinf(spec_sum):
                print(f"Warning: Zero or invalid sum for spectrogram {eeg_id}, channel {channel}, using default centroid 0")
                spectral_centroid = 0.0  # Default value
            else:
                spectral_centroid = np.sum(freqs * spec_channel) / (spec_sum + 1e-10)
                if np.isnan(spectral_centroid) or np.isinf(spectral_centroid):
                    print(f"Warning: NaN/Inf in spectral centroid for spectrogram {eeg_id}, channel {channel}, using default 0")
                    spectral_centroid = 0.0
            features.append(spectral_centroid)
        
        features = np.array(features, dtype=np.float32)
        expected_feature_size = EXPECTED_CHANNELS * FEATURES_PER_CHANNEL
        if features.shape[0] != expected_feature_size:
            print(f"Error: Feature vector for {eeg_id} has shape {features.shape}, expected ({expected_feature_size},)")
            return None
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"Warning: NaN or Inf in final features for spectrogram {eeg_id}")
            return None
        
        try:
            np.save(cache_file, features)
        except Exception as e:
            print(f"Error saving cached features for {eeg_id}: {e}")
        return features
    except Exception as e:
        print(f"Error loading features for {eeg_id}: {e}")
        return None

# Check preprocessed data
if not os.path.exists(PREPROCESSED_PATH):
    print(f"Error: Preprocessed spectrogram path not found at {PREPROCESSED_PATH}")
    print("Available files in /kaggle/working/:", os.listdir("/kaggle/working/") if os.path.exists("/kaggle/working/") else "Path does not exist")
    exit()

# Load features
print("Loading features...")
X = []
y_soft = []
y_hard = []
eeg_ids = []
skipped_samples = 0
for idx, row in train.iterrows():
    eeg_id = str(row['eeg_id'])
    # Original features
    features = load_and_cache_features(eeg_id, augment=False)
    if features is not None:
        X.append(features)
        y_soft.append(row[TARGETS].values)
        y_hard.append(row['label'])
        eeg_ids.append(eeg_id)
    else:
        skipped_samples += 1
    # Augmented features
    features_aug = load_and_cache_features(eeg_id, augment=True)
    if features_aug is not None:
        X.append(features_aug)
        y_soft.append(row[TARGETS].values)
        y_hard.append(row['label'])
        eeg_ids.append(eeg_id + "_aug")
    else:
        skipped_samples += 1

if skipped_samples > 0:
    print(f"Warning: Skipped {skipped_samples} samples due to missing or invalid spectrogram data")

if len(X) == 0:
    print("Error: No features could be loaded. Check preprocessed data in", PREPROCESSED_PATH)
    exit()

try:
    X = np.array(X)
    y_soft = np.array(y_soft, dtype=np.float64)
    y_hard = np.array(y_hard, dtype=int)
except ValueError as e:
    print(f"Error converting X to array: {e}")
    print("Debugging feature shapes:")
    for i, (features, eeg_id) in enumerate(zip(X, eeg_ids)):
        print(f"Spectrogram {eeg_id}: Feature shape {np.array(features).shape}")
    exit()

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X)
print(f"PCA reduced feature dimension from {X.shape[1]} to {X_pca.shape[1]}")
X = X_pca  # Replace X with PCA-transformed features

print(f"Loaded features for {len(X)} samples")
print(f"Feature vector shape: {X.shape}")

# Train-validation split
X_train, X_val, y_train, y_val, y_soft_train, y_soft_val, train_eeg_ids, val_eeg_ids = train_test_split(
    X, y_hard, y_soft, eeg_ids, test_size=0.2, random_state=42, stratify=y_hard
)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Verify class distribution
train_class_counts = pd.Series(y_train).value_counts(normalize=True)
train_class_counts.index = [CLASSES[i] for i in train_class_counts.index]
val_class_counts = pd.Series(y_val).value_counts(normalize=True)
val_class_counts.index = [CLASSES[i] for i in val_class_counts.index]
print("\nClass distribution in training set:")
print(train_class_counts)
print("\nClass distribution in validation set:")
print(val_class_counts)

# Plot class distribution
plt.figure(figsize=(8, 5))
plt.bar(train_class_counts.index, train_class_counts.values, alpha=0.5, label='Training')
plt.bar(val_class_counts.index, val_class_counts.values, alpha=0.5, label='Validation')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'class_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()

# Set up multiprocessing
try:
    max_cores = min(2, multiprocessing.cpu_count() - 1)
except:
    max_cores = 1
print(f"Using {max_cores} cores")

# Fixed hyperparameters for XGBoost
fixed_params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'lambda': 1.0,
    'alpha': 0.1,
    'random_state': 42,
    'n_jobs': max_cores,
    'objective': 'multi:softprob',
    'num_class': N_CLASSES,
    'eval_metric': 'mlogloss',
    'tree_method': check_gpu_availability(),
    'max_bin': 64
}

# Compute class weights for training
class_counts = pd.Series(y_train).value_counts()
class_weights = {i: len(y_train) / (N_CLASSES * count) for i, count in class_counts.items()}
sample_weights = np.array([class_weights[label] for label in y_train])

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(**fixed_params, early_stopping_rounds=10)
try:
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False
    )
    print(f"Number of boosting rounds used: {xgb_model.get_booster().num_boosted_rounds()}")
    
    # Evaluate XGBoost on training set
    y_train_pred = xgb_model.predict(X_train)
    y_train_pred_proba = xgb_model.predict_proba(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_kl = kl_divergence(y_soft_train, y_train_pred_proba)
    train_ll = log_loss(y_train, y_train_pred_proba)
    print(f"XGBoost Training Accuracy: {train_acc:.6f}")
    print(f"XGBoost Training KL Divergence: {train_kl:.6f}")
    print(f"XGBoost Training Log Loss: {train_ll:.6f}")
    
    # Evaluate XGBoost on validation set
    y_val_pred = xgb_model.predict(X_val)
    y_val_pred_proba = xgb_model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_kl = kl_divergence(y_soft_val, y_val_pred_proba)
    val_ll = log_loss(y_val, y_val_pred_proba)
    print(f"XGBoost Validation Accuracy: {val_acc:.6f}")
    print(f"XGBoost Validation KL Divergence: {val_kl:.6f}")
    print(f"XGBoost Validation Log Loss: {val_ll:.6f}")
    
    # Save XGBoost model
    xgb_model_path = os.path.join(MODEL_OUTPUT_PATH, "xgb_model_final.pkl")
    with open(xgb_model_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print("XGBoost model saved to:", xgb_model_path)
except Exception as e:
    print(f"Error training XGBoost model: {e}")
    exit()

# Feature importance visualization for XGBoost
try:
    feature_names = []
    for channel in range(EXPECTED_CHANNELS):
        feature_names.extend([
            f"Channel_{channel}_Mean", f"Channel_{channel}_Var",
            f"Channel_{channel}_Skew", f"Channel_{channel}_Kurtosis",
            f"Channel_{channel}_Delta", f"Channel_{channel}_Theta",
            f"Channel_{channel}_Alpha", f"Channel_{channel}_Beta",
            f"Channel_{channel}_Centroid"
        ])
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names[:len(xgb_model.feature_importances_)],
        'Importance': xgb_model.feature_importances_
    })
    top_features = feature_importance_df.sort_values('Importance', ascending=False)
    print("Top feature importances for XGBoost:")
    print(top_features)
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'][:10], top_features['Importance'][:10])
    plt.title('Top 10 Feature Importances for XGBoost')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'xgb_feature_importance.png'), dpi=100, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error creating XGBoost feature importance: {e}")

# Confusion matrix for XGBoost validation set
try:
    cm = confusion_matrix(y_val, y_val_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues')
    plt.title(f'Normalized Confusion Matrix - XGBoost Validation (KL: {val_kl:.6f})')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'xgb_confusion_matrix.png'), dpi=100, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error creating XGBoost confusion matrix: {e}")

# Learning Curves for XGBoost
try:
    train_sizes, train_scores, val_scores = learning_curve(
        XGBClassifier(**fixed_params),
        X, y_hard,
        cv=5,
        scoring='neg_log_loss',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=max_cores
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train Log Loss')
    plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation Log Loss')
    plt.title('XGBoost Learning Curves')
    plt.xlabel('Training Samples')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'xgb_learning_curves.png'), dpi=100, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error creating XGBoost learning curves: {e}")

# Precision-Recall Curves for XGBoost
try:
    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(CLASSES):
        y_val_binary = (y_val == i).astype(int)
        y_val_pred_proba_cls = y_val_pred_proba[:, i]
        precision, recall, _ = precision_recall_curve(y_val_binary, y_val_pred_proba_cls)
        auc_pr = auc(recall, precision)
        plt.plot(recall, precision, label=f'{cls} (AUC = {auc_pr:.2f})')
    plt.title('XGBoost Precision-Recall Curves by Class')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'xgb_pr_curves.png'), dpi=100, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error creating XGBoost PR curves: {e}")

print("\nTraining completed!")
print(f"XGBoost Training KL Divergence: {train_kl:.6f}")
print(f"XGBoost Validation KL Divergence: {val_kl:.6f}")
print(f"XGBoost Training Accuracy: {train_acc:.6f}")
print(f"XGBoost Validation Accuracy: {val_acc:.6f}")
print(f"XGBoost Training Log Loss: {train_ll:.6f}")
print(f"XGBoost Validation Log Loss: {val_ll:.6f}")
print(f"Files saved in: {MODEL_OUTPUT_PATH}")

