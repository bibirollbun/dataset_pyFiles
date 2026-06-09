# Basic libraries
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# LightGBM
import lightgbm as lgb

# For signal processing
from scipy import signal

# For the inference server
import kaggle_evaluation.cmi_inference_server

# For saving/loading models
import pickle

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Define paths for local testing
TRAIN_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
TRAIN_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
TEST_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
TEST_DEMO_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'

# Create a directory for models
os.makedirs('models', exist_ok=True)

print("Environment setup complete!")


# Load the data using polars for efficiency
print("Loading data...")
train_data = pl.scan_csv(TRAIN_PATH).fetch()
train_demo = pl.scan_csv(TRAIN_DEMO_PATH).fetch()

# Convert to pandas for easier manipulation
train_df = train_data.to_pandas()
demo_df = train_demo.to_pandas()

print(f"Train data shape: {train_df.shape}")
print(f"Demographics data shape: {demo_df.shape}")

# Basic information about the dataset
print("\nUnique values in key columns:")
print(f"Sequences: {train_df['sequence_id'].nunique()}")
print(f"Subjects: {train_df['subject'].nunique()}")
print(f"Gestures: {train_df['gesture'].nunique()}")
print(f"Behaviors: {train_df['behavior'].nunique()}")

# Display the unique gestures and their counts
gesture_counts = train_df.groupby('sequence_id')['gesture'].first().value_counts()
print("\nGesture distribution:")
print(gesture_counts)

# Display target vs non-target distribution
target_type = train_df.groupby('sequence_id')['sequence_type'].first().value_counts()
print("\nTarget vs Non-target distribution:")
print(target_type)


# Function to plot sensor data for a single sequence
def plot_sequence_data(sequence_id):
    seq_data = train_df[train_df['sequence_id'] == sequence_id]
    
    # Check if data exists for this sequence
    if len(seq_data) == 0:
        print(f"No data found for sequence {sequence_id}")
        return
    
    gesture = seq_data['gesture'].iloc[0]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot IMU data
    axes[0].set_title(f"Sequence {sequence_id} - Gesture: {gesture}")
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_x'], label='acc_x')
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_y'], label='acc_y')
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_z'], label='acc_z')
    axes[0].legend()
    axes[0].set_ylabel('Acceleration (m/s²)')
    
    # Plot thermopile data if available
    thm_cols = [col for col in seq_data.columns if col.startswith('thm_')]
    if not seq_data[thm_cols].isnull().all().all():
        for col in thm_cols:
            axes[1].plot(seq_data['sequence_counter'], seq_data[col], label=col)
        axes[1].legend()
        axes[1].set_ylabel('Temperature (°C)')
    else:
        axes[1].text(0.5, 0.5, 'No thermopile data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1].transAxes)
    
    # Mark behavior changes
    behaviors = seq_data[['sequence_counter', 'behavior']].drop_duplicates()
    for ax in axes:
        for _, row in behaviors.iterrows():
            ax.axvline(x=row['sequence_counter'], color='r', linestyle='--', alpha=0.5)
            ax.text(row['sequence_counter'], ax.get_ylim()[1]*0.9, row['behavior'], 
                   rotation=90, verticalalignment='top')
    
    # Plot rotation data
    axes[2].plot(seq_data['sequence_counter'], seq_data['rot_w'], label='rot_w')
    axes[2].plot(seq_data['sequence_counter'], seq_data['rot_x'], label='rot_x')
    axes[2].plot(seq_data['sequence_counter'], seq_data['rot_y'], label='rot_y')
    axes[2].plot(seq_data['sequence_counter'], seq_data['rot_z'], label='rot_z')
    axes[2].legend()
    axes[2].set_ylabel('Rotation')
    axes[2].set_xlabel('Sequence Counter')
    
    plt.tight_layout()
    plt.show()

# Plot a few example sequences
print("Plotting example sequences...")

# Get examples of target gesture types with error handling
target_sequences = train_df[train_df['sequence_type'] == 'target']
if len(target_sequences) > 0:
    # Get unique gestures and their first sequence_id
    target_gestures = target_sequences.groupby('gesture')['sequence_id'].first()
    
    # Convert to list and get up to 2 examples if available
    target_examples = list(target_gestures.items())[:min(2, len(target_gestures))]
    
    for _, seq_id in target_examples:
        plot_sequence_data(seq_id)
else:
    print("No target sequences found in the dataset")

# Get one example of a non-target gesture with error handling
non_target_sequences = train_df[train_df['sequence_type'] == 'non_target']
if len(non_target_sequences) > 0:
    non_target_seq = non_target_sequences['sequence_id'].iloc[0]
    plot_sequence_data(non_target_seq)
else:
    print("No non-target sequences found in the dataset")

print("Visualization complete!")


# Check for missing data
print("Analyzing missing data...")
missing_data = train_df.isnull().mean() * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

print(f"Columns with missing data (percentage):")
print(missing_data)

# Check if some sequences have entirely missing sensor types
def check_sensor_availability(df):
    sequence_sensor_status = {}
    
    for seq_id in tqdm(df['sequence_id'].unique(), desc="Checking sensors"):
        seq_data = df[df['sequence_id'] == seq_id]
        
        # Check IMU data
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        has_imu = not seq_data[imu_cols].isnull().all().all()
        
        # Check thermopile data
        thm_cols = [col for col in seq_data.columns if col.startswith('thm_')]
        has_thm = not seq_data[thm_cols].isnull().all().all()
        
        # Check time-of-flight data
        tof_cols = [col for col in seq_data.columns if col.startswith('tof_')]
        has_tof = not seq_data[tof_cols].isnull().all().all()
        
        sequence_sensor_status[seq_id] = {
            'has_imu': has_imu,
            'has_thm': has_thm,
            'has_tof': has_tof,
            'has_full_sensors': has_thm and has_tof
        }
    
    return pd.DataFrame.from_dict(sequence_sensor_status, orient='index')

# Check sensor availability
sensor_status = check_sensor_availability(train_df)
print("\nSensor availability in sequences:")
print(sensor_status.sum())
print(f"\nPercentage of sequences with full sensors: {sensor_status['has_full_sensors'].mean() * 100:.2f}%")
print(f"Percentage of sequences with IMU only: {(~sensor_status['has_full_sensors']).mean() * 100:.2f}%")


def extract_features_from_sequence(sequence_df):
    """
    Extract features from a single sequence.
    
    Args:
        sequence_df: DataFrame containing a single sequence
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Basic metadata
    features['sequence_id'] = sequence_df['sequence_id'].iloc[0]
    features['subject'] = sequence_df['subject'].iloc[0]
    
    if 'gesture' in sequence_df.columns:
        features['gesture'] = sequence_df['gesture'].iloc[0]
    
    if 'sequence_type' in sequence_df.columns:
        features['sequence_type'] = sequence_df['sequence_type'].iloc[0]
    
    # Get behavior phases
    phases = {}
    for behavior in ['Transition', 'Pause', 'Gesture']:
        phases[behavior] = sequence_df[sequence_df['behavior'] == behavior]
    
    # Extract IMU features for each phase
    for phase_name, phase_df in phases.items():
        if len(phase_df) == 0:
            continue
        
        # Accelerometer features
        for axis in ['x', 'y', 'z']:
            col = f'acc_{axis}'
            if col in phase_df.columns:
                values = phase_df[col].dropna()
                if len(values) > 0:
                    prefix = f"{phase_name.lower()}_acc_{axis}"
                    features[f"{prefix}_mean"] = values.mean()
                    features[f"{prefix}_std"] = values.std()
                    features[f"{prefix}_min"] = values.min()
                    features[f"{prefix}_max"] = values.max()
                    features[f"{prefix}_median"] = values.median()
                    features[f"{prefix}_range"] = values.max() - values.min()
                    
                    # Add features for the rate of change
                    diff = values.diff().dropna()
                    if len(diff) > 0:
                        features[f"{prefix}_diff_mean"] = diff.mean()
                        features[f"{prefix}_diff_std"] = diff.std()
                        
                    # Add frequency domain features if we have enough points
                    if len(values) >= 5:
                        try:
                            # Compute FFT
                            fft_values = np.abs(np.fft.rfft(values.values))
                            fft_freq = np.fft.rfftfreq(len(values), d=1)
                            
                            # Skip the DC component
                            if len(fft_values) > 1:
                                fft_values = fft_values[1:]
                                fft_freq = fft_freq[1:]
                                
                                # Get dominant frequency
                                idx_max = np.argmax(fft_values)
                                features[f"{prefix}_dom_freq"] = fft_freq[idx_max]
                                features[f"{prefix}_dom_amp"] = fft_values[idx_max]
                        except:
                            # Skip if there's an error in FFT computation
                            pass
        
        # Rotation features
        for axis in ['w', 'x', 'y', 'z']:
            col = f'rot_{axis}'
            if col in phase_df.columns:
                values = phase_df[col].dropna()
                if len(values) > 0:
                    prefix = f"{phase_name.lower()}_rot_{axis}"
                    features[f"{prefix}_mean"] = values.mean()
                    features[f"{prefix}_std"] = values.std()
                    features[f"{prefix}_min"] = values.min()
                    features[f"{prefix}_max"] = values.max()
                    features[f"{prefix}_median"] = values.median()
                    features[f"{prefix}_range"] = values.max() - values.min()
                    
                    # Add features for the rate of change
                    diff = values.diff().dropna()
                    if len(diff) > 0:
                        features[f"{prefix}_diff_mean"] = diff.mean()
                        features[f"{prefix}_diff_std"] = diff.std()
        
        # Thermopile features (if available)
        thm_cols = [col for col in phase_df.columns if col.startswith('thm_')]
        if thm_cols and not phase_df[thm_cols].isnull().all().all():
            for col in thm_cols:
                values = phase_df[col].dropna()
                if len(values) > 0:
                    prefix = f"{phase_name.lower()}_{col}"
                    features[f"{prefix}_mean"] = values.mean()
                    features[f"{prefix}_std"] = values.std()
                    features[f"{prefix}_min"] = values.min()
                    features[f"{prefix}_max"] = values.max()
                    features[f"{prefix}_median"] = values.median()
        
        # Time-of-flight features (if available)
        # Due to the high dimensionality, we'll compute summary statistics
        for i in range(1, 6):  # 5 ToF sensors
            tof_cols = [col for col in phase_df.columns if col.startswith(f'tof_{i}_')]
            if tof_cols and not phase_df[tof_cols].isnull().all().all():
                # Calculate mean across all pixels for each frame
                tof_means = phase_df[tof_cols].replace(-1, np.nan).mean(axis=1).dropna()
                if len(tof_means) > 0:
                    prefix = f"{phase_name.lower()}_tof_{i}"
                    features[f"{prefix}_mean"] = tof_means.mean()
                    features[f"{prefix}_std"] = tof_means.std()
                    features[f"{prefix}_min"] = tof_means.min()
                    features[f"{prefix}_max"] = tof_means.max()
    
    # Calculate features across the entire sequence
    features['sequence_length'] = len(sequence_df)
    
    # Calculate cross-sensor features (relationships between sensors)
    for phase_name, phase_df in phases.items():
        if len(phase_df) == 0:
            continue
            
        # Correlation between acc_x, acc_y, acc_z
        acc_cols = ['acc_x', 'acc_y', 'acc_z']
        if all(col in phase_df.columns for col in acc_cols):
            acc_data = phase_df[acc_cols].dropna()
            if len(acc_data) >= 5:  # Need enough points for correlation
                try:
                    corr_matrix = acc_data.corr()
                    features[f"{phase_name.lower()}_acc_xy_corr"] = corr_matrix.loc['acc_x', 'acc_y']
                    features[f"{phase_name.lower()}_acc_xz_corr"] = corr_matrix.loc['acc_x', 'acc_z']
                    features[f"{phase_name.lower()}_acc_yz_corr"] = corr_matrix.loc['acc_y', 'acc_z']
                except:
                    # Skip if there's an error in correlation computation
                    pass
    
    return features

def extract_all_features(df):
    """
    Extract features for all sequences in the dataset.
    
    Args:
        df: DataFrame containing all sequences
        
    Returns:
        DataFrame with one row per sequence and all extracted features
    """
    all_features = []
    
    for seq_id in tqdm(df['sequence_id'].unique(), desc="Extracting features"):
        try:
            # Extract features for this sequence
            seq_data = df[df['sequence_id'] == seq_id]
            features = extract_features_from_sequence(seq_data)
            all_features.append(features)
        except Exception as e:
            print(f"Error extracting features for sequence {seq_id}: {e}")
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features)
    
    return feature_df

# Extract features from training data
print("Extracting features from training data...")
train_features = extract_all_features(train_df)
print(f"Extracted feature dataset shape: {train_features.shape}")

# Check for constant or near-constant features
constant_features = [col for col in train_features.columns 
                     if col not in ['sequence_id', 'subject', 'gesture', 'sequence_type'] and
                     train_features[col].nunique() <= 1]
print(f"Number of constant features to drop: {len(constant_features)}")

# Drop constant features
train_features = train_features.drop(columns=constant_features, errors='ignore')
print(f"Feature dataset after dropping constant features: {train_features.shape}")

# Add demographics information
train_features_with_demo = train_features.merge(demo_df, on='subject', how='left')
print(f"Feature dataset with demographics: {train_features_with_demo.shape}")

# Check missing values
missing_pct = train_features_with_demo.drop(columns=['sequence_id', 'subject', 'gesture', 'sequence_type'], 
                                           errors='ignore').isnull().mean() * 100
print(f"Columns with >50% missing values: {sum(missing_pct > 50)}")


# Prepare data for modeling
print("Preparing data for modeling...")

# Identify full-sensor and IMU-only sequences
sequence_sensor_status = sensor_status.copy()
train_features_with_demo['has_full_sensors'] = train_features_with_demo['sequence_id'].map(
    sequence_sensor_status['has_full_sensors'])

# Split into features and target
X_full = train_features_with_demo.drop(columns=['sequence_id', 'subject', 'gesture', 'sequence_type', 
                                               'has_full_sensors'], errors='ignore')
y_gesture = train_features_with_demo['gesture']
y_type = train_features_with_demo['sequence_type']

# Encode target
label_encoder = LabelEncoder()
y_gesture_encoded = label_encoder.fit_transform(y_gesture)
print(f"Gesture classes: {label_encoder.classes_}")

# Create a mapping from encoded class to gesture name
class_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}

# Handle missing values with imputation
print("Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_full_imputed = pd.DataFrame(
    imputer.fit_transform(X_full), 
    columns=X_full.columns, 
    index=X_full.index
)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_full_scaled = pd.DataFrame(
    scaler.fit_transform(X_full_imputed),
    columns=X_full_imputed.columns,
    index=X_full_imputed.index
)

# Save preprocessing objects for later use
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
    
with open('models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X_full.columns), f)

print("Preprocessing complete!")


# Define the competition metric
def competition_metric(y_true, y_pred_proba):
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert numeric predictions back to class names
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_true)
    
    # Function to determine if a gesture is a target or non-target
    def is_target(gesture):
        return not any(keyword in gesture for keyword in 
                      ['Text', 'Drink', 'Glasses', 'Pull air', 'Pinch knee', 
                       'Scratch knee', 'Write name', 'Feel', 'Wave'])
    
    # Binary F1: Target vs Non-Target
    y_true_binary = np.array(['target' if is_target(y) else 'non_target' for y in y_true_labels])
    y_pred_binary = np.array(['target' if is_target(y) else 'non_target' for y in y_pred_labels])
    
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label='target')
    
    # For macro F1, collapse all non-target gestures
    y_true_macro = ['non_target' if not is_target(y) else y for y in y_true_labels]
    y_pred_macro = ['non_target' if not is_target(y) else y for y in y_pred_labels]
    
    macro_f1 = f1_score(y_true_macro, y_pred_macro, average='macro')
    
    # Final score is the average
    final_score = (binary_f1 + macro_f1) / 2
    
    return final_score

# Create a cross-validation strategy that accounts for subject dependence
def create_cv_folds(df, n_splits=5):
    # Use subject as a grouping variable to avoid data leakage
    subjects = df['subject'].unique()
    np.random.shuffle(subjects)
    
    # Assign each subject to a fold
    subject_fold = {}
    for i, subject in enumerate(subjects):
        subject_fold[subject] = i % n_splits
    
    # Create fold IDs for each sequence
    fold_ids = df['subject'].map(subject_fold).values
    
    return fold_ids

# Create fold IDs
print("Creating cross-validation folds...")
fold_ids = create_cv_folds(train_features_with_demo, n_splits=5)

# Check fold distribution
fold_counts = pd.Series(fold_ids).value_counts().sort_index()
print("Fold distribution:")
print(fold_counts)


# Train LightGBM models
def train_lgbm_model(X, y, fold_ids, has_full_sensors=None, n_splits=5):
    """
    Train a LightGBM model with cross-validation
    
    Args:
        X: Feature matrix
        y: Target variable
        fold_ids: Fold assignments for cross-validation
        has_full_sensors: Boolean mask indicating which sequences have full sensor data
        n_splits: Number of cross-validation folds
        
    Returns:
        Trained models, OOF predictions, and CV score
    """
    models = []
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    feature_importances = pd.DataFrame(0, index=X.columns, columns=['importance'])
    
    # If has_full_sensors is provided, filter for relevant sequences
    if has_full_sensors is not None:
        X = X[has_full_sensors]
        y = y[has_full_sensors]
        fold_ids = fold_ids[has_full_sensors]
    
    print(f"Training on {len(X)} sequences")
    
    for fold in range(n_splits):
        print(f"Training fold {fold+1}/{n_splits}")
        
        # Split data
        train_idx = fold_ids != fold
        val_idx = fold_ids == fold
        
        # Skip if no validation data in this fold
        if np.sum(val_idx) == 0:
            print(f"No validation data in fold {fold}, skipping...")
            continue
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED + fold,
            'n_jobs': -1
        }
        
        # Callbacks for early stopping - FIX: Use callbacks instead of early_stopping_rounds
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks,  # Fixed: Use callbacks instead of early_stopping_rounds
            verbose_eval=100
        )
        
        # Save model
        models.append(model)
        
        # OOF predictions
        oof_indices = np.where(val_idx)[0]
        oof_preds[oof_indices] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance()
        })
        feature_importances['importance'] += importances.set_index('feature')['importance']
    
    # Normalize feature importances
    if len(models) > 0:  # Only normalize if we have models
        feature_importances['importance'] /= len(models)
    
        # Calculate OOF score
        cv_score = competition_metric(y, oof_preds)
        print(f"Cross-validation score: {cv_score:.4f}")
    else:
        cv_score = 0
        print("Warning: No models were trained!")
    
    return models, oof_preds, feature_importances, cv_score


# Function to analyze predictions and errors
def analyze_predictions(y_true, y_pred_proba, label_encoder):
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    # Target vs Non-target performance
    def is_target(gesture):
        return not any(keyword in gesture for keyword in 
                      ['Text', 'Drink', 'Glasses', 'Pull air', 'Pinch knee', 
                       'Scratch knee', 'Write name', 'Feel', 'Wave'])
    
    y_true_binary = ['target' if is_target(y) else 'non_target' for y in y_true_labels]
    y_pred_binary = ['target' if is_target(y) else 'non_target' for y in y_pred_labels]
    
    print("\nTarget vs Non-Target Classification Report:")
    print(classification_report(y_true_binary, y_pred_binary))

# Check if model training was successful before analysis
print("Checking model availability...")

# Define a dictionary to track which models are available
model_availability = {
    'imu': 'oof_preds_imu' in locals() or 'oof_preds_imu' in globals(),
    'full': 'oof_preds_full' in locals() or 'oof_preds_full' in globals()
}

# Display availability
for model_type, available in model_availability.items():
    print(f"{model_type.upper()} model: {'Available' if available else 'Not available'}")

# Analyze IMU-only model predictions if available
if model_availability['imu']:
    print("\nAnalysis of IMU-only model predictions:")
    analyze_predictions(y_gesture_encoded, oof_preds_imu, label_encoder)
else:
    print("\nIMU model analysis skipped - model not available")

# Analyze full-sensor model predictions if available
if model_availability['full']:
    print("\nAnalysis of full-sensor model predictions:")
    full_sensor_mask = train_features_with_demo['has_full_sensors'].values
    analyze_predictions(y_gesture_encoded[full_sensor_mask], oof_preds_full, label_encoder)
else:
    print("\nFull-sensor model analysis skipped - model not available")

# Add current date/time and user info
print(f"\nAnalysis performed by: AdilShamim8")
print(f"Analysis timestamp (UTC): 2025-08-10 06:35:46")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predict the gesture type for a given sequence.
    
    Args:
        sequence: A Polars DataFrame containing sensor data for one sequence
        demographics: A Polars DataFrame containing demographic information
        
    Returns:
        str: The predicted gesture
    """
    # Convert to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    
    # First, let's ensure we have access to valid gestures from the training data
    global imputer, scaler, label_encoder, feature_columns
    global lgbm_models_imu, lgbm_models_full
    global valid_gestures, default_gesture
    
    # Initialize models loaded flag
    models_loaded = False
    
    # Check if valid gestures are already defined
    if 'valid_gestures' not in globals():
        try:
            # Try to load valid gestures from training data
            train_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
            train_sample = pl.scan_csv(train_path).select(['gesture']).unique('gesture').fetch()
            valid_gestures = train_sample['gesture'].to_list()
            default_gesture = valid_gestures[0]  # Use the first gesture as default
            print(f"Loaded {len(valid_gestures)} valid gestures from training data")
        except Exception as e:
            print(f"Error loading gestures from train data: {e}")
            # Hardcode some common gestures that should be in the data
            valid_gestures = ['Hair pull', 'Skin picking', 'Nail biting', 'Text on phone', 
                              'Drink from cup', 'Glasses adjustment']
            default_gesture = 'Hair pull'  # Use this as a guess
    
    # Now let's try to load our model files if they're not already loaded
    if 'label_encoder' not in globals() or 'imputer' not in globals() or 'scaler' not in globals():
        try:
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            
            with open('models/imputer.pkl', 'rb') as f:
                imputer = pickle.load(f)
                
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
            with open('models/feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
                
            # If we get here, we've successfully loaded the preprocessing objects
            models_loaded = True
            
            # Update valid gestures and default from label encoder classes
            valid_gestures = label_encoder.classes_.tolist()
            default_gesture = valid_gestures[0]
            
            # Load LightGBM models
            lgbm_models_imu = []
            for i in range(5):  # 5-fold CV
                try:
                    model = lgb.Booster(model_file=f'models/lgbm_imu_fold_{i}.txt')
                    lgbm_models_imu.append(model)
                except Exception as e:
                    print(f"Warning: Could not load IMU model fold {i}")
            
            lgbm_models_full = []
            for i in range(5):  # 5-fold CV
                try:
                    model = lgb.Booster(model_file=f'models/lgbm_full_fold_{i}.txt')
                    lgbm_models_full.append(model)
                except Exception as e:
                    print(f"Warning: Could not load full-sensor model fold {i}")
        
        except Exception as e:
            print(f"Error loading models and preprocessing objects: {e}")
            models_loaded = False
    else:
        # If the label encoder is already loaded, assume everything else is too
        models_loaded = True
    
    try:
        # If we don't have models loaded, return a valid gesture from our list
        if not models_loaded:
            print("No models loaded, returning default gesture")
            return default_gesture
            
        # Extract features from the sequence
        features = extract_features_from_sequence(seq_df)
        features_df = pd.DataFrame([features])
        
        # Add demographics data
        subject = seq_df['subject'].iloc[0]
        subject_demo = demo_df[demo_df['subject'] == subject]
        if not subject_demo.empty:
            for col in subject_demo.columns:
                if col != 'subject':
                    features_df[col] = subject_demo[col].values[0]
        
        # Check if this is a full-sensor sequence
        thm_cols = [col for col in seq_df.columns if col.startswith('thm_')]
        tof_cols = [col for col in seq_df.columns if col.startswith('tof_')]
        has_thm = not seq_df[thm_cols].isnull().all().all()
        has_tof = not seq_df[tof_cols].isnull().all().all()
        has_full_sensors = has_thm and has_tof
        
        # Prepare features for prediction - add missing columns
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0  # Add missing columns with default value
        
        X_pred = features_df[feature_columns]
        
        # Apply preprocessing
        X_pred_imputed = pd.DataFrame(
            imputer.transform(X_pred),
            columns=X_pred.columns
        )
        
        X_pred_scaled = pd.DataFrame(
            scaler.transform(X_pred_imputed),
            columns=X_pred_imputed.columns
        )
        
        # Make predictions based on available models
        if has_full_sensors and len(lgbm_models_full) > 0:
            # Use full-sensor models
            predictions = []
            for model in lgbm_models_full:
                pred = model.predict(X_pred_scaled.values)
                predictions.append(pred)
            
            # Average predictions
            avg_pred = np.mean(predictions, axis=0)
            
        elif len(lgbm_models_imu) > 0:
            # Use IMU-only models
            predictions = []
            for model in lgbm_models_imu:
                pred = model.predict(X_pred_scaled.values)
                predictions.append(pred)
            
            # Average predictions
            avg_pred = np.mean(predictions, axis=0)
        else:
            # No models available - return a valid gesture
            return default_gesture
        
        # Get predicted class
        predicted_class = np.argmax(avg_pred)
        predicted_gesture = label_encoder.inverse_transform([predicted_class])[0]
        
        # Double-check that the prediction is a valid gesture
        if predicted_gesture in valid_gestures:
            return predicted_gesture
        else:
            # If somehow we get an invalid prediction, return a known valid one
            print(f"Warning: Invalid prediction '{predicted_gesture}', returning default")
            return default_gesture
        
    except Exception as e:
        # If there's any error, log it and return a default gesture
        print(f"Error in prediction: {e}")
        return default_gesture

# Set up the inference server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

print(f"Inference pipeline created by: AdilShamim8")
print(f"Last updated (UTC): 2025-08-10 06:38:39")

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            TEST_PATH,
            TEST_DEMO_PATH,
        )
    )










