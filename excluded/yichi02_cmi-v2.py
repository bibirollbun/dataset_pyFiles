# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print("Pandas version:", pd.__version__)


# Configuration
RANDOM_STATE = 42
MISSING_VALUE = -1.0

# Feature definitions
IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
THERMOPILE_FEATURES = [f'thm_{i}' for i in range(1, 6)]
TOF_FEATURES = []
for i in range(1, 6):
    for j in range(64):
        TOF_FEATURES.append(f'tof_{i}_v{j}')

# Target gestures (BFRB)
TARGET_GESTURES = [
    'Above ear - pull hair',
    'Cheek - pinch skin', 
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch'
]

print(f"IMU features: {len(IMU_FEATURES)}")
print(f"Thermopile features: {len(THERMOPILE_FEATURES)}")
print(f"ToF features: {len(TOF_FEATURES)}")
print(f"Target gestures: {len(TARGET_GESTURES)}")


# Load data
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Unique train sequences: {train_df['sequence_id'].nunique()}")
print(f"Unique test sequences: {test_df['sequence_id'].nunique()}")
print(f"Unique gestures: {train_df['gesture'].nunique()}")


# Explore gesture distribution
print("Gesture distribution:")
gesture_counts = train_df['gesture'].value_counts()
print(gesture_counts)

# Check BFRB vs non-BFRB distribution
bfrb_count = sum(gesture_counts[gesture] for gesture in TARGET_GESTURES if gesture in gesture_counts.index)
total_count = gesture_counts.sum()
print(f"\nBFRB samples: {bfrb_count} ({bfrb_count/total_count*100:.1f}%)")
print(f"Non-BFRB samples: {total_count - bfrb_count} ({(total_count - bfrb_count)/total_count*100:.1f}%)")


def extract_sequence_features(df, use_subset=True):
    """
    Extract statistical features from sequences
    """
    print("Extracting sequence features...")
    
    # Group by sequence_id
    sequences = df.groupby('sequence_id')
    
    features_list = []
    labels_list = []
    seq_ids_list = []
    
    # Use subset of features for efficiency
    if use_subset:
        sensor_features = IMU_FEATURES + THERMOPILE_FEATURES + TOF_FEATURES[:50]
    else:
        sensor_features = IMU_FEATURES + THERMOPILE_FEATURES + TOF_FEATURES
    
    for seq_id, seq_data in sequences:
        # Extract features for this sequence
        seq_features = []
        
        for feature in sensor_features:
            if feature in seq_data.columns:
                values = pd.to_numeric(seq_data[feature], errors='coerce')
                # Replace -1.0 with NaN
                values = values.replace(-1.0, np.nan)
                
                # Calculate statistics
                if not values.isna().all():
                    seq_features.extend([
                        values.mean(),
                        values.std(),
                        values.min(),
                        values.max(),
                        values.median(),
                        values.count()  # Number of non-missing values
                    ])
                else:
                    seq_features.extend([0, 0, 0, 0, 0, 0])
            else:
                seq_features.extend([0, 0, 0, 0, 0, 0])
        
        features_list.append(seq_features)
        
        # Extract label if available
        if 'gesture' in seq_data.columns:
            labels_list.append(seq_data['gesture'].iloc[0])
        else:
            labels_list.append(None)
            
        seq_ids_list.append(seq_id)
    
    return np.array(features_list), np.array(labels_list), seq_ids_list


def calculate_competition_metric(y_true, y_pred):
    """
    Calculate the competition metric (average of Binary F1 and Macro F1)
    """
    # Convert to binary (target vs non-target)
    y_true_binary = [1 if gesture in TARGET_GESTURES else 0 for gesture in y_true]
    y_pred_binary = [1 if gesture in TARGET_GESTURES else 0 for gesture in y_pred]
    
    # Binary F1 score
    binary_f1 = f1_score(y_true_binary, y_pred_binary)
    
    # Macro F1 score (treating non-target as single class)
    y_true_macro = []
    y_pred_macro = []
    
    for true_gesture, pred_gesture in zip(y_true, y_pred):
        true_label = true_gesture if true_gesture in TARGET_GESTURES else 'non_target'
        pred_label = pred_gesture if pred_gesture in TARGET_GESTURES else 'non_target'
        y_true_macro.append(true_label)
        y_pred_macro.append(pred_label)
    
    # Get unique labels for macro F1
    unique_labels = list(set(y_true_macro + y_pred_macro))
    macro_f1 = f1_score(y_true_macro, y_pred_macro, labels=unique_labels, average='macro')
    
    # Final score
    final_score = (binary_f1 + macro_f1) / 2
    
    return {
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'final_score': final_score
    }


# Extract features from training data
X_train, y_train, train_seq_ids = extract_sequence_features(train_df, use_subset=True)

print(f"Training features shape: {X_train.shape}")
print(f"Number of training sequences: {len(train_seq_ids)}")
print(f"Number of unique gestures: {len(set(y_train))}")

# Handle any remaining NaN values
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

print("Training data preprocessing completed.")


# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)

print(f"Train split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")


# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_split)
y_val_encoded = label_encoder.transform(y_val_split)

print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_}")


# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_encoded)
print("Model training completed.")


# Make predictions on validation set
print("Evaluating model on validation set...")
y_val_pred_encoded = model.predict(X_val_scaled)
y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

# Calculate competition metrics
metrics = calculate_competition_metric(y_val_split, y_val_pred)

print("\n=== VALIDATION RESULTS ===")
print(f"Binary F1: {metrics['binary_f1']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Final Score: {metrics['final_score']:.4f}")

print("\nClassification Report:")
print(classification_report(y_val_split, y_val_pred, zero_division=0))


# Extract features from test data
print("Processing test data...")
X_test, _, test_seq_ids = extract_sequence_features(test_df, use_subset=True)

print(f"Test features shape: {X_test.shape}")
print(f"Number of test sequences: {len(test_seq_ids)}")

# Handle NaN values
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

# Scale test features
X_test_scaled = scaler.transform(X_test)

print("Test data preprocessing completed.")


# Make predictions on test data
print("Making predictions on test data...")
y_test_pred_encoded = model.predict(X_test_scaled)
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

# Get prediction probabilities for confidence analysis
y_test_proba = model.predict_proba(X_test_scaled)

print(f"Number of test predictions: {len(y_test_pred)}")
print(f"Unique predicted gestures: {len(set(y_test_pred))}")


# Analyze predictions
print("\nPrediction Analysis:")
for i, (seq_id, gesture) in enumerate(zip(test_seq_ids, y_test_pred)):
    max_prob = np.max(y_test_proba[i])
    gesture_type = "BFRB" if gesture in TARGET_GESTURES else "Non-BFRB"
    print(f"  {seq_id}: {gesture} ({gesture_type}, confidence: {max_prob:.3f})")

# Check prediction distribution
bfrb_count = sum(1 for gesture in y_test_pred if gesture in TARGET_GESTURES)
non_bfrb_count = len(y_test_pred) - bfrb_count

print(f"\nPrediction Distribution:")
print(f"  BFRB gestures: {bfrb_count}")
print(f"  Non-BFRB gestures: {non_bfrb_count}")


# Create submission dataframe
submission_df = pd.DataFrame({
    'sequence_id': test_seq_ids,
    'gesture': y_test_pred
})

print("Submission DataFrame:")
print(submission_df)

# Verify submission format
print(f"\nSubmission shape: {submission_df.shape}")
print(f"Required columns: ['sequence_id', 'gesture']")
print(f"Actual columns: {list(submission_df.columns)}")
print(f"All sequence_ids present: {set(test_seq_ids) == set(submission_df['sequence_id'])}")


# Save submission file in parquet format (REQUIRED by this competition)
submission_df.to_parquet('submission.parquet', index=False)
print("âœ… Submission file saved as 'submission.parquet' (REQUIRED FORMAT)")


# Verify the parquet file
verification_df = pd.read_parquet('submission.parquet')
print(f"\nâœ… Parquet file verification:")
print(f"   Shape: {verification_df.shape}")
print(f"   Columns: {list(verification_df.columns)}")
print(f"   Data types: {verification_df.dtypes.to_dict()}")

# Display final submission
print("\nFinal Submission Content:")
print(verification_df.to_string(index=False))

