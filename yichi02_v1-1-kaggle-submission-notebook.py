# Import required libraries
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# Configuration
MISSING_VALUE = -1.0

# Feature definitions (must match training)
IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
THERMOPILE_FEATURES = [f'thm_{i}' for i in range(1, 6)]
TOF_FEATURES = []
for i in range(1, 6):
    for j in range(64):
        TOF_FEATURES.append(f'tof_{i}_v{j}')

# Use subset of features (must match training)
SENSOR_FEATURES = IMU_FEATURES + THERMOPILE_FEATURES + TOF_FEATURES[:50]

print(f"Total features for inference: {len(SENSOR_FEATURES)}")


# Load pre-trained model components
print("Loading pre-trained model components...")

# In Kaggle environment, these would be uploaded as dataset
# For now, we'll create dummy components that match the expected structure

# Note: In actual Kaggle submission, you would load from uploaded dataset:
# with open('/kaggle/input/your-model-dataset/trained_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# For demonstration, we'll create a simple model structure
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create dummy model components (replace with actual loading in Kaggle)
model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Dummy classes (replace with actual classes from training)
classes = [
    'Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
    'Eyebrow - pull hair', 'Eyelash - pull hair', 'Feel around in tray and pull out an object',
    'Forehead - pull hairline', 'Forehead - scratch', 'Glasses on/off',
    'Neck - pinch skin', 'Neck - scratch', 'Pinch knee/leg skin',
    'Pull air toward your face', 'Scratch knee/leg skin', 'Text on phone',
    'Wave hello', 'Write name in air', 'Write name on leg'
]

label_encoder.classes_ = np.array(classes)

print(f"Model loaded with {len(classes)} classes")
print("Note: In actual Kaggle submission, load from uploaded model dataset")


# Load test data only
print("Loading test data...")
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')

print(f"Test shape: {test_df.shape}")
print(f"Unique test sequences: {test_df['sequence_id'].nunique()}")
print(f"Test columns: {list(test_df.columns)[:10]}...")


def extract_sequence_features(df):
    """
    Extract statistical features from sequences (inference only)
    Must match exactly the feature extraction used during training
    """
    print("Extracting sequence features for inference...")
    
    # Group by sequence_id
    sequences = df.groupby('sequence_id')
    
    features_list = []
    seq_ids_list = []
    
    for seq_id, seq_data in sequences:
        # Extract features for this sequence
        seq_features = []
        
        for feature in SENSOR_FEATURES:
            if feature in seq_data.columns:
                values = pd.to_numeric(seq_data[feature], errors='coerce')
                # Replace -1.0 with NaN
                values = values.replace(-1.0, np.nan)
                
                # Calculate statistics (must match training exactly)
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
        seq_ids_list.append(seq_id)
    
    return np.array(features_list), seq_ids_list


# Extract features from test data
print("Processing test data...")
X_test, test_seq_ids = extract_sequence_features(test_df)

print(f"Test features shape: {X_test.shape}")
print(f"Number of test sequences: {len(test_seq_ids)}")

# Handle any remaining NaN values
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print("Test data preprocessing completed.")


# Prepare features for inference
print("Preparing features for inference...")

# Scale test features using the loaded scaler
# Note: In actual Kaggle submission, the scaler would be loaded from dataset
# For demonstration, we fit a dummy scaler
scaler.fit(X_test)  # This would be replaced with loaded scaler
X_test_scaled = scaler.transform(X_test)

print(f"Scaled test features shape: {X_test_scaled.shape}")


# Make predictions using the loaded model
print("Making predictions on test data...")

# For demonstration, we'll fit a dummy model
# In actual Kaggle submission, this would be the loaded pre-trained model
dummy_y = np.random.randint(0, len(classes), size=X_test_scaled.shape[0])
model.fit(X_test_scaled, dummy_y)  # This would be replaced with loaded model

y_test_pred_encoded = model.predict(X_test_scaled)
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

# Get prediction probabilities
y_test_proba = model.predict_proba(X_test_scaled)

print(f"Number of test predictions: {len(y_test_pred)}")
print(f"Unique predicted gestures: {len(set(y_test_pred))}")


# Analyze predictions
print("\nPrediction Analysis:")
for i, (seq_id, gesture) in enumerate(zip(test_seq_ids, y_test_pred)):
    max_prob = np.max(y_test_proba[i])
    print(f"  {seq_id}: {gesture} (confidence: {max_prob:.3f})")

print(f"\nTotal predictions: {len(y_test_pred)}")
print(f"Unique predicted gestures: {len(set(y_test_pred))}")


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


# Save submission file in parquet format (required by Kaggle)
submission_df.to_parquet('submission.parquet', index=False)
print("Submission file saved as 'submission.parquet'")

# Also save as CSV for verification
submission_df.to_csv('submission.csv', index=False)
print("Verification CSV saved as 'submission.csv'")

# Display final submission
print("\nFinal Submission:")
print(submission_df.to_string(index=False))

# Verify parquet file can be read
verification_df = pd.read_parquet('submission.parquet')
print(f"\nParquet verification - shape: {verification_df.shape}")
print(f"Parquet verification - columns: {list(verification_df.columns)}")

