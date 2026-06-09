import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# --- Load All 4 Data Files ---
try:
    # Main sensor data
    train_df = pd.read_csv('/kaggle/input/cmi-data-set/train.csv')
    test_df = pd.read_csv('/kaggle/input/cmi-data-set/test.csv')
    
    # Demographics files
    train_demo_df = pd.read_csv('/kaggle/input/cmi-data-set/train_demographics.csv')
    test_demo_df = pd.read_csv('/kaggle/input/cmi-data-set/test_demographics.csv') # Corrected typo

    print("All 4 CSV files loaded successfully.")

except FileNotFoundError:
    print("One or more CSV files not found. Please check file paths.")
    
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train Demographics shape: {train_demo_df.shape}")
print(f"Test Demographics shape: {test_demo_df.shape}")


print("Extracting labels from train.csv...")

# Create the labels_df by taking the unique sequence_id and gesture pairs
labels_df = train_df[['sequence_id', 'gesture']].drop_duplicates().reset_index(drop=True)

print(f"Labels extracted. Found {len(labels_df)} unique sequences.")
print(labels_df.head())


def create_features(df):
    # 'gesture' is already in metadata_cols, so it will be ignored correctly
    metadata_cols = ['row_id', 'sequence_id', 'sequence_counter', 'subject', 
                     'orientation', 'behavior', 'phase', 'gesture', 'sequence_type']
    
    sensor_cols = [col for col in df.columns if col not in metadata_cols]
    
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    
    print(f"Starting sensor feature engineering with {len(sensor_cols)} columns...")
    
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
    df[sensor_cols] = df[sensor_cols].fillna(0) 

    grouped_df = df.groupby('sequence_id')[sensor_cols].agg(agg_funcs)
    
    new_cols = []
    for col in grouped_df.columns:
        new_cols.append(f"{col[0]}_{col[1]}")
    grouped_df.columns = new_cols
    
    print("Sensor feature engineering complete.")
    return grouped_df.reset_index()

# Create sensor features for train and test data
X_train_sensor_features = create_features(train_df)
X_test_sensor_features = create_features(test_df)

print(f"Train sensor features shape: {X_train_sensor_features.shape}")
print(f"Test sensor features shape: {X_test_sensor_features.shape}")


print("Starting demographic feature processing...")

# --- Handle Categorical Demo Features ---
categorical_cols = ['adult_child', 'sex', 'handedness']

train_demo_df['is_train'] = 1
test_demo_df['is_train'] = 0
combined_demo_df = pd.concat([train_demo_df, test_demo_df], ignore_index=True)
combined_demo_df = pd.get_dummies(combined_demo_df, columns=categorical_cols, drop_first=True)

train_demo_processed = combined_demo_df[combined_demo_df['is_train'] == 1].drop(columns=['is_train'])
test_demo_processed = combined_demo_df[combined_demo_df['is_train'] == 0].drop(columns=['is_train'])

# --- Get subject mapping from main files ---
subject_map_train = train_df[['sequence_id', 'subject']].drop_duplicates()
subject_map_test = test_df[['sequence_id', 'subject']].drop_duplicates()

# --- Merge Demo Features with Sensor Features ---
X_train_features = pd.merge(X_train_sensor_features, subject_map_train, on='sequence_id')
X_test_features = pd.merge(X_test_sensor_features, subject_map_test, on='sequence_id')

X_train_features = pd.merge(X_train_features, train_demo_processed, on='subject', how='left')
X_test_features = pd.merge(X_test_features, test_demo_processed, on='subject', how='left')

# Fill NaNs
demo_feature_cols = [col for col in train_demo_processed.columns if col != 'subject']
X_train_features[demo_feature_cols] = X_train_features[demo_feature_cols].fillna(X_train_features[demo_feature_cols].median())
X_test_features[demo_feature_cols] = X_test_features[demo_feature_cols].fillna(X_train_features[demo_feature_cols].median()) 

print("Demographic features merged.")
print(f"Final train features shape: {X_train_features.shape}")
print(f"Final test features shape: {X_test_features.shape}")


# --- Prepare Training Data (X and y) ---

# Merge features with labels (from Step 2)
train_data = pd.merge(labels_df[['sequence_id', 'gesture']], X_train_features, on='sequence_id', how='inner')

# --- Label Encoding for the target 'gesture' ---
le = LabelEncoder()
train_data['gesture_encoded'] = le.fit_transform(train_data['gesture'])

# Get the list of ALL feature columns
feature_columns = [
    col for col in X_train_features.columns if col not in ['sequence_id', 'subject', 'gesture']
]

# Define our final X and y
X = train_data[feature_columns]
y = train_data['gesture_encoded']

# Align test set columns
X_test = X_test_features[feature_columns]

print(f"Final X_train shape: {X.shape}")
print(f"Final y_train shape: {y.shape}")
print(f"Final X_test shape: {X_test.shape}")
print(f"Found {len(le.classes_)} unique gestures: {le.classes_}")


print("\n--- Step 6: Start Model Training (with Random Forest) ---")

# Import the new model
from sklearn.ensemble import RandomForestClassifier

N_CLASSES = len(le.classes_)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), N_CLASSES)) 
test_preds = np.zeros((len(X_test), N_CLASSES)) 
models = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,      
        random_state=42,
        n_jobs=-1,             
        min_samples_leaf=5,    
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Store predictions
    val_preds = model.predict_proba(X_val)
    oof_preds[val_index] = val_preds
    
    # Predict on test set
    test_preds += model.predict_proba(X_test) / N_SPLITS
    models.append(model)
    
    # --- LOCAL SCORE CHECK ---
    
    # !! ==================== FIX IS HERE ==================== !!
    # !! You MUST update this list with your competition's "target" gestures !!
    # !! Use the output from print(list(le.classes_)) to see your options.
    
    TARGET_GESTURES = ['A', 'C', 'F'] # <--- EDIT THIS LINE (This is just an example)
    
    # !! ===================================================== !!
    
    
    y_val_labels_pred = le.inverse_transform(np.argmax(val_preds, axis=1))
    y_val_labels_true = le.inverse_transform(y_val)
    
    y_val_binary_pred = [1 if g in TARGET_GESTURES else 0 for g in y_val_labels_pred]
    y_val_binary_true = [1 if g in TARGET_GESTURES else 0 for g in y_val_labels_true]
    
    y_val_macro_pred = [g if g in TARGET_GESTURES else 'non_target' for g in y_val_labels_pred]
    y_val_macro_true = [g if g in TARGET_GESTURES else 'non_target' for g in y_val_labels_true]

    binary_f1 = f1_score(y_val_binary_true, y_val_binary_pred, average='binary', zero_division=0)
    macro_f1 = f1_score(y_val_macro_true, y_val_macro_pred, average='macro', zero_division=0)
    final_score = (binary_f1 + macro_f1) / 2
    
    print(f"Fold {fold+1} Binary F1: {binary_f1:.4f}")
    print(f"Fold {fold+1} Macro F1: {macro_f1:.4f}")
    print(f"Fold {fold+1} Final Score: {final_score:.4f}")

print("\n--- Training Complete ---")


# --- Step 7: Create Submission File (Unchanged) ---
print("\n--- Step 7: Create Submission File ---")

# Get the class with the highest probability
final_test_preds_encoded = np.argmax(test_preds, axis=1)

# Decode the integer predictions back to string labels
final_test_preds_labels = le.inverse_transform(final_test_preds_encoded)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'sequence_id': X_test_features['sequence_id'],
    'gesture': final_test_preds_labels
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission_df.head())




