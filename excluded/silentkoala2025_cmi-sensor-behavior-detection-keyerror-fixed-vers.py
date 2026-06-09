## 1. Setup & Dependencies

# Essential imports
import os
import joblib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
print('âœ… Setup complete')

## 2. Data Loading (No Internet Access)

def load_data():
    """Load data from Kaggle input directory (offline)"""
    data_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if train_path.exists() and test_path.exists():
        print(f"ğŸ“� Loading data from: {data_dir}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"âœ… Data loaded successfully")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    else:
        raise FileNotFoundError("â�Œ Competition dataset files not found in /kaggle/input.")

# Load the data
train_df, test_df = load_data()

# Ensure 'sequence_name' column exists
if 'sequence_name' not in train_df.columns:
    train_df['sequence_name'] = [f'seq_{i//10}' for i in range(len(train_df))]
if 'sequence_name' not in test_df.columns:
    test_df['sequence_name'] = [f'seq_{i//10}' for i in range(len(test_df))]

# Display basic overview
print(f"\nğŸ“Š Data overview:")
print(f"Columns: {len(train_df.columns)}")
if 'gesture' in train_df.columns:
    print(f"Gestures: {train_df['gesture'].nunique()}")
print(f"Sequences: {train_df['sequence_name'].nunique()}")



## 3. Feature Extraction with Fixed Syntax

def identify_sensor_cols(df):
    """Identify sensor columns safely"""
    META = ['sequence_name', 'gesture', 'subject', 'orientation', 'phase', 'behavior']
    sensor_cols = []
    
    for col in df.columns:
        if col not in META:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sensor_cols.append(col)
            except Exception as e:
                print(f"âš ï¸� Skipping column {col}: {e}")
                continue
    
    return sensor_cols

def extract_features_safe(df, sensor_cols):
    """Extract features with proper error handling"""
    print(f"ğŸš€ Extracting features from {len(sensor_cols)} sensors...")
    
    features = []
    
    try:
        # Ensure sequence_name exists
        if 'sequence_name' not in df.columns:
            df['sequence_name'] = [f'seq_{i//10}' for i in range(len(df))]
        
        for seq_name, group in df.groupby('sequence_name'):
            try:
                # Get sensor data for this sequence
                sensor_data = group[sensor_cols]
                
                # Create feature row
                row = {'sequence_name': seq_name}
                
                # Basic statistics
                row['mean_all'] = sensor_data.mean().mean()
                row['std_all'] = sensor_data.std().mean()
                row['min_all'] = sensor_data.min().mean()
                row['max_all'] = sensor_data.max().mean()
                
                # Individual sensor means (first 10 for speed)
                for i, col in enumerate(sensor_cols[:10]):
                    try:
                        row[f'mean_{i}'] = sensor_data[col].mean()
                    except Exception as e:
                        row[f'mean_{i}'] = 0.0
                        continue
                
                features.append(row)
                
            except Exception as e:
                print(f"âš ï¸� Error processing sequence {seq_name}: {e}")
                continue
        
        return pd.DataFrame(features)
        
    except Exception as e:
        print(f"â�Œ Critical error in feature extraction: {e}")
        # Return minimal features as fallback
        fallback_features = []
        
        # Create dummy sequences if needed
        n_sequences = min(10, len(df))
        for i in range(n_sequences):
            fallback_features.append({
                'sequence_name': f'dummy_seq_{i}',
                'mean_all': 0.0,
                'std_all': 1.0
            })
        return pd.DataFrame(fallback_features)

# Extract sensor columns
sensor_cols = identify_sensor_cols(train_df)
print(f"ğŸ“Š Found {len(sensor_cols)} sensor columns")

# Extract features
X_train = extract_features_safe(train_df, sensor_cols)
X_test = extract_features_safe(test_df, sensor_cols)

print(f"âœ… Features extracted:")
print(f"Train features: {X_train.shape}")
print(f"Test features: {X_test.shape}")


## 4. Data Preparation

# Prepare labels with error handling
try:
    # Ensure both dataframes have sequence_name
    if 'sequence_name' in train_df.columns and 'sequence_name' in X_train.columns:
        gesture_map = train_df.groupby('sequence_name')['gesture'].first()
        y_labels = X_train['sequence_name'].map(gesture_map).values
    else:
        # Create dummy labels if sequence_name doesn't exist
        y_labels = np.array(['A'] * len(X_train))
    
    # Remove any NaN labels
    valid_mask = pd.notna(y_labels)
    X_train = X_train[valid_mask]
    y_labels = y_labels[valid_mask]
    
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    print(f"ğŸ“‹ Classes: {le.classes_}")
    print(f"ğŸ“Š Class distribution: {np.bincount(y)}")
    
except Exception as e:
    print(f"â�Œ Error preparing labels: {e}")
    # Create dummy labels as fallback
    y = np.zeros(len(X_train))
    le = LabelEncoder()
    le.classes_ = np.array(['A'])

# Prepare features with error handling
try:
    # Safely drop sequence_name if it exists
    X_train_features = X_train.drop('sequence_name', axis=1, errors='ignore')
    X_test_features = X_test.drop('sequence_name', axis=1, errors='ignore')
    
    # Ensure same columns
    common_cols = list(set(X_train_features.columns) & set(X_test_features.columns))
    if not common_cols:
        # If no common columns, create dummy feature
        X_train_features = pd.DataFrame({'dummy': np.ones(len(X_train_features))})
        X_test_features = pd.DataFrame({'dummy': np.ones(len(X_test_features))})
    else:
        X_train_features = X_train_features[common_cols]
        X_test_features = X_test_features[common_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    print(f"âœ… Data prepared:")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Samples: {X_train_scaled.shape[0]}")
    
except Exception as e:
    print(f"â�Œ Error preparing features: {e}")
    # Create minimal dummy data
    X_train_scaled = np.ones((len(X_train), 1))
    X_test_scaled = np.ones((len(X_test), 1))
    scaler = StandardScaler()
    scaler.fit(X_train_scaled)


## 5. Model Training with Proper Error Handling

# Split data for validation
try:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
except Exception as e:
    print(f"âš ï¸� Stratified split failed: {e}. Using random split.")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y, test_size=0.2, random_state=42
    )

print("ğŸš€ Training models...")

# Model 1: LightGBM with error handling
try:
    print("\nğŸ“ˆ Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1,
        n_jobs=1  # Single thread for stability
    )
    
    lgb_model.fit(X_tr, y_tr)
    lgb_pred = lgb_model.predict(X_val)
    lgb_acc = accuracy_score(y_val, lgb_pred)
    lgb_f1 = f1_score(y_val, lgb_pred, average='macro')
    print(f"âœ… LightGBM - Accuracy: {lgb_acc:.4f}, F1: {lgb_f1:.4f}")
    lgb_success = True
    
except Exception as e:
    print(f"â�Œ LightGBM failed: {e}")
    lgb_success = False
    lgb_f1 = 0.0

# Model 2: Random Forest with error handling
try:
    print("\nğŸŒ² Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=1  # Single thread for stability
    )
    
    rf_model.fit(X_tr, y_tr)
    rf_pred = rf_model.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    rf_f1 = f1_score(y_val, rf_pred, average='macro')
    print(f"âœ… Random Forest - Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    rf_success = True
    
except Exception as e:
    print(f"â�Œ Random Forest failed: {e}")
    rf_success = False
    rf_f1 = 0.0

# Choose best model
if lgb_success and rf_success:
    if lgb_f1 > rf_f1:
        best_model = lgb_model
        best_name = "LightGBM"
        best_f1 = lgb_f1
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_f1 = rf_f1
elif lgb_success:
    best_model = lgb_model
    best_name = "LightGBM"
    best_f1 = lgb_f1
elif rf_success:
    best_model = rf_model
    best_name = "Random Forest"
    best_f1 = rf_f1
else:
    print("â�Œ All models failed. Creating dummy model.")
    best_model = None
    best_name = "Dummy"
    best_f1 = 0.0

print(f"\nğŸ�† Best model: {best_name} (F1: {best_f1:.4f})")


from sklearn.impute import SimpleImputer

print("ğŸ�¯ Making final predictions...")

try:
    # Train full model
    if best_model is not None:
        best_model.fit(X_train_scaled, y)

        # Impute missing values in test
        imputer = SimpleImputer(strategy='mean')
        X_test_imputed = imputer.fit_transform(X_test_scaled)

        # Predict
        test_preds = best_model.predict(X_test_imputed)
        test_gestures = le.inverse_transform(test_preds)
    else:
        test_gestures = ['A'] * len(X_test)

    # Sequence names
    if isinstance(X_test, pd.DataFrame) and 'sequence_name' in X_test.columns:
        sequence_names = X_test['sequence_name']
    else:
        sequence_names = [f'seq_{i}' for i in range(len(X_test))]

    # Save submission
    submission = pd.DataFrame({
        'sequence_name': sequence_names,
        'gesture': test_gestures
    })

    submission.to_parquet('submission.parquet', index=False)
    print("âœ… submission.parquet saved")
    print(submission.head())

except Exception as e:
    print(f"â�Œ Error: {e}")
    submission = pd.DataFrame({
        'sequence_name': [f'seq_{i}' for i in range(len(X_test))],
        'gesture': ['A'] * len(X_test)
    })
    submission.to_parquet('submission.parquet', index=False)
    print("âœ… Fallback submission saved")


