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


!pip install protobuf==3.20.3 --force-reinstall --no-cache-dir



import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import os, time, math, random, gc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV


# === CONFIGURATION ===
# Using the user-provided file paths
TRAIN_CSV = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
TEST_CSV = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
DEMOG_TRAIN = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
DEMOG_TEST = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'

FEATURE_OUT = '/kaggle/working/sequence_features.csv'
RANDOM_SEED = 42
WINDOW_SIZE = 30 # A tunable hyperparameter

# Define sensor column groups
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
IMU_COLS = ACC_COLS + ROT_COLS
THM_COLS = [f'thm_{i}' for i in range(1, 6)]
TOF_COLS = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(0, 64)]


print("DONE")



# STEP 1: FEATURE ENGINEERING FUNCTIONS
# ===============================================

def extract_imu_features(window_df):
    """Extracts Time-Domain, Frequency-Domain, and statistical IMU features."""
    features = {}
    
    # ... (other code for Time-Domain features)
    
    # Time-Domain and Statistical Features (CORRECTED SECTION)
    for col in IMU_COLS:
        col_prefix = col
        features[f'{col_prefix}_mean'] = window_df[col].mean()
        features[f'{col_prefix}_var'] = window_df[col].var()
        
        # Explicitly check sample size before calculating skew and kurtosis
        data = window_df[col].dropna()
        MIN_SAMPLE_SIZE = 8 # Set minimum required samples (e.g., 8 is a common threshold)
        
        if len(data) >= MIN_SAMPLE_SIZE:
            features[f'{col_prefix}_skew'] = skew(data)
            features[f'{col_prefix}_kurt'] = kurtosis(data)
        else:
            # If sample is too small, set to NaN (which the imputer will handle later)
            features[f'{col_prefix}_skew'] = np.nan
            features[f'{col_prefix}_kurt'] = np.nan
    #Signal Magnitude Area (SMA) and Total Magnitude
    acc_mag = np.sqrt(np.square(window_df[ACC_COLS]).sum(axis=1))
    features['acc_SMA'] = acc_mag.sum() / WINDOW_SIZE
    features['acc_mag_mean'] = acc_mag.mean()
    features['acc_mag_var'] = acc_mag.var()

    # Cross-Axis Correlation
    features['acc_corr_xy'] = window_df['acc_x'].corr(window_df['acc_y'])
    
    # Frequency-Domain Features (FFT magnitude of first few non-DC components)
    for col in IMU_COLS:
        if len(window_df[col].dropna()) > 0:
            fft_values = np.abs(fft(window_df[col].fillna(window_df[col].mean()).values))
            for i in range(1, 6):
                features[f'{col}_fft_mag_{i}'] = fft_values[i]
        else:
             for i in range(1, 6):
                features[f'{col}_fft_mag_{i}'] = np.nan
            
    return features


def extract_thermopile_features(window_df):
    """Extracts features from Thermopile data (thm_1 to thm_5), including gradients."""
    features = {}

    # Simple Statistical Features (Average Heat Intensity, Temporal Variance)
    for col in THM_COLS:
        col_prefix = col
        features[f'{col_prefix}_mean'] = window_df[col].mean()
        features[f'{col_prefix}_var'] = window_df[col].var()

    # Thermopile Gradient Features (Difference between adjacent sensors)
    for i in range(1, 5):
        grad_col = f'thm_grad_{i}_{i+1}'
        gradient = window_df[f'thm_{i}'] - window_df[f'thm_{i+1}']
        features[f'{grad_col}_mean'] = gradient.mean()
        features[f'{grad_col}_var'] = gradient.var()
        
    return features


def extract_tof_features(window_df):
    """Extracts spatial and temporal features from Time-of-Flight (ToF) data."""
    features = {}

    # Replace the 'no sensor response' value (-1) with NaN for statistics
    tof_data = window_df[TOF_COLS].replace(-1, np.nan)
    
    # Overall Proximity Features
    features['tof_min_proximity'] = tof_data.min().min() # Closest object
    features['tof_max_proximity'] = tof_data.max().max() # Farthest object
    
    # Per-Sensor (5 sensors) Spatial and Temporal Features
    for i in range(1, 6):
        sensor_cols = [f'tof_{i}_v{j}' for j in range(0, 64)]
        tof_sensor_data = tof_data[sensor_cols]
        prefix = f'tof_{i}'

        # 1. Spatial Statistics
        spatial_mean_series = tof_sensor_data.mean(axis=1) # Mean distance per row
        features[f'{prefix}_spatial_mean'] = spatial_mean_series.mean()
        features[f'{prefix}_spatial_var'] = spatial_mean_series.var()
        
        # 2. Number of 'Active' Pixels (Non-negative)
        active_pixels = (window_df[sensor_cols] != -1).sum(axis=1)
        features[f'{prefix}_active_pixels_mean'] = active_pixels.mean()
        features[f'{prefix}_active_pixels_var'] = active_pixels.var()

        # 3. Temporal Slope / Trend
        avg_proximity_series = spatial_mean_series.dropna()
        if len(avg_proximity_series) >= 2:
            # Calculate temporal slope (change in average distance over time)
            slope = np.polyfit(np.arange(len(avg_proximity_series)), avg_proximity_series, 1)[0]
            features[f'{prefix}_temporal_slope'] = slope
        else:
            features[f'{prefix}_temporal_slope'] = np.nan

    return features


# STEP 2: DATA PIPELINE (Windowing, Extraction, Merging)
# ===============================================

def run_feature_pipeline(ts_path, demog_path, is_train=True):
    """Loads, processes time-series data, and merges with demographics."""
    print(f"--- Running Pipeline for {'Train' if is_train else 'Test'} Data ---")
    
    # Load Time-Series and Demographics
    df_ts = pd.read_csv(ts_path)
    df_demog = pd.read_csv(demog_path)
    
    features_list = []
    
    # Iterate through each unique sequence
    sequence_groups = df_ts.groupby('sequence_id')
    for seq_id, group in sequence_groups:
        
        # --- Windowing Strategy ---
        # For a full sequence (Transition, Pause, Gesture), the critical event 
        # is the Gesture, which is at the end of the sequence. We use the last 
        # WINDOW_SIZE rows to capture this final motion.
        window_df = group.tail(WINDOW_SIZE).copy()
        
        if len(window_df) < WINDOW_SIZE:
            # Skip short sequences that can't form a reliable feature vector
            continue
            
        # 1. Feature Extraction
        features = {}
        features.update(extract_imu_features(window_df))
        features.update(extract_thermopile_features(window_df))
        features.update(extract_tof_features(window_df))
        
        # 2. Add identifying and target columns
        features['sequence_id'] = seq_id
        features['subject'] = window_df['subject'].iloc[0]
        if is_train:
            # The gesture column is only present in the training set
            features['gesture'] = window_df['gesture'].iloc[-1] 
        
        features_list.append(features)
        
    df_features = pd.DataFrame(features_list)
    
    # 3. Merge with Demographics (Static Features Fusion)
    df_final = pd.merge(df_features, df_demog, on='subject', how='left')
    
    print(f"Generated {len(df_final)} feature vectors. Total features: {len(df_final.columns)}")
    return df_final


# Execute pipeline for both datasets
df_train_features = run_feature_pipeline(TRAIN_CSV, DEMOG_TRAIN, is_train=True)
df_test_features = run_feature_pipeline(TEST_CSV, DEMOG_TEST, is_train=False)

# Save the features (optional, for debugging/reuse)
# df_train_features.to_csv(FEATURE_OUT.replace(".csv", "_train.csv"), index=False)


# --- Imbalance Handling Note ---
# Class Weighting (class_weight='balanced') is used for LR, DT, and RF as it's built-in 
# and often preferred inside pipelines. SMOTE, though effective, requires an external 
# library (imblearn) and must be applied BEFORE scaling, which complicates the pipeline.
def train_and_evaluate_model(df_train_features, df_test_features, random_seed):
    """Defines, trains, and evaluates all baseline, tree-based, and ensemble models."""
    
    print("\nStarting Model Development and Evaluation...")
    
    # --- Data Preparation ---
    X_train_full = df_train_features.drop(columns=['sequence_id', 'subject', 'gesture'])
    y_train_full = df_train_features['gesture']
    
    # Use a split of the TRAIN data for internal evaluation (Macro F1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.3, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    # Prepare Test Data for final predictions
    X_test_seq_id = df_test_features['sequence_id']
    X_test = df_test_features.drop(columns=['sequence_id', 'subject'])
    
    # --- Pipeline Components ---
    IMPUTATION_STEP = ('imputer', SimpleImputer(strategy='median'))
    SCALING_STEP = ('scaler', StandardScaler())
    
    
    # --- 1. Baseline Models ---
    
    # 1A. Logistic Regression (Imbalanced handled by class_weight)
    lr_pipe = Pipeline([
        IMPUTATION_STEP, SCALING_STEP,
        ('clf', LogisticRegression(random_state=random_seed, solver='liblinear', multi_class='ovr', class_weight='balanced'))
    ])
    
    # 1B. k-Nearest Neighbors (No inherent imbalance handling, scaling is crucial)
    knn_pipe = Pipeline([
        IMPUTATION_STEP, SCALING_STEP,
        ('clf', KNeighborsClassifier(n_neighbors=5)) # n_neighbors is tunable
    ])
    
    
    # --- 2. Tree-based Models ---
    
    # 2A. Decision Tree (Imbalanced handled by class_weight)
    dt_pipe = Pipeline([
        IMPUTATION_STEP,
        ('clf', DecisionTreeClassifier(random_state=random_seed, class_weight='balanced'))
    ])
    
    # 2B. Random Forest (Imbalanced handled by class_weight)
    rf_pipe = Pipeline([
        IMPUTATION_STEP,
        ('clf', RandomForestClassifier(n_estimators=500, random_state=random_seed, class_weight='balanced', n_jobs=-1))
    ])
    
    
    # --- 3. Ensemble Model: Voting Classifier ---
    
    # Use the best performing and most diverse base models: LR, RF, and DT.
    # The 'named_estimators' must use the name of the step defined above.
    ensemble_clf = VotingClassifier(
        estimators=[
            ('lr', lr_pipe),
            ('rf', rf_pipe),
            ('dt', dt_pipe)
        ],
        voting='soft', # Use 'soft' voting for weighted probability combination
        n_jobs=-1
    )
    
    # Wrap the ensemble in a pipeline for consistent data preprocessing
    voting_pipe = Pipeline([
        ('ensemble', ensemble_clf)
    ])
    
    
    # --- 4. Training and Evaluation Loop ---
    
    models = {
        "Logistic Regression": lr_pipe,
        "k-Nearest Neighbors": knn_pipe,
        "Decision Tree": dt_pipe,
        "Random Forest": rf_pipe,
        "Voting Classifier": voting_pipe
    }
    
    best_model_name = ""
    best_f1 = -1.0
    
    results = {}
    for name, pipe in models.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        y_pred_val = pipe.predict(X_val)
        
        macro_f1 = f1_score(y_val, y_pred_val, average='macro')
        results[name] = macro_f1
        print(f"{name} Macro F1 Score (Validation): {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            
    # --- 5. Final Prediction using the BEST model ---
    
    final_model = models[best_model_name]
    
    # Retrain the best model on the FULL training set for best performance
    print(f"\nRetraining BEST model ({best_model_name}) on full training data...")
    final_model.fit(X_train_full, y_train_full)
    
    y_pred_test = final_model.predict(X_test)
    
    # --- Submission File Generation ---
    df_submission = pd.DataFrame({
        'sequence_id': X_test_seq_id,
        'gesture': y_pred_test
    })
    
    # submission_path = os.path.join('/kaggle/working', 'submission_ensemble.csv')
    # df_submission.to_csv(submission_path, index=False)
    
    print("\n--- Summary of Validation F1 Scores ---")
    for name, f1 in results.items():
         print(f"{name}: {f1:.4f}")
         
    print(f"\nFinal prediction made using: {best_model_name}")
    # print(f"Submission file created at: {submission_path}")
    # print(df_submission.head())
    
# Execute the final modeling step
train_and_evaluate_model(df_train_features, df_test_features, RANDOM_SEED)


def train_and_evaluate_xgb(df_train_features, df_test_features, random_seed):

    print("\n=== Improving Best Model: Using Optimized XGBoost ===")
    le = LabelEncoder()
    y_full = le.fit_transform(df_train_features['gesture'])

    # -------------------------
    # Feature Matrix
    # -------------------------
    X_full = df_train_features.drop(columns=['sequence_id', 'subject', 'gesture'])

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.3,
        random_state=random_seed, stratify=y_full
    )

    X_test_seq_id = df_test_features['sequence_id']
    X_test = df_test_features.drop(columns=['sequence_id', 'subject'])

    # -------------------------
    # Pipeline with Transformer
    # -------------------------
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('qt', QuantileTransformer(output_distribution='normal', random_state=random_seed)),
        ('clf', xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=random_seed,
            nthread=-1
        ))
    ])

    # -------------------------
    # Hyperparameter Search
    # -------------------------
    param_dist = {
        'clf__n_estimators': [300, 500, 700, 900],
        'clf__max_depth': [3, 4, 5, 6, 7],
        'clf__learning_rate': np.linspace(0.01, 0.2, 10),
        'clf__subsample': np.linspace(0.6, 1.0, 5),
        'clf__colsample_bytree': np.linspace(0.6, 1.0, 5),
        'clf__reg_lambda': np.logspace(-2, 2, 8),
        'clf__reg_alpha': np.logspace(-2, 1, 6)
    }

    print("\nRunning RandomizedSearchCV... (This takes ~10–25 minutes)")
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_iter=30,
        random_state=random_seed,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest Parameters Found:")
    print(search.best_params_)

    # -------------------------
    # Validation Performance
    # -------------------------
    y_val_pred = best_model.predict(X_val)
    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"\nOpdtimized XGBoost Macro F1 on Validation: {macro_f1:.4f}")

    # -------------------------
    # Retrain on Full Training Data
    # -------------------------
    print("\nRetraining Optimized Model on FULL training data...")
    best_model.fit(X_full, y_full)

    # -------------------------
    # Final Test Predictions
    # -------------------------
    y_test_pred = best_model.predict(X_test)

    df_submission = pd.DataFrame({
        'sequence_id': X_test_seq_id,
        'gesture': y_test_pred
    })

    # submission_path = os.path.join('/kaggle/working', 'submission_xgb.csv')
    # df_submission.to_csv(submission_path, index=False)

    # print(f"\nSubmission saved to: {submission_path}")
    # print(df_submission.head())

# Run improved model
train_and_evaluate_xgb(df_train_features, df_test_features, RANDOM_SEED)


os.environ['TF_USE_CUDNN'] = '0'
os.environ['TF_USE_CUBLAS'] = '0'


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization, Input, Flatten, Bidirectional, LSTM
)
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam



X_train_full = df_train_features.drop(columns=['sequence_id', 'subject', 'gesture'])
y_train_full = df_train_features['gesture']

# Encode y
le = LabelEncoder()
y_train_full_enc = le.fit_transform(y_train_full)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full_enc,
    test_size=0.3,
    random_state=42,
    stratify=y_train_full_enc
)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)


# Convert to numpy
X_train_np = X_train.values
X_val_np   = X_val.values

# Replace NaN → 0, keep safe values for CNN
X_train_clean = np.nan_to_num(X_train_np, nan=0.0)
X_val_clean   = np.nan_to_num(X_val_np, nan=0.0)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_val_scaled   = scaler.transform(X_val_clean)


X_train_cnn = X_train_scaled.reshape(-1, 119, 1)
X_val_cnn   = X_val_scaled.reshape(-1, 119, 1)

num_classes = len(np.unique(y_train))
print("Reshaped Train:", X_train_cnn.shape)
print("Number of classes:", num_classes)



num_classes = 18   # detected from your data

model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(119, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(256, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


y_train = y_train.astype(int)
y_val = y_val.astype(int)


history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=50,
    batch_size=64
)


y_val_pred = np.argmax(model.predict(X_val_cnn), axis=1)

val_acc = (y_val_pred == y_val).mean()
val_f1  = f1_score(y_val, y_val_pred, average='macro')

print("Validation Accuracy:", val_acc)
print("Validation Macro F1:", val_f1)



#To improve the val_acc and val_f1, adding more dropouts
model1 = Sequential([
    # Block 1
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(119, 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    # Block 2
    Conv1D(filters=256, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    # Block 3
    Conv1D(filters=256, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    # Block 4 (new)
    Conv1D(filters=256, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),

    # Dense Layer 1 (more capacity)
    Dense(256, activation='relu'),
    Dropout(0.4),

    # Dense Layer 2 (new)
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(18, activation='softmax')
])

# Lower learning rate for better generalization
optimizer = Adam(learning_rate=0.0005)

model1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model1.summary()


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))


history1 = model1.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=60,
    batch_size=64,
    class_weight=class_weights
)





y_val_pred = np.argmax(model1.predict(X_val_cnn), axis=1)

val_acc = (y_val_pred == y_val).mean()
val_f1  = f1_score(y_val, y_val_pred, average='macro')

print("Validation Accuracy:", val_acc)
print("Validation Macro F1:", val_f1)


# Prepare test data
X_test = df_test_features.drop(columns=['sequence_id', 'subject'])

X_test_cnn = X_test.values.reshape(-1, X_test.shape[1], 1)

# Predictions
test_preds = np.argmax(model.predict(X_test_cnn), axis=1)

# Convert back to original gesture labels
test_gestures = le.inverse_transform(test_preds)

# Build submission dataframe
submission = pd.DataFrame({
    "sequence_id": df_test_features["sequence_id"],
    "gesture": test_gestures
})

# submission.to_csv("submission_1dcnn.csv", index=False)

submission.head()



X_test = df_test_features.drop(columns=['sequence_id', 'subject'])

# Reshape for CNN
X_test_cnn = X_test.values.astype('float32').reshape(-1, X_test.shape[1], 1)

# Random 10
n_samples = min(10, len(X_test))
random_indices = np.random.choice(len(X_test), size=n_samples, replace=False)

X_sample_cnn = X_test_cnn[random_indices]

# Predict
sample_preds = np.argmax(model.predict(X_sample_cnn), axis=1)
sample_gestures = le.inverse_transform(sample_preds)

# Results
results = pd.DataFrame({
    "sequence_id": df_test_features.iloc[random_indices]["sequence_id"].values,
    "predicted_gesture": sample_gestures
})

print(results)


import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_NAME = "AdamW"



N, T, H, W, C = 100, 10, 8, 8, 3   
X = np.random.rand(N, T, H, W, C).astype(np.float32)
y = np.random.randint(0, 2, size=(N,))  

X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]


class GraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, X, A):
        return torch.matmul(A, self.lin(X))

class SimpleGCN(torch.nn.Module):
    def __init__(self, H, W, in_ch, nclass):
        super().__init__()
        self.N = H * W
        self.gc1 = GraphConv(in_ch, 32)
        self.gc2 = GraphConv(32, 64)
        self.fc = torch.nn.Linear(64, nclass)

    def forward(self, x, A):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.reshape(B, T, H * W, C)
        outs = []
        for t in range(T):
            h = F.relu(self.gc1(x[:, t], A))
            h = F.relu(self.gc2(h, A))
            pooled = h.mean(1)
            outs.append(pooled)
        out = torch.stack(outs, 1).mean(1)
        return self.fc(out)

class SimpleGAT(torch.nn.Module):
    def __init__(self, H, W, in_ch, nclass, heads=4):
        super().__init__()
        self.N = H * W
        self.fc1 = torch.nn.Linear(in_ch, 32)
        self.attn = torch.nn.MultiheadAttention(32, num_heads=heads, batch_first=True)
        self.fc2 = torch.nn.Linear(32, nclass)

    def forward(self, x, A=None):  # A không dùng, nhưng giữ để tương thích
        B, T, H, W, C = x.shape
        x = x.reshape(B, T, H * W, C)
        outs = []
        for t in range(T):
            h = F.relu(self.fc1(x[:, t]))
            attn_out, _ = self.attn(h, h, h)
            pooled = attn_out.mean(1)
            outs.append(pooled)
        out = torch.stack(outs, 1).mean(1)
        return self.fc2(out)

class CNN2D(torch.nn.Module):
    def __init__(self, nclass, in_ch):
        super().__init__()
        self.frame = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.temporal = torch.nn.LSTM(64, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, nclass)

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t].permute(0, 3, 1, 2)  # -> (B, C, H, W)
            feat = self.frame(xt).flatten(1)
            feats.append(feat)
        z = torch.stack(feats, 1)
        z, _ = self.temporal(z)
        z = z.mean(1)
        return self.fc(z)

class CNN3D(torch.nn.Module):
    def __init__(self, nclass, in_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((2,2,2)),
            torch.nn.Conv3d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1)
        )
        self.fc = torch.nn.Linear(32, nclass)

    def forward(self, x):
        # x: (B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # -> (B, C, T, H, W)
        z = self.net(x).flatten(1)
        return self.fc(z)


X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

scaler = StandardScaler().fit(X_train_reshaped)
X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val = scaler.transform(X_val_reshaped).reshape(X_val.shape)


def run_experiments(X_train, y_train, X_val, y_val, H=8, W=8, mask_channel=None, epochs=80):
    import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, pandas as pd
    from sklearn.metrics import f1_score, accuracy_score

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(np.unique(y_train))
    in_channels = X_train.shape[-1]

    # === Helper: adjacency 8-neighbor ===
    def build_adjacency(H, W):
        A = torch.zeros(H*W, H*W)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W:
                            n_idx = ni * W + nj
                            A[idx, n_idx] = 1
        D_inv = torch.diag(1.0 / (A.sum(1) + 1e-5))
        return (D_inv @ A).to(DEVICE)

    A_hat = build_adjacency(H, W)

    # === DataLoader ===
    bs = 32
    dl_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.long)),
        batch_size=bs, shuffle=True)
    dl_va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                       torch.tensor(y_val, dtype=torch.long)),
        batch_size=bs, shuffle=False)

    # === Models ===
    models = {
        "GCN": lambda: SimpleGCN(H, W, in_ch=in_channels, nclass=n_classes),
        "GAT": lambda: SimpleGAT(H, W, in_ch=in_channels, nclass=n_classes),
        "CNN2D": lambda: CNN2D(nclass=n_classes, in_ch=in_channels),
        "CNN3D": lambda: CNN3D(nclass=n_classes, in_ch=in_channels),
    }

    # === Evaluate ===
    @torch.no_grad()
    def evaluate(model, dl, A=None):
        model.eval()
        y_true, y_pred = [], []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, A) if A is not None else model(xb)
            y_pred += logits.argmax(1).cpu().tolist()
            y_true += yb.cpu().tolist()
        macro = f1_score(y_true, y_pred, average='macro')
        binary = f1_score(y_true, y_pred, average='binary') if len(set(y_true)) == 2 else 0
        acc = accuracy_score(y_true, y_pred)
        return binary, macro, acc

    results = []

    # === Train each model ===
    for name, fn in models.items():
        print(f"\n▶ Training {name} ...")
        model = fn().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        best_macro = 0
        patience = 0
        start = time.time()

        for ep in range(1, epochs+1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb, A_hat) if name in ["GCN", "GAT"] else model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
            scheduler.step()

            bin_f1, mac_f1, acc = evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
            if ep % 5 == 0:
                print(f"Epoch {ep}/{epochs}: loss={loss.item():.3f}, Macro={mac_f1:.3f}, Acc={acc:.3f}")
            if mac_f1 > best_macro:
                best_macro = mac_f1
                patience = 0
            else:
                patience += 1
                if patience > 10:
                    print(f"⏹ Early stop at epoch {ep}")
                    break

        train_time = time.time() - start
        start_inf = time.time()
        evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
        inf_time = (time.time() - start_inf) / len(dl_va.dataset)

        bin_f1, mac_f1, acc = evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
        results.append({
            "features_used": "ToF_Grid",
            "window_size": 10,
            "optimizer/solver": "AdamW+Cosine",
            "params (M)": round(params_m, 4),
            "Binary": round(bin_f1, 4),
            "Macro": round(mac_f1, 4),
            "Final Score": round(mac_f1/2, 4),
            "val_acc": round(acc, 4),
            "train_time": round(train_time, 2),
            "inference_time": round(inf_time, 4),
        })

    df = pd.DataFrame(results)
    print("\n=== RESULTS ===")
    print(df)
    return df


report = run_experiments(X_train, y_train, X_val, y_val, H=8, W=8, epochs=80)

# Define output path
output_path = "/kaggle/working/submission.parquet"

# Save the submission file in the correct format and location
report.to_parquet(output_path, index=False)

print(f"✅ Submission file saved successfully at: {output_path}")


import kaggle_evaluation.cmi_inference_server


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

