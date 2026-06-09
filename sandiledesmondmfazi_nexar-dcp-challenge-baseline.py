# Core Imports
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
from joblib import Parallel, delayed
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')
import keras_tuner as kt

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import EfficientNetB0

# Machine Learning
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV


def extract_critical_frames(video_path, alert_time, event_time, num_frames=8, sampling_interval=30):
    """Optimized frame extraction without redundant capture"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sampling_interval == 0:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        frame_count += 1
    
    cap.release()
    return np.array(frames[:num_frames])  # Return exactly num_frames

def calculate_optical_flow(frames):
    """Calculate dense optical flow between consecutive frames"""
    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(np.linalg.norm(flow, axis=2))
        prev_gray = gray
    
    return np.array(flows)


# Initialize feature extractor
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
cnn_feature_dim = base_model.output_shape[-1]

def get_hybrid_features(video_path, alert_time, event_time):
    """Optimized feature extraction with proper resource handling"""
    # Use optimized frame extraction
    frames = extract_critical_frames(
        video_path, 
        alert_time, 
        event_time,
        num_frames=8,  # Reduced from 16
        sampling_interval=30  # Process 1 frame per second (30fps video)
    )
    
    if len(frames) == 0:
        return np.zeros(1280 + 1)  # EfficientNetB0 features + flow feature
    
    # Batch process spatial features
    spatial_features = base_model.predict(
        preprocess_input(frames.astype('float32')),
        batch_size=32,  # Process 32 frames at once
        verbose=0
    )
    
    # Simplified temporal feature
    flow_feature = 0.0
    if len(frames) > 1:
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_feature = np.mean(np.linalg.norm(flow, axis=2))
    
    return np.concatenate([
        np.mean(spatial_features, axis=0),
        [flow_feature]
    ])


def build_temporal_model(hp):
    model = Sequential([
        # Tune the number of LSTM units
        Bidirectional(LSTM(
            units=hp.Int("lstm_units", min_value=32, max_value=256, step=32),
            input_shape=(1, X_train_scaled.shape[1])
        )),
        
        # Fully connected layer (Dense)
        Dense(
            units=hp.Int("dense_units", min_value=16, max_value=128, step=16),
            activation="relu"
        ),
        
        # Dropout for regularization
        Dropout(hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)),
        
        # Output layer
        Dense(1, activation="sigmoid")
    ])
    
    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    
    return model


# Load and preprocess data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
train_df['id'] = train_df['id'].apply(lambda x: f"{int(float(x)):05d}")
train_df.fillna({'time_of_alert': 0, 'time_of_event': 0}, inplace=True)

# Feature extraction
print("Extracting hybrid features...")
features = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Hybrid features extracted:"):
    video_path = f"/kaggle/input/nexar-collision-prediction/train/{row['id']}.mp4"
    features.append(get_hybrid_features(
        video_path, row['time_of_alert'], row['time_of_event']
    ))

X = np.array(features)
y = train_df['target'].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Temporal model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reshape for LSTM [samples, timesteps, features]
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_3d = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Initialize the tuner
tuner = kt.Hyperband(
    build_temporal_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="kt_logs",
    project_name="temporal_model_tuning"
)

# Perform hyperparameter search
tuner.search(
    X_train_3d, y_train,
    validation_data=(X_val_3d, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# Retrieve the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()


# temporal_model = build_temporal_model((1, X_train_scaled.shape[1]))
history = best_model.fit(
    X_train_3d, y_train,
    validation_data=(X_val_3d, y_val),
    epochs=30,  # Reduced epochs
    batch_size=64,  # Increased batch size
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        LearningRateScheduler(lambda epoch: 0.001 * (0.95 ** epoch))
    ]
)


# Generate temporal model predictions
train_probs = best_model.predict(X_train_3d).flatten()
val_probs = best_model.predict(X_val_3d).flatten()

# Create ensemble dataset
X_train_ensemble = np.column_stack([train_probs, X_train_scaled])
X_val_ensemble = np.column_stack([val_probs, X_val_scaled])

# Define hyperparameter search space
param_dist = {
    'device': ['cuda'],
    'tree_method': ['hist'],
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),  # 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'n_estimators': randint(100, 500)
}

# Create Bayesian-optimized search
optimizer = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False
        # tree_method='gpu_hist'  # Enable GPU acceleration
    ),
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter combinations
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Run optimization
optimizer.fit(X_train_ensemble, y_train)

# Best model evaluation
best_xgb = optimizer.best_estimator_
best_xgb.fit(X_train_ensemble, y_train,
            eval_set=[(X_val_ensemble, y_val)],
            early_stopping_rounds=20,
            verbose=False)

ensemble_val_probs = best_xgb.predict_proba(X_val_ensemble)[:, 1]
print(f"Optimized AUC: {roc_auc_score(y_val, ensemble_val_probs):.4f}")
print("Best parameters:", optimizer.best_params_)


# Process test set
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
test_df['id'] = test_df['id'].apply(lambda x: f"{int(float(x)):05d}")

test_features = []
for _, row in tqdm(test_df.iterrows(), desc="Processing Test Videos"):
    video_path = f"/kaggle/input/nexar-collision-prediction/test/{row['id']}.mp4"
    test_features.append(get_hybrid_features(video_path, 0, 0))  # No event times in test

X_test = scaler.transform(np.array(test_features))
X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Generate predictions
temporal_probs = best_model.predict(X_test_3d).flatten()
X_test_ensemble = np.column_stack([temporal_probs, X_test])

final_probs = best_xgb.predict_proba(X_test_ensemble)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'score': final_probs
})
submission.to_csv('submission.csv', index=False)

print("\nSubmission Summary:")
print(submission.describe())

