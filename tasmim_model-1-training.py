import os, json, joblib, numpy as np, pandas as pd
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl
from scipy.spatial.transform import Rotation as R

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything(seed=42)

# Training Configuration
TRAIN = True
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")  # Update this to your data directory
EXPORT_DIR = Path("./models")
EXPORT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 100
PATIENCE = 40
N_FOLDS = 5

print("â–¶ Training mode activated Â· tensorflow", tf.__version__)

#=============================================================================
# Utility Functions (same as inference)
#=============================================================================

def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = se_block(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                         kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def remove_gravity_from_acc(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data
    
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
    
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    
    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]
        
        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)
    
    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]
        
        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
    
    return angular_dist

def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)
    
    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)
    
    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    
    merged = Concatenate()([x1, x2])
    
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)
    
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)
    
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

#=============================================================================
# Data Loading and Feature Engineering
#=============================================================================

def load_and_preprocess_data():
    """Load and preprocess the training data"""
    print("Loading training data...")
    
    # Load main training data
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    train_demographics = pd.read_csv(RAW_DIR / "train_demographics.csv")
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Unique sequences: {train_df['sequence_id'].nunique()}")
    print(f"Gesture distribution:\n{train_df['gesture'].value_counts()}")
    
    return train_df, train_demographics

def engineer_features(df_seq):
    """Apply feature engineering to a sequence"""
    # Remove gravity from accelerometer data
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'] = linear_accel[:, 0]
    df_seq['linear_acc_y'] = linear_accel[:, 1] 
    df_seq['linear_acc_z'] = linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + 
                                      df_seq['linear_acc_y']**2 + 
                                      df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    # Calculate angular velocity
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'] = angular_vel[:, 0]
    df_seq['angular_vel_y'] = angular_vel[:, 1]
    df_seq['angular_vel_z'] = angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)
    
    # TOF sensor aggregations
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df_seq[pixel_cols].replace(-1, np.nan)
        df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
        df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
        df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
    
    return df_seq

def prepare_dataset(train_df, train_demographics):
    """Prepare the complete dataset for training"""
    print("Preparing dataset with feature engineering...")
    
    # Define feature columns (same as inference)
    base_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # Thermal features
    thermal_features = [f'thm_{i}' for i in range(1, 6)]
    
    # Engineered features
    engineered_features = [
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 
        'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
        'angular_distance'
    ]
    
    # TOF aggregated features
    tof_agg_features = []
    for i in range(1, 6):
        tof_agg_features.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    
    # TOF pixel features
    tof_pixel_features = []
    for i in range(1, 6):
        for p in range(64):
            tof_pixel_features.append(f'tof_{i}_v{p}')
    
    feature_cols = base_features + thermal_features + engineered_features + tof_agg_features + tof_pixel_features
    
    # Process sequences
    sequences = []
    labels = []
    subjects = []
    sequence_ids = []
    
    unique_seq_ids = train_df['sequence_id'].unique()
    total_sequences = len(unique_seq_ids)
    
    for idx, seq_id in enumerate(unique_seq_ids):
        if idx % 100 == 0:
            print(f"Processing sequence {idx+1}/{total_sequences}: {seq_id}")
            
        seq_data = train_df[train_df['sequence_id'] == seq_id].copy()
        
        # Apply feature engineering
        seq_data = engineer_features(seq_data)
        
        # Extract features and fill missing values
        seq_features = seq_data[feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        
        sequences.append(seq_features)
        labels.append(seq_data['gesture'].iloc[0])
        subjects.append(seq_data['subject'].iloc[0])
        sequence_ids.append(seq_id)
    
    # Calculate padding length
    sequence_lengths = [len(seq) for seq in sequences]
    pad_len = int(np.percentile(sequence_lengths, PAD_PERCENTILE))
    
    print(f"Sequence length statistics:")
    print(f"  Min: {min(sequence_lengths)}")
    print(f"  Max: {max(sequence_lengths)}")
    print(f"  Mean: {np.mean(sequence_lengths):.1f}")
    print(f"  {PAD_PERCENTILE}th percentile (pad_len): {pad_len}")
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=pad_len, padding='post', 
                     truncating='post', dtype='float32')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)
    
    # Fit scaler
    print("Fitting scaler on all training data...")
    all_data = np.vstack([seq.reshape(-1, seq.shape[-1]) for seq in X])
    scaler = StandardScaler()
    scaler.fit(all_data)
    
    # Scale the data
    X_scaled = np.zeros_like(X)
    for i in range(len(X)):
        X_scaled[i] = scaler.transform(X[i])
    
    print(f"Final dataset shape: {X_scaled.shape}")
    print(f"Feature dimensions: {len(feature_cols)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return {
        'X': X_scaled,
        'y': y_categorical,
        'y_encoded': y_encoded,
        'subjects': np.array(subjects),
        'sequence_ids': np.array(sequence_ids),
        'labels': labels,
        'feature_cols': feature_cols,
        'pad_len': pad_len,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'gesture_classes': label_encoder.classes_
    }

#=============================================================================
# Training Pipeline
#=============================================================================

def train_model(data_dict, fold_idx, train_idx, val_idx, model_name_prefix="model"):
    """Train a single model for one fold"""
    print(f"\n{'='*60}")
    print(f"Training {model_name_prefix} - Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    X, y = data_dict['X'], data_dict['y']
    pad_len = data_dict['pad_len']
    n_classes = len(data_dict['gesture_classes'])
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Calculate class weights
    y_train_encoded = data_dict['y_encoded'][train_idx]
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_encoded), 
        y=y_train_encoded
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Determine feature dimensions
    imu_dim = 7  # acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z
    total_dim = X.shape[-1]
    tof_dim = total_dim - imu_dim
    
    # Build model
    model = build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=WD)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LR_INIT, weight_decay=WD),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_path = EXPORT_DIR / f"{model_name_prefix}_{fold_idx}.h5"
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Create mixup generator
    mixup_gen = MixupGenerator(X_train, y_train, BATCH_SIZE, MIXUP_ALPHA)
    
    # Train model
    history = model.fit(
        mixup_gen,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Load best weights and evaluate
    model.load_weights(model_path)
    val_pred = model.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_pred, axis=1)
    val_true_classes = np.argmax(y_val, axis=1)
    
    # Print evaluation metrics
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"Validation Accuracy: {np.mean(val_pred_classes == val_true_classes):.4f}")
    
    return history, model

def main():
    """Main training pipeline"""
    print("ðŸš€ Starting BFRB Classification Training Pipeline")
    
    # Load and prepare data
    train_df, train_demographics = load_and_preprocess_data()
    data_dict = prepare_dataset(train_df, train_demographics)
    
    # Save preprocessing artifacts
    print("\nSaving preprocessing artifacts...")
    np.save(EXPORT_DIR / "feature_cols.npy", data_dict['feature_cols'])
    np.save(EXPORT_DIR / "sequence_maxlen.npy", data_dict['pad_len'])
    np.save(EXPORT_DIR / "gesture_classes.npy", data_dict['gesture_classes'])
    joblib.dump(data_dict['scaler'], EXPORT_DIR / "scaler.pkl")
    
    # Setup cross-validation
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Train models
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(data_dict['X'], data_dict['y_encoded'], data_dict['subjects'])
    ):
        # Train first model type (D-111)
        history1, model1 = train_model(
            data_dict, fold_idx, train_idx, val_idx, "D-111"
        )
        
        # Train second model type (v0629) 
        history2, model2 = train_model(
            data_dict, fold_idx, train_idx, val_idx, "v0629"
        )
        
        fold_results.append({
            'fold': fold_idx,
            'history1': history1,
            'history2': history2
        })
    
    print(f"\nðŸŽ‰ Training completed! Models saved in {EXPORT_DIR}")
    print("Preprocessing artifacts saved:")
    print("  - feature_cols.npy")
    print("  - sequence_maxlen.npy") 
    print("  - gesture_classes.npy")
    print("  - scaler.pkl")
    print(f"  - {N_FOLDS}x2 model files (.h5)")

if __name__ == "__main__":
    main()




