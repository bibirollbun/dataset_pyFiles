import os
import gc
import numpy as np
import pandas as pd
import random
import warnings
from pathlib import Path
from tqdm import tqdm
import joblib

# Import all necessary Keras components
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense, Dropout, Concatenate, GlobalAveragePooling1D, GaussianNoise, add, Multiply, Reshape, MaxPooling1D, GRU, Bidirectional, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, pad_sequences, Sequence
from tensorflow.keras.losses import CategoricalCrossentropy

# Import all necessary Sklearn components
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.spatial.transform import Rotation as R
from scipy.fft import rfft

warnings.filterwarnings("ignore")

# ===================================================================================
# Configuration
# ===================================================================================
class CFG:
    # Set the mode here: 'TRAIN' or 'INFERENCE'
    MODE = 'INFERENCE' 

    # --- Paths ---
    OUTPUT_DIR = Path("/kaggle/working/")
    MODEL_INPUT_DIR = Path("/kaggle/input/cmi-detection-training-output/") 

    # --- General ---
    SEED = 42
    
    # --- Training Params ---
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 5e-4
    PATIENCE = 35 
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    PAD_LEN_PERCENTILE = 95
    MIXUP_ALPHA = 0.4
    LABEL_SMOOTHING = 0.1
    
    # --- NEW: Inference Param ---
    TTA_STEPS = 5 # Number of augmentations for Test-Time Augmentation

def seed_everything(seed): os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
seed_everything(CFG.SEED)

# ===================================================================================
# Augmentation Function (Needed in both modes)
# ===================================================================================
def _augment_sequence(sequence):
    # Magnitude Scaling
    scale = np.random.uniform(0.9, 1.1)
    sequence = sequence * scale
    # Time Shifting
    shift = np.random.randint(-10, 10)
    sequence = np.roll(sequence, shift, axis=0)
    return sequence

# ===================================================================================
# Feature Engineering (Needed in both modes)
# ===================================================================================
def calculate_angular_velocity(quat_values, time_delta=1/50):
    angular_vel = np.zeros((len(quat_values), 3));
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1];
        if np.any(np.isnan(q_t)) or np.any(np.isnan(q_t_plus_dt)): continue
        try: rot_t = R.from_quat(q_t); rot_t_plus_dt = R.from_quat(q_t_plus_dt); delta_rot = rot_t.inv() * rot_t_plus_dt; angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except: pass
    return angular_vel
def remove_gravity(df_group):
    acc = df_group[['acc_x', 'acc_y', 'acc_z']].values; quat = df_group[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc); gravity = np.array([0, 0, 9.81])
    for i in range(len(acc)):
        try: rotation = R.from_quat(quat[i]); gravity_sensor = rotation.apply(gravity, inverse=True); linear_accel[i, :] = acc[i, :] - gravity_sensor
        except: linear_accel[i, :] = acc[i, :]
    return linear_accel
def engineer_features(df):
    processed = [];
    for _, group in tqdm(df.groupby('sequence_id'), desc="Engineering Features", leave=False, bar_format='{l_bar}{bar:10}{r_bar}'):
        g = group.copy(); linear_accel = remove_gravity(g); g['linear_acc_x'], g['linear_acc_y'], g['linear_acc_z'] = linear_accel.T; angular_vel = calculate_angular_velocity(g[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values); g['angular_vel_x'], g['angular_vel_y'], g['angular_vel_z'] = angular_vel.T
        for axis in ['x', 'y', 'z']: g[f'linear_acc_{axis}_jerk'] = g[f'linear_acc_{axis}'].diff().fillna(0)
        g['tof_neg1_pct'] = (g[[c for c in g.columns if c.startswith('tof_') and '_v' in c]] == -1).sum(axis=1) / 320.0; motion_features = ['acc_x', 'acc_y', 'acc_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
        for col in motion_features: g[f'{col}_std_roll5'] = g[col].rolling(window=5, min_periods=1).std().fillna(0); g[f'{col}_skew_roll5'] = g[col].rolling(window=5, min_periods=1).skew().fillna(0)
        for col in motion_features:
            fft_vals = np.abs(rfft(g[col].values))
            if len(fft_vals) > 4: top_freq_indices = np.argsort(fft_vals[1:])[-3:] + 1; g[f'{col}_fft_mag1'] = fft_vals[top_freq_indices[-1]] if len(top_freq_indices) > 0 else 0; g[f'{col}_fft_mag2'] = fft_vals[top_freq_indices[-2]] if len(top_freq_indices) > 1 else 0; g[f'{col}_fft_mag3'] = fft_vals[top_freq_indices[-3]] if len(top_freq_indices) > 2 else 0
            else: g[f'{col}_fft_mag1'], g[f'{col}_fft_mag2'], g[f'{col}_fft_mag3'] = 0, 0, 0
        processed.append(g)
    return pd.concat(processed).fillna(0)

# ===================================================================================
# Model Architectures & Data Generators
# ===================================================================================
class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2, shuffle=True):
        self.X_imu, self.X_thm_tof = X[0], X[1]; self.y, self.batch_size, self.alpha, self.shuffle = y, batch_size, alpha, shuffle; self.indices = np.arange(len(self.y)); self.on_epoch_end()
    def __len__(self): return int(np.floor(len(self.y) / self.batch_size))
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]; X_imu_aug = np.array([_augment_sequence(x) for x in self.X_imu[indices]]); X_thm_tof_aug = np.array([_augment_sequence(x) for x in self.X_thm_tof[indices]]); lam = np.random.beta(self.alpha, self.alpha); perm = np.random.permutation(len(indices)); X_imu_mix = lam * X_imu_aug + (1 - lam) * X_imu_aug[perm]; X_thm_tof_mix = lam * X_thm_tof_aug + (1 - lam) * X_thm_tof_aug[perm]; y_mix = lam * self.y[indices] + (1 - lam) * self.y[indices[perm]]; return (X_imu_mix, X_thm_tof_mix), y_mix
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)
class MixupGeneratorIMU(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2, shuffle=True):
        self.X, self.y, self.batch_size, self.alpha, self.shuffle = X, y, batch_size, alpha, shuffle; self.indices = np.arange(len(self.y)); self.on_epoch_end()
    def __len__(self): return int(np.floor(len(self.y) / self.batch_size))
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]; X_aug = np.array([_augment_sequence(x) for x in self.X[indices]]); lam = np.random.beta(self.alpha, self.alpha); perm = np.random.permutation(len(indices)); X_mix = lam * X_aug + (1 - lam) * X_aug[perm]; y_mix = lam * self.y[indices] + (1 - lam) * self.y[indices[perm]]; return X_mix, y_mix
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)
def se_block(x, reduction=8):
    channels = x.shape[-1]; se = GlobalAveragePooling1D()(x); se = Dense(channels // reduction, activation='relu')(se); se = Dense(channels, activation='sigmoid')(se); return Multiply()([x, Reshape((1, channels))(se)])
def residual_se_cnn_block(x, filters, kernel_size, wd, pool_size=2, drop=0.4):
    shortcut = x; x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x)
    if shortcut.shape[-1] != filters: shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
    x = add([x, shortcut]); x = Activation('relu')(x); x = MaxPooling1D(pool_size)(x); x = Dropout(drop)(x); return x
def build_slim_model(pad_len, imu_dim, thm_tof_dim, n_classes, wd=5e-4):
    inp_imu = Input(shape=(pad_len, imu_dim), name='imu_input'); x1 = Dense(64, activation='relu')(inp_imu); x1 = GaussianNoise(0.02)(x1); x1 = residual_se_cnn_block(x1, 32, 5, wd=wd); x1 = residual_se_cnn_block(x1, 64, 3, wd=wd); x1 = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(wd)))(x1); attention_out = Attention()([x1, x1]); x1 = GlobalAveragePooling1D()(attention_out)
    inp_thm_tof = Input(shape=(pad_len, thm_tof_dim), name='thm_tof_input'); x2 = Dense(32, activation='relu')(inp_thm_tof); x2 = GaussianNoise(0.02)(x2); x2 = residual_se_cnn_block(x2, 32, 5, wd=wd); x2 = GlobalAveragePooling1D()(x2)
    merged = Concatenate()([x1, x2]); x = Dense(128, activation='relu', kernel_regularizer=l2(wd))(merged); x = Dropout(0.5)(x); out = Dense(n_classes, activation='softmax')(x); return Model(inputs=[inp_imu, inp_thm_tof], outputs=out)
def build_imu_only_slim_model(pad_len, imu_dim, n_classes, wd=5e-4):
    inp_imu = Input(shape=(pad_len, imu_dim), name='imu_input'); x1 = Dense(64, activation='relu')(inp_imu); x1 = GaussianNoise(0.02)(x1); x1 = residual_se_cnn_block(x1, 32, 5, wd=wd); x1 = residual_se_cnn_block(x1, 64, 3, wd=wd); x1 = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(wd)))(x1); attention_out = Attention()([x1, x1]); x1 = GlobalAveragePooling1D()(attention_out)
    x = Dense(128, activation='relu', kernel_regularizer=l2(wd))(x1); x = Dropout(0.5)(x); out = Dense(n_classes, activation='softmax')(x); return Model(inputs=inp_imu, outputs=out)

###################################################################################
# --- MAIN EXECUTION BLOCK ---
###################################################################################
if __name__ == "__main__":
    
    if CFG.MODE == 'TRAIN':
        print("Script is in TRAIN mode. Please re-run this script in a separate notebook to generate artifacts.")
        # The training code would go here. It remains unchanged from the previous version.
    
    else: # INFERENCE MODE
        print("Running in INFERENCE mode.")
        if os.path.exists('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'):
            import kaggle_evaluation.cmi_inference_server
        else: # Mock server for local testing
            class MockInferenceServer:
                def __init__(self, pred_func): print("Mock server initialized.")
                def serve(self): print("Mock server `serve()` called.")
                def run_local_gateway(self, data_paths): print(f"Mock server `run_local_gateway` called.")
            kaggle_evaluation = type('Kaggle', (), {'cmi_inference_server': type('CMI', (), {'CMIInferenceServer': MockInferenceServer})})()

        print("Loading artifacts..."); LE = joblib.load(CFG.MODEL_INPUT_DIR / 'label_encoder.joblib'); PAD_LEN = joblib.load(CFG.MODEL_INPUT_DIR / 'pad_len.joblib'); ADV_IMU_COLS = joblib.load(CFG.MODEL_INPUT_DIR / 'adv_imu_cols.joblib'); THM_TOF_COLS = joblib.load(CFG.MODEL_INPUT_DIR / 'thm_tof_cols.joblib'); SCALER_IMU = joblib.load(CFG.MODEL_INPUT_DIR / 'scaler_imu.joblib'); SCALER_THM_TOF = joblib.load(CFG.MODEL_INPUT_DIR / 'scaler_thm_tof.joblib'); N_CLASSES = len(LE.classes_)
        print("Loading models..."); MODEL_FULL = build_slim_model(PAD_LEN, len(ADV_IMU_COLS), len(THM_TOF_COLS), N_CLASSES); MODEL_FULL.load_weights(CFG.MODEL_INPUT_DIR / 'full_model.weights.h5'); MODEL_IMU_ONLY = build_imu_only_slim_model(PAD_LEN, len(ADV_IMU_COLS), N_CLASSES); MODEL_IMU_ONLY.load_weights(CFG.MODEL_INPUT_DIR / 'imu_only_model.weights.h5'); print("Artifacts loaded.")

        def predict(sequence_df_polars, demographics_df_polars):
            sequence_df = sequence_df_polars.to_pandas()
            demographics_df = demographics_df_polars.to_pandas()
            sequence_df = pd.merge(sequence_df, demographics_df, on='subject', how='left')
            sensor_cols_inf = [c for c in sequence_df.columns if c.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
            for col in sensor_cols_inf: sequence_df[col] = pd.to_numeric(sequence_df[col], errors='coerce')
            sequence_df['sequence_id'] = 0; processed_df = engineer_features(sequence_df)
            is_imu_only = processed_df[THM_TOF_COLS].isnull().all().all()
            
            # --- Store predictions from each TTA step ---
            tta_preds = []
            
            # --- Extract original data once ---
            original_imu_data = processed_df[ADV_IMU_COLS].ffill().bfill().fillna(0).values
            if not is_imu_only:
                original_thm_tof_data = processed_df[THM_TOF_COLS].ffill().bfill().fillna(0).values
            
            for i in range(CFG.TTA_STEPS):
                # Use original data on the first step, augment on subsequent steps
                imu_data = _augment_sequence(np.copy(original_imu_data)) if i > 0 else original_imu_data
                
                if is_imu_only:
                    scaled_data = SCALER_IMU.transform(imu_data)
                    padded_data = pad_sequences([scaled_data], maxlen=PAD_LEN, dtype='float32')
                    preds = MODEL_IMU_ONLY.predict_on_batch(padded_data)
                else:
                    thm_tof_data = _augment_sequence(np.copy(original_thm_tof_data)) if i > 0 else original_thm_tof_data
                    scaled_imu = SCALER_IMU.transform(imu_data)
                    scaled_thm_tof = SCALER_THM_TOF.transform(thm_tof_data)
                    padded_imu = pad_sequences([scaled_imu], maxlen=PAD_LEN, dtype='float32')
                    padded_thm_tof = pad_sequences([scaled_thm_tof], maxlen=PAD_LEN, dtype='float32')
                    preds = MODEL_FULL.predict_on_batch([padded_imu, padded_thm_tof])
                
                tta_preds.append(preds)
            
            # Average the predictions across all TTA steps
            final_preds = np.mean(tta_preds, axis=0)
            
            predicted_class_index = np.argmax(final_preds, axis=1)[0]
            predicted_class_name = LE.inverse_transform([predicted_class_index])[0]
            
            return predicted_class_name

        print("Setting up Kaggle inference server...")
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

