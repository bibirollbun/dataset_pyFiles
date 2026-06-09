import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
register = tf.keras.utils.register_keras_serializable

@register(package="custom_layers")
class SqueezeLastAxis(Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)

@register(package="custom_layers")
class ExpandLastAxis(Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=-1)

@register(package="custom_layers")
class TimeSum(Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

@register(package="custom_layers")
class SliceLayer(Layer):
    def __init__(self, start, end=None, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = None if end is None else int(end)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'start': self.start, 'end': self.end})
        return cfg

    def call(self, inputs):
        if self.end is None:
            return inputs[:, :, self.start:]
        return inputs[:, :, self.start:self.end]

@register(package="custom_layers")
class SEBlock(Layer):
    def __init__(self, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        ch = int(input_shape[-1])
        from tensorflow.keras.layers import Dense
        self.dense1 = Dense(ch // self.reduction, activation='relu')
        self.dense2 = Dense(ch, activation='sigmoid')
        super().build(input_shape)

    def call(self, x):
        from tensorflow.keras.layers import Reshape, Multiply
        se = tf.reduce_mean(x, axis=1, keepdims=False)   # GlobalAveragePooling1D
        se = self.dense1(se)
        se = self.dense2(se)
        se = Reshape((1, tf.shape(se)[-1]))(se)
        return Multiply()([x, se])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reduction': self.reduction})
        return cfg

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def time_sum(x):
    return K.sum(x, axis=1)

def se_block(x, reduction=8):
    from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Reshape, Multiply
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

custom_objects = {
    'SqueezeLastAxis': SqueezeLastAxis,
    'ExpandLastAxis': ExpandLastAxis,
    'TimeSum': TimeSum,
    'SliceLayer': SliceLayer,
    'SEBlock': SEBlock,
    'squeeze_last_axis': squeeze_last_axis,
    'expand_last_axis': expand_last_axis,
    'time_sum': time_sum,
    'se_block': se_block,
}



import os
import pandas as pd
import polars as pl
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import backend as K
from scipy.spatial.transform import Rotation as R
import gc
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    """Squeeze-and-Excitation block"""
    from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Reshape, Multiply
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])


# Load preprocessing objects
scaler = joblib.load('/kaggle/input/cmi-final-model-artifacts/scaler.pkl')
le = joblib.load('/kaggle/input/cmi-final-model-artifacts/label_encoder.pkl')
gesture_classes = le.classes_

models = []
print("Loading models...")

for i in range(1, 6):
    path = f"/kaggle/input/cmi-final-model-artifacts/gesture_model_fold_{i}.h5"
    try:
        m = load_model(path, custom_objects=custom_objects, compile=False)
        models.append(m)
        print(f"Loaded model {i}")
    except Exception as e:
        print(f"Could not load model {i}: {e}")

if len(models) == 0:
    print("Trying to load only the best model...")
    try:
        best_model = load_model('/kaggle/input/cmi-final-model-artifacts/best_model_after_training.h5',
                        custom_objects=custom_objects,
                        compile=False)

        models = [best_model]
        print("Successfully loaded best model")
    except Exception as e:
        print(f"Could not load best model: {e}")

if len(models) == 0:
    print("Error: No models could be loaded!")
    # Fallback: create a dummy prediction function
    def fallback_predict():
        return 'Text on phone'
else:
    print(f"Successfully loaded {len(models)} models")


pad_len = 127  # Use the same pad_len from training

def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity from acceleration data"""
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
    """Calculate angular velocity from quaternions"""
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
    """Calculate angular distance between consecutive rotations"""
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

def create_advanced_features(df_seq):
    """Create advanced features from raw sensor data"""
    # Gravity removal
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'] = linear_accel[:, 0]
    df_seq['linear_acc_y'] = linear_accel[:, 1]
    df_seq['linear_acc_z'] = linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 +
                                      df_seq['linear_acc_y']**2 +
                                      df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)

    # Angular velocity features
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'] = angular_vel[:, 0]
    df_seq['angular_vel_y'] = angular_vel[:, 1]
    df_seq['angular_vel_z'] = angular_vel[:, 2]
    df_seq['angular_vel_mag'] = np.sqrt(angular_vel[:, 0]**2 +
                                       angular_vel[:, 1]**2 +
                                       angular_vel[:, 2]**2)
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)

    # ToF features
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols):
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)

            # Statistical features
            df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
            df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
            df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
            df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
            df_seq[f'tof_{i}_median'] = tof_data.median(axis=1)
            df_seq[f'tof_{i}_range'] = df_seq[f'tof_{i}_max'] - df_seq[f'tof_{i}_min']

            # Spatial features
            try:
                tof_reshaped = tof_data.values.reshape(-1, 8, 8)
                df_seq[f'tof_{i}_center'] = np.nanmean(tof_reshaped[:, 3:5, 3:5], axis=(1,2))
                df_seq[f'tof_{i}_edge'] = np.nanmean(np.concatenate([
                    tof_reshaped[:, 0, :].reshape(-1, 8),
                    tof_reshaped[:, -1, :].reshape(-1, 8),
                    tof_reshaped[:, 1:-1, 0].reshape(-1, 6),
                    tof_reshaped[:, 1:-1, -1].reshape(-1, 6)
                ], axis=1), axis=1)
            except:
                df_seq[f'tof_{i}_center'] = 0
                df_seq[f'tof_{i}_edge'] = 0

    for i in range(1, 6):
        if f'thm_{i}' in df_seq.columns:
            df_seq[f'thm_{i}_diff'] = df_seq[f'thm_{i}'].diff().fillna(0)
            df_seq[f'thm_{i}_rolling_mean'] = df_seq[f'thm_{i}'].rolling(
                window=5, center=True).mean().fillna(df_seq[f'thm_{i}'])

    return df_seq
    
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Predict gesture for a single sequence using ensemble of trained models"""
    try:
        df_seq = sequence.to_pandas()

        # Feature engineering
        df_seq = create_advanced_features(df_seq)

        # Define features
        imu_features = [
            'acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z',
            'linear_acc_x','linear_acc_y','linear_acc_z','linear_acc_mag',
            'linear_acc_mag_jerk','angular_vel_x','angular_vel_y','angular_vel_z',
            'angular_vel_mag','angular_distance'
        ]
        tof_features = [f'tof_{i}_{feat}' for i in range(1,6) for feat in 
                        ['mean','std','min','max','median','range','center','edge']]
        thm_features = [f for i in range(1,6) for f in 
                        [f'thm_{i}',f'thm_{i}_diff',f'thm_{i}_rolling_mean']]
        FEATURE_COLS = imu_features + tof_features + thm_features

        # Ensure all features exist
        for col in FEATURE_COLS:
            if col not in df_seq.columns:
                df_seq[col] = 0.0
        df_seq[FEATURE_COLS] = df_seq[FEATURE_COLS].ffill().bfill().fillna(0)

        # Scale + pad
        mat_unscaled = df_seq[FEATURE_COLS].values.astype('float32')
        mat_scaled = scaler.transform(mat_unscaled)
        pad_input = pad_sequences([mat_scaled], maxlen=pad_len,
                                  padding='post', truncating='post', dtype='float32')

        # Ensemble prediction
        if len(models) > 0:
            all_preds = [m.predict(pad_input, verbose=0)[0] for m in models]
            ensemble_pred = np.mean(all_preds, axis=0)
            predicted_idx = np.argmax(ensemble_pred)
            return gesture_classes[predicted_idx]
        else:
            return 'Text on phone'  # fallback
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 'Text on phone'

        

# For Kaggle submission
if __name__ == "__main__":
    print("Setting up inference server...")
    
    # Import kaggle evaluation
    import kaggle_evaluation.cmi_inference_server
    
    # Create inference server
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    # Check if running in competition environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in competition mode...")
        inference_server.serve()
    else:
        print("Running in local test mode...")
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
    
    print("Submission complete!")


import polars as pl
import pandas as pd
from tqdm.auto import tqdm
import os

# File paths
TEST_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
DEM_CSV  = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
OUTPUT_CSV = "/kaggle/working/predictions.csv"
LIMIT_SEQS = None

def find_seq_col(df):
    candidates = ['sequence_id', 'seq_id', 'group', 'id', 'session_id', 'sequence', 'group_id']
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in df.columns:
        nunique = df[c].nunique()
        if 2 < nunique < 10000 and nunique < len(df) // 2:
            return c
    raise ValueError("No sequence id column detected. Set SEQ_COL manually.")

print("Loading test CSV...")
test_pl = pl.read_csv(TEST_CSV)
print("Loading demographics CSV...")
dem_pl = pl.read_csv(DEM_CSV)

try:
    SEQ_COL = find_seq_col(test_pl.to_pandas())
    print(f"Detected sequence id column: '{SEQ_COL}'")
except Exception as e:
    print("Detection failed:", e)
    raise

seq_ids = test_pl.select(SEQ_COL).unique().to_series().to_list()
if LIMIT_SEQS:
    seq_ids = seq_ids[:LIMIT_SEQS]

preds = []
errors = []

print(f"Running predictions for {len(seq_ids)} sequences (limit={LIMIT_SEQS})...")
for sid in tqdm(seq_ids):
    try:
        seq_df = test_pl.filter(pl.col(SEQ_COL) == sid)
        dem_row = pl.DataFrame({})

        if 'subject' in dem_pl.columns:
            if 'subject' in seq_df.columns:
                subj_vals = seq_df.select('subject').unique().to_series().to_list()
                if len(subj_vals) > 0:
                    dem_row = dem_pl.filter(pl.col('subject') == subj_vals[0])

        if dem_row.height == 0:
            common = list(set(dem_pl.columns).intersection(set(seq_df.columns)))
            fallback_keys = ['participant_id', 'person_id', 'session_id', 'id', 'subject']
            for key in fallback_keys:
                if key in common:
                    val_list = seq_df.select(key).unique().to_series().to_list()
                    if len(val_list) > 0:
                        dem_row = dem_pl.filter(pl.col(key) == val_list[0])
                        break

        if dem_row.height == 0:
            dem_row = pl.DataFrame({})

        prediction = predict(seq_df, dem_row)
        preds.append((sid, prediction))

    except Exception as e:
        errors.append((sid, str(e)))
        preds.append((sid, "ERROR"))
        print(f"Error predicting for seq {sid}: {e}")

pred_df = pd.DataFrame(preds, columns=[SEQ_COL, 'predicted_gesture'])
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")

print("\nSample predictions (first 10):")
print(pred_df.head(10))

print("\nPrediction counts:")
print(pred_df['predicted_gesture'].value_counts().head(20))

if errors:
    print(f"\nThere were {len(errors)} prediction errors. Sample:")
    print(errors[:5])


