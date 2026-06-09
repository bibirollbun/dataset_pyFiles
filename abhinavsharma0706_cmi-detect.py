import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold


import random
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_everything(seed=42)


# (Competition metric will only be imported when TRAINing)
TRAIN = True                     # ← set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/pretrained-model")  # used when TRAIN=False
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40


print("▶ imports ready · tensorflow", tf.__version__)


#Tensor Manipulations
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

# Residual CNN Block with SE
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


# Normalizes and cleans the time series sequence. 

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

# MixUp the data argumentation in order to regularize the neural network. 

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

tmp_model = build_two_branch_model(127,7,325,18)



if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    df_for_groups = pd.merge(df.copy(), train_dem_df, on='subject', how='left')

    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    gesture_classes = le.classes_

    print("  Calculating engineered IMU features (magnitude, angle)...")
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    
    print("  Calculating engineered IMU derivatives (jerk, angular velocity)...")
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}

    imu_cols = [c for c in df.columns if c.startswith('acc_') and c not in ['acc_mag', 'acc_mag_jerk']]
    imu_cols.extend([c for c in df.columns if c.startswith('rot_') and c not in ['rot_angle', 'rot_angle_vel']])
    imu_cols.extend(['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel'])

    thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
    
    tof_aggregated_cols_template = []
    for i in range(1, 6):
        tof_aggregated_cols_template.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])

    final_feature_cols = imu_cols + thm_cols_original + tof_aggregated_cols_template
    imu_dim_final = len(imu_cols)
    tof_thm_aggregated_dim_final = len(thm_cols_original) + len(tof_aggregated_cols_template)
    
    print(f"  IMU (incl. engineered & derivatives) {imu_dim_final} | THM + Aggregated TOF {tof_thm_aggregated_dim_final} | total {len(final_feature_cols)} features")
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

    print("  Building sequences with aggregated TOF and preparing data for scaler...")
    seq_gp = df.groupby('sequence_id') 
    
    all_steps_for_scaler_list = []
    X_list_unscaled, y_list_int_for_stratify, lens = [], [], [] 

    for seq_id, seq_df_orig in seq_gp:
        seq_df = seq_df_orig.copy()

        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
            seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
        
        mat_unscaled = seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        
        all_steps_for_scaler_list.append(mat_unscaled)
        X_list_unscaled.append(mat_unscaled)
        y_list_int_for_stratify.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(mat_unscaled))

    print("  Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    del all_steps_for_scaler_list, all_steps_concatenated

    print("  Scaling and padding sequences...")
    X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
    del X_list_unscaled

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    
    X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    del X_scaled_list
    
    y_int_for_stratify = np.array(y_list_int_for_stratify)
    y = to_categorical(y_int_for_stratify, num_classes=len(le.classes_))

    print("  Splitting data and preparing for training...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=82, stratify=y_int_for_stratify)

    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_int_for_stratify)
    class_weight = dict(enumerate(cw_vals))

    model = build_two_branch_model(pad_len, imu_dim_final, tof_thm_aggregated_dim_final, len(le.classes_), wd=WD)
    
    steps = len(X_tr) // BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(5e-4, first_decay_steps=15 * steps) 
    
    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1, monitor='val_accuracy', mode='max')
    
    print("  Starting model training...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=class_weight, callbacks=[cb], verbose=1)

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")
    print("✔ Training done – artefacts saved in", EXPORT_DIR)

    from cmi_2025_metric_copy_for_import_py import CompetitionMetric
    preds_val = model.predict(X_val).argmax(1)
    true_val_int  = y_val.argmax(1)
    
    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[true_val_int]}),
        pd.DataFrame({'gesture': le.classes_[preds_val]}))
    print("Hold‑out H‑F1 =", round(h_f1, 4))

else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    temp_imu_cols = [c for c in final_feature_cols if c.startswith('acc_') or c.startswith('rot_')]
    imu_dim_final = len(temp_imu_cols)
    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final

    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer,
    }
    model = load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
                       compile=False, custom_objects=custom_objs)
    print("  Model, scaler, feature_cols, pad_len loaded – ready for evaluation")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()

    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
    df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)
    
    for i in range(1, 6): 
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
        tof_sensor_data = df_seq[pixel_cols_tof].replace(-1, np.nan)
        
        df_seq[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
        df_seq[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
        df_seq[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
        df_seq[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
        
    df_seq_reordered = pd.DataFrame(columns=final_feature_cols)
    for col in final_feature_cols:
        if col in df_seq.columns:
            df_seq_reordered[col] = df_seq[col]

    mat_unscaled = df_seq_reordered.ffill().bfill().fillna(0).values.astype('float32')
    
    mat_scaled = scaler.transform(mat_unscaled)
    
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    
    idx = int(model.predict(pad_input, verbose=0).argmax(1)[0])
    return str(gesture_classes[idx])


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

