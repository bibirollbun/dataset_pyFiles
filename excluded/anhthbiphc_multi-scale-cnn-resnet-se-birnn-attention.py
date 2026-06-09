import pandas as pd
from pathlib import Path


# Giáº£ sá»­ báº¡n Ä‘Ã£ add dataset vá»›i Ä‘Æ°á»�ng dáº«n lÃ  /kaggle/input/eda-cleaned-data
df_path = "/kaggle/input/eda-data"

train_df = pd.read_parquet(f"{df_path}/train_cleaned.parquet")
train_dem_df = pd.read_parquet(f"{df_path}/train_demographics.parquet")
test_df = pd.read_parquet(f"{df_path}/test_cleaned.parquet")
test_dem_df = pd.read_parquet(f"{df_path}/test_demographics.parquet")


USE_PROCESSED_DATA = True  # â†� chuyá»ƒn thÃ nh False náº¿u muá»‘n dÃ¹ng data gá»‘c
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EDA_DIR = Path("/kaggle/input/eda-data")  # â†� Add Dataset chá»©a file .parquet
import pandas as pd

if USE_PROCESSED_DATA:
    print("ğŸ“¦ Loading EDA-processed data...")
    train_df = pd.read_parquet(EDA_DIR / "train_cleaned.parquet")
    train_dem_df = pd.read_parquet(EDA_DIR / "train_demographics.parquet")
    test_df = pd.read_parquet(EDA_DIR / "test_cleaned.parquet")
    test_dem_df = pd.read_parquet(EDA_DIR / "test_demographics.parquet")
else:
    print("ğŸ“¦ Loading RAW data...")
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")
    test_dem_df = pd.read_csv(RAW_DIR / "test_demographics.csv")


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
    Lambda, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam,AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl


# (Competition metric will only be imported when TRAINing)
TRAIN = False                     # â†� set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi025-trained-model")  # used when TRAIN=False
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 1e-3
WD = 3e-4
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40


print("â–¶ imports ready Â· tensorflow", tf.__version__)


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling1D, Flatten
from keras.models import Model

# --- SE Block (Squeeze-and-Excitation Block) ---
def enhanced_se_block(x, reduction=8):
    ch = x.shape[-1]
    se1 = GlobalMaxPooling1D()(x)
    se2 = GlobalAveragePooling1D()(x)
    se = Concatenate()([se1, se2])
    
    se = Dense(max(4, ch // reduction), activation='swish')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

# --- MS-TCN Block (Multiscale Temporal Convolutional Network) ---
def ms_tcn_block(x, filters, dilation_rates=[1, 2, 4, 8], wd=1e-4):
    branches = []
    for rate in dilation_rates:
        b = Conv1D(filters, 3, padding='same', dilation_rate=rate, 
                   use_bias=False, kernel_regularizer=l2(wd))(x)
        b = BatchNormalization()(b)
        b = Activation('swish')(b)
        branches.append(b)
    out = Concatenate(axis=-1)(branches)
    out = Conv1D(filters, 1, use_bias=False, kernel_regularizer=l2(wd))(out)
    out = BatchNormalization()(out)
    return Activation('swish')(out)

# --- Cross Attention Fusion ---
def cross_attention_fusion(x1, x2, filters, wd=1e-4):
    q1 = Conv1D(filters//2, 1, use_bias=False, kernel_regularizer=l2(wd))(x1)
    k1 = Conv1D(filters//2, 1, use_bias=False, kernel_regularizer=l2(wd))(x2)
    v1 = Conv1D(filters, 1, use_bias=False, kernel_regularizer=l2(wd))(x2)

    def compute_attention(inputs):
        q, k, v = inputs
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, v)

    attn1 = Lambda(compute_attention)([q1, k1, v1])

    q2 = Conv1D(filters//2, 1, use_bias=False, kernel_regularizer=l2(wd))(x2)
    k2 = Conv1D(filters//2, 1, use_bias=False, kernel_regularizer=l2(wd))(x1)
    v2 = Conv1D(filters, 1, use_bias=False, kernel_regularizer=l2(wd))(x1)

    attn2 = Lambda(compute_attention)([q2, k2, v2])

    fused = Concatenate()([attn1, attn2])
    fused = Conv1D(filters, 1, use_bias=False, kernel_regularizer=l2(wd))(fused)
    return BatchNormalization()(fused)

class TimeAlign(layers.Layer):
    def __init__(self, target_steps, **kwargs):
        super().__init__(**kwargs)
        self.target_steps = target_steps

    def call(self, x):
        current_steps = tf.shape(x)[1]
        diff = self.target_steps - current_steps

        def pad_fn():
            paddings = [[0, 0], [0, diff], [0, 0]]
            return tf.pad(x, paddings)

        def crop_fn():
            return x[:, :self.target_steps, :]

        return tf.cond(diff > 0, pad_fn, crop_fn)

    def get_config(self):
        return {'target_steps': self.target_steps}

def pyramid_fusion(imu_features, tof_features, target_steps, wd=1e-4):
    fusion_layers = []

    for imu, tof in zip(imu_features, tof_features):
        imu = TimeAlign(target_steps)(imu)
        tof = TimeAlign(target_steps)(tof)

        fused = cross_attention_fusion(imu, tof, 128, wd)
        fused = enhanced_se_block(fused)

        fusion_layers.append(fused)

    fusion_output = Concatenate(axis=-1)(fusion_layers)
    return fusion_output

class AdaptiveSampling(Layer):
    def __init__(self, input_steps, output_steps, **kwargs):
        super().__init__(**kwargs)
        self.input_steps = input_steps
        self.output_steps = output_steps

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="sampling_kernel",
            shape=(self.input_steps, self.output_steps),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        weights = tf.nn.softmax(self.kernel, axis=-1)
        return tf.einsum('btc,ts->bsc', inputs, weights)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_steps, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"input_steps": self.input_steps, "output_steps": self.output_steps})
        return config

def build_enhanced_model(pad_len, imu_dim, tof_dim, n_classes, wd):
    imu_input = Input(shape=(pad_len, imu_dim), name='imu_input')
    tof_input = Input(shape=(pad_len, tof_dim), name='tof_input')

    x1 = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(wd))(imu_input)
    x1 = Dropout(0.2)(x1)
    x1 = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(wd))(x1)

    x2 = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(wd))(tof_input)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(wd))(x2)

    fused = pyramid_fusion([x1], [x2], pad_len, wd)
    fused = AdaptiveSampling(input_steps=pad_len, output_steps=128)(fused)


    context = GlobalMaxPooling1D()(fused)

    x = context
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation='softmax')(x)

    return Model(inputs=[imu_input, tof_input], outputs=output)






import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# --- Tensor Manipulation Helper Functions ---
def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

# --- SE Block (Squeeze-and-Excitation Block) ---
def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

# --- Multi-Scale CNN Branch ---
def multi_scale_cnn_branch(x, filters=64, wd=1e-4):
    convs = []
    for ks in [3, 5, 7, 11]:
        conv = Conv1D(filters, ks, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        convs.append(conv)
    out = Concatenate()(convs)
    return out

# --- ResNet + SE Block ---
def resnet_se_block(x, filters, wd=1e-4):
    shortcut = x
    x = Conv1D(filters, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

# --- Attention Layer ---
def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

# --- TimeAlign Layer (for padding/truncating sequences to a fixed length) ---
class TimeAlign(Layer):
    def __init__(self, target_steps, **kwargs):
        super().__init__(**kwargs)
        self.target_steps = target_steps

    def call(self, x):
        current_steps = tf.shape(x)[1]
        diff = self.target_steps - current_steps

        def pad_fn():
            paddings = [[0, 0], [0, diff], [0, 0]]
            return tf.pad(x, paddings)

        def crop_fn():
            return x[:, :self.target_steps, :]

        return tf.cond(diff > 0, pad_fn, crop_fn)

    def get_config(self):
        return {'target_steps': self.target_steps}

# --- Building the Two-Branch Model ---
def build_enhanced_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim + tof_dim), name='main_input')

    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    imu_features = []
    x1 = imu
    for i in range(3):
        x1 = ms_tcn_block(x1, filters=64*(i+1), wd=wd)
        x1 = enhanced_se_block(x1)
        imu_features.append(x1)
        x1 = Conv1D(x1.shape[-1], 3, strides=2, padding='same', kernel_regularizer=l2(wd))(x1)
        x1 = Dropout(0.2 + 0.1*i)(x1)

    tof_features = []
    x2 = tof
    for i in range(3):
        x2 = ms_tcn_block(x2, filters=32*(i+1), wd=wd)
        x2 = enhanced_se_block(x2)
        tof_features.append(x2)
        x2 = Conv1D(x2.shape[-1], 3, strides=2, padding='same', kernel_regularizer=l2(wd))(x2)
        x2 = Dropout(0.2 + 0.1*i)(x2)

    fused = pyramid_fusion(imu_features, tof_features, pad_len, wd)
    fused = AdaptiveSampling(input_steps=pad_len, output_steps=128)(fused)


    x = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(fused)
    x = MultiHeadAttention(num_heads=4, key_dim=32, kernel_regularizer=l2(wd))(x, x)
    context = GlobalMaxPooling1D()(x)

    x = context
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)

    model = Model(inputs=inp, outputs=out)
    return model


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
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # --- IMU å¤šå°ºåº¦ CNN + ResNet + SE ---
    x1 = multi_scale_cnn_branch(imu, filters=64, wd=wd)
    x1 = resnet_se_block(x1, filters=128, wd=wd)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    # --- TOF åˆ†æ”¯ä¿�æŒ�å�Ÿæ · ---
    x2 = multi_scale_cnn_branch(tof, filters=64, wd=wd)
    x2 = resnet_se_block(x2, filters=128, wd=wd)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Dropout(0.3)(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)

    # --- å�ˆå¹¶ ---
    merged = Concatenate()([x1, x2])
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    # --- å…¨è¿�æ�¥å±‚ ---
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)





if TRAIN:
    print("â–¶ TRAIN MODE â€“ loading dataset â€¦")
    df = pd.read_csv(RAW_DIR / "train.csv")

    # label encoding
    le = LabelEncoder(); df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols  = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols  = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # global scaler
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # build sequences
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, lens = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        lens.append(len(mat))
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))

    X = pad_sequences(X_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    y = to_categorical(y_list, num_classes=len(le.classes_))

    # split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_list)

    # class weights
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_list)
    class_weight = dict(enumerate(cw_vals))
    
    # model
    model = build_enhanced_model(pad_len=pad_len,imu_dim=len(imu_cols),tof_dim=len(tof_cols),n_classes=len(le.classes_),wd=WD)
    
    #model = build_two_branch_model(pad_len, len(imu_cols), len(tof_cols), len(le.classes_), wd=WD)
    steps = len(X_tr)//BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)
    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=class_weight, callbacks=[cb], verbose=1)

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")
    print("âœ” Training done â€“ artefacts saved in", EXPORT_DIR)

    # quick metric
    from cmi_2025_metric_copy_for_import import CompetitionMetric
    preds = model.predict(X_val).argmax(1)
    true  = y_val.argmax(1)
    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[true]}),
        pd.DataFrame({'gesture': le.classes_[preds]}))
    print("Holdâ€‘out Hâ€‘F1 =", round(h_f1, 4))

else:
    print("â–¶ INFERENCE MODE â€“ loading artefacts from", PRETRAINED_DIR)
    feature_cols   = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'multi_scale_cnn_branch': multi_scale_cnn_branch,
        'resnet_se_block':resnet_se_block,
        'attention_layer': attention_layer,
    }
    model = load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
                       compile=False, custom_objects=custom_objs)
    print("  model, scaler, pads loaded â€“ ready for evaluation")


from sklearn.metrics import f1_score, classification_report

# TÃ­nh láº¡i F1-score trÃªn validation set tá»‘t nháº¥t
model.set_weights(best_model_state)
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = batch_y.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

# F1-score
val_f1_macro = f1_score(all_targets, all_preds, average='macro')
val_f1_weighted = f1_score(all_targets, all_preds, average='weighted')

print(f"\nğŸ”� Final Validation Results on Best Model:")
print(f"Macro F1-score:    {val_f1_macro:.4f}")
print(f"Weighted F1-score: {val_f1_weighted:.4f}")

# Náº¿u báº¡n cÃ³ gesture_classes (label names), in báº£ng chi tiáº¿t:
print(classification_report(all_targets, all_preds, target_names=gesture_classes))


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure the model and gesture_classes are only loaded once
gesture_classes = None
model = None
scaler = None
pad_len = None

def load_model_and_resources():
    global model, gesture_classes, scaler, pad_len
    if model is None:
        # Load the pretrained model
        model = tf.keras.models.load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5")
        print("Model loaded.")
        
    if gesture_classes is None:
        # Load gesture classes
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
        print("Gesture classes loaded.")
        
    if scaler is None:
        # Load scaler
        scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
        print("Scaler loaded.")
        
    if pad_len is None:
        # Load padding length
        pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
        print("Padding length loaded.")

# Call this function at the start of the script
load_model_and_resources()

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: object) -> np.ndarray:
    # Ensure all missing values are handled, and the sequence is scaled properly
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def predict(sequence: pd.DataFrame, demographics: pd.DataFrame) -> str:
    # Ensure model and resources are loaded
    load_model_and_resources()

    # Convert the sequence to pandas if it's a polars dataframe
    df_seq = sequence.to_pandas()

    # Preprocess sequence and pad it
    mat = preprocess_sequence(df_seq, feature_cols, scaler)
    pad = pad_sequences([mat], maxlen=pad_len, padding='post', truncating='post', dtype='float32')

    # Make prediction and retrieve the class
    pred_idx = int(model.predict(pad, verbose=0).argmax(1)[0])
    return str(gesture_classes[pred_idx])




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

