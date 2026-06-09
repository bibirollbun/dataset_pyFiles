# Twoâ€‘Branch Humanâ€‘Activityâ€‘Recognition Pipeline (IMUÂ +Â Thermopile/TOF  +Â SEâ€‘CNNÂ +Â BiLSTMÂ +Â Attention)
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl



# (Competition metric will only be imported when TRAINing)
TRAIN = True                     # â†� set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/pretrained-model")  # used when TRAIN=False
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 1e-3
WD = 3e-4
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40


print("â–¶ imports ready Â· tensorflow", tf.__version__)


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



# æ™‚ç³»åˆ—ãƒ‡ãƒ¼ã‚¿ã�®å‰�å‡¦ç�†ï¼ˆæ¬ æ��å€¤ã�®è£œå®Œã�¨æ¨™æº–åŒ–ï¼‰
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values  # æ¬ æ��å€¤ã‚’è£œå®Œï¼ˆå‰�æ–¹ãƒ»å¾Œæ–¹ï¼‰ã€�ã�•ã‚‰ã�«ã‚¼ãƒ­åŸ‹ã‚�
    return scaler.transform(mat).astype('float32')  # æ¨™æº–åŒ–ï¼ˆå¹³å�‡0ãƒ»åˆ†æ•£1ã�«å¤‰æ�›ï¼‰ã€�ãƒ‡ãƒ¼ã‚¿å�‹ã‚’float32ã�«å¤‰æ�›

# MixUpãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã‚’è¡Œã�„ã€�ãƒ‹ãƒ¥ãƒ¼ãƒ©ãƒ«ãƒ�ãƒƒãƒˆãƒ¯ãƒ¼ã‚¯ã�®æ­£å‰‡åŒ–ã‚’å¼·åŒ–ã�™ã‚‹ã‚¯ãƒ©ã‚¹
class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y  # å…¥åŠ›ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ©ãƒ™ãƒ«
        self.batch = batch_size  # ãƒ�ãƒƒãƒ�ã‚µã‚¤ã‚º
        self.alpha = alpha  # ãƒ™ãƒ¼ã‚¿åˆ†å¸ƒã�®ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ï¼ˆMixUpã�®æ¯”ç�‡èª¿æ•´ï¼‰
        self.indices = np.arange(len(X))  # ãƒ‡ãƒ¼ã‚¿ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’ä½œæˆ�
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))  # ãƒŸãƒ‹ãƒ�ãƒƒãƒ�ã�®ç·�æ•°ã‚’è¨ˆç®—
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]  # ãƒ�ãƒƒãƒ�ã�«å¯¾å¿œã�™ã‚‹ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’å�–å¾—
        Xb, yb = self.X[idx], self.y[idx]  # å¯¾å¿œã�™ã‚‹ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ©ãƒ™ãƒ«ã‚’å�–å¾—
        
        lam = np.random.beta(self.alpha, self.alpha)  # ãƒ™ãƒ¼ã‚¿åˆ†å¸ƒã�‹ã‚‰MixUpä¿‚æ•°ã‚’å�–å¾—
        perm = np.random.permutation(len(Xb))  # ãƒ‡ãƒ¼ã‚¿ã�®ã‚·ãƒ£ãƒƒãƒ•ãƒ«ç”¨ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹
        
        X_mix = lam * Xb + (1 - lam) * Xb[perm]  # MixUpã�«ã‚ˆã‚Šãƒ‡ãƒ¼ã‚¿ã‚’ç·šå½¢è£œé–“
        y_mix = lam * yb + (1 - lam) * yb[perm]  # ãƒ©ãƒ™ãƒ«ã‚‚å�Œæ§˜ã�«MixUp
        
        return X_mix, y_mix  # MixUpå¾Œã�®ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ©ãƒ™ãƒ«ã‚’è¿”ã�™
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)  # ã‚¨ãƒ�ãƒƒã‚¯çµ‚äº†æ™‚ã�«ãƒ‡ãƒ¼ã‚¿ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’ã‚·ãƒ£ãƒƒãƒ•ãƒ«




from tensorflow.keras.layers import Lambda

# âœ… FFTã‚’ç”¨ã�„ã�Ÿä½�ãƒ»ä¸­ãƒ»é«˜å‘¨æ³¢ã�®ç‰¹å¾´æŠ½å‡ºé–¢æ•°ï¼ˆ3åˆ†å‰²ï¼‰
def fft_low_mid_high_features(tensor):
    fft_values = tf.signal.fft(tf.cast(tensor, dtype=tf.complex64))  # FFTå¤‰æ�›
    fft_magnitude = tf.abs(fft_values)  # æŒ¯å¹…ã‚¹ãƒšã‚¯ãƒˆãƒ«ï¼ˆçµ¶å¯¾å€¤ï¼‰

    freq_len = tf.shape(fft_magnitude)[-1]  # å‘¨æ³¢æ•°æ•°ï¼ˆã‚»ãƒ³ã‚µãƒ¼ãƒ‡ãƒ¼ã‚¿ã�®é•·ã�•ï¼‰

    # æ¯”ç�‡ã‚’ TensorFlow å®šæ•°ã�«å¤‰æ�›ï¼ˆå�‹å®‰å…¨ã�«ï¼‰
    ratio_30 = tf.constant(0.3, dtype=tf.float32)
    ratio_70 = tf.constant(0.7, dtype=tf.float32)

    # ä¸‰åˆ†å‰²ãƒ�ã‚¤ãƒ³ãƒˆï¼ˆlow: 0ã€œ30%ã€�mid: 30ã€œ70%ã€�high: 70ã€œ100%ï¼‰
    low_end = tf.cast(tf.math.floor(tf.cast(freq_len, tf.float32) * ratio_30), tf.int32)
    high_start = tf.cast(tf.math.floor(tf.cast(freq_len, tf.float32) * ratio_70), tf.int32)

    # å�„å‘¨æ³¢æ•°å¸¯ã�®çµ±è¨ˆç‰¹å¾´ã‚’æŠ½å‡º
    low_freq = tf.reduce_mean(fft_magnitude[:, :, :low_end], axis=-1, keepdims=True)
    mid_freq = tf.reduce_mean(fft_magnitude[:, :, low_end:high_start], axis=-1, keepdims=True)
    high_freq = tf.reduce_mean(fft_magnitude[:, :, high_start:], axis=-1, keepdims=True)

    # ä½�ãƒ»ä¸­ãƒ»é«˜å‘¨æ³¢ç‰¹å¾´ã‚’çµ�å�ˆ
    return tf.concat([low_freq, mid_freq, high_freq], axis=-1)

# âœ… 2ãƒ–ãƒ©ãƒ³ãƒ�ãƒ¢ãƒ‡ãƒ«ã�®æ§‹ç¯‰ï¼ˆFFTä¸‰åˆ†å‰²ç‰ˆï¼‰
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim + tof_dim))

    # å…¥åŠ›ã‚’IMU / TOFã�«åˆ†é›¢
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # âœ… FFTç‰¹å¾´ã‚’IMUã�«è¿½åŠ ï¼ˆä½�ãƒ»ä¸­ãƒ»é«˜å‘¨æ³¢ã�®ä¸‰åˆ†å‰²ï¼‰
    imu_freq_features = Lambda(fft_low_mid_high_features)(imu)
    imu = Concatenate()([imu, imu_freq_features])

    # IMUå‡¦ç�†ãƒ–ãƒ©ãƒ³ãƒ�ï¼ˆResidual SE-CNNï¼‰
    x1 = residual_se_cnn_block(imu, 128, 3, drop=0.4, wd=wd)
    x1 = residual_se_cnn_block(x1, 256, 5, drop=0.4, wd=wd)

    # TOFå‡¦ç�†ãƒ–ãƒ©ãƒ³ãƒ�ï¼ˆè»½é‡�Conv1Dï¼‰
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)

    # çµ�å�ˆ â†’ LSTM â†’ Attention
    merged = Concatenate()([x1, x2])
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.5)(x)
    x = attention_layer(x)

    # Denseå±¤ã�«ã‚ˆã‚‹åˆ†é¡�
    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)



if TRAIN:
    print("â–¶ TRAIN MODE â€“ loading dataset â€¦")
    df = pd.read_csv(RAW_DIR / "train.csv")  # å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã‚’èª­ã�¿è¾¼ã‚€

    # ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
    le = LabelEncoder(); df['gesture_int'] = le.fit_transform(df['gesture'])  # gestureã‚’æ•´æ•°ã�«å¤‰æ�›
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)  # å¤‰æ�›å¾Œã�®ã‚¯ãƒ©ã‚¹æƒ…å ±ã‚’ä¿�å­˜

    # ç‰¹å¾´é‡�ãƒªã‚¹ãƒˆã�®ä½œæˆ�
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}  # ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã�®åˆ—
    feature_cols = [c for c in df.columns if c not in meta_cols]  # ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã‚’é™¤ã�„ã�Ÿç‰¹å¾´é‡�ã�®åˆ—

    imu_cols  = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]  # IMUã‚»ãƒ³ã‚µãƒ¼ãƒ‡ãƒ¼ã‚¿
    tof_cols  = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]  # TOF/THMã‚»ãƒ³ã‚µãƒ¼ãƒ‡ãƒ¼ã‚¿
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")  # ç‰¹å¾´é‡�æ•°ã‚’å‡ºåŠ›

    # [ä¿®æ­£] ã‚»ãƒ³ã‚µãƒ¼ãƒ¢ãƒ€ãƒªãƒ†ã‚£ã�”ã�¨ã�«åˆ¥ã€…ã�®StandardScalerã‚’å­¦ç¿’ã�•ã�›ã‚‹
    #  IMUã�¯å‹•ã��ã�®ã‚¹ã‚±ãƒ¼ãƒ«ã€�TOF/THMã�¯è·�é›¢ãƒ»æ¸©åº¦ã�¨ã�„ã�†ç•°ã�ªã‚‹ç‰©ç�†é‡�ã‚’æŒ�ã�¤ã�Ÿã‚�ã€�åˆ¥ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°ã�Œæœ‰åŠ¹
    scaler_imu = StandardScaler().fit(df[imu_cols].fillna(0))  # IMUã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼
    scaler_tof = StandardScaler().fit(df[tof_cols].fillna(0))  # TOF/THMã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼

    # [ä¿®æ­£] ã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼ã‚’ä¿�å­˜ï¼ˆinferenceç”¨ã�«ï¼‰
    joblib.dump(scaler_imu, EXPORT_DIR / "scaler_imu.pkl")
    joblib.dump(scaler_tof, EXPORT_DIR / "scaler_tof.pkl")

    # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã�®æ§‹ç¯‰
    seq_gp = df.groupby('sequence_id')  # sequence_idã�”ã�¨ã�«ã‚°ãƒ«ãƒ¼ãƒ—åŒ–
    X_list, y_list, lens = [], [], []

    for seq_id, seq in seq_gp:
        # [ä¿®æ­£] ãƒ¢ãƒ€ãƒªãƒ†ã‚£ã�”ã�¨ã�«æ¨™æº–åŒ–å‡¦ç�†ã‚’åˆ†é›¢ã�—ã�¦é�©ç”¨
        imu = scaler_imu.transform(seq[imu_cols].fillna(0))
        tof = scaler_tof.transform(seq[tof_cols].fillna(0))
        mat = np.hstack([imu, tof]).astype('float32')  # çµ�å�ˆã�—ã�¦1ã�¤ã�®å…¥åŠ›é…�åˆ—ã�«ã�™ã‚‹
        X_list.append(mat)  # å‰�å‡¦ç�†æ¸ˆã�¿ç‰¹å¾´é‡�ã‚’ä¿�å­˜
        y_list.append(seq['gesture_int'].iloc[0])  # ã‚¸ã‚§ã‚¹ãƒ�ãƒ£ãƒ¼ã‚¯ãƒ©ã‚¹ã�®ä¿�å­˜
        lens.append(len(mat))  # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã�®é•·ã�•ã‚’è¨˜éŒ²

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))  # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·ã�®ãƒ‘ãƒ‡ã‚£ãƒ³ã‚°é–¾å€¤ã‚’è¨­å®š
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)  # æœ€å¤§ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·ã‚’ä¿�å­˜
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))  # ä½¿ç”¨ã�™ã‚‹ç‰¹å¾´é‡�ãƒªã‚¹ãƒˆã‚’ä¿�å­˜

    X = pad_sequences(X_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')  # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ã‚’ãƒ‘ãƒ‡ã‚£ãƒ³ã‚°
    y = to_categorical(y_list, num_classes=len(le.classes_))  # One-hotã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�§ã‚«ãƒ†ã‚´ãƒªåŒ–

    # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«åˆ†å‰²
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_list)  # stratified split

    # ã‚¯ãƒ©ã‚¹ã�®é‡�ã�¿ä»˜ã�‘ã‚’è¨ˆç®—
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_list)  # ã‚¯ãƒ©ã‚¹ãƒ�ãƒ©ãƒ³ã‚¹ã‚’è€ƒæ…®ã�—ã�Ÿé‡�ã�¿
    class_weight = dict(enumerate(cw_vals))  # é‡�ã�¿ã‚’è¾�æ›¸ã�«å¤‰æ�›

    # ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰
    model = build_two_branch_model(pad_len, len(imu_cols), len(tof_cols), len(le.classes_), wd=WD)  # ãƒ¢ãƒ‡ãƒ«ã‚’å®šç¾©
    steps = len(X_tr)//BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(LR_INIT, first_decay_steps=5*steps)  # å­¦ç¿’ç�‡ã�®ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ«è¨­å®š
    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])  # ãƒ¢ãƒ‡ãƒ«ã�®ã‚³ãƒ³ãƒ‘ã‚¤ãƒ«

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)  # Mixupã�«ã‚ˆã‚‹ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µ
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)  # æ—©æœŸå�œæ­¢ã�®è¨­å®š
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=class_weight, callbacks=[cb], verbose=1)  # ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")  # è¨“ç·´æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
    print("âœ” Training done â€“ artefacts saved in", EXPORT_DIR)  # è¨“ç·´å®Œäº†ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸

    # Hold-outãƒ‡ãƒ¼ã‚¿ã�®è©•ä¾¡
    from cmi_2025_metric_copy_for_import import CompetitionMetric
    preds = model.predict(X_val).argmax(1)  # æ�¨è«–çµ�æ�œã‚’å�–å¾—
    true  = y_val.argmax(1)  # æ­£è§£ãƒ©ãƒ™ãƒ«
    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[true]}),
        pd.DataFrame({'gesture': le.classes_[preds]}))  # ãƒ„ãƒªãƒ¼æ§‹é€ ã�«åŸºã�¥ã��F1ã‚¹ã‚³ã‚¢ã�®è¨ˆç®—
    print("Holdâ€‘out Hâ€‘F1 =", round(h_f1, 4))  # ã‚¹ã‚³ã‚¢ã‚’å‡ºåŠ›


# [ä¿®æ­£] æ­£ã�—ã�„ä¿�å­˜å…ˆï¼ˆworkingãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªï¼‰ã‚’æŒ‡å®š
PRETRAINED_DIR = Path("/kaggle/working")

# make sure gesture_classes exists in both modes
if TRAIN:
    gesture_classes = le.classes_

# [ä¿®æ­£] æ�¨è«–æ™‚ã�«ä½¿ç”¨ã�™ã‚‹2ã�¤ã�®ã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼ã‚’ãƒ­ãƒ¼ãƒ‰ï¼ˆIMUç”¨ãƒ»TOF/THMç”¨ï¼‰
scaler_imu = joblib.load(PRETRAINED_DIR / "scaler_imu.pkl")
scaler_tof = joblib.load(PRETRAINED_DIR / "scaler_tof.pkl")

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    df_seq = sequence.to_pandas()

    # IMUã�¨TOF/THMã‚’å€‹åˆ¥ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°ã�—ã€�çµ�å�ˆã�™ã‚‹
    imu = scaler_imu.transform(df_seq[imu_cols].fillna(0))
    tof = scaler_tof.transform(df_seq[tof_cols].fillna(0))
    mat = np.hstack([imu, tof]).astype('float32')

    # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·ã‚’ãƒ‘ãƒ‡ã‚£ãƒ³ã‚°ã�—ã�¦ãƒ¢ãƒ‡ãƒ«å…¥åŠ›ã�«æ•´å½¢
    pad = pad_sequences([mat], maxlen=pad_len, padding='post', truncating='post', dtype='float32')

    # äºˆæ¸¬ã‚¯ãƒ©ã‚¹ã‚’è¿”ã�™
    idx = int(model.predict(pad, verbose=0).argmax(1)[0])
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

