# # IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# # RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
# import kagglehub
# kagglehub.login()


# # IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# # THEN FEEL FREE TO DELETE THIS CELL.
# # NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# # ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# # NOTEBOOK.
# ENVIRONMENT = 'kaggle'

# if ENVIRONMENT == 'colab':
#     DATA_PATH = kagglehub.competition_download('cmi-detect-behavior-with-sensor-data')
# elif ENVIRONMENT == 'kaggle':
#     DATA_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
# else:
#     print("ENvironment is not recognized")
# print('Data source imported to: ', DATA_PATH)


print("import libraries...")

# statistical modules
import numpy as np
import pandas as pd

# visualization moodules
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# machine learning modules
import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, losses, metrics, initializers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import pad_sequences, to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# other modules
from tqdm.notebook import tqdm
from pprint import pprint
import warnings
import joblib
import random
import sys
import os
import gc # garbage collection module


DEBUG_MODE = "ignore"
warnings.filterwarnings(action=DEBUG_MODE)

# Enable Mixed Precision Training
# This can significantly speed up training on modern GPUs/TPUs by using float16 for computations
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)


# Enable XLA
# tf.config.optimizer.set_jit(True)


# Sets a uniform random seed for reproducibility across different libraries.
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

seed_everything(42)

# print tensorflow metrics
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# # load tain dataset
# train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

# # Optional unimportant datasets
# # test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
# # train_demographics_df = pd.read_csv(os.path.join(DATA_PATH, 'train_demographics.csv'))
# # test_demographics_df = pd.read_csv(os.path.join(DATA_PATH, 'test_demographics.csv'))


# implemented in another notebook


# train_df.groupby("sequence_id").agg({"sequence_counter":"count"}).hist(bins=100)


# imu_thm_cols = [c for c in train_df.columns if c.startswith(('acc_','rot_', 'thm_'))]
# tof_cols = [c for c in train_df.columns if c.startswith('tof_')]


# # Nulls ratio per IMU/THM column per sequence_id
# imu_thm_nulls_ratio = train_df.groupby('sequence_id')[imu_thm_cols].apply(lambda x: x.isna().mean())
# # All-TOF-Pixels Nulls ratio per sequence_id
# tof_nulls_ratio = (
#     train_df.groupby('sequence_id')[tof_cols]
#     .apply(lambda x: x.isna().sum().sum() / x.size)
#     .rename('tof')
# )
# # Nulls ratio summary
# summary = pd.concat([imu_thm_nulls_ratio, tof_nulls_ratio], axis=1)

# # final mask: any ratio >= 10%
# mask = summary[imu_thm_cols].ge(0.1).any(axis=1) | summary['tof'].ge(0.1)
# to_ignore = summary.index[mask].tolist()

# # Clean train_df
# train_df = train_df[~train_df['sequence_id'].isin(to_ignore)]
# train_df[imu_thm_cols+tof_cols] = train_df[imu_thm_cols+tof_cols].interpolate().ffill().bfill()


# Utility functions

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        LAMBDA = np.random.beta(a=alpha, b=alpha)
    else:
        LAMBDA = tf.constant(1.0, dtype=tf.float32) # Use tf.constant for consistency

    batch_size = tf.shape(x)[0]
    # Use tf.random.shuffle for shuffling within TensorFlow graph
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_x = LAMBDA * x + (1 - LAMBDA) * tf.gather(x, indices)
    mixed_y = LAMBDA * y + (1 - LAMBDA) * tf.gather(y, indices)
    return mixed_x, mixed_y

def plot_interactive_history(history):
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # minimal gap between rows :contentReference[oaicite:1]{index=1}
        subplot_titles=["Learning Rate", "Loss", "F1 Score", "Accuracy"]
    )
    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='#098C18')), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='#33B84E', dash='dash')), row=2, col=1)

    # Accuracy
    # fig.add_trace(go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', line=dict(color='#A34833')), row=4, col=1)
    # fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', line=dict(color='#CF796C', dash='dash')), row=4, col=1)

    # F1 Score
    fig.add_trace(go.Scatter(x=epochs, y=history['train_f1'], name='Train F1', line=dict(color='#3AA69F')), row=3, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_f1'], name='Val F1', line=dict(color='#64D9D4', dash='dash')), row=3, col=1)

    # Learning Rate
    fig.add_trace(go.Scatter(x=epochs, y=history['learning_rate'], name='Learning Rate', line=dict(color='#A9B033')), row=1, col=1)

    fig.update_layout(
        height=1200, width=800,
        margin=dict(t=50, b=50, l=50, r=50),  # smaller margins :contentReference[oaicite:2]{index=2}
        showlegend=False
    )    
    fig.update_xaxes(title_text="Epoch", row=4, col=1)
    fig.update_yaxes(title_text="Lr", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="F1", row=3, col=1)
    fig.update_yaxes(title_text="Accuracy", row=4, col=1)
    
    fig.show()

def unify_len(seq_df: pd.DataFrame, unified_len: int, padding_strategy: str = "post", padding_value = 0.0) -> pd.DataFrame:
    """
    Processes a single sequence in a DataFrame to a unified length by truncating or padding.
    Returns A new DataFrame with the sequence processed to the unified length.
    """
    if padding_strategy not in ["pre", "post"]:
        raise ValueError("padding_strategy must be either 'pre' or 'post'")

    n_rows = len(seq_df)
    processed_df = seq_df # Default to no change

    if n_rows > unified_len:
        # Truncation: Keep the last `unified_len` rows
        processed_df = seq_df.tail(unified_len)
        
    elif n_rows < unified_len:
        # Padding: Add rows of NaNs to meet the unified_len
        pad_size = unified_len - n_rows
        
        # Create a padding DataFrame with NaN values
        pad_df = pd.DataFrame(padding_value, index=range(pad_size), columns=seq_df.columns)

        if padding_strategy == "pre":
            # Add padding before the sequence data
            processed_df = pd.concat([pad_df, seq_df], ignore_index=True)
        else:  # "post"
            # Add padding after the sequence data
            processed_df = pd.concat([seq_df, pad_df], ignore_index=True)
            
    return processed_df.reset_index(drop=True)

# macro F1Score for multiclass classification
class F1ScoreMacro(metrics.Metric):
    """
    Computes the macro F1 score from a confusion matrix.
    """
    def __init__(self, num_classes, name='f1_score_macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # State variable for the confusion matrix
        self.confusion_matrix = self.add_weight(
            name='cm',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and labels to indices
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)

        # Update the confusion matrix for the current batch
        batch_cm = tf.math.confusion_matrix(
            y_true_labels, y_pred_labels, num_classes=self.num_classes, dtype=tf.int32
        )
        self.confusion_matrix.assign_add(batch_cm)

    def result(self):
        cm = self.confusion_matrix
        # Calculate True Positives, False Positives, False Negatives for each class
        tp = tf.cast(tf.linalg.tensor_diag_part(cm), dtype=tf.float32)
        fp = tf.cast(tf.reduce_sum(cm, axis=0), dtype=tf.float32) - tp
        fn = tf.cast(tf.reduce_sum(cm, axis=1), dtype=tf.float32) - tp


        # Calculate precision and recall for each class
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())

        # Calculate F1 score for each class
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        # Handle cases where a class has no examples by masking out NaN values.
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

        # Compute the macro average
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1

    def reset_state(self):
        # Reset the confusion matrix to zeros
        for v in self.variables:
            v.assign(tf.zeros(shape=v.shape, dtype=v.dtype))


class AttentionResampler(layers.Layer):
    def __init__(self, pad_len, out_dim, key_dim=64, **kw):
        super().__init__(**kw)
        self.pad_len = pad_len
        self.key_dim = key_dim
        self.query = self.add_weight("queries", shape=(pad_len, key_dim),
                                     initializer=initializers.TruncatedNormal(stddev=0.02))
        self.to_k = layers.Dense(key_dim, use_bias=False)
        self.to_v = layers.Dense(out_dim, use_bias=False)

    def call(self, x, mask=None):
        # x: [B, T, C]
        K = self.to_k(x)          # [B, T, key_dim]
        V = self.to_v(x)          # [B, T, out_dim]
        Q = tf.expand_dims(self.query, axis=0)  # [1, pad_len, key_dim]
        Q = tf.tile(Q, [tf.shape(x)[0], 1, 1])  # [B, pad_len, key_dim]

        # Scaled dot-product attention
        attn_logits = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        if mask is not None:
            # mask: [B, T] with 1 = valid. Convert to addable mask
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=1)   # [B, 1, T]
            attn_logits += (1.0 - mask) * -1e9

        attn = tf.nn.softmax(attn_logits, axis=-1)  # [B, pad_len, T]
        out = tf.matmul(attn, V)                    # [B, pad_len, out_dim]
        return out


# # Prepare labels
# print("Label encoding ...")
# label_encoder = LabelEncoder()
# train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))

# gesture_classes = label_encoder.classes_
# num_classes = len(gesture_classes)
# np.save('./gesture_classes.npy', gesture_classes)

# # Get relevan features
# imu_cols = [c for c in train_df.columns if (c.startswith('acc_') or c.startswith('rot_'))]
# tof_thm_cols = [c for c in train_df.columns if (c.startswith('thm_') or c.startswith('tof_'))]
# feature_cols = imu_cols + tof_thm_cols

# np.save("./feature_cols.npy", feature_cols)
# num_features = len(feature_cols)
# print(f"Got {num_features} features.")


# # Prepare Global Scaler
# print("Fitting StandardScaler...")
# values_for_scaling = train_df[feature_cols].astype('float32').interpolate().ffill().bfill().fillna(0).values
# scaler = StandardScaler().fit(values_for_scaling)
# joblib.dump(scaler, './global_scaler.pkl')

# del values_for_scaling
# gc.collect()


# # Sequence-wise Processing
# print("Sequences-wise Processing Started...")
# print("Flow: [Sort -> Scale -> Pad]")

# # get seuqnces
# sequences_grouped = train_df.groupby('sequence_id')
# n_samples = len(sequences_grouped)
# print(f"Found {n_samples} sequences.")

# # del train_df
# gc.collect()

# # Get Pad length
# # pad_len = int(sequences_grouped.size().max())
# pad_len = 200
# np.save("./sequence_maxlen.npy", pad_len)

# # Prepare X and y
# X = np.zeros((n_samples, pad_len, num_features), dtype='float32')
# y = np.zeros(n_samples, dtype='int32')

# for i, (_, seq_df) in tqdm(enumerate(sequences_grouped), total=n_samples, desc="ğŸ’ Progress"):

#     # 1. Sorting
#     _sequence = seq_df.sort_values(by="sequence_counter")[feature_cols].astype('float32')

#     # unify length
#     _sequence = unify_len(_sequence, unified_len= pad_len)
    
#     # clean
#     _sequence = _sequence.interpolate().ffill().bfill().fillna(0)
    
#     # 3. Scaling
#     _sequence = scaler.transform(_sequence.values)

#     # 4. Padding
#     _sequence = pad_sequences(
#         [_sequence],
#         maxlen=pad_len,
#         dtype='float32',
#         padding='post',
#         truncating='post'
#     )[0]

#     # Resutls
#     X[i] = _sequence
#     y[i] = seq_df['gesture'].iloc[0]

# # Cleanup
# del sequences_grouped
# gc.collect()


# # Split the entire dataset into 90% dev and 10% test
# print("Splitting data into 90% dev and 10% test...")
# X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
# print(f"Dev set shape: {X_dev.shape}, Test set shape: {X_test.shape}")

# del X, y
# gc.collect()


# # Split the dev set into 90% train and 10% validation
# print("Splitting dev set into 90% train and 10% validation...")
# X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1, stratify=y_dev)
# total_train_samples = X_train.shape[0]
# print(f"Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

# del X_dev, y_dev
# gc.collect()


# # Create tf.data.Dataset pipelines for train, validation, and test sets
# print("Creating tf.data.Dataset for train, validation, and test sets...")
# AUTOTUNE = tf.data.AUTOTUNE
# BATCH_SIZE = 64

# # Train Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = (
#     train_dataset
#     .shuffle(buffer_size=len(y_train))
#     .map(lambda features, label: (features, tf.one_hot(label, depth=num_classes)), num_parallel_calls=AUTOTUNE)
#     .batch(BATCH_SIZE, drop_remainder=True)
#     .prefetch(AUTOTUNE)
# )
# print("âœ”ï¸�Train Dataset Ready.")
# del X_train, y_train
# gc.collect()


# # Validation Dataset
# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_dataset = (
#     val_dataset
#     .map(lambda features, label: (features, tf.one_hot(label, depth=num_classes)), num_parallel_calls=AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTOTUNE)
# )
# print("âœ”ï¸�Validation Dataset Ready.")
# del X_val, y_val
# gc.collect()


# # Test Dataset
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# test_dataset = (
#     test_dataset
#     .map(lambda features, label: (features, tf.one_hot(label, depth=num_classes)), num_parallel_calls=AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTOTUNE)
# )
# print("âœ”ï¸�Test Dataset Ready.")
# del X_test, y_test
# gc.collect()


# 4. Model definition 
# Two-branche: IMU: SE-CNN + TOF/Thm: CNN -> BiLSTM -> Attention -> Densely Head
print("Model definition...")

class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.fc1 = layers.Dense(channels // reduction, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, x):                  # x: (batch, seq_len, channels)
        se = tf.reduce_mean(x, axis=1)  # (batch, channels)
        se = self.fc1(se)               # (batch, channels//reduction)
        se = self.fc2(se)               # (batch, channels)
        se = tf.expand_dims(se, axis=1) # (batch, 1, channels)
        return x * se                   # scale channels: (batch, seq_len, channels) * (batch, 1, channels)

class ResidualSEBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3, reg=regularizers.l2(5e-4), **kwargs):
        super(ResidualSEBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(out_channels, kernel_size, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv1D(out_channels, kernel_size, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn2 = layers.BatchNormalization() 
        self.se = SEBlock(out_channels, reduction=8)
        self.pool = layers.MaxPooling1D(pool_size=pool_size)
        self.dropout = layers.Dropout(dropout_rate)
        self.shortcut_conv = None
        self.shortcut_bn = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        out_channels = self.conv1.filters
        if in_channels != out_channels:
            self.shortcut_conv = layers.Conv1D(out_channels, kernel_size=1, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, x, training=False):

        # prepare shortcut (for skip connection)
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        # out = self.relu(out)

        # Squeeze-and-Excitation
        out = self.se(out)

        # Skip connection
        out = layers.add([out, shortcut])

        # Head
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out, training=training)

        return out

class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.score_fc = layers.Dense(1, activation='tanh')

    def call(self, x):                              # x: (batch, seq_len, features)
        scores = self.score_fc(x)                   # (batch, seq_len, 1)
        weights = layers.Softmax(axis=1)(scores)    # (batch, seq_len, 1)
        context = x * weights                       # (batch, seq_len, features)
        context = tf.reduce_sum(context, axis=1)    # (batch, features)
        return context

def TwoBranchModel(
    total_features, imu_dim, tof_thm_dim, 
    pad_len, num_classes, reg=regularizers.l2(5e-4)):
    
    # Input shape: (batch, seq_len, features)
    input_layer = layers.Input(shape=(pad_len, total_features))

    # Mask zero padding (experimental)
    # mask = layers.Masking(mask_value=0.0)(input_layer)

    # Split inputs
    x_imu = input_layer[:, :, :imu_dim]
    x_ttf = input_layer[:, :, imu_dim:]

    # IMU branch
    # b1 = ResidualSEBlock(64, kernel_size=3, pool_size=2, dropout_rate=0.3)(x_imu)
    b1 = ResidualSEBlock(64, kernel_size=5, pool_size=4, dropout_rate=0.3, reg=reg)(x_imu)

    # TOF/Thermal branch
    b2 = layers.Conv1D(32, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=reg)(x_ttf)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)
    b2 = layers.MaxPooling1D(pool_size=2)(b2)
    b2 = layers.Dropout(0.3)(b2)

    b2 = layers.Conv1D(64, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=reg)(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)
    b2 = layers.MaxPooling1D(pool_size=2)(b2)
    b2 = layers.Dropout(0.3)(b2)

    # Concatenate branches along channel dimension
    merged = layers.Concatenate(axis=-1)([b1, b2])

    # BiLSTM
    lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(merged)
    lstm_out = layers.Dropout(0.4)(lstm_out)

    # Attention
    context = Attention()(lstm_out)

    # Dense head
    x = layers.Dense(128, use_bias=False, kernel_regularizer=reg)(context)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  
    x = layers.Dropout(0.3)(x)

    # Output layer
    out = layers.Dense(num_classes)(x) # Logits output

    return Model(inputs=input_layer, outputs=out)


# # 5. Training setup (optimizer, loss, scheduler)


# # total_steps = num_epochs * np.ceil(total_train_samples / BATCH_SIZE).astype('int16')
# # decay_rate = (lr / 100) ** (1 / total_steps)
# # lr_scheduler = optimizers.schedules.ExponentialDecay(
# #     initial_learning_rate=lr,
# #     decay_steps=1,         # apply decay every step
# #     decay_rate=decay_rate, # tiny per-step multiplicative factor
# #     staircase=False        # continuous (set True for discrete jumps)
# # )

# # steps_per_epoch = np.ceil(total_train_samples / BATCH_SIZE).astype('int16')
# # lr_scheduler = FactorDecay(
# #     initial_lr=lr, 
# #     steps_per_epoch=steps_per_epoch,
# #     decay_every_epochs=15, 
# #     decay_factor=2.0)

# # Main Training hyperparamet/erslr = 1e-3

# lr = 2e-3
# num_epochs = 160
# patience = 30

# # CosineAnnealingWarmRestarts equivalent
# first_decay_steps = 5 * np.ceil(total_train_samples / BATCH_SIZE)
# lr_scheduler = optimizers.schedules.CosineDecayRestarts(
#     initial_learning_rate=lr,
#     first_decay_steps=first_decay_steps,  # First Decay after x Epochs
#     t_mul=1.5,                            # After restart multiply period len by t_mul
#     m_mul=0.9,                           # After restart multiply current lr by m_mul
#     alpha=1e-5 / lr                       # Minimum value lr will reach (eta_min)
# )
# optimizer = optimizers.AdamW(learning_rate=lr_scheduler)
# # optimizer = optimizers.Adam(learning_rate=lr_scheduler, weight_decay=5e-4)

# # Loss: CategoricalCrossentropy with from_logits=True handles soft labels and log_softmax
# loss_fn = losses.CategoricalCrossentropy(from_logits=True)

# # Metrics
# train_loss_metric = metrics.Mean(name='train_loss')
# val_loss_metric = metrics.Mean(name='val_loss')
# train_accuracy_metric = metrics.CategoricalAccuracy(name='train_accuracy')
# val_accuracy_metric = metrics.CategoricalAccuracy(name='val_accuracy')
# train_f1_metric = F1ScoreMacro(num_classes=18, name='train_f1')
# val_f1_metric = F1ScoreMacro(num_classes=18, name='val_f1')


# # Instantiate model
# model = TwoBranchModel(
#     total_features=len(feature_cols),
#     imu_dim=len(imu_cols),
#     tof_thm_dim=len(tof_thm_cols),
#     pad_len=pad_len,
#     num_classes=18,
#     reg=regularizers.l2(5e-4)
# )

# # Reset monitoring indices
# best_val_loss = np.inf
# best_f1 = -np.inf
# epochs_no_improve = 0
# best_weights = None
# history = {
#     'train_loss': [],
#     'train_f1': [],
#     'val_loss': [],
#     'val_f1': [],
#     'learning_rate': []
#     # 'train_acc': [],
#     # 'val_acc': [],
# }


# # 6. Training with Mixup
# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         predictions = model(x, training=True)
#         loss = loss_fn(y, predictions)

#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss_metric.update_state(loss)
#     train_f1_metric.update_state(y, predictions)
#     # train_accuracy_metric.update_state(y, predictions)

# @tf.function
# def val_step(x, y):
#     predictions = model(x, training=False)
#     loss = loss_fn(y, predictions)

#     val_loss_metric.update_state(loss)
#     val_f1_metric.update_state(y, predictions)
#     # val_accuracy_metric.update_state(y, predictions)


# print("Train is Starting...")
# # For each epoch
# for epoch in range(1, num_epochs + 1):
#     # Reset metrics at the start of each epoch
#     train_loss_metric.reset_state()
#     train_f1_metric.reset_state()
#     val_loss_metric.reset_state()
#     val_f1_metric.reset_state()
#     # train_accuracy_metric.reset_state()
#     # val_accuracy_metric.reset_state()

#     # Training loop
#     for batch_x, batch_y in tqdm(train_dataset, desc=f"ğŸ’ Epoch {epoch}/{num_epochs}"):
#         mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=0.4)
#         train_step(mixed_x, mixed_y)

#     # Validation loop
#     for batch_x_val, batch_y_val in val_dataset:
#         val_step(batch_x_val, batch_y_val)

#     # Metrics monitoring
#     train_loss = train_loss_metric.result()
#     train_f1 = train_f1_metric.result()
#     val_loss = val_loss_metric.result()
#     val_f1 = val_f1_metric.result()
#     current_lr = optimizer.learning_rate
#     # train_acc = train_accuracy_metric.result()
#     # val_acc = val_accuracy_metric.result()
    
#     # Save metrics
#     history['train_loss'].append(train_loss.numpy())
#     history['train_f1'].append(train_f1.numpy())
#     history['val_loss'].append(val_loss.numpy())
#     history['val_f1'].append(val_f1.numpy())
#     history['learning_rate'].append(current_lr.numpy())
#     # history['train_acc'].append(train_acc.numpy())
#     # history['val_acc'].append(val_acc.numpy())
    
#     print(
#         f"  Tr Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}, ",
#         f"Tr F1 = {train_f1:.3f},  Val F1 = {val_f1:.3f}, Lr = {current_lr.numpy():.5f}",
#     )

#     # Earlyâ€‘stopping check
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         epochs_no_improve = 0
#         print(f"  â†—ï¸� New best val loss: {best_val_loss:.4f}")
#         model.save_weights(f"best_model.weights.h5") # Save best model weights
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             model.save_weights(f"best_model_f1.weights.h5") # Save best model weights
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"âš ï¸� Early stopping at epoch {epoch}.")
#             model.load_weights(f"best_model.weights.h5") # Restore best model weights
#             break


# plot_interactive_history(history)


# # Evaluate the model on the test set

# # Initialize metrics for testing
# test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
# # test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
# test_f1_metric = F1ScoreMacro(name='test_f1_score', num_classes=18) # Using the custom F1Score metric

# # Reset metrics before evaluation
# test_loss_metric.reset_state()
# # test_accuracy_metric.reset_state()
# test_f1_metric.reset_state()

# # Evaluate the model on the test dataset
# print("Evaluating model on the test dataset...")
# y_true = []
# y_pred = []

# for batch_x_test, batch_y_test in tqdm(test_dataset, desc="Testing"):
#     predictions = inference_model(batch_x_test, training=False)
#     loss = loss_fn(batch_y_test, predictions)

#     test_loss_metric.update_state(loss)
#     # test_accuracy_metric.update_state(batch_y_test, predictions)
#     test_f1_metric.update_state(batch_y_test, predictions)

#     # Collect true and predicted labels for classification report
#     y_true.extend(tf.argmax(batch_y_test, axis=-1).numpy())
#     y_pred.extend(tf.argmax(predictions, axis=-1).numpy())


# test_loss = test_loss_metric.result()
# # test_acc = test_accuracy_metric.result()
# test_f1 = test_f1_metric.result()

# print(f"Test Results: Loss = {test_loss:.4f}, F1 Score = {test_f1:.4f}")

# # Generate and print classification report
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=gesture_classes))


# 7. Inference function for Kaggle evaluation

import kaggle_evaluation.cmi_inference_server
import polars as pl

# Load artifacts
print("load model artifacts...")
gesture_classes = np.load("/kaggle/input/artifacts/gesture_classes.npy", allow_pickle=True)
pad_len = 200
# pad_len = int(np.load("/kaggle/input/artifacts/sequence_maxlen.npy", allow_pickle=True))
scaler = joblib.load("/kaggle/input/artifacts/global_scaler.pkl")
feature_cols = np.load("/kaggle/input/artifacts/feature_cols.npy", allow_pickle=True)
# Recreate model and load weights
print("Creating model and loading weights...")
inference_model = TwoBranchModel(
    total_features=332,
    imu_dim=7,
    tof_thm_dim=325,
    pad_len=pad_len,
    num_classes=len(gesture_classes),
    reg=regularizers.l2(5e-4)
)
# Build the model by calling it on a dummy input
dummy_input = np.zeros((1, pad_len, 332), dtype=np.float32)
_ = inference_model(dummy_input)
inference_model.load_weights("/kaggle/input/artifacts/best_model.weights.h5")


def preprocess_sequence(df_seq: pd.DataFrame):
    data = df_seq[feature_cols].interpolate().ffill().bfill().fillna(0).values
    scaled = scaler.transform(data)  # (seq_len_actual, total_features)
    # Pad/truncate
    padded = pad_sequences(
        [scaled],
        maxlen=pad_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )  # (1, pad_len, total_features)
    return tf.convert_to_tensor(padded, dtype=tf.float32)

def predict(sequence, demographics) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    sequence: polars DataFrame for a single sequence
    Returns: predicted gesture string
    """
    df_seq = sequence.to_pandas()
    x_tensor = preprocess_sequence(df_seq) # (1, pad_len, features)

    outputs = inference_model(x_tensor, training=False) # (1, num_classes)
    pred_idx = tf.argmax(outputs, axis=1).numpy()[0]

    return str(gesture_classes[pred_idx])
    return "Drink from bottle/cup"


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    print("local gateway running...")
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
    print("Done.")


# if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     print(pd.read_parquet("submission.parquet"))




