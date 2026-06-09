import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add, 
    Bidirectional, LSTM, Dropout, Dense, GlobalAveragePooling1D,
    GaussianNoise, MultiHeadAttention, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#score 0.63 - 0.65, good luck for you. Upvote for me if you like it 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Bidirectional, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import polars as pl
import kaggle_evaluation.cmi_inference_server
print("Imports loaded")
from tensorflow.keras.layers import Input, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Custom layer definitions
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], input_shape[-1]), 
                                  initializer='glorot_uniform', 
                                  trainable=True)
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-1]), 
                                  initializer='glorot_uniform', 
                                  trainable=True)
        self.Wv = self.add_weight(shape=(input_shape[-1], input_shape[-1]), 
                                  initializer='glorot_uniform', 
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        Q = tf.matmul(inputs, self.Wq)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, V)

# Residual blocks
def residual_block_tof(x, filters, kernel_size=3, dilation_rate=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return Activation('relu')(x)

def residual_block_imu(x, filters):
    shortcut = Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    x = Conv1D(filters, 3, padding='same', activation='relu')(x)
    x = Conv1D(filters, 3, padding='same')(x)
    return Activation('relu')(Add()([shortcut, x]))

# Data preparation
def prepare_data(df):
    # Encode labels
    label_encoder = LabelEncoder()
    df['gesture'] = label_encoder.fit_transform(df['gesture'].astype(str))
    gesture_classes = label_encoder.classes_
    
    # Define feature sets
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    
    excluded_cols = {'gesture', 'sequence_type', 'behavior', 'orientation', 
                    'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    excluded_cols.update([col for col in df.columns if col.startswith(('thm_', 'tof_'))])
    imu_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Initialize scalers
    tof_scaler = StandardScaler()
    tof_scaler.fit(df[tof_cols].dropna())
    
    imu_scaler = StandardScaler()
    imu_scaler.fit(df[imu_cols].dropna())
    np.save("tof_scaler_mean.npy", tof_scaler.mean_)
    np.save("tof_scaler_scale.npy", tof_scaler.scale_)
    np.save("imu_scaler_mean.npy", imu_scaler.mean_)
    np.save("imu_scaler_scale.npy", imu_scaler.scale_)
    # Process sequences
    sequences = df.groupby('sequence_id')
    X_tof, X_imu, y, seq_ids = [], [], [], []
    tof_lengths, imu_lengths = [], []

    for seq_id, seq_df in sequences:
        # Process ToF data
        tof_data = seq_df[tof_cols].ffill().bfill().fillna(0)
        X_tof.append(tof_scaler.transform(tof_data))
        tof_lengths.append(len(tof_data))
        
        # Process IMU data
        imu_data = seq_df[imu_cols].ffill().bfill().fillna(0)
        X_imu.append(imu_scaler.transform(imu_data))
        imu_lengths.append(len(imu_data))
        
        # Store labels and IDs
        y.append(seq_df['gesture'].iloc[0])
        seq_ids.append(seq_id)
    
    # Pad sequences
    tof_pad_len = int(np.percentile(tof_lengths, 90))
    imu_pad_len = int(np.percentile(imu_lengths, 90))
    
    X_tof = pad_sequences(X_tof, maxlen=tof_pad_len, dtype='float32', padding='post', truncating='post')
    X_imu = pad_sequences(X_imu, maxlen=imu_pad_len, dtype='float32', padding='post', truncating='post')
    
    # Convert labels
    y = to_categorical(y, num_classes=len(gesture_classes))
    
    return X_tof, X_imu, y, tof_pad_len, imu_pad_len, gesture_classes

# Model architecture
def create_multimodal_model(tof_input_shape, imu_input_shape, num_classes):
    # ToF branch
    tof_input = Input(shape=tof_input_shape, name='tof_input')
    x_tof = Conv1D(64, 3, padding='same', activation='relu')(tof_input)
    x_tof = residual_block_tof(x_tof, 64)
    x_tof = residual_block_tof(x_tof, 64, dilation_rate=2)
    x_tof = residual_block_tof(x_tof, 64, dilation_rate=4)
    x_tof = Bidirectional(LSTM(64, return_sequences=True))(x_tof)
    x_tof = SelfAttention()(x_tof)
    tof_branch = GlobalAveragePooling1D()(x_tof)
    
    # IMU branch
    imu_input = Input(shape=imu_input_shape, name='imu_input')
    x_imu = GaussianNoise(0.1)(imu_input)
    x_imu = Conv1D(64, 3, padding='same', activation='relu')(x_imu)
    x_imu = residual_block_imu(x_imu, 64)
    x_imu = MaxPooling1D(2)(x_imu)
    x_imu = residual_block_imu(x_imu, 128)
    x_imu = MaxPooling1D(2)(x_imu)
    x_imu = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x_imu)
    imu_branch = GlobalAveragePooling1D()(x_imu)
    
    # Combined model
    combined = Concatenate()([tof_branch, imu_branch])
    combined = Dense(384, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(192, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    output = Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs=[tof_input, imu_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main workflow



df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
X_tof, X_imu, y, tof_pad_len, imu_pad_len, gesture_classes = prepare_data(df)
# Train-validation split
(X_tof_train, X_tof_val,X_imu_train, X_imu_val, 
y_train, y_val) = train_test_split(X_tof, X_imu, y, test_size=0.2, random_state=42)
    
    # Create model
model = create_multimodal_model(
        tof_input_shape=(tof_pad_len, X_tof_train.shape[2]),
        imu_input_shape=(imu_pad_len, X_imu_train.shape[2]),
        num_classes=y_train.shape[1]
)
    
    # Train model
callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]
    
history = model.fit(
        [X_tof_train, X_imu_train],
        y_train,
        epochs=60,
        batch_size=64,
        validation_data=([X_tof_val, X_imu_val], y_val),
        callbacks=callbacks
    )
    
    # Save model and configuration
model.save("multimodal_model_2.h5")
np.save("tof_pad_len.npy", tof_pad_len)
np.save("imu_pad_len.npy", imu_pad_len)
np.save("gesture_classes.npy", gesture_classes)




from cmi_2025_metric_copy_for_import import CompetitionMetric
import numpy as np
import pandas as pd

# Get predicted labels for the validation set
print("Predicting on validation set...")
y_val_pred_probs = model.predict([X_tof_val, X_imu_val], verbose=0)  # Dual input
y_val_pred = np.argmax(y_val_pred_probs, axis=1)
y_val_true = np.argmax(y_val, axis=1)
val_pred_labels = pd.Series(y_val_pred).map(lambda i: gesture_classes[i])
val_true_labels = pd.Series(y_val_true).map(lambda i: gesture_classes[i])

# Build DataFrames for the metric
val_submission = pd.DataFrame({'gesture': val_pred_labels})
val_solution = pd.DataFrame({'gesture': val_true_labels})

# Run competition metric
metric = CompetitionMetric()
score = metric.calculate_hierarchical_f1(val_solution, val_submission)
print(f"Estimated leaderboard (val) score: {score:.4f}")


import polars as pl
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Convert to pandas
    df_seq = sequence.to_pandas()
    
    # Load configuration
    gesture_classes = np.load("/kaggle/working/gesture_classes.npy", allow_pickle=True)
    tof_pad_len = int(np.load("/kaggle/working/tof_pad_len.npy"))
    imu_pad_len = int(np.load("/kaggle/working/imu_pad_len.npy"))
    
    # Define feature columns
    tof_cols = [col for col in df_seq.columns if col.startswith('tof_')]
    
    excluded_cols = {'sequence_type', 'behavior', 'orientation', 
                    'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    excluded_cols.update([col for col in df_seq.columns if col.startswith('thm_')])
    imu_cols = [col for col in df_seq.columns if col not in excluded_cols and not col.startswith('tof_')]
    
    # Load scaler parameters
    tof_mean = np.load("/kaggle/working/tof_scaler_mean.npy")
    tof_scale = np.load("/kaggle/working/tof_scaler_scale.npy")
    imu_mean = np.load("/kaggle/working/imu_scaler_mean.npy")
    imu_scale = np.load("/kaggle/working/imu_scaler_scale.npy")
    
    # Handle zero-variance features
    tof_scale[tof_scale == 0] = 1.0
    imu_scale[imu_scale == 0] = 1.0
    
    # Preprocess ToF data
    tof_data = df_seq[tof_cols].copy().ffill().bfill().fillna(0)
    tof_processed = (tof_data - tof_mean) / tof_scale
    
    # Preprocess IMU data
    imu_data = df_seq[imu_cols].copy().ffill().bfill().fillna(0)
    imu_processed = (imu_data - imu_mean) / imu_scale
    
    # Pad sequences
    tof_padded = pad_sequences(
        [tof_processed.values], 
        maxlen=tof_pad_len, 
        dtype='float32', 
        padding='post', 
        truncating='post'
    )
    
    imu_padded = pad_sequences(
        [imu_processed.values], 
        maxlen=imu_pad_len, 
        dtype='float32', 
        padding='post', 
        truncating='post'
    )
    
    # Load model and predict
    # model = load_model("/kaggle/working/multimodal_model.h5")
    prediction = model.predict([tof_padded, imu_padded], verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]
    
    return gesture_classes[predicted_index]


# Launch inference server
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

