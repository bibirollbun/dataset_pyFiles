import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Disable XLA JIT to avoid CTC compatibility issues
tf.config.optimizer.set_jit(False)


class Config:
    SEQ_LENGTH = 384       
    LANDMARK_SIZE = 84     
    BATCH_SIZE = 64
    EPOCHS = 15
    CHAR_MAX_LEN = 32      
    RATE = 0.2             
    EMBED_DIM = 256        
    
config = Config()


def get_landmark_cols():
    cols = []
    for coord in ['x', 'y']:
        for hand in ['left_hand', 'right_hand']:
            for i in range(21):
                cols.append(f'{coord}_{hand}_{i}')
    return cols


LANDMARK_COLS = get_landmark_cols()


ROOT_PATH = "/kaggle/input/asl-fingerspelling"
train_df = pd.read_csv(f"{ROOT_PATH}/train.csv")
char_to_num = keras.layers.StringLookup(vocabulary=list("abcdefghijklmnopqrstuvwxyz' !"), oov_token="")


def load_and_preprocess_sequence(file_id, sequence_id):
    """Load and preprocess individual sequence"""
    file_path = f"{ROOT_PATH}/train_landmarks/{file_id}.parquet"
    df = pd.read_parquet(file_path)
    
    # Filter by sequence_id
    seq_df = df[df.index == sequence_id]
    
    # Handle missing sequences
    if seq_df.empty:
        return np.zeros((config.SEQ_LENGTH, config.LANDMARK_SIZE))
    
    # Extract relevant landmarks
    seq_data = seq_df[LANDMARK_COLS].fillna(0).values.astype(np.float32)
    n_frames = seq_data.shape[0]
    
    # Pad or truncate to fixed length
    if n_frames < config.SEQ_LENGTH:
        pad_len = config.SEQ_LENGTH - n_frames
        seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
    else:
        seq_data = seq_data[:config.SEQ_LENGTH]
    
    return seq_data


# Preprocess subset of data
subset_df = train_df.head(1000).copy()
X = []
y = []

for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Loading and preprocess sequences"):
    seq_data = load_and_preprocess_sequence(row['file_id'], row['sequence_id'])
    X.append(seq_data)
    y.append(row['phrase'])

X = np.array(X)
y = np.array(y)

# Encode labels
y_encoded = char_to_num(tf.strings.unicode_split(y, input_encoding="UTF-8"))
# Convert RaggedTensor to padded tensor
y_padded = y_encoded.to_tensor(default_value=0)
# Truncate sequences longer than max length
y_padded = y_padded[:, :config.CHAR_MAX_LEN]
# Pad to fixed length
pad_len = config.CHAR_MAX_LEN - tf.shape(y_padded)[1]
y_padded = tf.pad(y_padded, [[0, 0], [0, pad_len]], constant_values=0)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y_padded.numpy(), test_size=0.1, random_state=42)


def build_model():
    inputs = keras.Input(shape=(config.SEQ_LENGTH, config.LANDMARK_SIZE))
    
    # Encoder
    x = layers.Dense(config.EMBED_DIM, activation="relu")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(config.RATE)(x)
    
    # Transformer Blocks
    for _ in range(2):
        # Self-attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        attn = layers.Dropout(config.RATE)(attn)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward network
        ffn = layers.Dense(4 * config.EMBED_DIM, activation="relu")(x)
        ffn = layers.Dense(config.EMBED_DIM)(ffn)  # Maintain dimension
        ffn = layers.Dropout(config.RATE)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
    
    # CTC requires time-distributed outputs
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(char_to_num.get_vocabulary()) + 1)(x)  # +1 for CTC blank
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()


def ctc_loss(y_true, y_pred):
    # Calculate label lengths (number of non-zero characters)
    mask = tf.not_equal(y_true, 0)
    label_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
    
    # Calculate input lengths (all sequences are full length)
    batch_size = tf.shape(y_true)[0]
    input_length = tf.fill([batch_size], tf.shape(y_pred)[1])
    
    # Use TensorFlow's native CTC loss
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=False
    )
    return tf.reduce_mean(loss)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=ctc_loss,
    metrics=[]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


def decode_pred(pred):
    """Convert model output to text"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(
        pred, 
        input_length=input_len, 
        greedy=True
    )[0][0]
    texts = tf.strings.reduce_join(char_to_num(results), axis=-1)
    return [str(t.numpy(), "utf-8") for t in texts]


# Check if test data exists
TEST_CSV_PATH = f"{ROOT_PATH}/test.csv"

if os.path.exists(TEST_CSV_PATH):
    # Load test data
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_test = []

    for _, row in test_df.iterrows():
        seq_data = load_and_preprocess_sequence(row['file_id'], row['sequence_id'])
        X_test.append(seq_data)

    X_test = np.array(X_test)

    # Predict
    predictions = model.predict(X_test)
    decoded_preds = decode_pred(predictions)

    # Create submission
    submission = pd.DataFrame({
        "sequence_id": test_df.sequence_id,
        "phrase": decoded_preds
    })
    submission.to_csv("submission.csv", index=False)
    print("Submission file created with test predictions!")
else:
    # Create a sample submission file for local testing
    print("Test data not found. Creating sample submission file.")
    sample_submission = pd.DataFrame({
        "sequence_id": [1, 2, 3],
        "phrase": ["hello", "world", "asl"]
    })
    sample_submission.to_csv("submission.csv", index=False)
    print("Sample submission file created for testing.")

