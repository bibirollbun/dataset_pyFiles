import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Attention, Input, Add, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical




# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load Data
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=3)


SEQ_LEN = x_train.shape[1]
VOCAB_SIZE = len(np.unique(x_train))
NUM_CLASSES = 3




# Optimize data pipeline for GPU
def create_tf_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(len(x)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(x_train, y_train, batch_size=512)

# Define Model Using Functional API
inputs = Input(shape=(SEQ_LEN,))
x = Embedding(input_dim=VOCAB_SIZE, output_dim=256, input_length=SEQ_LEN, dtype='float32')(inputs)
x = Bidirectional(LSTM(512, return_sequences=True))(x)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)

# Transformer Block
attention = MultiHeadAttention(num_heads=4, key_dim=256)(x, x)
x = Add()([x, attention])
x = LayerNormalization()(x)
x = Dropout(0.3)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(256, activation='relu', dtype='float32')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)

# Compile Model with Label Smoothing
optimizer = Adam(learning_rate=0.001)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=optimizer,
    metrics=['accuracy']
)

# Learning Rate Scheduler (Cosine Decay)
lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=10000, alpha=0.0001)

# Train Model
model.fit(train_dataset, epochs=100, callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)])


# Predict
y_test_hat = model.predict(x_test)
y_test_hat = tf.argmax(y_test_hat, axis=1).numpy()
y_test_hat_pd = pd.DataFrame({'Id': list(range(len(x_test))), 'Predicted': y_test_hat.reshape(-1)})




assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 10000

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('trail6.csv', index=False)

