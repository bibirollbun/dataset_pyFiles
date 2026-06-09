import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping





# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')

print(x_train.shape, x_test.shape, y_train.shape)


# Define Model
model = tf.keras.models.Sequential([
    Flatten(input_shape=(64,)),

    Dense(512, activation='relu', kernel_regularizer=l2(0.0004)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1024, activation='relu', kernel_regularizer=l2(0.0004)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(512, activation='relu', kernel_regularizer=l2(0.0004)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu', kernel_regularizer=l2(0.0004)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation='relu', kernel_regularizer=l2(0.0004)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(20, activation='softmax')  # Multi-class classification
])

# Optimizer with fine-tuned learning rate
optimizer = SGD(learning_rate=0.007, momentum=0.9, nesterov=True)

# Compile the Model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'],
)

# Learning Rate Scheduler (Reduce LR faster)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)

# Train Model with larger batch size for better gradient estimates
history = model.fit(
    x_train, y_train,
    epochs=120,   # Train a little longer
    batch_size=128,  # More stable updates
    validation_split=0.1,
    callbacks=[lr_scheduler]
)


y_test_hat = model.predict(x_test)
y_test_hat = tf.argmax(y_test_hat, axis=1) # Convert from probabilities to classes
y_test_hat = y_test_hat.numpy()
y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(x_test))),
    'Predicted': y_test_hat.reshape(-1),
})


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 5000

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat28.csv', index=False)

