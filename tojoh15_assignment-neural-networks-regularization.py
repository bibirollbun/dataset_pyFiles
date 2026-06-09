import numpy as np
import pandas as pd
import tensorflow as tf


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')

print(x_train.shape, x_test.shape, y_train.shape)


# Initialize
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3072,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(20, activation='softmax'),
])

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    )

# Fit
model.fit(x_train, y_train, epochs=10)


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
assert len(y_test_hat_pd) == 10000

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)

