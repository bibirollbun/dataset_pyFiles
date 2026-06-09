import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


target = "BeatsPerMinute"

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv").drop("id", axis=1)
y = train.pop(target)
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test_id = pd.DataFrame(test.pop("id"))
test.head()


X = train


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)


import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
import tensorflow.keras.regularizers as regularizers


normalizer = layers.Normalization()
normalizer.adapt(X_train.values)


inp = layers.Input((X_train.shape[1], ))
norm = normalizer(inp)

x1 = layers.Dense(32, activation="mish")(norm)
x1 = layers.Dropout(0.2)(x1)

x2 = layers.Concatenate()([norm, x1])
x2 = layers.Dense(32, activation="mish")(x2)
x2 = layers.Dropout(0.2)(x2)

x3 = layers.Concatenate()([norm, x1, x2])
x3 = layers.Dense(32, activation="mish")(x3)
x3 = layers.Dropout(0.2)(x3)

x4 = layers.Concatenate()([norm, x1, x2, x3])
x4 = layers.Dense(32, activation="mish")(x4)
x4 = layers.Dropout(0.2)(x4)

out = layers.Dense(1)(x4)
model = models.Model(inputs=inp, outputs=out)
model.summary()


# model = models.Sequential([
#     layers.Input((X_train.shape[1], )),
#     normalizer,
#     layers.Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.1)),
#     layers.Dropout(0.1),
#     layers.Dense(64, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.1)),
#     layers.Dropout(0.1),
#     layers.Dense(32, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.1)),
#     layers.Dropout(0.1),
#     layers.Dense(1)
# ])
# model.summary()
model.compile(optimizers.AdamW(0.001), loss=losses.MeanSquaredError(name="mse"), metrics=[metrics.RootMeanSquaredError(name="rmse")])


weights_path = "chkpt.weights.h5"
cb = [
    callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_steps_per_second=False,
        update_freq='epoch',
    ),
    callbacks.ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    )
]

history = model.fit(
    X_train,
    y_train,
    batch_size=1024,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=cb
)


model.load_weights(weights_path)


import matplotlib.pyplot as plt


plt.semilogy(history.history["loss"])
plt.semilogy(history.history["val_loss"])
plt.show()

plt.semilogy(history.history["rmse"])
plt.semilogy(history.history["val_rmse"])



test_id[target] = model.predict(test[X.columns], batch_size=512).squeeze()
test_id


test_id.to_csv("submission.csv", index=False)


!head submission.csv

