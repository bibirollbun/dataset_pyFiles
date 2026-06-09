# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
os.makedirs("/kaggle/working/models", exist_ok=True)
os.listdir('/kaggle/working')



flat_data_a = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy')
flat_vel_a = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy')

curve_data_a = np.load('/kaggle/input/waveform-inversion/train_samples/CurveVel_A/data/data2.npy')
curve_vel_a = np.load('/kaggle/input/waveform-inversion/train_samples/CurveVel_A/model/model2.npy')

style_data_a = np.load('/kaggle/input/waveform-inversion/train_samples/Style_A/data/data2.npy')
style_vel_a = np.load('/kaggle/input/waveform-inversion/train_samples/Style_B/model/model2.npy')



print("Loaded")


#500 échantillons
print(flat_data_a.shape)#70x70
print(flat_vel_a.shape)#5x1000x70

print(curve_data_a.shape)#70x70
print(curve_vel_a.shape)#5x1000x70

print(style_data_a.shape)#70x70
print(style_vel_a.shape)#5x1000x70


import matplotlib.pyplot as plt

sample = 400

fig, axs = plt.subplots(1, 3, figsize=(12, 4)) 

def plot_velocity(ax, velocity_tensor, title, sample=400):
    img = ax.imshow(velocity_tensor[sample, 0, :, :], cmap='jet')
    ax.set_xticks(range(0, 70, 10))
    ax.set_xticklabels(range(0, 700, 100))
    ax.set_yticks(range(0, 70, 10))
    ax.set_yticklabels(range(0, 700, 100))
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_xlabel('Offset (m)', fontsize=10) #=largeur, != decalage
    ax.set_title(title, fontsize=11)

    clb = plt.colorbar(img, ax=ax, shrink=0.7)
    clb.ax.set_title('km/s', fontsize=8)

plot_velocity(axs[0], flat_vel_a, "Flat Velocity A", sample)
plot_velocity(axs[1], curve_vel_a, "Curve Velocity A", sample)
plot_velocity(axs[2], style_vel_a, "Style Velocity A", sample)

plt.tight_layout()
plt.show()



source=0
receiver=35

waveform = style_data_a[sample, source, :, receiver]
plt.figure(figsize=(12,4))
plt.plot(waveform)
plt.title(f"Forme d'onde captée - Echantillon 0, Source {source}, Récepteur {receiver}")
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


sample_selected=128


X = np.concatenate([flat_data_a[:sample_selected],curve_data_a[:sample_selected],style_data_a[:sample_selected]])
y = np.concatenate([flat_vel_a[:sample_selected],curve_vel_a[:sample_selected],style_vel_a[:sample_selected]])

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

print(X.shape,y.shape)


print(X.min(), X.max())
print(y.min(), y.max())


X_min = np.min(X)
X_max = np.max(X)
y_min = np.min(y)
y_max = np.max(y)

scale_X = (X_max - X_min)
scale_y = (y_max - y_min)


X_normalized = (X - X_min) / scale_X
y_normalized = (y - y_min) / scale_y

print("X min/max/mean/std après normalisation:", X_normalized.min(), X_normalized.max(), X_normalized.mean(), X_normalized.std())
print("y min/max/mean/std après normalisation:", y_normalized.min(), y_normalized.max(), y_normalized.mean(), y_normalized.std())


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]

y_train = y_train.squeeze()
y_val = y_val.squeeze()

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape) 
print("y_val shape:", y_val.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam

def build_ConvLSTM2D():
    model = Sequential([
        Input(shape=(5, 1000, 70, 1)),

        ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", return_sequences=False, activation='relu'),
        BatchNormalization(),

        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(70 * 70, activation='linear'),
        Reshape((70, 70))
    ])
    return model

def mae(y_true, y_pred):
    y_true_phys = y_true * scale_y + y_min
    y_pred_phys = y_pred * scale_y + y_min
    return tf.reduce_mean(tf.abs(y_true_phys - y_pred_phys))

model=build_ConvLSTM2D()
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer , loss='mae', metrics=["mae"])
model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,             
    restore_best_weights=True  
)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16,callbacks=[early_stop], verbose=1)


mae = model.evaluate(X_val, y_val)
print(f"MAE sur le jeu de validation : {mae}")


#model.save_weights("/kaggle/working/models/W1_(350).weights.h5")


#model.save_weights("/kaggle/working/models/W2_(266).weights.h5")


model.save_weights("/kaggle/working/models/W3_(random).weights.h5")


plt.figure(figsize=(10, 5))

plt.plot(history.history['mae'], label='MAE (train)')
plt.plot(history.history['val_mae'], label='MAE (val)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Évolution de la MAE pendant l’entraînement')
plt.legend()
plt.grid(True)
plt.show()



index = 12

pred = model.predict(X_val[:index])[0] * scale_y + y_min
true = y_val[index] * scale_y + y_min

# 4. Visualisation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Vitesse prédite")
plt.imshow(pred, cmap='RdYlGn')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Vitesse réelle")
plt.imshow(true, cmap='RdYlGn')
plt.colorbar()
plt.show()

