import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

np.random.seed(42)
tf.random.set_seed(42)

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
WAVELET_SCALE = 1e-3



def get_data_paths(data_dir):
    input_files = []
    output_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                if 'data' in file or 'seis' in file:
                    input_files.append(full_path)
                    output_file = full_path.replace('data', 'model').replace('seis', 'vel')
                    if os.path.exists(output_file):
                        output_files.append(output_file)
    return input_files, output_files

def load_and_preprocess_data(input_files, output_files):
    X, y = [], []
    for inp, out in tqdm(zip(input_files, output_files), total=len(input_files)):
        try:
            seismic = np.load(inp)
            velocity = np.load(out)
            seismic = np.mean(seismic[0, :, :, :], axis=0)
            velocity = velocity[0, 0, :, :]
            seismic = (seismic - np.mean(seismic)) / (np.std(seismic) + 1e-8) * WAVELET_SCALE
            velocity = (velocity - 1500) / 4500
            X.append(seismic)
            y.append(velocity)
        except Exception as e:
            print(f"Error loading {inp} or {out}: {str(e)}")
    return np.array(X), np.array(y)

train_data_dir = '/kaggle/input/waveform-inversion/train_samples'
test_data_dir = '/kaggle/input/waveform-inversion/test'

all_inputs, all_outputs = get_data_paths(train_data_dir)

X, y = load_and_preprocess_data(all_inputs, all_outputs)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def build_pinn_model(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(np.prod(output_shape), activation='relu')(x)
    outputs = layers.Reshape(output_shape)(x)
    model = models.Model(inputs, outputs)
    return model

input_shape = X_train.shape[1:]
output_shape = y_train.shape[1:]
model = build_pinn_model(input_shape, output_shape)
model.summary()



def physics_loss(y_true, y_pred):
    data_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    y_pred_4d = tf.expand_dims(y_pred, axis=-1)
    dy_dx = tf.image.image_gradients(y_pred_4d)
    physics_constraint = tf.reduce_mean(tf.abs(dy_dx[0]) + tf.reduce_mean(tf.abs(dy_dx[1])))
    return data_loss + 0.1 * physics_constraint

model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
              loss=physics_loss,
              metrics=['mae'])



callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=5),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='val')
plt.legend()
plt.title('MAE')
plt.show()



def generate_submission(model, test_dir):
    test_files = [f for f in Path(test_dir).rglob('*.npy')]
    x_cols = [f"x_{i}" for i in range(1, 70, 2)]
    header = ["oid_ypos"] + x_cols
    submission = [",".join(header)]
    
    for file in tqdm(test_files):
        oid = file.stem
        data = np.load(file)

        if data.ndim == 4:
            data = data[0]
        if data.ndim == 3:
            data = np.mean(data, axis=0)

        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * WAVELET_SCALE

        pred = model.predict(np.expand_dims(data, 0))[0]
        pred = pred * 4500 + 1500  

        for y_pos in range(70):
            odd_cols = pred[y_pos, 1::2]

            line = f"{oid}_y_{y_pos}," + ",".join([f"{v:.1f}" for v in odd_cols])
            submission.append(line)

    with open('submission.csv', 'w') as f:
        f.write("\n".join(submission))

generate_submission(model, test_data_dir)

