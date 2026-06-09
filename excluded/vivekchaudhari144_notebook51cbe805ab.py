# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


IMG_SIZE = 512
NUM_CLASSES = 28
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = '/kaggle/input/human-protein-atlas-image-classification/train/'  # Update if path is different
CSV_PATH = '/kaggle/input/human-protein-atlas-image-classification/train.csv'


def encode_labels(label_str):
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    for l in label_str.split():
        label[int(l)] = 1.0
    return label


def load_image_and_label(image_id, label_str, image_dir=TRAIN_DIR, size=(IMG_SIZE, IMG_SIZE)):
    channels = []
    for color in ['red', 'green', 'blue', 'yellow']:
        path = os.path.join(image_dir, f"{image_id}_{color}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        channels.append(img)
    image = np.stack(channels, axis=-1) / 255.0
    label = encode_labels(label_str)
    return image.astype(np.float32), label



def create_tf_dataset(df, batch_size=BATCH_SIZE, shuffle=True):
    def generator():
        for _, row in df.iterrows():
            yield load_image_and_label(row['Id'], row['Target'])

    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((IMG_SIZE, IMG_SIZE, 4), (NUM_CLASSES,))
    )
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Lambda

def build_model():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 4))
    x = Conv2D(3, (1, 1), padding='same')(input_tensor)  # Convert 4 -> 3 channels
    base = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_out = base(x)
    x = GlobalAveragePooling2D()(base_out)
    output = Dense(NUM_CLASSES, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output)
    return model


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    # Sample only 20% for quick training
    df_small = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    train_df, val_df = train_test_split(df_small, test_size=0.1, random_state=42)

    train_ds = create_tf_dataset(train_df)
    val_ds = create_tf_dataset(val_df, shuffle=False)

    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='AUC')])

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_AUC', save_best_only=True, mode='max', verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint])


# Load the best model
model.load_weights('best_model.h5')

# Evaluate on validation dataset
results = model.evaluate(val_ds)
print(f"\nðŸ“Š Validation Loss: {results[0]:.4f}")
print(f"âœ… Validation AUC: {results[1]:.4f}")



import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm



TEST_DIR = '/kaggle/input/human-protein-atlas-image-classification/test/'
MODEL_PATH = '/kaggle/working/best_model.h5'
IMG_SIZE = 512
NUM_CLASSES = 28


model = load_model(MODEL_PATH)


def load_test_image(image_id, image_dir=TEST_DIR, size=(IMG_SIZE, IMG_SIZE)):
    channels = []
    for color in ['red', 'green', 'blue', 'yellow']:
        path = os.path.join(image_dir, f"{image_id}_{color}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        img = cv2.resize(img, size)
        channels.append(img)
    image = np.stack(channels, axis=-1) / 255.0
    return image.astype(np.float32)




test_files = os.listdir(TEST_DIR)
image_ids = sorted(set(f.split('_')[0] for f in test_files if f.endswith('.png')))


results = []
for image_id in tqdm(image_ids):
    img = load_test_image(image_id)
    pred = model.predict(np.expand_dims(img, axis=0))[0]  # shape (28,)
    pred_labels = [str(i) for i, p in enumerate(pred) if p > 0.5]
    results.append({'Id': image_id, 'Predicted': ' '.join(pred_labels)})

# --- Save as submission.csv ---
submission_df = pd.DataFrame(results)
submission_df.to_csv('submission.csv', index=False)

print("\nâœ… Done! Predictions saved in submission.csv")
print(submission_df.head())





