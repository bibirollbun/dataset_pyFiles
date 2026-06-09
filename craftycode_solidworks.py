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


import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_DIR =  Path('/kaggle/input/solidworks-ai-hackathon/train/train')
TEST_DIR = Path('/kaggle/input/solidworks-ai-hackathon/test/test')

trn_labels = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_labels.csv')
trn_labels.head()


import pandas as pd

labels = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_labels.csv')
labels.head()



import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32

TRAIN_DIR = Path('/kaggle/input/solidworks-ai-hackathon/train/train')

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

paths = labels['image_name'].apply(lambda x: str(TRAIN_DIR / x)).values
targets = labels[['bolt','locatingpin','nut','washer']].values.astype('float32')

ds = tf.data.Dataset.from_tensor_slices((paths, targets))
ds = ds.shuffle(1024).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False  

inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='relu')(x)

model = tf.keras.Model(inputs, outputs)





import tensorflow as tf

def exact_match_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_pred = tf.clip_by_value(y_pred, 0, 100)

    matches = tf.reduce_all(tf.equal(y_true, y_pred), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))



model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mae',
    metrics=['mae', exact_match_accuracy]
)



from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    labels,
    test_size=0.2,
    random_state=42
)


def make_dataset(df, shuffle=False):
    paths = df['image_name'].apply(lambda x: str(TRAIN_DIR / x)).values
    targets = df[['bolt','locatingpin','nut','washer']].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, shuffle=True)
val_ds   = make_dataset(val_df)



early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_exact_match_accuracy',
    mode='max',
    patience=7,
    restore_best_weights=True
)



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop]
)


base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mae'
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)



TEST_DIR = Path('/kaggle/input/solidworks-ai-hackathon/test/test')

test_images = sorted([p.name for p in TEST_DIR.glob('*.png')])


IMG_SIZE = 224
BATCH_SIZE = 32

def load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

test_paths = [str(TEST_DIR / name) for name in test_images]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



preds = model.predict(test_ds, verbose=1)



preds = np.round(preds).astype(int)
preds = np.clip(preds, 0, None)  


submission = pd.DataFrame({
    'image_name': test_images,
    'bolt': preds[:, 0],
    'locatingpin': preds[:, 1],
    'nut': preds[:, 2],
    'washer': preds[:, 3],
})


submission.head()


submission.to_csv('submission_mec(2).csv', index=False)




