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


import tensorflow as tf

GCS_PATH = '/kaggle/input/flower-classification-with-tpus/tfrecords-jpeg-192x192'
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')


IMAGE_SIZE = [192, 192]
AUTO = tf.data.AUTOTUNE

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def read_tfrecord(example, labeled):
    if labeled:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.string)
        }
    else:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string)
        }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    idnum = example['id']

    if labeled:
        label = tf.cast(example['class'], tf.int32)
        return image, label
    else:
        return image, idnum


def load_dataset(filenames, labeled=True):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTO)
    return dataset

def get_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda x: read_tfrecord(x, labeled),
        num_parallel_calls=AUTO
    )
    return dataset


strategy = tf.distribute.get_strategy()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

train_dataset = (
    load_dataset(TRAINING_FILENAMES, labeled=True)
    .shuffle(2048)
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    load_dataset(VALIDATION_FILENAMES, labeled=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

with strategy.scope():
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(192,192,3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(104, activation='softmax')  
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4, weight_decay=5e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0000001)

history = model.fit(
    train_dataset,
    epochs=35,
    steps_per_epoch=100,
    validation_data=valid_dataset,
    validation_steps=20,
    callbacks=[reduce_lr]
)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()


test_ds = get_dataset(TEST_FILENAMES, labeled=False, ordered=True)

test_ids = []
for _, idnum in test_ds.as_numpy_iterator():
    test_ids.append(idnum.decode('utf-8'))

predict_ds = test_ds.map(lambda image, idnum: image, num_parallel_calls=AUTO)
predict_ds = predict_ds.batch(BATCH_SIZE).prefetch(AUTO)

preds = model.predict(predict_ds, verbose=1)
pred_labels = np.argmax(preds, axis=-1)

import pandas as pd
submission = pd.DataFrame({'id': test_ids, 'label': pred_labels})
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()


import os
print("submission.csv exists:", os.path.exists('submission.csv'))


!ls -lh /kaggle/working/

