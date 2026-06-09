import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from sklearn.model_selection import train_test_split
import os


print("GPU доступен:", tf.config.list_physical_devices('GPU'))



IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 64

DIR = Path('/kaggle/input/mushroom-multiclass-classification-1')
DATASET_DIR = Path('/kaggle/input/mushroom-multiclass-classification-1/dataset/dataset')
MAIN_DIR = Path('/kaggle/working') 
MAIN_DIR.mkdir(exist_ok=True)


class_mapping = {
    'amanita': 0,
    'boletus': 1,
    'chantelle': 2,
    'deterrimus': 3,
    'rufus': 4,
    'torminosus': 5,
    'aurantiacum': 6,
    'procera': 7,
    'involutus': 8,
    'russula': 9
}



train = pd.read_csv(DIR / 'train.csv', dtype={'Image': str})
train['Image'] = str(DATASET_DIR) + '/' + train['Image'].astype(str) + '.jpg'

# Разделение данных
train_val, test_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train['Mushroom']
)

train_df, val_df = train_test_split(
    train_val,
    test_size=0.2,
    random_state=42,
    stratify=train_val['Mushroom']
)



def load_image(image_path, label):
    # label = tf.strings.as_string(label, width=5, fill='0')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    image = tf.image.resize(image, size=IMAGE_SIZE[:2])
    image = tf.cast(image, tf.float32)
    return image, label

# Создание tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_df['Image'].values, train_df['Mushroom'].values))
train_ds = train_ds.shuffle(buffer_size=len(train_ds), reshuffle_each_iteration=True)
train_ds = train_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['Image'].values, val_df['Mushroom'].values))
val_ds = val_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

test_ds = tf.data.Dataset.from_tensor_slices((test_df['Image'].values, test_df['Mushroom'].values))
test_ds = test_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)



from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    min_delta=0.000001,
    cooldown=0,
    min_lr=0.0000001
)

checkpointer = ModelCheckpoint(
    MAIN_DIR / 'best_model.weights.h5',
    save_best_only=True,
    verbose=1,
    save_weights_only=True
)

stopper = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    min_delta=0.0001,
    verbose=1
)

logger = CSVLogger(MAIN_DIR / "training_log.csv", separator=';', append=True)

callbacks = [reduce_learning_rate, checkpointer, stopper, logger]



conv_base = tf.keras.applications.EfficientNetB0(
    input_shape=IMAGE_SIZE,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

inputs = tf.keras.Input(shape=IMAGE_SIZE)
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = conv_base(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_mapping), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)



history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=callbacks
)



model.trainable = True
for layer in model.layers[2].layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=callbacks
)



model.save(MAIN_DIR / 'model.keras')



test_sub = pd.read_csv(DIR / 'test.csv', dtype={'Image': str})
test_sub['Image_path'] = str(DATASET_DIR) + '/' + test_sub['Image'].astype(str) + '.jpg'

def load_image_for_prediction(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, size=IMAGE_SIZE[:2])
    image = tf.cast(image, tf.float32)
    return image

pred_ds = tf.data.Dataset.from_tensor_slices(test_sub['Image_path'].values)
pred_ds = pred_ds.map(load_image_for_prediction, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

predictions = model.predict(pred_ds)
predicted_classes = predictions.argmax(axis=1)

results = pd.DataFrame({
    'Id': range(len(test_sub)),
    'Predicted': predicted_classes
})

results.to_csv(MAIN_DIR / 'submission.csv', index=False)
print("Файл submission.csv успешно создан!")


