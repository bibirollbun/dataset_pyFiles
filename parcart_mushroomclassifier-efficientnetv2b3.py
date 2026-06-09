from pathlib import Path
import pandas as pd

import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

from datetime import datetime

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}") 

try:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    print(f"Количество устройств: {strategy.num_replicas_in_sync}")
except Exception as e:
    print(f"Ошибка инициализации GPU: {e}")
    strategy = tf.distribute.get_strategy()


import os
# Dataset files
for files in os.listdir('/kaggle/input/mushroom-multiclass-classification'):
    print(files)


DATASET_DIR = Path('/kaggle/input/mushroom-multiclass-classification')
DATASET_IMAGE_DIR = DATASET_DIR.joinpath("dataset").joinpath("dataset")
BATCH_SIZE = 64
IMAGE_SIZE = (300,300,3)
MAIN_DIR = Path('/result')
MAIN_DIR.mkdir(parents=True, exist_ok=True)

label_dict = {
    0: "amanita",
    1: "boletus",
    2:"chantelle",
    3:"deterrimus",
    4: "rufus",
    5: "torminosus",
    6: "aurantiacum",
    7: "procera",
    8: "involutus",
    9: "russula",
}


test_df = pd.read_csv(DATASET_DIR.joinpath("test.csv"))
train_df = pd.read_csv(DATASET_DIR.joinpath("train.csv"))


train_df


def set_image_path_by_id(image_id):
    return DATASET_IMAGE_DIR.joinpath(f"{image_id:05d}.jpg").as_posix()

train_df["Image"] = train_df["Image"].apply(set_image_path_by_id)
test_df["Image"] = test_df["Image"].apply(set_image_path_by_id)


def get_label(label_id):
    return label_dict[label_id]

train_df.rename(columns={"Mushroom": "Label_i"}, inplace=True)
train_df["Label"] = train_df["Label_i"].apply(get_label)
train_df['Label'].value_counts().plot(kind='bar', figsize=(10,6))


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, train_size=0.8, stratify=train_df['Label_i'], random_state=1337)


from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler(sampling_strategy='auto')
train_df, _ = ROS.fit_resample(train_df, train_df['Label'])
train_df['Label'].value_counts().plot.bar(figsize=((12,5)), title='Resampled classes')


train_df


test_df


train_df.to_csv(MAIN_DIR.joinpath('train_mash.csv'), index=False)
val_df.to_csv(MAIN_DIR.joinpath('val_mash.csv'), index=False)
test_df.to_csv(MAIN_DIR.joinpath('test_mash.csv'), index=False)


train_df = pd.read_csv(MAIN_DIR.joinpath('train_mash.csv'))
val_df = pd.read_csv(MAIN_DIR.joinpath('val_mash.csv'))
test_df = pd.read_csv(MAIN_DIR.joinpath('test_mash.csv'))


def load_image(image_path, label_i):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)

    image = tf.image.resize(image, size=IMAGE_SIZE[:2])
    image = tf.cast(image, tf.float32)
    return image, label_i

train_ds = tf.data.Dataset.from_tensor_slices((train_df["Image"], train_df["Label_i"]))
train_ds = train_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

val_ds = tf.data.Dataset.from_tensor_slices((val_df["Image"], val_df["Label_i"]))
val_ds = val_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

test_ds = tf.data.Dataset.from_tensor_slices((test_df["Image"], None))
test_ds = test_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)


im_batch_plus_lbl, lbl_batch = next(iter(test_ds))
im_batch = im_batch_plus_lbl

fig = plt.figure(figsize=(35, 35))
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(4, 8),  
                 axes_pad=0.3,  
                 )
for i, grid in enumerate(zip(grid, im_batch)):
    if i == 16:
        break
    ax, im = grid
    ax.imshow(tf.cast(im,tf.int32))

plt.show()


def show_results(history):
    accuracy = history.history['sparse_categorical_accuracy']
    val_accuracy = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(accuracy)+1)
    f1_score = history.history['f1_score']
    val_f1_score = history.history['val_f1_score']

    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.legend()
    plt.title("Training and validation accuracy")
    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.legend()
    plt.title("Training and validation loss")
    plt.figure()

    plt.plot(epochs, f1_score, "go", label="Training F1-score")
    plt.plot(epochs, val_f1_score, "g", label="Validation F1-score")
    plt.title("Training and Validation F1-Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.legend()

    plt.tight_layout()

    plt.show()


##CALLBACKS
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

MAIN_DIR.joinpath('cp').mkdir(parents=True, exist_ok=True)
MAIN_DIR.joinpath('logs').mkdir(parents=True, exist_ok=True)

reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy',  
    factor=0.5,  
    patience=10,  
    min_lr=1e-5,
    mode='max',
    verbose=1
)

chk_path = MAIN_DIR.joinpath('cp', 'ckpt_{epoch:02d}.weights.h5')

checkpointer = tf.keras.callbacks.ModelCheckpoint(chk_path, save_best_only=True, verbose=1, save_weights_only=True, mode='min', monitor='val_loss')

stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  
    min_delta=0.001,  
    mode='min',
    restore_best_weights=True
)

logger = CSVLogger(MAIN_DIR.joinpath("logs", f"training_log.csv"), separator=';', append=True)

logs = "logs/" + datetime.now().strftime("thresh_%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch='200,205')

callbacks = [reduce_learning_rate,
             checkpointer,
             stopper,
             logger,
             ]

print('Done')


from keras.api import optimizers
from tensorflow.keras import regularizers

from tensorflow.keras.metrics import F1Score

class SparseF1Score(F1Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse y_true to one-hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(label_dict))
        super().update_state(y_true_one_hot, y_pred, sample_weight)


with strategy.scope():
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': False,
        'remapping': False
    })
    
    conv_base = tf.keras.applications.EfficientNetV2B3(
        include_top=False,
        weights='imagenet',
        input_shape=IMAGE_SIZE,
        pooling='avg'
    )
    
    
    conv_base.trainable = True
    
    for layer in conv_base.layers[:int(len(conv_base.layers)*0.75)]:
        layer.trainable = False
    
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.GaussianNoise(0.02), 
    ], name='augmentation')
    
    input = tf.keras.layers.Input(IMAGE_SIZE)
    x = data_augmentation(input)
    
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = conv_base(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    outputs = tf.keras.layers.Dense(len(label_dict), activation='softmax')(x)
    
    model = tf.keras.Model(input, outputs, name='MushroomClassifier_V2B3')
    
    model.summary()
    
    
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=1000,
    #     end_learning_rate=1e-6
    # )
    
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics= [tf.keras.metrics.SparseCategoricalAccuracy(), SparseF1Score(average='macro')]
                 # metrics=['accuracy', SparseF1Score(average='macro', threshold=None, name='f1_score')])
                 )



def configure_dataset(dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options)

train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)


history = model.fit(
     train_ds,
     epochs=100,
     validation_data=val_ds,
     callbacks=callbacks)


show_results(history)


result_test = model.predict(test_ds)
predicted_classes = tf.argmax(result_test, axis=1).numpy()


def create_submission(predictions, filename):
    with open(filename + '.csv', 'w') as solution_file:
        solution_file.write('Id,Predicted\n')
        for i, prediction in enumerate(predictions):
            solution_file.write(f"{i},{prediction}\n")

create_submission(predicted_classes, 'submission')

