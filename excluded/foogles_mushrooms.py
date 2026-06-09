from pathlib import Path

DATASET_DIR = Path('/kaggle/input/mushroom-multiclass-classification')
IMG_DIR = Path('/kaggle/input/mushroom-multiclass-classification/dataset/dataset')
TRAIN_PATH = Path('/kaggle/input/mushroom-multiclass-classification/train.csv')
TEST_PATH = Path('/kaggle/input/mushroom-multiclass-classification/test.csv')
MAIN_DIR= Path('/kaggle/working/')
MAIN_DIR.mkdir(exist_ok=True)
IMAGE_SIZE = (224,224,3)
BATCH_SIZE = 64


from sklearn.model_selection import train_test_split
import albumentations as A
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import pandas as pd


train_df = pd.read_csv(DATASET_DIR.joinpath("train.csv"))
test_df = pd.read_csv(DATASET_DIR.joinpath("test.csv"))
train_df['Mushroom'].value_counts().plot(kind='bar', figsize=(10,6))
index_to_label = train_df.groupby('Mushroom')['Image'].first()
print(index_to_label)


train_unsampled, val = train_test_split(train_df, stratify=train_df['Mushroom'], random_state=89)
val, test = train_test_split(val, stratify=val['Mushroom'], random_state=89)
print(train_unsampled.shape, val.shape, test.shape)


ROS = RandomOverSampler(sampling_strategy='auto')
train, _ = ROS.fit_resample(train_unsampled, train_unsampled['Mushroom'])
train['Mushroom'].value_counts().plot.bar(figsize=((12,5)), title='Resampled classes')
plt.show()

print(f'Total positive items: {len(train)+len(val)}')
print(f'Train: {len(train)}, val: {len(val)}, test: {len(test)}')


train.to_csv(MAIN_DIR.joinpath('train.csv'), index=False)
val.to_csv(MAIN_DIR.joinpath('val.csv'), index=False)
test.to_csv(MAIN_DIR.joinpath('test.csv'), index=False)


train = pd.read_csv(MAIN_DIR.joinpath('train.csv'))
val = pd.read_csv(MAIN_DIR.joinpath('val.csv'))
test = pd.read_csv(MAIN_DIR.joinpath('test.csv'))
train = train.groupby('Mushroom').head(1000)


train['Image'] = train['Image'].astype(str) 
val['Image'] = val['Image'].astype(str) 
test['Image'] = test['Image'].astype(str) 

train.loc[:,'Image'] = train.Image.apply(lambda x: str(IMG_DIR)+"/" +str(x).zfill(5)+".jpg")
val.loc[:,'Image'] = val.Image.apply(lambda x: str(IMG_DIR)+"/" +str(x).zfill(5)+".jpg")
test.loc[:,'Image'] = test.Image.apply(lambda x: str(IMG_DIR)+"/" +str(x).zfill(5)+".jpg")


test


import tensorflow as tf
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    image = tf.image.resize(image, size=IMAGE_SIZE[:2])
    image = tf.cast(image, tf.float32)
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((train['Image'].values, train['Mushroom'].values))


train_ds = train_ds.shuffle(buffer_size=len(train_ds), reshuffle_each_iteration=True)
train_ds = train_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

val_ds = tf.data.Dataset.from_tensor_slices((val['Image'].values, val['Mushroom'].values))
val_ds = val_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

test_ds = tf.data.Dataset.from_tensor_slices((test['Image'].values, test['Mushroom'].values))
test_ds = test_ds.map(load_image, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from datetime import datetime


MAIN_DIR.joinpath('cp').mkdir(parents=True, exist_ok=True)
MAIN_DIR.joinpath('logs').mkdir(parents=True, exist_ok=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=2,
                                         verbose=1,
                                         min_delta=0.000001,
                                         cooldown=0,
                                         min_lr=0.0000001)

chk_path = MAIN_DIR.joinpath('cp', 'ckpt_{epoch:02d}.weights.h5')

checkpointer = tf.keras.callbacks.ModelCheckpoint(chk_path, save_best_only=True, verbose=1, save_weights_only=True)

stopper = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, min_delta=0.0001, verbose=1)

logger = CSVLogger(MAIN_DIR.joinpath("logs", f"training_log.csv"), separator=';', append=True)

logs = "logs/" + datetime.now().strftime("thresh_%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '200,205')


callbacks = [reduce_learning_rate,
             checkpointer,
             stopper,
             logger,
]

print('Done')


import matplotlib.pyplot as plt

def show_results(history):
    accuracy = history.history['f1_score']
    val_accuracy = history.history['val_f1_score']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(accuracy)+1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.legend()
    plt.title("Training and validation accuracy")
    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.legend()
    plt.title("Training and validation loss")
    plt.show()



from tensorflow.keras.metrics import F1Score

class SparseF1Score(F1Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
        super().update_state(y_true_one_hot, y_pred, sample_weight)


# from tensorflow.keras.applications import EfficientNetV2B3

# base = EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
# base.trainable = True
# for layer in base.layers[:-20]:  
#     layer.trainable = False

# inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# x = base(inputs)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(256, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

# model = tf.keras.models.Model(inputs, outputs, name='Mushroom-10')

# model.summary()

# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics= [tf.keras.metrics.SparseCategoricalAccuracy(),SparseF1Score(average='macro', threshold=None, name='f1_score')])

# history = model.fit(
#     train_ds,
#     epochs=1000,
#     validation_data=val_ds,
#     callbacks=callbacks
# )


# show_results(history)


# model.save('/kaggle/working/SparseCategorical.keras')


model = tf.keras.models.load_model('/kaggle/input/sparsecategorical/keras/sparsecategorical/1/SparseCategorical.keras')


# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test accuracy: {test_acc:.3f}")

# from sklearn.metrics import f1_score
# f1 = f1_score(val_ds, train_ds)


df_res = test_df.copy()
df_res['Image'] = df_res['Image'].astype(str) 
df_res.loc[:,'Image'] = df_res.Image.apply(lambda x: str(IMG_DIR)+"/" +str(x).zfill(5)+".jpg")
print("Done")


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


import numpy as np
import tensorflow as tf


def predict(model_r, dataset):
    preds = []
    for image_path in dataset:
        img = load_image(image_path)
        img = tf.expand_dims(img, axis=0)
        output = model_r(img, training=False)
        pred_idx = np.argmax(output.numpy(), axis=1)[0]
        preds.append(pred_idx)
    return preds

def create_submission(predictions, filename):
    with open(filename + '.csv', 'w') as solution_file:
        solution_file.write('Id,Predicted\n')
        for i, prediction in enumerate(predictions):
            solution_file.write(f"{i},{prediction}\n")
        
test_ds_1 = test_ds
test_ds_1 = test_ds_1.map(lambda x, y: x) 
test_ds_1 = test_ds_1.map(lambda x: tf.keras.applications.efficientnet.preprocess_input(x))
test_ds_1 = test_ds_1.batch(64, drop_remainder=True)
preds = predict(model, df_res["Image"])
create_submission(preds, "submission")

