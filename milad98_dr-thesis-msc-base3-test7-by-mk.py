import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, cohen_kappa_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything()



train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


print('Number of Train Samples: ', train.shape[0])
print('Number of Test Samples: ', test.shape[0])
display(train.head())


f, ax = plt.subplots(figsize=(14, 8.7))
ax = sns.countplot(x="diagnosis", data=train, palette="GnBu_d")
sns.despine()
plt.show()


sns.set_style("white")
count = 1
plt.figure(figsize=[20, 20])
for img_name in train['id_code'][:15]:
    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/%s.png" % img_name)[...,[2, 1, 0]]
    plt.subplot(5, 5, count)
    plt.imshow(img)
    plt.title("Image %s" % count)
    count += 1
    
plt.show()


# Model parameters
BATCH_SIZE = 8
EPOCHS = 20
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-4
WARMUP_LEARNING_RATE = 1e-3
HEIGHT = 512
WIDTH = 512
CANAL = 3
N_CLASSES = train['diagnosis'].nunique()
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5


# Preprocecss data
train["id_code"] = train["id_code"].apply(lambda x: x + ".png")
test["id_code"] = test["id_code"].apply(lambda x: x + ".png")
train['diagnosis'] = train['diagnosis'].astype('str')
train.head()


train_datagen=ImageDataGenerator(rescale=1./255, 
                                 validation_split=0.2,
                                 horizontal_flip=True)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=train,
    directory="../input/aptos2019-blindness-detection/train_images/",
    x_col="id_code",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(HEIGHT, WIDTH),
    subset='training')

valid_generator=train_datagen.flow_from_dataframe(
    dataframe=train,
    directory="../input/aptos2019-blindness-detection/train_images/",
    x_col="id_code",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",    
    target_size=(HEIGHT, WIDTH),
    subset='validation')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  
        dataframe=test,
        directory = "../input/aptos2019-blindness-detection/test_images/",
        x_col="id_code",
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        shuffle=False,
        class_mode=None)


def create_model(input_shape, n_out):
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(input_shape[0], input_shape[1], 3)
    )

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_out, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model



model = create_model(input_shape=(HEIGHT, WIDTH, 3), n_out=N_CLASSES)

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-5:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.Adam(
    learning_rate=WARMUP_LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()



history_warmup = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=WARMUP_EPOCHS,
    verbose=1
)



for layer in model.layers:
    layer.trainable = True

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=ES_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

rlrop = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=RLROP_PATIENCE,
    factor=DECAY_DROP,
    min_lr=1e-6,
    verbose=1
)

callback_list = [es, rlrop]

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()




history_finetunning = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callback_list,
    verbose=1
).history



history_warmup_dict = history_warmup.history
history_finetunning_dict = history_finetunning

history = {
    'loss': history_warmup_dict['loss'] + history_finetunning_dict['loss'],
    'val_loss': history_warmup_dict['val_loss'] + history_finetunning_dict['val_loss'],
    'accuracy': history_warmup_dict.get('accuracy', history_warmup_dict.get('acc', [])) + 
                history_finetunning_dict.get('accuracy', history_finetunning_dict.get('acc', [])),
    'val_accuracy': history_warmup_dict.get('val_accuracy', history_warmup_dict.get('val_acc', [])) + 
                    history_finetunning_dict.get('val_accuracy', history_finetunning_dict.get('val_acc', []))
}

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['accuracy'], label='Train Accuracy')
ax2.plot(history['val_accuracy'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()

history.keys()

history['loss']
history['val_loss']
history['accuracy']
history['val_accuracy']

history['loss'][-1], history['val_loss'][-1], history['accuracy'][-1], history['val_accuracy'][-1]



complete_datagen = ImageDataGenerator(rescale=1./255)
complete_generator = complete_datagen.flow_from_dataframe(  
        dataframe=train,
        directory="../input/aptos2019-blindness-detection/train_images/",
        x_col="id_code",
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        shuffle=False,
        class_mode=None)

STEP_SIZE_COMPLETE = complete_generator.n // complete_generator.batch_size

train_preds = model.predict(complete_generator, steps=STEP_SIZE_COMPLETE, verbose=1)
train_preds = [np.argmax(pred) for pred in train_preds]



labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
cnf_matrix = confusion_matrix(train['diagnosis'].astype('int'), train_preds)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
plt.show()


print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train['diagnosis'].astype('int'), weights='quadratic'))


test_generator.reset()
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

preds = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predictions = [np.argmax(pred) for pred in preds]



filenames = test_generator.filenames
results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])
results.to_csv('submission.csv',index=False)
results.head(10)


f, ax = plt.subplots(figsize=(14, 8.7))
ax = sns.countplot(x="diagnosis", data=results, palette="GnBu_d")
sns.despine()
plt.show()


history.keys()

history['loss']
history['val_loss']
history['accuracy']
history['val_accuracy']


