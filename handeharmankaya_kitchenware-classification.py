import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3 
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/kitchenware-classification/train.csv')
img_path='/kaggle/input/kitchenware-classification/images/'


df.head()


df['filename'] = df['Id'].astype(str).apply(lambda x: x.zfill(4) + '.jpg')


df.head()


df.shape


df.isnull().sum()


sample_test = df.sample(20).reset_index(drop=True)
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    img_name = sample_test.loc[i, 'filename']
    label_name = sample_test.loc[i, 'label']  
    full_path = os.path.join(img_path, img_name)
    img = plt.imread(full_path)
    ax.imshow(img)
    ax.set_title(label_name, fontsize=12)  
    ax.axis('off')
plt.tight_layout()
plt.show()


df['label'].value_counts()


sns.countplot(x=df['label'], palette='Spectral');


train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)


IMG_SIZE = 300 
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=30,
                                   width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
                                   zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,directory=img_path,x_col='filename',
                                                    y_col='label',target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE,class_mode='categorical',shuffle=True)

val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,directory=img_path,x_col='filename',
                                                y_col='label',target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE,class_mode='categorical',shuffle=False)


def create_model(num_classes):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False 

    model = models.Sequential([base_model,
                               layers.GlobalAveragePooling2D(),
                               layers.BatchNormalization(),
                               layers.Dropout(0.3),
                               layers.Dense(256, activation='relu'),
                               layers.Dropout(0.2),
                               layers.Dense(num_classes, activation='softmax')    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

num_classes = df['label'].nunique()
model = create_model(num_classes)
model.summary()


checkpoint = ModelCheckpoint('best_kitchen_model.keras',monitor='val_accuracy',
                             save_best_only=True,mode='max',verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,min_lr=1e-6,verbose=1)

early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,verbose=1)

callbacks = [checkpoint, reduce_lr, early_stop]


history_warmup = model.fit(train_generator,epochs=5,validation_data=val_generator,callbacks=callbacks)


history_warmup.history['accuracy'][-1]


model.save("kitchen.h5")


model.save_weights("kitchen_weights.weights.h5")


tf.keras.models.save_model(model,"kitchen1.keras")


plt.plot(history_warmup.history['accuracy'],label='Accuracy')
plt.plot(history_warmup.history['val_accuracy'],label='Val_Accuracy')
plt.plot(history_warmup.history['loss'], label='Loss')
plt.plot(history_warmup.history['val_loss'], label='Val_Loss')
plt.legend();


test_df = pd.read_csv('/kaggle/input/kitchenware-classification/test.csv')
test_df['filename'] = test_df['Id'].astype(str).apply(lambda x: x.zfill(4) + '.jpg')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,directory=img_path,x_col='filename',
                                                  y_col=None,target_size=(300, 300),batch_size=32,
                                                  class_mode=None,shuffle=False)

predictions = model.predict(test_generator, verbose=1)
predicted_indices = np.argmax(predictions, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predicted_labels = [labels[k] for k in predicted_indices]

submission = pd.DataFrame({'Id': test_df['Id'],'label': predicted_labels})

submission.to_csv('submission_warmup.csv', index=False)


model.layers[0].trainable = True 

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

history_finetune = model.fit(train_generator,epochs=15,validation_data=val_generator,callbacks=callbacks)


history_finetune.history['accuracy'][-1]


tf.keras.models.save_model(model,"kitchen2.keras")


acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history_warmup.history['loss'] + history_finetune.history['loss']
val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()


test_df = pd.read_csv('/kaggle/input/kitchenware-classification/test.csv')
test_df['filename'] = test_df['Id'].astype(str).apply(lambda x: x.zfill(4) + '.jpg')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,directory=img_path,x_col='filename',
                                                  y_col=None,target_size=(300, 300),batch_size=32,
                                                  class_mode=None,shuffle=False)

predictions = model.predict(test_generator, verbose=1)
predicted_indices = np.argmax(predictions, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predicted_labels = [labels[k] for k in predicted_indices]

submission = pd.DataFrame({'Id': test_df['Id'],'label': predicted_labels})

submission.to_csv('submission_finetune.csv', index=False)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
class_names = [labels[k] for k in sorted(labels.keys())]
print(class_names)

