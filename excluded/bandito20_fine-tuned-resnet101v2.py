import numpy as np 
from tensorflow import keras
import os, shutil, pathlib
from tensorflow.keras.utils import image_dataset_from_directory


!unzip -qq /kaggle/input/dogs-vs-cats/train.zip


!unzip -qq /kaggle/input/dogs-vs-cats/test1.zip


original_train_dir = pathlib.Path('/kaggle/working/train')  
new_base_dir = pathlib.Path('cats_vs_dogs_train')

def make_subset(subset_name, start_index, end_index):
  for category in ('cat', 'dog'):
    dir = new_base_dir / subset_name / category
    os.makedirs(dir, exist_ok=True)
    fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
    for fname in fnames:
      shutil.copyfile(src=original_train_dir / fname,
                      dst=dir / fname)
      
make_subset('train', start_index=0, end_index=10000)
make_subset('validation', start_index=10000, end_index=12500)


train_dataset = image_dataset_from_directory(
    new_base_dir / 'train',
    image_size=(180, 180),
    batch_size=32
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / 'validation',
    image_size=(180, 180),
    batch_size=32
)


test_dir = "/kaggle/working/test1"  
img_size = (180, 180)  
batch_size = 32

test_dataset = image_dataset_from_directory(
    directory=test_dir,
    labels=None,            
    image_size=img_size,    
    batch_size=batch_size,
    shuffle=False,         
)


from tensorflow.keras import layers

conv_base = keras.applications.ResNet101V2(
    weights='imagenet',
    include_top=False
)
conv_base.trianable = False


data_augmentation = keras.Sequential(
    [
 layers.RandomFlip("horizontal"),
 layers.RandomRotation(0.1),
 layers.RandomZoom(0.2),
    ]
 )


inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet_v2.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)


conv_base.trainable = True
for layer in conv_base.layers[:-5]:
  layer.trainable = False


model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="fine_tuning.keras",
    save_best_only=True,
    monitor="val_loss")
 ]


history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=callbacks
)


import matplotlib.pyplot as plt

def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_title("Loss")
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('binary_crossentropy')
  ax1.legend()
  ax1.grid(True)


  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_title("Accuracy")
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)
  ax2.legend()
  plt.show()


plot_history(history)

