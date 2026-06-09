import numpy as np
from tensorflow import keras
import os, shutil, pathlib
from tensorflow.keras.utils import image_dataset_from_directory
import warnings
warnings.filterwarnings('ignore')


!unzip -qq /kaggle/input/dogs-vs-cats/train.zip


!unzip -qq /kaggle/input/dogs-vs-cats/test1.zip


original_train_dir = pathlib.Path('/kaggle/working/train')  # Now a Path object
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
#make_subset('test', start_index=1500, end_index=2500)


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


from tensorflow.keras.preprocessing import image_dataset_from_directory

test_dir = "/kaggle/working/test1"  # Path to your test folder
img_size = (180, 180)  # Should match your model's expected input shape
batch_size = 32

# Load test dataset (unlabeled)
test_dataset = image_dataset_from_directory(
    directory=test_dir,
    labels=None,            # No labels available
    image_size=img_size,    # Resize to model's expected input
    batch_size=batch_size,
    shuffle=False,          # Keep original order for submission
)


for data_batch, labels_batch in train_dataset:
  print(data_batch.shape)
  print(labels_batch.shape)
  break


from tensorflow.keras import layers


conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False
)
conv_base.trainable = False


data_augmentation = keras.Sequential(
    [
 layers.RandomFlip("horizontal"),
 layers.RandomRotation(0.1),
 layers.RandomZoom(0.2),
    ]
 )


inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)


conv_base.trainable = True
for layer in conv_base.layers[:-4]:
  layer.trainable = False


model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])
callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="fine_tuning.keras",
    save_best_only=True,
    monitor="val_loss")
 ]


model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=callbacks
)


import pandas as pd
test_dir = "/kaggle/working/test1"  # Adjust if different


test_filenames = os.listdir(test_dir)

test_df = pd.DataFrame({
    'filename': test_filenames
})


predictions = model.predict(test_dataset)
test_df['label'] = (predictions > 0.5).astype(int).ravel()
test_df['id'] = test_df['filename'].str.extract('(\d+)').astype(int)

# Create submission
submission_df = test_df[['id', 'label']]
submission_df.to_csv('submission.csv', index=False)


model.save('cats_vs_dogs_model.h5')


model.save_weights('model_weights.weights.h5')

