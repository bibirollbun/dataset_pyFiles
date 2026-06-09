import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 0.0001
FINE_TUNE_LR = 0.00001
FINE_TUNE_EPOCHS = 20


# Prepare data generators with aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Create data generators
train_dir = '/kaggle/input/heads-or-tails-image-classification/train'
test_dir = '/kaggle/input/heads-or-tails-image-classification/test'

# Calculate class weights (address dataset imbalance)
heads_dir = os.path.join(train_dir, 'heads')
tails_dir = os.path.join(train_dir, 'tails')
num_heads = len(os.listdir(heads_dir))
num_tails = len(os.listdir(tails_dir))
total = num_heads + num_tails
class_weights = {
    0: total / (2 * num_tails),  # Tails class weight
    1: total / (2 * num_heads)   # Heads class weight
}

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True,
    seed=42
)



base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model layers
base_model.trainable = False


inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[AUC(name='auc'), 'accuracy']
)



early_stop = EarlyStopping(
    monitor='val_auc',
    patience=8,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
    mode='max'
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)


# Train initial model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Fine-tuning stage
base_model.trainable = True

# Freeze first 150 layers, unfreeze the rest
for layer in base_model.layers[:150]:
    layer.trainable = False
for layer in base_model.layers[150:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='binary_crossentropy',
    metrics=[AUC(name='auc'), 'accuracy']
)

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Load best model
model = tf.keras.models.load_model('best_model.h5')



# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.dirname(test_dir),
    classes=['test'],
    target_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode=None
)

# Predict on test set
test_generator.reset()
probabilities = model.predict(test_generator, verbose=1)


# Create submission file
file_names = test_generator.filenames
image_ids = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in file_names]

submission = pd.DataFrame({
    'prediction_id': image_ids,
    'probability_of_heads': probabilities.flatten()
})

submission = submission.sort_values('prediction_id')
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")

