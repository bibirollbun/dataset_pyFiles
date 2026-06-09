import os
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models


# Unzip the training images
zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
extract_path = "/kaggle/working/train"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Move images into cats/ and dogs/ folders
os.makedirs(f"{extract_path}/cats", exist_ok=True)
os.makedirs(f"{extract_path}/dogs", exist_ok=True)

image_dir = os.path.join(extract_path, "train")  # actual image folder inside zip
for fname in os.listdir(image_dir):
    src = os.path.join(image_dir, fname)
    if fname.startswith("cat"):
        shutil.move(src, os.path.join(extract_path, "cats", fname))
    elif fname.startswith("dog"):
        shutil.move(src, os.path.join(extract_path, "dogs", fname))

# Clean up leftover folder
shutil.rmtree(image_dir)



datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    extract_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    extract_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # output is probability of dog
])



model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



#Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)



# Visualize Training Curves
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()


from tensorflow.keras import regularizers

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dropout(0.5),  # ðŸ”¥ NEW: turns off 50% neurons
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])



model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]




history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  
    callbacks=callbacks
)


# Visualize Training Curves
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()





# Grab one batch from val_gen
val_imgs, val_labels = next(val_gen)  # val_imgs.shape = (32, 150, 150, 3)

# Predict probabilities
val_preds = model.predict(val_imgs).flatten()
import matplotlib.pyplot as plt

class_names = ['Cat', 'Dog']

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(val_imgs[i])
    plt.axis('off')
    
    true_label = class_names[int(val_labels[i])]
    pred_prob = val_preds[i]
    pred_label = class_names[int(pred_prob > 0.5)]
    
    plt.title(f"Pred: {pred_label}\nProb: {pred_prob:.2f}\nTrue: {true_label}")
plt.tight_layout()
plt.show()



from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (300, 300)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    "/kaggle/working/train",
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "/kaggle/working/train",
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load base model with 300x300 input
base_model = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(300, 300, 3)
)
base_model.trainable = False  # freeze initially

# Add custom classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train head
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks
)



# Unfreeze top layers
base_model.trainable = True

# Freeze earlier layers, fine-tune last ~30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Re-compile with lower learning rate
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune entire model
fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)



# STEP 1: Data Generator with Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    "/kaggle/working/train",   # Make sure cats/ and dogs/ are subfolders here
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "/kaggle/working/train",
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



# STEP 2: Build ResNet50V2 model with custom head
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models

base_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze initial layers

# Custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

# Final model
model = models.Model(inputs=base_model.input, outputs=output)

# STEP 3: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# STEP 4: Add callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]



# STEP 5: Train the classifier head
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks
)



# STEP 6: Fine-tune the top layers of ResNet
base_model.trainable = True

# Optionally freeze earlier layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Re-compile with lower LR for fine-tuning
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training
fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)



from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tqdm import tqdm

# Path to test folder (after proper unzip)
test_dir = "/kaggle/working/test_flat"  # <- update this if different
test_files = sorted(os.listdir(test_dir))

X_test = []
ids = []

for fname in tqdm(test_files):
    img_path = os.path.join(test_dir, fname)
    img = image.load_img(img_path, target_size=(224, 224))  # âœ… match ResNet input
    img_array = image.img_to_array(img) / 255.0
    X_test.append(img_array)
    ids.append(int(fname.split('.')[0]))

X_test = np.array(X_test)



# Predict all at once
preds = model.predict(X_test, batch_size=64).flatten()  # Binary probabilities
import pandas as pd

submission = pd.DataFrame({
    'id': ids,
    'label': preds
})

submission = submission.sort_values('id')  # Ensure order
submission.to_csv("submission.csv", index=False)



# STEP 1: Imports
from tensorflow.keras.applications import ResNet50V2, EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# STEP 2: Data Generators
img_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    "/kaggle/working/train",
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "/kaggle/working/train",
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# STEP 3: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)

# STEP 4: Build Base Model Function
def build_model(base_fn):
    base_model = base_fn(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss=BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

# STEP 5: Train ResNet50V2 (no fine-tuning)
model_resnet = build_model(ResNet50V2)
model_resnet.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr])

# STEP 6: Train EfficientNetB3 (no fine-tuning)
model_effnet = build_model(EfficientNetB3)
model_effnet.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr])



shutil.rmtree("/kaggle/working/test_flat", ignore_errors=True)

import zipfile

# Extract directly to /kaggle/working/test_flat
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

# Rename from /test â†’ /test_flat to make it clear
import os
os.rename("/kaggle/working/test", "/kaggle/working/test_flat")


# STEP 7: TTA + Ensemble Predictions
test_dir = "/kaggle/working/test_flat"
test_files = sorted(os.listdir(test_dir))
batch_size = 500
ids = []
ensemble_preds = []

for i in tqdm(range(0, len(test_files), batch_size)):
    batch_files = test_files[i:i+batch_size]
    batch_imgs = []
    batch_imgs_flipped = []
    batch_ids = []

    for fname in batch_files:
        path = os.path.join(test_dir, fname)
        img = image.load_img(path, target_size=img_size)
        arr = image.img_to_array(img) / 255.0
        flipped = np.fliplr(arr)

        batch_imgs.append(arr)
        batch_imgs_flipped.append(flipped)
        batch_ids.append(int(fname.split('.')[0]))

    batch_imgs = np.array(batch_imgs)
    batch_imgs_flipped = np.array(batch_imgs_flipped)

    resnet_orig = model_resnet.predict(batch_imgs, batch_size=64).flatten()
    resnet_flip = model_resnet.predict(batch_imgs_flipped, batch_size=64).flatten()

    effnet_orig = model_effnet.predict(batch_imgs, batch_size=64).flatten()
    effnet_flip = model_effnet.predict(batch_imgs_flipped, batch_size=64).flatten()

    resnet_tta = (resnet_orig + resnet_flip) / 2
    effnet_tta = (effnet_orig + effnet_flip) / 2

    ensemble_pred = (resnet_tta + effnet_tta) / 2
    ensemble_pred = np.clip(ensemble_pred ** 1.2, 1e-5, 1 - 1e-5)

    ids.extend(batch_ids)
    ensemble_preds.extend(ensemble_pred)

# STEP 8: Create Submission
submission = pd.DataFrame({'id': ids, 'label': ensemble_preds})
submission = submission.sort_values('id')
submission.to_csv("submission_ensemble2.csv", index=False)



shutil.rmtree("/kaggle/working/test_flat", ignore_errors=True)



shutil.rmtree("/kaggle/working/test_flat", ignore_errors=True)

import zipfile

# Extract directly to /kaggle/working/test_flat
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

# Rename from /test â†’ /test_flat to make it clear
import os
os.rename("/kaggle/working/test", "/kaggle/working/test_flat")



from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Path to the test images (ensure this folder has all .jpgs flat)
test_dir = "/kaggle/working/test_flat"
test_files = sorted(os.listdir(test_dir))

# Setup
batch_size = 500  # Number of images to load per mini-batch
ids = []
preds = []

# Predict in batches
for i in tqdm(range(0, len(test_files), batch_size)):
    batch_files = test_files[i:i+batch_size]
    batch_imgs = []
    batch_ids = []

    for fname in batch_files:
        img_path = os.path.join(test_dir, fname)
        img = image.load_img(img_path, target_size=(224, 224))  # match model input
        img_array = image.img_to_array(img) / 255.0
        batch_imgs.append(img_array)
        batch_ids.append(int(fname.split('.')[0]))

    batch_array = np.array(batch_imgs)
    batch_preds = model.predict(batch_array, batch_size=64).flatten()

    ids.extend(batch_ids)
    preds.extend(batch_preds)



submission = pd.DataFrame({
    'id': ids,
    'label': preds
})
submission = submission.sort_values('id')
submission.to_csv("submission2.csv", index=False)



test_dir = "/kaggle/working/test_flat"

from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

test_files = sorted(os.listdir(test_dir))

X_test = []
ids = []

for fname in tqdm(test_files):
    img_path = os.path.join(test_dir, fname)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    X_test.append(img_array)
    ids.append(int(fname.split('.')[0]))

X_test = np.array(X_test)




# Predict all at once
preds = model.predict(X_test, batch_size=64).flatten()

# Create submission
submission = pd.DataFrame({'id': ids, 'label': preds})
submission = submission.sort_values('id')
submission.to_csv("submission.csv", index=False)



# Predict probabilities
preds_effnet = model_effnet.predict(X_test, batch_size=64).flatten()
preds_resnet = model_resnet.predict(X_test, batch_size=64).flatten()

# Average (simple ensemble)
final_preds = (preds_effnet + preds_resnet) / 2


