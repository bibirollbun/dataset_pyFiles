# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Note: Ignore the file walk as there are thousands of files.
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
df.head()


print(df['label'].value_counts())
print(df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt

label_counts = df['label'].value_counts()
total = len(df)

# Plot
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='label')

for p in ax.patches:
    count = p.get_height()
    percent = f'{100 * count / total:.2f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.text(x, y + total * 0.01, percent, ha='center', va='bottom', fontsize=12)

# Format
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.ylim(0, label_counts.max() * 1.1)
plt.show()


import cv2
import numpy as np
import os

def load_image(image_id, base_path='/kaggle/input/histopathologic-cancer-detection/train'):
    path = os.path.join(base_path, f"{image_id}.tif")
    return cv2.imread(path)

def show_samples(df, label, n=5):
    samples = df[df['label'] == label].sample(n)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for img_id, ax in zip(samples['id'], axes):
        img = load_image(img_id)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.suptitle(f"Label: {label}")
    plt.show()

show_samples(df, label=0)
show_samples(df, label=1)


img = load_image(df['id'][0])
print("Image shape:", img.shape)


def plot_color_distribution_histogram(image):
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        plt.hist(image[..., i].ravel(), bins=256, color=color, alpha=0.5)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()

img_tumor = load_image(df[df['label'] == 1].sample(1).iloc[0]['id'])
img_normal = load_image(df[df['label'] == 0].sample(1).iloc[0]['id'])

plot_color_distribution_histogram(img_tumor)
plot_color_distribution_histogram(img_normal)


def tissue_ratio(img, threshold=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray < threshold)

# Use only a sample for EDA
sample_ids = df['id'].sample(2000, random_state=42)

ratios = sample_ids.apply(lambda id_: tissue_ratio(load_image(id_)))
df['tissue_ratio'] = ratios

sns.histplot(data=df, x='tissue_ratio', hue='label', bins=30, kde=True)
plt.title("Tissue Area Ratio Distribution")
plt.show()


df['label_str'] = df['label'].astype(str)
df['filename'] = df['id'] + '.tif'


from sklearn.model_selection import train_test_split

df_sampled, _ = train_test_split(
    df,
    train_size=40000,
    stratify=df['label'],
    random_state=42
)

train_df, val_df = train_test_split(
    df_sampled,
    test_size=0.2,                # 20% for validation
    stratify=df_sampled['label'],         # maintain class balance
    random_state=42               # reproducibility
)

print("Train label distribution:")
print(train_df['label'].value_counts(normalize=True))

print("\nValidation label distribution:")
print(val_df['label'].value_counts(normalize=True))


import cv2
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

def load_image_cv2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def data_generator(paths, labels):
    for path, label in zip(paths, labels):
        img = load_image_cv2(path)
        yield img, label

train_paths = [f"/kaggle/input/histopathologic-cancer-detection/train/{f}" for f in train_df['filename']]
train_labels = train_df['label'].values.astype(np.float32)

val_paths = [f"/kaggle/input/histopathologic-cancer-detection/train/{f}" for f in val_df['filename']]
val_labels = val_df['label'].values.astype(np.float32)

train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).shuffle(1024).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


for images, labels in val_data.take(1):
    print(np.unique(labels.numpy(), return_counts=True))


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_basic(input_shape=(96, 96, 3)):
    model = models.Sequential(name="Basic_Model")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

basic_model = build_basic()
basic_model.summary()


import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg_like(input_shape=(96, 96, 3)):
    model = models.Sequential(name="VGGNet")
    model.add(layers.Input(shape=input_shape))
    
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

vgg_model = build_vgg_like()
vgg_model.summary()


steps_per_epoch = len(train_df) // BATCH_SIZE


import tensorflow as tf
from tensorflow.keras import layers, models

basic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    # This helps stop the model early
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    # Reduces learning rate when plateued
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_basic_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

bm_history = basic_model.fit(
    train_data, 
    validation_data = val_data, 
    epochs=10, 
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list
)
basic_model.evaluate(val_data)


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(bm_history.history['accuracy'], label='Train Accuracy')
plt.plot(bm_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(bm_history.history['loss'], label='Train Loss')
plt.plot(bm_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


import numpy as np

def get_y_true_and_y_pred(val_data, model, threshold):

    # Predict on validation set
    y_true = []
    y_pred_probs = []
    
    for batch_x, batch_y in val_data:
        preds = model.predict(batch_x, verbose=0)
        y_pred_probs.extend(preds.ravel())  # Flatten to 1D list
        y_true.extend(batch_y.numpy().ravel())  # Convert from tensor to NumPy array
    
    # Convert to NumPy arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs > threshold).astype(int)
    return y_true, y_pred


y_true_bm, y_pred_bm = get_y_true_and_y_pred(val_data, basic_model, 0.5)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(y_true_bm, y_pred_bm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true_bm, y_pred_probs_bm)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).shuffle(1024).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

vgg_history = vgg_model.fit(
    train_data, 
    validation_data = val_data, 
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list
)
vgg_model.evaluate(val_data)


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(vgg_history.history['accuracy'], label='Train Accuracy')
plt.plot(vgg_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history.history['loss'], label='Train Loss')
plt.plot(vgg_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


y_true_vgg, y_pred_vgg = get_y_true_and_y_pred(val_data, vgg_model, 0.5)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(y_true_vgg, y_pred_vgg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true_vgg, y_pred_probs_vgg)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).shuffle(1024).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(32, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

new_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

vgg_model.compile(optimizer=new_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model_v1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

vgg_history_v2 = vgg_model.fit(
    train_data, 
    validation_data = val_data, 
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list
)
vgg_model.evaluate(val_data)


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(vgg_history_v2.history['accuracy'], label='Train Accuracy')
plt.plot(vgg_history_v2.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history_v2.history['loss'], label='Train Loss')
plt.plot(vgg_history_v2.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg_like_v2(input_shape=(96, 96, 3)):
    model = models.Sequential(name="VGGNetv2")
    model.add(layers.Input(shape=input_shape))
    
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

vgg_model_v2 = build_vgg_like_v2()
vgg_model_v2.summary()


train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).shuffle(1024).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


new_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

vgg_model_v2.compile(optimizer=new_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model_v2.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

vgg_history_v3 = vgg_model_v2.fit(
    train_data, 
    validation_data = val_data, 
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list
)
vgg_model_v2.evaluate(val_data)


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(vgg_history_v3.history['accuracy'], label='Train Accuracy')
plt.plot(vgg_history_v3.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history_v3.history['loss'], label='Train Loss')
plt.plot(vgg_history_v3.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg_like_v3(input_shape=(96, 96, 3)):
    model = models.Sequential(name="VGGNetv3")
    model.add(layers.Input(shape=input_shape))
    
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

vgg_model_v3 = build_vgg_like_v3()
vgg_model_v3.summary()


train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).shuffle(1024).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

new_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

vgg_model_v3.compile(optimizer=new_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # save best model
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model_v3.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

vgg_history_v4 = vgg_model_v3.fit(
    train_data, 
    validation_data = val_data, 
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list
)
vgg_model_v3.evaluate(val_data)


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(vgg_history_v4.history['accuracy'], label='Train Accuracy')
plt.plot(vgg_history_v4.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history_v4.history['loss'], label='Train Loss')
plt.plot(vgg_history_v4.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


y_true_vgg, y_pred_vgg = get_y_true_and_y_pred(val_data, vgg_model, 0.5)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(y_true_vgg, y_pred_vgg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true_vgg, y_pred_probs_vgg)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


#import os
#import pandas as pd

# Get all .tif filenames from the test folder
#test_dir = "/kaggle/input/histopathologic-cancer-detection/test/"
#test_filenames = sorted([f for f in os.listdir(test_dir) if f.endswith(".tif")])

# Extract IDs by removing ".tif"
#test_ids = [f[:-4] for f in test_filenames]


#import tensorflow as tf
#import cv2
#import numpy as np

#def test_data_generator(paths):
#    for path in paths:
#        img = load_image_cv2(path)
#        yield img

# Create full paths
#test_paths = [os.path.join(test_dir, fname) for fname in test_filenames]

# Create tf.data.Dataset
#test_dataset = tf.data.Dataset.from_generator(
#    lambda: test_data_generator(test_paths),
#    output_signature=tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32)
#).batch(32).prefetch(tf.data.AUTOTUNE)


#from tensorflow.keras.models import load_model

#best_model = load_model('best_vgg_model_v3.keras')

#pred_probs = best_model.predict(test_dataset, verbose=1)
#pred_labels = (pred_probs > 0.5).astype(int).flatten()


#submission_df = pd.DataFrame({
#    'id': test_ids,
#    'label': pred_labels
#})

#submission_df.to_csv("submission.csv", index=False)


from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.models import load_model

basic_model = load_model('best_basic_model.keras')
y_true_bm, y_pred_bm = get_y_true_and_y_pred(val_data, basic_model, 0.5)

bm_rc = recall_score(y_true_bm, y_pred_bm)
bm_ac = accuracy_score(y_true_bm, y_pred_bm)


from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.models import load_model

vgg_model = load_model('best_vgg_model.keras')
y_true_vgg, y_pred_vgg = get_y_true_and_y_pred(val_data, vgg_model, 0.5)

vgg_rc = recall_score(y_true_vgg, y_pred_vgg)
vgg_ac = accuracy_score(y_true_vgg, y_pred_vgg)


from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.models import load_model

vgg_model_lr = load_model('best_vgg_model_v1.keras')
y_true_vgg_lr, y_pred_vgg_lr = get_y_true_and_y_pred(val_data, vgg_model_lr, 0.5)

vgg_lr_rc = recall_score(y_true_vgg_lr, y_pred_vgg_lr)
vgg_lr_ac = accuracy_score(y_true_vgg_lr, y_pred_vgg_lr)


from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.models import load_model

vgg_model_v2 = load_model('best_vgg_model_v2.keras')
y_true_vgg_v2, y_pred_vgg_v2 = get_y_true_and_y_pred(val_data, vgg_model_v2, 0.5)

vgg_v2_rc = recall_score(y_true_vgg_v2, y_pred_vgg_v2)
vgg_v2_ac = accuracy_score(y_true_vgg_v2, y_pred_vgg_v2)


from sklearn.metrics import recall_score, accuracy_score
from tensorflow.keras.models import load_model

vgg_model_v3 = load_model('best_vgg_model_v3.keras')
y_true_vgg_v3, y_pred_vgg_v3 = get_y_true_and_y_pred(val_data, vgg_model_v3, 0.5)

vgg_v3_rc = recall_score(y_true_vgg_v3, y_pred_vgg_v3)
vgg_v3_ac = accuracy_score(y_true_vgg_v3, y_pred_vgg_v3)


results = {
    'Model': ['Basic Model', 'VGG Model', 'VGG Model with 1e-4 LR', 'VGG Model v2', 'VGG Model v3'],
    'Accuracy': [bm_ac, vgg_ac, vgg_lr_ac, vgg_v2_ac, vgg_v3_ac],
    'Recall': [bm_rc, vgg_rc, vgg_lr_rc, vgg_v2_rc, vgg_v3_rc],
    'Miss Rate': [1 - bm_rc, 1 - vgg_rc, 1 - vgg_lr_rc, 1 - vgg_v2_rc, 1 - vgg_v3_rc]
}

df_result = pd.DataFrame(results).sort_values(by=['Miss Rate'], ascending=True)
df_result

