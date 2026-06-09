# TensorFlow and Keras Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

# Data Handling Libraries
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Visualization Libraries
import matplotlib.pyplot as plt

# File and Operating System Libraries
import os

# Warnings Management
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))

TRAIN_IMG_PATH = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'
TEST_IMG_PATH = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'



train_metadata = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
test_metadata = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')


train_metadata.info()


train_metadata.label.value_counts()


# Compute class weights based on the training labels
class_weights = compute_class_weight(
    'balanced',  # This will automatically adjust for class imbalance
    classes=np.unique(train_metadata.label),  # Unique classes in the target
    y=train_metadata.label  # The target class labels for training
)

# Create a dictionary mapping each class to its weight
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


def custom_preprocessing(image):
    # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    image = image / 255.0  # Rescale to [0, 1]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

train_datagen = ImageDataGenerator(
    rotation_range=15,            # Random rotation
    horizontal_flip=True,         # Random horizontal flip
    brightness_range=[0.8, 1.2],  # Approximation for brightness jitter
    preprocessing_function=custom_preprocessing,  # Custom normalization
    validation_split=0.2          # Validation split
)

datagen = ImageDataGenerator(rescale=1./255)

train_img = train_datagen.flow_from_dataframe(train_metadata,
    directory=TRAIN_IMG_PATH,
    x_col='image',
    y_col='label',
    weight_col=None,
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='training')

val_img = train_datagen.flow_from_dataframe(train_metadata,
    directory=TRAIN_IMG_PATH,
    x_col='image',
    y_col='label',
    weight_col=None,
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='validation')

test_img = datagen.flow_from_dataframe(test_metadata,
    directory=TEST_IMG_PATH,
    x_col='image',
    y_col=None,
    weight_col=None,
    target_size=(512, 512),
    batch_size=8,
    class_mode=None,
    shuffle='False')


class_names = {v: k for k, v in train_img.class_indices.items()}
class_names


_, ax = plt.subplots(2,4,figsize=(8,4))
ax = ax.flatten()
imgs, clss = next(iter(train_img))
for i in range(8):
    ax[i].imshow(imgs[i])
    ax[i].set_title(class_names[clss[i]])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()


input_shape = (512,512,3)
num_classes = 1

image_input = layers.Input(shape=input_shape)
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
x = effnet(image_input)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)  
x = layers.Dense(1024, activation='relu')(x) 
x = layers.Dense(num_classes,activation='sigmoid')(x)
model =  models.Model(inputs=image_input, outputs=x)
model.summary()


# Compiling the model with the combined loss and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy','AUC']) 

chkpnt_loss = ModelCheckpoint(
    'best_model_loss.keras',            # Path to save the model
    monitor='val_loss',         # Metric to monitor 
    verbose=1,                  # Print messages when saving the model
    save_best_only=True,        # Save only the best model (with highest metric)
    mode='min',                 
    save_weights_only=False,     # Save the entire model (not just weights)
)

chkpnt_auc = ModelCheckpoint(
    'best_model_auc.keras',            # Path to save the model
    monitor='val_AUC',         # Metric to monitor 
    verbose=1,                  # Print messages when saving the model
    save_best_only=True,        # Save only the best model (with highest metric)
    mode='max',                 # 'max' means the model with the highest metric score will be saved
    save_weights_only=False,     # Save the entire model (not just weights)
)




history = model.fit(train_img,
                    validation_data=val_img,
                    epochs=64,
                    class_weight=class_weight_dict,
                    callbacks=[chkpnt_loss,chkpnt_auc])


# Plot training & validation loss values
plt.figure(figsize=(8,6))
plt.subplot(221)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(222)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Metric Value')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(223)
plt.plot(history.history['AUC'], label='Train AUC')
plt.plot(history.history['val_AUC'], label='Validation AUC')
plt.title('Model AUC')
plt.ylabel('Metric Value')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()


imgs,clss = next(iter(val_img))
pred_prob = np.squeeze(model.predict(imgs,verbose=0))

fig, ax = plt.subplots(2,4,figsize=(8,4))
ax = ax.flatten()
for i in range(8):
    ax[i].imshow(imgs[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    # Get the actual class from one-hot encoding (if it's one-hot encoded)
    ax[i].set_title(f'Actual: {class_names[clss[i]]}\n'
                    f'Pred: {class_names[np.round(pred_prob[i])]} ({pred_prob[i]:0.2f})', fontsize=9)
plt.tight_layout()
plt.show()


pred_prob = np.squeeze(model.predict(test_img,verbose=0))
test_metadata.label = pred_prob
test_metadata.to_csv('submision.csv',index=False)

