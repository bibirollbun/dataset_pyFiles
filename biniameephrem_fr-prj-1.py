import os, math, json, gc, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input, Add, GlobalAveragePooling2D, SpatialDropout2D, Activation
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam




# Load data
df = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')


print(df.shape)


emotions = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}


# Display the head of the train DataFrame. 
df.head(10)


df['emotion'].value_counts()


plt.figure(figsize=(9, 8))


sns.countplot(x=df.emotion)
_ = plt.title('Label Distribution')
_ = plt.xticks(ticks=range(0, 7), labels=[emotions[i] for i in range(0, 7)], )


# Use original dataframe with 'pixels' and 'emotion'
sample = df.sample(n=10).reset_index()

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

plt.figure(figsize=(12, 6))

for i, row in sample.iterrows():
    # Convert pixel string to numpy array
    img = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)

    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(emotions[int(row['emotion'])])
        
    plt.axis('off')

plt.tight_layout()
plt.show()


X = np.array([np.fromstring(pixel, sep=' ') for pixel in df['pixels']], dtype='float32')
X = X.reshape(-1, 48, 48, 1)
X_rgb = np.repeat(X, 3, axis=-1)  # Convert grayscale to RGB

# Resize to match MobileNetV2 expected input
X_resized = tf.image.resize(X_rgb, [128, 128]).numpy()


# Label encoding and one-hot encoding
le = LabelEncoder()
y = le.fit_transform(df['emotion'])
y_cat = to_categorical(y, num_classes=7)


# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X_resized, y_cat, test_size=0.2, stratify=y, random_state=42)


print(y_train)



model = Sequential([
    Input(shape=(48, 48, 1)), 

    Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, kernel_initializer='he_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)




# Define data generators with augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)



# Load MobileNetV2 base model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

# Add custom classification head
inputs = Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=False)

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_gen.flow(X_train, y_train, batch_size=64),
    validation_data=val_gen.flow(X_val, y_val, batch_size=64),
    epochs=30,
    callbacks=[early_stop]
)



base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen.flow(X_train, y_train, batch_size=64),
    validation_data=val_gen.flow(X_val, y_val, batch_size=64),
    epochs=10,
    callbacks=[early_stop]
)



model.summary()


# preprocessing as training
val_eval = val_gen.flow(X_val, y_val, batch_size=64, shuffle=False)

# Evaluate & predict
val_loss, val_acc = model.evaluate(val_eval, verbose=0)
print(f"Validation -> loss: {val_loss:.4f}  acc: {val_acc:.4f}")

y_prob = model.predict(val_eval, verbose=0)
y_pred_classes = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_val, axis=1)

print(classification_report(y_true, y_pred_classes, digits=3, zero_division=0))


h = history_finetune.history 
plt.plot(h['accuracy'],     label='Train Acc')
plt.plot(h['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.title('Training vs Validation Accuracy'); plt.show()



model.save('FR_PRJ_1_model.h5')
pickle.dump(history, open(f'FR_PRJ_history.pkl', 'wb'))

