import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import keras

from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Sequential



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Dataset loaded')
print('shape of train images', x_train.shape)
print('shape of train labels', y_train.shape)
print(x_train.shape[0], 'number of training images')
print(x_test.shape[0], 'number of testing images')


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

random_indices = np.random.choice(len(x_train), size=3, replace=False)

plt.figure(figsize=(10, 4))
for i, index in enumerate(random_indices):
    image = x_train[index]
    label = y_train[index][0]
    
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.title(classes[label])
    plt.axis('off')

plt.tight_layout()
plt.show()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# CIFAR-10 channel mean and std for each channel
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range=0.1 
)

train_generator = datagen.flow(x_train, y_train, batch_size=64)


# Function to denormalize images for display purposes only
def denormalize(imgs):
    return imgs * std + mean

# Select random images and labels
sample_images = x_train[np.random.choice(len(x_train), size=5, replace=False)]
sample_labels = y_train[np.random.choice(len(x_train), size=5, replace=False)]

# Create a generator for just these samples
augmented_generator = datagen.flow(sample_images, sample_labels, batch_size=5, shuffle=False)
augmented_images, _ = next(augmented_generator)

sample_images_denorm = denormalize(sample_images)
augmented_images_denorm = denormalize(augmented_images)

plt.figure(figsize=(5 * 2, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.clip(sample_images_denorm[i], 0, 1))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(np.clip(augmented_images_denorm[i], 0, 1))
    plt.title("Augmented")
    plt.axis('off')

plt.tight_layout()
plt.show()



from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, ReLU, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


model_1 = Sequential([
    Input(shape=(32, 32, 3)),
    
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),


    # next, converting the image into a 1D vector (array)
    Flatten(),

    # a layer with 256 nodes
    Dense(256, activation='relu'),
    Dropout(0.7),

    # the output is a vector of probabilities for each class (like being 0.2 certain the digit is a 0)
    Dense(10, activation='softmax')
])


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
loss = CategoricalCrossentropy(from_logits=False)
optimizer = Adam(learning_rate=1e-3)

model_1.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)


history = model_1.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 64,
    validation_data=(x_test, y_test),
    epochs=1, 
    callbacks=[lr_scheduler, early_stopping]
)


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

y_pred_probs = model_1.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


model_2 = Sequential([
    Input(shape=(32, 32, 3)),
    
    Conv2D(32, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    Conv2D(32, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),
    
    Conv2D(64, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    Conv2D(64, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),

    Conv2D(128, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, 3, padding='same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),


    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    
    Dense(10, activation='softmax')
])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
loss = CategoricalCrossentropy(from_logits=False)
optimizer = Adam(learning_rate=1e-3)

model_2.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)


history = model_2.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 64,
    validation_data=(x_test, y_test),
    epochs=1, 
    callbacks=[lr_scheduler, early_stopping]
)


import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

y_pred_probs = model_2.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


model_3 = Sequential([
    Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'), 
    BatchNormalization(),

    Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),

    Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'), 
    BatchNormalization(),

    Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'), 
    BatchNormalization(),

    Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.35),

    Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'), 
    BatchNormalization(),

    Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.45),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# model_3 = Sequential()
# model_3.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())

# model_3.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())
# model_3.add(MaxPooling2D(pool_size=(2,2)))
# model_3.add(Dropout(0.2))

# model_3.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())

# model_3.add(Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())
# model_3.add(MaxPooling2D(pool_size=(2,2)))
# model_3.add(Dropout(0.25))

# model_3.add(Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())

# model_3.add(Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())
# model_3.add(MaxPooling2D(pool_size=(2,2)))
# model_3.add(Dropout(0.35))

# model_3.add(Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())

# model_3.add(Conv2D(256,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
# model_3.add(BatchNormalization())
# model_3.add(MaxPooling2D(pool_size=(2,2)))
# model_3.add(Dropout(0.45))

# model_3.add(Flatten())
# model_3.add(Dense(256, activation='relu'))
# model_3.add(BatchNormalization())
# model_3.add(Dropout(0.5))
# model_3.add(Dense(10, activation='softmax'))
model_3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
loss = CategoricalCrossentropy(from_logits=False)
optimizer = Adam(learning_rate=1e-3)

model_3.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)


history = model_3.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 64,
    validation_data=(x_test, y_test),
    epochs=50, 
    callbacks=[lr_scheduler, early_stopping]
)


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


y_pred_probs = model_3.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

y_pred_probs = model_3.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

wrong_indices = np.where(y_pred_classes != y_true)[0]
np.random.shuffle(wrong_indices)
wrong_images = wrong_indices[:6]

# Function to invert normalization, for visualisation purposes
def invert_normalization(img):
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

plt.figure(figsize=(10, 6))
for i, idx in enumerate(wrong_images):
    img = invert_normalization(x_test[idx])
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    # plt.title(f"Predicted: {class_names[y_pred_classes[idx]]}\nTrue: {class_names[y_true[idx]]}")
    plt.title(f"True: {class_names[y_true[idx]]}\nPredicted: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


y_pred_probs = model_3.predict(x_test)
y_pred_indices = np.argmax(y_pred_probs, axis=1)
y_pred_labels = [class_names[i] for i in y_pred_indices]

submission_df = pd.DataFrame({
    'id': np.arange(len(y_pred_labels)),
    'label': y_pred_labels
})

submission_df.to_csv('submission.csv', index=False)

