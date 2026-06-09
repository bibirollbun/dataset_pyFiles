# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


#Remove Warnings
import warnings
warnings.filterwarnings('ignore')


!unzip -q "/kaggle/input/dogs-vs-cats/train.zip"


!unzip -q "/kaggle/input/dogs-vs-cats/test1.zip"


def prepare_data(train_path, val_size=0.15, test_size=0.15, random_state=42):

    train_filenames = os.listdir(train_path)
    train_categories = [1 if filename.split(".")[0] == 'dog' else 0 for filename in train_filenames]

    df = pd.DataFrame({
        'filename': train_filenames,
        'category': train_categories
    })

    
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), stratify=df["category"], random_state=random_state)

    val_ratio = val_size / (val_size + test_size)  
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio), stratify=temp_df["category"], random_state=random_state)

    return train_df, val_df, test_df


train_path = "/kaggle/working/train"
train_df, val_df, test_df = prepare_data(train_path)
print(f"Total Training Images: {len(train_df)}")
print(f"Total Validation Images: {len(val_df)}")
print(f"Total Test Images: {len(test_df)}")


train_df.head()


val_df.head()


test_df.head()


def subsampled_dataset(df, source_path, base_dir, dataset_type):
    dogs_dir = os.path.join(base_dir, dataset_type, "dogs")
    cats_dir = os.path.join(base_dir, dataset_type, "cats")
    os.makedirs(dogs_dir, exist_ok=True)
    os.makedirs(cats_dir, exist_ok=True)
    source_path = Path(source_path)

    for category, num, folder in [("cat", 0, cats_dir), ("dog", 1, dogs_dir)]:
        for i, filename in enumerate(df[df["category"] == num]["filename"]):
            src = source_path / filename
            dst = os.path.join(folder, f"{category}{i}.jpg") 
            shutil.copyfile(src, dst)




base_dir = "cats_vs_dogs"
subsampled_dataset(train_df, train_path, base_dir, "train")


subsampled_dataset(val_df, train_path, base_dir, "validation")


subsampled_dataset(test_df, train_path, base_dir, "test")


cats = len(os.listdir("/kaggle/working/cats_vs_dogs/train/cats"))
dogs = len(os.listdir("/kaggle/working/cats_vs_dogs/train/dogs"))
cats+dogs


def display_images(df, image_path, num_images=6):

    sample_data = df.sample(n=num_images).reset_index(drop=True)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        filename = sample_data.loc[i, "filename"]
        img = Image.open(os.path.join(image_path, filename))

        img = img.resize((128, 128))
        
        label = "Dog" if sample_data.loc[i, "category"] == 1 else "Cat"

        ax.imshow(img)
        ax.set_title(label, fontsize=20)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


display_images(train_df, train_path)


new_base_dir = Path(base_dir)


#Training
train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=42
)


#Validation
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=42
)


#Test
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=42
)


for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)   
    print("labels batch shape:", labels_batch.shape)
    break



inputs = keras.Input(shape=(180, 180, 3))  
x = layers.Rescaling(1./255)(inputs)     

# Conv Block 1
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv Block 2
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv Block 3
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv Block 4
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv Block 5
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)

# Flatten and Dense Layers
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(loss="binary_crossentropy",
      optimizer="rmsprop",
      metrics=["accuracy"])


callbacks = [
keras.callbacks.ModelCheckpoint(
 filepath="convnet_from_scratch.keras",
 save_best_only=True,
 monitor="val_loss")
 ]


history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)


model.summary()


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset) 
print(f"Test accuracy: {test_acc:.3f}")


# Define the model with regularization
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs)

# Conv Block 1
x = layers.Conv2D(filters=32, kernel_size=3, activation=None)(x)  
x = layers.BatchNormalization()(x)  # Batch normalization
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.3)(x) # Dropout

# Conv Block 2
x = layers.Conv2D(filters=64, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 3
x = layers.Conv2D(filters=128, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 4
x = layers.Conv2D(filters=256, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 5
x = layers.Conv2D(filters=256, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# Flatten and Dense Layers
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)  
outputs = layers.Dense(1, activation="sigmoid")(x)

model1 = keras.Model(inputs=inputs, outputs=outputs)


model1.compile(loss="binary_crossentropy",
      optimizer="rmsprop",
      metrics=["accuracy"])


callbacks = [
keras.callbacks.ModelCheckpoint(
 filepath="convnet_from_scratch1.keras",
 save_best_only=True,
 monitor="val_loss")
 ]


history = model1.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


def plot_confusion_matrix(model, test_dataset, class_names=["cat", "dog"]):
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        # Convert probabilities to binary predictions (threshold = 0.5)
        preds_binary = (preds > 0.5).astype(int).flatten()
        y_true.extend(labels.numpy())
        y_pred.extend(preds_binary)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (cat as cat): {tn}")
    print(f"False Positives (cat as dog): {fp}")
    print(f"False Negatives (dog as cat): {fn}")
    print(f"True Positives (dog as dog): {tp}")


test_model = keras.models.load_model("convnet_from_scratch1.keras")
test_loss, test_acc = test_model.evaluate(test_dataset) 
print(f"Test accuracy: {test_acc:.3f}")


plot_confusion_matrix(keras.models.load_model("convnet_from_scratch1.keras"), test_dataset)


def display_predictions(model, test_dataset, class_names=["cat", "dog"], num_images=6):

    images, labels = next(iter(test_dataset))
    
    # Predict probabilities
    preds = model.predict(images, verbose=0)
    preds_binary = (preds > 0.5).astype(int).flatten()
    
    # Convert to numpy arrays
    images = images.numpy()
    labels = labels.numpy()
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns
        plt.imshow(images[i].astype("uint8"))  # Assuming images are normalized
        actual_label = class_names[labels[i]]
        pred_label = class_names[preds_binary[i]]
        # Set title color: green if correct, red if wrong
        title_color = "green" if actual_label == pred_label else "red"
        plt.title(f"Actual: {actual_label}\nPred: {pred_label}", color=title_color, fontsize=12)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


display_predictions(keras.models.load_model("convnet_from_scratch1.keras"), test_dataset)


data_augmentation = keras.Sequential(
    [
     layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.1),
     layers.RandomZoom(0.2),
    ])


plt.figure(figsize=(15, 15)) 
for images, _ in train_dataset.take(1):   
    for i in range(9):
       augmented_images = data_augmentation(images)  
       ax = plt.subplot(3, 3, i + 1)
       plt.imshow(augmented_images[0].numpy().astype("uint8"))   
       plt.axis("off")


# Define the model with regularization
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

# Conv Block 1
x = layers.Conv2D(filters=32, kernel_size=3, activation=None)(x)  
x = layers.BatchNormalization()(x)  # Batch normalization
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.3)(x) # Dropout

# Conv Block 2
x = layers.Conv2D(filters=64, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 3
x = layers.Conv2D(filters=128, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 4
x = layers.Conv2D(filters=256, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Conv Block 5
x = layers.Conv2D(filters=256, kernel_size=3, activation=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# Flatten and Dense Layers
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)  
outputs = layers.Dense(1, activation="sigmoid")(x)

model2 = keras.Model(inputs=inputs, outputs=outputs)


model2.compile(loss="binary_crossentropy",
      optimizer="rmsprop",
      metrics=["accuracy"])


callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="convnet_from_scratch_with_augmentation.keras",
    save_best_only=True,
     monitor="val_loss")
  ]


history = model2.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


plot_confusion_matrix(keras.models.load_model("convnet_from_scratch_with_augmentation.keras"), test_dataset)


display_predictions(keras.models.load_model("convnet_from_scratch_with_augmentation.keras"), test_dataset)


conv_base  = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3))


conv_base.summary()


 conv_base.trainable = False


inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)          
x = keras.applications.vgg16.preprocess_input(x)   
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model3 = keras.Model(inputs, outputs)



model3.compile(loss="binary_crossentropy",
      optimizer="rmsprop",
      metrics=["accuracy"])


callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="feature_extraction_with_data_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")
 ]


history = model3.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


test_model = keras.models.load_model("feature_extraction_with_data_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


plot_confusion_matrix(keras.models.load_model("feature_extraction_with_data_augmentation.keras"), test_dataset)


display_predictions(keras.models.load_model("feature_extraction_with_data_augmentation.keras"), test_dataset)


conv_base.trainable = True
for layer in conv_base.layers[:-8]:
    layer.trainable = False


model3.compile(loss="binary_crossentropy",
      optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
      metrics=["accuracy"])


callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="fine_tuning.keras",
    save_best_only=True,
    monitor="val_loss")
 ]


history = model3.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset) 
print(f"Test accuracy: {test_acc:.3f}")


plot_confusion_matrix(keras.models.load_model("fine_tuning.keras"), test_dataset)


display_predictions(keras.models.load_model("fine_tuning.keras"), test_dataset)


model = keras.models.load_model("fine_tuning.keras")
test_dir = "/kaggle/working/test1"

def load_test_images(test_dir, image_size=(180, 180)):
    image_ids = []
    images = []
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            image_ids.append(int(filename.split(".")[0]))
    return np.array(images), image_ids

test_images, test_ids = load_test_images(test_dir)
predictions = model.predict(test_images, batch_size=32, verbose=1)
predictions = (predictions > 0.5).astype(int).flatten()

submission_df = pd.DataFrame({"id": test_ids, "label": predictions})
submission_df = submission_df.sort_values(by="id")
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

