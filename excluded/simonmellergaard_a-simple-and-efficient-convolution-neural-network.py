!unzip -qq /kaggle/input/dogs-vs-cats/train.zip


import os, shutil, pathlib
from tensorflow.keras.utils import image_dataset_from_directory

original_dir = pathlib.Path("/kaggle/working/train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname,
                            dst=dir / fname)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)


from tensorflow import keras
from tensorflow.keras import layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)


inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)

x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

for size in [32, 64, 128, 256, 512]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(
        size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="model.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)


import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, title='', exclude_indexes=None):
    # Extract loss and accuracy
    history_dict = history.history
    train_loss = np.array(history_dict['loss'])
    val_loss = np.array(history_dict['val_loss'])
    train_accuracy = np.array(history_dict['accuracy'])
    val_accuracy = np.array(history_dict['val_accuracy'])

    # Convert exclude_indexes to a set of valid indices
    if exclude_indexes is None:
        exclude_indexes = set()
    else:
        exclude_indexes = set(exclude_indexes)

    # Build a mask to keep only desired indexes
    all_indexes = np.arange(len(train_loss))
    mask = np.array([i not in exclude_indexes for i in all_indexes])

    # Apply mask
    train_loss = train_loss[mask]
    val_loss = val_loss[mask]
    train_accuracy = train_accuracy[mask]
    val_accuracy = val_accuracy[mask]
    epochs = np.arange(1, len(history_dict['loss']) + 1)[mask]

    # Plotting
    plt.figure(figsize=(14, 5))
    plt.suptitle(f'{title}', fontsize=16)

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'o-', label='Training Loss', color = 'purple')
    plt.plot(epochs, val_loss, 'o-', label='Validation Loss', color = 'navy')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'o-', label='Training Accuracy', color = 'purple')
    plt.plot(epochs, val_accuracy, 'o-', label='Validation Accuracy', color = 'navy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_best_val_loss_and_accuracy(history):
    # Find the best model index
    history_dict = history.history
    best_val_loss_index = np.argmin(history_dict['val_loss'])
    best_val_loss = history_dict['val_loss'][best_val_loss_index]
    best_val_accuracy = history_dict['val_accuracy'][best_val_loss_index]

    # Print results
    print(f"Best Validation Loss: {'{0:.5g}'.format(best_val_loss)} at epoch {best_val_loss_index + 1}")
    print(f"Validation Accuracy at Best Loss: {'{0:.5g}'.format(100 * best_val_accuracy)}%")


plot_loss(history=history, title='Classification of cats and dogs with an Xception-like model')
print_best_val_loss_and_accuracy(history)


test_model = keras.models.load_model("model.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


!unzip -qq /kaggle/input/dogs-vs-cats/test1.zip


original_dir = pathlib.Path("/kaggle/working/test1")
new_base_dir = pathlib.Path("/kaggle/working/test_comp")
def make_subset2(subset_name, start_index, end_index):
    dir = new_base_dir / subset_name
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    fnames = [f"{i}.jpg" for i in range(start_index, end_index)]
    for fname in fnames:
        shutil.copyfile(src=original_dir / fname,
                        dst=dir / fname)
make_subset2("test_comp", start_index=1, end_index=12501)


dire = pathlib.Path("/kaggle/working/test_comp/test_comp")
test_set = image_dataset_from_directory(
    dire,
    labels=None,
    image_size=(180, 180),
    batch_size=32)


preds = test_model.predict(test_set)
predicted_labels = (preds > 0.5).astype(int)


import pandas as pd
predicted_labels = predicted_labels.tolist()
index = [i for i in range(1, 12501, 1)]
output = pd.DataFrame({'id': index,
                       'label': predicted_labels})
output.to_csv('submission.csv', index=False)

