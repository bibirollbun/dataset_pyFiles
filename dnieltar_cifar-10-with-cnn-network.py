# Download with (x_train, y_train), (x_test, y_test) = cifar10.load_data() #was slow to download
# or 
# https://www.cs.toronto.edu/~kriz/cifar.html #as well was slow
# or 
# https://www.kaggle.com/competitions/cifar-10/data #faster, but filled with junk files in test (against cheating in the competition)


import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras.backend as K
import itertools
from functools import partial

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

if physical_devices:
    print(f"TensorFlow is using GPU: {physical_devices[0].name}")
else:
    print("No GPU found. TensorFlow is running on the CPU.")


object_mapping = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
RANDOM_SEED = 42


image_dir = r"datasets/train"
csv_path = r"datasets/trainLabels.csv"


labels_df = pd.read_csv(csv_path)
labels_df.sort_values(by="id", inplace=True)
labels_sorted = tf.convert_to_tensor(labels_df[["label"]], dtype=tf.string)


unique_labels = tf.convert_to_tensor(list(object_mapping.keys()), dtype=tf.string)


train_ids, val_ids, train_labels, val_labels = train_test_split(labels_df["id"], labels_df["label"], test_size=0.2, stratify=labels_df["label"], random_state=42)


train_ids = train_ids.values
val_ids = val_ids.values


@tf.function
def load_image(id):
    id_str = tf.strings.as_string(id)
    image_path = tf.strings.join([image_dir, '/', id_str, '.png'])
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        
    label = tf.argmax(tf.equal(unique_labels, labels_sorted[id-1])) ##because index starts from 1 at pictures
    label = tf.cast(label, tf.float16)
    label = tf.reshape(label, [])
    
    return image, label


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.GaussianNoise(0.01),
])


@tf.function
def apply_data_augmentation(image, label):
    return data_augmentation(image), label


train_dataset = tf.data.Dataset.from_tensor_slices(train_ids)
train_dataset = train_dataset.map(lambda id: load_image(id),
    num_parallel_calls=tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(val_ids)
val_dataset = val_dataset.map(lambda id: load_image(id), 
                      num_parallel_calls=tf.data.AUTOTUNE)


def configure_performance(ds, is_train_ds=True):
    if is_train_ds:
        ds = ds.cache()
        ds = ds.shuffle(1000)
        ds = ds.map(apply_data_augmentation, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.cache()
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = configure_performance(train_dataset)
val_dataset = configure_performance(val_dataset, is_train_ds=False)


image_batch, labels_batch = next(iter(train_dataset))


plt.figure(figsize=(6, 6))

for i in range(10):
  ax = plt.subplot(4, 3, i + 1)
  label_idx = tf.argmax(labels_batch.numpy() == i).numpy()
  plt.imshow(image_batch[label_idx])
  plt.title(f"Label {i}, {[objct for objct, value in object_mapping.items() if value ==i][0]}", fontweight="bold")
  plt.axis("off")

plt.tight_layout()






#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay 
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule for custom scheduling

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_target, warmup_steps, decay_steps, alpha=0.0):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate + (self.warmup_target - self.initial_learning_rate) * (step / self.warmup_steps)
        
        step_cosine = step - self.warmup_steps
        decay_steps_adjusted = self.decay_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * step_cosine / decay_steps_adjusted))
        decayed_lr = (1 - self.alpha) * cosine_decay + self.alpha
        decayed_lr = self.warmup_target * decayed_lr

        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decayed_lr
        )


# input=W, filter=F, padding=P, S=Stride, => (W−F+2P)/S+1


model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),


    # Flatten and Dense
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


lr_schedule = WarmupCosineDecay(
    initial_learning_rate=1e-5,
    warmup_target=0.001,
    warmup_steps=5000,
    decay_steps=62500,
    alpha=0.0
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
)


model.summary()


tf.keras.utils.plot_model(model)


def visualize_filter_responses(model, layer_index, input_image, nrows=int, ncols=int, figsize=(10, 6)):
    intermediate_model = tf.keras.Model(
        inputs=model.input, 
        outputs=model.layers[layer_index].output
    )
    
    responses = intermediate_model.predict(tf.expand_dims(input_image, 0))
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(min(64, responses.shape[-1])):
        row = i // 8
        col = i % 8
        axs[row, col].set_title(f"Filter:{i}")
        axs[row, col].imshow(responses[0, :, :, i], cmap='viridis')
        axs[row, col].axis('off')
        
    plt.suptitle(f"Convolution layer indexed: {layer_index} applied on a sample image")
    plt.tight_layout()
    plt.show()


sample_image = next(iter(train_dataset))[0][0]
plt.title("Sample Image")
plt.imshow(sample_image)
plt.show()


visualize_filter_responses(model, 2, sample_image, nrows=4, ncols=8)


visualize_filter_responses(model, 6, sample_image, nrows=8, ncols=8, figsize=(12, 12))


predictions = model.predict(val_dataset)


predicted_labels = tf.argmax(predictions, axis=1)
predicted_labels = tf.cast(predicted_labels, tf.float16)
prediction_confidence = tf.reduce_max(predictions, axis=1)


validation_imgs = []
actual_labels = []
for imgs, labels in val_dataset:
    validation_imgs.extend(imgs.numpy())
    actual_labels.extend(labels.numpy())

validation_imgs = tf.convert_to_tensor(validation_imgs)
actual_labels = tf.convert_to_tensor(actual_labels)


cm = confusion_matrix(actual_labels, predicted_labels)
accuracy = accuracy_score(actual_labels, predicted_labels)


plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(f"Validation Accuracy: {accuracy:.4f}")


precision = cm.diagonal() / cm.sum(axis=0)
recall = cm.diagonal() / cm.sum(axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)
output_nodes = 10

for i in range(output_nodes):
    print(f"Class {i}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-score: {f1_score[i]:.4f}")

print(f"\nMacro-average F1-score: {np.mean(f1_score):.4f}")


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(13, 5))
plt.suptitle("Accurate Predictions")

for i, ax in enumerate(axes.flat):
    random_index = np.random.choice(np.where(predicted_labels==actual_labels)[0])
    image = validation_imgs[random_index]
    fig = plt.figure
    ax.set_title(f"Predicted: {predicted_labels[random_index].numpy()}: {unique_labels[int(predicted_labels[random_index].numpy())].numpy().decode('utf8')}\n Confidence: {round(prediction_confidence[random_index].numpy()*100, 2)}%")
    ax.imshow(image)
    ax.set_axis_off()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(13, 5))
plt.suptitle("Wrong Predictions")

for i, ax in enumerate(axes.flat):
    random_index = np.random.choice(np.where(predicted_labels!=actual_labels)[0])
    image = validation_imgs[random_index]
    fig = plt.figure
    ax.set_title(f"Predicted Label: {predicted_labels[random_index].numpy()}: {unique_labels[int(predicted_labels[random_index].numpy())].numpy().decode('utf8')}\n Confidence: {round(prediction_confidence[random_index].numpy()*100, 2)}%")
    ax.imshow(image)
    ax.set_axis_off()

plt.tight_layout()
plt.show()





# Just for reproducibility purpose
random_splits_seeds = [10, 523, 3210, 5623113, 1244]


def return_model():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    lr_schedule = WarmupCosineDecay(
    initial_learning_rate=1e-5,
    warmup_target=0.001,
    warmup_steps=5000,
    decay_steps=62500,
    alpha=0.0
    )

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    
    return model


accuracies = []
for random_state in random_splits_seeds:

    train_ids, val_ids, train_labels, val_labels = train_test_split(labels_df["id"], labels_df["label"], test_size=0.2, stratify=labels_df["label"], random_state=random_state)
    train_ids = train_ids.values
    val_ids = val_ids.values

    train_dataset = tf.data.Dataset.from_tensor_slices(train_ids)
    train_dataset = train_dataset.map(lambda id: load_image(id),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.map(apply_data_augmentation, 
                                    num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_ids)
    val_dataset = val_dataset.map(lambda id: load_image(id), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = configure_performance(train_dataset, is_train_ds=True)
    val_dataset = configure_performance(val_dataset, is_train_ds=False)

    model = return_model()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
    )
    
    predictions = model.predict(val_dataset)

    predicted_labels = tf.argmax(predictions, axis=1)
    predicted_labels = tf.cast(predicted_labels, tf.float16)

    actual_labels = []

    for _, labels in val_dataset:
        actual_labels.extend(labels.numpy())

    actual_labels = tf.convert_to_tensor(actual_labels)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print(f"Current Fold Accuracy: {round(accuracy, 4)}")
    accuracies.append(accuracy)    
    del val_dataset, model



print(f"5 Fold avg. accuracy: {round(np.mean(accuracies)*100, 2)}%")





train_ids = labels_df.id.values
train_dataset = tf.data.Dataset.from_tensor_slices(train_ids)
train_dataset = train_dataset.map(lambda id: load_image(id),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.map(apply_data_augmentation, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = configure_performance(train_dataset)


model = return_model()
model.fit(
    train_dataset,
    epochs=50,
)


# Import test img-s
image_dir = r"datasets/test"


@tf.function
def load_image_test(id):
    id_str = tf.strings.as_string(id)
    image_path = tf.strings.join([image_dir, '/', id_str, '.png'])
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        
    return image


test_ids = np.arange(1, 300001)


test_dataset = tf.data.Dataset.from_tensor_slices(test_ids)
test_dataset = test_dataset.map(lambda id: load_image_test(id), 
                        num_parallel_calls=tf.data.AUTOTUNE)


test_dataset = configure_performance(test_dataset, is_train_ds=False)


image_batch = next(iter(test_dataset))


plt.figure(figsize=(6, 6))

for i in range(10):
  ax = plt.subplot(4, 3, i + 1)
  plt.imshow(image_batch[i])
  plt.axis("off")

plt.tight_layout()



predictions = model.predict(test_dataset)


predicted_labels = tf.argmax(predictions, axis=1)
predicted_labels = tf.cast(predicted_labels, tf.int32)


submission = pd.DataFrame(np.arange(1, 300001), columns=["id"])
submission["label"] = predicted_labels.numpy()


submission.replace({"label": dict(zip(object_mapping.values(), object_mapping.keys()))}, inplace=True)


submission.to_csv("submission.csv", index=False)







