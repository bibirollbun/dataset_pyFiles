#  import libraries
import os

# data handling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder as le

# Deep learning
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Exploratory Data Analysis (EDA)
# Load the dataset
df = pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
df.head()


# data shape
df.shape


# unique values in label
df['label'].unique().tolist()


# unique values in variety
df['variety'].unique().tolist()


# stats for Paddy's age
df['age'].describe()


# number varieties present using Seaborn
plt.figure(figsize=(10, 5))
sns.countplot(x='variety', data=df, palette='Set1')
plt.title('Number of Varieties')
plt.xticks(rotation=45)
plt.show()


# Make plot of images
# Extract 5 normal and 5 dead images for ADT45
normal = df[df['label'] == 'normal']
normal = normal[normal['variety'] == 'ADT45']
five_normal = normal.sample(5)['image_id'].tolist()

dead = df[df['label'] == 'dead_heart']
dead = dead[dead['variety'] == 'ADT45']
five_dead = dead.sample(5)['image_id'].tolist()

# Define path to images
path = '/kaggle/input/paddy-disease-classification/train_images/'
col = 5

# plot images
plt.figure(figsize=(20, 10))
for i, image_loc in enumerate(np.concatenate((five_normal, five_dead))):
    plt.subplot(10 // col + 1, col, i + 1)

    if i < 5:
        image = plt.imread(path + "normal/" + image_loc)
        plt.title('Normal')
    else:
        image = plt.imread(path + "dead_heart/" + image_loc)
        plt.title('Dead')
    plt.imshow(image)


# Extract 1 image_id for each label
# Create a list of unique labels
temp_path = df['label'].unique().tolist()
# append the path (dataset path) to temp_path
path = []
for i in temp_path:
    path.append('/kaggle/input/paddy-disease-classification/train_images/' + i)

# Extract 1 image_id for each label
image_id = []
for i in temp_path:
    image_id.append(df[df['label'] == i].sample(1)['image_id'].values[0])

# append image_id to path
path = [path[i] + '/' + image_id[i] for i in range(len(image_id))]

# plot images
plt.figure(figsize=(20, 10))
for i, image_loc in enumerate(path):
    plt.subplot(10 // col + 1, col, i + 1)
    image = plt.imread(image_loc)
    plt.title(temp_path[i])
    plt.imshow(image)


# define parameters
batch_size = 16
img_height = 224
img_width = 224
data_dir = '/kaggle/input/paddy-disease-classification/train_images/'


# Train dataset
train_df = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123,
    )
class_names = train_df.class_names  # List of class labels (index -> label mapping)


# Validation dataset
val_df = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123)


# image dimension per batch
for image_batch, labels_batch in train_df:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# normalization
normalized_train = train_df.map(lambda x, y: (normalization_layer(x), y))
normalized_val = val_df.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_train))
# pixel values after normalization
print(np.min(image_batch[0]), np.max(image_batch[0]))


# Autotuning the data loading
AUTOTUNE = tf.data.AUTOTUNE
train_df = train_df.cache().prefetch(buffer_size=AUTOTUNE)
val_df = val_df.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(temp_path), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


%%time
early_stop = EarlyStopping(patience=12)
history = model.fit(train_df, validation_data= val_df, epochs=100, callbacks=[early_stop])

# evaluate the model
loss = model.evaluate(val_df)

plt.plot(history.history['loss'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Val Accuracy')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.show()


# plot confusion matrix for validation data
predictions = model.predict(val_df)
predictions = np.argmax(predictions, axis=1)

# Extract true labels from the validation dataset
true_labels = []
for _, labels in val_df:
	true_labels.extend(labels.numpy())

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(true_labels, predictions, target_names=temp_path))


# Image size and batch_size should match those from your training pipeline
img_height = 224
img_width = 224
batch_size = 32
test_dir = '/kaggle/input/paddy-disease-classification/test_images/'

test_df = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# Predict the Test Data
predictions = model.predict(test_df)  # shape: (num_test_images, num_classes)

# Convert to predicted class indices
predicted_indices = np.argmax(predictions, axis=1)

# Map predicted indices to class names
predicted_labels = [class_names[idx] for idx in predicted_indices]


# CSV creation in (image_id, label) format
file_paths = test_df.file_paths  # List of full paths to each test image
# Extract only the file name (e.g., "abc.jpg") from the full path
image_names = [os.path.basename(path) for path in file_paths]

submission_df = pd.DataFrame({
    "image_id": image_names,
    "label": predicted_labels
})

# For viewing
print(submission_df.head())

# For submission
submission_df.to_csv("submission.csv", index=True)

