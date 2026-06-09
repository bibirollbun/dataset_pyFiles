# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# Define paths
BASE_FOLDER = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
TRAIN_PATH = os.path.join(BASE_FOLDER, "Train")  # Use os.path.join for path construction
LABELS_CSV_FILEPATH = os.path.join(BASE_FOLDER, "train.csv")  

NUM_IMAGES = 720
BATCH_SIZE = 8
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

# Function to load labels
def load_labels_from_csv_file():
    with open(LABELS_CSV_FILEPATH) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # Skip the header row
        image_label_pairs = [(row[0], 1 if row[1] == "real" else 0) for row in reader]
        image_label_pairs = sorted(image_label_pairs, key=lambda t: t[0])
        labels = [t[1] for t in image_label_pairs]
        return labels

# Function to create DataFrame
def create_df_from_csv_and_folder(csv_filepath, folder_path):
    with open(csv_filepath) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        image_label_pairs = [(row[0], 1 if row[1] == "real" else 0) for row in reader]

    df = pd.DataFrame(image_label_pairs, columns=['image', 'label'])
    df['image'] = df['image'].astype(str)
    df['filepath'] = df['image'].apply(lambda img: os.path.join(folder_path, img))
    return df

# Create and split the DataFrame
train_df = create_df_from_csv_and_folder(LABELS_CSV_FILEPATH, TRAIN_PATH)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=100)

def load_and_preprocess_image(image_path, label, augment=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Or decode_png
    img = tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_crop(img, size=[224, 224, 3])

    # Normalize (adjust mean and std if needed)
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    img = (img - mean) / std

    return img, label

# Create tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath'], train_df['label']))
val_dataset = tf.data.Dataset.from_tensor_slices((val_df['filepath'], val_df['label']))


# Map the load_and_preprocess_image function
train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_image(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: load_and_preprocess_image(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE)


# Batch and shuffle
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(buffer_size=len(train_df))
val_dataset = val_dataset.batch(BATCH_SIZE)


import matplotlib.pyplot as plt

# Take the first 10 elements from the train_dataset
data_to_plot = train_dataset.take(10)  

# Iterate through the elements and plot images with labels
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 images
axes = axes.flatten()  # Flatten the axes array for easier iteration

for i, (image_batch, label_batch) in enumerate(data_to_plot):
    # Iterate through images in the batch
    for j, image in enumerate(image_batch):
        # Only plot 10 images
        if i * BATCH_SIZE + j < 10:
            axes[i * BATCH_SIZE + j].imshow(image)  # Display individual image
            axes[i * BATCH_SIZE + j].set_title(f"Label: {label_batch[j].numpy()}")  # Display the label as the title
            axes[i * BATCH_SIZE + j].axis("off")  # Turn off axis ticks
        else:
            break  # Exit the inner loop if we've plotted 10 images

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


 import tensorflow as tf
 from tensorflow.keras.applications import EfficientNetV2L
 from tensorflow.keras.models import Model
 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
 from tensorflow.keras.optimizers import Adam
 from tensorflow.keras.regularizers import l2

 IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
 # Load pre-trained EfficientNetV2L
 base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

 # Add custom classification layers
 x = base_model.output
 x = GlobalAveragePooling2D()(x)
 x = Dropout(0.7)(x)
 x = Dense(2, activation='softmax', kernel_regularizer=l2(0.1))(x)  # 2 classes, softmax activation

 # Create final model
 model = Model(inputs=base_model.input, outputs=x)

 # Create an Adam optimizer
 optimizer = Adam(learning_rate=1e-4)

#l # Compile the model
 model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping 
import time


start_time = time.time()
# Define the learning rate (lr)
lr = 1e-4 
gamma = 0.9 



# Define loss, optimizer, and scheduler
loss_fn = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=lr)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,           # Number of epochs with no improvement after which to stop
    restore_best_weights=True  # Restore the weights of the best epoch
)

def scheduler(epoch, lr):
    return lr * gamma

lr_scheduler = LearningRateScheduler(scheduler)

history = model.fit(
    train_dataset,
    epochs=50,  # Adjust as needed
    validation_data=val_dataset,
    callbacks=[lr_scheduler,early_stopping]  # Include the scheduler
)

end_time = time.time()
print(f"Training time 16: {end_time - start_time} seconds")


loss, accuracy = model.evaluate(val_dataset)
print('Validation accuracy:', accuracy)
train_accuracy = history.history['accuracy'][-1]  
print(f"Training accuracy: {train_accuracy}")
print(f"Validation loss: {loss}")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# ... (your existing code for loading and preprocessing data, model training, etc.) ...

# Get predictions on the validation dataset
y_pred_probs = model.predict(val_dataset)  # Get predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels (0 or 1)

# Get true labels from the validation dataset
y_true = np.concatenate([y for x, y in val_dataset], axis=0)  # Extract true labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
test_image_folder = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test" 
test_images = os.listdir(test_image_folder)

submission = []


for test_image in test_images:
    img_path = os.path.join(test_image_folder, test_image)
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img) / 255.0           
    img_array = np.expand_dims(img_array, axis=0)    
    

    prediction = model.predict(img_array)
    label = 1 if prediction[0][0] > 0.5 else 0 
    

    submission.append({"image": test_image, "label": label})


submission_df = pd.DataFrame(submission)

# save to CSV
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)

print(f"Submission file saved to {submission_csv_path}")
submission_df.head()




