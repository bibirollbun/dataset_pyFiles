import tensorflow as tf
from tensorflow import keras
import shutil
from tqdm.notebook import tqdm
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks, optimizers, losses, metrics



labels_df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')


labels_df.head()


labels_df['label'].value_counts()


import matplotlib.pyplot as plt

counts = labels_df['label'].value_counts() 

plt.figure(figsize=(8,5))
plt.bar(counts.index, counts.values) 
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Count of each label')
plt.show()



CSV_PATH = '/kaggle/input/cassava-leaf-disease-classification/train.csv'
SOURCE_IMAGE_DIR = '/kaggle/input/cassava-leaf-disease-classification/train_images'
OUTPUT_DIR = '/kaggle/working/data'

try:
    train_df = pd.read_csv(CSV_PATH)
    print("âœ… Successfully loaded train.csv.")
    print(f"DataFrame contains {len(train_df)} records.")
except FileNotFoundError:
    print(f"Error: Could not find train.csv at {CSV_PATH}")
    train_df = pd.DataFrame()

if not train_df.empty:
    print(f"\nCreating base directory at: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_labels = sorted(train_df['label'].unique())
    
    print(f"Found unique labels: {unique_labels}")
    for label in unique_labels:
        label_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(label_dir, exist_ok=True)
    
    print(f"âœ… Created {len(unique_labels)} subdirectories for each label.")

    print("\nStarting to copy images. This may take a few minutes...")
    
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        image_filename = row['image_id']
        label = str(row['label'])
        
        source_path = os.path.join(SOURCE_IMAGE_DIR, image_filename)
        destination_path = os.path.join(OUTPUT_DIR, label, image_filename)
        
        shutil.copy(source_path, destination_path)
        
    print("\nğŸ�‰ Successfully sorted all images into their label folders!")

    print("\nVerifying the new directory structure...")
    created_dirs = sorted(os.listdir(OUTPUT_DIR))
    print(f"Folders in output directory: {created_dirs}")

    sample_label = str(unique_labels[0])
    sample_dir_path = os.path.join(OUTPUT_DIR, sample_label)
    num_files = len(os.listdir(sample_dir_path))
    print(f"Found {num_files} images in the '{sample_label}' folder.")


DATA_DIR = '/kaggle/working/data'
IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("âœ… Datasets created successfully.")

class_names = train_ds.class_names
print(f"\nInferred class names: {class_names}")

print("\nVerifying a batch from the training dataset:")
for images, labels in train_ds.take(1):
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("\nâœ… Performance optimization applied to both datasets.")


class_map = {
    '0': 'Cassava Bacterial Blight (CBB)',
    '1': 'Cassava Brown Streak Disease (CBSD)',
    '2': 'Cassava Green Mottle (CGM)',
    '3': 'Cassava Mosaic Disease (CMD)',
    '4': 'Healthy'
}

def plot_class_samples(label_index, dataset = train_ds, class_map = class_map, num_images=9):
    target_label_name = class_map[str(label_index)]
    
    filtered_ds = dataset.unbatch().filter(lambda image, label: label == label_index)
    
    images_to_plot = [image.numpy() for image, label in filtered_ds.take(num_images)]
    
    if images_to_plot:
        print(f"Displaying {len(images_to_plot)} sample images for class: {target_label_name}")
        
        grid_size = int(np.ceil(np.sqrt(len(images_to_plot))))
        plt.figure(figsize=(grid_size * 3, grid_size * 3))
        
        for i, img in enumerate(images_to_plot):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(img.astype("uint8"))
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Could not find any images for label {label_index} ({target_label_name}).")


plot_class_samples(0)


plot_class_samples(1)


plot_class_samples(2)


plot_class_samples(3)


plot_class_samples(4)


IMG_SIZE = 300
NUM_CLASSES = 5
EPOCHS = 5

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
], name="data_augmentation")

base_model = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = keras.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()



model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    'best_cassava_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


print("\nStarting model training")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, checkpoint]
)


plt.figure(figsize=(8, 6))
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


