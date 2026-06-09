import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(224, 224), 
                 shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.n = len(self.image_paths)
        self.indexes = np.arange(self.n)
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ) if augment else ImageDataGenerator(rescale=1./255)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in indexes]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(indexes), *self.img_size, 3))
        batch_y = np.array([self.labels[i] for i in indexes])
        
        # Load and preprocess images
        for i, path in enumerate(batch_paths):
            img = tf.keras.preprocessing.image.load_img(
                path, target_size=self.img_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.datagen.standardize(img)
            batch_x[i] = img
            
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def create_model():
    """Create and return the model architecture"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model


def train_model(train_data_path, labels_path, model_save_path='best_model.keras', 
                batch_size=32, epochs=20, validation_split=0.2):
    """Train the model using custom data generator"""
    # Read labels
    df_labels = pd.read_csv(labels_path)
    print(df_labels.columns)
    
    # Create full image paths
    image_paths = [os.path.join(train_data_path, img) for img in df_labels['image_name']]
    #labels = df_labels['label'].values
    labels = df_labels['label'].astype(float).values

    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=validation_split, random_state=42
    )
    
    # Create data generators
    train_generator = CustomDataGenerator(
        train_paths, train_labels, batch_size=batch_size, augment=True
    )
    val_generator = CustomDataGenerator(
        val_paths, val_labels, batch_size=batch_size, augment=False
    )
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history, train_generator, val_generator


def fine_tune_model(model, train_generator, val_generator, epochs=10):
    """Fine-tune the model by unfreezing some layers"""
    base_model = model.layers[0]
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    
    return model, history


def predict_test_images(model, test_dir, submission_path):
    """Generate predictions for test images"""
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    test_paths = [os.path.join(test_dir, f) for f in test_files]
    
    # Create test generator
    test_generator = CustomDataGenerator(
        test_paths,
        labels=np.zeros(len(test_paths)),  # Dummy labels
        batch_size=32,
        shuffle=False,
        augment=False
    )
    
    # Generate predictions
    predictions = model.predict(test_generator)
    
    # Create submission DataFrame
    df_submit = pd.DataFrame({
        'image': test_files,
        'label': predictions.flatten()
    })
    #labels = df_labels['label'].astype(float).values

    
    # Save submission file
    df_submit.to_csv(submission_path, index=False)


import os

# Print contents of base input directory
print("\nContents of input directory:")
print(os.listdir('../input'))

# Print contents of competition directory (if it exists)
try:
    print("\nContents of competition directory:")
    print(os.listdir('../input/cidaut'))
except FileNotFoundError:
    print("\nCouldn't find the competition directory. Please check the path.")


if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train"  # Directory containing training images
    TEST_DIR = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test"    # Directory containing test images
    LABELS_PATH = "/kaggle/input/ttaaaa/train.csv"  # Path to CSV file with labels
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 20
    FINE_TUNE_EPOCHS = 10
    #labels = df_labels['label'].astype(float).values

    
    # Train the model
    model, history, train_generator, val_generator = train_model(
        TRAIN_DIR,
        LABELS_PATH,
        model_save_path='best_model.keras',
        batch_size=BATCH_SIZE,
        epochs=INITIAL_EPOCHS
    )
    
    # Fine-tune the model
    model, ft_history = fine_tune_model(
        model,
        train_generator,
        val_generator,
        epochs=FINE_TUNE_EPOCHS
    )
    
    # Generate predictions and submission file
    predict_test_images(model, TEST_DIR, 'submission.csv')

