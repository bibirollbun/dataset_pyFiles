# Import libraries
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import warnings
warnings.filterwarnings('ignore')

# Print TensorFlow version
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# Load train.csv
try:
    train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
    print(f"Loaded train1.csv with {len(train_df)} rows")
except FileNotFoundError:
    print("Error: train1.csv not found. Ensure dataset is attached.")
    raise

# Define 14 condition labels
label_columns = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# Verify label columns exist
missing_cols = [col for col in label_columns if col not in train_df.columns]
if missing_cols:
    print(f"Error: Missing columns in train1.csv: {missing_cols}")
    raise KeyError(f"Missing columns: {missing_cols}")

# Split into train and validation (80/20)
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['No Finding']
)
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")


# Custom data generator for images
class ChestXRayGenerator(Sequence):
    def __init__(self, df, batch_size=32, img_size=(128, 128), is_test=False):
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_test = is_test
        self.label_columns = label_columns
        self.image_dir = '/kaggle/input/grand-xray-slam-division-a/train1/' if not is_test else '/kaggle/input/grand-xray-slam-division-a/test1/'

        # Verify image directory
        if not os.path.exists(self.image_dir):
            print(f"Error: Image directory {self.image_dir} not found.")
            raise FileNotFoundError(f"Directory {self.image_dir} missing.")

    def __len__(self):
        return (len(self.df) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.df))
        batch = self.df.iloc[start_idx:end_idx]
        images = []
        labels = []

        for _, row in batch.iterrows():
            img_path = os.path.join(self.image_dir, row['Image_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Image {img_path} not found, using zero array.")
                img = np.zeros(self.img_size)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            if not self.is_test:
                labels.append(row[self.label_columns].values.astype(np.float32))

        images = np.array(images, dtype=np.float32)
        if not self.is_test:
            labels = np.array(labels, dtype=np.float32)
            return images, labels
        return images

# Create generators
batch_size = 128
try:
    train_generator = ChestXRayGenerator(train_data, batch_size=batch_size)
    val_generator = ChestXRayGenerator(val_data, batch_size=batch_size)
    print("Train and validation generators created successfully.")
except Exception as e:
    print(f"Error creating generators: {e}")
    raise


# Build simple CNN model
def create_cnn_model(input_shape=(128, 128, 1), num_classes=14):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label
    ])
    return model

# Create and compile model
model = create_cnn_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC']
)
model.summary()


# Train for 1 epoch
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,
    verbose=1
)

# Print validation AUC-ROC
val_auc = history.history['val_AUC'][-1] if 'val_AUC' in history.history else 0.0
print(f"Validation AUC-ROC: {val_auc:.4f}")


# Load sample submission and test data
try:
    sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')
    print(f"Loaded sample_submission_1.csv with {len(sample_submission)} rows")
except FileNotFoundError:
    print("Error: sample_submission_1.csv not found.")
    raise

test_generator = ChestXRayGenerator(sample_submission, batch_size=batch_size, is_test=True)

# Predict on test set
predictions = []
for i in range(len(test_generator)):
    batch_images = test_generator[i]
    batch_preds = model.predict(batch_images, verbose=0)
    predictions.append(batch_preds)

# Combine predictions
predictions = np.vstack(predictions)
predictions = predictions[:len(sample_submission)]

# Create submission file
submission_df = sample_submission.copy()
submission_df[label_columns] = predictions
submission_df.to_csv('sample_submission.csv', index=False)
print("Submission file created: submission.csv")




