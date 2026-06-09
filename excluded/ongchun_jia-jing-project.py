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


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam







#dataset
# Set up the paths for Kaggle dataset
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'


def extract_frames(video_path, frame_rate=30):
    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # Capture every `frame_rate`-th frame
        if int(cap.get(1)) % frame_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return np.array(frames)



# Preprocess the frames: resize and normalize
def preprocess_frames(frames, target_size=(128, 128)):
    print(f"Preprocessing {len(frames)} frames: resizing and normalizing")
    frames_resized = [cv2.resize(frame, target_size) for frame in frames]
    frames_normalized = np.array(frames_resized) / 255.0  # Normalize pixel values
    print(f"Preprocessing completed. Shape of processed frames: {frames_normalized.shape}")
    return frames_normalized



# Load and preprocess data
def load_data_from_directory(directory_path, frame_rate=30, target_size=(128, 128)):
    print(f"Loading data from directory: {directory_path}")
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.mp4')]
    frames = []
    labels = []
    
    for file_path in file_paths:
        video_frames = extract_frames(file_path, frame_rate)
        video_frames = preprocess_frames(video_frames, target_size)
        frames.extend(video_frames)
        
        # Labels: 0 for real, 1 for fake (adjust based on directory structure or metadata)
        label = 0 if 'real' in file_path else 1
        labels.extend([label] * len(video_frames))
    
    print(f"Loaded {len(frames)} frames from {len(file_paths)} videos.")
    return np.array(frames), np.array(labels)



def create_model(input_shape=(128, 128, 3)):
    model = Sequential()
    
    # First Conv Layer with L2 Regularization and Batch Normalization
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01), name="conv1"))
    model.add(BatchNormalization())  # Batch Normalization
    model.add(MaxPooling2D((2, 2), name="maxpool1"))
    
    # Second Conv Layer with L2 Regularization and Dropout
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), name="conv2"))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(MaxPooling2D((2, 2), name="maxpool2"))
    
    model.add(Flatten(name="flatten"))
    
    # Dense Layer with Dropout
    model.add(Dense(128, activation='relu', name="dense1"))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Dense(1, activation='sigmoid', name="output"))  # Binary classification: real or fake
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



from sklearn.model_selection import train_test_split


# Example usage
def main():
    # Load training data
    X_train, y_train = load_data_from_directory(train_dir, frame_rate=30, target_size=(128, 128))
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Apply augmentation to the training data
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    # Create the CNN model
    model = create_model()

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with augmented data and store the history
    history = model.fit(train_generator, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc}")

    # Plot training and validation loss and accuracy
    import matplotlib.pylab as plt

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()




import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up the paths for Kaggle dataset
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'

# Function to extract frames
def extract_frames(video_path, frame_rate=30):
    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return np.array(frames)

# Preprocess the frames: resize and normalize
def preprocess_frames(frames, target_size=(128, 128)):
    print(f"Preprocessing {len(frames)} frames: resizing and normalizing")
    frames_resized = [cv2.resize(frame, target_size) for frame in frames]
    frames_normalized = np.array(frames_resized, dtype=np.float32) / 255.0
    print(f"Preprocessing completed. Shape of processed frames: {frames_normalized.shape}")
    return frames_normalized

# Load and preprocess data
def load_data_from_directory(directory_path, frame_rate=30, target_size=(128, 128)):
    print(f"Loading data from directory: {directory_path}")
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.mp4')]
    frames = []
    labels = []
    
    for file_path in file_paths:
        video_frames = extract_frames(file_path, frame_rate)
        video_frames = preprocess_frames(video_frames, target_size)
        frames.extend(video_frames)
        
        label = 0 if 'real' in file_path else 1
        labels.extend([label] * len(video_frames))
    
    print(f"Loaded {len(frames)} frames from {len(file_paths)} videos.")
    return np.array(frames), np.array(labels)

# Model creation
def create_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01), name="conv1", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D((2, 2), name="maxpool1"))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.01), name="conv2"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2), name="maxpool2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation='relu', name="dense1"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name="output"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Predict real or fake for a single video
def predict_video(video_path, model, frame_rate=30, target_size=(128, 128)):
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path, frame_rate=frame_rate)
    if len(frames) == 0:
        print("No frames extracted. Skipping this video.")
        return "Error: No frames"
    preprocessed_frames = preprocess_frames(frames, target_size=target_size)
    predictions = model.predict(preprocessed_frames)
    mean_prediction = np.mean(predictions.flatten())
    print(f"Mean prediction score: {mean_prediction}")
    video_label = "Fake" if mean_prediction >= 0.5 else "Real"
    return video_label

# Main function
def main():
    # Load and preprocess training data
    X_train, y_train = load_data_from_directory(train_dir, frame_rate=30, target_size=(128, 128))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Create model and train
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])

    # Evaluate model
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation accuracy: {val_acc}")

    # Confusion matrix and classification report
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    print("Classification Report:\n", classification_report(y_val, y_pred))
    conf_matrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Test on a single video
    test_video_path = os.path.join(test_dir, '/kaggle/input/deepfake-detection-challenge/test_videos/aassnaulhq.mp4')  # Change this to your test video path
    video_label = predict_video(test_video_path, model)
    print(f"The video '{os.path.basename(test_video_path)}' is predicted to be: {video_label}")

    # Batch prediction on all test videos
    video_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.mp4')]
    for video_path in video_files:
        video_label = predict_video(video_path, model)
        print(f"Video: {os.path.basename(video_path)} - Predicted as: {video_label}")

# Run the main function
if __name__ == "__main__":
    main()



import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up the paths for Kaggle dataset
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'

# Function to extract frames
def extract_frames(video_path, frame_rate=30):
    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return np.array(frames)

# Preprocess the frames: resize and normalize
def preprocess_frames(frames, target_size=(128, 128)):
    print(f"Preprocessing {len(frames)} frames: resizing and normalizing")
    frames_resized = [cv2.resize(frame, target_size) for frame in frames]
    frames_normalized = np.array(frames_resized, dtype=np.float32) / 255.0
    print(f"Preprocessing completed. Shape of processed frames: {frames_normalized.shape}")
    return frames_normalized

# Load and preprocess data
def load_data_from_directory(directory_path, frame_rate=30, target_size=(128, 128)):
    print(f"Loading data from directory: {directory_path}")
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.mp4')]
    frames = []
    labels = []
    
    for file_path in file_paths:
        video_frames = extract_frames(file_path, frame_rate)
        video_frames = preprocess_frames(video_frames, target_size)
        frames.extend(video_frames)
        
        label = 0 if 'real' in file_path else 1
        labels.extend([label] * len(video_frames))
    
    print(f"Loaded {len(frames)} frames from {len(file_paths)} videos.")
    return np.array(frames), np.array(labels)

# Model creation
def create_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01), name="conv1", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D((2, 2), name="maxpool1"))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.01), name="conv2"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2), name="maxpool2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation='relu', name="dense1"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name="output"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Predict real or fake for a single video
def predict_video(video_path, model, frame_rate=30, target_size=(128, 128)):
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path, frame_rate=frame_rate)
    if len(frames) == 0:
        print("No frames extracted. Skipping this video.")
        return "Error: No frames"
    preprocessed_frames = preprocess_frames(frames, target_size=target_size)
    predictions = model.predict(preprocessed_frames)
    mean_prediction = np.mean(predictions.flatten())
    print(f"Mean prediction score: {mean_prediction}")
    video_label = "Fake" if mean_prediction >= 0.5 else "Real"
    return video_label

# Main function
def main():
    # Load and preprocess training data
    X_train, y_train = load_data_from_directory(train_dir, frame_rate=30, target_size=(128, 128))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Create model and train
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])

    # Evaluate model
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation accuracy: {val_acc}")

    # Plot Training & Validation Accuracy and Loss
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion matrix and classification report
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    print("Classification Report:\n", classification_report(y_val, y_pred))
    conf_matrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Test on a single video
    test_video_path = os.path.join(test_dir, '/kaggle/input/deepfake-detection-challenge/test_videos/aassnaulhq.mp4')  # Change this to your test video path
    video_label = predict_video(test_video_path, model)
    print(f"The video '{os.path.basename(test_video_path)}' is predicted to be: {video_label}")

    # Batch prediction on all test videos
    video_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.mp4')]
    for video_path in video_files:
        video_label = predict_video(video_path, model)
        print(f"Video: {os.path.basename(video_path)} - Predicted as: {video_label}")

# Run the main function
if __name__ == "__main__":
    main()



import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
plt.style.use('ggplot')
from IPython.display import Video
from IPython.display import HTML
!ls -GFlash ../input/deepfake-detection-challenge
!du -sh ../input/deepfake-detection-challenge/


train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.head()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# Load metadata
metadata_path = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'
train_sample_metadata = pd.read_json(metadata_path).T

# Display the first few rows of the metadata
print("Metadata Sample:")
print(train_sample_metadata.head())

# Visualize the distribution of fake vs. real videos
plt.figure(figsize=(8, 6))
sns.countplot(data=train_sample_metadata, x='label', palette='Set1')
plt.title('Distribution of Fake vs. Real Videos')
plt.xlabel('Label (Real vs Fake)')
plt.ylabel('Number of Videos')
plt.show()

# Further analysis: Count of videos by category (real/fake)
category_counts = train_sample_metadata['label'].value_counts()
print("Category Counts:")
print(category_counts)

# Pie chart visualization of class distribution
plt.figure(figsize=(8, 6))
category_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'], legend=False)
plt.title('Class Distribution (Real vs Fake)')
plt.ylabel('')
plt.show()

# Check if the 'duration' column exists
if 'duration' in train_sample_metadata.columns:
    # Further analysis: Length of videos based on label (Real vs Fake)
    train_sample_metadata['duration'] = train_sample_metadata['duration'].astype(float)  # Ensure duration is float
    real_videos_duration = train_sample_metadata[train_sample_metadata['label'] == 'REAL']['duration']
    fake_videos_duration = train_sample_metadata[train_sample_metadata['label'] == 'FAKE']['duration']

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='duration', data=train_sample_metadata, palette='Set2')
    plt.title('Video Duration by Label (Real vs Fake)')
    plt.xlabel('Label')
    plt.ylabel('Video Duration (seconds)')
    plt.show()

    # Additional Graph: Duration of Real vs Fake Videos Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(real_videos_duration, color='lightgreen', label='Real Videos', kde=True, bins=30)
    sns.histplot(fake_videos_duration, color='salmon', label='Fake Videos', kde=True, bins=30)
    plt.title('Distribution of Video Durations (Real vs Fake)')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
else:
    print("The 'duration' column is not present in the metadata. Skipping video duration analysis.")

# Define a function to display a frame from a video
def display_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()
    else:
        print(f"Failed to read video: {video_path}")

# Path to the video folder
video_folder = '../input/deepfake-detection-challenge/train_sample_videos/'

# Display some sample frames from fake videos
fake_videos = train_sample_metadata[train_sample_metadata['label'] == 'FAKE'].index
print("Sample frames from Fake videos:")
for video in fake_videos[:3]:  # Display the first 3 fake videos
    print(f"Video: {video}")
    display_video_frame(os.path.join(video_folder, video))

# Display some sample frames from real videos
real_videos = train_sample_metadata[train_sample_metadata['label'] == 'REAL'].index
print("Sample frames from Real videos:")
for video in real_videos[:3]:  # Display the first 3 real videos
    print(f"Video: {video}")
    display_video_frame(os.path.join(video_folder, video))

# Additional Analysis: Real vs Fake Videos in Terms of Aspect Ratio
def get_video_aspect_ratio(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        height, width, _ = frame.shape
        return width / height
    return None

# Get aspect ratio for real and fake videos
real_videos_aspect_ratio = [get_video_aspect_ratio(os.path.join(video_folder, video)) for video in real_videos]
fake_videos_aspect_ratio = [get_video_aspect_ratio(os.path.join(video_folder, video)) for video in fake_videos]

# Remove None values (in case of failed aspect ratio extraction)
real_videos_aspect_ratio = [ratio for ratio in real_videos_aspect_ratio if ratio is not None]
fake_videos_aspect_ratio = [ratio for ratio in fake_videos_aspect_ratio if ratio is not None]

# Plot aspect ratios of real vs fake videos
plt.figure(figsize=(10, 6))
sns.kdeplot(real_videos_aspect_ratio, color='lightgreen', label='Real Videos', shade=True)
sns.kdeplot(fake_videos_aspect_ratio, color='salmon', label='Fake Videos', shade=True)
plt.title('Aspect Ratio Distribution of Real vs Fake Videos')
plt.xlabel('Aspect Ratio (Width/Height)')
plt.ylabel('Density')
plt.legend()
plt.show()




!pip install mtcnn tensorflow opencv-python pandas numpy scikit-learn


import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialize MTCNN face detector
detector = MTCNN()

# Paths and setup
video_folder = '../input/deepfake-detection-challenge/train_sample_videos/'
metadata_path = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'

# Load metadata
train_sample_metadata = pd.read_json(metadata_path).T

# Split metadata into training and validation sets
train_metadata, val_metadata = train_test_split(train_sample_metadata, test_size=0.2, random_state=42)

class VideoFrameGenerator(Sequence):
    def __init__(self, metadata, batch_size=32, target_size=(224, 224), shuffle=True):
        self.metadata = metadata
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]
        
        X, y_labels = self.__data_generation(batch_metadata)
        return X, y_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_metadata):
        X = []
        y_labels = []
        
        for video_name, row in batch_metadata.iterrows():
            video_path = os.path.join(video_folder, video_name)
            label = 1 if row['label'] == 'FAKE' else 0
            
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                
                for face in faces:
                    x, y, width, height = face['box']
                    face_img = frame_rgb[y:y+height, x:x+width]
                    face_img = cv2.resize(face_img, self.target_size)  # Resize face to target_size
                    face_array = img_to_array(face_img) / 255.0  # Normalize pixel values
                    
                    X.append(face_array)
                    y_labels.append(label)
                    
                    if len(X) >= self.batch_size:
                        cap.release()
                        return np.array(X), np.array(y_labels)
            
            cap.release()
        
        # If we exit the loop and don't have enough samples, pad with the first few samples
        while len(X) < self.batch_size:
            X.append(X[0])
            y_labels.append(y_labels[0])
        
        return np.array(X), np.array(y_labels)

# Instantiate the generators
batch_size = 32
train_generator = VideoFrameGenerator(train_metadata, batch_size=batch_size)
val_generator = VideoFrameGenerator(val_metadata, batch_size=batch_size)

# Build the model (with deeper CNN layers)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model 
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Perform predictions on the validation set
y_true = []
y_pred = []

for X_batch, y_batch in val_generator:
    y_true.extend(y_batch)
    y_pred_batch = model.predict(X_batch)
    y_pred.extend((y_pred_batch > 0.5).astype(int))  # Convert predictions to binary (0 or 1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# You can also calculate precision, recall, and F1-score individually:
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

   




    # Plot training and validation loss and accuracy
    import matplotlib.pylab as plt

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

   


import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize MTCNN face detector
detector = MTCNN()

# Load the trained model
# model = load_model('path/to/your/trained_model.h5')  # Update with the actual path to your model

# Function to detect and preprocess faces from a video
def extract_faces_from_video(video_path, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(frame_rgb)

        for face in detected_faces:
            x, y, width, height = face['box']
            face_img = frame_rgb[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, target_size)
            face_array = img_to_array(face_img) / 255.0
            faces.append(face_array)
    
    cap.release()
    return np.array(faces)

# Function to predict if the video is fake or real
def predict_video(video_path):
    faces = extract_faces_from_video(video_path)
    if len(faces) == 0:
        print("No faces detected in the video.")
        return None

    predictions = model.predict(faces)
    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        print(f"The video '{video_path}' is predicted to be FAKE.")
    else:
        print(f"The video '{video_path}' is predicted to be REAL.")

    return avg_prediction

# Test the prediction function with a sample video
video_path = '/kaggle/input/deepfake-detection-challenge/test_videos'  # Update with the actual path to the test video
predict_video(video_path)

