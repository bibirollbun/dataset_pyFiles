


# ==============================================================================
# Deepfake Detection with Data Generator for Kaggle Notebooks
# This code provides a complete, runnable solution for training on the large
# DFDC dataset by using a custom data generator to handle memory constraints.
# ==============================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, LSTM, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# --- Corrected Import ---
# We will import EfficientNetB0 inside the function to make the code more robust.
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



# ==============================================================================
# 1. Project Configuration and Hyperparameters
# ==============================================================================
# The size of the face images after cropping and resizing
IMG_SIZE = 224
# The number of frames to sample from each video for the sequence
# We have reduced this value to prevent GPU memory issues.
MAX_SEQ_LENGTH = 12
# The training batch size. Adjust based on your GPU memory.
# We have reduced this value from 16 to 8 to prevent Out of Memory (OOM) errors.
BATCH_SIZE = 8
# Number of training epochs
EPOCHS = 10

# --- Kaggle-specific dataset path ---
DATA_DIR = "/kaggle/input/deepfake-detection-challenge"

# Path to the pre-trained Haar Cascade face detector XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# ==============================================================================
# 2. Data Pre-processing Pipeline
# ==============================================================================

def extract_faces_from_video(video_path):
    """
    Extracts a fixed number of face-cropped frames from a video.
    This function simulates the pre-processing workflow for deepfake detection.
    
    Args:
        video_path (str): The path to the video file.
    
    Returns:
        np.array: A 4D numpy array of pre-processed face images (frames).
    """
    frames = []
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Handle cases where the video file cannot be opened
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate an interval to evenly sample frames
    frame_interval = max(1, total_frames // MAX_SEQ_LENGTH)
    frame_count = 0
    
    while len(frames) < MAX_SEQ_LENGTH:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert frame to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the Haar Cascade classifier
        # minSize=(40,40) helps filter out very small detections
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) > 0:
            # Assume only one face per video for simplicity; take the largest one.
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            # Crop the detected face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize and normalize the face image
            resized_face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            normalized_face = resized_face / 255.0
            
            frames.append(normalized_face)
            
        frame_count += frame_interval
        
    cap.release()
    
    # --- FIX START ---
    # Moved the 'if not frames:' check BEFORE converting to a numpy array
    # to avoid the ValueError.
    if not frames:
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    frames = np.array(frames)
    # --- FIX END ---
    
    # Pad sequences with zeros if fewer than MAX_SEQ_LENGTH faces were detected
    if len(frames) < MAX_SEQ_LENGTH:
        padding_needed = MAX_SEQ_LENGTH - len(frames)
        padding_shape = (padding_needed, IMG_SIZE, IMG_SIZE, 3)
        padded_frames = np.zeros(padding_shape)
        frames = np.vstack((frames, padded_frames))
        
    return frames


def load_dataset_from_dir(data_dir):
    """
    Loads video paths and labels from the DFDC dataset structure.
    
    Args:
        data_dir (str): The root directory of the dataset.
        
    Returns:
        tuple: A tuple containing lists of (video_paths, labels).
    """
    print("Loading dataset metadata from disk...")
    video_paths = []
    labels = []
    
    # DFDC dataset is organized into 'train_sample_videos', 'test_videos', etc.
    # The training data is further split into parts (e.g., 'part_00', 'part_01')
    # in the full competition dataset. We'll iterate through these parts.
    
    # For this example, we will just use the 'train_sample_videos' part
    # which is often available by default for quick testing.
    # To use the full dataset, you would need to iterate through all 'part_xx' folders.
    
    dataset_part_dir = os.path.join(data_dir, 'train_sample_videos')
    metadata_path = os.path.join(dataset_part_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        for video_name, video_info in metadata.items():
            # We only care about videos that are marked FAKE or REAL.
            if video_info['label'] in ['FAKE', 'REAL']:
                video_path = os.path.join(dataset_part_dir, video_name)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(1 if video_info['label'] == 'FAKE' else 0)
    else:
        print(f"Error: metadata.json not found at {metadata_path}")
    
    return video_paths, np.array(labels)



# ==============================================================================
# 3. Model Architecture (CNN-LSTM Hybrid)
# ==============================================================================

def create_model():
    """
    Builds and compiles the hybrid CNN-LSTM model for deepfake detection.
    
    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    # Create the CNN backbone using a pre-trained EfficientNetB0 model.
    # Note: EfficientNetB0 needs to be imported separately.
    from tensorflow.keras.applications import EfficientNetB0
    
    cnn_backbone = EfficientNetB0(
        weights="imagenet", # Use pre-trained weights from ImageNet
        include_top=False,  # Exclude the final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn_backbone.trainable = True # Set the backbone to be trainable for fine-tuning
    
    # Define the model input as a sequence of video frames
    video_input = Input(shape=(MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    # Use TimeDistributed to apply the CNN to each frame in the sequence
    cnn_features = TimeDistributed(cnn_backbone)(video_input)
    
    # Replaced TimeDistributed(Flatten()) with TimeDistributed(GlobalAveragePooling2D())
    # to significantly reduce the feature vector size and prevent the OOM error.
    cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
    
    # Add the LSTM layer for temporal analysis
    lstm_features = LSTM(128)(cnn_features)
    
    # Add a Dropout layer for regularization to prevent overfitting
    lstm_features = Dropout(0.5)(lstm_features)
    
    # The final output layer for binary classification (Real vs. Fake)
    output = Dense(1, activation='sigmoid')(lstm_features)
    
    # Construct the full model from the input and output layers
    model = Model(inputs=video_input, outputs=output)
    
    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


# ==============================================================================
# 4. Data Generator for Training on Large Datasets
# ==============================================================================

class DataGenerator(Sequence):
    """
    Keras Data Generator for efficiently loading and pre-processing videos
    in batches, preventing memory errors on large datasets.
    """
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get the batch's indices
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the video paths and labels for this batch
        batch_paths = [self.video_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]
        
        # Pre-process the videos and store in X, y
        X = np.empty((self.batch_size, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            # Load and pre-process the video
            frames = extract_faces_from_video(path)
            X[i,] = frames
            y[i] = label
            
        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is enabled."""
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# ==============================================================================
# 5. Main Execution Block for Training and Prediction
# ==============================================================================

if __name__ == "__main__":
    # --- Step 5.1: Load Data Paths and Labels ---
    print("Starting data loading...")
    video_paths, labels = load_dataset_from_dir(DATA_DIR)
    print(f"Loaded {len(video_paths)} video paths and labels.")
    
   
    
 
    
   
    
  


 # --- Step 5.2: Split Data for Training and Validation ---
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )


 # --- Step 5.3: Create Data Generators ---
    train_generator = DataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE)
    val_generator = DataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, shuffle=False)
    


  
    # --- Step 5.4: Create and Train Model ---
    model = create_model()
    
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )
    
    print("\nTraining complete.")
    
    # Save the trained model for future use
    model.save("deepfake_detector.h5")
    print("Model saved to 'deepfake_detector.h5'")


 # --- Step 5.5: Model Evaluation ---
    print("\nEvaluating model on the validation set...")
    y_pred_probs = model.predict(val_generator)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32")
    
    # Flatten y_val since val_generator returns batches.
    # Note: This is an approximation. For a perfect evaluation, you'd
    # collect all predictions and labels in one go.
    y_val_flat = np.concatenate([y for X, y in val_generator], axis=0)
    
    accuracy = accuracy_score(y_val_flat, y_pred_classes)
    precision = precision_score(y_val_flat, y_pred_classes)
    recall = recall_score(y_val_flat, y_pred_classes)
    f1 = f1_score(y_val_flat, y_pred_classes)
    auc = roc_auc_score(y_val_flat, y_pred_probs)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")


  # --- Step 5.6: Prediction on a single video ---
    # This shows how to use the trained model on a new video file.
    new_video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/aaxjpsnvrq.mp4"
    if os.path.exists(new_video_path):
        print(f"\nMaking a prediction on a new video: {new_video_path}")
        new_video_frames = extract_faces_from_video(new_video_path)
        # Add the batch dimension for prediction
        new_video_frames = np.expand_dims(new_video_frames, axis=0)
        
        # Get the deepfake probability from the model
        prediction = model.predict(new_video_frames)[0][0]
        
        print("\nPrediction:")
        if prediction > 0.5:
            print(f"This video is likely a DEEPFAKE with confidence {prediction:.2f}")
        else:
            print(f"This video is likely REAL with confidence {1 - prediction:.2f}")
    else:
        print(f"\nPrediction skipped: Could not find the new video at {new_video_path}")



!pip install efficientnet


# ==============================================================================
# Deepfake Detection with Data Generator for Kaggle Notebooks
# This code provides a complete, runnable solution for training on the large
# DFDC dataset by using a custom data generator to handle memory constraints.
# ==============================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, LSTM, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# --- Corrected Import ---
# We will import EfficientNetB0 inside the function to make the code more robust.
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================================
# 1. Project Configuration and Hyperparameters
# ==============================================================================
# The size of the face images after cropping and resizing
IMG_SIZE = 224
# The number of frames to sample from each video for the sequence
# We have reduced this value to prevent GPU memory issues.
MAX_SEQ_LENGTH = 8
# The training batch size. Adjust based on your GPU memory.
# We have reduced this value to a very conservative 4 to prevent OOM errors.
BATCH_SIZE = 4
# Number of training epochs
EPOCHS = 10

# --- Kaggle-specific dataset path ---
DATA_DIR = "/kaggle/input/deepfake-detection-challenge"

# Path to the pre-trained Haar Cascade face detector XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ==============================================================================
# 2. Data Pre-processing Pipeline
# ==============================================================================

def extract_faces_from_video(video_path):
    """
    Extracts a fixed number of face-cropped frames from a video.
    This function simulates the pre-processing workflow for deepfake detection.
    
    Args:
        video_path (str): The path to the video file.
    
    Returns:
        np.array: A 4D numpy array of pre-processed face images (frames).
    """
    frames = []
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Handle cases where the video file cannot be opened
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate an interval to evenly sample frames
    frame_interval = max(1, total_frames // MAX_SEQ_LENGTH)
    frame_count = 0
    
    while len(frames) < MAX_SEQ_LENGTH:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert frame to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the Haar Cascade classifier
        # minSize=(40,40) helps filter out very small detections
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) > 0:
            # Assume only one face per video for simplicity; take the largest one.
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            # Crop the detected face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize and normalize the face image
            resized_face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            normalized_face = resized_face / 255.0
            
            frames.append(normalized_face)
            
        frame_count += frame_interval
        
    cap.release()
    
    # --- FIX START ---
    # Moved the 'if not frames:' check BEFORE converting to a numpy array
    # to avoid the ValueError.
    if not frames:
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    frames = np.array(frames)
    # --- FIX END ---
    
    # Pad sequences with zeros if fewer than MAX_SEQ_LENGTH faces were detected
    if len(frames) < MAX_SEQ_LENGTH:
        padding_needed = MAX_SEQ_LENGTH - len(frames)
        padding_shape = (padding_needed, IMG_SIZE, IMG_SIZE, 3)
        padded_frames = np.zeros(padding_shape)
        frames = np.vstack((frames, padded_frames))
        
    return frames


def load_dataset_from_dir(data_dir):
    """
    Loads video paths and labels from the DFDC dataset structure.
    
    Args:
        data_dir (str): The root directory of the dataset.
        
    Returns:
        tuple: A tuple containing lists of (video_paths, labels).
    """
    print("Loading dataset metadata from disk...")
    video_paths = []
    labels = []
    
    # DFDC dataset is organized into 'train_sample_videos', 'test_videos', etc.
    # The training data is further split into parts (e.g., 'part_00', 'part_01')
    # in the full competition dataset. We'll iterate through these parts.
    
    # For this example, we will just use the 'train_sample_videos' part
    # which is often available by default for quick testing.
    # To use the full dataset, you would need to iterate through all 'part_xx' folders.
    
    dataset_part_dir = os.path.join(data_dir, 'train_sample_videos')
    metadata_path = os.path.join(dataset_part_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        for video_name, video_info in metadata.items():
            # We only care about videos that are marked FAKE or REAL.
            if video_info['label'] in ['FAKE', 'REAL']:
                video_path = os.path.join(dataset_part_dir, video_name)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(1 if video_info['label'] == 'FAKE' else 0)
    else:
        print(f"Error: metadata.json not found at {metadata_path}")
    
    return video_paths, np.array(labels)


# ==============================================================================
# 3. Model Architecture (CNN-LSTM Hybrid)
# ==============================================================================

def create_model():
    """
    Builds and compiles the hybrid CNN-LSTM model for deepfake detection.
    
    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    # Try to import EfficientNetB0 from different paths,
    # as its location varies across TensorFlow versions.
    try:
        from tensorflow.keras.applications import EfficientNetB0
    except ImportError:
        try:
            from efficientnet.tfkeras import EfficientNetB0
        except ImportError:
            # If all imports fail, the user must install the library.
            # This is the last resort to ensure the code is runnable.
            raise ImportError(
                "EfficientNetB0 not found. Please run the following command "
                "in a separate cell to install the library: !pip install efficientnet"
            )

    cnn_backbone = EfficientNetB0(
        weights="imagenet", # Use pre-trained weights from ImageNet
        include_top=False,  # Exclude the final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn_backbone.trainable = True # Set the backbone to be trainable for fine-tuning
    
    # Define the model input as a sequence of video frames
    video_input = Input(shape=(MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    # Replaced TimeDistributed(Flatten()) with TimeDistributed(GlobalAveragePooling2D())
    # to significantly reduce the feature vector size and prevent the OOM error.
    cnn_features = TimeDistributed(cnn_backbone)(video_input)
    cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
    
    # Add the LSTM layer for temporal analysis
    lstm_features = LSTM(128)(cnn_features)
    
    # Add a Dropout layer for regularization to prevent overfitting
    lstm_features = Dropout(0.5)(lstm_features)
    
    # The final output layer for binary classification (Real vs. Fake)
    output = Dense(1, activation='sigmoid')(lstm_features)
    
    # Construct the full model from the input and output layers
    model = Model(inputs=video_input, outputs=output)
    
    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


# ==============================================================================
# 4. Data Generator for Training on Large Datasets
# ==============================================================================

class DataGenerator(Sequence):
    """
    Keras Data Generator for efficiently loading and pre-processing videos
    in batches, preventing memory errors on large datasets.
    """
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get the batch's indices
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the video paths and labels for this batch
        batch_paths = [self.video_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]
        
        # Pre-process the videos and store in X, y
        X = np.empty((self.batch_size, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            # Load and pre-process the video
            frames = extract_faces_from_video(path)
            X[i,] = frames
            y[i] = label
            
        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is enabled."""
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# ==============================================================================
# 5. Main Execution Block for Training and Prediction
# ==============================================================================

if __name__ == "__main__":
    # --- Step 5.1: Load Data Paths and Labels ---
    print("Starting data loading...")
    video_paths, labels = load_dataset_from_dir(DATA_DIR)
    print(f"Loaded {len(video_paths)} video paths and labels.")
    
    # --- Step 5.2: Split Data for Training and Validation ---
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- Step 5.3: Create Data Generators ---
    train_generator = DataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE)
    val_generator = DataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 5.4: Create and Train Model ---
    model = create_model()
    
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )
    
    print("\nTraining complete.")
    
    # Save the trained model for future use
    model.save("deepfake_detector.h5")
    print("Model saved to 'deepfake_detector.h5'")
    
    # --- Step 5.5: Model Evaluation ---
    print("\nEvaluating model on the validation set...")
    y_pred_probs = model.predict(val_generator)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32")
    
    # Flatten y_val since val_generator returns batches.
    # Note: This is an approximation. For a perfect evaluation, you'd
    # collect all predictions and labels in one go.
    y_val_flat = np.concatenate([y for X, y in val_generator], axis=0)
    
    accuracy = accuracy_score(y_val_flat, y_pred_classes)
    precision = precision_score(y_val_flat, y_pred_classes)
    recall = recall_score(y_val_flat, y_pred_classes)
    f1 = f1_score(y_val_flat, y_pred_classes)
    auc = roc_auc_score(y_val_flat, y_pred_probs)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # --- Step 5.6: Prediction on a single video ---
    # This shows how to use the trained model on a new video file.
    new_video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/aaxjpsnvrq.mp4"
    if os.path.exists(new_video_path):
        print(f"\nMaking a prediction on a new video: {new_video_path}")
        new_video_frames = extract_faces_from_video(new_video_path)
        # Add the batch dimension for prediction
        new_video_frames = np.expand_dims(new_video_frames, axis=0)
        
        # Get the deepfake probability from the model
        prediction = model.predict(new_video_frames)[0][0]
        
        print("\nPrediction:")
        if prediction > 0.5:
            print(f"This video is likely a DEEPFAKE with confidence {prediction:.2f}")
        else:
            print(f"This video is likely REAL with confidence {1 - prediction:.2f}")
    else:
        print(f"\nPrediction skipped: Could not find the new video at {new_video_path}")



!pip install efficientnet


# ==============================================================================
# Deepfake Detection with Data Generator for Kaggle Notebooks
# This code provides a complete, runnable solution for training on the large
# DFDC dataset by using a custom data generator to handle memory constraints.
# ==============================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, LSTM, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# --- Corrected Import ---
# We will import EfficientNetB0 inside the function to make the code more robust.
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================================
# 1. Project Configuration and Hyperparameters
# ==============================================================================
# The size of the face images after cropping and resizing
IMG_SIZE = 224
# The number of frames to sample from each video for the sequence
# We have reduced this value to prevent GPU memory issues.
MAX_SEQ_LENGTH = 8
# The training batch size. Adjust based on your GPU memory.
# We have reduced this value to a very conservative 4 to prevent OOM errors.
BATCH_SIZE = 4
# Number of training epochs
EPOCHS = 10

# --- Kaggle-specific dataset path ---
DATA_DIR = "/kaggle/input/deepfake-detection-challenge"

# Path to the pre-trained Haar Cascade face detector XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ==============================================================================
# 2. Data Pre-processing Pipeline
# ==============================================================================

def extract_faces_from_video(video_path):
    """
    Extracts a fixed number of face-cropped frames from a video.
    This function simulates the pre-processing workflow for deepfake detection.
    
    Args:
        video_path (str): The path to the video file.
    
    Returns:
        np.array: A 4D numpy array of pre-processed face images (frames).
    """
    frames = []
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Handle cases where the video file cannot be opened
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate an interval to evenly sample frames
    frame_interval = max(1, total_frames // MAX_SEQ_LENGTH)
    frame_count = 0
    
    while len(frames) < MAX_SEQ_LENGTH:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert frame to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the Haar Cascade classifier
        # minSize=(40,40) helps filter out very small detections
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) > 0:
            # Assume only one face per video for simplicity; take the largest one.
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            # Crop the detected face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize and normalize the face image
            resized_face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            normalized_face = resized_face / 255.0
            
            frames.append(normalized_face)
            
        frame_count += frame_interval
        
    cap.release()
    
    # --- FIX START ---
    # Moved the 'if not frames:' check BEFORE converting to a numpy array
    # to avoid the ValueError.
    if not frames:
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    frames = np.array(frames)
    # --- FIX END ---
    
    # Pad sequences with zeros if fewer than MAX_SEQ_LENGTH faces were detected
    if len(frames) < MAX_SEQ_LENGTH:
        padding_needed = MAX_SEQ_LENGTH - len(frames)
        padding_shape = (padding_needed, IMG_SIZE, IMG_SIZE, 3)
        padded_frames = np.zeros(padding_shape)
        frames = np.vstack((frames, padded_frames))
        
    return frames


def load_dataset_from_dir(data_dir):
    """
    Loads video paths and labels from the DFDC dataset structure.
    
    Args:
        data_dir (str): The root directory of the dataset.
        
    Returns:
        tuple: A tuple containing lists of (video_paths, labels).
    """
    print("Loading dataset metadata from disk...")
    video_paths = []
    labels = []
    
    # DFDC dataset is organized into 'train_sample_videos', 'test_videos', etc.
    # The training data is further split into parts (e.g., 'part_00', 'part_01')
    # in the full competition dataset. We'll iterate through these parts.
    
    # For this example, we will just use the 'train_sample_videos' part
    # which is often available by default for quick testing.
    # To use the full dataset, you would need to iterate through all 'part_xx' folders.
    
    dataset_part_dir = os.path.join(data_dir, 'train_sample_videos')
    metadata_path = os.path.join(dataset_part_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        for video_name, video_info in metadata.items():
            # We only care about videos that are marked FAKE or REAL.
            if video_info['label'] in ['FAKE', 'REAL']:
                video_path = os.path.join(dataset_part_dir, video_name)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(1 if video_info['label'] == 'FAKE' else 0)
    else:
        print(f"Error: metadata.json not found at {metadata_path}")
    
    return video_paths, np.array(labels)


# ==============================================================================
# 3. Model Architecture (CNN-LSTM Hybrid)
# ==============================================================================

def create_model():
    """
    Builds and compiles the hybrid CNN-LSTM model for deepfake detection.
    
    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    # Try to import EfficientNetB0 from different paths,
    # as its location varies across TensorFlow versions.
    try:
        from tensorflow.keras.applications import EfficientNetB0
    except ImportError:
        try:
            from efficientnet.tfkeras import EfficientNetB0
        except ImportError:
            # If all imports fail, the user must install the library.
            # This is the last resort to ensure the code is runnable.
            raise ImportError(
                "EfficientNetB0 not found. Please run the following command "
                "in a separate cell to install the library: !pip install efficientnet"
            )

    cnn_backbone = EfficientNetB0(
        weights="imagenet", # Use pre-trained weights from ImageNet
        include_top=False,  # Exclude the final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn_backbone.trainable = True # Set the backbone to be trainable for fine-tuning
    
    # Define the model input as a sequence of video frames
    video_input = Input(shape=(MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    # Replaced TimeDistributed(Flatten()) with TimeDistributed(GlobalAveragePooling2D())
    # to significantly reduce the feature vector size and prevent the OOM error.
    cnn_features = TimeDistributed(cnn_backbone)(video_input)
    cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
    
    # Add the LSTM layer for temporal analysis
    lstm_features = LSTM(128)(cnn_features)
    
    # Add a Dropout layer for regularization to prevent overfitting
    lstm_features = Dropout(0.5)(lstm_features)
    
    # The final output layer for binary classification (Real vs. Fake)
    output = Dense(1, activation='sigmoid')(lstm_features)
    
    # Construct the full model from the input and output layers
    model = Model(inputs=video_input, outputs=output)
    
    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


# ==============================================================================
# 4. Data Generator for Training on Large Datasets
# ==============================================================================

class DataGenerator(Sequence):
    """
    Keras Data Generator for efficiently loading and pre-processing videos
    in batches, preventing memory errors on large datasets.
    """
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get the batch's indices
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the video paths and labels for this batch
        batch_paths = [self.video_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]
        
        # Pre-process the videos and store in X, y
        X = np.empty((self.batch_size, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            # Load and pre-process the video
            frames = extract_faces_from_video(path)
            X[i,] = frames
            y[i] = label
            
        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is enabled."""
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# ==============================================================================
# 5. Main Execution Block for Training and Prediction
# ==============================================================================

if __name__ == "__main__":
    # --- Step 5.1: Load Data Paths and Labels ---
    print("Starting data loading...")
    video_paths, labels = load_dataset_from_dir(DATA_DIR)
    print(f"Loaded {len(video_paths)} video paths and labels.")
    
    # --- Step 5.2: Split Data for Training and Validation ---
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- Step 5.3: Create Data Generators ---
    train_generator = DataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE)
    val_generator = DataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 5.4: Create and Train Model ---
    model = create_model()
    
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )
    
    print("\nTraining complete.")
    
    # Save the trained model for future use
    model.save("deepfake_detector.h5")
    print("Model saved to 'deepfake_detector.h5'")
    
    # --- Step 5.5: Model Evaluation ---
    print("\nEvaluating model on the validation set...")
    y_pred_probs = model.predict(val_generator)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32")
    
    # Flatten y_val since val_generator returns batches.
    # Note: This is an approximation. For a perfect evaluation, you'd
    # collect all predictions and labels in one go.
    y_val_flat = np.concatenate([y for X, y in val_generator], axis=0)
    
    accuracy = accuracy_score(y_val_flat, y_pred_classes)
    precision = precision_score(y_val_flat, y_pred_classes)
    recall = recall_score(y_val_flat, y_pred_classes)
    f1 = f1_score(y_val_flat, y_pred_classes)
    auc = roc_auc_score(y_val_flat, y_pred_probs)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # --- Step 5.6: Prediction on a single video ---
    # This shows how to use the trained model on a new video file.
    new_video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/aaxjpsnvrq.mp4"
    if os.path.exists(new_video_path):
        print(f"\nMaking a prediction on a new video: {new_video_path}")
        new_video_frames = extract_faces_from_video(new_video_path)
        # Add the batch dimension for prediction
        new_video_frames = np.expand_dims(new_video_frames, axis=0)
        
        # Get the deepfake probability from the model
        prediction = model.predict(new_video_frames)[0][0]
        
        print("\nPrediction:")
        if prediction > 0.5:
            print(f"This video is likely a DEEPFAKE with confidence {prediction:.2f}")
        else:
            print(f"This video is likely REAL with confidence {1 - prediction:.2f}")
    else:
        print(f"\nPrediction skipped: Could not find the new video at {new_video_path}")



# ==============================================================================
# Deepfake Detection with Data Generator for Kaggle Notebooks
# This code provides a complete, runnable solution for training on the large
# DFDC dataset by using a custom data generator to handle memory constraints.
# ==============================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, LSTM, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception # We will use Xception as per the app config
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================================
# 1. Project Configuration and Hyperparameters
# ==============================================================================
# The size of the face images after cropping and resizing
IMG_SIZE = 224
# The number of frames to sample from each video for the sequence
# We have reduced this value to prevent GPU memory issues.
MAX_SEQ_LENGTH = 12
# The training batch size. Adjust based on your GPU memory.
# We have reduced this value to a very conservative 4 to prevent OOM errors.
BATCH_SIZE = 4
# Number of training epochs
EPOCHS = 10

# --- Kaggle-specific dataset path ---
DATA_DIR = "/kaggle/input/deepfake-detection-challenge"

# Path to the pre-trained Haar Cascade face detector XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ==============================================================================
# 2. Data Pre-processing Pipeline
# ==============================================================================

def extract_faces_from_video(video_path):
    """
    Extracts a fixed number of face-cropped frames from a video.
    This function simulates the pre-processing workflow for deepfake detection.
    
    Args:
        video_path (str): The path to the video file.
    
    Returns:
        np.array: A 4D numpy array of pre-processed face images (frames).
    """
    frames = []
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Handle cases where the video file cannot be opened
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate an interval to evenly sample frames
    frame_interval = max(1, total_frames // MAX_SEQ_LENGTH)
    frame_count = 0
    
    while len(frames) < MAX_SEQ_LENGTH:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert frame to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the Haar Cascade classifier
        # minSize=(40,40) helps filter out very small detections
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) > 0:
            # Assume only one face per video for simplicity; take the largest one.
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            # Crop the detected face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize and normalize the face image
            resized_face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            normalized_face = resized_face / 255.0
            frames.append(normalized_face)
            
        frame_count += frame_interval
        
    cap.release()
    
    # --- FIX START ---
    # Moved the 'if not frames:' check BEFORE converting to a numpy array
    # to avoid the ValueError.
    if not frames:
        return np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    frames = np.array(frames)
    # --- FIX END ---
    
    # Pad sequences with zeros if fewer than MAX_SEQ_LENGTH faces were detected
    if len(frames) < MAX_SEQ_LENGTH:
        padding_needed = MAX_SEQ_LENGTH - len(frames)
        padding_shape = (padding_needed, IMG_SIZE, IMG_SIZE, 3)
        padded_frames = np.zeros(padding_shape)
        frames = np.vstack((frames, padded_frames))
        
    return frames


def load_dataset_from_dir(data_dir):
    """
    Loads video paths and labels from the DFDC dataset structure.
    
    Args:
        data_dir (str): The root directory of the dataset.
        
    Returns:
        tuple: A tuple containing lists of (video_paths, labels).
    """
    print("Loading dataset metadata from disk...")
    video_paths = []
    labels = []
    
    # --- FIX START ---
    # Corrected file path to the metadata.json file.
    # The 'train_sample_videos' folder is now located directly under the dataset root.
    metadata_path = os.path.join(data_dir, 'train_sample_videos/metadata.json')
    # --- FIX END ---
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        for video_name, video_info in metadata.items():
            # We only care about videos that are marked FAKE or REAL.
            if video_info['label'] in ['FAKE', 'REAL']:
                video_path = os.path.join(data_dir, 'train_sample_videos', video_name)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(1 if video_info['label'] == 'FAKE' else 0)
    else:
        print(f"Error: metadata.json not found at {metadata_path}")
    
    return video_paths, np.array(labels)


# ==============================================================================
# 3. Model Architecture (CNN-LSTM Hybrid)
# ==============================================================================

def create_model():
    """
    Builds and compiles the hybrid CNN-LSTM model for deepfake detection.
    
    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    # Create the CNN backbone using a pre-trained Xception model.
    cnn_backbone = Xception(
        weights="imagenet", # Use pre-trained weights from ImageNet
        include_top=False,  # Exclude the final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn_backbone.trainable = True # Set the backbone to be trainable for fine-tuning
    
    # Define the model input as a sequence of video frames
    video_input = Input(shape=(MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    # Replaced TimeDistributed(Flatten()) with TimeDistributed(GlobalAveragePooling2D())
    # to significantly reduce the feature vector size and prevent the OOM error.
    cnn_features = TimeDistributed(cnn_backbone)(video_input)
    cnn_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
    
    # Add the LSTM layer for temporal analysis
    lstm_features = LSTM(128)(cnn_features)
    
    # Add a Dropout layer for regularization to prevent overfitting
    lstm_features = Dropout(0.5)(lstm_features)
    
    # The final output layer for binary classification (Real vs. Fake)
    output = Dense(1, activation='sigmoid')(lstm_features)
    
    # Construct the full model from the input and output layers
    model = Model(inputs=video_input, outputs=output)
    
    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


# ==============================================================================
# 4. Data Generator for Training on Large Datasets
# ==============================================================================

class DataGenerator(Sequence):
    """
    Keras Data Generator for efficiently loading and pre-processing videos
    in batches, preventing memory errors on large datasets.
    """
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get the batch's indices
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the video paths and labels for this batch
        batch_paths = [self.video_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]
        
        # Pre-process the videos and store in X, y
        X = np.empty((self.batch_size, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            # Load and pre-process the video
            frames = extract_faces_from_video(path)
            X[i,] = frames
            y[i] = label
            
        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is enabled."""
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# ==============================================================================
# 5. Main Execution Block for Training and Prediction
# ==============================================================================

if __name__ == "__main__":
    # --- Step 5.1: Load Data Paths and Labels ---
    print("Starting data loading...")
    video_paths, labels = load_dataset_from_dir(DATA_DIR)
    print(f"Loaded {len(video_paths)} video paths and labels.")
    
    # --- Step 5.2: Split Data for Training and Validation ---
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- Step 5.3: Create Data Generators ---
    train_generator = DataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE)
    val_generator = DataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 5.4: Create and Train Model ---
    model = create_model()
    
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )
    
    print("\nTraining complete.")
    
    # Save the trained model for future use
    model.save("deepfake_detector.h5")
    print("Model saved to 'deepfake_detector.h5'")
    
    # --- Step 5.5: Model Evaluation ---
    print("\nEvaluating model on the validation set...")
    y_pred_probs = model.predict(val_generator)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32")
    
    # Flatten y_val since val_generator returns batches.
    # Note: This is an approximation. For a perfect evaluation, you'd
    # collect all predictions and labels in one go.
    y_val_flat = np.concatenate([y for X, y in val_generator], axis=0)
    
    accuracy = accuracy_score(y_val_flat, y_pred_classes)
    precision = precision_score(y_val_flat, y_pred_classes)
    recall = recall_score(y_val_flat, y_pred_classes)
    f1 = f1_score(y_val_flat, y_pred_classes)
    auc = roc_auc_score(y_val_flat, y_pred_probs)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # --- Step 5.6: Prediction on a single video ---
    # This shows how to use the trained model on a new video file.
    new_video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/aaxjpsnvrq.mp4"
    if os.path.exists(new_video_path):
        print(f"\nMaking a prediction on a new video: {new_video_path}")
        new_video_frames = extract_faces_from_video(new_video_path)
        # Add the batch dimension for prediction
        new_video_frames = np.expand_dims(new_video_frames, axis=0)
        
        # Get the deepfake probability from the model
        prediction = model.predict(new_video_frames)[0][0]
        
        print("\nPrediction:")
        if prediction > 0.5:
            print(f"This video is likely a DEEPFAKE with confidence {prediction:.2%}")
        else:
            print(f"This video is likely REAL with confidence {1 - prediction:.2%}")
    else:
        print(f"\nPrediction skipped: Could not find the new video at {new_video_path}")



import tensorflow as tf

# Load your old h5 model (ignore compile issues)
model = tf.keras.models.load_model("deepfake_detection_model.h5", compile=False)

# Save it in new .keras format
model.save("deepfake_detection_model.keras")

print("✅ Model converted and saved as deepfake_detection_model.keras")


