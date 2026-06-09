from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os


IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


DATA_FOLDER = '/kaggle/input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")


train_sample_metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.head()


train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()


train_sample_metadata.shape


fake_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='FAKE'].sample(10).index)
fake_train_sample_video


import cv2
import matplotlib.pyplot as plt

def show_first_frame(video_file_path):
    
    video_capture = cv2.VideoCapture(video_file_path)
    
    # Verify that the video file was opened successfully
    if not video_capture.isOpened():
        video_capture.release()  # Ensure resources are released
        raise FileNotFoundError(f"Failed to access the video at {video_file_path}")
    
    # Attempt to capture the first frame
    successful, frame = video_capture.read()
    
    # Ensure that a frame was successfully captured
    if not successful:
        video_capture.release()  # Ensure resources are released before raising an error
        raise RuntimeError("No frames could be read from the video file")
    
    # Adjust the frame's color format for displaying
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    
    # Close the video capture object to free resources
    video_capture.release()


for video_file in fake_train_sample_video:
    show_first_frame(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


real_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='REAL'].sample(10).index)
real_train_sample_video


for video_file in real_train_sample_video:
    show_first_frame(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


train_sample_metadata['original'].value_counts()[0:10]


import os
import cv2
import matplotlib.pyplot as plt

def show_frames_from_videos(video_files, base_folder=TRAIN_SAMPLE_FOLDER):
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    # Loop through the first six videos in the list
    for index, video_name in enumerate(video_files[:6]):
        video_full_path = os.path.join(DATA_FOLDER, base_folder, video_name)
        video_capture = cv2.VideoCapture(video_full_path)
        
        success, frame = video_capture.read()
        if not success:
            print(f"Failed to read from {video_name}")
            axes[index // 3, index % 3].set_title("Failed to load video")
            axes[index // 3, index % 3].axis('off')
            continue
        
        # Convert the color from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the image in the respective subplot
        axes[index // 3, index % 3].imshow(frame_rgb)
        axes[index // 3, index % 3].set_title(video_name)
        axes[index // 3, index % 3].axis('on')  # Keep the axis on for clarity
        
        # Release the video capture object
        video_capture.release()

    plt.tight_layout()
    plt.show()


same_original_fake_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.original=='meawmsgiti.mp4'].index)
show_frames_from_videos(same_original_fake_train_sample_video)


test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])


test_videos.head(10)


show_first_frame(os.path.join(DATA_FOLDER, TEST_FOLDER, test_videos.iloc[3].video))


fake_videos = list(train_sample_metadata.loc[train_sample_metadata.label=='FAKE'].index)


from IPython.display import HTML
from base64 import b64encode
import os

def embed_video_in_notebook(video_filename, directory=TRAIN_SAMPLE_FOLDER):
    try:
        # Construct the full path to the video file
        video_path = os.path.join(DATA_FOLDER, directory, video_filename)
        
        # Read the video file as binary data
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()

        # Encode the video data in base64 and create the data URL
        video_base64 = b64encode(video_data).decode('utf-8')
        data_url = f"data:video/mp4;base64,{video_base64}"

        # Return an HTML object that contains the video element
        return HTML(f'<video width="500" controls><source src="{data_url}" type="video/mp4"></video>')
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The video file {video_filename} could not be found in {directory}.")

# Example usage:
# Assuming 'fake_videos[10]' contains the filename of the video to play
video_to_play = fake_videos[14] 
embed_video_in_notebook(video_to_play)


import cv2
import numpy as np

def square_crop_frame(image):
    
    height, width = image.shape[:2]
    min_dimension = min(height, width)
    start_x = (width - min_dimension) // 2
    start_y = (height - min_dimension) // 2
    return image[start_y:start_y + min_dimension, start_x:start_x + min_dimension]

def process_video_frames(video_path, max_frames=0, resize_dims=(IMG_SIZE, IMG_SIZE)):

    capture = cv2.VideoCapture(video_path)
    processed_frames = []
    try:
        while True:
            read_success, frame = capture.read()
            if not read_success:
                break
            frame = square_crop_frame(frame)
            frame = cv2.resize(frame, resize_dims)
            # Convert BGR to RGB for standard color format
            frame = frame[..., ::-1]
            processed_frames.append(frame)

            if max_frames > 0 and len(processed_frames) >= max_frames:
                break
    finally:
        capture.release()
    return np.array(processed_frames)


import tensorflow as tf
from tensorflow import keras

def build_feature_extractor(model_name='ResNet50'):
    # Get model class and preprocessing function
    base_model_class = getattr(keras.applications, model_name)
    preprocess_input = getattr(keras.applications, model_name.lower()).preprocess_input

    # Base model without top layers
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Build sequential model
    model = keras.Sequential([
        keras.layers.Lambda(preprocess_input, input_shape=(IMG_SIZE, IMG_SIZE, 3), name='preprocessing'),
        base_model
    ], name=f"{model_name}_feature_extractor_seq")

    return model


from sklearn.model_selection import train_test_split

Train_set, Test_set = train_test_split(train_sample_metadata,test_size=0.1,random_state=42,stratify=train_sample_metadata['label'])

print(Train_set.shape, Test_set.shape )


import numpy as np
import os

feature_extractor = build_feature_extractor('ResNet50')

def extract_video_features(dataframe, directory):
    
    total_videos = len(dataframe)
    video_file_paths = dataframe.index.tolist()
    binary_labels = np.array(dataframe["label"].values == 'FAKE', dtype=int)

    # Initialize arrays to hold data for all videos
    video_masks = np.zeros((total_videos, MAX_SEQ_LENGTH), dtype=bool)
    video_features = np.zeros((total_videos, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Process each video individually
    for video_idx, video_file in enumerate(video_file_paths):
        full_video_path = os.path.join(directory, video_file)
        video_data = process_video_frames(full_video_path)
        video_data = np.expand_dims(video_data, axis=0)  # Add a batch dimension

        # Temporary storage for this video's data
        current_video_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
        current_video_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        # Frame-by-frame feature extraction
        frames_to_process = min(MAX_SEQ_LENGTH, video_data.shape[1])
        for frame_idx in range(frames_to_process):
            frame = video_data[:, frame_idx, :]
            extracted_features = feature_extractor.predict(frame[None, :])
            current_video_features[0, frame_idx, :] = extracted_features

        current_video_mask[0, :frames_to_process] = True  # Mark frames as valid

        # Store the extracted data in the corresponding arrays
        video_features[video_idx] = current_video_features.squeeze()
        video_masks[video_idx] = current_video_mask.squeeze()

    return (video_features, video_masks), binary_labels


train_data, train_labels = extract_video_features(Train_set, "train")
test_data, test_labels = extract_video_features(Test_set, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


from tensorflow.keras import layers, models, regularizers, metrics

frame_features_input = layers.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = layers.Input((MAX_SEQ_LENGTH,), dtype="bool")

x = layers.LSTM(
    16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)
)(frame_features_input, mask=mask_input)

x = layers.LSTM(
    8, kernel_regularizer=regularizers.l2(0.01)
)(x)

x = layers.Dropout(0.5)(x)
x = layers.Dense(8, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model([frame_features_input, mask_input], output)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        metrics.Recall(name="recall"),
        metrics.Precision(name="precision")
    ]
)

model.summary()


import os
from tensorflow.keras import callbacks, models

# Define the directory for storing model checkpoints and the final model
checkpoint_dir = './model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

# Setup the model checkpoint callback to save only the best model during training
checkpoint_filepath = os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_loss:.2f}.h5')
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

# EarlyStopping callback to stop training early if no improvement
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Model training
history = model.fit(
    [train_data[0], train_data[1]],
    train_labels,
    validation_data=([test_data[0], test_data[1]], test_labels),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=1
)

# Save the final model after training
final_model_path = os.path.join(checkpoint_dir, 'final_model6.h5')
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

# Optionally, print the history of training
print("Training history:", history.history)


# Evaluate the model on the test set
test_loss, test_accuracy, test_recall, test_precision = model.evaluate(
    [test_data[0], test_data[1]],  # Test features and masks
    test_labels,                   # Test labels
    batch_size=8                  # Use the batch size consistent with training
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Precision: {test_precision:.4f}")


import matplotlib.pyplot as plt

# Assuming `history` is the result of model.fit()
hist = history.history

# Create 2x2 subplot
plt.figure(figsize=(14, 10))

# Plot Accuracy
plt.subplot(2, 2, 1)
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(2, 2, 2)
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Recall
plt.subplot(2, 2, 3)
plt.plot(hist['recall'], label='Train Recall')
plt.plot(hist['val_recall'], label='Val Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# Plot Precision
plt.subplot(2, 2, 4)
plt.plot(hist['precision'], label='Train Precision')
plt.plot(hist['val_precision'], label='Val Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
plt.show()


y_pred_prob = model.predict([test_data[0], test_data[1]], batch_size=8)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Classification report
print(classification_report(test_labels, y_pred, digits=4))


# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


import numpy as np
import os

feature_extractor = build_feature_extractor('VGG16')

def extract_video_features(dataframe, directory):
    
    total_videos = len(dataframe)
    video_file_paths = dataframe.index.tolist()
    binary_labels = np.array(dataframe["label"].values == 'FAKE', dtype=int)

    # Initialize arrays to hold data for all videos
    video_masks = np.zeros((total_videos, MAX_SEQ_LENGTH), dtype=bool)
    video_features = np.zeros((total_videos, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Process each video individually
    for video_idx, video_file in enumerate(video_file_paths):
        full_video_path = os.path.join(directory, video_file)
        video_data = process_video_frames(full_video_path)
        video_data = np.expand_dims(video_data, axis=0)  # Add a batch dimension

        # Temporary storage for this video's data
        current_video_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
        current_video_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        # Frame-by-frame feature extraction
        frames_to_process = min(MAX_SEQ_LENGTH, video_data.shape[1])
        for frame_idx in range(frames_to_process):
            frame = video_data[:, frame_idx, :]
            extracted_features = feature_extractor.predict(frame[None, :])
            current_video_features[0, frame_idx, :] = extracted_features

        current_video_mask[0, :frames_to_process] = True  # Mark frames as valid

        # Store the extracted data in the corresponding arrays
        video_features[video_idx] = current_video_features.squeeze()
        video_masks[video_idx] = current_video_mask.squeeze()

    return (video_features, video_masks), binary_labels


from tensorflow.keras import layers, models, regularizers, metrics

frame_features_input = layers.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = layers.Input((MAX_SEQ_LENGTH,), dtype="bool")

x = layers.Bidirectional(layers.LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(
    frame_features_input, mask=mask_input
)
x = layers.Bidirectional(layers.LSTM(8, kernel_regularizer=regularizers.l2(0.01)))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(8, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model([frame_features_input, mask_input], output)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        metrics.Recall(name="recall"),
        metrics.Precision(name="precision")
    ]
)

model.summary()


import os
from tensorflow.keras import callbacks, models

# Define the directory for storing model checkpoints and the final model
checkpoint_dir = './model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

# Setup the model checkpoint callback to save only the best model during training
checkpoint_filepath = os.path.join(checkpoint_dir, 'bid-model-{epoch:02d}-{val_loss:.2f}.h5')
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

# EarlyStopping callback to stop training early if no improvement
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Model training
history = model.fit(
    [train_data[0], train_data[1]],
    train_labels,
    validation_data=([test_data[0], test_data[1]], test_labels),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=1
)

# Save the final model after training
final_model_path = os.path.join(checkpoint_dir, 'final_model6.h5')
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

# Optionally, print the history of training
print("Training history:", history.history)


# Evaluate the model on the test set
test_loss, test_accuracy, test_recall, test_precision = model.evaluate(
    [test_data[0], test_data[1]],  # Test features and masks
    test_labels,                   # Test labels
    batch_size=8                  # Use the batch size consistent with training
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Precision: {test_precision:.4f}")


import matplotlib.pyplot as plt

# Assuming `history` is the result of model.fit()
hist = history.history

# Create 2x2 subplot
plt.figure(figsize=(14, 10))

# Plot Accuracy
plt.subplot(2, 2, 1)
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(2, 2, 2)
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Recall
plt.subplot(2, 2, 3)
plt.plot(hist['recall'], label='Train Recall')
plt.plot(hist['val_recall'], label='Val Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# Plot Precision
plt.subplot(2, 2, 4)
plt.plot(hist['precision'], label='Train Precision')
plt.plot(hist['val_precision'], label='Val Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
plt.show()


y_pred_prob = model.predict([test_data[0], test_data[1]], batch_size=8)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Classification report
print(classification_report(test_labels, y_pred, digits=4))


# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(test_labels, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

