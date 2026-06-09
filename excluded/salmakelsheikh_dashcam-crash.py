import warnings
warnings.filterwarnings("ignore")
import os

data_path = "/kaggle/input/nexar-collision-prediction/"
print(os.listdir(data_path))


import pandas as pd

# Load train.csv
train_df = pd.read_csv(f"{data_path}/train.csv")

print(train_df.head())


train_df.info()


#Explore data img
import os

# Count images in train folder
train_images = os.listdir(f"{data_path}/train")
test_images = os.listdir(f"{data_path}/test")

print(f"Number of training images: {len(train_images)}")
print(f"Number of test images: {len(test_images)}")


# Fill missing values with the mean of each column
train_df['time_of_event'].fillna(train_df['time_of_event'].mean(), inplace=True)
train_df['time_of_alert'].fillna(train_df['time_of_alert'].mean(), inplace=True)

# Check if there are still missing values
print(train_df.isnull().sum())


# Check the file extensions of images
image_extensions = [os.path.splitext(img)[1] for img in train_images]
print(set(image_extensions))  # Display unique file extensions


from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Image size (e.g., 224x224 for CNN)
img_size = (224, 224)

# Load images
images = []
for img_name in train_images[:10]:  # Load the first 10 images
    img_path = os.path.join(data_path, 'train', img_name)
    try:
        img = image.load_img(img_path, target_size=img_size)  
        img_array = image.img_to_array(img) / 255.0  
        images.append(img_array)
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")

# Convert list to numpy array
images = np.array(images)
print(images.shape)



!pip install opencv-python



import cv2
import os

# Function to extract frames from a video file
def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If the video has more than max_frames, we pick frames evenly spaced
    step = total_frames // max_frames
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:
            # Resize and normalize
            frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
            frame = frame / 255.0  # Normalize pixel values to [0, 1]
            frames.append(frame)
    
    cap.release()
    return frames

# Example: Extract frames from the first video
video_path = os.path.join(data_path, 'train', '02059.mp4')  
frames = extract_frames(video_path)
print(f"Extracted {len(frames)} frames from {video_path}")


#Process All Video Files
all_frames = []
video_names = [video for video in train_images if video.endswith('.mp4')]

for video_name in video_names[:10]:  
    video_path = os.path.join(data_path, 'train', video_name)
    frames = extract_frames(video_path)
    all_frames.extend(frames)

# Convert list of frames to numpy array
all_frames = np.array(all_frames)
print(f"Extracted {all_frames.shape[0]} frames in total.")


import matplotlib.pyplot as plt

# Plot the distribution of the target variable
train_df['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Target Variable")
plt.xlabel("Target (0 = No Collision, 1 = Collision)")
plt.ylabel("Count")
plt.show()



# Correlation matrix
import seaborn as sns
correlation_matrix = train_df[['time_of_event', 'time_of_alert', 'target']].corr()

# Plotting the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


image_files = [f for f in os.listdir(os.path.join(data_path, 'train')) if f.endswith(('.jpg', '.png'))]
video_files = [f for f in os.listdir(os.path.join(data_path, 'train')) if f.endswith('.mp4')]

print(f"Number of image files: {len(image_files)}")
print(f"Number of video files: {len(video_files)}")


import cv2
import matplotlib.pyplot as plt

# Function to extract a frame from a video file
def extract_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Make sure the frame index is within range
    if frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Resize and normalize the frame
            frame_resized = cv2.resize(frame, (224, 224))
            frame_resized = frame_resized / 255.0  
            return frame_resized
        else:
            print(f"Error: Couldn't read frame {frame_idx} from {video_path}")
            return None
    else:
        print(f"Error: Frame {frame_idx} exceeds total frames {total_frames} in {video_path}")
        cap.release()
        return None

# Extract and display frames from the first video
video_path = os.path.join(data_path, 'train', video_files[8]) 
frame = extract_frame(video_path, frame_idx=10) 

if frame is not None:
    plt.imshow(frame)
    plt.axis('off')
    plt.show()
else:
    print("Could not extract valid frame.")



# Display frames from the first videos
num_videos_to_display = min(9, len(video_files))  
fig, axes = plt.subplots(1, num_videos_to_display, figsize=(15, 10))

for i, ax in enumerate(axes):
    video_path = os.path.join(data_path, 'train', video_files[i])
    frame = extract_frame(video_path, frame_idx=10) 
    
    if frame is not None:
        ax.imshow(frame)
        ax.axis('off')
    else:
        ax.set_title(f"Error in video {i}")
        ax.axis('off')

plt.show()



import matplotlib.pyplot as plt

# Display the first 5 extracted frames
fig, axes = plt.subplots(1, 5, figsize=(15, 10))
for i, ax in enumerate(axes):
    ax.imshow(all_frames[i])
    ax.axis('off')
plt.show()


import pandas as pd

# Load the train.csv file
train_csv_path = "/kaggle/input/nexar-collision-prediction/train.csv"
train_data = pd.read_csv(train_csv_path)

# Check the first few rows
print(train_data.head())



import numpy as np
import os
import cv2
import pandas as pd

# Load train.csv
train_csv_path = "/kaggle/input/nexar-collision-prediction/train.csv"
train_data = pd.read_csv(train_csv_path)

# Function to extract frames from a video
def extract_frame(video_path, frame_idx=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_resized = cv2.resize(frame, target_size)
            frame_resized = frame_resized / 255.0  # Normalize
            return frame_resized
    cap.release()
    return None

# Function to process multiple videos
def preprocess_frames(video_files, num_frames=5):
    X, y = [], []

    for video_file in video_files:
        video_id = int(video_file.split('.')[0])  # Extract ID
        video_path = os.path.join('/kaggle/input/nexar-collision-prediction/train', video_file)
        
        # Match ID to target value
        target_row = train_data[train_data['id'] == video_id]
        if target_row.empty:
            print(f"Warning: No target found for {video_file}")
            continue  
        
        target = target_row['target'].values[0]  # Get label (0 or 1)
        
        frames = []
        for frame_idx in np.linspace(0, 50, num_frames, dtype=int):  # Sample 5 frames
            frame = extract_frame(video_path, frame_idx)
            if frame is not None:
                frames.append(frame)

        if frames:
            X.append(np.array(frames))
            y.append(target)

    return np.array(X), np.array(y)

# Get video filenames
video_files = [f for f in os.listdir('/kaggle/input/nexar-collision-prediction/train') if f.endswith('.mp4')]

# Process videos (first 20 for testing)
X, y = preprocess_frames(video_files[:20])

print("Data shape:", X.shape, "Labels shape:", y.shape)


import numpy as np

def preprocess_frames(video_files, num_frames=5, target_size=(224, 224)):
    X = []
    y = []

    # Iterate through video files and extract frames
    for video_file in video_files:
        video_path = os.path.join(data_path, 'train', video_file)
        
        # Get the target (collision or not)
        video_id = int(video_file.split('.')[0])  # Assuming the filename is the ID
        target = train_data[train_data['id'] == video_id]['target'].values[0]
        
        # Extract frames from the video
        frames = []
        for frame_idx in range(num_frames):
            frame = extract_frame(video_path, frame_idx)
            if frame is not None:
                frames.append(frame)
        
        if len(frames) > 0:
            X.append(np.array(frames))
            y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Preprocess frames from the first 10 videos
X, y = preprocess_frames(video_files[:10]) 


print("X shape:", X.shape) 
print("y shape:", y.shape) 


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

# Define input shape for a single frame
input_shape = (224, 224, 3)

# Load Pretrained Model (without top layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

# Add a classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(512, activation="relu")(x)  
x = Dense(1, activation="sigmoid")(x)  # (collision or not)

# Define final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()


#Train the model
import numpy as np

# Convert X from shape (10, 5, 224, 224, 3) → (50, 224, 224, 3) (Flatten Frames)
X_train_flat = X.reshape(-1, 224, 224, 3)  # Treat each frame as a separate image
y_train_expanded = np.repeat(y, 5)  # Duplicate labels (each frame gets same label)

# Train the CNN model
history = model.fit(X_train_flat, y_train_expanded, epochs=5, batch_size=8, validation_split=0.2)


# Unfreeze last few layers of ResNet
for layer in base_model.layers[-20:]:  
    layer.trainable = True

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])


X, y = preprocess_frames(video_files[:700]) 


# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,         # Rotat
    width_shift_range=0.1,    # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    horizontal_flip=True    # Flip images
)


from sklearn.model_selection import train_test_split

# Flatten frames (Convert shape from (200, 5, 224, 224, 3) → (1000, 224, 224, 3))
X_flat = X.reshape(-1, 224, 224, 3)
y_flat = np.repeat(y, 5)  # Repeat labels (each frame gets same label)

# Split into 80% training and 20% validation
X_train_flat, X_val_flat, y_train_flat, y_val_flat = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train_flat.shape, y_train_flat.shape)
print("Validation Data Shape:", X_val_flat.shape, y_val_flat.shape)


history = model.fit(X_train_flat, y_train_flat, epochs=10, batch_size=8, validation_data=(X_val_flat, y_val_flat))


#Reduce Overfitting
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.6)(x)
output = Dense(1, activation='sigmoid')(x)


from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)


history = model.fit(X_train_flat, y_train_flat, epochs=15, batch_size=8, 
                    validation_data=(X_val_flat, y_val_flat),
                    callbacks=[reduce_lr])


datagen = ImageDataGenerator(
    rotation_range=20,     
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    zoom_range=0.2  # Zoom in or out
)


model.save("final_resnet50_model.h5")


import glob

# Get all test video file paths
test_video_files = sorted(glob.glob("/kaggle/input/nexar-collision-prediction/test/*.mp4"))

print("Total test videos:", len(test_video_files))
print("Example paths:", test_video_files[:5])  


import os

# Extract video IDs from filenames
test_video_ids = [int(os.path.basename(f).split('.')[0]) for f in test_video_files]

print("Example test video IDs:", test_video_ids[:5])  


import cv2
import numpy as np

def extract_frames(video_paths, num_frames=30, target_size=(224, 224)):
    """
    Extracts 'num_frames' frames from each video and resizes them to target_size.
    """
    X_test = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in np.linspace(0, total_frames-1, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, target_size)  
                frame = frame / 255.0  
                frames.append(frame)
            else:
                break 

        cap.release()

        # Ensure all videos have 'num_frames' frames (padding if necessary)
        while len(frames) < num_frames:
            frames.append(np.zeros((target_size[0], target_size[1], 3)))  # Black frame

        X_test.append(frames)

    return np.array(X_test)

# Extract frames from all test videos
X_test = extract_frames(test_video_files, num_frames=30)

print("X_test shape:", X_test.shape)  # (num_videos, 30, 224, 224, 3)


# Ensure model expects (None, 30, 224, 224, 3)
predictions = model.predict(X_test)

# Convert probabilities to binary labels (0 or 1)
predictions = (predictions > 0.5).astype(int)

print("Predictions shape:", predictions.shape)  # Should be (1344, 1)
print(predictions[:5])  


import pandas as pd

# Flatten predictions to match video IDs
submission = pd.DataFrame({'video_id': test_video_ids, 'prediction': predictions.flatten()})
submission.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv")
print(submission.head())




