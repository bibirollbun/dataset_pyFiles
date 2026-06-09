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


!pip install decord torch torchvision yolov5  # Install required libraries




import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yolov5")

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from decord import VideoReader
import cv2
import torch
from yolov5 import YOLOv5

# Load Excel file
excel_path = '/kaggle/input/nexar-collision-prediction/train.csv'  # Update this path
df = pd.read_csv(excel_path)

print("Excel Data Sample:")
print(df.head())
print("\nNaN Check:")
print(df.isna().sum())

# Video folder path
video_folder = '/kaggle/input/nexar-collision-prediction/train'

def get_video_path(video_id):
    return os.path.join(video_folder, f"{str(video_id).zfill(5)}.mp4")

df['video_path'] = df['id'].apply(get_video_path)
df['video_exists'] = df['video_path'].apply(os.path.exists)
print("\nMissing Videos:")
print(df[~df['video_exists']])
df = df[df['video_exists']]

# Select 300 videos with target = 1 and 300 with target = 0
df_target_1 = df[df['target'] == 1].head(300)
df_target_0 = df[df['target'] == 0].head(300)
df_train = pd.concat([df_target_1, df_target_0])
print(f"Total training samples before processing: {len(df_train)}")

# Load YOLOv5 model
yolo_model = YOLOv5("yolov5s.pt", device="cuda" if torch.cuda.is_available() else "cpu")

# Function to extract frame and bounding boxes
def extract_frame_and_boxes(video_path, timestamp):
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    if pd.isna(timestamp):
        frame_num = np.random.randint(0, total_frames) if total_frames > 0 else 0
    else:
        frame_num = min(int(timestamp * fps), total_frames - 1)
    
    if frame_num >= total_frames or frame_num < 0:
        return None, []
    
    frame = vr[frame_num].asnumpy()
    results = yolo_model.predict(frame)
    boxes = []
    for det in results.pred[0]:
        if det[5] in [2, 7]:  # Car or truck
            x1, y1, x2, y2 = map(int, det[:4])
            boxes.append([x1, y1, x2, y2])
    return frame, boxes

# Preprocess for MobileNetV2
def preprocess_input(image):
    if image is None:
        return None
    resized = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
    return np.expand_dims(normalized, axis=0)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = models.Sequential([base_model, layers.GlobalAveragePooling2D()])
feature_extractor.trainable = False

# Extract features and bounding box info
def extract_features_and_boxes(frame, boxes, model):
    preprocessed = preprocess_input(frame)
    if preprocessed is None:
        return None, None, None
    
    features = model.predict(preprocessed, verbose=0).flatten()
    dashcam_pos = (frame.shape[1] // 2, frame.shape[0])
    box_features = []
    for box in boxes:
        x1, y1, x2, y2 = box
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = np.sqrt((centroid[0] - dashcam_pos[0])**2 + (centroid[1] - dashcam_pos[1])**2)
        box_features.append(distance)
    
    min_distance = min(box_features) if box_features else 1000
    combined_features = np.concatenate([features, [min_distance]])
    return combined_features, boxes, min_distance

# Collect training data with dynamic Collision threshold
X_train = []
y_train = []
distances_target_1 = []

for index, row in df_train.iterrows():
    video_path = row['video_path']
    time_of_event = row['time_of_event']
    time_of_alert = row['time_of_alert']
    target = row['target']
    
    frame, boxes = extract_frame_and_boxes(video_path, time_of_event)
    if frame is not None:
        features, _, min_distance = extract_features_and_boxes(frame, boxes, feature_extractor)
        if features is not None:
            X_train.append(features)
            if target == 0:
                y_train.append(0)  # Normal
            else:
                distances_target_1.append(min_distance)  # Collect distances for target = 1
            y_train.append(target)  # Temporary label (0 or 1)
        else:
            print(f"Failed to extract features for {video_path}")
    else:
        print(f"Failed to extract frame for {video_path}")

# Calculate dynamic Collision threshold (25th percentile of target = 1 distances)
COLLISION_THRESHOLD = np.percentile(distances_target_1, 25) if distances_target_1 else 200
print(f"Dynamic Collision Threshold: {COLLISION_THRESHOLD:.2f} pixels")

# Reassign labels with Collision class
for i in range(len(y_train)):
    if y_train[i] == 1:  # Only for target = 1 samples
        min_distance = X_train[i][-1]
        time_diff = abs(df_train.iloc[i]['time_of_event'] - df_train.iloc[i]['time_of_alert']) if not pd.isna(df_train.iloc[i]['time_of_alert']) else float('inf')
        if min_distance < COLLISION_THRESHOLD or time_diff < 0.5:
            y_train[i] = 2  # Collision
        else:
            y_train[i] = 1  # Caution

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Collected {len(X_train)} samples with features: {X_train.shape}")
print(f"Initial distribution: Normal={np.sum(y_train == 0)}, Caution={np.sum(y_train == 1)}, Collision={np.sum(y_train == 2)}")

# Oversample Collision to 150 for emphasis (half of Normal/Caution)
collision_indices = np.where(y_train == 2)[0]
target_collision_count = 150
if len(collision_indices) > 0 and len(collision_indices) < target_collision_count:
    oversample_indices = np.random.choice(collision_indices, target_collision_count - len(collision_indices), replace=True)
    X_train = np.concatenate([X_train, X_train[oversample_indices]])
    y_train = np.concatenate([y_train, y_train[oversample_indices]])

print(f"Balanced distribution: Normal={np.sum(y_train == 0)}, Caution={np.sum(y_train == 1)}, Collision={np.sum(y_train == 2)}")

# Build and train model
model = models.Sequential([
    layers.Input(shape=(1281,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save('/kaggle/working/collision_model_3class.h5')

# Test on 00023.mp4
video_path = '/kaggle/input/nexar-collision-prediction/test/00314.mp4'
output_video_path = '/kaggle/working/predicted_00314_3rdclass.mp4'

vr = VideoReader(video_path)
fps = vr.get_avg_fps()
total_frames = len(vr)
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

caution_count = 0
collision_count = 0
frame_count = 0

for i in range(total_frames):
    frame = vr[i].asnumpy()
    features, boxes, min_distance = extract_features_and_boxes(frame, extract_frame_and_boxes(video_path, i/fps)[1], feature_extractor)
    if features is not None:
        pred = model.predict(np.expand_dims(features, axis=0), verbose=0)
        class_id = np.argmax(pred)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if class_id == 1:
            cv2.putText(frame, "CAUTION: Vehicle Close!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            border_color = (0, 255, 255)
            caution_count += 1
        elif class_id == 2:
            cv2.putText(frame, "COLLISION IMMINENT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            border_color = (0, 0, 255)
            collision_count += 1
        else:
            border_color = (0, 255, 0)
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 10)
        video_writer.write(frame)
        frame_count += 1

cap.release()
video_writer.release()
print(f"Processed {frame_count} frames. Caution frames: {caution_count} ({(caution_count / frame_count) * 100:.2f}%), Collision frames: {collision_count} ({(collision_count / frame_count) * 100:.2f}%)")
print(f"Output saved to: {output_video_path}")

