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


#It is impossible to run my full model in 1 go ... I have done some parts in local 
#jupyter notebook(anaconda) and some in kaggle notebook 


#step 1
#finding the number of frames in one video and making jpg of each frame
#this code should be wrote in local notebook

import cv2
import os

def extract_frames_from_videos(input_folder, output_base_folder, frame_rate=5):
    """
    Extracts frames from all videos in a folder, saving them in individual folders for each video.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_base_folder (str): Path to the base folder where extracted frames will be saved.
        frame_rate (int): Number of frames to save per second of the video.
    """
    # Create the base output folder if it doesn't exist
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # Process each video in the input folder
    for video_filename in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_filename)
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Skipping non-video file: {video_filename}")
            continue

        # Create an output folder for the current video
        video_name = os.path.splitext(video_filename)[0]
        output_folder = os.path.join(output_base_folder, video_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_filename}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
        frame_interval = int(fps / frame_rate)  # Interval between frames to save

        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame if it matches the interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"Saved {saved_frame_count} frames for video: {video_filename}")

# Input folder containing the videos
input_folder = "path/to/your/videos/folder"  # Replace with the folder containing your videos

# Output base folder to save extracted frames
output_base_folder = "path/to/output/folder"  # Replace with the desired output folder path

# Extract frames
extract_frames_from_videos(input_folder, output_base_folder, frame_rate=5)



#step 2
#for making the csv file of arrays
#this code should be wrote in local notebook
import torch
from pathlib import Path
import csv

def save_bounding_boxes_to_csv(input_base_folder, csv_file_path, model_path='yolov5s.pt', conf_threshold=0.25, img_size=640):
    """
    Detects objects in images within a folder of folders using YOLOv5 and saves
    bounding box coordinates (min/max x, y) to a CSV file.

    Args:
        input_base_folder (str): Path to the base folder containing subfolders of images.
        csv_file_path (str): Path to the CSV file for saving bounding box data.
        model_path (str): Path to the YOLOv5 model (e.g., 'yolov5s.pt').
        conf_threshold (float): Confidence threshold for detection.
        img_size (int): Input image size for YOLOv5.

    Returns:
        None
    """
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = conf_threshold  # Set confidence threshold
    model.img_size = img_size    # Set input image size

    # Open CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Video Folder", "Frame Name", "Min X", "Min Y", "Max X", "Max Y"])

        # Loop through all subfolders in the input base folder
        for folder in Path(input_base_folder).iterdir():
            if not folder.is_dir():
                continue

            # Process each image in the current subfolder
            for image_path in folder.glob('*.*'):
                if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                # Run inference
                results = model(str(image_path), size=img_size)

                # Get bounding boxes
                detections = results.xyxy[0].cpu().numpy()

                # Initialize min/max values
                min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

                # Calculate min/max bounding box coordinates
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Update min/max coordinates
                    min_x, min_y = min(min_x, x1), min(min_y, y1)
                    max_x, max_y = max(max_x, x2), max(max_y, y2)

                # Write bounding box data to the CSV file
                if min_x != float('inf') and max_x != float('-inf'):  # Ensure there was a detection
                    csv_writer.writerow([folder.name, image_path.name, min_x, min_y, max_x, max_y])

            print(f"Processed folder: {folder.name}")

# Specify input folder and CSV file path
input_base_folder = "path_to_input_base_folder"  # Replace with the folder containing subfolders of images
csv_file_path = "path_to_csv_file/bounding_box_data.csv"  # Replace with the path to save the CSV file
model_path = "yolov5s.pt"  # Replace with the path to your YOLOv5 model

# Run the function
save_bounding_boxes_to_csv(input_base_folder, csv_file_path, model_path=model_path, conf_threshold=0.001, img_size=640)



#step3 for prediction of first 4 attributes
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths for Kaggle dataset
train_videos_path = "../input/dataset-name/Training_Data/Train_Videos"
train_csv_path = "../input/dataset-name/Training_Data/train.csv"
test_videos_path = "../input/dataset-name/Testing_Data/Test_Videos"
test_output_csv = "./submission.csv"

# Function to preprocess video data
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 64x64 and convert to grayscale
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    # Create a 2D video summary by averaging frames
    video_summary = np.mean(frames, axis=0).flatten()
    return video_summary

# Load training data
train_data = pd.read_csv(train_csv_path)

# Prepare training features and labels
X_train = []
y_train = train_data.drop(columns=["video_id"])

for video_id in train_data["video_id"]:
    video_path = os.path.join(train_videos_path, f"{video_id}.mp4")
    X_train.append(preprocess_video(video_path))

X_train = np.array(X_train)

# Encode categorical labels
label_encoders = {}
for column in y_train.columns:
    if column != "video_summary":  # Skip video_summary
        le = LabelEncoder()
        y_train[column] = le.fit_transform(y_train[column])
        label_encoders[column] = le

# Drop 'video_summary' column from y_train
y_train = y_train.drop(columns=["video_summary"])

# Split data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train classifiers for each attribute (excluding 'video_summary')
classifiers = {}
for column in y_train.columns:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_split, y_train_split[column])
    classifiers[column] = clf

# Validate classifiers
for column in y_train.columns:
    y_pred = classifiers[column].predict(X_val)
    accuracy = accuracy_score(y_val[column], y_pred)
    print(f"Accuracy for {column}: {accuracy:.2f}")

# Predict for test data
X_test = []
test_video_ids = []

for video_file in os.listdir(test_videos_path):
    if video_file.endswith(".mp4"):
        video_id = os.path.splitext(video_file)[0]
        test_video_ids.append(video_id)
        video_path = os.path.join(test_videos_path, video_file)
        X_test.append(preprocess_video(video_path))

X_test = np.array(X_test)

y_test_pred = {}
for column in y_train.columns:
    y_test_pred[column] = classifiers[column].predict(X_test)

# Decode predictions and save to CSV (excluding 'video_summary')
output_data = {"video_id": test_video_ids}
for column, predictions in y_test_pred.items():
    output_data[column] = label_encoders[column].inverse_transform(predictions)

output_df = pd.DataFrame(output_data)
output_df.to_csv(test_output_csv, index=False)

print(f"Predictions saved to {test_output_csv}")



#i ran this code power prediction only as an experiment and it gave me better result due to changed test/train ration
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths for Kaggle dataset
train_videos_path = "../input/dataset-name/Training_Data/Train_Videos"
train_csv_path = "../input/dataset-name/Training_Data/train.csv"
test_videos_path = "../input/dataset-name/Testing_Data/Test_Videos"
test_output_csv = "./submission.csv"

# Function to preprocess video data
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 64x64 and convert to grayscale
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    # Create a 2D video summary by averaging frames
    video_summary = np.mean(frames, axis=0).flatten()
    return video_summary

# Load training data
train_data = pd.read_csv(train_csv_path)

# Prepare training features and labels
X_train = []
y_train = train_data["power"]  # Only target the 'power' attribute

for video_id in train_data["video_id"]:
    video_path = os.path.join(train_videos_path, f"{video_id}.mp4")
    X_train.append(preprocess_video(video_path))

X_train = np.array(X_train)

# Encode 'power' labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Split data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train Random Forest Classifier for 'power'
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_split, y_train_split)

# Validate classifier
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy for 'power': {accuracy:.2f}")

# Predict for test data
X_test = []
test_video_ids = []

for video_file in os.listdir(test_videos_path):
    if video_file.endswith(".mp4"):
        video_id = os.path.splitext(video_file)[0]
        test_video_ids.append(video_id)
        video_path = os.path.join(test_videos_path, video_file)
        X_test.append(preprocess_video(video_path))

X_test = np.array(X_test)

# Predict 'power' for test videos
y_test_pred = clf.predict(X_test)

# Decode predictions and save to CSV
output_data = {
    "video_id": test_video_ids,
    "power": label_encoder.inverse_transform(y_test_pred)
}

output_df = pd.DataFrame(output_data)
output_df.to_csv(test_output_csv, index=False)

print(f"'Power' predictions saved to {test_output_csv}")



#after this i repeated 1st and second step for test vidoes to get thier array csv file


#this is the last code which predict video summary using the arrays ... i uplaoded the output csvs from 2nd step onto the kaggle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load training data
train_data = pd.read_csv('/kaggle/input/train-dataset/train.csv')

# Load data for 3000 videos
test_data = pd.read_csv('/kaggle/input/test-dataset/test.csv')

# Function to preprocess data
def preprocess_data(data):
    # Convert array columns to numerical arrays
    array_columns = ["Min X Array", "Min Y Array", "Max X Array", "Max Y Array"]
    for col in array_columns:
        data[col] = data[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.zeros(20))
    
    # Encode categorical columns
    categorical_columns = ['element', 'motion', 'power']
    for col in categorical_columns:
        if col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
    
    return data

# Preprocess the training data
train_data = preprocess_data(train_data)

# Split video_summary into two columns
train_data[['summary_x', 'summary_y']] = train_data['video_summary'].str.strip('()').str.split(',', expand=True).astype(float)

# Prepare the feature set (X) and target set (y) for training
features = ['element', 'motion', 'power', 'speed'] + ["Min X Array", "Min Y Array", "Max X Array", "Max Y Array"]
X_train = np.hstack([train_data[features[:4]].values] + [np.vstack(train_data[col].values) for col in features[4:]])
y_train = train_data[['summary_x', 'summary_y']].values

# Train a Ridge regression model
model = Ridge(random_state=42)
model.fit(X_train, y_train)

# Preprocess the test data
test_data = preprocess_data(test_data)

# Prepare the feature set (X) for predictions
X_test = np.hstack([test_data[features[:4]].values] + [np.vstack(test_data[col].values) for col in features[4:]])

# Predict video summaries for the 3000 videos
predictions = model.predict(X_test)

# Include video_id in the predictions (if present in test.csv)
if 'video_id' in test_data.columns:
    prediction_df['video_id'] = test_data['video_id']

# Save predictions to a CSV file
output_path = '/kaggle/working/predictions.csv'
prediction_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")



#then i just compiled the 5 columns to make my best submission

