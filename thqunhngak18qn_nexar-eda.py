import numpy as np
import pandas as pd

import pandas.api.types

import sklearn.metrics
import os
import random
import glob
import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML, Video
from base64 import b64encode
import re
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)



import matplotlib.pyplot as plt
import os
import glob

# Đường dẫn thư mục train và test
dataset_train_dir = "/kaggle/input/nexar-collision-prediction/train"  
dataset_test_dir = "/kaggle/input/nexar-collision-prediction/test"

# Hàm kiểm tra và đếm số lượng file video
def count_video_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return 0
    files = os.listdir(directory)
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))]
    return len(video_files)

# Đếm số lượng file video trong mỗi tập
train_video_count = count_video_files(dataset_train_dir)
test_video_count = count_video_files(dataset_test_dir)

# Hiển thị số lượng file video trong mỗi tập
print("--- Number of file video in each dataset---")
print(f"Train: {train_video_count} video files")
print(f"Test: {test_video_count} video files")
print(f"Total: {train_video_count + test_video_count} video files")

# Biểu đồ biểu thị số lượng file video
labels = ['Train', 'Test']
counts = [train_video_count, test_video_count]

plt.figure(figsize=(6, 6))
bars = plt.bar(labels, counts, color=['skyblue', 'lightgreen'])

# Thêm số lượng chính xác lên trên mỗi cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height),
             ha='center', va='bottom', fontsize=12)

plt.xlabel('Dataset')
plt.ylabel('Number of video files')
plt.title('Number of file video in each dataset')
plt.tight_layout()
plt.show()



train = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
test = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')

ss = pd.read_csv('/kaggle/input/nexar-collision-prediction/sample_submission.csv')


train = train.sort_values(by='id')
test = test.sort_values(by='id')
ss = ss.sort_values(by='id')


train.head()


test.head()


ss.head()


# Read the image locations
train_filenames = glob.glob('/kaggle/input/nexar-collision-prediction/train/*.mp4')
test_filenames = glob.glob('/kaggle/input/nexar-collision-prediction/test/*.mp4')

# Sort by id
train_filenames = sorted(train_filenames)
test_filenames = sorted(test_filenames)


# Hàm trích xuấxuất ID từ tên file 
video_path = '/kaggle/input/nexar-collision-prediction/train/00000.mp4'
def get_id(video_path):
    match = re.search(r'(\d+)\.mp4$', video_path)
    return match.group(1) if match else None
get_id(video_path)


WIDTH = 1280
HEIGHT = 720


def get_metadata(video_paths):
    fps_values = []
    frame_counts = []
    total_durations = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Error: Cannot open video file.')
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps if fps > 0 else 0
        
        fps_values.append(fps)
        frame_counts.append(frame_count)
        total_durations.append(total_duration)
        cap.release()

    results = {
        'fps': fps_values,
        'frame_count': frame_counts,
        'total_duration': total_durations
        }
    
    return results


# Thêm metadata vào train/test (nếu chưa có)
if 'fps' not in train.columns:
    train_metadata = get_metadata(train_filenames)
    for key in train_metadata:
        train[key] = train_metadata[key]
if 'fps' not in test.columns:
    test_metadata = get_metadata(test_filenames)
    for key in test_metadata:
        test[key] = test_metadata[key]

# --- Phân tích metadata ---
print("--- Các cột trong train.csv ---")
print(train.info())
print("\n--- Thống kê mô tả ---")
print(train.describe())
    


def get_metadata(video_paths):
    fps_values = []
    frame_counts = []
    total_durations = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Error: Cannot open video file.')
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps if fps > 0 else 0
        
        fps_values.append(fps)
        frame_counts.append(frame_count)
        total_durations.append(total_duration)
        cap.release()

    results = {
        'fps': fps_values,
        'frame_count': frame_counts,
        'total_duration': total_durations
        }
    
    return results


# Thêm metadata vào train/test (nếu chưa có)
if 'fps' not in train.columns:
    train_metadata = get_metadata(train_filenames)
    for key in train_metadata:
        train[key] = train_metadata[key]
if 'fps' not in test.columns:
    test_metadata = get_metadata(test_filenames)
    for key in test_metadata:
        test[key] = test_metadata[key]

# --- Phân tích metadata ---
print("--- Các cột trong test.csv ---")
print(test.info())
print("\n--- Thống kê mô tả ---")
print(test.describe())
    


# 1. Phân phối nhãn (target)
plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='target')
plt.title('Label Distribution (0: No Collision, 1: Collision)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Check for imbalance
label_counts = train['target'].value_counts()
print("\n--- Label Proportions ---")
print(f"No Collision (0): {label_counts[0]} ({label_counts[0]/len(train)*100:.2f}%)")
print(f"Collision (1): {label_counts[1]} ({label_counts[1]/len(train)*100:.2f}%)")


# 2. Phân tích thời lượng video
plt.figure(figsize=(10, 6))
sns.histplot(data=test, x='total_duration', bins=30)
plt.title('Video Duration Distribution (seconds)')
plt.xlabel('Duration (s)')
plt.ylabel('Frequency')
plt.show()


# 3. Phân tích số frame
plt.figure(figsize=(10, 6))
sns.histplot(data=test, x='frame_count', bins=30)
plt.title('Frame Count Distribution')
plt.xlabel('Number of Frames')
plt.ylabel('Frequency')
plt.show()


# 4. Phân tích FPS
plt.figure(figsize=(10, 6))
sns.histplot(data=test, x='fps', bins=10)
plt.title('FPS Distribution')
plt.xlabel('FPS')
plt.ylabel('Frequency')
plt.show()


# 5. Phân tích thời gian sự kiện (time_of_event)
plt.figure(figsize=(10, 6))
sns.histplot(data=train[train['target'] == 1], x='time_of_event', bins=30, label='Collision', color='red', alpha=0.5)
sns.histplot(data=train[train['target'] == 0], x='time_of_event', bins=30, label='No Collision', color='blue', alpha=0.5)
plt.title('Event Time Distribution (time_of_event)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# 6. Phân tích thời gian cảnh báo (time_of_alert)
plt.figure(figsize=(10, 6))
sns.histplot(data=train[train['target'] == 1], x='time_of_alert', bins=30, label='Collision', color='red', alpha=0.5)
sns.histplot(data=train[train['target'] == 0], x='time_of_alert', bins=30, label='No Collision', color='blue', alpha=0.5)
plt.title('Alert Time Distribution (time_of_alert)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# 7. So sánh time_of_event và time_of_alert
train['alert_event_diff'] = train['time_of_alert'] - train['time_of_event']
plt.figure(figsize=(10, 6))
sns.histplot(data=train[train['target'] == 1], x='alert_event_diff', bins=30, color='green')
plt.title('Distribution of Time Difference (time_of_alert - time_of_event) for Collisions')
plt.xlabel('Time Difference (s)')
plt.ylabel('Frequency')
plt.show()


# The NEW main training directory of the dataset after moving
dataset_train_dir = "/kaggle/input/nexar-collision-prediction/train"

# Check if the target directory exists
if not os.path.isdir(dataset_train_dir):
    print(f"Error: Directory not found: {dataset_train_dir}")
    print(f"Please check if the '{target_subfolder}' folder exists inside '{dataset_train_dir}'")
else:
    # List all files in the target video directory
    files = os.listdir(dataset_train_dir)

    # Filter for video files based on common extensions
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))]

    if not video_files:
        print(f"No video files found in the directory: {dataset_train_dir}")
    else:
        print(f"Found {len(video_files)} video files in {dataset_train_dir}:")
        # for i, video_file in enumerate(video_files):
        #     print(f"{i+1}. {video_file}")

        # --- Displaying a random video using HTML embedding ---

        # Check if there's at least one video file
        if video_files:
            # Choose a random video from the list
            # chosen_video_name = random.choice(video_files)
            chosen_video_name = '00822.mp4'
            video_file_path = os.path.join(dataset_train_dir, chosen_video_name)

            print(f"\nDisplaying a random video using HTML embedding: {chosen_video_name}")

            try:
                # Read the video file in binary mode
                with open(video_file_path, 'rb') as f:
                    mp4 = f.read()

                # Encode the video data in base64
                data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

                # Create and display the HTML video player
                display(HTML("""
                <video width=400 controls>
                      <source src="%s" type="video/mp4">
                </video>
                """ % data_url))

            except FileNotFoundError:
                print(f"Error: Video file not found at: {video_file_path}")
            except Exception as e:
                print(f"An error occurred during HTML embedding: {e}")
        else:
            print("No video files available to display.")


# --- Phân tích video ---
def display_frames(video_path, num_frames=5, time_of_event=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Cannot open {video_path}')
        return
    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if time_of_event is not None and fps > 0:
        # Di chuyển đến frame gần time_of_event
        cap.set(cv2.CAP_PROP_POS_MSEC, time_of_event * 1000)
        frame_count = int(time_of_event * fps)
    while frame_count < (int(time_of_event * fps) + num_frames if time_of_event else num_frames):
        ret, frame = ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
    cap.release()
    plt.figure(figsize=(15, 3))
    for i, frame in enumerate(frames):
        plt.subplot(1, min(num_frames, len(frames)), i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f'Frame {i+1}')
    plt.show()

# Hiển thị frame từ video có va chạm và không va chạm
collision_video = train[train['target'] == 1]['id'].iloc[0]
no_collision_video = train[train['target'] == 0]['id'].iloc[0]
collision_path = [f for f in train_filenames if get_id(f) == str(collision_video).zfill(5)][0]
no_collision_path = [f for f in train_filenames if get_id(f) == str(no_collision_video).zfill(5)][0]

print(f"\nHiển thị video có va chạm (ID: {collision_video})")
display_frames(collision_path, time_of_event=train[train['id'] == collision_video]['time_of_event'].iloc[0])
print(f"\nHiển thị video không va chạm (ID: {no_collision_video})")
display_frames(no_collision_path)


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

