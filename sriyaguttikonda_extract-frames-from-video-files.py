import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set paths
metadata_file = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json"
input_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
face_save_path = "/kaggle/working/"  # Change save path to "faces"

# Constants
STRIDE = 1.0
FACE_SIZE = (128, 128)  # Resize cropped face to consistent dimensions
MAX_IMAGE_SIZE = 1024

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_frames_from_video(video_file, stride=1.0):
    """Extract frames from video based on stride interval."""
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    i = 0.
    frames = []
    frame_times = []

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            frame_times.append(i)
            i += stride
            video.set(1, round(i * fps))  # Move to next frame position
        else:
            video.release()
            break
    return frames, frame_times

def detect_and_crop_face(image):
    """Detect and crop the largest face in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No face detected

    # Select the largest detected face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_crop = image[y:y+h, x:x+w]  # Crop face region

    # Resize face to maintain uniform size
    face_crop = cv2.resize(face_crop, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    return face_crop

def process_video(input_path, save_path, key, stride, max_image_size):
    """Extract frames, detect faces, crop, and save cropped faces."""
    video_file = os.path.join(input_path, key)
    frames, frame_times = get_frames_from_video(video_file, stride)

    file_name, _ = key.split(".")
    face_folder = os.path.join(save_path, file_name)

    if not os.path.isdir(face_folder):
        os.makedirs(face_folder)

    for frame, frame_time in zip(frames, frame_times):
        face = detect_and_crop_face(frame)

        if face is not None:
            image_name = str(round(frame_time, 3)).replace(".", "_")
            cv2.imwrite(os.path.join(face_folder, f"{image_name}.jpg"), face)  # Save cropped face

# Load metadata
with open(metadata_file) as json_file:
    metadata = json.load(json_file)

# Process first video
key = list(metadata.keys())[0]
process_video(input_path, face_save_path, key, STRIDE, MAX_IMAGE_SIZE)

# Compute storage efficiency
def get_folder_size(path):
    total_size = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(path) for f in filenames if not os.path.islink(os.path.join(dirpath, f)))
    return total_size

video_size = os.path.getsize(os.path.join(input_path, key))
faces_folder = os.path.join(face_save_path, key.split(".")[0])
total_faces_size = get_folder_size(faces_folder)

# Print storage comparison
print(f"Video file size: {video_size}")
print(f"Total extracted face images size: {total_faces_size}")
print(f"Compression Ratio: {video_size / total_faces_size:.4f}")


import matplotlib.pyplot as plt
import cv2
import os

def print_cropped_faces(faces_folder, max_faces=10):
    """Displays cropped face images in a grid layout, similar to original frames."""
    face_images = sorted(os.listdir(faces_folder))[:max_faces]  # Limit number of displayed faces

    plt.figure(figsize=(20, 10))
    columns = 5  # Number of columns in the grid
    for i, face_file in enumerate(face_images):
        img = cv2.imread(os.path.join(faces_folder, face_file))  # Load face image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format

        plt.subplot(len(face_images) // columns + 1, columns, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Frame {face_file.split('.')[0].replace('_', '.')}s")  # Show frame time

    plt.tight_layout()
    plt.show()

# Example usage
faces_folder = "/kaggle/working/aagfhgtpmv"  # Replace with actual folder path
print_cropped_faces(faces_folder)


import matplotlib.pyplot as plt
import cv2
import os

def print_cropped_faces(faces_folder, max_faces=10):
    """Displays cropped face images in a grid and prints X/Y axis values."""
    face_images = sorted(os.listdir(faces_folder))[:max_faces]  # Limit number of displayed faces

    plt.figure(figsize=(20, 10))
    columns = 5  # Number of columns in the grid
    axes_list = []  # Store axis references

    for i, face_file in enumerate(face_images):
        img = cv2.imread(os.path.join(faces_folder, face_file))  # Load face image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format

        ax = plt.subplot(len(face_images) // columns + 1, columns, i + 1)
        plt.imshow(img)
        plt.axis('on')  # Keep axis visible for values
        plt.title(f"Frame {face_file.split('.')[0].replace('_', '.')}s")

        axes_list.append(ax)

    plt.tight_layout()
    plt.show()

    # ðŸ”¹ Print X and Y axis limits for each subplot
    for i, ax in enumerate(axes_list):
        x_lim = ax.get_xlim()  # Get X-axis range
        y_lim = ax.get_ylim()  # Get Y-axis range
        print(f"Grid {i+1}: X-axis range {x_lim}, Y-axis range {y_lim}")

# Example usage
faces_folder = "/kaggle/working/aagfhgtpmv"  # Replace with your folder path
print_cropped_faces(faces_folder)


import shutil

# Define the path of the directory you want to delete
dir_path = "/kaggle/working/aagfhgtpmv"  # Replace with your actual folder path

# Delete the entire directory
shutil.rmtree(dir_path)

print(f"Deleted directory: {dir_path}")

