import shutil

# Delete the entire folder and its contents
shutil.rmtree('/kaggle/working/output_frames')# This Python 3 environment comes with many helpful analytics libraries installed
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


import shutil

# Delete the entire folder and its contents
shutil.rmtree('/kaggle/working')


!pip install ultralytics


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from ultralytics import YOLO

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video and save them as images.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file.
    output_folder : str
        Folder where frames will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f'{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def detect_objects_in_frames(model, input_folder, output_folder, conf_threshold=0.2):
    """
    Use the YOLOv8 model to detect objects in each frame in the input folder.
    
    Parameters:
    -----------
    model : YOLO
        The pre-trained YOLOv8 model.
    input_folder : str
        Folder containing frames to detect objects in.
    output_folder : str
        Folder to save the detected frames.
    conf_threshold : float, optional
        Confidence threshold for object detection (default: 0.2)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    for frame_filename in tqdm(frames, desc="Detecting Objects", unit="frame"):
        frame_path = os.path.join(input_folder, frame_filename)
        frame = cv2.imread(frame_path)
        
        # Run object detection on the frame
        results = model(frame, conf=conf_threshold)
        
        # Annotate the frame with detection results
        annotated_frame = results[0].plot()  # Plot the detected objects
        
        # Save the annotated frame
        annotated_frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(annotated_frame_path, annotated_frame)
    
    print(f"Object detection complete. Annotated frames saved to {output_folder}")

def create_animation(ims):
    """
    Create an animation from a list of image frames.
    
    Parameters:
    -----------
    ims : list or numpy array
        List of image frames.
    
    Returns:
    --------
    matplotlib animation object
    """
    fig = plt.figure(figsize=(10, 6))
    im = plt.imshow(cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.close()

    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
        return [im]
    
    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//3)

def main():
    # Path to the video file
    video_path = '/kaggle/input/nexar-collision-prediction/test/00092.mp4'
    
    # Define folders to store extracted frames and annotated frames
    frames_folder = '/kaggle/working/frames'
    annotated_frames_folder = '/kaggle/working/annotated_frames'
    
    # Extract frames from the video
    extract_frames(video_path, frames_folder)
    
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')
    
    # Run object detection on extracted frames
    detect_objects_in_frames(model, frames_folder, annotated_frames_folder)
    
    # Load the annotated frames to create animation
    annotated_frames = sorted([cv2.imread(os.path.join(annotated_frames_folder, f)) for f in os.listdir(annotated_frames_folder) if f.endswith('.png')])
    
    # Create and display the animation
    anim = create_animation(np.array(annotated_frames))
    from IPython.display import HTML
    HTML(anim.to_jshtml())

if __name__ == '__main__':
    main()



import cv2
import os

# Path to the directory containing the annotated frames
input_frames_dir = '/kaggle/working/annotated_frames'
output_video_path = '/kaggle/working/annotated_video.mp4'

# Get sorted list of frames
frames = sorted([os.path.join(input_frames_dir, f) for f in os.listdir(input_frames_dir) if f.endswith('.png')])

# Read the first frame to get the frame size
frame = cv2.imread(frames[0])
height, width, layers = frame.shape

# Create a VideoWriter object to write the frames into a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 fps

# Add frames to the video
for frame_path in frames:
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()



import os
import cv2

# Path to annotated frames folder
annotated_frames_folder = '/kaggle/working/annotated_frames'

# Get a sorted list of valid image paths
frame_paths = sorted([os.path.join(annotated_frames_folder, f) for f in os.listdir(annotated_frames_folder) if f.endswith('.png')])

# Read images while ensuring no None values
annotated_frames = []
for path in frame_paths:
    img = cv2.imread(path)
    if img is not None:
        annotated_frames.append(img)  # Append only valid images

# Check if any valid images were loaded
if not annotated_frames:
    raise ValueError("Error: No valid annotated frames found. Check if the folder contains PNG files.")

print(f"Loaded {len(annotated_frames)} annotated frames successfully!")



###LOGIC1

import os
import cv2
import numpy as np

# Path settings
annotated_frames_folder = '/kaggle/working/annotated_frames'
output_video_path = '/kaggle/working/traffic_caution_collision_video.mp4'

# Get sorted list of valid image paths
frame_paths = sorted([os.path.join(annotated_frames_folder, f) for f in os.listdir(annotated_frames_folder) if f.endswith('.png')])

# Load frames
annotated_frames = []
for path in frame_paths:
    img = cv2.imread(path)
    if img is not None:
        annotated_frames.append(img)

if not annotated_frames:
    raise ValueError("Error: No valid annotated frames found. Check if the folder contains PNG files.")

print(f"Loaded {len(annotated_frames)} annotated_frames successfully!")

# Get frame dimensions from first frame
height, width, layers = annotated_frames[0].shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# Function to check proximity in heavy traffic
def check_dashcam_proximity(frame, caution_area=20000, collision_area=100000, 
                          caution_bottom=0.15, collision_bottom=0.02, debug=False):
    """
    Adjusted for heavy traffic:
    - Caution: For typical close proximity in traffic
    - Collision Possible: Only when vehicles are extremely close (near contact)
    """
    # Placeholder contour detection (replace with your bounding box data)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caution = False
    collision = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            bottom_y = y + h
            
            # Caution conditions (lenient for traffic)
            caution_large = box_area > caution_area
            caution_close = bottom_y > height * (1 - caution_bottom)
            
            # Collision conditions (very strict)
            collision_large = box_area > collision_area
            collision_close = bottom_y > height * (1 - collision_bottom)
            
            if debug:
                print(f"Box: Area={box_area}, Bottom_Y={bottom_y}, Frame_Height={height}")
                print(f"Caution: Large={caution_large}, Close={caution_close}")
                print(f"Collision: Large={collision_large}, Close={collision_close}")
            
            # Collision check (highest priority)
            if collision_large and collision_close:  # Both conditions for collision
                collision = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.putText(frame, "COLLISION POSSIBLE!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if debug:
                    print("Collision possible triggered!")
            
            # Caution check (only if no collision)
            elif caution_large or caution_close:
                caution = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "CAUTION: Vehicle Close!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if debug:
                    print("Caution triggered!")
    
    return frame, caution, collision

# Process and write frames
caution_count = 0
collision_count = 0
for i, frame in enumerate(annotated_frames):
    # Check proximity with debug for first few frames
    debug = i < 5
    processed_frame, caution_detected, collision_detected = check_dashcam_proximity(
        frame.copy(),
        caution_area=20000,         # Suitable for traffic proximity
        collision_area=100000,      # Very large for collision
        caution_bottom=0.15,        # Bottom 15% for caution in traffic
        collision_bottom=0.02,      # Bottom 2% for collision
        debug=debug
    )
    
    if collision_detected:
        collision_count += 1
        cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 0, 255), 10)  # Red border
    elif caution_detected:
        caution_count += 1
        cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 255, 255), 10)  # Yellow border
    
    # Write frame to video
    video_writer.write(processed_frame)

# Release video writer and print summary
video_writer.release()
print(f"Video processing complete.")
print(f"Caution frames: {caution_count} ({(caution_count / len(annotated_frames)) * 100:.2f}%)")
print(f"Collision possible frames: {collision_count} ({(collision_count / len(annotated_frames)) * 100:.2f}%)")
print(f"Output saved to: {output_video_path}")


###LOGIC2

import os
import cv2
import numpy as np
from scipy.spatial import distance

# Path settings
annotated_frames_folder = '/kaggle/working/annotated_frames'
output_video_path = '/kaggle/working/hybrid_collision_video_v2.mp4'

# Get sorted list of valid image paths
frame_paths = sorted([os.path.join(annotated_frames_folder, f) for f in os.listdir(annotated_frames_folder) if f.endswith('.png')])

# Load frames
annotated_frames = []
for path in frame_paths:
    img = cv2.imread(path)
    if img is not None:
        annotated_frames.append(img)

if not annotated_frames:
    raise ValueError("Error: No valid annotated_frames found. Check if the folder contains PNG files.")

print(f"Loaded {len(annotated_frames)} annotated_frames successfully!")

# Get frame dimensions from first frame
height, width, layers = annotated_frames[0].shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# Function for hybrid spatial-temporal collision detection
def check_hybrid_proximity(frame, prev_centroids, frame_num, caution_dist=100, collision_dist=50, 
                          closure_rate_threshold=10, debug=False):
    """
    Adjusted hybrid model:
    - Relaxed collision_dist and closure_rate_threshold
    - Improved centroid tracking with stricter matching
    """
    dashcam_pos = (width // 2, height)
    
    # Placeholder contour detection (replace with bounding box data)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get current centroids
    current_centroids = {}
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            centroid_id = f"{x}_{y}"
            current_centroids[centroid_id] = (centroid_x, centroid_y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    caution = False
    collision = False
    for curr_id, curr_centroid in current_centroids.items():
        dist_to_dashcam = distance.euclidean(dashcam_pos, curr_centroid)
        
        # Temporal check with improved matching
        closure_rate = None
        if prev_centroids and frame_num > 0:
            # Match centroids within a reasonable radius (e.g., 50 pixels) to avoid mismatches
            prev_dists = {prev_id: distance.euclidean(prev_centroid, curr_centroid) 
                         for prev_id, prev_centroid in prev_centroids.items()}
            if prev_dists:
                closest_prev_id = min(prev_dists, key=prev_dists.get)
                if prev_dists[closest_prev_id] < 50:  # Only match if close enough
                    prev_dist_to_dashcam = distance.euclidean(dashcam_pos, prev_centroids[closest_prev_id])
                    closure_rate = prev_dist_to_dashcam - dist_to_dashcam
        
        if debug:
            print(f"Frame {frame_num}, Centroid {curr_centroid}: Dist={dist_to_dashcam:.2f}, "
                  f"Closure Rate={closure_rate if closure_rate is not None else 'N/A'}")
        
        # Collision: Relaxed distance and closure rate
        if (dist_to_dashcam < collision_dist and 
            (closure_rate is not None and closure_rate > closure_rate_threshold)):
            collision = True
            cv2.circle(frame, curr_centroid, 5, (0, 0, 255), -1)
            cv2.putText(frame, "COLLISION POSSIBLE!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if debug:
                print("Collision possible triggered!")
        
        # Caution: Moderately close
        elif dist_to_dashcam < caution_dist:
            caution = True
            cv2.circle(frame, curr_centroid, 5, (0, 255, 255), -1)
            cv2.putText(frame, "CAUTION: Vehicle Close!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if debug:
                print("Caution triggered!")
    
    cv2.circle(frame, dashcam_pos, 5, (255, 0, 0), -1)
    return frame, caution, collision, current_centroids

# Process and write frames
caution_count = 0
collision_count = 0
prev_centroids = None
for i, frame in enumerate(annotated_frames):
    debug = i < 10  # Extended debug for first 10 frames
    processed_frame, caution_detected, collision_detected, current_centroids = check_hybrid_proximity(
        frame.copy(),
        prev_centroids,
        frame_num=i,
        caution_dist=80,           # Unchanged
        collision_dist=40,          # Relaxed from 30 to 50
        closure_rate_threshold=15,  # Relaxed from 20 to 10
        debug=debug
    )
    
    if collision_detected:
        collision_count += 1
        cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 0, 255), 10)
    elif caution_detected:
        caution_count += 1
        cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 255, 255), 10)
    
    video_writer.write(processed_frame)
    prev_centroids = current_centroids

# Release video writer and print summary
video_writer.release()
print(f"Video processing complete.")
print(f"Caution frames: {caution_count} ({(caution_count / len(annotated_frames)) * 100:.2f}%)")
print(f"Collision possible frames: {collision_count} ({(collision_count / len(annotated_frames)) * 100:.2f}%)")
print(f"Output saved to: {output_video_path}")

