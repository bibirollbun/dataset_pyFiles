pip install --upgrade nbconvert traitlets



import numpy as np 
import pandas as pd 
import os

# Load the training and test data
df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')

# Pad the 'id' column with leading zeros to ensure 5-digit IDs
df["id"] = df["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

# Define directories containing train and test videos
train_dir = "/kaggle/input/nexar-collision-prediction/train/"
test_dir = "/kaggle/input/nexar-collision-prediction/test/"

# Create video filenames by appending ".mp4" to the ID
df["train_videos"] = df["id"] + ".mp4"
df_test["test_videos"] = df_test["id"] + ".mp4"

# Display sample IDs
print(f"Sample Train IDs:\n{df['id'].head()}")
print(f"Sample Test IDs:\n{df_test['id'].head()}")

# Display the total number of videos
print(f"Total Train Videos: {len(df['train_videos'])}")
print(f"Total Test Videos: {len(df_test['test_videos'])}")



import cv2
from multiprocessing import Pool, cpu_count

# Optical Flow calculation function (CPU version)
def compute_optical_flow_gpu(video_info):  
    video_name, video_dir, alert_time, event_time = video_info
    video_path = os.path.join(video_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"â�Œ Video couldnt opened: {video_path}")
        return None  

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default FPS 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    alert_frame = int(alert_time * fps) if not pd.isna(alert_time) else 0
    event_frame = int(event_time * fps) if not pd.isna(event_time) else 0

    flow_features = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for frame_count in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break  
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CPU Optical Flow Calculation
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        std_magnitude = np.std(magnitude)

        flow_features.append([video_name, frame_count, mean_magnitude, max_magnitude, std_magnitude])

        prev_gray = gray

    cap.release()

    df = pd.DataFrame(flow_features, columns=["video_id", "frame", "mean_magnitude", "max_magnitude", "std_magnitude"])
    return df

# Process all videos and compute Optical Flow (CPU version)
def process_videos_cpu(df, train_dir):  
    video_list = df[['train_videos', 'time_of_alert', 'time_of_event']].values.tolist()
    video_list = [(video_name, train_dir, alert_time, event_time) for video_name, alert_time, event_time in video_list]

    print(f"ğŸ”„ {len(video_list)} videos will be processed with CPU...")

    with Pool(cpu_count()) as pool:
        results = list(pool.map(compute_optical_flow_gpu, video_list))

    all_results = pd.concat([df for df in results if df is not None], ignore_index=True)
    
    return all_results

# Run CPU Optical Flow
optical_flow_results = process_videos_cpu(df.iloc[0:2], train_dir)



# Save results to CSV 
optical_flow_results.to_csv("optical_flow_results.csv", index=False)



print(optical_flow_results.head())

