# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

# Limit the number of files to display (e.g., 10)
max_files = 10
file_count = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        file_count += 1
        if file_count >= max_files:  # Stop when reaching maximum count
            print("... more files omitted ...")
            break
    if file_count >= max_files:
        break
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import libraries
import cv2
from tqdm import tqdm

# Suppress NaN warning messages
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np


# 1. Load CSV files
def load_data(train_csv_path='train.csv', test_csv_path='test.csv'):
    """
    Load training and testing data from CSV files.
    """
    # Load CSV files
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Format IDs as 5-digit numbers
    train_df['id'] = train_df['id'].apply(lambda x: f"{int(float(x)):05d}")
    test_df['id'] = test_df['id'].apply(lambda x: f"{int(float(x)):05d}")
    
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)
    
    return train_df, test_df


# 2. Get video file information
def check_video_info(video_path):
    """
    Check basic information about a video file.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration_seconds': duration
    }


# 3. Extract sample frames from videos
def extract_sample_frames(video_path, num_frames=5):
    """
    Extract sample frames from a video for visualization.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


# 4. Explore sample data (both positive and negative examples)
def explore_sample_data(train_df, video_dir='train', num_samples=2):
    """
    Explore sample positive and negative examples.
    """
    # Select positive samples
    positive_samples = train_df[train_df['target'] == 1].sample(num_samples)
    
    # Select negative samples
    negative_samples = train_df[train_df['target'] == 0].sample(num_samples)
    
    samples = pd.concat([positive_samples, negative_samples])
    
    results = []
    for _, row in samples.iterrows():
        video_path = os.path.join(video_dir, f"{row['id']}.mp4")
        video_info = check_video_info(video_path)
        
        if video_info:
            result = {
                'id': row['id'],
                'target': row['target'],
                'time_of_event': row.get('time_of_event', 'N/A'),
                'time_of_alert': row.get('time_of_alert', 'N/A'),
                'video_info': video_info
            }
            results.append(result)
    
    return results


# Main function
def explore_data(train_csv='train.csv', test_csv='test.csv', train_dir='train'):
    """
    Main function to explore the dataset.
    """
    # 1. Load CSV files
    train_df, test_df = load_data(train_csv, test_csv)
    
    # 2. Print basic data information
    print("\nTraining data columns:", train_df.columns.tolist())
    print("\nSample of training data:")
    print(train_df.head())
    
    # 3. Check label distribution
    if 'target' in train_df.columns:
        print("\nTarget distribution:")
        print(train_df['target'].value_counts())
    
    # 4. Explore sample data
    print("\nExploring sample videos...")
    sample_results = explore_sample_data(train_df, train_dir)
    
    for i, result in enumerate(sample_results):
        print(f"\nSample {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  Target: {result['target']} ({'Positive/Accident' if result['target'] == 1 else 'Negative/Normal'})")
        print(f"  Event time: {result['time_of_event']}")
        print(f"  Alert time: {result['time_of_alert']}")
        print(f"  Video info: {result['video_info']}")
    
    return train_df, test_df


# Run the code
if __name__ == "__main__":
    # Find CSV file paths
    train_csv_path = None
    test_csv_path = None
    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename == 'train.csv':
                train_csv_path = os.path.join(dirname, filename)
            elif filename == 'test.csv':
                test_csv_path = os.path.join(dirname, filename)
    
    # Find train video directory path
    train_video_dir = None
    for dirname, _, filenames in os.walk('/kaggle/input'):
        if os.path.basename(dirname) == 'train' and any(f.endswith('.mp4') for f in filenames):
            train_video_dir = dirname
            break
    
    # Run the exploration if paths are found
    if train_csv_path and test_csv_path and train_video_dir:
        print(f"CSV file paths:\n- train.csv: {train_csv_path}\n- test.csv: {test_csv_path}")
        print(f"Video directory: {train_video_dir}")
        
        train_df, test_df = explore_data(train_csv_path, test_csv_path, train_video_dir)
    else:
        print("Could not find required files or directories.")


def perform_eda(train_df, train_video_dir):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        train_df: Training dataframe
        train_video_dir: Directory containing training videos
    """
    print("\n## Exploratory Data Analysis ##")
    
    # 1. Analyze time-related features for accident cases
    analyze_time_features(train_df)
    
    # 2. Analyze video properties
    analyze_video_properties(train_df, train_video_dir)
    
    # 3. Analyze optical flow across samples (movement detection)
    analyze_sample_optical_flow(train_df, train_video_dir)


def analyze_time_features(train_df):
    """
    Analyze time-related features for accident cases
    """
    # Focus on positive examples (accident cases)
    accident_df = train_df[train_df['target'] == 1].copy()
    
    # Calculate reaction time (difference between event and alert)
    accident_df['reaction_time'] = accident_df['time_of_event'] - accident_df['time_of_alert']
    
    print(f"\n1. Time Feature Analysis (Accident Cases Only - {len(accident_df)} samples)")
    print("\nReaction time statistics (seconds):")
    print(accident_df['reaction_time'].describe())
    
    # Create visualization for reaction time
    plt.figure(figsize=(10, 6))
    sns.histplot(accident_df['reaction_time'], bins=20, kde=True)
    plt.title('Distribution of Reaction Time (Event Time - Alert Time)')
    plt.xlabel('Reaction Time (seconds)')
    plt.ylabel('Count')
    plt.axvline(accident_df['reaction_time'].mean(), color='red', linestyle='--', 
               label=f'Mean: {accident_df["reaction_time"].mean():.2f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze event timing distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(accident_df['time_of_event'], bins=20, kde=True)
    plt.title('Distribution of Event Times in Videos')
    plt.xlabel('Event Time (seconds)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_video_properties(train_df, train_video_dir, sample_size=100):
    """
    Analyze properties of videos in the dataset
    
    Args:
        train_df: Training dataframe
        train_video_dir: Directory containing training videos
        sample_size: Number of videos to sample for analysis
    """
    print(f"\n2. Video Properties Analysis (Sample of {sample_size} videos)")
    
    # Sample videos for analysis
    sampled_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    
    # Collect video properties
    video_properties = []
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Analyzing videos"):
        video_path = os.path.join(train_video_dir, f"{row['id']}.mp4")
        video_info = check_video_info(video_path)
        
        if video_info:
            video_info['target'] = row['target']
            video_properties.append(video_info)
    
    # Convert to DataFrame
    video_df = pd.DataFrame(video_properties)
    
    if len(video_df) > 0:
        # Display summary statistics
        print("\nVideo property statistics:")
        print(video_df.describe())
        
        # Compare duration between positive and negative cases
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='target', y='duration_seconds', data=video_df)
        plt.title('Video Duration by Class')
        plt.xlabel('Class (0=Normal, 1=Accident)')
        plt.ylabel('Duration (seconds)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Check resolution distribution
        plt.figure(figsize=(10, 6))
        video_df['resolution'] = video_df['width'].astype(str) + 'x' + video_df['height'].astype(str)
        resolution_counts = video_df['resolution'].value_counts()
        resolution_counts.plot(kind='bar')
        plt.title('Video Resolution Distribution')
        plt.xlabel('Resolution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No valid video properties found in the sample.")


def analyze_sample_optical_flow(train_df, train_video_dir, num_samples=5):
    """
    Analyze optical flow in sample videos to detect motion patterns
    
    Args:
        train_df: Training dataframe
        train_video_dir: Directory containing training videos
        num_samples: Number of samples to analyze from each class
    """
    print(f"\n3. Optical Flow Analysis (Sample of {num_samples*2} videos)")
    
    # Select samples from each class
    positive_samples = train_df[train_df['target'] == 1].sample(num_samples)
    negative_samples = train_df[train_df['target'] == 0].sample(num_samples)
    samples = pd.concat([positive_samples, negative_samples])
    
    # Function to calculate average optical flow magnitude
    def get_optical_flow(video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # Get total frames and calculate frame indices
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - num_frames - 1, num_frames, dtype=int)
        
        flows = []
        prev_frame = None
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Calculate magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flows.append(np.mean(magnitude))
            
            prev_frame = gray
            
        cap.release()
        return flows if flows else None
    
    # Calculate optical flow for samples
    results = []
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="Calculating optical flow"):
        video_path = os.path.join(train_video_dir, f"{row['id']}.mp4")
        flows = get_optical_flow(video_path)
        
        if flows:
            results.append({
                'id': row['id'],
                'target': row['target'],
                'mean_flow': np.mean(flows),
                'max_flow': np.max(flows),
                'flow_values': flows
            })
    
    if results:
        # Convert to DataFrame
        flow_df = pd.DataFrame([
            {'id': r['id'], 'target': r['target'], 'mean_flow': r['mean_flow'], 'max_flow': r['max_flow']} 
            for r in results
        ])
        
        # Display results
        print("\nOptical flow statistics by class:")
        print(flow_df.groupby('target')[['mean_flow', 'max_flow']].describe())
        
        # Visualize mean flow by class
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='target', y='mean_flow', data=flow_df)
        plt.title('Mean Optical Flow Magnitude by Class')
        plt.xlabel('Class (0=Normal, 1=Accident)')
        plt.ylabel('Mean Flow Magnitude')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot flow over time for a few samples
        plt.figure(figsize=(12, 8))
        for i, result in enumerate(results[:4]):  # Plot first 4 samples
            plt.subplot(2, 2, i+1)
            plt.plot(result['flow_values'])
            plt.title(f"Sample {result['id']} (Class {result['target']})")
            plt.xlabel('Frame Index')
            plt.ylabel('Flow Magnitude')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid optical flow results found in the samples.")


# Feature extraction for model building
def extract_basic_features(train_df, train_video_dir, sample_size=None):
    """
    Extract basic features from videos for model building
    
    Args:
        train_df: Training dataframe
        train_video_dir: Directory containing training videos
        sample_size: Number of videos to sample (None for all)
        
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of extracted features
    """
    # Sample data if requested
    if sample_size is not None:
        df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    else:
        df = train_df
    
    features = []
    labels = []
    
    print("Extracting features from videos...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(train_video_dir, f"{row['id']}.mp4")
        
        # Check if video can be opened
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        # Basic video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames for analysis
        num_sample_frames = min(10, frame_count)
        indices = np.linspace(0, frame_count - 1, num_sample_frames, dtype=int)
        
        # Features to extract
        avg_brightness = []
        avg_motion = []
        prev_gray = None
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness.append(np.mean(gray))
            
            # Motion (optical flow)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_motion.append(np.mean(magnitude))
                
            prev_gray = gray
        
        cap.release()
        
        # Compile features
        video_features = [
            width, 
            height,
            fps,
            duration,
            np.mean(avg_brightness) if avg_brightness else 0,
            np.std(avg_brightness) if len(avg_brightness) > 1 else 0,
            np.mean(avg_motion) if avg_motion else 0,
            np.max(avg_motion) if avg_motion else 0,
            np.std(avg_motion) if len(avg_motion) > 1 else 0
        ]
        
        # For positive samples, add time features
        if row['target'] == 1:
            video_features.extend([
                row['time_of_event'],
                row['time_of_alert'],
                row['time_of_event'] - row['time_of_alert']
            ])
        else:
            # For negative samples, use zeros for time features
            video_features.extend([0, 0, 0])
        
        features.append(video_features)
        labels.append(row['target'])
    
    # Feature names for reference
    feature_names = [
        'width', 'height', 'fps', 'duration', 
        'avg_brightness', 'std_brightness',
        'avg_motion', 'max_motion', 'std_motion',
        'event_time', 'alert_time', 'reaction_time'
    ]
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Extracted {X.shape[1]} features from {X.shape[0]} videos")
    
    return X_scaled, y, feature_names


# Add this to your main function
def analyze_and_extract_features(train_df, test_df, train_video_dir):
    """
    Perform analysis and feature extraction
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        train_video_dir: Directory containing training videos
        
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of extracted features
    """
    # Perform exploratory data analysis
    perform_eda(train_df, train_video_dir)
    
    # Extract features for model building (optional)
    # Uncomment to extract features
    # X, y, feature_names = extract_basic_features(train_df, train_video_dir, sample_size=100)
    # return X, y, feature_names
    
    return None, None, None


# Run the code
if __name__ == "__main__":
    import os

    # Find CSV file paths
    train_csv_path = None
    test_csv_path = None
    train_video_dir = None
    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename == 'train.csv':
                train_csv_path = os.path.join(dirname, filename)
            elif filename == 'test.csv':
                test_csv_path = os.path.join(dirname, filename)
    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        if os.path.basename(dirname) == 'train' and any(f.endswith('.mp4') for f in filenames):
            train_video_dir = dirname
            break
    
    # Run EDA functions only
    if train_csv_path and test_csv_path and train_video_dir:
        train_df, test_df = explore_data(train_csv_path, test_csv_path, train_video_dir)
        
        if train_df is not None:
            print("\nStarting additional data analysis...")
            
            # Time features analysis
            analyze_time_features(train_df)
            
            # Video properties analysis (reduced sample size)
            analyze_video_properties(train_df, train_video_dir, sample_size=20)
            
            # Optical flow analysis (reduced sample size)
            analyze_sample_optical_flow(train_df, train_video_dir, num_samples=2)


