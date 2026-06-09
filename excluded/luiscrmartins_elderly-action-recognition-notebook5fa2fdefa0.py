!pip install fiftyone > /dev/null


!pip install -U transformers > /dev/null


!pip install torch > /dev/null


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

from datetime import datetime

import random
from IPython.display import display, Video

import cv2

import matplotlib.pyplot as plt

from IPython.display import HTML

import fiftyone as fo
import fiftyone.core.labels as fol

import os

import shutil


class CFG:
    # paths
    input_test_path = "/kaggle/input/elderly-action-recognition-challenge-at-wacv-2025/eval_FO_ids/"
    
    ETRI_sample_path = "/kaggle/input/etri-activity-simplified/ETRI-Activity3D_Sample_en/"
    ETRI_sample_videos_path = "SampleVideos/"
    ETRI_sample_overview = "ETRI-Activity3D_Overview.xlsx"
    working_path = "/kaggle/working/"

    # Structure the categories and actions into a DataFrame
    data = {
        "Category": [
            "locomotion",
            "manipulation",
            "communication",
            "hygiene",
            "eating_drinking",
            "leisure"
        ],
        "Actions": [
            "walking, entering a space, leaving a space, sitting, standing, sitting down, getting up, lying down",
            "cooking, making coffee, making tea, wiping table, spreading bedding/folding bedding, cleaning dishes, vacuuming, looking for something, putting on/taking off shoes, putting on/taking off glasses, cleaning, writing",
            "talking, beckoning, waving a hand, clapping, pointing, shaking hands, phone calls, using a telephone, hugging",
            "washing face, washing hands, brushing teeth, brushing hair, massaging a shoulder oneself, taking medicine",
            "eating, drinking",
            "watching tv, reading, using a laptop, using a tablet, exercising"
        ]
    }
    
    # Convert to DataFrame
    df_categories_actions = pd.DataFrame(data)


def analyze_files(path):
    """
    Analyzes files in a given directory path and prints general information.

    Parameters:
        path (str): Path to the directory containing files.
    """
    file_list = []
    extensions = Counter()
    file_sizes = []

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            file_list.append(filename)
            ext = os.path.splitext(filename)[1].lower()  # Get file extension
            extensions[ext] += 1
            file_path = os.path.join(dirname, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            file_sizes.append(file_size)

    # Print general information
    print(f"Total number of files: {len(file_list)}")
    print("File types and counts:", extensions)
    print("Sample file names:", file_list[:10])
    print(f"Average file size: {sum(file_sizes) / len(file_sizes):.2f} MB")
    print(f"Min file size: {min(file_sizes):.2f} MB")
    print(f"Max file size: {max(file_sizes):.2f} MB")
# Example usage:
# analyze_files("path/to/directory")



def analyze_video_metadata(path):
    """
    Analyzes video metadata (FPS, frame count, duration) in a given directory 
    and generates histograms and a frequency table.

    Parameters:
        path (str): Path to the directory containing video files.
    """
    # Lists to store metadata
    fps_list = []
    frame_count_list = []
    duration_list = []

    # Get video files
    video_files = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(dirname, filename))

    # Extract metadata
    for video in video_files:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if fps > 0 and frame_count > 0:
            fps_list.append(round(fps))  # Round FPS to nearest integer
            frame_count_list.append(frame_count)
            duration_list.append(duration)

    if not fps_list:
        print("No valid video metadata found.")
        return

    # Compute FPS frequencies
    fps_counts = pd.Series(fps_list).value_counts().sort_index()

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # FPS Histogram
    bins = np.arange(min(fps_list), max(fps_list) + 2) - 0.5  # Create bins for each unique FPS value
    n, bins, patches = axes[0].hist(fps_list, bins=bins, edgecolor='black', color='green')

    # Add labels above bars
    for count, patch in zip(n, patches):
        if count > 0:
            axes[0].text(patch.get_x() + patch.get_width() / 2, count, str(int(count)), 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[0].set_xlabel("Frames Per Second (FPS)")
    axes[0].set_ylabel("Number of Videos")
    axes[0].set_title("Distribution of FPS")

    # Frame Count Histogram
    axes[1].hist(frame_count_list, bins='auto', edgecolor='black', color='green')
    axes[1].set_xlabel("Frame Count")
    axes[1].set_ylabel("Number of Videos")
    axes[1].set_title("Distribution of Frame Counts")

    # Duration Histogram
    axes[2].hist(duration_list, bins='auto', edgecolor='black', color='green')
    axes[2].set_xlabel("Duration (seconds)")
    axes[2].set_ylabel("Number of Videos")
    axes[2].set_title("Distribution of Video Durations")

    plt.tight_layout()
    plt.show()

    # Print FPS frequency table
    fps_freq_table = pd.DataFrame({'FPS': fps_counts.index, 'Frequency': fps_counts.values})
    print(fps_freq_table)

# Example usage:
# analyze_video_metadata("path/to/directory")


def extract_video_metadata(input_path):
    """
    Extracts metadata from video files in the given directory and returns a DataFrame.

    Parameters:
        input_path (str): Path to the directory containing video files.

    Returns:
        pd.DataFrame: DataFrame containing video metadata (video name, path, FPS, frame count, duration).
    """
    # List to store video metadata
    video_metadata = []

    # Iterate through files in the input folder
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                file_path = os.path.join(dirname, filename)
                
                # Extracting metadata using OpenCV
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Append metadata to the list
                video_metadata.append({
                    "video_name": filename,
                    "path": file_path,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration": duration
                })

    # Create and return DataFrame
    return pd.DataFrame(video_metadata)

# Example usage:
# df_videos = extract_video_metadata(CFG.input_test_path)
# print(df_videos.head())  # To preview the DataFrame

def analyze_video_channels(directory_path):
    """
    Scans a directory for .mp4 videos, analyzes their color channels,
    and determines if they are RGB or RGB+D.

    Parameters:
        directory_path (str): Path to the directory containing video files.

    Returns:
        pd.DataFrame: A DataFrame with file names and detected format (RGB or RGB+D).
    """
    video_data = []

    # Iterate through all .mp4 files in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)

                # Open video with OpenCV
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    num_channels = frame.shape[2] if len(frame.shape) == 3 else 1
                    format_type = "RGB+D" if num_channels == 4 else "RGB"
                else:
                    format_type = "Could not read"

                # Store results
                video_data.append({"file_name": file, "format": format_type})

    # Convert to DataFrame
    df_videos = pd.DataFrame(video_data)

    return df_videos

# Example usage:
# df_results = analyze_video_channels("/path/to/videos")
# print(df_results)


analyze_files(CFG.input_test_path)


analyze_video_metadata(CFG.input_test_path)


df_videos = extract_video_metadata(CFG.input_test_path)


df_videos.to_csv(f"{CFG.working_path}video_metadata.csv", index=False)


pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent automatic line breaks
pd.set_option("display.width", 1000)  # Set output width
print(df_videos.head(10))  # Display more rows if needed


# Structure the categories and actions into a DataFrame
print(CFG.df_categories_actions)


xls_file = f'{CFG.ETRI_sample_path}{CFG.ETRI_sample_overview}'
SampleVideos_folder = f'{CFG.ETRI_sample_path}{CFG.ETRI_sample_videos_path}'


# Check if the Excel file exists and load it
if os.path.exists(xls_file):
    print(f"âœ… Excel file found: {CFG.ETRI_sample_overview}")
    xls = pd.ExcelFile(xls_file, engine="openpyxl")
else:
    print("â�Œ Excel file not found.")

# List files in the video folder
if os.path.exists(SampleVideos_folder):
    video_files = os.listdir(SampleVideos_folder)
    print(f"ğŸ“‚ Videos in '{SampleVideos_folder}':")
    for file in video_files:
        print(f"   - {file}")
else:
    print("â�Œ Video folder not found.")


print("Available sheets:", xls.sheet_names)


# Load only the 'Overview' sheet
df_overview = pd.read_excel(xls, sheet_name='Sheet1', engine="openpyxl")


display(df_overview.head())  # Show first few rows


analyze_files(SampleVideos_folder)


analyze_video_metadata(SampleVideos_folder)


df_ETRI_videos = extract_video_metadata(SampleVideos_folder)


pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent automatic line breaks
pd.set_option("display.width", 1000)  # Set output width
print(df_ETRI_videos.head(15))  # Display more rows if needed


df_results = analyze_video_channels(SampleVideos_folder)
print(df_results)


print(f"âœ… 'Overview' shape: {df_overview.shape}")
display(df_overview.head())


# # Create a new dataset
# dataset = fo.Dataset("train_ds")

# # Add videos to the dataset
# dataset.add_dir(
#     SampleVideos_folder,
#     dataset_type=fo.types.VideoDirectory,
# )


# # Launch the FiftyOne app (May not work fully in Kaggle)
# session = fo.launch_app(dataset)


# # Generate a unique annotation key with a timestamp
# anno_key = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# print("Annotation Key:", anno_key)


# # Export videos to CVAT for annotation
# dataset.annotate(
#     anno_key=anno_key,
#     label_field="ground_truth",
#     backend="cvat",
#     launch_editor=True,  # Opens the CVAT annotation tool
# )
















