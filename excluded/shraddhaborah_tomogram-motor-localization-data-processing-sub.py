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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Define file paths
train_labels_path = "/kaggle/input/train-data/train_labels.csv"  # Ensure this file is uploaded
train_dir = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"  # Root directory of tomogram slices

# Load training labels with error handling
try:
    train_labels_df = pd.read_csv(train_labels_path)
    print("Training labels loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file {train_labels_path} was not found. Please upload it.")

# Function to load tomogram slices into a 3D numpy array
def load_tomogram(tomo_id):
    tomo_path = os.path.join(train_dir, tomo_id)
    
    # Check if tomogram directory exists
    if not os.path.exists(tomo_path):
        print(f"Error: Tomogram directory {tomo_path} not found.")
        return None
    
    slice_files = sorted(os.listdir(tomo_path))  # Sort to maintain slice order
    slices = []
    
    for f in slice_files:
        slice_path = os.path.join(tomo_path, f)
        image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            slices.append(image)
        else:
            print(f"Warning: Could not read {slice_path}")
    
    if len(slices) == 0:
        print(f"Error: No valid images found in {tomo_path}")
        return None
    
    return np.array(slices)

# Select a sample tomogram from train_labels.csv
if not train_labels_df.empty:
    sample_tomo_id = train_labels_df["tomo_id"].iloc[0]
    sample_tomogram = load_tomogram(sample_tomo_id)

    if sample_tomogram is not None:
        # Get motor locations for this tomogram
        motor_positions = train_labels_df[train_labels_df["tomo_id"] == sample_tomo_id][["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values

        # 3D Visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot motor locations
        ax.scatter(motor_positions[:, 2], motor_positions[:, 1], motor_positions[:, 0], c='r', marker='o', s=40, label='Motors')

        # Set plot limits and labels
        ax.set_xlim([0, sample_tomogram.shape[2]])
        ax.set_ylim([0, sample_tomogram.shape[1]])
        ax.set_zlim([0, sample_tomogram.shape[0]])
        ax.set_xlabel("X-axis (Width)")
        ax.set_ylabel("Y-axis (Height)")
        ax.set_zlabel("Z-axis (Slices)")
        ax.set_title(f"3D Tomogram View: {sample_tomo_id}")

        plt.legend()
        plt.show()
    else:
        print("Failed to load tomogram. Please check the file structure.")
else:
    print("Error: Training labels file is empty or not loaded.")


import pandas as pd

# Define the output file path
output_csv_path = "/kaggle/working/submission.csv"

# Initialize an empty list to store rows before creating a DataFrame
submission_data = []

# Process each tomogram in the dataset
if not train_labels_df.empty:
    for tomo_id in train_labels_df["tomo_id"].unique():
        # Get motor locations for this tomogram
        motors = train_labels_df[train_labels_df["tomo_id"] == tomo_id][["Motor axis 0", "Motor axis 1", "Motor axis 2"]]

        # Store motor locations in a list
        for _, row in motors.iterrows():
            submission_data.append([tomo_id, row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])

    # Convert the list to a DataFrame
    submission_df = pd.DataFrame(submission_data, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])

    # Save the formatted results to a CSV file
    submission_df.to_csv(output_csv_path, index=False)
    print(f"✅ Submission file generated successfully: {output_csv_path}")

else:
    print("❌ Error: No valid training data found.")


