# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np
import math
import shutil


!mkdir -p /root/.cache/torch/hub/checkpoints

src = '/kaggle/input/fasterrcnn_resnet50_fpn/pytorch/default/1/fasterrcnn_resnet50_fpn.pth'   #'../input/offline-resnet50/resnet50'
dst = '/root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

shutil.copy(src, dst)


def get_folder_names(directory_path):
    """
    Gets a list of folder names in the specified directory.

    Args:
        directory_path: The path to the directory.

    Returns:
        A list of folder names.
    """
    folder_names = [entry.name for entry in os.scandir(directory_path) if entry.is_dir()]
    return folder_names


INPUT_PATH = '/kaggle/input/'
DATA_PATH = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TEST_PATH = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'
OUTPUT_PATH = 'kaggle/working'
CHECKPOINT_PATH = '/kaggle/input/bb50_fasterrcnn_resnet50_epoch_100/pytorch/default/1/fasterrcnn_resnet50_epoch_100.pth'
test_tomograms = get_folder_names('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test')


# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #model = fasterrcnn_resnet50_fpn
    

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# Initialize the model
num_classes = 2 # Background + motor

# Load the trained model
model = get_model(num_classes)

# Load the checkpoint dictionary
if torch.cuda.is_available():
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
else:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'), weights_only=False)

# Extract and load only the model's state_dict
model.load_state_dict(checkpoint['model_state_dict'])


model.to(device)
model.eval()  # Set the model to evaluation mode


def prepare_image(image_path, device):
    """Prepares a single image for model inference."""
    image = Image.open(image_path).convert("RGB") # Ensure image is RGB if your model expects it
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)


# `prediction` contains:
# - boxes: predicted bounding boxes
# - labels: predicted class labels
# - scores: predicted scores for each box (confidence level)
COCO_CLASSES = {0: "Background", 1: "Motor"}

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")



def get_motor_detections_3d(image_series_path, model, device, score_threshold=0.5, decimation=2):
    """
    Performs object detection on a series of images and extracts 3D locations and scores.

    Args:
        image_series_path (str): Path to the directory containing the image series.
        model: The trained object detection model.
        device: The device to run inference on (e.g., 'cuda' or 'cpu').
        score_threshold (float): Confidence threshold for detected objects.

    Returns:
        pandas.DataFrame: DataFrame with 'x', 'y', 'z', 'score' columns for detected motor centers.
    """
    detections_3d = []
    image_filenames = sorted(os.listdir(image_series_path)) # Assuming filenames allow sorting by slice number

    for i in range(0, len(image_filenames), decimation):
        filename = image_filenames[i]
        if filename.endswith(".jpg"): # Adjust file extension if needed
            image_path = os.path.join(image_series_path, filename)
            image_tensor = prepare_image(image_path, device)

            with torch.no_grad():
                prediction = model(image_tensor)

            # Process predictions
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # Extract center (x, y) and set z as slice number, also include score
            for box, label, score in zip(boxes, labels, scores):
                if score > score_threshold and label == 1: # Assuming label 1 is 'motor'
                    x_min, y_min, x_max, y_max = box
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    # Use 'i' as the z-coordinate (slice number)
                    detections_3d.append({'x': center_x, 'y': center_y, 'z': i, 'score': score})

    return pd.DataFrame(detections_3d)

def cluster_and_average_3d_points_with_scores(detections_df, eps=20, min_samples=5):
    """
    Clusters 3D points (including scores) and averages the locations within each cluster,
    also keeping track of the maximum score in each cluster.

    Args:
        detections_df (pandas.DataFrame): DataFrame with 'x', 'y', 'z', 'score' columns.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        pandas.DataFrame: DataFrame with 'x', 'y', 'z', 'max_score' columns for averaged 3D motor locations
                          and the maximum score within each cluster.
    """
    if detections_df.empty:
        return pd.DataFrame(columns=['x', 'y', 'z', 'max_score'])

    X = detections_df[['x', 'y', 'z']].values
    scores = detections_df['score'].values

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)

    averaged_locations_with_scores = []
    # Iterate through unique cluster labels (excluding noise, labeled as -1)
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:
            cluster_points = X[clusters == cluster_id]
            cluster_scores = scores[clusters == cluster_id]

            avg_x = np.mean(cluster_points[:, 0])
            avg_y = np.mean(cluster_points[:, 1])
            avg_z = np.mean(cluster_points[:, 2])
            max_score = np.max(cluster_scores) # Get the maximum score in the cluster

            averaged_locations_with_scores.append({'x': avg_x, 'y': avg_y, 'z': avg_z, 'max_score': max_score})

    return pd.DataFrame(averaged_locations_with_scores)

def process_tomograms_and_generate_csv_single_best(tomograms_path, model, device, output_csv_path):
    """
    Cycles through tomogram folders, processes them, and generates a CSV with the single averaged
    motor location with the highest associated confidence score for each tomogram.
    If no motors are found in a tomogram, returns -1 for all 3 motor axis values.

    Args:
        tomograms_path (str): Path to the directory containing tomogram folders.
        model: The trained object detection model.
        device: The device to run inference on (e.g., 'cuda' or 'cpu').
        output_csv_path (str): Path to save the output CSV file.
    """
    all_best_averaged_locations = []
    tomogram_folders = get_folder_names(tomograms_path)

    for tomo_id in tomogram_folders:
        image_series_path = os.path.join(tomograms_path, tomo_id)

        # Get 3D detections with scores
        motor_detections_3d_df = get_motor_detections_3d(image_series_path, model, device, score_threshold=0.5, decimation=1)

        # Cluster and average detections, also getting max score per cluster
        averaged_motor_locations_with_scores_df = cluster_and_average_3d_points_with_scores(motor_detections_3d_df, eps=25, min_samples=2)

        if averaged_motor_locations_with_scores_df.empty:
            # No motors found, add a row with -1
            all_best_averaged_locations.append({
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1
            })
        else:
            # Motors found, find the row with the highest 'max_score'
            best_row = averaged_motor_locations_with_scores_df.loc[averaged_motor_locations_with_scores_df['max_score'].idxmax()]

            all_best_averaged_locations.append({
                'tomo_id': tomo_id,
                'Motor axis 0': best_row['z'],
                'Motor axis 1': best_row['y'],
                'Motor axis 2': best_row['x']
            })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(all_best_averaged_locations)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)
    print(f"Single best averaged motor locations saved to {output_csv_path}")


# Example usage:
output_csv_filename = 'submission.csv'
output_csv_path = os.path.join('/kaggle/working/', output_csv_filename)

process_tomograms_and_generate_csv_single_best(TEST_PATH, model, device, output_csv_path)


#df = pd.read_csv('/kaggle/working/submission.csv')

#print(df)




