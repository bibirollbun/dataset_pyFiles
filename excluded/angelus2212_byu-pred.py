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


!pip install numpy scipy scikit-image


import os
import glob
import csv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress FutureWarnings (e.g., for torch.cuda.amp.autocast)
warnings.filterwarnings('ignore', category=FutureWarning)

# --------------------------------------------------------------------
# Load YOLO Model (using YOLOv5s pretrained on COCO)
# --------------------------------------------------------------------
def load_yolo_model():
    """
    Loads the YOLOv5s model using torch.hub.
    This model is pretrained on COCO and serves as a placeholder.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
    return model

# --------------------------------------------------------------------
# Process a Single 2D Slice
# --------------------------------------------------------------------
def process_slice(image, model, conf_threshold=0.5):
    """
    Processes a single 2D image slice using the YOLO model.
    
    Parameters:
      image: PIL.Image instance (can be grayscale or RGB)
      model: YOLO detection model
      conf_threshold: confidence threshold to filter detections

    Returns:
      A list of detections; each detection is an array containing:
      [xmin, ymin, xmax, ymax, confidence, class]
    """
    # Convert to RGB if not already, since YOLO expects 3 channels.
    if image.mode != 'RGB':
        image = image.convert('RGB')
    results = model(image)
    # The results are stored in results.xyxy[0] as a tensor.
    detections = results.xyxy[0].cpu().numpy()
    # Filter out detections with confidence below the threshold.
    detections = [det for det in detections if det[4] >= conf_threshold]
    return detections

# --------------------------------------------------------------------
# Process an Entire Tomogram (Directory of Slices)
# --------------------------------------------------------------------
def process_tomogram_yolo(tomo_dir, model):
    """
    Processes all slices in a tomogram directory. For each slice, it runs object
    detection using the YOLO model and tracks the detection with the highest
    confidence. The best detection’s bounding box center is combined with the
    slice index to form a 3D coordinate [z, y, x].
    
    Parameters:
      tomo_dir: directory path containing JPEG slices
      model: YOLO detection model
      
    Returns:
      A list with the 3D coordinate or [-1, -1, -1] if no detection is found.
    """
    slice_paths = sorted(glob.glob(os.path.join(tomo_dir, "*.jpg")))
    best_detection = None
    best_conf = -1
    best_slice_index = -1

    for i, slice_path in enumerate(slice_paths):
        try:
            image = Image.open(slice_path)
        except Exception as e:
            print(f"Error loading {slice_path}: {e}")
            continue

        detections = process_slice(image, model, conf_threshold=0.5)
        for det in detections:
            conf = det[4]
            if conf > best_conf:
                best_conf = conf
                best_detection = det
                best_slice_index = i

    if best_detection is not None:
        xmin, ymin, xmax, ymax, conf, cls = best_detection
        # Compute the center of the bounding box.
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        return [best_slice_index, y_center, x_center]
    else:
        return [-1, -1, -1]

# --------------------------------------------------------------------
# Save Submission CSV File
# --------------------------------------------------------------------
def save_submission(results, output_file='submission.csv'):
    """
    Saves the list of results to a CSV file in the required submission format.
    
    Each row is: [tomo_id, Motor axis 0, Motor axis 1, Motor axis 2]
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
        writer.writerows(results)

# --------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------
def main():
    # Set the test directory path (ensure this path is correct in your Kaggle environment)
    test_dir = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
    
    # Load the YOLO model (using YOLOv5s pretrained on COCO)
    model = load_yolo_model()
    results = []
    
    # Get a list of tomogram directories.
    tomo_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    # Process each tomogram with a progress bar.
    for tomo_id in tqdm(tomo_dirs, desc="Processing tomograms"):
        tomo_path = os.path.join(test_dir, tomo_id)
        coord = process_tomogram_yolo(tomo_path, model)
        results.append([tomo_id] + coord)
    
    # Save the results to a CSV file.
    save_submission(results)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()





