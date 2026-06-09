!pip install ultralytics


import glob  # helps in finding files by matching patterns
import os  # helps in interacting with the computer’s operating system
import random  # helps in generating random numbers or making random choices
import shutil  # helps in copying and moving files/folders

from IPython.display import Image, display  # helps in showing images in a notebook
import matplotlib.pyplot as plt # helps in arranging images
import matplotlib.image as mpimg # helps in displaying images
import kagglehub  # helps with working on Kaggle projects and data
from ultralytics import YOLO  # lets us use the YOLO tool for object detection
import yaml  # helps in reading and writing YAML files (a text format)
from tqdm import tqdm # to display progress bar
import pandas as pd # # enables the use of all Pandas functions and features, such as creating data frames, reading CSV files, and performing data analysis.


!yolo train \
    model=yolo11n.pt \
    data=/kaggle/input/stem-challenge-task-3-competition/task_3_train_val/data.yaml \
    epochs=20 \
    imgsz=1280 \
    device=0,1 \
    project=/kaggle/working


# View confusion matrix
display(Image(filename='/kaggle/working/train/confusion_matrix.png'))


MODEL_WEIGHTS = "/kaggle/working/train/weights/best.pt"


def yolo_kaggle_predict(input_model_weights: str, input_image_dir: str, output_csv: str) -> None:
    """
    Predict bounding boxes on images in input_image_dir using the YOLO model specified by input_model_weights.
    Writes predictions to output_csv in the format: (conf, class_id, x_center, y_center, width, height).

    Args:
        input_model_weights (str): Path to the trained YOLO model weights file.
        input_image_dir (str): Directory path containing images to run inference on.
        output_csv (str): Path where the resulting CSV file containing predictions will be saved.

    Returns:
        None: This function saves results directly to a CSV file and does not return any values.
    """
    # Load the YOLO model with the provided model weights.
    model = YOLO(input_model_weights)
    # Run the YOLO model inference on all images in the provided directory.
    preds = model(input_image_dir)
    # Initialize an empty DataFrame to store predictions with columns for image_id and bounding boxes.
    output_df = pd.DataFrame(columns=["image_id", "bbox"])
    # Iterate through each prediction (each corresponding to one image).
    for pred in preds:
        # Extract the filename of the current image to serve as a unique image identifier.
        image_id = os.path.basename(pred.path)
        # Initialize an empty list to store bounding boxes for the current image.
        boxes = []
        # Iterate over each detected bounding box in the prediction.
        for bbox in pred.boxes:
            # Extract confidence score of the bounding box prediction.
            conf = float(bbox.conf)
            # Extract predicted class ID for the detected object.
            class_id = int(bbox.cls)
            # Extract normalized bounding box coordinates (x_center, y_center, width, height).
            xywhn = [float(val) for val in bbox.xywhn[0]]
            print('bbox:', bbox.xywhn[0])
            # Append the bounding box data as a list to the boxes list.
            boxes.append([conf, class_id, *xywhn])
        # Create a new DataFrame row containing the image ID and its associated bounding boxes.
        new_row = pd.DataFrame({"image_id": [image_id], "bbox": [boxes]})

        # Concatenate the new row to the existing output DataFrame.
        output_df = pd.concat([output_df, new_row], ignore_index=True)

    # Save the final predictions DataFrame to a CSV file without row indices.
    output_df.to_csv(output_csv, index=False)

    # Print a message indicating how many rows (images) were successfully processed and saved.
    print(f"Wrote {len(output_df)} rows to {output_csv}")

yolo_kaggle_predict(MODEL_WEIGHTS, "/kaggle/input/stem-challenge-task-3-competition/task_3_test/images/test", "/kaggle/working/submission.csv")




