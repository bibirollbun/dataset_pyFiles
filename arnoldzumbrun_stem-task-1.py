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
from tqdm import tqdm # helps in display progress bar
import pandas as pd # enables the use of all Pandas functions and features, such as creating data frames, reading CSV files, and performing data analysis.


# Load the pre-trained YOLO model
yolo_model = YOLO("yolo11n.pt")


# List all the test images and randomly select 4
test_images_dir = "/kaggle/input/stem-challenge-task-1-competition/task_1_test/images/test"
test_images = os.listdir(test_images_dir)
selected_images = [os.path.join(test_images_dir, random_image) for random_image in random.sample(test_images, 4)]

# Create directory for predictions if it doesn't exist
predictions_dir = "/kaggle/working/predictions"
os.makedirs(predictions_dir, exist_ok=True)

# Run the model prediction on each image and save the result
predictions = []
for image in selected_images:
    results = yolo_model(image)[0] 
    output_path = os.path.join(predictions_dir, os.path.basename(image))
    # save an image with the detected bounding boxes drawn on it, along with class labels and confidence scores
    results.save(filename=output_path) 
    predictions.append(output_path)

# Create a set of subplots arranged in a single row, where each subplot corresponds to an item in the predictions
fig, axes = plt.subplots(1, len(predictions), figsize=(15, 5))
for ax, prediction in zip(axes, predictions):
    img = mpimg.imread(prediction)
    ax.imshow(img) # # Display each prediction on its subplot
    ax.axis('off')  # Hide the axes
plt.tight_layout()
plt.show()


def yolo_kaggle_predict(input_model_weights: str, input_image_dir: str, output_csv: str) -> None:
    """Predict bounding boxes on images in input_image_dir using the YOLO model in input_model_weights.
    Write the predictions to output_csv in the format: (class_id, x_center, y_center, width, height)"""
    # Load the model
    model = YOLO(input_model_weights)
    preds = model(input_image_dir)
    output_df = pd.DataFrame(columns=["image_id", "bbox"])
    for pred in preds:
        image_id = os.path.basename(pred.path)
        # Get the bounding boxes
        boxes = []
        for bbox in pred.boxes:
            conf = float(bbox.conf)
            class_id = int(bbox.cls)
            xywhn = [float(val) for val in bbox.xywhn[0]]
            boxes.append([conf, class_id, *xywhn])
        new_row = pd.DataFrame({"image_id": [image_id], "bbox": [boxes]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)
    # Write the output_csv file using pandas
    output_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(output_df)} rows to {output_csv}")

yolo_kaggle_predict("yolo11n.pt", "/kaggle/input/stem-challenge-task-1-competition/task_1_test/images/test", "/kaggle/working/submission.csv")




