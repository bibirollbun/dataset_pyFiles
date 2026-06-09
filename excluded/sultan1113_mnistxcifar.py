# Install the Ultralytics YOLO package (only needed if not already installed)
!pip3 install ultralytics  

# Import necessary libraries
import os  # Provides functions for interacting with the operating system
from ultralytics import YOLO  # Imports the YOLO model from Ultralytics for object detection
import shutil  # Used for file operations like copying and deleting files
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img  # Functions for image processing in Keras
import numpy as np  # Library for numerical computations
from PIL import Image  # Python Imaging Library for handling image files
import random  # Provides functions for generating random numbers



# Ensure the "Dataset/train" and "Dataset/test" directories exist, creating them if necessary.

# Define the folder name for the training dataset
folder_name = "Dataset/train"  # Replace with your desired folder name

# Check if the folder exists; if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)  # Create the directory
    print(f"Folder '{folder_name}' created successfully.")  # Notify user
else:
    print(f"Folder '{folder_name}' already exists.")  # Notify if it already exists

# Define the folder name for the testing dataset
folder_name = "Dataset/test"  # Replace with your desired folder name

# Check if the folder exists; if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)  # Create the directory
    print(f"Folder '{folder_name}' created successfully.")  # Notify user
else:
    print(f"Folder '{folder_name}' already exists.")  # Notify if it already exists


# Copy files and subdirectories from the source to the destination, ensuring the destination exists.

source = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/train"  # Update with your source path
destination = "/kaggle/working/Dataset/train"  # Destination path

# Ensure the destination folder exists
os.makedirs(destination, exist_ok=True)

# Copy files instead of the whole directory to avoid errors
for item in os.listdir(source):
    src_path = os.path.join(source, item)
    dst_path = os.path.join(destination, item)
    
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)  # Allows overwriting
    else:
        shutil.copy2(src_path, dst_path)

print("Folder copied successfully in Kaggle!")



# Move 15% of images from each class in the training dataset to the test dataset for validation.

# Define source and destination folders
source_folder = "/kaggle/working/Dataset/train"
destination_folder = "/kaggle/working/Dataset/test"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Iterate through each class folder in the source directory
for class_name in os.listdir(source_folder):
    class_source_path = os.path.join(source_folder, class_name)
    class_destination_path = os.path.join(destination_folder, class_name)

    # Create class folder in destination if it doesn't exist
    os.makedirs(class_destination_path, exist_ok=True)

    # Get list of images in the current class folder
    images = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]

    # Calculate the number of images to move (20%)
    num_images_to_move = int(0.15 * len(images))

    # Randomly select images to move
    images_to_move = random.sample(images, num_images_to_move)

    # Move the selected images to the destination folder
    for image in images_to_move:
        source_path = os.path.join(class_source_path, image)
        destination_path = os.path.join(class_destination_path, image)
        shutil.move(source_path, destination_path)


# Load a model
model = YOLO("yolo11l-cls.yaml")  # build a new model from YAML
# Train the model
results = model.train(data="/kaggle/working/Dataset", epochs=60, imgsz=248)


!zip -r /kaggle/working/file.zip /kaggle/working/runs


# Load a trained YOLO model and predict labels for test images, then save results to a CSV file.

# Load the YOLO model
model = YOLO("/kaggle/working/runs/classify/train/weights/last.pt") # Path to your trained model

# Define the directory containing the images
image_dir = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/test" # Path to your images

# Initialize an empty list to store the results
results_list = []

# Iterate through all the image files in the directory
count = 0
for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)
    results = model(image_path)  # predict on an image
    top_prediction = results[0].probs.top1

    results_list.append([filename.split(".")[0], int(top_prediction)])
    print(filename, top_prediction, count)
    count += 1



import csv
with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Label'])
    writer.writerows(results_list)










