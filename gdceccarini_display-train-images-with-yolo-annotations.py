from glob import glob
import os

import numpy as np
import pandas as pd


from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

from matplotlib import pyplot as plt
from matplotlib import patches

import seaborn as sns


glob('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/*')


submission_example = pd.read_csv('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/sample_submission.csv')
submission_example.head()


# Function to load images from a list of file paths
def load_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        # Open the image file
        img = Image.open(path)
        # Resize the image to the target size
        img = img.resize(target_size)
        # Convert the image to an array
        img_array = img_to_array(img)
        # Normalize the image array (optional, depending on your model)
        img_array /= 255.0
        images.append(img_array)
    
    # Convert the list of images to a numpy array
    return np.array(images)

# Load images
training_images_paths = glob('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/images/*')
images = load_images(training_images_paths)

# Now 'images' is a numpy array that can be used with Keras
print(images.shape)  # This will show the shape of the loaded images



def get_label(filename):
    filename = filename.split('.')[0]
    label_paths = glob('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train/labels/*')
    label_path = [x for x in label_paths if filename in x][0]
    with open(label_path, 'r') as f:
        label = f.read()
    # print(label_path)
    # print(label)
    return label

get_label('000000115.png')


images[0].shape


for i in range(10):
    plt.figure(figsize=(2, 2))
    width = 224
    height = 224
    
    filename = training_images_paths[i].split('/')[-1]
    
    label = get_label(filename)
    label_array = label.split(' ')
    yolo_coords = [float(x) for x in label_array][1:]
    # Extract YOLO coordinates
    x_center = yolo_coords[0]
    y_center = yolo_coords[1]
    box_width = yolo_coords[2]
    box_height = yolo_coords[3]

    # Convert normalized coordinates to absolute pixel values
    x_min = int((x_center - (box_width / 2)) * width)
    y_min = int((y_center - (box_height / 2)) * height)
    x_max = int((x_center + (box_width / 2)) * width)
    y_max = int((y_center + (box_height / 2)) * height)

    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(images[i])
    
    # Create a rectangle patch
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='r', facecolor='none')
    
    # Add the rectangle to the plot
    ax.add_patch(rect)
    
    plt.title(f'{filename}')
    plt.show()

