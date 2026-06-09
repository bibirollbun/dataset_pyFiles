!pip install ultralytics matplotlib seaborn opencv-python pandas tqdm


import os, shutil

# Define input paths (these come from the Kaggle dataset)
input_train_dir = "/kaggle/input/wheat-plant-diseases/data/train"
input_valid_dir = "/kaggle/input/wheat-plant-diseases/data/valid"
input_test_dir  = "/kaggle/input/wheat-plant-diseases/data/test"

# Define working directories
base_dir = "/kaggle/working/datasets/GlobalWheat2021"
train_img_dir = os.path.join(base_dir, "images", "train_flat")
valid_img_dir = os.path.join(base_dir, "images", "valid_flat")
test_img_dir  = os.path.join(base_dir, "images", "test")
train_lbl_dir = os.path.join(base_dir, "labels", "train")
valid_lbl_dir = os.path.join(base_dir, "labels", "valid")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(valid_lbl_dir, exist_ok=True)

# Create a mapping from class names to numeric labels by listing subdirectories in input_train_dir
classes = sorted([d for d in os.listdir(input_train_dir) if os.path.isdir(os.path.join(input_train_dir, d))])
if len(classes)==0:
    classes = ["wheat_disease"]
class_to_id = {cls: i for i, cls in enumerate(classes)}
print("Class mapping:", class_to_id)

# Function to copy images from a source folder (which may contain subfolders) to a destination folder
# and create a corresponding YOLO annotation (full image: 0.5,0.5,1.0,1.0) based on the folder (class) name.
def process_folder(src_folder, dest_img_dir, dest_lbl_dir):
    for cls in os.listdir(src_folder):
        cls_path = os.path.join(src_folder, cls)
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Copy image to destination (use the original filename)
                    src_img = os.path.join(cls_path, file)
                    dest_img = os.path.join(dest_img_dir, file)
                    shutil.copy(src_img, dest_img)
                    # Write label file: one line with <class_id> 0.5 0.5 1.0 1.0
                    label_file = os.path.join(dest_lbl_dir, f"{os.path.splitext(file)[0]}.txt")
                    with open(label_file, "w") as f:
                        f.write(f"{class_to_id.get(cls,0)} 0.5 0.5 1.0 1.0\n")
        else:
            # If no subfolders (flat structure), treat all images as class 0
            if cls.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_img = os.path.join(src_folder, cls)
                dest_img = os.path.join(dest_img_dir, cls)
                shutil.copy(src_img, dest_img)
                label_file = os.path.join(dest_lbl_dir, f"{os.path.splitext(cls)[0]}.txt")
                with open(label_file, "w") as f:
                    f.write("0 0.5 0.5 1.0 1.0\n")

# Process training and validation sets
process_folder(input_train_dir, train_img_dir, train_lbl_dir)
process_folder(input_valid_dir, valid_img_dir, valid_lbl_dir)

# For test images, we only need to copy them (no labels)
if os.path.isdir(input_test_dir):
    for file in os.listdir(input_test_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_img = os.path.join(input_test_dir, file)
            dest_img = os.path.join(test_img_dir, file)
            shutil.copy(src_img, dest_img)
            
print("Data copying and label creation complete.")



%%writefile GlobalWheat2021.yaml
path: /kaggle/working/datasets/GlobalWheat2021
train: images/train_flat
val: images/valid_flat
test: images/test

names:
  0: wheat_disease
# If there are more classes, list them here (order according to class_to_id)


import cv2
import matplotlib.pyplot as plt
import os

def show_image_with_annotation(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    # Read the annotation
    with open(lbl_path, "r") as f:
        line = f.readline().strip().split()
    # YOLO format: class x_center y_center width height (normalized)
    cls, xc, yc, w, h = line
    h_img, w_img = img.shape[:2]
    # Denormalize box coordinates
    xc, yc, w, h = float(xc)*w_img, float(yc)*h_img, float(w)*w_img, float(h)*h_img
    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title(f"{os.path.basename(img_path)} | Class: {cls}")
    plt.axis('off')
    plt.show()

# Show 3 random training images with annotations
import random
train_imgs = os.listdir(train_img_dir)
for img_file in random.sample(train_imgs, 3):
    img_path = os.path.join(train_img_dir, img_file)
    lbl_path = os.path.join(train_lbl_dir, f"{os.path.splitext(img_file)[0]}.txt")
    if os.path.exists(lbl_path):
        show_image_with_annotation(img_path, lbl_path)



!wget -O yolov12n.pt https://github.com/yourusername/yolov12/releases/download/v1.0/yolov12n.pt



# Download the YOLOv12 configuration file
!wget -O yolo12.yaml https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/12/yolo12.yaml

# Import and instantiate the model using the config file
from ultralytics import YOLO
model = YOLO('yolo12.yaml')

# Print model summary using the info() method
model.info()


print(model)


import os, cv2, shutil
from tqdm import tqdm

# Define input directories (as provided by the Kaggle dataset)
# (These paths depend on the dataset structure on Kaggle.)
input_train_dir = "/kaggle/input/wheat-plant-diseases/data/train"
input_valid_dir = "/kaggle/input/wheat-plant-diseases/data/valid"
input_test_dir  = "/kaggle/input/wheat-plant-diseases/data/test"

# Define working (output) directories in Kaggle working area
base_dir = "/kaggle/working/datasets/GlobalWheat2021"
train_img_out = os.path.join(base_dir, "images", "train_flat")
valid_img_out = os.path.join(base_dir, "images", "valid_flat")
test_img_out  = os.path.join(base_dir, "images", "test")

train_lbl_out = os.path.join(base_dir, "labels", "train_flat")
valid_lbl_out = os.path.join(base_dir, "labels", "valid_flat")

# Create directories if they don't exist
for d in [train_img_out, valid_img_out, test_img_out, train_lbl_out, valid_lbl_out]:
    os.makedirs(d, exist_ok=True)

# Function to process a folder of images: copy valid images to a flat folder and generate a label file.
# For each valid image, we generate a label with a full-image box: "0 0.5 0.5 1.0 1.0"
def process_images(input_dir, img_out_dir, lbl_out_dir):
    # If images are organized in subfolders, traverse them
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                src = os.path.join(root, file)
                # Try to read image using cv2; if not valid, skip.
                img = cv2.imread(src)
                if img is None:
                    # Skip corrupt image formats (e.g. GIF) or unreadable files.
                    continue
                # Save to flat folder using the original filename.
                dest_img = os.path.join(img_out_dir, file)
                shutil.copy(src, dest_img)
                # Generate label file (full image box)
                # YOLO expects: <class> <x_center> <y_center> <width> <height> (all normalized)
                # For a full image box, that is: "0 0.5 0.5 1.0 1.0"
                label_filename = os.path.splitext(file)[0] + ".txt"
                dest_lbl = os.path.join(lbl_out_dir, label_filename)
                with open(dest_lbl, "w") as f:
                    f.write("0 0.5 0.5 1.0 1.0\n")
                count += 1
    print(f"Processed {count} images from {input_dir}.")

# Process training, validation, and test images.
process_images(input_train_dir, train_img_out, train_lbl_out)
process_images(input_valid_dir, valid_img_out, valid_lbl_out)

# For test images, we only need to copy them (no labels needed)
def process_test_images(input_dir, img_out_dir):
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                src = os.path.join(root, file)
                img = cv2.imread(src)
                if img is None:
                    continue
                dest_img = os.path.join(img_out_dir, file)
                shutil.copy(src, dest_img)
                count += 1
    print(f"Processed {count} test images from {input_dir}.")

process_test_images(input_test_dir, test_img_out)



import cv2
import matplotlib.pyplot as plt
import random

def show_image_with_label(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    with open(lbl_path, "r") as f:
        line = f.readline().strip().split()
    # YOLO format: class x_center y_center width height (normalized)
    cls, xc, yc, w, h = line
    h_img, w_img = img.shape[:2]
    # Denormalize bounding box coordinates
    xc, yc, w, h = float(xc)*w_img, float(yc)*h_img, float(w)*w_img, float(h)*h_img
    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title(f"{os.path.basename(img_path)} | Label: {line}")
    plt.axis('off')
    plt.show()

# Get a few random training images
train_imgs = os.listdir(train_img_out)
for file in random.sample(train_imgs, 3):
    img_path = os.path.join(train_img_out, file)
    lbl_path = os.path.join(train_lbl_out, os.path.splitext(file)[0] + ".txt")
    if os.path.exists(lbl_path):
        show_image_with_label(img_path, lbl_path)



import os
import cv2
import shutil
from tqdm import tqdm

# Define input validation folder (from Kaggle dataset) and output folders for validation images/labels.
input_valid_dir = "/kaggle/input/wheat-plant-diseases/data/valid"
valid_img_out = "/kaggle/working/datasets/GlobalWheat2021/images/valid_flat"
valid_lbl_out = "/kaggle/working/datasets/GlobalWheat2021/labels/valid_flat"

os.makedirs(valid_img_out, exist_ok=True)
os.makedirs(valid_lbl_out, exist_ok=True)

# Since this is a classification dataset, we create a pseudo-detection label for each image.
# Each image gets a label file with a full-image bounding box: "0 0.5 0.5 1.0 1.0"
def process_valid_images(input_dir, dest_img_dir, dest_lbl_dir):
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                src_img = os.path.join(root, file)
                # Attempt to read image to skip corrupt ones
                img = cv2.imread(src_img)
                if img is None:
                    continue
                # Copy the image to the flat destination
                dest_img = os.path.join(dest_img_dir, file)
                shutil.copy(src_img, dest_img)
                # Write label file: a full-image bounding box for class 0.
                label_filename = os.path.splitext(file)[0] + ".txt"
                dest_lbl = os.path.join(dest_lbl_dir, label_filename)
                with open(dest_lbl, "w") as f:
                    f.write("0 0.5 0.5 1.0 1.0\n")
                count += 1
    print(f"Processed {count} validation images.")

process_valid_images(input_valid_dir, valid_img_out, valid_lbl_out)



import os
import cv2
import matplotlib.pyplot as plt
import random

# Define directories for validation images and labels
valid_img_dir = "/kaggle/working/datasets/GlobalWheat2021/images/valid_flat"
valid_lbl_dir = "/kaggle/working/datasets/GlobalWheat2021/labels/valid_flat"

def show_valid_image_with_label(image_file):
    # Construct full paths for the image and its label file
    img_path = os.path.join(valid_img_dir, image_file)
    lbl_path = os.path.join(valid_lbl_dir, os.path.splitext(image_file)[0] + ".txt")
    
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    h, w = img.shape[:2]
    
    # Check if label file exists and then read the annotation(s)
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            lines = f.readlines()
        # For each annotation in the label file
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            # YOLO format: <class> <x_center> <y_center> <width> <height> (all normalized)
            cls, xc, yc, bw, bh = parts
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            # Denormalize bounding box coordinates
            box_w = bw * w
            box_h = bh * h
            x_center = xc * w
            y_center = yc * h
            x1 = int(x_center - box_w/2)
            y1 = int(y_center - box_h/2)
            x2 = int(x_center + box_w/2)
            y2 = int(y_center + box_h/2)
            # Draw the bounding box (red rectangle)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No label file found for", image_file)
    
    # Convert BGR to RGB and display the image using matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(image_file)
    plt.show()

# Get a list of all validation images
valid_images = os.listdir(valid_img_dir)
if len(valid_images) == 0:
    print("No validation images found in", valid_img_dir)
else:
    # Display 3 random validation images with their labels
    for image_file in random.sample(valid_images, min(3, len(valid_images))):
        show_valid_image_with_label(image_file)



from ultralytics import YOLO

# If you have a pretrained YOLOv12 checkpoint, use it here.
# For example, if you have 'yolov12n.pt', you can load it like this:
# model = YOLO('yolov12n.pt')

# Otherwise, we instantiate the model from the config file (which builds the architecture).
# This example uses the YOLOv12 configuration we downloaded earlier.
model = YOLO('yolo12.yaml')

# Now train the model using the YAML configuration file that defines the dataset.
# Adjust epochs, image size, and batch size as needed.
model.train(data='/kaggle/working/GlobalWheat2021.yaml', epochs=100, imgsz=640, batch=16)



model.save('trained_100_yolov12.pt')


import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Define the test images directory
test_img_dir = '/kaggle/input/wheat-plant-diseases/data/test'
test_images = os.listdir(test_img_dir)

if len(test_images) == 0:
    print("No test images found in:", test_img_dir)
else:
    # Loop through a few test images (e.g., first 5 images)
    for img_file in test_images[:5]:
        img_path = os.path.join(test_img_dir, img_file)
        # Run inference on the image
        results = model(img_path)
        # results is a list; iterate and call .show() on each result object
        for result in results:
            result.show()  # This displays the image with predictions inline





