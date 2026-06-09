import pandas as pd
import os
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import numpy.linalg as la


# A helpful little cell that is essentially a reset button
# shutil.rmtree("/kaggle/working/dataset")


# This variable controls how many neighboring slices from each side are used as positive examples.
num = 4


# This cell sets up the structure the YOLO model is expecting
# Define paths for the dataset directory
working_dir = "/kaggle/working/dataset"

# Define paths for the label directories, for both train and test
labels_train = os.path.join(working_dir, "labels", "train")
labels_val = os.path.join(working_dir, "labels", "val")
labels_test = os.path.join(working_dir, "labels", "test")

# Create directories
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)
os.makedirs(labels_test, exist_ok=True)



# Define the same path as earlier for the label directory, to use when adding files to this directory
label_dir = "/kaggle/working/dataset/labels/"

# As we randomly sort tomogram labels into the test and train sets, we'll keep track so we can add
# the tomogram images to the right set later on
test_ids = set()
val_ids = set()

# Load up the dataframe of labels of the train set
df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

# This line turns the training labels DataFrame into an easily iterable dictionary
tomo_dict = {tomo_id: group.drop(columns="tomo_id").to_dict(orient="records") 
             for tomo_id, group in df.groupby("tomo_id")}

# Iterate through the keys of tomo_dict, which are the tomo_id tags
for tomo_id in tomo_dict:
    # The values of tomo_dict are lists of dictionaries, each dict representing a motor
    motors = tomo_dict.get(tomo_id, []) 
    
    if motors[0]['Number of motors'] == 0:
        continue # Skip non-motor slices

    box_size = 40 # Size of the bounding box around the motor

    # Get the width and height of the tomogram
    img_width, img_height = int(motors[0]['Array shape (axis 1)']), int(motors[0]['Array shape (axis 2)'])

    # Randomly sort this tomogram into train or test, with an 80-20 split
    token = np.random.rand()
    if token > 0.9:
        label_base = os.path.join(label_dir, "test") # We will add the label to this directory later
        test_ids.add(tomo_id)
    elif token > 0.8:
        label_base = os.path.join(label_dir, "val")
        val_ids.add(tomo_id)
    else:
        label_base = os.path.join(label_dir, "train") # We will add the label to this directory later

    # Now we iterate through the motors within the tomogram
    for i, motor in enumerate(motors):
        # Get the (x, y) coordinate of the motor on its slice
        # We don't save the z coordinate, or the slice number, because we are training on 2-D images
        y_coord, x_coord =  int(motor['Motor axis 1']), int(motor['Motor axis 2'])

        # Here we normalize the (x,y) coordinates and the width of the bounding box using the width and height of the image
        # Each value will be in between 0 and 1, representing a proportion of the image's dimensions
        x_center = x_coord / img_width
        y_center = y_coord / img_height
        width = box_size / img_width
        height = box_size / img_height

        # Create labels for the two tomograms above and below the true-labeled slice
        # We will use the same x,y coordinate, as it should still be centered. This will bolster our training set.
        for n in range(-num, num + 1):
            # Create the label name, made from the tomo_id, the number of the motor, and how many slices above or below the true label
            label_name = f"{tomo_id}_{i}_{n}.txt"
    
            # Combine that label name with label_base (train or test) to create the full file path
            label_path = os.path.join(label_base, label_name)
    
            # Write the label to the given file path in the YOLO format
            with open(label_path, "w") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

print("Labels sorted!")


# Both contrast enhancement and gaussian noise can easily be toggled off when this function is called
def augment_image(img, contrast=False, gaussian_noise=False):
    """Applies contrast adjustment and Gaussian noise to an image."""
    # Contrast enhancement (CLAHE)
    if contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

    # Gaussian noise
    if gaussian_noise:
        # Changing the values passed into the following function change how much noise we are adding in
        noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img

# Define which directory contains the images we're moving
base_dir = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"

# Name output directories, to where we're moving the images
train_output_dir = "/kaggle/working/dataset/images/train"
val_output_dir = "/kaggle/working/dataset/images/val"
test_output_dir = "/kaggle/working/dataset/images/test"

# Ensure output directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Load DataFrame
df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

# Create a dictionary mapping each tomo_id to its rows
tomo_dict = {tomo_id: group.drop(columns="tomo_id").to_dict(orient="records") 
             for tomo_id, group in df.groupby("tomo_id")}

# Iterate through tomograms in tomo_dict
for tomo_id in tomo_dict:
    # The values of tomo_dict are lists of dictionaries, each dict representing a motor
    motors = tomo_dict.get(tomo_id, [])

    if motors[0]['Number of motors'] == 0: 
        continue # Skip non-motor slices

    # Set tomo_path to be the directory containing slices of the current tomogram
    tomo_path = os.path.join(base_dir, tomo_id)

    # Sort the tomogram into the same train/test bin as its label
    output_dir = test_output_dir if tomo_id in test_ids else (val_output_dir if tomo_id in val_ids else train_output_dir)

    # Iterate through the motors contained in the tomogram
    for i, motor in enumerate(motors):
        # Save the true labeled slice index as true_z_index
        true_z_index = int(motor['Motor axis 0'])

        # We will save the two slice-images on either side of the true labeled slice, to match their labels
        for n in range(-num, num + 1):
            # Get the index of the slice of the tomogram n slices below the true index (above if n<0)
            z_index = true_z_index + n
            slice_index = f"slice_{z_index:04d}.jpg"
            
            # Create the path for the image's output spot
            selected_slice_path = os.path.join(base_dir, tomo_id, slice_index)
            label_name = f"{tomo_id}_{i}_{n}.jpg"
            target_path = os.path.join(output_dir, label_name)
    
            # Read image and apply augmentation
            img = cv2.imread(selected_slice_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue # Skip if the slice does not exist
            img_aug = augment_image(img)
    
            # Save the augmented image to the target path
            cv2.imwrite(target_path, img_aug)

print('Slices copied and augmented successfully!')



# This cell is a sanity check or a spot-check to make sure we are correctly labeling the right images
# If everything works, we should see slices of tomagrams with a box around a little pod on the edge of the cell walls

# Define the directory where the images and their labels are located
image_dir = "/kaggle/working/dataset/images/train"
label_dir = "/kaggle/working/dataset/labels/train"

# Get a list of the names of the files in image_dir
tomograms = os.listdir(image_dir)

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.ravel()

# We'll iterate through a small interval of the saved tomogram slice images
for i in range(4):
    # Get the path of the tomogram slice, cut the extension off the basename to get the id, and print the id
    tomo = os.path.join(image_dir, tomograms[i])
    tomo_id = os.path.basename(tomo)[:-4]

    # Save the image
    image = cv2.imread(tomo)

    # Find the label .txt file and read it into the variable label
    label_path = os.path.join(label_dir, tomo_id + ".txt")
    with open(label_path, 'r') as f:
        label = f.read().split()
        # read().split() will give a list of strings describing the location and dimensions of the bounding box:
        # ['{class}', '{x_center}', '{y_center}', '{box_width}', '{box_height}']
        # Each besides class (which is always 0) will be a string representation of a float between 0 and 1

    # Get the coordinates of the motor. They are expressed in terms of the proportion of the width or height of the image, respectively
    x_center = float(label[1])
    y_center = float(label[2])

    # Get the width and height of the tomogram slice, and set the box_width and box_heigh accordingly
    width, height = image.shape[0], image.shape[1]
    box_width = float(label[3]) * width
    box_height = float(label[4]) * height

    # Multiply x_center, y_center by width and height to get their 'cartesian' coordinates
    x_coord, y_coord = int(x_center * width), int(y_center * height)

    # Use the coordinates and box dims to set up the bounding box, to be used with Matplotlib.patches
    x_min, x_max = int(x_coord - (box_width / 2)), int(x_coord + (box_width / 2))
    y_min, y_max = int(y_coord - (box_height / 2)), int(y_coord + (box_height / 2))

    # Initialize the figure and display the image in grayscale
    ax[i].imshow(image, cmap='gray')
    ax[i].set_title(f'Label for {tomo_id}')
    
    # Create and add the bounding box
    rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
    ax[i].add_patch(rect)  # Use ax.add_patch instead of plt.add_patch

    
# Show plot
plt.tight_layout()
plt.show()


# Write contents of config .yaml file. This file lets the YOLO model know where to find everything.
config = """path: /kaggle/working/dataset  # Root directory of dataset
train: images/train  # Training images relative to root
val: images/val  # Validation images relative to root
test: images/test  # Testing images relative to root

nc: 1  # Number of classes
names: ["motor"]  # Class name"""


# Create filepath
filepath = "/kaggle/working/dataset/dataset.yaml"

# Write contents to location of filepath
with open(filepath, 'w') as config_file:
    config_file.write(config)

print(f".yaml file saved to {filepath}")



# Install ultralytics package in order to load up a pretrained YOLO model.
!pip install ultralytics > /dev/null


from ultralytics import YOLO # YOLO contains a family of YOLO models

model = YOLO("yolov8n.pt")  # Load YOLOv8 small model with pretrained weights

# Train with 50 epochs, 256x256 images, with a batch size of 32 images

results = model.train(data="/kaggle/working/dataset/dataset.yaml", 
            epochs=50, # Training cap at 50 epochs
            imgsz=256, # Compress images to be 256x256
            batch=32, # Batch size of 32 images
            device='cuda', # Use tehe gpu we are connected to
            conf=0.5, # 50% of the predicted bounded box must overlap with the true bounding box to count
            val=True, # At checkpoints, test against the validation set and adjust learning rate accordingly, to avoid overfit
            patience=5, # Early stopping; if training doesn't improve five epochs in a row, stop
           save_period=5, # How often save checkpoints are, saves a bit on memory
           verbose=False) # I personally think all the default logs are excessive


dataset_dir = '/kaggle/working/dataset'

def get_ground_truth(img_file, size, dataset_dir=dataset_dir):
    """Given the file location of an image, find the true label, create a bounding box,
    and return it as a plt.patch.Rectangle object
    Args:
        img_file(str): A file path of a tomogram slice with a motor in it
        size(tup(int)): A tuple of integers representing the original dimensions of the image
        dataset_dir(str): A file path to the directory containing our images and labels
    Return:
        rect(plt.patches.Rectangle): A graph-able bounding box"""
    label_name = img_file[:-4]
    label_path = os.path.join(dataset_dir, 'labels', 'val', label_name + '.txt')

    with open(label_path, 'r') as f:
        label = f.read().split()
        # read().split() will give a list of strings describing the location and dimensions of the bounding box:
        # ['{class}', '{x_center}', '{y_center}', '{box_width}', '{box_height}']
        # Each besides class (which is always 0) will be a string representation of a float between 0 and 1

    # Get the coordinates of the motor. They are expressed in terms of the proportion of the width or height of the image, respectively
    x_center = float(label[1])
    y_center = float(label[2])

    # Get the width and height of the tomogram slice, and set the box_width and box_height accordingly
    width, height = size[0], size[1]
    box_width = float(label[3]) * width
    box_height = float(label[4]) * height

    # Multiply x_center, y_center by width and height to get their 'cartesian' coordinates
    x_coord, y_coord = int(x_center * width), int(y_center * height)

    # Use the coordinates and box dims to set up the bounding box, to be used with Matplotlib.patches
    x_min, x_max = int(x_coord - (box_width / 2)), int(x_coord + (box_width / 2))
    y_min, y_max = int(y_coord - (box_height / 2)), int(y_coord + (box_height / 2))

    rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=1, edgecolor='g', facecolor='none')

    return rect

def predict_on_samples(model, num_samples=4):
    """
    Run predictions on random validation samples and display results
    
    Args:
        model: Trained YOLO model
        num_samples (int): Number of random samples to test

    Prints out a visualization of four random predictions and the ground truth labels
    """
    # Get validation images
    val_dir = os.path.join(dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        # Try train directory instead if val doesn't exist
        val_dir = os.path.join(dataset_dir, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")

    # Create a list of the file paths to the images in the validation images directory
    val_images = os.listdir(val_dir)

    # If there are no validation images, print an error statement and end
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Iterate through validation image files
    for i, img_file in enumerate(samples):
        # Only predict on as many images as we have axes
        if i >= len(axes):
            break

        # Get the entire path to the current image
        img_path = os.path.join(val_dir, img_file)
        
        # Run prediction
        results = model.predict(img_path, conf=0.1)[0]
        
        # Load and display the image
        img = Image.open(img_path)
        img_array = np.array(img)
        axes[i].imshow(img_array, cmap='gray')
        
        # Draw ground truth box (from filename), calling our previously defined function
        m, n = img_array.shape[:2]  # Get image height (m) and width (n)
        ground_truth_rect = get_ground_truth(img_file, (m, n))
        axes[i].add_patch(ground_truth_rect)
        
        # Draw predicted boxes (red)
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            # Make a box for each prediction
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                rect_pred = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                
        # Title our figure
        axes[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")

    # Make sure nothing overlaps
    plt.tight_layout()
    
    # Save the predictions plot
    plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
    plt.show()

# Call our function and check out the images
predict_on_samples(model)



# Run validation on our model and save results to runs/validation
results = model.val(
    project="runs",  # Custom directory to save results
    name="validation"   # Subdirectory for this run
)


# Define YOLO validation results directory and filenames
val_dir = "runs/validation"
image_files = ["confusion_matrix.png", "F1_curve.png", "PR_curve.png"]
titles = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]

# Load and convert images, handling NaN and Inf values
images = []
for f in image_files:
    image_path = os.path.join(val_dir, f)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

# Create subplots
fig, ax = plt.subplots(3, 1, figsize=(18, 54))

# Display images using a loop
for i, (img, title) in enumerate(zip(images, titles)):
    ax[i].imshow(img)
    ax[i].set_title(title)
    ax[i].axis("off")

plt.tight_layout()
plt.show()



test_dir = Path('/kaggle/working/dataset/images/test')
test_ids = [tomo_id.name[:11] for tomo_id in test_dir.iterdir() if tomo_id.is_file()]
test_ids = set(test_ids)


def get_best_slice_batch(tomogram_dir, model, threshold=0.25, batch_size=16, device="cuda"):
        """
    Identifies the best slice in a tomogram based on the highest confidence prediction from a model.

    Parameters:
        tomogram_dir (str): Path to the directory containing tomogram slices as image files.
        model (torch.nn.Module): The model we trained
        threshold (float, optional): Minimum confidence threshold for valid detections. Defaults to 0.25.
        batch_size (int, optional): Number of slices to process in a batch. Defaults to 16.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        tuple:
            float: Best z-coordinate of the identified object.
            float: Best y-coordinate of the identified object.
            float: Best x-coordinate of the identified object.
            float: Confidence score of the best detection.
    """
    # Ensure model is on the correct device
    model.to(device)

    # Initialize variables that we will keep track of and update
    confidence_to_beat = 0
    best_slice_name = None
    best_z_coord, x_coord, y_coord = -1, -1, -1

    slice_images, slice_names, z_coords = [], [], []
    original_sizes = {}  # Store original image dimensions

    # Load and preprocess images
    for slice_name in os.listdir(tomogram_dir):
        if slice_name.endswith('.jpg'):
            slice_path = os.path.join(tomogram_dir, slice_name)

            # Extract z-coord from filename (adjust parsing if needed)
            try:
                z_coord = int(slice_name.split('_')[1].split('.')[0])  
            except (IndexError, ValueError):
                print(f"Warning: Could not extract z-coord from {slice_name}")
                continue

            # Read image (BGR)
            slice_image = cv2.imread(slice_path)  
            if slice_image is None:
                continue

            # Store original size so we can rescale the predictions
            original_height, original_width = slice_image.shape[:2]  
            original_sizes[slice_name] = (original_width, original_height)

            # Convert image to RGB, resize the image, and save the image, name, and z_coord to lists
            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(slice_image, (256, 256))
            slice_images.append(resized_image)
            slice_names.append(slice_name)
            z_coords.append(z_coord)

    if not slice_images:
        print("No valid images found.")
        return -1, -1, -1, -99

    # Convert to Tensor (B, C, H, W)
    slice_images = np.array(slice_images).astype(np.float32) / 255.0  # Normalize
    slice_images = torch.tensor(slice_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    # Process in batches
    for i in range(0, len(slice_images), batch_size):
        batch = slice_images[i:i + batch_size]
        batch_names = slice_names[i:i + batch_size]
        batch_z_coords = z_coords[i:i + batch_size]

        # Run inference on the batch in order to speed things up
        results = model.predict(batch, verbose=False)

        # Iterate through the results of the batch
        for j, result in enumerate(results):
            # Check if predictions were made
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                boxes = result.boxes.xywh.cpu().numpy()  # Bounding box coordinates

                if len(confidences) == 0:
                    continue

                # Find the slice in the batch with the highest confidence and save it
                best_idx_batch = np.argmax(confidences)
                best_confidence = confidences[best_idx_batch]

                # If this is the best slice so far, update our tracker variables
                if best_confidence > confidence_to_beat:
                    confidence_to_beat = best_confidence
                    best_slice_name = batch_names[j]
                    best_z_coord = batch_z_coords[j]  # Store best z-coordinate
                    box = boxes[best_idx_batch]
                    x_coord, y_coord = box[0], box[1]  # Center coordinates in resized image

                    # Get original image size
                    original_width, original_height = original_sizes[best_slice_name]

                    # Scale coordinates back to original size and update
                    scale_x = original_width / 256
                    scale_y = original_height / 256
                    x_coord_original = x_coord * scale_x
                    y_coord_original = y_coord * scale_y

                    x_coord, y_coord = x_coord_original, y_coord_original

    # Return the coordinates of the highest-confidence prediction
    return float(best_z_coord), y_coord, x_coord, confidence_to_beat


def run_inference(pred_path, test_path='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train', model=model, test_ids=test_ids):
    """Iterate through sub-directories containing tomogram images, make predictions on each one using get_best_slice_batch(),
    and save the coordinates into a dictionary that we then convert into a DataFrame that follows the format described in the competition overview. 
    We then save it to 'submission.csv', which is where Kaggle will look for our predictions when scoring.

    args: 
        pred_path(str): Location to save predictions csv to
        test_path(str): Location of the test directory containing the test set of tomograms
        model(torch.nn.Module): The trained model
        test_ids(list()): List of tomo_ids in the test set

    Saves the prediction .csv to 'submission.csv'
    """
    # Set test_dir to be a Path() object 
    test_dir = Path(test_path)

    # Initialize a dictionary to store our predictions
    sub_table = {'tomo_id': [],
                'Motor axis 0': [],
                'Motor axis 1': [],
                'Motor axis 2': []}

    # Iterate through the tomo_ids in our list of test tomograms
    for tomo_id in test_ids:
        # Get the path of the directory of images and get our model's prediction
        tomo_dir = os.path.join(test_dir, tomo_id)
        z, y, x, conf = get_best_slice_batch(tomo_dir, model)

        # Save the results in the dictionary
        sub_table['tomo_id'].append(tomo_id)
        sub_table['Motor axis 0'].append(z)
        sub_table['Motor axis 1'].append(y)
        sub_table['Motor axis 2'].append(x)

    # Convert to a pd.DataFrame and save to a csv file
    sub_df = pd.DataFrame(sub_table)
    sub_df.to_csv(pred_path, index=False)
    return sub_df


# Set our pred_path and test_dir, then make predictions
# We use the training set as our test_dir because it contains all of the directories of tomogram images
# We only predict on the tomograms in our test set
pred_path = 'test_predictions.csv'
test_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
run_inference(pred_path, test_dir)
print('Finished making predictions!')


# Our inference code only predicts one motor per tomogram, since the test data will only include tomograms with one motor or zero motors. 
# Some of the tomograms in our personal test set have more than one motor. We match the prediction with the nearest motor and score based off of that.

true_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv'
pred_path = '/kaggle/working/test_predictions.csv'
threshold = 1000

def convert_to_float(t):
    "Ensures the coordinates are floats or np.nan, which are easier to work with"
    return tuple(float(x) if x is not None and x != -1 else np.nan for x in t)

def eval(true_path=true_path, pred_path=pred_path, beta = 2, threshold=threshold):
    """
    Evaluates the model's performance according to the competiton's custom metric

    Args:
        true_path (str): Path to the CSV file containing ground truth motor locations.
        pred_path (str): Path to the CSV file containing predicted motor locations.
        beta (float, optional): The beta parameter for the F-beta score, defaulting to 2.
        threshold (float, optional): The maximum acceptable distance (in angstroms) between 
                                     a predicted motor and the nearest true motor for it to 
                                     be considered a true positive. Default is 1000.

    Returns:
        tuple: (f_beta, precision, recall, TP, TN, FP, FN), where:
            - f_beta (float): The computed F-beta score.
            - precision (float): The precision of the predictions.
            - recall (float): The recall of the predictions.
            - TP (int): Count of true positives.
            - TN (int): Count of true negatives.
            - FP (int): Count of false positives.
            - FN (int): Count of false negatives."""
    # Read in the predictions csv as a pd.DataFrame, and read in the labels as a pd.DataFrame
    prediction = pd.read_csv(pred_path)
    true = pd.read_csv(true_path)

    # Turn both the labels and predictions into easily iterable and searchable dictionaries
    key_dict = {tomo_id: group.drop(columns="tomo_id").to_dict(orient="records") 
             for tomo_id, group in true.groupby("tomo_id")}
    pred_dict = {tomo_id: group.drop(columns="tomo_id").to_dict(orient="records") 
             for tomo_id, group in prediction.groupby("tomo_id")}

    # Initialize True Positive, True Negative, False Positive, and False negative counts
    TP, TN, FP, FN = 1e-6, 1e-6, 1e-6, 1e-6 # Set these to near-zero values to avoid division by zero errors

    
    def get_coords_array(tomo_id):
        "Makes an array of coordinates for multiple motors in the image"
        coords = []
        for motor in key_dict[tomo_id]:
            coord = convert_to_float((motor['Motor axis 0'], motor['Motor axis 1'], motor['Motor axis 2']))
            coords.append(coord)
    
        return np.array(coords)

    def match_motor(pred_coords, coords_array):
        "Match a predicted motor with its nearest true motor"
        pred_coords_array = np.tile(pred_coords, (len(coords_array), 1))
        diff = coords_array - pred_coords_array
        dist = la.norm(diff, axis=1)

        return np.argmin(dist)

    # Iterate through predictions for the test set
    for tomo_id in test_ids:
        # Get the prediction for the current tomogram. If there isn't one, add to the False Negative count
        pred = pred_dict.get(tomo_id, None)
        if not pred:
            FN += 1
            continue

        # Call convert_to_float() on both the true label and the predicted location
        # We use the first of each (if there are multiple) because if there are no motors in the image, the first label will let us know
        # Likewise, if there are motors in the image, the first labeled motor will let us know. 
        # If there were multiple predictions made (There won't be using the inference code in this notebook), we only consider the first.
        X, Y, Z = convert_to_float((key_dict[tomo_id][0]['Motor axis 0'], key_dict[tomo_id][0]['Motor axis 1'], key_dict[tomo_id][0]['Motor axis 2']))
        x, y, z = convert_to_float((pred_dict[tomo_id][0]['Motor axis 0'], pred_dict[tomo_id][0]['Motor axis 1'], pred_dict[tomo_id][0]['Motor axis 2']))

        true_coord = np.array([X, Y, Z])
        pred_coord = np.array([x, y, z])

        # Check if there is a true motor.
        if np.all(np.isnan(true_coord)):
            # If there is no motor and none was predicted, add to the True Negative count
            if np.all(np.isnan(pred_coord)):
                TN += 1
            # If there is no motor and one was predicted, add to False Positive count
            elif np.all(~np.isnan(pred_coord)):
                FP += 1
            else:
                raise ValueError(f'Prediction row for {tomo_id} must contain all -1 or all numbers.')

        else:
            # If there is a true motor and none were predicted, add to False Negative count
            if np.all(np.isnan(pred_coord)):
                FN += 1
            # If there is a true motor and one was predicted, check if it is close enough
            elif np.all(~np.isnan(pred_coord)):

                # Find the nearest true motor to the predicted motor
                true_coords_array = get_coords_array(tomo_id)
                j = match_motor(pred_coord, true_coords_array)
                X, Y, Z = convert_to_float((key_dict[tomo_id][j]['Motor axis 0'], key_dict[tomo_id][j]['Motor axis 1'], key_dict[tomo_id][j]['Motor axis 2']))
                true_coord = np.array([X, Y, Z])

                # Find the euclidean distance between the motors (in voxels) and scale it to angstroms using the Voxel spacing value
                voxel_distance = la.norm(true_coord - pred_coord)
                angstrom_distance = voxel_distance * key_dict[tomo_id][0]['Voxel spacing']

                # If not within the threshold, add to False Negatives count
                if angstrom_distance > threshold:
                    FN += 1
                # If within the threshold, add to True Positives count
                else:
                    TP += 1
            else:
               raise ValueError(f'Prediction row for {tomo_id} must contain all na or all numbers.') 

    # Calculate precision, recall, and f_beta
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # Format and print results
    printer = f"""F Beta Score:\t{f_beta}\nPrecision:\t{precision}\nRecall:\t\t{recall}\n"""
    print(printer)

    return f_beta, precision, recall, TP, TN, FP, FN

# Then call the function to evaluate how the model did
eval_results = eval()
        


# This cell will save the weights of our model into our output directory. 
# These same weights can be used to initialize our trained model in a different notebook.
model_name = 'demo_yolo'
model.save(f'{model_name}.pt')


# Import statements
import numpy as np
import pandas as pd
import os
import cv2
import torch
from pathlib import Path
import csv
from ultralytics import YOLO


# Load in our trained model
model_name = 'PUT YOUR MODEL PATH HERE'
model = YOLO(f"{model_path}", task='detect')


def get_best_slice_batch(tomogram_dir, model, threshold=0.25, batch_size=16, device="cuda"):
    """Identifies the best slice in a tomogram based on the highest confidence prediction from a model.

    Parameters:
        tomogram_dir (str): Path to the directory containing tomogram slices as image files.
        model (torch.nn.Module): The model we trained
        threshold (float, optional): Minimum confidence threshold for valid detections. Defaults to 0.25.
        batch_size (int, optional): Number of slices to process in a batch. Defaults to 16.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        tuple:
            float: Best z-coordinate of the identified object.
            float: Best y-coordinate of the identified object.
            float: Best x-coordinate of the identified object.
            float: Confidence score of the best detection.
    """
    # Ensure model is on the correct device
    model.to(device)

    confidence_to_beat = 0
    best_slice_name = None
    best_z_coord, x_coord, y_coord = -1, -1, -1

    slice_images, slice_names, z_coords = [], [], []
    original_sizes = {}  # Store original image dimensions

    # Load and preprocess images
    for slice_name in os.listdir(tomogram_dir):
        if slice_name.endswith('.jpg'):
            slice_path = os.path.join(tomogram_dir, slice_name)

            # Extract z-coord from filename (adjust parsing if needed)
            try:
                z_coord = int(slice_name.split('_')[1].split('.')[0])  
            except (IndexError, ValueError):
                print(f"Warning: Could not extract z-coord from {slice_name}")
                continue

            slice_image = cv2.imread(slice_path)  # Read image (BGR)
            if slice_image is None:
                continue

            original_height, original_width = slice_image.shape[:2]  # Store original size
            original_sizes[slice_name] = (original_width, original_height)

            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            resized_image = cv2.resize(slice_image, (256, 256))  # Resize
            slice_images.append(resized_image)
            slice_names.append(slice_name)
            z_coords.append(z_coord)

    if not slice_images:
        print("No valid images found.")
        return -1, -1, -1, -99

    # Convert to Tensor (B, C, H, W)
    slice_images = np.array(slice_images).astype(np.float32) / 255.0  # Normalize
    slice_images = torch.tensor(slice_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    # Process in batches
    for i in range(0, len(slice_images), batch_size):
        batch = slice_images[i:i + batch_size]
        batch_names = slice_names[i:i + batch_size]
        batch_z_coords = z_coords[i:i + batch_size]

        # Run inference on the batch
        results = model.predict(batch, verbose=False)

        for j, result in enumerate(results):
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                boxes = result.boxes.xywh.cpu().numpy()  # Bounding box coordinates

                if len(confidences) == 0:
                    continue

                best_idx_batch = np.argmax(confidences)
                best_confidence = confidences[best_idx_batch]

                if best_confidence > confidence_to_beat:
                    confidence_to_beat = best_confidence
                    best_slice_name = batch_names[j]
                    best_z_coord = batch_z_coords[j]  # Store best z-coordinate
                    box = boxes[best_idx_batch]
                    x_coord, y_coord = box[0], box[1]  # Center coordinates in resized image

                    # Get original image size
                    original_width, original_height = original_sizes[best_slice_name]

                    # Scale coordinates back to original size
                    scale_x = original_width / 256
                    scale_y = original_height / 256
                    x_coord_original = x_coord * scale_x
                    y_coord_original = y_coord * scale_y

                    x_coord, y_coord = x_coord_original, y_coord_original

    return float(best_z_coord), y_coord, x_coord, confidence_to_beat



def run_inference(test_path='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'):
    """Iterate through sub-directories containing tomogram images, make predictions on each one using get_best_slice_batch(),
    and save the coordinates into a dictionary that we then convert into a DataFrame that follows the format described in the competition overview. 
    We then save it to 'submission.csv', which is where Kaggle will look for our predictions when scoring.

    args: test_path(str): location of the test directory containing the test set of tomograms

    Saves the prediction .csv to 'submission.csv'
    """
    test_dir = Path(test_path)
    
    sub_table = {'tomo_id': [],
                'Motor axis 0': [],
                'Motor axis 1': [],
                'Motor axis 2': []}
    
    for tomo_dir in test_dir.iterdir():
        z, y, x, conf = get_best_slice_batch(tomo_dir)
        
        sub_table['tomo_id'].append(tomo_dir.name)
        sub_table['Motor axis 0'].append(z)
        sub_table['Motor axis 1'].append(y)
        sub_table['Motor axis 2'].append(x)
    
    sub_df = pd.DataFrame(sub_table)
    sub_df.to_csv('submission.csv', index=False)


run_inference()

