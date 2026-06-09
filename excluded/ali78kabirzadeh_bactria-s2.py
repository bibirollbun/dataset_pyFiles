!tar xfvz /kaggle/input/ultralytics-offlineinstall-yolo12-weights/archive.tar.gz
#!tar xfvz /kaggle/input/ultralytics-for-offline-instal/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages



#Ø§ØµÙ„Ø§Ø­ Ø´Ø§Ø±Ù¾
import os
import shutil
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
# Main settings

DATA_PATH = "/kaggle/input/cryoet-flagellar-motors-dataset/"
TRAIN_DIR = os.path.join(DATA_PATH, "jpgs")
TRAIN_CSV = os.path.join(DATA_PATH, "labels.csv")
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Load data
labels_df = pd.read_csv(TRAIN_CSV)
tomo_with_motors = labels_df['tomo_id'].unique()

print(f"ğŸ”� Number of tomograms with motors: {len(tomo_with_motors)}")

def organize_clean_slices():
    """Organize slices by keeping only the exact slice where the motor is located"""
    processed_slices = set()  # To avoid duplication

    for tomo_id in tqdm(tomo_with_motors, desc="ğŸ§ª Processing tomograms"):
        # Create output folder for this tomogram
        tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
        os.makedirs(tomo_output_dir, exist_ok=True)

        tomo_path = os.path.join(TRAIN_DIR, tomo_id)
        all_slices = sorted([f for f in os.listdir(tomo_path) if f.endswith('.jpg')])

        # Get motors for this tomogram with valid (non-negative) coordinates
        tomo_motors = labels_df[(labels_df['tomo_id'] == tomo_id) &
                                (labels_df['z'] >= 0) &
                                (labels_df['x'] >= 0) &
                                (labels_df['y'] >= 0)]

        for _, motor in tomo_motors.iterrows():
            z_center = int(motor['z'])
            y_center = int(motor['y'])
            x_center = int(motor['x'])  # âœ… Ø§ØµÙ„Ø§Ø­ Ø´Ø¯ (Ù‚Ø¨Ù„Ø§Ù‹ y Ø§Ø´ØªØ¨Ø§Ù‡ÛŒ Ø¬Ø§ÛŒ x Ø¨ÙˆØ¯)

            # Build the slice file name
            slice_name = f"slice_{z_center:04d}.jpg"
            src_path = os.path.join(tomo_path, slice_name)

            unique_key = f"{tomo_id}_{slice_name}"

            if os.path.exists(src_path) and unique_key not in processed_slices:
                rel_pos = 0  # Ù�Ù‚Ø· Ù‡Ù…ÙˆÙ† Ø§Ø³Ù„Ø§ÛŒØ³
                dest_name = f"motor{z_center}-{y_center}-{x_center}_slice{z_center:04d}_pos{rel_pos:+d}.jpg"
                dest_path = os.path.join(tomo_output_dir, dest_name)

                shutil.copy2(src_path, dest_path)
                processed_slices.add(unique_key)

# Run the function
print("ğŸ”� Organizing slices (only next 5 slices from motor)...")
organize_clean_slices()

# Show final result
print(f"\nâœ… Organization completed successfully!")
print(f"ğŸ“� Main output folder: {BASE_OUTPUT_DIR}")
print(f"ğŸ“Š Number of processed tomograms: {len(os.listdir(BASE_OUTPUT_DIR))}")

# Show sample structure of one tomogram folder
sample_tomo = os.listdir(BASE_OUTPUT_DIR)[0] if os.listdir(BASE_OUTPUT_DIR) else None
if sample_tomo:
    sample_path = os.path.join(BASE_OUTPUT_DIR, sample_tomo)
    print(f"\nğŸ“‚ Sample tomogram folder structure:")
    print(f"{sample_tomo}/")
    print("â”‚")
    sample_files = os.listdir(sample_path)[:3]  # Show first 3 files
    for f in sample_files:
        print(f"â”œâ”€â”€ {f}")
    if len(os.listdir(sample_path)) > 3:
        print(f"â””â”€â”€ ... ({len(os.listdir(sample_path)) - 3} more files)")




# Main configuration
DATA_PATH = "/kaggle/input/cryoet-flagellar-motors-dataset/"
TRAIN_DIR = os.path.join(DATA_PATH, "jpgs")
TRAIN_CSV = os.path.join(DATA_PATH, "labels.csv")
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean"
TRUST_RANGE = 0  # Only next 5 slices
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Load dataset
labels_df = pd.read_csv(TRAIN_CSV)
tomo_with_motors = labels_df['tomo_id'].unique()

print(f"ğŸ”� Number of tomograms containing motors: {len(tomo_with_motors)}")

def rotate_image_and_coords(image, angle, x, y, img_width, img_height):
    """Rotate the image and calculate the new coordinates of the motor based on the angle"""
    if angle == 0:
        rotated_image = image.copy()
        new_x, new_y = x, y
    elif angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_x, new_y = img_width - y, x  # Rotation formula for 90 degrees
    elif angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        new_x, new_y = img_width - x, img_height - y  # Rotation formula for 180 degrees
    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_x, new_y = y, img_width - x  # Rotation formula for 270 degrees

    return rotated_image, new_x, new_y

def organize_clean_slices():
    """Organize slices and rotate images while saving the new motor coordinates"""
    processed_slices = set()  # To avoid duplication
    
    for tomo_id in tqdm(tomo_with_motors, desc="ğŸ§ª Processing tomograms"):
        # Create output folder for the current tomogram
        tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
        os.makedirs(tomo_output_dir, exist_ok=True)
        
        tomo_path = os.path.join(TRAIN_DIR, tomo_id)
        all_slices = sorted([f for f in os.listdir(tomo_path) if f.endswith('.jpg')])
        total_slices = len(all_slices)
        
        # Get valid motors for this tomogram (positive coordinates only)
        tomo_motors = labels_df[(labels_df['tomo_id'] == tomo_id) & 
                               (labels_df['z'] >= 0) &
                               (labels_df['y'] >= 0) &
                               (labels_df['x'] >= 0)]
        
        for _, motor in tomo_motors.iterrows():
            z_center = int(motor['z'])
            y_center = int(motor['y'])
            x_center = int(motor['x'])
            
            # Slice range (only next 5 slices after motor)
            z_start = z_center
            z_end = min(total_slices - 1, z_center + TRUST_RANGE)
            
            for z in range(z_start, z_end + 1):
                slice_name = f"slice_{z:04d}.jpg"
                src_path = os.path.join(tomo_path, slice_name)
                
                # Unique key to avoid duplicate processing
                unique_key = f"{tomo_id}_{slice_name}"
                
                if os.path.exists(src_path) and unique_key not in processed_slices:
                    # Output filename with motor information
                    rel_pos = z - z_center  # Relative position (0 to +5)
                    dest_name = f"motor{z_center}-{y_center}-{x_center}_slice{z:04d}_pos{rel_pos:+d}.jpg"
                    dest_path = os.path.join(tomo_output_dir, dest_name)
                    
                    # Copy the original image
                    shutil.copy2(src_path, dest_path)
                    
                    # Read and rotate image at different angles
                    img = cv2.imread(src_path)
                    img_height, img_width = img.shape[:2]
                    
                    for angle in [0, 90, 180, 270]:
                        rotated_img, new_x, new_y = rotate_image_and_coords(
                            img, angle, x_center, y_center, img_width, img_height
                        )
                        
                        # Save rotated image
                        rotated_dest_name = f"motor{z_center}-{y_center}-{x_center}_slice{z:04d}_rot{angle}.jpg"
                        rotated_dest_path = os.path.join(tomo_output_dir, rotated_dest_name)
                        cv2.imwrite(rotated_dest_path, rotated_img)
                        
                        # Save new coordinates into CSV
                        rotated_entry = {
                            "tomo_id": tomo_id,
                            "slice_number": z,
                            "rotation_angle": angle,
                            "Motor axis 0": z_center,
                            "Motor axis 1": new_y,
                            "Motor axis 2": new_x
                        }
                        rotated_df = pd.DataFrame([rotated_entry])
                        
                        # Append or create CSV file for rotated coordinates
                        rotated_csv_path = os.path.join(tomo_output_dir, "rotated_motor_coordinates.csv")
                        if not os.path.exists(rotated_csv_path):
                            rotated_df.to_csv(rotated_csv_path, index=False)
                        else:
                            rotated_df.to_csv(rotated_csv_path, mode='a', header=False, index=False)
                    
                    processed_slices.add(unique_key)

# Execute the function
print("ğŸ”� Organizing slices and rotating images...")
organize_clean_slices()

# Final result summary
print(f"\nâœ… Organization completed successfully!")
print(f"ğŸ“� Main output directory: {BASE_OUTPUT_DIR}")
print(f"ğŸ“Š Number of processed tomograms: {len(os.listdir(BASE_OUTPUT_DIR))}")

# Show example folder structure
sample_tomo = os.listdir(BASE_OUTPUT_DIR)[0] if os.listdir(BASE_OUTPUT_DIR) else None
if sample_tomo:
    sample_path = os.path.join(BASE_OUTPUT_DIR, sample_tomo)
    print(f"\nğŸ“‚ Sample tomogram folder structure:")
    print(f"{sample_tomo}/")
    print("â”‚")
    sample_files = os.listdir(sample_path)[:3]  # Show first 3 files
    for f in sample_files:
        print(f"â”œâ”€â”€ {f}")
    if len(os.listdir(sample_path)) > 3:
        print(f"â””â”€â”€ ... ({len(os.listdir(sample_path)) - 3} more files)")




def delete_pos_zero_images(tomo_output_dir):
    """Delete images with _pos+0 in the filename"""
    for root, dirs, files in os.walk(tomo_output_dir):
        for file in files:
            if "_pos" in file:  # If the filename contains _pos+0
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Delete the file
                print(f"ğŸ—‘ï¸� Deleted: {file_path}")

# Path to the tomogram folder
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean"

# Delete images with _pos+0 from each tomogram folder
for tomo_id in os.listdir(BASE_OUTPUT_DIR):
    tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
    if os.path.isdir(tomo_output_dir):
        print(f"ğŸ”� Processing folder: {tomo_id}")
        delete_pos_zero_images(tomo_output_dir)

print("âœ… All images with _pos+0 have been deleted.")





# Paths for the input and output folders
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean"
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"  # Final folder to store images
os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)

# Function to move images and add the tomogram name to the file name
def move_images_to_final_folder_with_tomo_id():
    for tomo_id in os.listdir(BASE_OUTPUT_DIR):
        tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
        if os.path.isdir(tomo_output_dir):
            for file in os.listdir(tomo_output_dir):
                if file.endswith('.jpg'):
                    # Rename the file by adding the tomo_id
                    new_file_name = f"{tomo_id}_{file}"
                    src_path = os.path.join(tomo_output_dir, file)
                    dest_path = os.path.join(FINAL_IMAGES_DIR, new_file_name)
                    
                    # Move the file with the new name
                    shutil.copy2(src_path, dest_path)  
                    print(f"ğŸ“‚ Image transferred: {new_file_name}")

# Run the function to move the images
move_images_to_final_folder_with_tomo_id()

print(f"âœ… All images have been moved to the folder {FINAL_IMAGES_DIR}.")




import os
import pandas as pd

# Ù…Ø³ÛŒØ±Ù‡Ø§ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean"
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates.xlsx"  # Ù…Ø³ÛŒØ± Ø¨Ø±Ø§ÛŒ Ù�Ø§ÛŒÙ„ Ù†Ù‡Ø§ÛŒÛŒ Excel

# Ù„ÛŒØ³ØªÛŒ Ø¨Ø±Ø§ÛŒ Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
all_data = []

# Ø®ÙˆØ§Ù†Ø¯Ù† Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ø§Ø² Ù‡Ø± ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù…
for tomo_id in os.listdir(BASE_OUTPUT_DIR):
    tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
    rotated_csv_path = os.path.join(tomo_output_dir, "rotated_motor_coordinates.csv")
    
    if os.path.exists(rotated_csv_path):
        # Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¬Ø¯ÙˆÙ„ Ù…Ø®ØªØµØ§Øª ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù…
        rotated_df = pd.read_csv(rotated_csv_path)
        
        # Ø§Ø¶Ø§Ù�Ù‡ Ú©Ø±Ø¯Ù† Ù†Ø§Ù… ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù… Ø¨Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
        rotated_df['tomo_id'] = tomo_id
        
        # Ø§Ù�Ø²ÙˆØ¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ Ù„ÛŒØ³Øª
        all_data.append(rotated_df)

# ØªØ±Ú©ÛŒØ¨ ØªÙ…Ø§Ù…ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ ÛŒÚ© DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ ÛŒÚ© Ù�Ø§ÛŒÙ„ Excel (Ø¨Ø¯ÙˆÙ† Ù†ÛŒØ§Ø² Ø¨Ù‡ engine Ø®Ø§Øµ)
final_df.to_excel(FINAL_EXCEL_PATH, index=False)

print(f"âœ… Ù�Ø§ÛŒÙ„ Ø§Ú©Ø³Ù„ Ù†Ù‡Ø§ÛŒÛŒ Ø¨Ø§ Ù†Ø§Ù… {FINAL_EXCEL_PATH} Ø§ÛŒØ¬Ø§Ø¯ Ø´Ø¯.")





# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"

# Count the total number of images
total_images = len([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

print(f"ğŸ”¢ Total number of images: {total_images}")





# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Total number of rows
total_rows = len(final_df)

print(f"ğŸ”¢ Total number of rows in the Excel file: {total_rows}")





# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find all the images in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

# Third image
third_image = all_images[2]  # Third image (index 2)

# Extract tomo_id, slice number, and rotation angle from the third image file name
tomo_id = third_image.split('_')[0]  # Example: "tomo_00e047"
slice_number = int(third_image.split('_')[2][5:])  # Example: "slice0169" -> 169

# Extract the rotation angle (we need to convert "rot270" to 270)
rotation_angle_str = third_image.split('_')[3]  # Example: "rot270.jpg"
rotation_angle = int(rotation_angle_str[3:6])  # Extract number from "rot270" -> 270

# Print the third image's name and extracted information
print(f"ğŸ“¸ Third image: {third_image}")
print(f"tomo_id: {tomo_id}")
print(f"Slice number: {slice_number}")
print(f"Rotation angle: {rotation_angle} degrees")

# Filter the Excel data for the specific image
image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                      (final_df['slice_number'] == slice_number) & 
                      (final_df['rotation_angle'] == rotation_angle)]

# Print the motor coordinates for the third image
motor_x = image_data['Motor axis 2'].values[0]  # X from column motor_axis_2
motor_y = image_data['Motor axis 1'].values[0]  # Y from column motor_axis_1
motor_z = image_data['Motor axis 0'].values[0]  # Z from column motor_axis_0
print(f"ğŸ”� Motor coordinates for the third image: X={motor_x}, Y={motor_y}, Z={motor_z}")

# Load the image
image_path = os.path.join(FINAL_IMAGES_DIR, third_image)
img = Image.open(image_path)

# Draw a red circle at the motor coordinates
draw = ImageDraw.Draw(img)
circle_radius = 10  # Circle radius

# Draw the red circle (at coordinates X, Y)
draw.ellipse([motor_x - circle_radius, motor_y - circle_radius,
              motor_x + circle_radius, motor_y + circle_radius],
             outline="red", width=3)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Hide x and y axes
plt.show()





# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find the third image in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

# The third image
third_image = all_images[1000]  # Third image (index 6602)

# Extract tomo_id, slice number, and rotation angle from the third image's filename
tomo_id = third_image.split('_')[0] # Example: "tomo_00e047"
slice_number = int(third_image.split('_')[2][5:])  # Example: "slice0169" -> 169

# Extract rotation angle (we need to convert "rot270" to 270)
rotation_angle_str = third_image.split('_')[3]  # Example: "rot270.jpg"
rotation_angle_str = rotation_angle_str.replace('.jpg', '')  # Remove the .jpg extension
rotation_angle = int(rotation_angle_str[3:])  # Extract the number from "rot270" -> 270


# Print the third image name and extracted information
print(f"ğŸ“¸ Third image: {third_image}")
print(f"tomo_id: {tomo_id}")
print(f"Slice number: {slice_number}")
print(f"Rotation angle: {rotation_angle} degrees")

# Filter the Excel data for the specific image
image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                      (final_df['slice_number'] == slice_number) & 
                      (final_df['rotation_angle'] == rotation_angle)]

# Print the motor coordinates for the third image
motor_x = image_data['Motor axis 2'].values[0]  # X from motor_axis_2 column
motor_y = image_data['Motor axis 1'].values[0]  # Y from motor_axis_1 column
motor_z = image_data['Motor axis 0'].values[0]  # Z from motor_axis_0 column
print(f"ğŸ”� Motor coordinates for the third image: X={motor_x}, Y={motor_y}, Z={motor_z}")

# Load the image
image_path = os.path.join(FINAL_IMAGES_DIR, third_image)
img = Image.open(image_path)

# Draw a red circle at the motor coordinates
draw = ImageDraw.Draw(img)
circle_radius = 10  # Circle radius

# Draw a red circle (at X, Y coordinates)
draw.ellipse([motor_x - circle_radius, motor_y - circle_radius,
              motor_x + circle_radius, motor_y + circle_radius],
             outline="red", width=3)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Remove x and y axes
plt.show()





# Path to the folder you want to delete
folder_path = "/kaggle/working/classified_motor_slices_clean"

# Check if the folder exists and delete it along with all files and subfolders
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)  # Delete the folder and all its contents
    print(f"âœ… Folder '{folder_path}' and all its files have been successfully deleted.")
else:
    print(f"âš ï¸� Folder '{folder_path}' not found.")





# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates.xlsx"

# Path to the folder to save YOLO label files
YOLO_LABELS_DIR = "/kaggle/working/yolo_labels"

# Create folder if it doesn't exist
if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR)

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find all images in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

for image_name in all_images:
    # Extract tomo_id, slice number, and rotation angle from the image file name
    tomo_id = image_name.split('_')[0] 
    slice_number = int(image_name.split('_')[2][5:])
    
    # Correctly extract the rotation angle
    rotation_angle_str = image_name.split('_')[3]
    rotation_angle_str = rotation_angle_str.replace('.jpg', '')  # Remove the .jpg extension
    rotation_angle = ''.join(filter(str.isdigit, rotation_angle_str))  # Extract only digits
    rotation_angle = int(rotation_angle)  # Convert to integer

    # Filter the Excel data for the specific image
    image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                          (final_df['slice_number'] == slice_number) & 
                          (final_df['rotation_angle'] == rotation_angle)]

    # Extract motor coordinates for the image
    motor_x = image_data['Motor axis 2'].values[0]  # X in the motor_axis_2 column
    motor_y = image_data['Motor axis 1'].values[0]  # Y in the motor_axis_1 column
    motor_z = image_data['Motor axis 0'].values[0]  # Z in the motor_axis_0 column

    # Load the image to get the dimensions
    image_path = os.path.join(FINAL_IMAGES_DIR, image_name)
    img = Image.open(image_path)

    # Image dimensions (width and height)
    image_width, image_height = img.size

    # Assuming the real dimensions of the object are as follows (here 50 is used as the object size)
    real_width = 30  # Actual object width (in pixels)
    real_height = 30  # Actual object height (in pixels)

    # Normalize the width and height
    width = real_width / image_width
    height = real_height / image_height

    # Normalize the motor coordinates (motor_x, motor_y)
    x_center = motor_x / image_width
    y_center = motor_y / image_height

    # Path to the YOLO label file for saving the text file with the same name as the image
    label_file_path = os.path.join(YOLO_LABELS_DIR, f"{os.path.splitext(image_name)[0]}.txt")

    # Save the data in YOLO format
    with open(label_file_path, 'w') as f:
        f.write(f"0 {x_center} {y_center} {width} {height}\n")

    # Display the image dimensions and normalized coordinates
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Coordinates (X, Y): ({motor_x}, {motor_y})")
    print(f"Normalized Bounding Box: x_center={x_center}, y_center={y_center}, width={width}, height={height}")



###############
# Main configuration
DATA_PATH = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TRAIN_CSV = os.path.join(DATA_PATH, "train_labels.csv")
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean2"
TRUST_RANGE = 0  # Only next 5 slices
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Load dataset
labels_df = pd.read_csv(TRAIN_CSV)
tomo_with_motors = labels_df['tomo_id'].unique()

print(f"ğŸ”� Number of tomograms containing motors: {len(tomo_with_motors)}")

def rotate_image_and_coords(image, angle, x, y, img_width, img_height):
    """Rotate the image and calculate the new coordinates of the motor based on the angle"""
    if angle == 0:
        rotated_image = image.copy()
        new_x, new_y = x, y
    elif angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_x, new_y = img_width - y, x  # Rotation formula for 90 degrees
    elif angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        new_x, new_y = img_width - x, img_height - y  # Rotation formula for 180 degrees
    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_x, new_y = y, img_width - x  # Rotation formula for 270 degrees

    return rotated_image, new_x, new_y

def organize_clean_slices():
    """Organize slices and rotate images while saving the new motor coordinates"""
    processed_slices = set()  # To avoid duplication
    
    for tomo_id in tqdm(tomo_with_motors, desc="ğŸ§ª Processing tomograms"):
        # Create output folder for the current tomogram
        tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
        os.makedirs(tomo_output_dir, exist_ok=True)
        
        tomo_path = os.path.join(TRAIN_DIR, tomo_id)
        all_slices = sorted([f for f in os.listdir(tomo_path) if f.endswith('.jpg')])
        total_slices = len(all_slices)
        
        # Get valid motors for this tomogram (positive coordinates only)
        tomo_motors = labels_df[(labels_df['tomo_id'] == tomo_id) & 
                               (labels_df['Motor axis 0'] >= 0) &
                               (labels_df['Motor axis 1'] >= 0) &
                               (labels_df['Motor axis 2'] >= 0)]
        
        for _, motor in tomo_motors.iterrows():
            z_center = int(motor['Motor axis 0'])
            y_center = int(motor['Motor axis 1'])
            x_center = int(motor['Motor axis 2'])
            
            # Slice range (only next 5 slices after motor)
            z_start = z_center
            z_end = min(total_slices - 1, z_center + TRUST_RANGE)
            
            for z in range(z_start, z_end + 1):
                slice_name = f"slice_{z:04d}.jpg"
                src_path = os.path.join(tomo_path, slice_name)
                
                # Unique key to avoid duplicate processing
                unique_key = f"{tomo_id}_{slice_name}"
                
                if os.path.exists(src_path) and unique_key not in processed_slices:
                    # Output filename with motor information
                    rel_pos = z - z_center  # Relative position (0 to +5)
                    dest_name = f"motor{z_center}-{y_center}-{x_center}_slice{z:04d}_pos{rel_pos:+d}.jpg"
                    dest_path = os.path.join(tomo_output_dir, dest_name)
                    
                    # Copy the original image
                    shutil.copy2(src_path, dest_path)
                    
                    # Read and rotate image at different angles
                    img = cv2.imread(src_path)
                    img_height, img_width = img.shape[:2]
                    
                    for angle in [0, 90, 180, 270]:
                        rotated_img, new_x, new_y = rotate_image_and_coords(
                            img, angle, x_center, y_center, img_width, img_height
                        )
                        
                        # Save rotated image
                        rotated_dest_name = f"motor{z_center}-{y_center}-{x_center}_slice{z:04d}_rot{angle}.jpg"
                        rotated_dest_path = os.path.join(tomo_output_dir, rotated_dest_name)
                        cv2.imwrite(rotated_dest_path, rotated_img)
                        
                        # Save new coordinates into CSV
                        rotated_entry = {
                            "tomo_id": tomo_id,
                            "slice_number": z,
                            "rotation_angle": angle,
                            "Motor axis 0": z_center,
                            "Motor axis 1": new_y,
                            "Motor axis 2": new_x
                        }
                        rotated_df = pd.DataFrame([rotated_entry])
                        
                        # Append or create CSV file for rotated coordinates
                        rotated_csv_path = os.path.join(tomo_output_dir, "rotated_motor_coordinates_2.csv")
                        if not os.path.exists(rotated_csv_path):
                            rotated_df.to_csv(rotated_csv_path, index=False)
                        else:
                            rotated_df.to_csv(rotated_csv_path, mode='a', header=False, index=False)
                    
                    processed_slices.add(unique_key)

# Execute the function
print("ğŸ”� Organizing slices and rotating images...")
organize_clean_slices()

# Final result summary
print(f"\nâœ… Organization completed successfully!")
print(f"ğŸ“� Main output directory: {BASE_OUTPUT_DIR}")
print(f"ğŸ“Š Number of processed tomograms: {len(os.listdir(BASE_OUTPUT_DIR))}")

# Show example folder structure
sample_tomo = os.listdir(BASE_OUTPUT_DIR)[0] if os.listdir(BASE_OUTPUT_DIR) else None
if sample_tomo:
    sample_path = os.path.join(BASE_OUTPUT_DIR, sample_tomo)
    print(f"\nğŸ“‚ Sample tomogram folder structure:")
    print(f"{sample_tomo}/")
    print("â”‚")
    sample_files = os.listdir(sample_path)[:3]  # Show first 3 files
    for f in sample_files:
        print(f"â”œâ”€â”€ {f}")
    if len(os.listdir(sample_path)) > 3:
        print(f"â””â”€â”€ ... ({len(os.listdir(sample_path)) - 3} more files)")



########

def delete_pos_zero_images(tomo_output_dir):
    """Delete images with _pos+0 in the filename"""
    for root, dirs, files in os.walk(tomo_output_dir):
        for file in files:
            if "_pos" in file:  # If the filename contains _pos+0
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Delete the file
                print(f"ğŸ—‘ï¸� Deleted: {file_path}")

# Path to the tomogram folder
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean2"

# Delete images with _pos+0 from each tomogram folder
for tomo_id in os.listdir(BASE_OUTPUT_DIR):
    tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
    if os.path.isdir(tomo_output_dir):
        print(f"ğŸ”� Processing folder: {tomo_id}")
        delete_pos_zero_images(tomo_output_dir)

print("âœ… All images with _pos+0 have been deleted.")




######################
# Paths for the input and output folders
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean2"
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"  # Final folder to store images
os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)

# Function to move images and add the tomogram name to the file name
def move_images_to_final_folder_with_tomo_id():
    for tomo_id in os.listdir(BASE_OUTPUT_DIR):
        tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
        if os.path.isdir(tomo_output_dir):
            for file in os.listdir(tomo_output_dir):
                if file.endswith('.jpg'):
                    # Rename the file by adding the tomo_id
                    new_file_name = f"{tomo_id}_{file}"
                    src_path = os.path.join(tomo_output_dir, file)
                    dest_path = os.path.join(FINAL_IMAGES_DIR, new_file_name)
                    
                    # Move the file with the new name
                    shutil.copy2(src_path, dest_path)  
                    print(f"ğŸ“‚ Image transferred: {new_file_name}")

# Run the function to move the images
move_images_to_final_folder_with_tomo_id()

print(f"âœ… All images have been moved to the folder {FINAL_IMAGES_DIR}.")



###########
import os
import pandas as pd

# Ù…Ø³ÛŒØ±Ù‡Ø§ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
BASE_OUTPUT_DIR = "/kaggle/working/classified_motor_slices_clean2"
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates2.xlsx"  # Ù…Ø³ÛŒØ± Ø¨Ø±Ø§ÛŒ Ù�Ø§ÛŒÙ„ Ù†Ù‡Ø§ÛŒÛŒ Excel

# Ù„ÛŒØ³ØªÛŒ Ø¨Ø±Ø§ÛŒ Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
all_data = []

# Ø®ÙˆØ§Ù†Ø¯Ù† Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ø§Ø² Ù‡Ø± ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù…
for tomo_id in os.listdir(BASE_OUTPUT_DIR):
    tomo_output_dir = os.path.join(BASE_OUTPUT_DIR, tomo_id)
    rotated_csv_path = os.path.join(tomo_output_dir, "rotated_motor_coordinates_2.csv")
    
    if os.path.exists(rotated_csv_path):
        # Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¬Ø¯ÙˆÙ„ Ù…Ø®ØªØµØ§Øª ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù…
        rotated_df = pd.read_csv(rotated_csv_path)
        
        # Ø§Ø¶Ø§Ù�Ù‡ Ú©Ø±Ø¯Ù† Ù†Ø§Ù… ØªÙˆÙ…ÙˆÚ¯Ø±Ø§Ù… Ø¨Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
        rotated_df['tomo_id'] = tomo_id
        
        # Ø§Ù�Ø²ÙˆØ¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ Ù„ÛŒØ³Øª
        all_data.append(rotated_df)

# ØªØ±Ú©ÛŒØ¨ ØªÙ…Ø§Ù…ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ ÛŒÚ© DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ù‡ ÛŒÚ© Ù�Ø§ÛŒÙ„ Excel (Ø¨Ø¯ÙˆÙ† Ù†ÛŒØ§Ø² Ø¨Ù‡ engine Ø®Ø§Øµ)
final_df.to_excel(FINAL_EXCEL_PATH, index=False)

print(f"âœ… Ù�Ø§ÛŒÙ„ Ø§Ú©Ø³Ù„ Ù†Ù‡Ø§ÛŒÛŒ Ø¨Ø§ Ù†Ø§Ù… {FINAL_EXCEL_PATH} Ø§ÛŒØ¬Ø§Ø¯ Ø´Ø¯.")



#####################

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates2.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Total number of rows
total_rows = len(final_df)

print(f"ğŸ”¢ Total number of rows in the Excel file: {total_rows}")




######################
# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"

# Count the total number of images
total_images = len([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

print(f"ğŸ”¢ Total number of images: {total_images}")



######################
# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates2.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find all the images in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

# Third image
third_image = all_images[2]  # Third image (index 2)

# Extract tomo_id, slice number, and rotation angle from the third image file name
tomo_id = third_image.split('_')[0] + "_" + third_image.split('_')[1]  # Example: "tomo_00e047"
slice_number = int(third_image.split('_')[3][5:])  # Example: "slice0169" -> 169

# Extract the rotation angle (we need to convert "rot270" to 270)
rotation_angle_str = third_image.split('_')[4]  # Example: "rot270.jpg"
rotation_angle = int(rotation_angle_str[3:6])  # Extract number from "rot270" -> 270

# Print the third image's name and extracted information
print(f"ğŸ“¸ Third image: {third_image}")
print(f"tomo_id: {tomo_id}")
print(f"Slice number: {slice_number}")
print(f"Rotation angle: {rotation_angle} degrees")

# Filter the Excel data for the specific image
image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                      (final_df['slice_number'] == slice_number) & 
                      (final_df['rotation_angle'] == rotation_angle)]

# Print the motor coordinates for the third image
motor_x = image_data['Motor axis 2'].values[0]  # X from column motor_axis_2
motor_y = image_data['Motor axis 1'].values[0]  # Y from column motor_axis_1
motor_z = image_data['Motor axis 0'].values[0]  # Z from column motor_axis_0
print(f"ğŸ”� Motor coordinates for the third image: X={motor_x}, Y={motor_y}, Z={motor_z}")

# Load the image
image_path = os.path.join(FINAL_IMAGES_DIR, third_image)
img = Image.open(image_path)

# Draw a red circle at the motor coordinates
draw = ImageDraw.Draw(img)
circle_radius = 10  # Circle radius

# Draw the red circle (at coordinates X, Y)
draw.ellipse([motor_x - circle_radius, motor_y - circle_radius,
              motor_x + circle_radius, motor_y + circle_radius],
             outline="red", width=3)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Hide x and y axes
plt.show()




######################
# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates2.xlsx"

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find the third image in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

# The third image
third_image = all_images[1100]  # Third image (index 6602)

# Extract tomo_id, slice number, and rotation angle from the third image's filename
tomo_id = third_image.split('_')[0] + "_" + third_image.split('_')[1]  # Example: "tomo_00e047"
slice_number = int(third_image.split('_')[3][5:])  # Example: "slice0169" -> 169

# Extract rotation angle (we need to convert "rot270" to 270)
rotation_angle_str = third_image.split('_')[4]  # Example: "rot270.jpg"
rotation_angle_str = rotation_angle_str.replace('.jpg', '')  # Remove the .jpg extension
rotation_angle = int(rotation_angle_str[3:])  # Extract the number from "rot270" -> 270


# Print the third image name and extracted information
print(f"ğŸ“¸ Third image: {third_image}")
print(f"tomo_id: {tomo_id}")
print(f"Slice number: {slice_number}")
print(f"Rotation angle: {rotation_angle} degrees")

# Filter the Excel data for the specific image
image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                      (final_df['slice_number'] == slice_number) & 
                      (final_df['rotation_angle'] == rotation_angle)]

# Print the motor coordinates for the third image
motor_x = image_data['Motor axis 2'].values[0]  # X from motor_axis_2 column
motor_y = image_data['Motor axis 1'].values[0]  # Y from motor_axis_1 column
motor_z = image_data['Motor axis 0'].values[0]  # Z from motor_axis_0 column
print(f"ğŸ”� Motor coordinates for the third image: X={motor_x}, Y={motor_y}, Z={motor_z}")

# Load the image
image_path = os.path.join(FINAL_IMAGES_DIR, third_image)
img = Image.open(image_path)

# Draw a red circle at the motor coordinates
draw = ImageDraw.Draw(img)
circle_radius = 10  # Circle radius

# Draw a red circle (at X, Y coordinates)
draw.ellipse([motor_x - circle_radius, motor_y - circle_radius,
              motor_x + circle_radius, motor_y + circle_radius],
             outline="red", width=3)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Remove x and y axes
plt.show()




##############################
# Path to the folder you want to delete
folder_path = "/kaggle/working/classified_motor_slices_clean2"

# Check if the folder exists and delete it along with all files and subfolders
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)  # Delete the folder and all its contents
    print(f"âœ… Folder '{folder_path}' and all its files have been successfully deleted.")
else:
    print(f"âš ï¸� Folder '{folder_path}' not found.")







##################
# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"

# Path to the final Excel file
FINAL_EXCEL_PATH = "/kaggle/working/final_motor_coordinates2.xlsx"

# Path to the folder to save YOLO label files
YOLO_LABELS_DIR = "/kaggle/working/yolo_labels"

# Create folder if it doesn't exist
if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR)

# Load the Excel file
final_df = pd.read_excel(FINAL_EXCEL_PATH)

# Find all images in the folder
all_images = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

for image_name in all_images:
    # Extract tomo_id, slice number, and rotation angle from the image file name
    tomo_id = image_name.split('_')[0] + "_" + image_name.split('_')[1]
    slice_number = int(image_name.split('_')[3][5:])
    
    # Correctly extract the rotation angle
    rotation_angle_str = image_name.split('_')[4]
    rotation_angle_str = rotation_angle_str.replace('.jpg', '')  # Remove the .jpg extension
    rotation_angle = ''.join(filter(str.isdigit, rotation_angle_str))  # Extract only digits
    rotation_angle = int(rotation_angle)  # Convert to integer

    # Filter the Excel data for the specific image
    image_data = final_df[(final_df['tomo_id'] == tomo_id) & 
                          (final_df['slice_number'] == slice_number) & 
                          (final_df['rotation_angle'] == rotation_angle)]

    # Extract motor coordinates for the image
    motor_x = image_data['Motor axis 2'].values[0]  # X in the motor_axis_2 column
    motor_y = image_data['Motor axis 1'].values[0]  # Y in the motor_axis_1 column
    motor_z = image_data['Motor axis 0'].values[0]  # Z in the motor_axis_0 column

    # Load the image to get the dimensions
    image_path = os.path.join(FINAL_IMAGES_DIR, image_name)
    img = Image.open(image_path)

    # Image dimensions (width and height)
    image_width, image_height = img.size

    # Assuming the real dimensions of the object are as follows (here 50 is used as the object size)
    real_width = 30  # Actual object width (in pixels)
    real_height =30  # Actual object height (in pixels)

    # Normalize the width and height
    width = real_width / image_width
    height = real_height / image_height

    # Normalize the motor coordinates (motor_x, motor_y)
    x_center = motor_x / image_width
    y_center = motor_y / image_height

    # Path to the YOLO label file for saving the text file with the same name as the image
    label_file_path = os.path.join(YOLO_LABELS_DIR, f"{os.path.splitext(image_name)[0]}.txt")

    # Save the data in YOLO format
    with open(label_file_path, 'w') as f:
        f.write(f"0 {x_center} {y_center} {width} {height}\n")

    # Display the image dimensions and normalized coordinates
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Coordinates (X, Y): ({motor_x}, {motor_y})")
    print(f"Normalized Bounding Box: x_center={x_center}, y_center={y_center}, width={width}, height={height}")





# Path to the final images folder
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images2"

# Path to the folder for YOLO label files
YOLO_LABELS_DIR = "/kaggle/working/yolo_labels"

# File for the fifth image (index 4)
image_name = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])[142]  # Fifth image

# Load the image
image_path = os.path.join(FINAL_IMAGES_DIR, image_name)
img = Image.open(image_path)

# Path to the corresponding YOLO text file for the image
label_file_path = os.path.join(YOLO_LABELS_DIR, f"{os.path.splitext(image_name)[0]}.txt")

# Read the YOLO file data
with open(label_file_path, 'r') as f:
    label_data = f.readlines()

# Extract normalized coordinates and dimensions from the YOLO file
for label in label_data:
    parts = label.split()
    class_id = int(parts[0])  # Class ID (0 for motor)
    x_center = float(parts[1])  # Normalized X center
    y_center = float(parts[2])  # Normalized Y center
    width = float(parts[3])  # Normalized width
    height = float(parts[4])  # Normalized height

    # Image dimensions
    image_width, image_height = img.size

    # Convert normalized coordinates to pixels
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    # Calculate the coordinates of the bounding box (for drawing)
    left = x_center_pixel - width_pixel / 2
    top = y_center_pixel - height_pixel / 2
    right = x_center_pixel + width_pixel / 2
    bottom = y_center_pixel + height_pixel / 2

    # Draw the bounding box on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

# Display the image with the bounding box
plt.imshow(img)
plt.axis('off')  # Remove x and y axis
plt.show()



import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Ù…Ø³ÛŒØ± Ù¾ÙˆØ´Ù‡ ØªØµØ§ÙˆÛŒØ±
FINAL_IMAGES_DIR = "/kaggle/working/all_motor_images"

# Ù…Ø³ÛŒØ± Ù¾ÙˆØ´Ù‡ Ù„ÛŒØ¨Ù„â€ŒÙ‡Ø§ÛŒ YOLO
YOLO_LABELS_DIR = "/kaggle/working/yolo_labels"

# Ø¨Ø±Ø±Ø³ÛŒ ÙˆØ¬ÙˆØ¯ Ù¾ÙˆØ´Ù‡ ØªØµØ§ÙˆÛŒØ±
if not os.path.exists(FINAL_IMAGES_DIR):
    os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)
    print(f"âœ… Directory '{FINAL_IMAGES_DIR}' created successfully.")
else:
    print(f"ğŸ“‚ Directory '{FINAL_IMAGES_DIR}' already exists.")

# Ø¨Ø±Ø±Ø³ÛŒ ÙˆØ¬ÙˆØ¯ Ù¾ÙˆØ´Ù‡ Ù„ÛŒØ¨Ù„â€ŒÙ‡Ø§
if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    print(f"âœ… Directory '{YOLO_LABELS_DIR}' created successfully.")
else:
    print(f"ğŸ“‚ Directory '{YOLO_LABELS_DIR}' already exists.")

# Ú¯Ø±Ù�ØªÙ† Ù„ÛŒØ³Øª ØªØµØ§ÙˆÛŒØ±
image_files = sorted([f for f in os.listdir(FINAL_IMAGES_DIR) if f.endswith('.jpg')])

# Ø¨Ø±Ø±Ø³ÛŒ Ø§ÛŒÙ†Ú©Ù‡ Ø­Ø¯Ø§Ù‚Ù„ 11 ØªØµÙˆÛŒØ± ÙˆØ¬ÙˆØ¯ Ø¯Ø§Ø´ØªÙ‡ Ø¨Ø§Ø´Ø¯
if len(image_files) <= 10:
    raise ValueError(f"âš ï¸� Not enough images in '{FINAL_IMAGES_DIR}' to select the 11th image.")

# Ø§Ù†ØªØ®Ø§Ø¨ ØªØµÙˆÛŒØ± ÛŒØ§Ø²Ø¯Ù‡Ù… (index 10)
image_name = image_files[10]
image_path = os.path.join(FINAL_IMAGES_DIR, image_name)

# Ø¨Ø§Ø² Ú©Ø±Ø¯Ù† ØªØµÙˆÛŒØ±
img = Image.open(image_path)

# Ù…Ø³ÛŒØ± Ù�Ø§ÛŒÙ„ Ù„ÛŒØ¨Ù„ Ù…Ø±Ø¨ÙˆØ· Ø¨Ù‡ Ø§ÛŒÙ† ØªØµÙˆÛŒØ±
label_file_path = os.path.join(YOLO_LABELS_DIR, f"{os.path.splitext(image_name)[0]}.txt")

# Ø¨Ø±Ø±Ø³ÛŒ ÙˆØ¬ÙˆØ¯ Ù�Ø§ÛŒÙ„ Ù„ÛŒØ¨Ù„
if not os.path.exists(label_file_path):
    raise FileNotFoundError(f"âš ï¸� Label file '{label_file_path}' not found.")

# Ø®ÙˆØ§Ù†Ø¯Ù† Ù„ÛŒØ¨Ù„â€ŒÙ‡Ø§ÛŒ YOLO
with open(label_file_path, 'r') as f:
    label_data = f.readlines()

# ØªØ±Ø³ÛŒÙ… Ø¨Ø§Ú©Ø³â€ŒÙ‡Ø§ÛŒ Ù…Ø±Ø²ÛŒ
draw = ImageDraw.Draw(img)
image_width, image_height = img.size

for label in label_data:
    parts = label.strip().split()
    if len(parts) != 5:
        print(f"âš ï¸� Skipping invalid label line: {label}")
        continue

    class_id, x_center, y_center, width, height = map(float, parts)

    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    left = x_center_pixel - width_pixel / 2
    top = y_center_pixel - height_pixel / 2
    right = x_center_pixel + width_pixel / 2
    bottom = y_center_pixel + height_pixel / 2

    # Ø±Ø³Ù… Ù…Ø³ØªØ·ÛŒÙ„
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

# Ù†Ù…Ø§ÛŒØ´ ØªØµÙˆÛŒØ±
plt.imshow(img)
plt.axis('off')  # Ø­Ø°Ù� Ù…Ø­ÙˆØ±Ù‡Ø§ÛŒ x Ùˆ y
plt.show()



import os
import shutil

source_folder = '/kaggle/working/all_motor_images2'
destination_folder = '/kaggle/working/all_motor_images'
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.lower().endswith('.jpg'):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        shutil.copy2(src_path, dst_path)
        os.remove(src_path)

        print(f"{filename} â†’ Ù…Ù†ØªÙ‚Ù„ Ùˆ Ø­Ø°Ù� Ø´Ø¯.")

print("âœ… Ø¹Ù…Ù„ÛŒØ§Øª Ø§Ù†ØªÙ‚Ø§Ù„ Ùˆ Ø­Ø°Ù� Ù�Ø§ÛŒÙ„â€ŒÙ‡Ø§ÛŒ jpg Ø¨Ø§ Ù…ÙˆÙ�Ù‚ÛŒØª Ø§Ù†Ø¬Ø§Ù… Ø´Ø¯.")



import shutil
import os

folders_to_delete = [
    "/kaggle/working/all_motor_images2"
]

for folder_path in folders_to_delete:
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Ù¾ÙˆØ´Ù‡ {folder_path} Ø¨Ø§ Ù…ÙˆÙ�Ù‚ÛŒØª Ø­Ø°Ù� Ø´Ø¯.")
    else:
        print(f"Ù¾ÙˆØ´Ù‡ {folder_path} Ù¾ÛŒØ¯Ø§ Ù†Ø´Ø¯.")






# Path to input data
IMAGES_DIR = "/kaggle/working/all_motor_images"
LABELS_DIR = "/kaggle/working/yolo_labels"

# Path for the final YOLO structure
BASE_DIR = "/kaggle/working/motor_dataset"
os.makedirs(BASE_DIR, exist_ok=True)

# Create the necessary directories for train and val
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# Get all filenames (without extension)
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
image_names = [os.path.splitext(f)[0] for f in image_files]

# Split into train and val (90% train, 10% val)
train_names, val_names = train_test_split(image_names, test_size=0.2, random_state=42)

# Function to copy image and label files
def copy_data(file_list, split):
    for name in file_list:
        img_src = os.path.join(IMAGES_DIR, name + ".jpg")
        label_src = os.path.join(LABELS_DIR, name + ".txt")

        img_dst = os.path.join(BASE_DIR, f"images/{split}", name + ".jpg")
        label_dst = os.path.join(BASE_DIR, f"labels/{split}", name + ".txt")

        if os.path.exists(img_src) and os.path.exists(label_src):
            shutil.copyfile(img_src, img_dst)
            shutil.copyfile(label_src, label_dst)

# Copy the files
copy_data(train_names, "train")
copy_data(val_names, "val")

# Create a dataset.yaml file for YOLOv8
yaml_path = os.path.join(BASE_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(
        f"path: {BASE_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: ['motor']\n"
    )

print("âœ… Directories have been created and data has been split.")
print(f"ğŸ“� Final path: {BASE_DIR}")
print(f"ğŸ“� The YAML file is also located at: {yaml_path}")





# Path of folders and files
all_motor_images_path = "/kaggle/working/all_motor_images"
final_motor_coordinates_path = "/kaggle/working/final_motor_coordinates.xlsx"

# Delete the all_motor_images folder and all its contents
if os.path.exists(all_motor_images_path) and os.path.isdir(all_motor_images_path):
    shutil.rmtree(all_motor_images_path)
    print(f"âœ… Folder '{all_motor_images_path}' and all its contents were successfully deleted.")
else:
    print(f"âš ï¸� Folder '{all_motor_images_path}' not found.")

# Delete the final_motor_coordinates.xlsx file
if os.path.exists(final_motor_coordinates_path):
    os.remove(final_motor_coordinates_path)
    print(f"âœ… File '{final_motor_coordinates_path}' was successfully deleted.")
else:
    print(f"âš ï¸� File '{final_motor_coordinates_path}' not found.")



# Path to the yaml file containing dataset information
dataset_yaml_path = "/kaggle/working/motor_dataset/dataset.yaml"


from ultralytics import YOLO

# Ù…Ø±Ø­Ù„Ù‡ 1: Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ ÙˆØ²Ù† Ø§ÙˆÙ„ÛŒÙ‡ Ùˆ Ø¢Ù…ÙˆØ²Ø´ 50 epoch Ø¨Ø¯ÙˆÙ† Ù�Ø±ÛŒØ²
model = YOLO("/kaggle/input/ultralytics-offlineinstall-yolo12-weights/yolov12-weights/yolo12s.pt")

model.train(
    data=dataset_yaml_path,
    epochs=1,
    imgsz=960,
    batch=8,
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    val=True,
    mosaic=1,
    mixup=0,
    degrees=10,
    scale=0.25,
    name='m1_initial'
)




# Ù…Ø±Ø­Ù„Ù‡ 2: Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ù…Ø¯Ù„ Ø¢Ù…ÙˆØ²Ø´â€ŒØ¯ÛŒØ¯Ù‡ Ø§Ø² Ù…Ø±Ø­Ù„Ù‡ Ù‚Ø¨Ù„
model = YOLO('runs/detect/m1_initial/weights/last.pt')  # Ù…Ø³ÛŒØ± Ù�Ø§ÛŒÙ„ Ø®Ø±ÙˆØ¬ÛŒ Ù…Ø±Ø­Ù„Ù‡ Ø§ÙˆÙ„

# Ù�Ø±ÛŒØ² Ú©Ø±Ø¯Ù† Ù„Ø§ÛŒÙ‡â€ŒÙ‡Ø§ÛŒ Backbone (Ù„Ø§ÛŒÙ‡â€ŒÙ‡Ø§ÛŒ 0 ØªØ§ 9)
for i, layer in enumerate(model.model.model[:10]):
    for param in layer.parameters():
        param.requires_grad = False

# Ø§Ø¯Ø§Ù…Ù‡ Ø¢Ù…ÙˆØ²Ø´ Ø¨Ø§ Ù�Ø±ÛŒØ² Ú©Ø±Ø¯Ù† - 150 epoch Ø¯ÛŒÚ¯Ø±
model.train(
    data=dataset_yaml_path,
    epochs=1,
    imgsz=960,
    batch=8,
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    val=True,
    mosaic=1,
    mixup=0,
    degrees=10,
    scale=0.25,
    name='m1_frozen'
)





# Load the trained model
model = YOLO("runs/detect/m1_frozen/weights/best.pt")

# Run evaluation (val) on the validation data defined in dataset.yaml
results = model.val(data="/kaggle/working/motor_dataset/dataset.yaml")



print(results.results_dict)





for fname in ["results.png", "confusion_matrix.png", "precision_recall_curve.png"]:
    fpath = os.path.join(results.save_dir, fname)
    if os.path.exists(fpath):
        img = Image.open(fpath)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(fname)
        plt.show()



import torch

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths for the test data and submission
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"

# Path to the best trained model (adjust if necessary)
model_path="runs/detect/m1_frozen/weights/best.pt"

# Define detection and processing parameters
CONFIDENCE_THRESHOLD = 0.45
MAX_DETECTIONS_PER_TOMO = 6
NMS_IOU_THRESHOLD = 0.2
CONCENTRATION = 1  # Process a fraction of slices for fast submission

# GPU profiling context manager for timing
class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

# Set device and dynamic batch size
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
if device.startswith('cuda'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
    print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 4


def normalize_slice(slice_data):
    """
    Normalize slice data using the 2nd and 98th percentiles.
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory."""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path))
        images.append(img)
    return images

def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
    """
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    final_detections = []
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)
    
    box_size = 24
    distance_threshold = box_size * iou_threshold
    
    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections

def process_tomogram(tomo_id, model, index=0, total=1):
    """
    Process a single tomogram and return the most confident motor detection.
    """
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]
    
    print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices (CONCENTRATION={CONCENTRATION})")
    all_detections = []
    
    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        streams = [None]
    
    next_batch_thread = None
    next_batch_images = None
    
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None
            
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
        
        sub_batches = np.array_split(batch_files, len(streams))
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        for box_idx, confidence in enumerate(result.boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    if not final_detections:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}
    
    best_detection = final_detections[0]
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }

def debug_image_loading(tomo_id):
    """
    Debug function to test image loading methods.
    """
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return
        
    print(f"Found {len(slice_files)} image files in {tomo_dir}")
    sample_file = slice_files[len(slice_files)//2]
    img_path = os.path.join(tomo_dir, sample_file)
    
    try:
        img_pil = Image.open(img_path)
        print(f"PIL Image shape: {np.array(img_pil).shape}, dtype: {np.array(img_pil).dtype}")
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        print("Image loading successful!")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        
    try:
        test_model = YOLO(model_path)
        test_results = test_model([img_path], verbose=False)
        print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")


from concurrent.futures import ThreadPoolExecutor
import threading

def generate_submission():
    """
    Main function to generate the submission file.
    """
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    print(f"Found {total_tomos} tomograms in test directory")
    
    if test_tomos:
        debug_image_loading(test_tomos[0])
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    model.to(device)
    if device.startswith('cuda'):
        model.fuse()
        if torch.cuda.get_device_capability(0)[0] >= 7:
            model.model.half()
            print("Using half precision (FP16) for inference")
    
    results = []
    motors_found = 0
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id
        
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = future.result()
                results.append(result)
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                results.append({'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1})
    
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")
    print("\nSubmission preview:")
    print(submission_df.head())
    return submission_df


import time

if __name__ == "__main__":
    start_time = time.time()
    submission = generate_submission()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

