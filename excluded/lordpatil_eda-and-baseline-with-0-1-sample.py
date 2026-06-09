import pandas as pd
import numpy as np
import pydicom
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set the base path to your data
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/'

print("Setup complete. Libraries imported and base path set.")


# Load the training labels and localizer data
df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
df_localizers = pd.read_csv(os.path.join(BASE_PATH, 'train_localizers.csv'))

print("train.csv shape:", df_train.shape)
print("train_localizers.csv shape:", df_localizers.shape)


# Display the first 5 rows of the training data
print("First 5 rows of train.csv:")
display(df_train.head())

# Analyze the distribution of the main target variable 'Aneurysm Present'
print("\nDistribution of 'Aneurysm Present':")
aneurysm_counts = df_train['Aneurysm Present'].value_counts()
print(aneurysm_counts)

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Aneurysm Present', data=df_train)
plt.title('Distribution of Aneurysm Presence')
plt.xlabel('Aneurysm Present (1 = Yes, 0 = No)')
plt.ylabel('Number of Scan Series')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()

# Print the percentage
print(f"\nPercentage of series with an aneurysm: {aneurysm_counts[1] / len(df_train) * 100:.2f}%")


# Display the first 5 rows of the localizers data
print("First 5 rows of train_localizers.csv:")
display(df_localizers.head())

# Merge the localizer data with the main training dataframe
# This will give us a dataframe containing only the positive cases with their coordinates
df_merged = pd.merge(df_train, df_localizers, on='SeriesInstanceUID', how='inner')

print(f"\nShape of merged dataframe: {df_merged.shape}")
print("This shape should match the number of rows in train_localizers.csv, which is 2254.")

print("\nFirst 5 rows of the merged dataframe (train + localizers):")
display(df_merged.head())


import ast

# The 'coordinates' column is a string representation of a dictionary.
# We use ast.literal_eval to safely convert it into an actual dictionary.
df_merged['coordinates_dict'] = df_merged['coordinates'].apply(ast.literal_eval)

# Now, create separate columns for 'x' and 'y' coordinates.
df_merged['x'] = df_merged['coordinates_dict'].apply(lambda d: d['x'])
df_merged['y'] = df_merged['coordinates_dict'].apply(lambda d: d['y'])

# We can drop the intermediate columns now
df_merged = df_merged.drop(columns=['coordinates', 'coordinates_dict'])

print("Parsed 'x' and 'y' coordinates.")
display(df_merged[['SeriesInstanceUID', 'SOPInstanceUID', 'x', 'y', 'location']].head())


# Select the first sample from our merged dataframe
sample = df_merged.iloc[5]

series_uid = sample['SeriesInstanceUID']
sop_uid = sample['SOPInstanceUID']
x_coord = sample['x']
y_coord = sample['y']
location_label = sample['location']

# Construct the full path to the specific DICOM file
dcm_path = os.path.join(BASE_PATH, 'series', series_uid, f"{sop_uid}.dcm")
print(f"Loading DICOM file from: {dcm_path}")

# Load the DICOM file using pydicom
dcm_file = pydicom.dcmread(dcm_path)

# Get the pixel data from the DICOM file
image = dcm_file.pixel_array

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')

# Plot a red circle on top of the image at the aneurysm's coordinates
plt.scatter(x_coord, y_coord, s=200, facecolors='none', edgecolors='r', linewidth=2)

plt.title(f"Aneurysm Location: '{location_label}'\n at ({x_coord:.1f}, {y_coord:.1f})", fontsize=14)
plt.axis('off') # Hide the x and y axes for a cleaner look
plt.show()


from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import cv2

# --- Configuration ---
SAMPLE_FRACTION = 0.1 
BBOX_SIZE = 30
OUTPUT_DIR = '/kaggle/working/dataset'

# --- Create Directory Structure ---
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# --- Sample and Split Data ---
data_sample = df_merged.sample(frac=SAMPLE_FRACTION, random_state=42)
train_df, val_df = train_test_split(data_sample, test_size=0.2, random_state=42, stratify=data_sample['location'])

print(f"Processing {len(train_df)} images for training and {len(val_df)} for validation.")

# --- Processing Function ---
def process_and_save_sample(df, split):
    """Reads DICOM, robustly converts to 2D grayscale PNG, and creates YOLO label file."""
    success_count = 0
    error_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {split} set'):
        dcm_path = os.path.join(BASE_PATH, 'series', row['SeriesInstanceUID'], f"{row['SOPInstanceUID']}.dcm")
        image_filename = f"{row['SOPInstanceUID']}.png"
        label_filename = f"{row['SOPInstanceUID']}.txt"
        image_save_path = os.path.join(OUTPUT_DIR, 'images', split, image_filename)
        label_save_path = os.path.join(OUTPUT_DIR, 'labels', split, label_filename)

        try:
            # 1. Read DICOM and get pixel array
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array

            # --- ROBUSTNESS FIX STARTS HERE ---
            # Handle different image dimensions and channels
            if image.ndim == 3:
                # Case 1: (frames, height, width) -> multi-frame grayscale
                # Heuristic: take the middle frame
                if image.shape[0] > 1 and image.shape[2] > 4: # check if last dim is not channels
                     middle_slice_idx = image.shape[0] // 2
                     image = image[middle_slice_idx]
                # Case 2: (height, width, channels) -> color image
                elif image.shape[2] in [3, 4]:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Case 3: (1, height, width) -> single-frame in a 3D array
                elif image.shape[0] == 1:
                    image = image[0]
            
            if image.ndim != 2:
                 raise ValueError(f"Image array not successfully converted to 2D. Shape is {image.shape}")
            # --- ROBUSTNESS FIX ENDS HERE ---

            # 2. Normalize to 0-255 and convert to 8-bit unsigned integer
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 3. Save as PNG
            cv2.imwrite(image_save_path, image)

            # 4. Create YOLO label file
            img_h, img_w = image.shape
            x_center_norm = row['x'] / img_w
            y_center_norm = row['y'] / img_h
            width_norm = BBOX_SIZE / img_w
            height_norm = BBOX_SIZE / img_h
            
            with open(label_save_path, 'w') as f:
                f.write(f"0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
            
            success_count += 1
        
        except Exception as e:
            # print(f"Skipping file {dcm_path} due to error: {e}")
            error_count += 1
    
    print(f"\nFinished processing {split} set.")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")

# --- Run Processing ---
process_and_save_sample(train_df, 'train')
process_and_save_sample(val_df, 'val')

print("\nData preparation complete. Check the '/kaggle/working/dataset' directory.")


import yaml
import os
import sys
import glob

# --- Part 1: Create the dataset.yaml file (Unchanged) ---
dataset_config = {
    'path': '/kaggle/working/dataset',
    'train': 'images/train',
    'val': 'images/val',
    'names': { 0: 'aneurysm' }
}
with open('dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)
print("--- dataset.yaml created ---")
with open('dataset.yaml', 'r') as f: print(f.read())
print("--------------------------")

# --- Part 2: Bypassing pip by extracting the .whl file directly from your dataset ---
print("\nBypassing pip. Using ultralytics by extracting the wheel file.")

# 1. Define the path to your dataset's 'packages' directory.
packages_dir = '/kaggle/input/queryplanner-ultralytics-for-offline-install/packages'

# 2. Find the exact path to the ultralytics wheel file inside that directory.
try:
    ultralytics_whl_path = glob.glob(f'{packages_dir}/ultralytics-*.whl')[0]
    print(f"Found wheel file: {ultralytics_whl_path}")
except IndexError:
    raise FileNotFoundError("Could not find the ultralytics wheel file in your dataset's 'packages' directory.")

# 3. A .whl file is just a zip file. Unzip it to extract the source code.
#    We will extract it into the current working directory.
!unzip -q {ultralytics_whl_path} -d /kaggle/working/

# 4. The source code is now in a folder named 'ultralytics'. Add this to the Python path.
#    This makes the module importable without any installation.
source_code_path = '/kaggle/working/ultralytics'
sys.path.insert(0, source_code_path)
print(f"Added {source_code_path} to Python path.")

# Now, the import will work.
from ultralytics import YOLO

# --- Part 3: Load the correct YOLOv8 model ---
model_weights_path = '/kaggle/input/yolov8/pytorch/default/1/yolov8s.pt'
print(f"Loading pre-trained YOLOv8 weights from: {model_weights_path}")
model = YOLO(model_weights_path)

# --- Part 4: Train the Model ---
print("\nStarting model training...")
results = model.train(
    data='dataset.yaml',
    epochs=25,
    imgsz=512,
    batch=8,
    project='runs',
    name='aneurysm_detection_v8_final'
)


import shutil
import polars as pl
from ultralytics import YOLO

# --- Load our trained model ---
# The path points to the best weights saved during training
MODEL_PATH = '/kaggle/working/runs/aneurysm_detection_v8_final/weights/best.pt'
model = YOLO(MODEL_PATH)

# --- Define Constants ---
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present',
]
CONFIDENCE_THRESHOLD = 0.05 # A low threshold to be safe

# --- The Main Prediction Function ---
def predict(series_path: str) -> pl.DataFrame:
    """
    Takes a path to a DICOM series, runs inference on each slice,
    and returns a prediction for the entire series.
    """
    series_id = os.path.basename(series_path)
    aneurysm_detected_in_series = False

    # Get all DICOM file paths in the series
    dcm_files = glob.glob(os.path.join(series_path, '*.dcm'))

    # Loop through each slice and run prediction
    for dcm_path in dcm_files:
        # We don't need the try-except block here because the test set should be clean,
        # but it's good practice.
        try:
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array

            # Robustly convert to 2D grayscale, same logic as data prep
            if image.ndim == 3:
                if image.shape[0] > 1 and image.shape[2] > 4:
                     image = image[image.shape[0] // 2]
                elif image.shape[2] in [3, 4]:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[0] == 1:
                    image = image[0]
            
            # Run YOLO model inference
            results = model.predict(image, imgsz=512, verbose=False)
            
            # Check if any detection has confidence above our threshold
            if len(results[0].boxes) > 0:
                if results[0].boxes.conf[0].item() > CONFIDENCE_THRESHOLD:
                    aneurysm_detected_in_series = True
                    break # Stop processing this series as we've found one
        except Exception as e:
            # Silently ignore problematic slices in test data
            pass
            
    # --- Create the submission DataFrame ---
    # For this baseline, if we detect *any* aneurysm, we'll give a high probability to 'Aneurysm Present'
    # and a low, non-zero probability to all specific locations.
    if aneurysm_detected_in_series:
        aneurysm_present_prob = 0.9
        location_prob = 0.1 # A small guess for all locations
    else:
        aneurysm_present_prob = 0.1
        location_prob = 0.01

    # Create a dictionary for the row data
    data = {ID_COL: [series_id]}
    for col in LABEL_COLS:
        if col == 'Aneurysm Present':
            data[col] = [aneurysm_present_prob]
        else:
            data[col] = [location_prob]

    # Create a Polars DataFrame
    predictions = pl.DataFrame(data)

    # --- Important Cleanup Step for Kaggle ---
    shutil.rmtree('/kaggle/shared', ignore_errors=True)

    return predictions.drop(ID_COL)

print("Submission logic is ready.")

# This part of the code is for generating a submission.csv when you run the notebook directly.
# The actual competition environment will use the `predict` function differently.
# We'll simulate it with one of our validation images' series.
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("\nRunning a local test...")
    
    # Get a sample series path from our validation set
    sample_series_path = os.path.join(BASE_PATH, 'series', val_df.iloc[0]['SeriesInstanceUID'])
    
    # Create a dummy submission file by calling our function
    sample_prediction = predict(sample_series_path)
    
    # Add the ID column back for the local file
    final_df = pl.DataFrame({ID_COL: [os.path.basename(sample_series_path)]}).hstack(sample_prediction)
    
    final_df.write_parquet('submission.parquet')
    print("Created a sample submission.parquet file:")
    display(final_df)




