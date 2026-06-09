# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


DIR = "/kaggle/input/rsna-lumbar-metadata/data/processed_metadata"


import pandas as pd


left = pd.read_csv(f"{DIR}/processed_metadata_LeftNeuralForaminalNarrowing.csv")
right = pd.read_csv(f"{DIR}/processed_metadata_RightNeuralForaminalNarrowing.csv")


left.shape


right.shape


left.head()


paths_df1 = set(left['image_path'].unique())
paths_df2 = set(right['image_path'].unique())


duplicate_paths = paths_df1.intersection(paths_df2)


print(f"Total unique paths in df1: {len(paths_df1)}")
print(f"Total unique paths in df2: {len(paths_df2)}")
print(f"Number of duplicate image paths across both: {len(duplicate_paths)}")


if duplicate_paths:
    print("Sample duplicate paths:")
    for path in list(duplicate_paths)[:5]:  # Show just a few for sanity check
        print(path)
else:
    print("No duplicate image paths found between the two DataFrames.")


left_sub = pd.read_csv(f"{DIR}/processed_metadata_LeftSubarticularStenosis.csv")
right_sub = pd.read_csv(f"{DIR}/processed_metadata_RightSubarticularStenosis.csv")


paths_df1 = set(left['image_path'].unique())
paths_df2 = set(right['image_path'].unique())


duplicate_paths = paths_df1.intersection(paths_df2)


print(f"Total unique paths in df1: {len(paths_df1)}")
print(f"Total unique paths in df2: {len(paths_df2)}")
print(f"Number of duplicate image paths across both: {len(duplicate_paths)}")


left = pd.read_csv(f"{DIR}/processed_metadata_LeftNeuralForaminalNarrowing.csv")
right = pd.read_csv(f"{DIR}/processed_metadata_RightNeuralForaminalNarrowing.csv")


severity_counts_df1 = left['severity_code'].value_counts().sort_index()
severity_counts_df2 = right['severity_code'].value_counts().sort_index()
print(severity_counts_df1)
print(severity_counts_df2)


severity_counts_df1_sub = left_sub['severity_code'].value_counts().sort_index()
severity_counts_df2_sub = right_sub['severity_code'].value_counts().sort_index()
print(severity_counts_df1_sub)
print(severity_counts_df2_sub)


spinal = pd.read_csv(f"{DIR}/processed_metadata_SpinalCanalStenosis.csv")


severity_counts_df_spinal = spinal['severity_code'].value_counts().sort_index()
print(severity_counts_df_spinal)





import os
import cv2
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A



# --- CONFIG ---
TARGET_TOTAL = 15000
CLASS_COUNTS = {0: 5000, 1: 5000, 2: 5000}
ROTATION_ANGLES = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]

# Define input/output paths
DIR = "/kaggle/input/rsna-lumbar-metadata/data/processed_metadata"
INPUT_CSV = f"{DIR}/processed_metadata_LeftNeuralForaminalNarrowing.csv"
OUTPUT_FOLDER = "/kaggle/working/augmented/left_neural"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# --- Load CSV ---
df = pd.read_csv(INPUT_CSV)

# --- Load DICOM and normalize ---
def load_image(path):
    try:
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array.astype(np.float32)
        image -= np.min(image)
        image /= np.max(image)
        image = (image * 255).astype(np.uint8)
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load DICOM: {path} | {e}")
        return None

# --- Save DICOM with augmented pixel data ---
def save_dicom_image(original_path, image_array, save_path):
    try:
        # Read the original DICOM
        ds = pydicom.dcmread(original_path)

        # Extract relevant DICOM information
        study_id = ds.StudyInstanceUID
        series_id = ds.SeriesInstanceUID
        instance_no = ds.InstanceNumber

        # Ensure correct type and dimensions
        if image_array.dtype != np.uint16:
            image_array = image_array.astype(np.uint16)

        # Update the DICOM metadata
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"  # for grayscale images
        ds.Rows, ds.Columns = image_array.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15  # Highest bit of a 16-bit image
        ds.PixelRepresentation = 0  # Unsigned integers
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Set the PixelData to the image (ensure it's in bytes)
        ds.PixelData = image_array.tobytes()

        # Update SOP Instance UID and MediaStorage SOP Instance UID to prevent conflicts
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        # --- Modify the filename to include study_id, series_id, and instance_no ---
        base_name = os.path.basename(save_path)  # Get the base name of the save path (e.g., file name)
        new_filename = f"aug_{study_id}_series{series_id}_inst{instance_no}_aug{base_name}"
        new_path = os.path.join(os.path.dirname(save_path), new_filename)

        # Save the modified DICOM
        ds.save_as(new_path)
        # print(f"âœ… Saved DICOM with 16-bit depth at {new_path}")
    except Exception as e:
        print(f"[ERROR] Saving DICOM failed for {save_path} | {e}")



# --- Rotation Transform ---
def get_rotation_transform(angle):
    return A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_REFLECT, p=1.0)
    ])

# --- Augment & Save as DICOM ---
def augment_and_save(images, required_count, class_id, base_df):
    saved_rows = []
    augmented = 0
    i = 0
    while augmented < required_count:
        image_path = images[i % len(images)]
        image = load_image(image_path)
        if image is None:
            i += 1
            continue

        for angle in ROTATION_ANGLES:
            if augmented >= required_count:
                break
            transform = get_rotation_transform(angle)
            augmented_image = transform(image=image)['image']

            new_filename = f"aug_sev{class_id}_{augmented}.dcm"
            new_path = os.path.join(OUTPUT_FOLDER, new_filename)
            save_dicom_image(image_path, augmented_image, new_path)

            row = base_df[base_df['image_path'] == image_path].iloc[0].copy()
            row['image_path'] = new_path
            saved_rows.append(row)
            augmented += 1
        i += 1
    return saved_rows


# --- Create balanced dataset ---
balanced_df = []

# ğŸŸ¢ Handle Normal (0)
normal_df = df[df['severity_code'] == 0].sample(CLASS_COUNTS[0], random_state=42)
for idx, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Processing Normal"):
    image = load_image(row['image_path'])
    if image is None:
        continue
    new_filename = f"normal_{idx}.dcm"
    new_path = os.path.join(OUTPUT_FOLDER, new_filename)
    save_dicom_image(row['image_path'], image, new_path)
    row['image_path'] = new_path
    balanced_df.append(row)

# ğŸŸ¡ Handle Moderate (1)
moderate_df = df[df['severity_code'] == 1]
moderate_images = moderate_df['image_path'].tolist()
needed = CLASS_COUNTS[1] - len(moderate_images)
balanced_df += moderate_df.to_dict('records')
balanced_df += augment_and_save(moderate_images, needed, 1, moderate_df)

# ğŸ”´ Handle Severe (2)
severe_df = df[df['severity_code'] == 2]
severe_images = severe_df['image_path'].tolist()
needed = CLASS_COUNTS[2] - len(severe_images)
balanced_df += severe_df.to_dict('records')
balanced_df += augment_and_save(severe_images, needed, 2, severe_df)



output_csv = '/kaggle/working/balanced_dataset.csv'
cleaned_balanced_df = [pd.Series(item) for item in balanced_df]
df = pd.DataFrame(cleaned_balanced_df)
df.to_csv(output_csv, index=False)
print(f"âœ… Balanced dataset saved to: {output_csv}")


balanced_df[0]


df.head()


df.shape


severity_counts_df1 = df['severity_code'].value_counts().sort_index()
print(severity_counts_df1)


import os
import matplotlib.pyplot as plt
import pydicom

# --- CONFIG ---
DATA_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
AUG_DIR = "/kaggle/working/augmented/left_neural"
image_dir = os.path.join(DATA_DIR, 'train_images')

# ğŸ”§ Function to generate original DICOM image paths
def generate_image_paths(study_id, series_id):
    series_dir = os.path.join(image_dir, str(study_id), str(series_id))
    images = sorted(os.listdir(series_dir))
    image_paths = [os.path.join(series_dir, img) for img in images]
    return image_paths

# ğŸ”� Function to construct the augmented filename using DICOM metadata
def get_augmented_path_from_dicom(orig_path):
    try:
        ds = pydicom.dcmread(orig_path)
        study_id = ds.StudyInstanceUID
        series_id = ds.SeriesInstanceUID
        instance_no = ds.InstanceNumber
        base_name = os.path.basename(orig_path)
        aug_filename = f"aug_{study_id}_series{series_id}_inst{instance_no}_aug{base_name}"
        aug_path = os.path.join(AUG_DIR, aug_filename)
        return aug_path if os.path.exists(aug_path) else None
    except Exception as e:
        print(f"[ERROR] Could not read DICOM: {orig_path} | {e}")
        return None

# ğŸ–¼ï¸� Display original vs augmented images using new naming convention
def display_original_vs_augmented(image_paths):
    n = min(len(image_paths), 6)  # Limit to first few for quick view
    plt.figure(figsize=(10, 5 * n))

    for i in range(n):
        orig_path = image_paths[i]
        aug_path = get_augmented_path_from_dicom(orig_path)

        if not aug_path:
            print(f"[WARN] No augmented file found for: {os.path.basename(orig_path)}")
            continue

        try:
            orig_ds = pydicom.dcmread(orig_path)
            aug_ds = pydicom.dcmread(aug_path)

            # Original
            plt.subplot(n, 2, 2 * i + 1)
            plt.imshow(orig_ds.pixel_array, cmap='gray')
            plt.title(f"Original: {os.path.basename(orig_path)}")
            plt.axis('off')

            # Augmented
            plt.subplot(n, 2, 2 * i + 2)
            plt.imshow(aug_ds.pixel_array, cmap='gray')
            plt.title(f"Augmented: {os.path.basename(aug_path)}")
            plt.axis('off')
        except Exception as e:
            print(f"[ERROR] Failed to load image pair {i}: {e}")
            continue

    plt.tight_layout()
    plt.show()


study_id = "3617698707"
series_id = "3406806779"
image_paths = generate_image_paths(study_id, series_id)
display_original_vs_augmented(image_paths)


balanced_df[0]['image_path']




