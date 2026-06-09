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


import pandas as pd
import os

# Load the labeled data metadata
base_path="/kaggle/input/forams-classification-2025"
labeled_df = pd.read_csv(os.path.join(base_path, 'labelled.csv'))

# Display the first few rows
print("Labeled Data Metadata:")
print(labeled_df.head())



import os

# Path to the labeled volume folder
volumes_labelled_path = '/kaggle/input/forams-classification-2025/volumes/volumes/labelled'

# List files in the labelled volumes folder
labeled_files = os.listdir(volumes_labelled_path)

# Display the first few files to confirm structure
print("Labeled Volume Files:")
print(labeled_files[:5])

# Extract the numeric ID from the first row in labelled.csv
sample_id = labeled_df.iloc[0]['id'].replace('labelled_', '')  # Remove 'labelled_' prefix

# Look for the correct file format (adjust the scale factor as per your understanding)
matching_files = [f for f in labeled_files if f.startswith(f"labelled_foram_{sample_id}")]

# Display matched files
print(f"Matching files for id {sample_id}:")
print(matching_files)



import tifffile
import matplotlib.pyplot as plt

# Path to the sample file
sample_file_path = '/kaggle/input/forams-classification-2025/volumes/volumes/labelled/labelled_foram_00000_sc_0_752.tif'

# Load the TIFF file as a 3D NumPy array
img_array = tifffile.imread(sample_file_path)

# Check the shape of the array to confirm it's 3D (128x128x128)
print(f"Shape of the volume: {img_array.shape}")

# Visualize a few slices along the z-axis (we'll show slices at index 30, 60, and 90)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Slice indices to display (valid indices between 0 and 127)
slice_indices = [30, 60, 90]

for i, slice_idx in enumerate(slice_indices):
    axes[i].imshow(img_array[slice_idx], cmap='gray')
    axes[i].set_title(f"Slice {slice_idx}")
    axes[i].axis('off')

plt.show()



# import numpy as np
# import pandas as pd
# import tifffile
# from glob import glob
# import os
# from concurrent.futures import ThreadPoolExecutor
# import gc
# from tqdm import tqdm

# # Paths
# labeled_volume_path = '/kaggle/input/forams-classification-2025/volumes/volumes/labelled/'
# unlabeled_volume_path = '/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled/'

# # Metadata
# # labeled_metadata = pd.read_csv('/kaggle/input/forams-classification-2025/labelled.csv')


# import os
# import re
# import pandas as pd
# import torch
# import torchvision.transforms as transforms
# import numpy as np
# import tifffile as tiff  # For loading .tif files

# # Paths
# volumes_path = "/kaggle/input/forams-classification-2025/volumes/volumes/"
# labelled_csv_path = "/kaggle/input/forams-classification-2025/labelled.csv"

# # Regex for filenames
# labelled_regex = re.compile(r"labelled_foram_(\d{5})_sc_(\d)_(\d{3})\.tif")
# unlabelled_regex = re.compile(r"foram_(\d{5})_sc_(\d)_(\d{3})\.tif")

# # Parse labelled data
# labelled_data = []
# for filename in os.listdir(os.path.join(volumes_path, "labelled")):
#     match = labelled_regex.match(filename)
#     if match:
#         file_id, int_scale, dec_scale = match.groups()
#         scaling_factor = int(int_scale) + int(dec_scale) / 1000
#         labelled_data.append({"id": int(file_id), "scaling_factor": scaling_factor, "filename": filename})

# # Parse unlabelled data
# unlabelled_data = []
# for filename in os.listdir(os.path.join(volumes_path, "unlabelled")):
#     match = unlabelled_regex.match(filename)
#     if match:
#         file_id, int_scale, dec_scale = match.groups()
#         scaling_factor = int(int_scale) + int(dec_scale) / 1000
#         unlabelled_data.append({"id": int(file_id), "scaling_factor": scaling_factor, "filename": filename})

# # Load labelled CSV
# # Read the CSV file, skipping the first row
# labels_df = pd.read_csv(labelled_csv_path)

# labels_df = labels_df[pd.to_numeric(labels_df['id'], errors='coerce').notnull()]

# labelled_df = pd.DataFrame(labelled_data)
# labelled_df["id"] = labelled_df["id"].astype(int)

# labelled_df = labelled_df.merge(labels_df, on="id", how="inner")

# print("Labelled Data:")
# print(labelled_df.head())

# print("\nUnlabelled Data:")
# print(pd.DataFrame(unlabelled_data).head())




# import numpy as np
# import tifffile
# from glob import glob
# import os
# import gc
# from tqdm import tqdm

# # Path for unlabeled volumes
# unlabeled_volume_path = '/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled/'

# # Function to load and normalize a single volume
# def load_volume(file_path):
#     try:
#         volume = tifffile.imread(file_path)
#         return volume / np.max(volume)  # Normalize to [0, 1]
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

# # Process unlabeled data one by one
# def process_unlabeled_incrementally():
#     unlabeled_files = sorted(glob(os.path.join(unlabeled_volume_path, "*.tif")))

#     for idx, file in enumerate(tqdm(unlabeled_files, desc="Processing Unlabeled Volumes")):
#         volume = load_volume(file)
#         if volume is not None:
#             # Save processed volume immediately
#             save_path = f'/kaggle/working/unlabeled_volume_{idx}.npz'
#             np.savez_compressed(save_path, X=volume)
#             print(f"Saved {save_path}")
        
#         # Explicitly release memory
#         del volume
#         gc.collect()

#     print("Unlabeled data processing completed.")

# # Execute

# process_unlabeled_incrementally()



# print(f"Training data shape: {X_train.shape}")
# print(f"Training labels shape: {y_train.shape}")



