import pandas as pd
import os

# Folder dir
INPUT_CSV = '/kaggle/input/asl-signs/train.csv'
NUM_CLASSES = 80

# 1. Load Original CSV
print("Reading CSV...")
df = pd.read_csv(INPUT_CSV)

# 2. Find Top 100 Words
print(f"Finding top {NUM_CLASSES} signs...")
top_100_signs = df['sign'].value_counts().head(NUM_CLASSES).index.tolist()

# 3. Filter the DataFrame
df_subset = df[df['sign'].isin(top_100_signs)].copy()

print(f"Selected {len(df_subset)} files for {len(top_100_signs)} signs.")
print(f"Signs: {top_100_signs[:10]}...")


import shutil
from tqdm.notebook import tqdm
import os

SOURCE_ROOT = '/kaggle/input/asl-signs'

# CHANGE: Build the dataset in the TEMP directory first
# This prevents your 19GB quota from filling up with loose files
DEST_ROOT = '/kaggle/temp/asl_subset_100' 

# 1. Reset/Create Directory
if os.path.exists(DEST_ROOT):
    print("Removing old subset folder in temp...")
    shutil.rmtree(DEST_ROOT)
    
os.makedirs(DEST_ROOT, exist_ok=True)
print(f"Created temp output folder: {DEST_ROOT}")

# 2. Copy Loop
print("Starting Copy Process (This keeps all Facial Features)...")

copied_count = 0

for index, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    # Original Path
    src_path = os.path.join(SOURCE_ROOT, row['path'])
    
    # Destination Path (in Temp)
    dest_path = os.path.join(DEST_ROOT, row['path'])
    
    # Make sure the sub-folder exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Copy the file
    shutil.copy2(src_path, dest_path)
    copied_count += 1

print(f"Successfully copied {copied_count} files to Temp storage.")


# 1. Save the filtered CSV inside the temp folder
csv_save_path = os.path.join(DEST_ROOT, 'train.csv')
df_subset.to_csv(csv_save_path, index=False)
print(f"Saved index file to: {csv_save_path}")

# 2. ZIP THE DATASET
# We zip from '/kaggle/temp/asl_subset_100' -> To -> '/kaggle/working/asl_data_compressed'
print("Zipping dataset... (This allows us to bypass the 500-file upload limit)")

shutil.make_archive(
    '/kaggle/working/asl_subset_100', # Destination (Output folder)
    'zip',                                 # Format
    DEST_ROOT                              # Source (Temp folder)
)

print("------------------------------------------------")
print("SUCCESS! A file named 'asl_data_compressed.zip' has been created.")
print("You can now Save & Run All. Use this ZIP file to create your dataset.")

