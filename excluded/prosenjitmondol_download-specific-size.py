import os

# 1. List everything in the main input folder
print("--- WHAT IS IN YOUR INPUT FOLDER? ---")
try:
    items = os.listdir("/kaggle/input")
    print(items)
except FileNotFoundError:
    print("ERROR: /kaggle/input folder does not exist.")

# 2. If it's empty '[]', the data is NOT added.
if len(items) == 0:
    print("\n>>> RESULT: The folder is EMPTY. Please complete Step 2 above.")
else:
    print(f"\n>>> RESULT: Found {len(items)} item(s).")
    # If we found something, let's see what the full path is
    for item in items:
        print(f"Path found: /kaggle/input/{item}")


import os
import shutil

# --- CONFIGURATION ---
# We build the path based on what you found
base_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
input_folder = os.path.join(base_path, "train_images")
output_folder = "/kaggle/working/train_images_subset"
target_size_gb = 4
# ---------------------

# 1. Verification Check
print(f"Checking for images in: {input_folder}...")
if not os.path.exists(input_folder):
    print("\nERROR: 'train_images' folder not found.")
    print(f"Contents of main folder '{base_path}':")
    print(os.listdir(base_path))
    print("\nPlease update the 'input_folder' variable if you see the image folder named differently above.")
else:
    # 2. Start Copying
    max_size_bytes = target_size_gb * 1024**3
    current_size_bytes = 0
    
    print(f"Folder found! Starting copy of ~{target_size_gb}GB...")
    
    # Clean output folder if it exists from previous attempts
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Get list of study folders
    studies = sorted(os.listdir(input_folder))
    count = 0
    
    for study in studies:
        if current_size_bytes >= max_size_bytes:
            break
            
        src_path = os.path.join(input_folder, study)
        dst_path = os.path.join(output_folder, study)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            count += 1
            
            # Calculate size
            for root, _, files in os.walk(dst_path):
                current_size_bytes += sum(os.path.getsize(os.path.join(root, name)) for name in files)

            if count % 20 == 0:
                print(f"Copied {count} studies... ({current_size_bytes / 1024**2:.0f} MB)")

    print(f"Copy Finished. Total Size: {current_size_bytes / 1024**3:.2f} GB")

    # 3. Zip the files
    print("Zipping files... (Please wait, this takes about 1-2 minutes)")
    shutil.make_archive("/kaggle/working/subset_4gb", 'zip', output_folder)
    
    print("\nSUCCESS! ===================================================")
    print("1. Look at the 'Output' section in the right sidebar.")
    print("2. Find 'subset_4gb.zip'.")
    print("3. Click the three dots (...) and select 'Download'.")


from IPython.display import FileLink, display

# Create a clickable link for the file
print("Click the blue link below to start the download:")
display(FileLink(r'subset_4gb.zip'))

