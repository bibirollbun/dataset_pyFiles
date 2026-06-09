import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm


train_paths = [
    "/kaggle/input/grocery-items-multi-class-object-detection/train/train/images",
    "/kaggle/input/ccs-v1/Output/2025-09-03-20-08-05/train/images",
    "/kaggle/input/ccs-v2/Output/2025-09-04-15-28-49/train/images", 
    "/kaggle/input/ccs-v3/Output/2025-09-06-16-31-47/train/images",
    "/kaggle/input/ccs-v4/Output/2025-09-06-18-03-18/train/images",
    "/kaggle/input/ccs-v5/Output/2025-09-08-13-05-38/train/images",
    "/kaggle/input/ccs-v6/Output/2025-09-08-14-44-05/train/images",
    "/kaggle/input/css-v7/Output/2025-09-07-09-44-39/train/images",
    "/kaggle/input/css-v8/Output/2025-09-07-10-18-51/train/images"
]
val_paths = [
    "/kaggle/input/grocery-items-multi-class-object-detection/val/val/images",
    "/kaggle/input/ccs-v1/Output/2025-09-03-20-08-05/val/images",
    "/kaggle/input/ccs-v2/Output/2025-09-04-15-28-49/val/images",
    "/kaggle/input/ccs-v3/Output/2025-09-06-16-31-47/val/images",
    "/kaggle/input/ccs-v4/Output/2025-09-06-18-03-18/val/images",
    "/kaggle/input/ccs-v5/Output/2025-09-08-13-05-38/val/images",
    "/kaggle/input/ccs-v6/Output/2025-09-08-14-44-05/val/images",
    "/kaggle/input/css-v7/Output/2025-09-07-09-44-39/val/images",
    "/kaggle/input/css-v8/Output/2025-09-07-10-18-51/val/images"
]


# Output dir
output_dir = "/kaggle/working/preprocess"
os.makedirs(output_dir, exist_ok=True)

# train and val dirs
train_output_dir = os.path.join(output_dir, "train")
val_output_dir = os.path.join(output_dir, "val")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# images and labels in train and val
train_images_dir = os.path.join(train_output_dir, "images")
train_labels_dir = os.path.join(train_output_dir, "labels")
val_images_dir = os.path.join(val_output_dir, "images")
val_labels_dir = os.path.join(val_output_dir, "labels")

for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(dir_path, exist_ok=True)


def copy_files(source_paths, output_images_dir, output_labels_dir):
    counter = 1
    for source_path in source_paths:
        # label dir
        labels_path = source_path.replace("images", "labels")
        
        # Check dirs
        if not os.path.exists(source_path):
            print(f"⚠️ Error {source_path}, not found")
            continue
        if not os.path.exists(labels_path):
            print(f"⚠️ Error {labels_path}, not found")
            continue
        
        # Take all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(source_path, ext)))
        
        print(f"Search {len(image_files)} images in {source_path}")

        for image_path in tqdm(image_files):
            filename = Path(image_path).stem
            image_extension = Path(image_path).suffix
            
            # Path to label
            label_path = os.path.join(labels_path, filename + ".txt")
            
            # New name
            new_filename = f"{counter:06d}"
            new_image_name = new_filename + image_extension
            new_label_name = new_filename + ".txt"
            
            # save path
            new_image_path = os.path.join(output_images_dir, new_image_name)
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            
            # Copy image
            shutil.copy2(image_path, new_image_path)
            
            # Copy label
            if os.path.exists(label_path):
                shutil.copy2(label_path, new_label_path)
            else:
                print(f"label {label_path} not found")
            
            counter += 1

print("train...")
copy_files(train_paths, train_images_dir, train_labels_dir)

print("val...")
copy_files(val_paths, val_images_dir, val_labels_dir)


train_images_count = len(os.listdir(train_images_dir))
train_labels_count = len(os.listdir(train_labels_dir))
val_images_count = len(os.listdir(val_images_dir))
val_labels_count = len(os.listdir(val_labels_dir))

print(f"Train images: {train_images_count}")
print(f"Train labels: {train_labels_count}")
print(f"Val images: {val_images_count}")
print(f"Val labels: {val_labels_count}")

