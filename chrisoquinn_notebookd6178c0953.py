"""
Kaggle Notebook: Create ImageNet Subset

Run this in a Kaggle notebook with the ImageNet dataset attached.
The output will be available in /kaggle/working/ for download.
"""

import os
import shutil
import random
from pathlib import Path

def create_imagenet_subset():
    """
    Create a subset from Kaggle ImageNet targeting ~10GB total size.
    Sample ~35 images per class to get approximately 35k total images.
    """
    print("Creating ImageNet subset targeting ~10GB...")
    
    # Kaggle ImageNet paths
    train_dir = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
    
    # Output directories
    output_base = '/kaggle/working/imagenet_subset'
    output_train = os.path.join(output_base, 'train')
    
    # Create output directories
    os.makedirs(output_train, exist_ok=True)
    
    # Get all class directories from training set
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(class_dirs)} classes in training set")
    
    # Target ~35 images per class for ~10GB total
    # (35k images * ~300KB average = ~10GB)
    samples_per_class = 35
    total_copied = 0
    
    print(f"Sampling {samples_per_class} images per class")
    print(f"Expected total: {len(class_dirs) * samples_per_class} images (~10GB)")
    
    # Process each class
    for class_name in class_dirs:
        class_train_dir = os.path.join(train_dir, class_name)
        
        # Get all images in this class
        image_files = [f for f in os.listdir(class_train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < samples_per_class:
            print(f"Warning: Class {class_name} only has {len(image_files)} images, using all")
            use_count = len(image_files)
        else:
            use_count = samples_per_class
        
        # Randomly sample images
        random.shuffle(image_files)
        selected_images = image_files[:use_count]
        
        # Create class directory in output
        output_class_dir = os.path.join(output_train, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Copy selected images
        for img_file in selected_images:
            src = os.path.join(class_train_dir, img_file)
            dst = os.path.join(output_class_dir, img_file)
            shutil.copy2(src, dst)
            total_copied += 1
        
        if (len([d for d in os.listdir(output_train) if os.path.isdir(os.path.join(output_train, d))]) % 100) == 0:
            print(f"Processed {len([d for d in os.listdir(output_train) if os.path.isdir(os.path.join(output_train, d))])} classes...")
    
    print(f"Subset creation complete!")
    print(f"Total images: {total_copied}")
    print(f"Expected: {len(class_dirs) * samples_per_class}")
    
    return output_base

def zip_subset(subset_dir):
    """
    Zip the subset for download from Kaggle.
    """
    print("Zipping subset...")
    
    zip_path = '/kaggle/working/imagenet_subset.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(subset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Get relative path for zip
                arcname = os.path.relpath(file_path, '/kaggle/working/')
                zipf.write(file_path, arcname)
    
    print(f"Zip created: {zip_path}")
    
    # Get zip file size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Zip file size: {zip_size_mb:.1f} MB")
    
    return zip_path

def main():
    """
    Main function to create subset (no zipping due to space constraints).
    """
    random.seed(42)  # For reproducibility
    
    # Create the subset
    subset_dir = create_imagenet_subset()
    
    print("Subset creation complete!")
    print("Files are in /kaggle/working/imagenet_subset/")
    print("Use Kaggle's dataset creation feature to save this as a dataset.")
    print("Then you can access it from Colab without downloading.")

if __name__ == "__main__":
    main()

