import os
import pandas as pd
import shutil
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def sample_severstal_dataset(input_dir='/kaggle/input/severstal-steel-defect-detection', 
                           output_dir='/kaggle/working/severstal-steel-defect-detection_sampling',
                           sample_ratio=0.2,
                           use_stratified=True):
    """
    Sample the Severstal Steel Defect Detection dataset with stratified sampling
    
    Args:
        input_dir: Path to the original dataset
        output_dir: Path where sampled dataset will be saved
        sample_ratio: Ratio of data to sample (0.2 = 20%)
        use_stratified: Whether to use stratified sampling based on defect classes
    """
    
    print(f"Starting dataset sampling with ratio: {sample_ratio}")
    print(f"Using stratified sampling: {use_stratified}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    train_images_out = output_path / 'train_images'
    test_images_out = output_path / 'test_images'
    
    # Create directories
    os.makedirs(train_images_out, exist_ok=True)
    os.makedirs(test_images_out, exist_ok=True)
    
    # Input paths
    input_path = Path(input_dir)
    train_images_in = input_path / 'train_images'
    test_images_in = input_path / 'test_images'
    train_csv_path = input_path / 'train.csv'
    sample_submission_path = input_path / 'sample_submission.csv'
    
    # Read train.csv to get image list
    print("Reading train.csv...")
    train_df = pd.read_csv(train_csv_path)
    
    if use_stratified:
        # Prepare data for stratified sampling
        print("Preparing data for stratified sampling...")
        
        # Create defect class labels for each image
        image_defects = {}
        for _, row in train_df.iterrows():
            img_id = row['ImageId']
            class_id = row['ClassId']
            
            if img_id not in image_defects:
                image_defects[img_id] = set()
            
            # Only add class if there's actually a defect (EncodedPixels is not NaN)
            if pd.notna(row['EncodedPixels']):
                image_defects[img_id].add(class_id)
        
        # Create stratification labels
        stratify_labels = []
        image_ids = []
        
        for img_id in train_df['ImageId'].unique():
            image_ids.append(img_id)
            defects = image_defects.get(img_id, set())
            
            if len(defects) == 0:
                # No defects
                label = 'no_defect'
            elif len(defects) == 1:
                # Single defect
                label = f'single_class_{list(defects)[0]}'
            else:
                # Multiple defects - create combined label
                sorted_defects = sorted(list(defects))
                label = f'multi_class_{"_".join(map(str, sorted_defects))}'
            
            stratify_labels.append(label)
        
        # Show class distribution
        label_counts = Counter(stratify_labels)
        print("\nOriginal class distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} images ({count/len(stratify_labels)*100:.1f}%)")
        
        print(f"\nTotal training images: {len(image_ids)}")
        
        # Perform stratified sampling
        try:
            sampled_train_images, _, sampled_labels, _ = train_test_split(
                image_ids, 
                stratify_labels,
                train_size=sample_ratio,
                stratify=stratify_labels,
                random_state=42
            )
            
            # Show sampled distribution
            sampled_label_counts = Counter(sampled_labels)
            print(f"\nSampled {len(sampled_train_images)} training images")
            print("Sampled class distribution:")
            for label, count in sorted(sampled_label_counts.items()):
                print(f"  {label}: {count} images ({count/len(sampled_train_images)*100:.1f}%)")
                
        except ValueError as e:
            print(f"Stratified sampling failed: {e}")
            print("Falling back to random sampling...")
            n_train_sample = int(len(image_ids) * sample_ratio)
            sampled_train_images = random.sample(image_ids, n_train_sample)
            print(f"Random sampling: {n_train_sample} training images")
    else:
        # Use simple random sampling
        unique_train_images = train_df['ImageId'].unique()
        print(f"Total training images in CSV: {len(unique_train_images)}")
        
        n_train_sample = int(len(unique_train_images) * sample_ratio)
        sampled_train_images = random.sample(list(unique_train_images), n_train_sample)
        print(f"Random sampling: {n_train_sample} training images")
    
    # Copy sampled training images
    print("Copying training images...")
    for img_id in tqdm(sampled_train_images, desc="Training images"):
        src_path = train_images_in / img_id
        dst_path = train_images_out / img_id
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {src_path} not found")
    
    # Filter train.csv for sampled images (this samples the labels too)
    print("Filtering train.csv and sampling labels...")
    sampled_train_df = train_df[train_df['ImageId'].isin(sampled_train_images)]
    sampled_train_df.to_csv(output_path / 'train.csv', index=False)
    
    # Verify label sampling
    original_defect_rows = train_df[train_df['EncodedPixels'].notna()]
    sampled_defect_rows = sampled_train_df[sampled_train_df['EncodedPixels'].notna()]
    
    print(f"Original train.csv: {len(train_df)} total rows")
    print(f"  - Rows with defects: {len(original_defect_rows)}")
    print(f"  - Rows without defects (NaN): {len(train_df) - len(original_defect_rows)}")
    
    print(f"Sampled train.csv: {len(sampled_train_df)} total rows")
    print(f"  - Rows with defects: {len(sampled_defect_rows)}")
    print(f"  - Rows without defects (NaN): {len(sampled_train_df) - len(sampled_defect_rows)}")
    
    # Show class distribution in labels
    print("\nOriginal label distribution by class:")
    for class_id in sorted(train_df['ClassId'].unique()):
        class_defects = original_defect_rows[original_defect_rows['ClassId'] == class_id]
        print(f"  Class {class_id}: {len(class_defects)} defect annotations")
    
    print("Sampled label distribution by class:")
    for class_id in sorted(sampled_train_df['ClassId'].unique()):
        class_defects = sampled_defect_rows[sampled_defect_rows['ClassId'] == class_id]
        print(f"  Class {class_id}: {len(class_defects)} defect annotations")
    
    # Sample test images
    test_images_list = list(test_images_in.glob('*.jpg'))
    n_test_sample = int(len(test_images_list) * sample_ratio)
    sampled_test_images = random.sample(test_images_list, n_test_sample)
    print(f"Sampling {n_test_sample} test images out of {len(test_images_list)}")
    
    # Copy sampled test images
    print("Copying test images...")
    sampled_test_image_ids = []
    for img_path in tqdm(sampled_test_images, desc="Test images"):
        dst_path = test_images_out / img_path.name
        shutil.copy2(img_path, dst_path)
        sampled_test_image_ids.append(img_path.name)
    
    # Filter sample_submission.csv for sampled test images (samples test labels too)
    print("Filtering sample_submission.csv and sampling test labels...")
    sample_submission_df = pd.read_csv(sample_submission_path)
    
    # Check the actual column names in sample_submission.csv
    print(f"Sample submission columns: {list(sample_submission_df.columns)}")
    
    # The sample_submission.csv appears to have ImageId, EncodedPixels, ClassId columns
    # Filter based on the sampled test image IDs
    sampled_submission_df = sample_submission_df[
        sample_submission_df['ImageId'].isin(sampled_test_image_ids)
    ]
    sampled_submission_df.to_csv(output_path / 'sample_submission.csv', index=False)
    
    print(f"Original sample_submission.csv: {len(sample_submission_df)} rows")
    print(f"Sampled sample_submission.csv: {len(sampled_submission_df)} rows")
    
    # Show test label sampling details
    original_test_images = sample_submission_df['ImageId'].unique()
    sampled_test_images_in_sub = sampled_submission_df['ImageId'].unique()
    
    print(f"Test images in original submission: {len(original_test_images)}")
    print(f"Test images in sampled submission: {len(sampled_test_images_in_sub)}")
    print(f"Classes per test image: {len(sample_submission_df) // len(original_test_images) if len(original_test_images) > 0 else 0}")
    
    # Print summary
    print("\n" + "="*50)
    print("SAMPLING SUMMARY")
    print("="*50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample ratio: {sample_ratio * 100}%")
    print(f"Training images: {len(sampled_train_images)}")
    print(f"Test images: {len(sampled_test_images)}")
    print(f"Train CSV rows: {len(sampled_train_df)}")
    print(f"Sample submission rows: {len(sampled_submission_df)}")
    print("="*50)
    
    return {
        'train_images': len(sampled_train_images),
        'test_images': len(sampled_test_images),
        'train_csv_rows': len(sampled_train_df),
        'submission_rows': len(sampled_submission_df)
    }

# Set random seed for reproducibility
random.seed(42)

# Run the sampling
if __name__ == "__main__":
    # Adjust these paths according to your Kaggle setup
    INPUT_DIR = '/kaggle/input/severstal-steel-defect-detection'
    OUTPUT_DIR = '/kaggle/working/severstal-steel-defect-detection_sampling'  # Saves to /kaggle/working
    SAMPLE_RATIO = 0.2  # 20% sampling
    
    # Alternative: Save directly to /kaggle/working without subdirectory
    # OUTPUT_DIR = '/kaggle/working'
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found!")
        print("Please check the dataset path in your Kaggle environment.")
    else:
        # Run sampling
        results = sample_severstal_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            sample_ratio=SAMPLE_RATIO,
            use_stratified=True  # Set to False for simple random sampling
        )
        
        print(f"\nSampling completed successfully!")
        print(f"Sampled dataset saved to: {OUTPUT_DIR}")

