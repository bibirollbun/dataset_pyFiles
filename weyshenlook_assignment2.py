# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# (1) Explore Dataset
# Dataset path
data_path = "/kaggle/input/siim-isic-melanoma-classification"

# List files
print(os.listdir(data_path))


# Load Metadata (CSV File)
train_df = pd.read_csv(f"{data_path}/train.csv")
test_df = pd.read_csv(f"{data_path}/test.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining data columns:", train_df.columns.tolist())
print("\nFirst 5 rows:")
print(train_df.head())


# (2) Metadata Cleaning and Preprocessing
def clean_metadata(df, is_train=True):
    """Clean and preprocess metadata"""
    df_clean = df.copy()
    
    # Handle missing values in age
    if 'age_approx' in df_clean.columns:
        # Fill missing age with median
        median_age = df_clean['age_approx'].median()
        df_clean['age_approx'].fillna(median_age, inplace=True)
        
        # Normalize age (0-1 scale)
        df_clean['age_normalized'] = df_clean['age_approx'] / 100.0
        print(f"Age missing values filled with median: {median_age}")
    
    # Handle missing values in sex
    if 'sex' in df_clean.columns:
        # Fill missing sex with mode
        mode_sex = df_clean['sex'].mode()[0] if not df_clean['sex'].mode().empty else 'male'
        df_clean['sex'].fillna(mode_sex, inplace=True)
        
        # One-hot encode sex
        sex_dummies = pd.get_dummies(df_clean['sex'], prefix='sex')
        df_clean = pd.concat([df_clean, sex_dummies], axis=1)
        print(f"Sex missing values filled with mode  : {mode_sex}")
    
    # Handle missing values in anatomical site
    if 'anatom_site_general_challenge' in df_clean.columns:
        # Fill missing site with 'unknown'
        df_clean['anatom_site_general_challenge'].fillna('unknown', inplace=True)
        
        # One-hot encode anatomical site
        site_dummies = pd.get_dummies(df_clean['anatom_site_general_challenge'], prefix='site')
        df_clean = pd.concat([df_clean, site_dummies], axis=1)
        print("Anatomical site missing values filled with 'unknown'")
    
    # Create additional features
    if is_train and 'target' in df_clean.columns:
        # Calculate class weights for imbalanced dataset
        target_counts = df_clean['target'].value_counts()
        print(f"\nClass distribution:")
        print(f"Benign (0)   : {target_counts[0]} ({target_counts[0]/len(df_clean)*100:.2f}%)")
        print(f"Malignant (1): {target_counts[1]} ({target_counts[1]/len(df_clean)*100:.2f}%)")
    
    return df_clean


# Clean training and test metadata
train_clean = clean_metadata(train_df, is_train=True)
test_clean = clean_metadata(test_df, is_train=False)

print(f"\nCleaned training data shape: {train_clean.shape}")
print(f"Cleaned test data shape    : {test_clean.shape}")


# (3) Patient-based Train/Validation Split
def create_patient_split(df, test_size=0.2, random_state=42):
    """Create train/validation split by patient ID to avoid data leakage"""
    
    # Get unique patients
    unique_patients = df['patient_id'].unique()
    print(f"Total unique patients: {len(unique_patients)}")
    
    # Split patients (not individual images)
    train_patients, val_patients = train_test_split(
        unique_patients, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Can't stratify by patient easily, would need more complex logic
    )
    
    # Create train/validation dataframes
    train_split = df[df['patient_id'].isin(train_patients)].copy()
    val_split = df[df['patient_id'].isin(val_patients)].copy()
    
    print(f"Training patients    : {len(train_patients)}")
    print(f"Validation patients  : {len(val_patients)}")
    print(f"Training images      : {len(train_split)}")
    print(f"Validation images    : {len(val_split)}")
    
    # Check target distribution in splits
    if 'target' in df.columns:
        print(f"\nTarget distribution in training split:")
        print(train_split['target'].value_counts(normalize=True))
        print(f"\nTarget distribution in validation split:")
        print(val_split['target'].value_counts(normalize=True))
    
    return train_split, val_split


# Create patient-based split
train_split, val_split = create_patient_split(train_clean)


# (4) Image Preprocessing Functions
class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values to [0, 1]
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            # Apply augmentation if specified
            if augment:
                image = self.apply_augmentation(image)
                
            return image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image):
        """Apply basic data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (small angle)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)
            
        return image
    
    def preprocess_batch(self, image_paths, augment=False, batch_size=32):
        """Preprocess a batch of images"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            img = self.load_and_preprocess_image(path, augment=augment)
            if img is not None:
                images.append(img)
                valid_paths.append(path)
                
        return np.array(images), valid_paths


# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)


# (5) Create Data Loading Functions
def create_image_paths(df, image_dir):
    """Create full image paths from dataframe"""
    return [os.path.join(image_dir, f"{img_id}.jpg") for img_id in df['image_name']]

def save_processed_data(train_df, val_df, test_df, output_dir='processed_data'):
    """Save processed dataframes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned metadata
    train_df.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_processed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    
    print(f"Processed data saved to {output_dir}/")
    
    # Save preprocessing summary
    with open(os.path.join(output_dir, 'preprocessing_summary.txt'), 'w') as f:
        f.write("SIIM-ISIC Data Preprocessing Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Image target size: {preprocessor.target_size}\n")
        f.write(f"Normalization applied: {preprocessor.normalize}\n")
        
        # Feature columns
        feature_cols = [col for col in train_df.columns if col not in ['image_name', 'patient_id', 'target']]
        f.write(f"\nFeature columns ({len(feature_cols)}):\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")


# (6) Execute Preprocessing Pipeline
# Create image paths
train_paths = create_image_paths(train_split, f"{data_path}/jpeg/train/")
val_paths = create_image_paths(val_split, f"{data_path}/jpeg/train/")
test_paths = create_image_paths(test_clean, f"{data_path}/jpeg/test/")


# Validate that images exist
def validate_image_paths(paths, df, split_name):
    """Validate that image files exist"""
    existing_paths = []
    valid_indices = []
    
    for i, path in enumerate(paths):
        if os.path.exists(path):
            existing_paths.append(path)
            valid_indices.append(i)
    
    print(f"{split_name}: {len(existing_paths)}/{len(paths)} images found")
    return df.iloc[valid_indices].reset_index(drop=True), existing_paths


# Validate all splits
train_final, train_paths_final = validate_image_paths(train_paths, train_split, "Training")
val_final, val_paths_final = validate_image_paths(val_paths, val_split, "Validation")
test_final, test_paths_final = validate_image_paths(test_paths, test_clean, "Test")


# Save processed data
save_processed_data(train_final, val_final, test_final)


print(f"Training samples   : {len(train_final)}")
print(f"Validation samples : {len(val_final)}")
print(f"Test samples       : {len(test_final)}")


# Load and display a sample image
if len(train_paths_final) > 0:
    # Load a sample image
    sample_image = preprocessor.load_and_preprocess_image(train_paths_final[0])
    if sample_image is not None:
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample image data type: {sample_image.dtype}")
        print(f"Sample image value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(sample_image)
        plt.title("Sample Preprocessed Image")
        plt.axis('off')
        plt.show()










