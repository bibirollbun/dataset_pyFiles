import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def load_and_display_samples(train_df, image_dir, n_samples=5):
    """
    Load and display sample images for each cancer subtype
    """
    # Set style for better visualization
    plt.style.use('seaborn')
    
    # Get unique subtypes
    subtypes = train_df['label'].unique()
    
    # Create a figure
    plt.figure(figsize=(20, 4*len(subtypes)))
    
    # For each subtype
    for idx, subtype in enumerate(subtypes):
        # Get sample images for this subtype
        subtype_df = train_df[train_df['label'] == subtype]
        valid_images = 0
        
        # Keep sampling until we get enough valid images
        for _, row in subtype_df.sample(frac=1).iterrows():  # Shuffle and iterate
            if valid_images >= n_samples:
                break
                
            image_path = os.path.join(image_dir, f"{str(row['image_id'])}_thumbnail.png")
            
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    plt.subplot(len(subtypes), n_samples, idx*n_samples + valid_images + 1)
                    plt.imshow(img)
                    plt.axis('off')
                    if valid_images == 0:  # Only add label for first image in row
                        plt.title(f'{subtype}\n(n={len(subtype_df)})', fontsize=12, pad=20)
                    valid_images += 1
            except:
                continue
    
    plt.tight_layout()
    plt.show()


def main():
    # Load the training data
    train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
    
    # Print initial data info
    print("Dataset Overview:")
    print(f"Total number of images: {len(train_df)}")
    print("\nDistribution of subtypes:")
    print(train_df['label'].value_counts())
    print("\nSample of image IDs:")
    print(train_df['image_id'].head())
    
    # Define image directory 
    train_image_dir = '/kaggle/input/UBC-OCEAN/train_thumbnails'
    
    # Verify directory exists
    if not os.path.exists(train_image_dir):
        print(f"Error: Directory '{train_image_dir}' not found")
        return
    
    # Display visualization
    print("\nDisplaying cancer subtype samples...")
    load_and_display_samples(train_df, train_image_dir)
    
    # Display image size statistics
    print("\nImage size statistics:")
    print("\nWidth:")
    print(train_df['image_width'].describe())
    print("\nHeight:")
    print(train_df['image_height'].describe())

if __name__ == "__main__":
    main()


train_df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")
train_df.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os


def perform_eda(train_df):
    """
    Perform comprehensive EDA on the UBC-OCEAN dataset
    """
    # Set the style for better visualizations
    plt.style.use('seaborn')
    
    # 1. Class Distribution Analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(x=train_df['label'].value_counts().index, 
                y=train_df['label'].value_counts().values)
    plt.title('Distribution of Cancer Subtypes')
    plt.xlabel('Subtype')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    #plt.savefig("class_distribution.png")
    plt.show()
    
    # 2. Image Size Distribution
    plt.figure(figsize=(15, 5))
    
    # Width distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x='image_width', bins=30)
    plt.title('Distribution of Image Widths')
    plt.xlabel('Width (pixels)')
    
    # Height distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x='image_height', bins=30)
    plt.title('Distribution of Image Heights')
    plt.xlabel('Height (pixels)')
    plt.tight_layout()
    plt.show()
    
    # 3. TMA vs WSI Analysis
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x='label', hue='is_tma')
    plt.title('Distribution of TMA vs WSI across Subtypes')
    plt.xlabel('Subtype')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Is TMA')
    plt.tight_layout()
    #plt.savefig("tma_wsi.png")
    plt.show()
    
    # 4. Image Aspect Ratio Analysis
    train_df['aspect_ratio'] = train_df['image_width'] / train_df['image_height']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='label', y='aspect_ratio')
    plt.title('Image Aspect Ratios by Subtype')
    plt.xlabel('Subtype')
    plt.ylabel('Aspect Ratio (Width/Height)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig("image_ratio.png")
    plt.show()
    
    # 5. Print Statistical Summary
    print("\nStatistical Summary by Subtype:")
    summary_stats = train_df.groupby('label').agg({
        'image_width': ['mean', 'std', 'min', 'max'],
        'image_height': ['mean', 'std', 'min', 'max'],
        'is_tma': 'sum'
    }).round(2)
    print(summary_stats)
    
    # 6. Image Size Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=train_df, x='image_width', y='image_height', 
                    hue='label', style='is_tma', alpha=0.6)
    plt.title('Image Dimensions by Subtype and Type (TMA vs WSI)')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig("scatter_plot.png")
    plt.show()



def main():
    # Load the data
    train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
    
    print("Starting Exploratory Data Analysis...")
    print("\nDataset Overview:")
    print(f"Total number of samples: {len(train_df)}")
    print(f"Number of unique subtypes: {train_df['label'].nunique()}")
    print(f"Number of TMA images: {train_df['is_tma'].sum()}")
    print(f"Number of WSI images: {(~train_df['is_tma']).sum()}")
    
    # Perform EDA
    perform_eda(train_df)

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def detect_outliers(train_df):
    """
    Perform comprehensive outlier detection analysis
    """
    # Create figure for multiple plots
    plt.figure(figsize=(20, 15))
    
    # 1. Z-score based outlier detection for image dimensions
    train_df['width_zscore'] = np.abs(stats.zscore(train_df['image_width']))
    train_df['height_zscore'] = np.abs(stats.zscore(train_df['image_height']))
    
    # 2. Calculate additional features for outlier detection
    train_df['area'] = train_df['image_width'] * train_df['image_height']
    train_df['aspect_ratio'] = train_df['image_width'] / train_df['image_height']
    
    # Plot 1: Area vs Aspect Ratio with outlier boundaries
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=train_df, x='area', y='aspect_ratio', hue='label', alpha=0.6)
    plt.title('Image Area vs Aspect Ratio\nPotential Outliers Detection')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Aspect Ratio')
    
    # Plot 2: Width vs Height with Outlier Boundaries
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=train_df, x='width_zscore', y='height_zscore', 
                    hue='label', alpha=0.6)
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.3)
    plt.title('Z-scores of Width vs Height\nRed lines indicate z-score = 3')
    plt.xlabel('Width Z-score')
    plt.ylabel('Height Z-score')
    
    # Plot 3: Box plot of image areas by subtype
    plt.subplot(2, 2, 3)
    sns.boxplot(data=train_df, x='label', y='area')
    plt.title('Distribution of Image Areas by Subtype')
    plt.xticks(rotation=45)
    plt.ylabel('Area (pixels²)')
    
    # Plot 4: Density plot of aspect ratios
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=train_df, x='aspect_ratio', hue='label')
    plt.title('Density Distribution of Aspect Ratios')
    plt.xlabel('Aspect Ratio')
    
    plt.tight_layout()
    #plt.savefig("outlier.png")
    plt.show()
    
    # Print statistical outliers
    print("\nPotential Outliers Analysis:")
    
    # Z-score based outliers (|z| > 3)
    width_outliers = train_df[train_df['width_zscore'] > 3]
    height_outliers = train_df[train_df['height_zscore'] > 3]
    
    print(f"\nImages with unusual width (Z-score > 3): {len(width_outliers)}")
    print(f"Images with unusual height (Z-score > 3): {len(height_outliers)}")
    
    # IQR based outlier detection for area
    Q1 = train_df['area'].quantile(0.25)
    Q3 = train_df['area'].quantile(0.75)
    IQR = Q3 - Q1
    area_outliers = train_df[(train_df['area'] < (Q1 - 1.5 * IQR)) | 
                            (train_df['area'] > (Q3 + 1.5 * IQR))]
    
    print(f"\nImages with unusual area (IQR method): {len(area_outliers)}")
    
    # Print summary of extreme cases
    print("\nExtreme Cases Summary:")
    extremes = pd.DataFrame({
        'Metric': ['Smallest Area', 'Largest Area', 'Most Square', 'Least Square'],
        'Image ID': [
            train_df.loc[train_df['area'].idxmin(), 'image_id'],
            train_df.loc[train_df['area'].idxmax(), 'image_id'],
            train_df.loc[(train_df['aspect_ratio'] - 1).abs().idxmin(), 'image_id'],
            train_df.loc[(train_df['aspect_ratio'] - 1).abs().idxmax(), 'image_id']
        ],
        'Subtype': [
            train_df.loc[train_df['area'].idxmin(), 'label'],
            train_df.loc[train_df['area'].idxmax(), 'label'],
            train_df.loc[(train_df['aspect_ratio'] - 1).abs().idxmin(), 'label'],
            train_df.loc[(train_df['aspect_ratio'] - 1).abs().idxmax(), 'label']
        ]
    })
    print(extremes)
    
    return width_outliers, height_outliers, area_outliers


def main():
    # Load the data
    train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
    
    print("Starting Outlier Detection Analysis...")
    width_outliers, height_outliers, area_outliers = detect_outliers(train_df)
    
    # Save outlier information for future reference
    outlier_summary = pd.DataFrame({
        'image_id': list(set(width_outliers['image_id'].tolist() + 
                           height_outliers['image_id'].tolist() + 
                           area_outliers['image_id'].tolist())),
        'is_outlier': True
    })
    
    print("\nTotal unique outliers detected:", len(outlier_summary))

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_class_imbalance(train_df):
    """
    Perform comprehensive class imbalance analysis
    """
    # Set style
    plt.style.use('seaborn')
    
    # Create figure for multiple plots
    plt.figure(figsize=(15, 10))
    
    # 1. Class Distribution Plot
    plt.subplot(2, 2, 1)
    class_counts = train_df['label'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Subtype')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for i, v in enumerate(class_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    # 2. Percentage Distribution
    plt.subplot(2, 2, 2)
    class_percentages = (class_counts / len(train_df) * 100).round(2)
    sns.barplot(x=class_percentages.index, y=class_percentages.values)
    plt.title('Class Distribution (%)')
    plt.xlabel('Subtype')
    plt.ylabel('Percentage')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(class_percentages.values):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Pie Chart
    plt.subplot(2, 2, 3)
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette('husl', n_colors=len(class_counts)))
    plt.title('Class Distribution (Pie Chart)')
    
    # 4. Imbalance Metrics Table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate imbalance metrics
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    metrics_text = (
        f'Imbalance Analysis:\n\n'
        f'Total Samples: {len(train_df)}\n'
        f'Number of Classes: {len(class_counts)}\n'
        f'Majority Class (HGSC): {majority_class}\n'
        f'Minority Class (MC): {minority_class}\n'
        f'Imbalance Ratio: {imbalance_ratio:.2f}:1\n\n'
        f'Class Distribution:\n'
    )
    
    for class_name, percentage in class_percentages.items():
        metrics_text += f'{class_name}: {percentage:.1f}%\n'
    
    plt.text(0.1, 0.9, metrics_text, fontsize=10, va='top')
    
    plt.tight_layout()
    #plt.savefig("class_imbalance.png")
    plt.show()
    
    # Print additional analysis
    print("\nDetailed Class Imbalance Analysis:")
    print("\nClass Counts:")
    print(class_counts)
    
    print("\nClass Percentages:")
    print(class_percentages)
    
    print("\nImbalance Ratios (relative to majority class):")
    imbalance_ratios = majority_class / class_counts
    print(imbalance_ratios)
    
    # Suggest potential strategies
    print("\nRecommended Strategies based on Imbalance:")
    if imbalance_ratio > 4:
        print("- Consider using class weights in model")
        print("- Implement oversampling techniques (e.g., SMOTE) for minority classes")
        print("- Use stratified sampling in train/validation split")
    if imbalance_ratio > 2:
        print("- Use balanced accuracy or F1-score as metrics")
        print("- Consider ensemble methods with balanced class weights")
    
    return class_counts, class_percentages, imbalance_ratio


def main():
    # Load the data
    train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
    
    print("Starting Class Imbalance Analysis...")
    class_counts, class_percentages, imbalance_ratio = analyze_class_imbalance(train_df)

if __name__ == "__main__":
    main()


!pip install -q -U albumentations


!pip install -q tiatoolbox


import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from collections import Counter
import albumentations as A
from imblearn.over_sampling import SMOTE
from scipy.ndimage import gaussian_filter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torchvision.models as models
from tiatoolbox import logger
from tiatoolbox.tools import stainnorm, patchextraction
from tiatoolbox.tools.stainaugment import StainAugmentor
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import PIL.Image
import gc
import warnings
warnings.filterwarnings("ignore")


class EnhancedPreprocessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224), 
                 wsi_magnification: float = 20.0,
                 tma_magnification: float = 40.0,
                 stain_norm_method: str = 'reinhard'):
        self.target_size = target_size
        self.wsi_magnification = wsi_magnification
        self.tma_magnification = tma_magnification
        
        # Initialize stain normalizer
        if stain_norm_method == 'macenko':
            self.normalizer = stainnorm.MacenkoNormalizer()
        elif stain_norm_method == 'vahadane':
            self.normalizer = stainnorm.VahadaneNormalizer()
        elif stain_norm_method == 'reinhard':
            self.normalizer = stainnorm.ReinhardNormalizer()
        elif stain_norm_method == 'ruifrok':
            self.normalizer = stainnorm.RuifrokNormalizer()
        else:
            raise ValueError(f"Unknown stain normalization method: {stain_norm_method}")

    def detect_image_type(self, image: np.ndarray) -> str:
        """Determine if image is WSI or TMA based on size"""
        height, width = image.shape[:2]
        if height <= 5000 and width <= 5000:
            return 'TMA'
        return 'WSI'
    
    def normalize_magnification(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Normalize image magnification"""
        if image_type == 'TMA':
            scale_factor = self.wsi_magnification / self.tma_magnification
            new_size = (int(image.shape[1] * scale_factor), 
                       int(image.shape[0] * scale_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def apply_stain_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply stain normalization with error handling"""
        try:
            self.normalizer.fit(image)
            normalized = self.normalizer.transform(image)
            return normalized
        except Exception as e:
            print(f"Error in stain normalization: {str(e)}")
            return image  # Return original image if normalization fails

    def detect_tissue(self, image: np.ndarray) -> np.ndarray:
        """Improved tissue detection using LAB color space and adaptive thresholding"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask > 0

    def extract_tissue_region(self, image: np.ndarray) -> np.ndarray:
        """Extract main tissue region"""
        try:
            # Get tissue mask
            tissue_mask = self.detect_tissue(image)
            
            # Find contours
            contours, _ = cv2.findContours(tissue_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract region with padding
            pad = 10
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image.shape[1], x + w + pad)
            y_end = min(image.shape[0], y + h + pad)
            
            return image[y_start:y_end, x_start:x_end]
            
        except Exception as e:
            print(f"Error in tissue extraction: {str(e)}")
            return image

    def handle_image_dimensions(self, image: np.ndarray) -> np.ndarray:
        """Handle different image dimensions based on size"""
        height, width = image.shape[:2]
        
        # Handle very large WSIs
        if width > 50000 or height > 50000:
            scale_factor = min(50000 / width, 50000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"Resizing large WSI from {width}x{height} to {new_width}x{new_height}")
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline"""
        try:
            # Handle large image dimensions first
            image = self.handle_image_dimensions(image)
            
            # Determine image type
            image_type = self.detect_image_type(image)
            
            # Normalize magnification
            image = self.normalize_magnification(image, image_type)
            
            # Extract tissue region
            image = self.extract_tissue_region(image)
            
            # Apply stain normalization
            image = self.apply_stain_normalization(image)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            return image

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            # Return resized original image as fallback
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)


def create_transforms(image_size: Tuple[int, int] = (224, 224), stain_augment_prob: float = 0.5):
    """Create augmentation transforms with advanced techniques"""
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.RandomGamma(p=1.0)
        ], p=0.5),
        # Geometric augmentations
        A.OneOf([
            A.ElasticTransform(p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5)
        ], p=0.3),
        # Cutout augmentation
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
        # Basic transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True)
    ])
    
    return train_transform, val_transform
    


class HistoDataset(Dataset):
    """Enhanced dataset with robust preprocessing"""
    def __init__(self, df: pd.DataFrame, image_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True,
                 apply_smote: bool = True,
                 stain_norm_method: str = 'reinhard'):
        
        self.df = df.copy()  # Make a copy to prevent modifications
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        self.apply_smote = apply_smote and is_training
        
        # Verify required columns exist
        required_columns = ['image_id', 'encoded_label']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Initialize enhanced preprocessor
        self.preprocessor = EnhancedPreprocessor(
            stain_norm_method=stain_norm_method
        )
        
        # Load and preprocess images
        print("Loading and preprocessing images...")
        self._load_images()
        
        if self.apply_smote and len(self.processed_images) > 0:
            self._apply_smote_preprocessing()
    
    def _load_images(self):
        """Load and preprocess all images"""
        valid_indices = []
        self.processed_images = []
        self.labels = []
        
        for idx in tqdm(range(len(self.df)), desc="Processing images"):
            try:
                image_path = os.path.join(self.image_dir,
                                        f"{self.df.iloc[idx]['image_id']}_thumbnail.png")
                if os.path.exists(image_path):
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image - {image_path}")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess image
                    processed_image = self.preprocessor.preprocess_image(image)
                    
                    self.processed_images.append(processed_image)
                    valid_indices.append(idx)
                    self.labels.append(self.df.iloc[idx]['encoded_label'])
                else:
                    print(f"Warning: Image not found - {image_path}")
            
            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue
        
        if len(valid_indices) == 0:
            raise ValueError("No valid images were loaded")
            
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.labels = np.array(self.labels)
        
        print(f"Successfully processed {len(self.processed_images)} images")

    def _apply_smote_preprocessing(self):
        """Apply Borderline-SMOTE and random undersampling"""
        print("\nApplying SMOTE and undersampling...")
        print("Class distribution before resampling:")
        print(self.df['label'].value_counts())
        
        # Reshape images for SMOTE
        features = [img.reshape(-1) for img in self.processed_images]
        features = np.array(features)
        
        # Define resampling pipeline
        smote = BorderlineSMOTE(random_state=42)
        under = RandomUnderSampler(random_state=42)
        pipeline = Pipeline([('smote', smote), ('under', under)])
        
        # Apply resampling
        features_resampled, labels_resampled = pipeline.fit_resample(features, self.labels)
        
        # Reconstruct images
        self.images = [feat.reshape(self.preprocessor.target_size[0], 
                                  self.preprocessor.target_size[1], 3) 
                      for feat in features_resampled]
        
        # Create new balanced dataframe
        new_data = []
        for idx, label in enumerate(labels_resampled):
            new_data.append({
                'image_id': f'synthetic_{idx}' if idx >= len(self.df) else self.df.iloc[idx]['image_id'],
                'label': self.df['label'].unique()[label],
                'encoded_label': label,
                'is_synthetic': idx >= len(self.df)
            })
        
        self.df = pd.DataFrame(new_data)
        print("\nClass distribution after resampling:")
        print(self.df['label'].value_counts())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            if hasattr(self, 'images'):  # If SMOTE was applied
                image = self.images[idx]
                label = self.df.iloc[idx]['encoded_label']
            else:  # Original image loading
                image = self.processed_images[idx]
                label = self.labels[idx]
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            return torch.zeros((3, 224, 224)), torch.tensor(0)


def prepare_data(df: pd.DataFrame, image_dir: str, batch_size: int = 32):
    """Prepare data loaders with TIAToolbox preprocessing"""
    try:
        # Create label encodings
        label_encoder = {'HGSC': 0, 'EC': 1, 'CC': 2, 'LGSC': 3, 'MC': 4}
        df['encoded_label'] = df['label'].map(label_encoder)
        
        # Stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=42
        )
        
        # Create transforms
        train_transform, val_transform = create_transforms(image_size=(224, 224), stain_augment_prob=0.5 )
        
        print("Creating training dataset...")
        train_dataset = HistoDataset(
            df=train_df,
            image_dir=image_dir,
            transform=train_transform,
            is_training=True,
            apply_smote=True,
            stain_norm_method='reinhard',# try 'macenko' or 'reinhard' or 'ruifrok'
        )
        
        print("\nCreating validation dataset...")
        val_dataset = HistoDataset(
            df=val_df,
            image_dir=image_dir,
            transform=val_transform,
            is_training=False,
            apply_smote=False,
            stain_norm_method='reinhard',
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise



def show_sample_images(loader):
    """Display sample images from the data loader"""
    plt.figure(figsize=(15, 5))
    images, labels = next(iter(loader))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        img = images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("preprocessing.png")
    plt.show()


def test_stain_normalization():
    image = cv2.imread("/kaggle/input/UBC-OCEAN/train_thumbnails/10077_thumbnail.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    preprocessor = EnhancedPreprocessor(stain_norm_method='reinhard')
    normalized_image = preprocessor.apply_stain_normalization(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_image)
    plt.title("Normalized Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#if __name__ == "__main__":
    #test_stain_normalization()


def test_tissue_detection():
    image = cv2.imread("/kaggle/input/UBC-OCEAN/train_thumbnails/10896_thumbnail.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    preprocessor = EnhancedPreprocessor()
    tissue_mask = preprocessor.detect_tissue(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tissue_mask, cmap='gray')
    plt.title("Tissue Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#if __name__ == "__main__":
    #test_tissue_detection()


def test_data_augmentation():
    image = cv2.imread("/kaggle/input/UBC-OCEAN/train_thumbnails/12222_thumbnail.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    train_transform, _ = create_transforms()
    augmented = train_transform(image=image)['image']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.title("Augmented Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#if __name__ == "__main__":
    #test_data_augmentation()


def test_smote_undersampling():
    """Test SMOTE and undersampling with proper label encoding"""
    try:
        # Load data
        df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
        image_dir = '/kaggle/input/UBC-OCEAN/train_thumbnails'
        
        # Add label encoding
        label_encoder = {'HGSC': 0, 'EC': 1, 'CC': 2, 'LGSC': 3, 'MC': 4}
        df['encoded_label'] = df['label'].map(label_encoder)
        
        print("Initial class distribution:")
        print(df['label'].value_counts())
        
        # Create dataset with SMOTE
        dataset = HistoDataset(
            df=df,
            image_dir=image_dir,
            transform=None,
            is_training=True,
            apply_smote=True
        )
        
        print("\nFinal class distribution after SMOTE:")
        print(dataset.df['label'].value_counts())
        
    except Exception as e:
        print(f"Error in test_smote_undersampling: {str(e)}")


#if __name__ == "__main__":
    #test_smote_undersampling()



def main():
    """Main function to prepare and store data loaders in global scope"""
    try:
        # Load data
        df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
        image_dir = '/kaggle/input/UBC-OCEAN/train_thumbnails'
        
        print("Initial class distribution:")
        print(df['label'].value_counts())
        
        # Create data loaders with full pipeline and store in global scope
        global train_loader, val_loader
        train_loader, val_loader = prepare_data(df, image_dir)
        
        # Test batch loading
        images, labels = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Labels: {labels.shape}")
        
        print("\nClass distribution in batch:")
        print(pd.Series(labels.numpy()).value_counts())
        
        # Show sample images
        print("\nDisplaying sample processed images:")
        show_sample_images(train_loader)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        # Memory cleanup even if there's an error
        gc.collect()
        torch.cuda.empty_cache()

#if __name__ == "__main__":
    #main()


if __name__ == "__main__":
    # Test preprocessing components
    test_stain_normalization()
    test_tissue_detection()
    test_data_augmentation()

    main()


from tiatoolbox.models.architecture import vanilla
from torchvision import models, transforms
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, Optional
import timm
import PIL.Image
from typing import Optional, Union, Dict
from tiatoolbox.models.models_abc import ModelABC
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
from tiatoolbox.models.engine.patch_predictor import PatchPredictor, IOPatchPredictorConfig
import gc
import math
import traceback
import warnings
warnings.filterwarnings("ignore")


class HistoPathModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Initialize both backbones
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Remove original classifier layers
        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()
        
        # Get feature dimensions
        self.resnet_dim = 2048  # ResNet101's output dimension
        self.efficient_dim = 1536  # EfficientNet-B3's output dimension
        self.feature_dim = 512
        
        # Calculate combined features dimension for ResNet
        # Global features (2048) + 3 processors (512 each) = 3584
        self.combined_resnet_dim = self.resnet_dim + (512 * 3)
        
        # Feature reduction layers
        self.resnet_reducer = nn.Sequential(
            nn.Linear(self.combined_resnet_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.efficient_reducer = nn.Sequential(
            nn.Linear(self.efficient_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )
        
        # Modify first conv layer for histology images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        # Add channel attention for EfficientNet
        self.efficient_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1536, 512, kernel_size=1),  # 1536 for EfficientNet-B3
            nn.ReLU(),
            nn.Conv2d(512, 1536, kernel_size=1),
            nn.Sigmoid()
        )

        # Add fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )

                
        # Feature processors for ResNet features
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 5, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout for stability
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )
    
  
        # Add feature gates
        self.feature_gates = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2),
            nn.Softmax(dim=1)
        )

    
        # Add a squeeze-excitation block
        self.se_block = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
         )
    
        
        # Self-attention for global context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Main classifier
        self.main_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_resnet_features(self, x):
        # Initial layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # ResNet blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # Shape: [B, 2048, H, W]
        
        # Apply attention
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        attended_features = x * spatial_weights * channel_weights
        
        # Global features
        global_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        # Process features through each processor
        processed_features = []
        for processor in self.feature_processors:
            features = processor(attended_features)
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            processed_features.append(pooled)
        
        # Concatenate global and processed features
        combined_features = torch.cat([global_features] + processed_features, dim=1)
        
        # Reduce features
        return self.resnet_reducer(combined_features)

    def extract_efficient_features(self, x):
        features = self.efficientnet.features(x)
    
        # Apply channel attention
        channel_weights = self.efficient_channel_attention(features)
        features = features * channel_weights
    
        features = self.efficientnet.avgpool(features)
        features = torch.flatten(features, 1)
        return self.efficient_reducer(features)
    

    def extract_features(self, x):
        resnet_features = self.extract_resnet_features(x)
        torch.cuda.empty_cache()
    
        efficient_features = self.extract_efficient_features(x)
        torch.cuda.empty_cache()
    
        combined_features = torch.cat([resnet_features, efficient_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
    
        # Residual connection
        if hasattr(self, 'feature_gates'):
            gates = self.feature_gates(combined_features)
            residual = resnet_features * gates[:, 0].unsqueeze(1) + efficient_features * gates[:, 1].unsqueeze(1)
            fused_features = fused_features + residual
    
        
        # Apply self-attention with gradient clipping
        attended_features, _ = self.self_attention(
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1)
        )
        
        return fused_features + 0.1 * attended_features.squeeze(1)
        
    
    def forward(self, x):
        # Extract combined features
        features = self.extract_features(x)
        
        if self.training:
            # During training, return both main and auxiliary outputs
            main_logits = self.main_classifier(features)
            aux_logits = self.aux_classifier(features)
            return main_logits, aux_logits
        else:
            # During inference, only return main classifier output
            return self.main_classifier(features)

    def __del__(self):
        # Clean up CUDA memory
        torch.cuda.empty_cache()


def create_model(device, learning_rate=5e-4):
    """Create model and associated components."""
    # Create enhanced model
    model = HistoPathModel(num_classes=5)
    model = model.to(device)
    
    # Create predictor configuration
    wsi_config = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        stride_shape=[224, 224]
    )
    
    # Create patch predictor
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        num_loader_workers=4
    )
    
    # Loss function with class weights
    #class_weights = torch.tensor([1.0, 1.8, 2.2, 4.7, 4.8]).to(device)
    class_weights = torch.tensor([1.2, 2.0, 2.4, 4.8, 4.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=28,
        epochs=50,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    return model, criterion, optimizer, scheduler, predictor, wsi_config


def main():
    """Main function to test model architecture"""
    try:
        print("Testing Model Architecture...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model and get all components
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_model(
            device, learning_rate=5e-4)
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\nModel Parameters:")
        print(f"Total: {total_params:,}")
        print(f"Trainable: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        print(f"\nInput shape: {x.shape}")
        
        # Set model to eval mode for testing
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            if isinstance(outputs, tuple):
                main_logits = outputs[0]  # Get main classifier output
                probs = F.softmax(main_logits, dim=1)
                print(f"Output shape: {main_logits.shape}")
                
                # Print mean probabilities for each class
                mean_probs = probs.mean(dim=0)
                print("\nMean class probabilities:")
                for i, p in enumerate(mean_probs):
                    print(f"Class {i}: {p:.4f}")
            else:
                print(f"Output shape: {outputs.shape}")
        
        print("\nChecking PatchPredictor configuration:")
        print(f"Batch size: {predictor.batch_size}")
        print(f"WSI config input shape: {wsi_config.patch_input_shape}")
        
        print("\nModel architecture test completed successfully!")
        
        # Clean up
        del model, predictor
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in model test: {str(e)}")
        traceback.print_exc()
        
        # Clean up even if there's an error
        try:
            del model, predictor
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import gc
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class MetricTracker:
    """Track training metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'loss': [],
            'acc': [],
            'balanced_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'best_val_acc': 0.0,
            'best_val_balanced_acc': 0.0
        }
    
    def update(self, phase: str, loss: float, acc: float, balanced_acc: float):
        if phase == 'train':
            self.metrics['loss'].append(loss)
            self.metrics['acc'].append(acc)
            self.metrics['balanced_acc'].append(balanced_acc)
        else:
            self.metrics['val_loss'].append(loss)
            self.metrics['val_acc'].append(acc)
            self.metrics['val_balanced_acc'].append(balanced_acc)
            if acc > self.metrics['best_val_acc']:
                self.metrics['best_val_acc'] = acc
            if balanced_acc > self.metrics['best_val_balanced_acc']:
                self.metrics['best_val_balanced_acc'] = balanced_acc
    
    def get_best_metrics(self) -> Dict:
        return {
            'best_val_acc': self.metrics['best_val_acc'],
            'best_val_balanced_acc': self.metrics['best_val_balanced_acc'],
            'best_epoch_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else float('inf'),
            'best_epoch_val_acc': max(self.metrics['val_acc']) if self.metrics['val_acc'] else 0,
            'best_epoch_val_balanced_acc': max(self.metrics['val_balanced_acc']) if self.metrics['val_balanced_acc'] else 0
        }


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, criterion,
                 optimizer, scheduler, predictor, wsi_config, epochs=50,
                 save_dir='./model_checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor
        self.wsi_config = wsi_config
        self.epochs = epochs
        self.save_dir = save_dir
        self.tracker = MetricTracker()
        
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_loss(self, main_logits, aux_logits, labels):
        """Compute combined loss from main and auxiliary outputs"""
        # Main classification loss
        main_loss = self.criterion(main_logits, labels)
        
        # Auxiliary classification loss
        aux_loss = self.criterion(aux_logits, labels)
        
        # Combined loss with weighting
        total_loss = main_loss + 0.3 * aux_loss
        
        return total_loss, main_loss, aux_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Get both main and auxiliary outputs
            main_logits, aux_logits = self.model(inputs)
            
            # Compute combined loss
            loss, main_loss, aux_loss = self.compute_loss(main_logits, aux_logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = main_logits.max(1)  # Use main classifier predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)
        return running_loss / len(self.train_loader), 100. * correct / total, balanced_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc='Validating')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # During validation, model only returns main classifier output
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)
        
        class_names = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=3,
            output_dict=True
        )
        
        return running_loss / len(self.val_loader), 100. * correct / total, balanced_acc, report

    def train(self) -> Dict:
        print(f"\nStarting training for {self.epochs} epochs...")
        best_val_balanced_acc = 0.0
    
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            print('-' * 20)
        
            # Get training metrics (now includes main and auxiliary losses)
            train_loss, train_acc, train_balanced_acc = self.train_epoch()
            self.tracker.update('train', train_loss, train_acc, train_balanced_acc)
        
            # Validation phase remains the same
            val_loss, val_acc, val_balanced_acc, report = self.validate()
            self.tracker.update('val', val_loss, val_acc, val_balanced_acc)
        
            # Print detailed training metrics
            print(f'\nTraining Results:')
            print(f'Total Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Balanced Acc: {train_balanced_acc:.2f}%')
        
            # Print validation metrics
            print(f'\nValidation Results:')
            print(f'Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Balanced Acc: {val_balanced_acc:.2f}%')
        
            # Print class-wise performance
            print('\nClass-wise Performance:')
            for cls_name in ['HGSC', 'EC', 'CC', 'LGSC', 'MC']:
                cls_metrics = report[cls_name]
                print(f'{cls_name} - Precision: {cls_metrics["precision"]:.3f}, '
                      f'Recall: {cls_metrics["recall"]:.3f}, '
                      f'F1: {cls_metrics["f1-score"]:.3f}')
            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                model_path = os.path.join(self.save_dir, 'best_model.pth')
            
                # Save model with additional metrics
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_balanced_acc': train_balanced_acc,
                    'train_loss': train_loss,
                 }, model_path)
                print(f'Saved new best model with validation balanced accuracy: {val_balanced_acc:.2f}%')
            
            # Memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        # Return final metrics
        best_metrics = self.tracker.get_best_metrics()
    
        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best validation balanced accuracy: {best_metrics['best_val_balanced_acc']:.2f}%")
        print(f"Best epoch validation loss: {best_metrics['best_epoch_val_loss']:.4f}")
    
        return best_metrics


def train_main():
    try:
        print("Initializing Training Pipeline...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Use global data loaders
        global train_loader, val_loader
        
        print("\nInitializing model...")
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_model(
            device, 
            learning_rate=5e-4,
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer with new components
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            predictor=predictor,
            wsi_config=wsi_config,
            epochs=50,  
            save_dir='./model_checkpoints'
        )
        
        # Start training
        print("\nStarting training process...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        best_metrics = trainer.train()
        
        # Print final results
        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best validation balanced accuracy: {best_metrics['best_val_balanced_acc']:.2f}%")
        print(f"Best epoch validation loss: {best_metrics['best_epoch_val_loss']:.4f}")
        
        # Additional metrics reporting
        print("\nDetailed metrics:")
        print(f"Best epoch validation accuracy: {best_metrics['best_epoch_val_acc']:.2f}%")
        print(f"Best epoch validation balanced accuracy: {best_metrics['best_epoch_val_balanced_acc']:.2f}%")
        
        # Clean up
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_metrics
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        
        # Clean up even if there's an error
        try:
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        return None

if __name__ == "__main__":
    train_main()


!pip install -q umap-learn


import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import torch
import gc


def extract_features(model, dataloader, device):
    """Extract features from the model's intermediate layer"""
    features = []
    positions = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            images = images.to(device)
            # Use model's extract_features method directly
            batch_features = model.extract_features(images)
            
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
            positions.append(np.array([(i * images.shape[0] + j, j) for j in range(images.shape[0])]))
    
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    positions = np.concatenate(positions)
    return features, labels, positions


def visualize_sample_predictions(model, val_loader, device, num_samples=10):
    """Display sample images with their predictions"""
    model.eval()
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC'] 
    
    # Get a batch of images
    images, labels = next(iter(val_loader))
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        if isinstance(outputs, tuple):  # Handle training mode output
            outputs = outputs[0]
        _, preds = torch.max(outputs, 1)
    
    # Create a figure to display images
    fig = plt.figure(figsize=(20, 4))
    for idx in range(min(num_samples, len(images))):
        ax = fig.add_subplot(1, num_samples, idx + 1, xticks=[], yticks=[])
        
        # Convert tensor to image
        img = images[idx].numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        
        # Add title with true and predicted labels
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx].cpu()]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(features, labels):
    """Plot the distribution of features across classes"""
    # Reduce dimensionality to 2D using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(features)
    
    # Create scatter plot with different colors for each class
    plt.figure(figsize=(12, 8))
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, cls in enumerate(classes):
        mask = labels == i
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[i], label=cls, alpha=0.6)
    
    plt.title('Feature Distribution Across Classes')
    plt.legend()
    plt.show()


def plot_feature_space(features, labels, title="Feature Space Visualization"):
    """Create UMAP visualization of feature space"""
    # Reduce dimensionality to 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    plt.colorbar(scatter, label='True Class')
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def visualize_predictions(model, dataloader, device):
    """Visualize model predictions vs actual labels with enhanced metrics"""
    predictions = []
    actuals = []
    probabilities = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle training mode output
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.numpy())
            probabilities.extend(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    probabilities = np.array(probabilities)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(actuals, predictions) * 100
    
    # Create confusion matrix plot
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(12, 8))
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix\nBalanced Accuracy: {balanced_acc:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, target_names=classes))
    
    # Plot per-class prediction confidence
    plt.figure(figsize=(12, 6))
    
    for i, cls in enumerate(classes):
        true_mask = actuals == i
        pred_mask = predictions == i
        
        # Correct predictions
        correct_mask = np.logical_and(true_mask, pred_mask)
        if np.any(correct_mask):
            plt.scatter(np.full(np.sum(correct_mask), i+0.1),
                       probabilities[correct_mask, i],
                       c='green', alpha=0.5, label='Correct' if i == 0 else '')
        
        # Wrong predictions
        wrong_mask = np.logical_and(true_mask, ~pred_mask)
        if np.any(wrong_mask):
            plt.scatter(np.full(np.sum(wrong_mask), i-0.1),
                       probabilities[wrong_mask, i],
                       c='red', alpha=0.5, label='Wrong' if i == 0 else '')
    
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylabel('Prediction Confidence')
    plt.title('Per-class Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return balanced_acc, cm, predictions, actuals, probabilities


def analyze_model(model, train_loader, val_loader, device):
    """Run comprehensive model analysis with enhanced visualizations"""
    print("1. Displaying sample predictions...")
    visualize_sample_predictions(model, val_loader, device)
    
    print("\n2. Extracting features...")
    train_features, train_labels, _ = extract_features(model, train_loader, device)
    val_features, val_labels, _ = extract_features(model, val_loader, device)
    
    print("\n3. Plotting feature distributions...")
    plot_feature_distribution(train_features, train_labels)
    
    print("\n4. Analyzing predictions and metrics...")
    balanced_acc, cm, predictions, actuals, probs = visualize_predictions(model, val_loader, device)
    
    print(f"\nOverall Balanced Accuracy: {balanced_acc:.2f}%")
    
    print("\n5. Plotting UMAP embeddings...")
    plot_feature_space(train_features, train_labels, "Training Data Feature Space")
    plot_feature_space(val_features, val_labels, "Validation Data Feature Space")
    
    # Additional per-class analysis
    print("\nPer-class Performance Summary:")
    for i, cls in enumerate(['HGSC', 'EC', 'CC', 'LGSC', 'MC']):
        class_mask = actuals == i
        class_acc = balanced_accuracy_score([1 if x == i else 0 for x in actuals],
                                          [1 if x == i else 0 for x in predictions]) * 100
        class_conf = probs[class_mask, i].mean() * 100
        print(f"{cls}:")
        print(f" Balanced Accuracy: {class_acc:.2f}%")
        print(f" Average Confidence: {class_conf:.2f}%")


def run_analysis():
    """Run the complete analysis pipeline"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load your best model
        model = HistoPathModel()
        checkpoint = torch.load('./model_checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print("\nStarting model analysis...")
        print(f"Model checkpoint metrics:")
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Validation Balanced Accuracy: {checkpoint['val_balanced_acc']:.2f}%")
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
        
        analyze_model(model, train_loader, val_loader, device)
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    run_analysis()


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet101, ResNet101_Weights


class OutlierHistoPathModel(nn.Module):
    def __init__(self, num_classes=5, feature_dim=512):
        super().__init__()
        # Base backbones (same as before)
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Remove original classifier layers
        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()
        
        # Feature dimensions
        self.resnet_dim = 2048
        self.efficient_dim = 1536
        self.feature_dim = feature_dim
        self.combined_resnet_dim = self.resnet_dim + (512 * 3)
        
        # Feature reduction layers (same as before)
        self.resnet_reducer = nn.Sequential(
            nn.Linear(self.combined_resnet_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.efficient_reducer = nn.Sequential(
            nn.Linear(self.efficient_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
         )
        
        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Attention modules (same as before)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        # Add channel attention for EfficientNet
        self.efficient_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1536, 512, kernel_size=1),  # 1536 for EfficientNet-B3
            nn.ReLU(),
            nn.Conv2d(512, 1536, kernel_size=1),
            nn.Sigmoid()
        )

        # Add fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )

        
        # Feature processors (same as before)
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 5, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout for stability
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )
    
  
        # Add feature gates
        self.feature_gates = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2),
            nn.Softmax(dim=1)
        )

    
        # Add a squeeze-excitation block
        self.se_block = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
         )
    
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Outlier detection head
        self.outlier_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim * 2)  # Mean and log variance
        )

    def extract_resnet_features(self, x):
        # Initial layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # ResNet blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Apply attention
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        attended_features = x * spatial_weights * channel_weights
        
        # Global features
        global_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        # Process features through each processor
        processed_features = []
        for processor in self.feature_processors:
            features = processor(attended_features)
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            processed_features.append(pooled)
        
        # Concatenate global and processed features
        combined_features = torch.cat([global_features] + processed_features, dim=1)
        
        return self.resnet_reducer(combined_features)

    def extract_efficient_features(self, x):
        features = self.efficientnet.features(x)
    
        # Apply channel attention
        channel_weights = self.efficient_channel_attention(features)
        features = features * channel_weights
    
        features = self.efficientnet.avgpool(features)
        features = torch.flatten(features, 1)
        return self.efficient_reducer(features)
    

    def extract_features(self, x):
        resnet_features = self.extract_resnet_features(x)
        torch.cuda.empty_cache()
    
        efficient_features = self.extract_efficient_features(x)
        torch.cuda.empty_cache()
    
        combined_features = torch.cat([resnet_features, efficient_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
    
        # Residual connection
        if hasattr(self, 'feature_gates'):
            gates = self.feature_gates(combined_features)
            residual = resnet_features * gates[:, 0].unsqueeze(1) + efficient_features * gates[:, 1].unsqueeze(1)
            fused_features = fused_features + residual
    
        
        # Apply self-attention with gradient clipping
        attended_features, _ = self.self_attention(
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1)
        )
        
        return fused_features + 0.1 * attended_features.squeeze(1)

    def compute_outlier_score(self, x):
        """Compute outlier score for input images"""
        with torch.no_grad():
            # Extract features
            features = self.extract_features(x)
            
            # Get distribution parameters
            dist_params = self.outlier_detector(features)
            mean, log_var = torch.chunk(dist_params, 2, dim=1)
            
            # Compute Mahalanobis distance as outlier score
            var = torch.exp(log_var)
            z_score = (features - mean) / torch.sqrt(var + 1e-6)
            outlier_score = torch.sum(z_score ** 2, dim=1)
            
            return outlier_score

    def forward(self, x):
        """Forward pass with both classification and outlier detection"""
        # Extract features
        features = self.extract_features(x)
        
        # Classification logits
        logits = self.classifier(features)
        
        # Outlier detection parameters
        dist_params = self.outlier_detector(features)
        mean, log_var = torch.chunk(dist_params, 2, dim=1)
        
        if self.training:
            return logits, mean, log_var
        else:
            return logits

    def __del__(self):
        torch.cuda.empty_cache()


class OutlierLoss(nn.Module):
    """Combined loss for classification and outlier detection"""
    def __init__(self, num_classes=5, outlier_weight=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.outlier_weight = outlier_weight
        
    def forward(self, logits, mean, log_var, labels):
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Feature distribution regularization
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Combined loss
        total_loss = ce_loss + self.outlier_weight * kl_loss
        
        return total_loss, ce_loss, kl_loss


def create_outlier_model(device, learning_rate=5e-4, epochs=30):
    """Create model with outlier detection capabilities"""
    model = OutlierHistoPathModel(num_classes=5)
    model = model.to(device)
    
    # Create predictor configuration
    wsi_config = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        stride_shape=[224, 224]
    )
    
    # Create patch predictor
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        num_loader_workers=4
    )
    
    # Combined loss function
    criterion = OutlierLoss(num_classes=5)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=28,
        epochs=epochs,
        pct_start=0.3
    )
    
    return model, criterion, optimizer, scheduler, predictor, wsi_config


class OutlierTrainer:
    def __init__(self, model, train_loader, val_loader, device, criterion,
                 optimizer, scheduler, predictor, wsi_config, epochs=30,
                 save_dir='./outlier_model_checkpoints', batch_size=16):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor
        self.wsi_config = wsi_config
        self.epochs = epochs
        self.save_dir = save_dir
        self.batch_size = batch_size
        
        # Create gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        os.makedirs(save_dir, exist_ok=True)

    def compute_auxiliary_loss(self, features, labels):
        """Auxiliary task: Feature clustering loss"""
        # Process in chunks to save memory
        chunk_size = 4
        total_center_loss = 0
        
        for i in range(0, len(features), chunk_size):
            chunk_features = features[i:i + chunk_size]
            chunk_labels = labels[i:i + chunk_size]
            
            # Compute center for each class in chunk
            centers = {}
            for cls in torch.unique(chunk_labels):
                centers[cls.item()] = chunk_features[chunk_labels == cls].mean(0)
            
            # Compute center loss for chunk
            chunk_loss = 0
            for cls in centers:
                cls_features = chunk_features[chunk_labels == cls]
                if len(cls_features) > 0:
                    chunk_loss += F.mse_loss(cls_features, 
                                           centers[cls].expand(len(cls_features), -1))
            
            total_center_loss += chunk_loss
            
            # Clear cache
            torch.cuda.empty_cache()
        
        return total_center_loss
    
    def compute_consistency_loss(self, image, augmented_image):
        """Consistency between different views of same image"""
        # Process in chunks
        chunk_size = 4
        total_consist_loss = 0
        
        for i in range(0, len(image), chunk_size):
            # Extract features for original and augmented chunks
            with torch.cuda.amp.autocast():
                orig_features = self.model.extract_features(image[i:i + chunk_size])
                aug_features = self.model.extract_features(augmented_image[i:i + chunk_size])
                chunk_loss = F.mse_loss(orig_features, aug_features)
            
            total_consist_loss += chunk_loss
            
            # Clear cache
            torch.cuda.empty_cache()
        
        return total_consist_loss
    
    def augment_batch(self, images):
        """Memory efficient augmentation"""
        augmented = []
        chunk_size = 4
        
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            chunk_aug = []
            
            for img in chunk:
                transform = A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
                ])
                
                # Move to CPU for transformation
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                aug_img = transform(image=img_np)['image']
                chunk_aug.append(torch.from_numpy(aug_img.transpose(2, 0, 1)))
            
            # Move augmented chunk back to GPU
            chunk_tensor = torch.stack(chunk_aug).to(self.device)
            augmented.append(chunk_tensor)
            
            # Clear cache
            torch.cuda.empty_cache()
        
        return torch.cat(augmented, dim=0)

    def train_epoch(self):
        self.model.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kl_loss = 0.0
        running_aux_loss = 0.0
        running_consist_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            # Limit batch size
            if len(inputs) > self.batch_size:
                inputs = inputs[:self.batch_size]
                labels = labels[:self.batch_size]
            
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Clear cache before augmentation
            torch.cuda.empty_cache()
            
            # Get augmented version
            augmented_inputs = self.augment_batch(inputs)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                # Get model outputs
                logits, mean, log_var = self.model(inputs)
                features = self.model.extract_features(inputs)
                
                # Calculate losses
                total_loss, ce_loss, kl_loss = self.criterion(logits, mean, log_var, labels)
                aux_loss = self.compute_auxiliary_loss(features, labels)
                consist_loss = self.compute_consistency_loss(inputs, augmented_inputs)
                
                # Combined loss
                total_loss = total_loss + 0.1 * aux_loss + 0.1 * consist_loss
            
            # Scaled backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()
            running_aux_loss += aux_loss.item()
            running_consist_loss += consist_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'total_loss': running_total_loss / len(self.train_loader),
            'ce_loss': running_ce_loss / len(self.train_loader),
            'kl_loss': running_kl_loss / len(self.train_loader),
            'aux_loss': running_aux_loss / len(self.train_loader),
            'consist_loss': running_consist_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kl_loss = 0.0
        correct = 0
        total = 0
        
        # For outlier detection metrics
        all_scores = []
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc='Validating')
        for inputs, labels in pbar:
            # Limit batch size
            if len(inputs) > self.batch_size:
                inputs = inputs[:self.batch_size]
                labels = labels[:self.batch_size]
            
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Use mixed precision for validation too
            with torch.cuda.amp.autocast():
                # Get model outputs (model returns tuple in training mode)
                self.model.train()  # Temporarily set to train mode to get all outputs
                logits, mean, log_var = self.model(inputs)
                self.model.eval()  # Set back to eval mode
                
                # Calculate losses
                total_loss, ce_loss, kl_loss = self.criterion(logits, mean, log_var, labels)
                
                # Calculate outlier scores
                outlier_scores = self.model.compute_outlier_score(inputs)
            
            # Update metrics
            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and scores
            all_scores.extend(outlier_scores.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Clear cache
            torch.cuda.empty_cache()
            
            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate average metrics
        avg_total_loss = running_total_loss / len(self.val_loader)
        avg_ce_loss = running_ce_loss / len(self.val_loader)
        avg_kl_loss = running_kl_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        outlier_metrics = {
            'scores': np.array(all_scores),
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
        
        return avg_total_loss, avg_ce_loss, avg_kl_loss, accuracy, outlier_metrics

    def train(self):
        print(f"\nStarting outlier detection training for {self.epochs} epochs...")
        best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            print('-' * 20)
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Clear cache before validation
            torch.cuda.empty_cache()
            
            # Validation phase
            val_total_loss, val_ce_loss, val_kl_loss, val_acc, outlier_metrics = self.validate()
            
            # Print epoch results
            print(f'\nTraining Results:')
            print(f"Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"CE Loss: {train_metrics['ce_loss']:.4f}")
            print(f"KL Loss: {train_metrics['kl_loss']:.4f}")
            print(f"Auxiliary Loss: {train_metrics['aux_loss']:.4f}")
            print(f"Consistency Loss: {train_metrics['consist_loss']:.4f}")
            print(f"Accuracy: {train_metrics['accuracy']:.2f}%")
            
            print(f'\nValidation Results:')
            print(f'Total Loss: {val_total_loss:.4f}')
            print(f'CE Loss: {val_ce_loss:.4f}')
            print(f'KL Loss: {val_kl_loss:.4f}')
            print(f'Accuracy: {val_acc:.2f}%')
            
            # Save best model
            if val_total_loss < best_metrics['best_val_loss']:
                best_metrics['best_val_loss'] = val_total_loss
                best_metrics['best_val_acc'] = val_acc
                best_metrics['best_epoch'] = epoch + 1
                
                model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'val_loss': val_total_loss,
                    'val_acc': val_acc,
                    'outlier_metrics': outlier_metrics
                }, model_path)
                print(f'Saved new best model with validation loss: {val_total_loss:.4f}')
                print(f'Saved new best model with validation accuracy: {val_acc:.2f}')
            
            # Memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        return best_metrics


def train_outlier_main():
    try:
        print("Initializing Outlier Detection Pipeline...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Use global data loaders
        global train_loader, val_loader
        
        print("\nInitializing model with outlier detection...")
        # Create model with outlier detection capabilities
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_outlier_model(
            device, 
            learning_rate=5e-4,  
            epochs=30
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer with memory optimizations
        trainer = OutlierTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            predictor=predictor,
            wsi_config=wsi_config,
            epochs=30,
            save_dir='./outlier_model_checkpoints',
            batch_size=16  # Add reduced batch size for memory efficiency
        )
        
        # Start training
        print("\nStarting training process...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        best_metrics = trainer.train()
        
        # Print final results
        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation loss: {best_metrics['best_val_loss']:.4f}")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best epoch: {best_metrics['best_epoch']}")
        
        # Clean up
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_metrics
        
    except Exception as e:
        print(f"Error in outlier detection pipeline: {str(e)}")
        traceback.print_exc()
        
        # Clean up even if there's an error
        try:
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        return None

if __name__ == "__main__":
    train_outlier_main()


def analyze_outliers(model, dataloader, device, threshold=3.0):
    """Analyze potential outliers in the dataset"""
    model.eval()
    outlier_scores = []
    predictions = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            # Compute outlier scores directly
            scores = model.compute_outlier_score(images)
            # Get predictions
            logits = model(images)  # Model returns only logits in eval mode
            
            outlier_scores.extend(scores.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    outlier_scores = np.array(outlier_scores)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Identify outliers using Z-score
    z_scores = (outlier_scores - np.mean(outlier_scores)) / np.std(outlier_scores)
    outliers = z_scores > threshold
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Outlier scores distribution
    plt.subplot(1, 2, 1)
    plt.hist(z_scores, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Distribution of Outlier Scores')
    plt.xlabel('Z-score')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot 2: Scatter plot of outlier scores vs predictions
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(predictions, z_scores, c=labels, cmap='viridis', alpha=0.6)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Outlier Scores vs Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Outlier Score (Z-score)')
    plt.legend()
    plt.colorbar(scatter, label='True Class')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nFound {np.sum(outliers)} potential outliers out of {len(outlier_scores)} samples")
    print(f"Outlier percentage: {100 * np.sum(outliers) / len(outlier_scores):.2f}%")
    
    return outlier_scores, z_scores, outliers


def visualize_outliers(model, dataloader, device, threshold=3.0, num_samples=10):
    """Display sample images identified as outliers"""
    model.eval()
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
    
    # Collect images and their outlier scores
    all_images = []
    all_scores = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Get predictions
            logits = model(images)  # Model returns only logits in eval mode
            # Get outlier scores
            scores = model.compute_outlier_score(images)
            preds = torch.argmax(logits, dim=1)
            
            # Store batch data
            all_images.extend(images.cpu())
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate z-scores
    z_scores = (all_scores - np.mean(all_scores)) / np.std(all_scores)
    
    # Find outlier indices
    outlier_indices = np.where(z_scores > threshold)[0]
    
    if len(outlier_indices) == 0:
        print("No outliers found with the current threshold.")
        return
    
    # Sort outliers by score for most extreme cases
    sorted_indices = outlier_indices[np.argsort(-z_scores[outlier_indices])]
    
    # Display top outliers
    n_cols = 5
    n_rows = (min(num_samples, len(sorted_indices)) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4*n_rows))
    
    for idx, outlier_idx in enumerate(sorted_indices[:num_samples]):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, xticks=[], yticks=[])
        
        # Get image and convert from tensor
        img = all_images[outlier_idx].numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        
        # Add title with prediction and outlier score
        true_label = classes[all_labels[outlier_idx]]
        pred_label = classes[all_preds[outlier_idx]]
        score = z_scores[outlier_idx]
        
        title = f'True: {true_label}\nPred: {pred_label}\nOutlier Score: {score:.2f}'
        ax.set_title(title, color='red', fontsize=10)
    
    plt.suptitle(f'Top {num_samples} Outliers (Threshold = {threshold})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nTotal outliers found: {len(outlier_indices)} out of {len(z_scores)} images")
    print(f"Percentage of outliers: {100 * len(outlier_indices) / len(z_scores):.2f}%")
    
    # Show class distribution of outliers
    print("\nClass distribution of outliers:")
    for i, cls in enumerate(classes):
        outlier_count = np.sum(all_labels[outlier_indices] == i)
        total_count = np.sum(all_labels == i)
        if total_count > 0:
            percentage = 100 * outlier_count / total_count
            print(f"{cls}: {outlier_count}/{total_count} ({percentage:.2f}%)")


def run_outlier_analysis():
    """Run the complete outlier analysis pipeline"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load trained model
        model = OutlierHistoPathModel()
        model.load_state_dict(torch.load('./outlier_model_checkpoints/best_model.pth')['model_state_dict'])
        model.to(device)
        
        # Different thresholds for comparison
        thresholds = [2.0, 2.5, 3.0]
        results = {}
        
        for threshold in thresholds:
            print(f"\nAnalyzing outliers with threshold {threshold}...")
            outlier_scores, z_scores, outliers = analyze_outliers(
                model,
                val_loader,
                device,
                threshold=threshold
            )
            results[threshold] = {
                'scores': outlier_scores,
                'z_scores': z_scores,
                'outliers': outliers
            }
            
            # Visualize outliers for each threshold
            print(f"\nVisualizing outliers for threshold {threshold}...")
            visualize_outliers(model, val_loader, device, threshold=threshold)
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return results
    
    except Exception as e:
        print(f"Error in outlier analysis: {str(e)}")
        traceback.print_exc()
        return None

run_outlier_analysis()


test_df = pd.read_csv("/kaggle/input/UBC-OCEAN/test.csv")
test_df.head()


sample_df = pd.read_csv("/kaggle/input/UBC-OCEAN/sample_submission.csv")
sample_df.head()


import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from torch.nn import functional as F
from torchvision.models import ResNet101_Weights
import albumentations as A
import os


def predict_single_image(image_path, model_path='./model_checkpoints/best_model.pth', device='cuda'):
    """
    Predict class for a single test image using the saved best model
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize preprocessor
        preprocessor = EnhancedPreprocessor()
        preprocessed_image = preprocessor.preprocess_image(image)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = transform(preprocessed_image).unsqueeze(0)
        
        # Load the saved model
        model = HistoPathModel()  # Initialize model architecture
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
        model = model.to(device)
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
        pred_class = classes[torch.argmax(probabilities).item()]
        confidence = torch.max(probabilities).item()
        
        # Get probabilities for all classes
        class_probs = {cls: prob.item() for cls, prob in zip(classes, probabilities[0])}
        
        return pred_class, confidence, class_probs
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise


def save_prediction_to_csv(image_id, predicted_class, output_path='submission.csv'):
    """
    Save prediction to CSV in the required format
    """
    df = pd.DataFrame({
        'image_id': [image_id],
        'label': [predicted_class]
    })
    df.to_csv(output_path, index=False)
    print(f"\nPrediction saved to {output_path}")
    print(f"Content preview:")
    print(df.to_string(index=False))


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    image_path = '/kaggle/input/UBC-OCEAN/test_thumbnails/41_thumbnail.png'
    model_path = './model_checkpoints/best_model.pth'
    
    # Get image ID (removing leading zeros)
    image_id = str(int(os.path.basename(image_path).split('_')[0]))
    
    try:
        # Make prediction
        pred_class, confidence, class_probs = predict_single_image(
            image_path, 
            model_path, 
            device
        )
        
        # Print detailed results
        print(f"\nPredicted class: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nClass probabilities:")
        for cls, prob in class_probs.items():
            print(f"{cls}: {prob:.2%}")
        
        # Save to CSV
        save_prediction_to_csv(image_id, pred_class)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()




