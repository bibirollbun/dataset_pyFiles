import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm

# Set style for better visualizations
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set Kaggle paths
HOME = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
# Define paths for train and test data
train_dir = Path(f'{HOME}/train')
test_dir = Path(f'{HOME}/test')
labels_file = Path(f'{HOME}/train_labels.csv')

print("ğŸ“� Setup Complete!")


# Load training labels
train_labels = pd.read_csv(labels_file)

# Display basic information
print("ğŸ“� Training Labels Info:")
print(f"Total number of labeled images: {len(train_labels)}")
print(f"\nFirst few entries:")
display(train_labels.head())

# Check for missing values
print("\nğŸ”� Missing Values Check:")
display(train_labels.isnull().sum())


# Calculate class distribution
class_dist = train_labels['label'].value_counts()

# Create a bar plot using matplotlib
plt.figure(figsize=(12, 6))
bars = plt.bar(class_dist.index, class_dist.values)
plt.title('Distribution of Sheep Breeds', pad=20)
plt.xlabel('Breed')
plt.ylabel('Number of Images')

# Color the bars using a color map
colors = plt.cm.viridis(np.linspace(0, 1, len(class_dist)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print class distribution statistics
print("\nğŸ“Š Class Distribution Statistics:")
for breed, count in class_dist.items():
    print(f"{breed}: {count} images ({count/len(train_labels)*100:.2f}%)")


def load_and_analyze_image(img_path):
    """Load and return image dimensions and basic stats"""
    img = Image.open(img_path)
    return img.size, np.array(img).mean()

# Analyze a sample of images
sample_size = min(100, len(train_labels))
sample_images = train_labels['filename'].sample(sample_size)

# Collect image statistics
image_sizes = []
image_means = []

print("ğŸ”� Analyzing sample images...")
for img_file in tqdm(sample_images):
    img_path = train_dir / img_file
    if img_path.exists():
        size, mean_val = load_and_analyze_image(img_path)
        image_sizes.append(size)
        image_means.append(mean_val)

# Convert to DataFrame for analysis
img_stats = pd.DataFrame({
    'width': [s[0] for s in image_sizes],
    'height': [s[1] for s in image_sizes],
    'mean_pixel_value': image_means
})

# Display statistics
print("\nğŸ“Š Image Statistics:")
display(img_stats.describe())

# Plot image dimensions distribution using matplotlib
plt.figure(figsize=(10, 8))
scatter = plt.scatter(img_stats['width'], img_stats['height'], 
                     c=img_stats['mean_pixel_value'], 
                     cmap='viridis')
plt.colorbar(scatter, label='Mean Pixel Value')
plt.title('Image Dimensions Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.tight_layout()
plt.show()


# Function to display sample images
def display_sample_images(df, samples_per_breed=5):
    # Get unique breeds
    breeds = df['label'].unique()
    n_breeds = len(breeds)
    
    # Create a figure with subplots for each breed
    plt.figure(figsize=(20, 4*n_breeds))
    
    # For each breed
    for breed_idx, breed in enumerate(breeds):
        # Get 5 random samples for this breed
        breed_samples = df[df['label'] == breed].sample(min(samples_per_breed, len(df[df['label'] == breed])))
        
        # Display each sample
        for sample_idx, (_, row) in enumerate(breed_samples.iterrows(), 1):
            plt.subplot(n_breeds, samples_per_breed, breed_idx * samples_per_breed + sample_idx)
            img = Image.open(train_dir / row['filename'])
            plt.imshow(img)
            plt.title(f"{breed}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("ğŸ–¼ï¸� Sample Images from Each Breed (5 samples per breed):")
display_sample_images(train_labels, samples_per_breed=5)

