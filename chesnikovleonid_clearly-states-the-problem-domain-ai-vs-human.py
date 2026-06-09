!pip install -q -U exifread seaborn
!pip install -q -U torch


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from IPython.display import display
import exifread
import seaborn as sns

pd.set_option("display.max_columns", 30)
sns.set_style("whitegrid")

class Config:
    # Paths
    TRAIN_CSV = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
    TEST_CSV = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
    DATA_DIR = "/kaggle/input/ai-vs-human-generated-dataset"

    # System
    SEED = 42
    NUM_WORKERS = 4


def analyze_dataset_structure():
    # Load data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    train_df = train_df[['file_name', 'label']]
    train_df.columns = ['id', 'label']
    test_df = pd.read_csv(Config.TEST_CSV)

    print("=== Basic Dataset Structure ===")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples:  {len(test_df):,}")
    print(f"Columns in train: {list(train_df.columns)}")
    print(f"Columns in test:  {list(test_df.columns)}\n")
    
    # Missing values check
    print("=== Missing Values Analysis ===")
    print("Train set:")
    print(train_df.isnull().sum())
    print("\nTest set:")
    print(test_df.isnull().sum())
    
    # File existence verification
    def check_file_existence(df, sample_size=100):
        missing = []
        sample = df.sample(sample_size, random_state=Config.SEED)
        for fname in sample['id']:
            if not os.path.exists(os.path.join(Config.DATA_DIR, fname)):
                missing.append(fname)
        return missing
    
    print("\n=== File Existence Check ===")
    train_missing = check_file_existence(train_df)
    test_missing = check_file_existence(test_df)
    print(f"Missing train files: {len(train_missing)}/{len(train_df)}")
    print(f"Missing test files:  {len(test_missing)}/{len(test_df)}")
    
    return train_df, test_df

train_df, test_df = analyze_dataset_structure()


def analyze_labels(train_df):
    print("=== Label Distribution Analysis ===")
    train_df['pair_id'] = train_df.index // 2
    # Class balance
    label_counts = train_df["label"].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()
    
    # Pair pattern verification
    print("\n=== Pair Pattern Analysis ===")
    pair_violations = sum(train_df["label"].diff()[1::2] != -1)
    total_pairs = len(train_df) // 2
    print(f"Strict 1-0 pattern violations: {pair_violations}/{total_pairs}")
    
    # Consecutive labels analysis
    consecutive_same = sum(train_df["label"].diff() == 0)
    print(f"Total consecutive same labels: {consecutive_same}")
    
    # Pair completeness check
    pair_sizes = train_df.groupby("pair_id")["id"].count().value_counts()
    print("\nPair size distribution:")
    print(pair_sizes)
    
analyze_labels(train_df)


def analyze_image_metadata(train_df, sample_size=1000):
    print("=== Image Metadata Analysis ===")
    
    # Sampling
    sample = train_df.sample(sample_size, random_state=Config.SEED)
    metadata = []
    
    for _, row in tqdm(sample.iterrows(), total=sample_size, desc="Processing images"):
        img_path = os.path.join(Config.DATA_DIR, row['id'])
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                format_ = img.format
        except:
            width, height, mode, format_ = (None,)*4
            
        metadata.append({
            "width": width,
            "height": height,
            "channels": len(mode) if mode else None,
            "format": format_
        })
    
    meta_df = pd.DataFrame(metadata)
    
    # Dimension analysis
    print("\n=== Dimension Statistics ===")
    print(f"Average dimensions: {meta_df['width'].mean():.1f}x{meta_df['height'].mean():.1f}")
    print("Dimension distribution:")
    print(meta_df[['width', 'height']].describe())
    
    # Format distribution
    plt.figure(figsize=(10, 5))
    meta_df['format'].value_counts().plot(kind='bar')
    plt.title("Image Format Distribution")
    plt.show()
    
    # Channel analysis
    channel_dist = meta_df['channels'].value_counts()
    print("\nChannel distribution:")
    print(channel_dist)
    
    return meta_df

meta_df = analyze_image_metadata(train_df)


def visualize_image_samples(train_df, n_samples=5):
    print("=== Visual Pattern Analysis ===")
    
    def plot_comparison(label, title):
        plt.figure(figsize=(15, 3))
        samples = train_df[train_df['label'] == label].sample(n_samples, random_state=Config.SEED)
        for i, (_, row) in enumerate(samples.iterrows()):
            img_path = os.path.join(Config.DATA_DIR, row['id'])
            try:
                img = Image.open(img_path)
                plt.subplot(1, n_samples, i+1)
                plt.imshow(img)
                plt.title(f"Label {label}\n{img.size[0]}x{img.size[1]}")
                plt.axis('off')
            except:
                plt.title("Failed to load")
        plt.suptitle(title)
        plt.show()
    
    # Compare AI vs Human
    plot_comparison(1, "AI-Generated Samples")
    plot_comparison(0, "Human-Created Samples")

visualize_image_samples(train_df)


def analyze_pixel_statistics(train_df, sample_size=500):
    print("=== Pixel Statistics Analysis ===")
    
    sample = train_df.sample(sample_size, random_state=Config.SEED)
    stats = []
    
    for _, row in tqdm(sample.iterrows(), total=sample_size, desc="Analyzing pixels"):
        img_path = os.path.join(Config.DATA_DIR, row['id'])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # Basic statistics
                stats.append({
                    "label": row['label'],
                    "mean": img.mean(),
                    "std": img.std(),
                    "min": img.min(),
                    "max": img.max(),
                    "entropy": calculate_image_entropy(img)
                })
        except:
            continue
    
    stats_df = pd.DataFrame(stats)
    
    # Distribution plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['mean', 'std', 'entropy']):
        plt.subplot(2, 2, i+1)
        sns.kdeplot(data=stats_df, x=col, hue="label", fill=True)
        plt.title(f"{col.capitalize()} Distribution")
    plt.tight_layout()
    plt.show()
    
    # Grouped statistics
    print("\nLabel-wise Statistics:")
    print(stats_df.groupby("label").agg(['mean', 'std']))
    
    return stats_df

def calculate_image_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

pixel_stats = analyze_pixel_statistics(train_df)


def analyze_exif_metadata(train_df, sample_size=200):
    print("=== EXIF Metadata Analysis ===")
    
    sample = train_df.sample(sample_size, random_state=Config.SEED)
    exif_data = []
    
    for _, row in tqdm(sample.iterrows(), total=sample_size, desc="Reading EXIF"):
        img_path = os.path.join(Config.DATA_DIR, row['id'])
        try:
            with open(img_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                exif_data.append({
                    "label": row['label'],
                    "software": str(tags.get('Software', '')),
                    "camera": str(tags.get('Image Model', '')),
                    "datetime": str(tags.get('EXIF DateTimeOriginal', ''))
                })
        except:
            continue
    
    exif_df = pd.DataFrame(exif_data)
    
    # Software analysis
    print("\n=== Software Distribution ===")
    print(exif_df['software'].value_counts().head(10))
    
    # Camera analysis
    print("\n=== Camera Models ===")
    print(exif_df['camera'].value_counts().head(10))
    
    return exif_df

exif_df = analyze_exif_metadata(train_df)


def analyze_file_sizes(train_df):
    print("=== File Size Analysis ===")
    
    train_df['file_size'] = train_df['id'].apply(
        lambda x: os.path.getsize(os.path.join(Config.DATA_DIR, x)))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='label', y='file_size', data=train_df)
    plt.yscale('log')
    plt.title("File Size Distribution by Label")
    plt.ylabel("File Size (bytes, log scale)")
    plt.xlabel("Label")
    plt.show()
    
    print("\nFile Size Statistics:")
    print(train_df.groupby('label')['file_size'].describe())
    
analyze_file_sizes(train_df)


def check_contamination(train_df, test_df):
    print("=== Train-Test Contamination Check ===")
    
    train_files = set(train_df['id'])
    test_files = set(test_df['id'])
    overlap = train_files & test_files
    
    print(f"Overlapping files: {len(overlap)}")
    if overlap:
        print("Example overlapping files:", list(overlap)[:3])
    
check_contamination(train_df, test_df)


def analyze_color_spaces(train_df, sample_size=200):
    print("=== Color Space Analysis ===")
    
    sample = train_df.sample(sample_size, random_state=Config.SEED)
    color_stats = []
    
    for _, row in tqdm(sample.iterrows(), total=sample_size, desc="Processing"):
        img_path = os.path.join(Config.DATA_DIR, row['id'])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                color_stats.append({
                    "label": row['label'],
                    "hue_mean": hsv[:,:,0].mean(),
                    "saturation_mean": hsv[:,:,1].mean(),
                    "value_mean": hsv[:,:,2].mean()
                })
        except:
            continue
    
    color_df = pd.DataFrame(color_stats)
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    for i, channel in enumerate(['hue_mean', 'saturation_mean', 'value_mean']):
        plt.subplot(1, 3, i+1)
        sns.kdeplot(data=color_df, x=channel, hue='label', fill=True)
        plt.title(f"{channel.split('_')[0].capitalize()} Distribution")
    plt.tight_layout()
    plt.show()
    
    return color_df

color_df = analyze_color_spaces(train_df)


def advanced_correlation_analysis(train_df, meta_df, pixel_stats):
    print("=== Advanced Feature Correlation Analysis ===")
    
    # Create composite dataframe
    analysis_df = train_df[['label', 'pair_id']].copy()
    analysis_df = analysis_df.join(meta_df[['width', 'height']])
    analysis_df = analysis_df.join(pixel_stats[['mean', 'std', 'entropy']])
    analysis_df = analysis_df.dropna()    
    # Calculate correlation matrix
    corr_matrix = analysis_df.corr()
    # Visualize feature correlations
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Calculate mutual information with labels
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(
        analysis_df.drop(['label','pair_id'], axis=1),
        analysis_df['label'],
        random_state=Config.SEED
    )
    
    # Display feature importance
    mi_df = pd.DataFrame({
        'feature': analysis_df.columns.drop(['label','pair_id']),
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("\nMutual Information with Labels:")
    display(mi_df)
    
    # Investigate multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["feature"] = analysis_df.columns.drop(['label','pair_id'])
    vif_data["VIF"] = [variance_inflation_factor(
        analysis_df.drop(['label','pair_id'], axis=1).values, i) 
        for i in range(len(analysis_df.columns.drop(['label','pair_id'])))]
    print("\nMulticollinearity Analysis (VIF):")
    display(vif_data.sort_values('VIF', ascending=False))

advanced_correlation_analysis(train_df, meta_df, pixel_stats)

