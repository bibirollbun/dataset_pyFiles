import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import os
import cv2
from tqdm import tqdm

from scipy import stats

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

np.random.seed(42)
tf.random.set_seed(42)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')

print("="*80)
print("TRAINING DATA SHAPE")
print("="*80)
print(f"Train shape: {train_df.shape}")
print(f"\nTest shape: {test_df.shape}")

print("\n" + "="*80)
print("FIRST FEW ROWS OF TRAINING DATA")
print("="*80)
print(train_df.head(10))

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(train_df.dtypes)

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(train_df.isnull().sum())

print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)
print(train_df.describe())


print("="*80)
print("TARGET VARIABLE (PAWPULARITY) STATISTICS")
print("="*80)
print(f"Mean: {train_df['Pawpularity'].mean():.2f}")
print(f"Median: {train_df['Pawpularity'].median():.2f}")
print(f"Std: {train_df['Pawpularity'].std():.2f}")
print(f"Min: {train_df['Pawpularity'].min():.2f}")
print(f"Max: {train_df['Pawpularity'].max():.2f}")
print(f"Q25: {train_df['Pawpularity'].quantile(0.25):.2f}")
print(f"Q75: {train_df['Pawpularity'].quantile(0.75):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(train_df['Pawpularity'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(train_df['Pawpularity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_df["Pawpularity"].mean():.2f}')
axes[0].axvline(train_df['Pawpularity'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {train_df["Pawpularity"].median():.2f}')
axes[0].set_xlabel('Pawpularity Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Pawpularity Scores', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(train_df['Pawpularity'], vert=True)
axes[1].set_ylabel('Pawpularity Score', fontsize=12)
axes[1].set_title('Boxplot of Pawpularity Scores', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

train_df['Pawpularity'].plot(kind='kde', ax=axes[2], color='purple', linewidth=2)
axes[2].set_xlabel('Pawpularity Score', fontsize=12)
axes[2].set_ylabel('Density', fontsize=12)
axes[2].set_title('Density Plot of Pawpularity Scores', fontsize=14, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()



metadata_cols = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
                 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']

print("="*80)
print("METADATA FEATURES DISTRIBUTION")
print("="*80)
for col in metadata_cols:
    print(f"\n{col}:")
    print(train_df[col].value_counts().sort_index())
    print(f"  Percentage of 1s: {(train_df[col].sum() / len(train_df) * 100):.2f}%")


fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.ravel()

for idx, col in enumerate(metadata_cols):
    counts = train_df[col].value_counts().sort_index()
    axes[idx].bar(counts.index, counts.values, color=['salmon', 'lightgreen'], edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}\n({train_df[col].sum()} have feature)', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Value', fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].set_xticks([0, 1])
    axes[idx].grid(alpha=0.3)

    for i, v in enumerate(counts.values):
        axes[idx].text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()



print("="*80)
print("CORRELATION: METADATA FEATURES vs PAWPULARITY")
print("="*80)

correlations = {}
for col in metadata_cols:
    corr = train_df[col].corr(train_df['Pawpularity'])
    correlations[col] = corr
    print(f"{col:20s}: {corr:7.4f}")

sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n" + "="*80)
print("SORTED BY ABSOLUTE CORRELATION")
print("="*80)
for feature, corr in sorted_corr:
    print(f"{feature:20s}: {corr:7.4f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

features = [x[0] for x in sorted_corr]
corr_values = [x[1] for x in sorted_corr]
colors = ['green' if x > 0 else 'red' for x in corr_values]

axes[0].barh(features, corr_values, color=colors, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Correlation with Pawpularity', fontsize=12)
axes[0].set_title('Feature Correlations with Pawpularity', fontsize=14, fontweight='bold')
axes[0].axvline(0, color='black', linewidth=1)
axes[0].grid(alpha=0.3)

corr_matrix = train_df[metadata_cols + ['Pawpularity']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1])
axes[1].set_title('Correlation Heatmap: All Features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.ravel()

for idx, col in enumerate(metadata_cols):
    data_to_plot = [train_df[train_df[col] == 0]['Pawpularity'],
                    train_df[train_df[col] == 1]['Pawpularity']]

    bp = axes[idx].boxplot(data_to_plot, labels=['No (0)', 'Yes (1)'], patch_artist=True)

    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Pawpularity Score', fontsize=10)
    axes[idx].grid(alpha=0.3)

    mean_0 = train_df[train_df[col] == 0]['Pawpularity'].mean()
    mean_1 = train_df[train_df[col] == 1]['Pawpularity'].mean()
    axes[idx].text(1, mean_0, f'{mean_0:.1f}', ha='center', va='bottom', fontweight='bold', color='red')
    axes[idx].text(2, mean_1, f'{mean_1:.1f}', ha='center', va='bottom', fontweight='bold', color='green')

plt.suptitle('Pawpularity Distribution by Metadata Features', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()




print("="*80)
print("T-TEST: PAWPULARITY DIFFERENCE BY METADATA FEATURES")
print("="*80)

for col in metadata_cols:
    group_0 = train_df[train_df[col] == 0]['Pawpularity']
    group_1 = train_df[train_df[col] == 1]['Pawpularity']

    t_stat, p_value = stats.ttest_ind(group_0, group_1)

    mean_0 = group_0.mean()
    mean_1 = group_1.mean()
    diff = mean_1 - mean_0

    print(f"\n{col}:")
    print(f"  Mean (No):  {mean_0:.2f}")
    print(f"  Mean (Yes): {mean_1:.2f}")
    print(f"  Difference: {diff:+.2f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")


print("="*80)
print("FEATURE COMBINATIONS - TOP PATTERNS")
print("="*80)

train_df['feature_sum'] = train_df[metadata_cols].sum(axis=1)

print("\nNumber of active features distribution:")
print(train_df['feature_sum'].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

feature_counts = train_df['feature_sum'].value_counts().sort_index()
axes[0].bar(feature_counts.index, feature_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Active Features', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Active Feature Count', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].boxplot([train_df[train_df['feature_sum'] == i]['Pawpularity'] for i in sorted(train_df['feature_sum'].unique())],
                labels=sorted(train_df['feature_sum'].unique()))
axes[1].set_xlabel('Number of Active Features', fontsize=12)
axes[1].set_ylabel('Pawpularity Score', fontsize=12)
axes[1].set_title('Pawpularity by Number of Active Features', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

corr = train_df['feature_sum'].corr(train_df['Pawpularity'])
print(f"\nCorrelation between feature_sum and Pawpularity: {corr:.4f}")


print("="*80)
print("PAIRWISE FEATURE INTERACTIONS")
print("="*80)

top_features = ['Blur', 'Subject Focus', 'Eyes', 'Face']

for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        feat1, feat2 = top_features[i], top_features[j]

        print(f"\n{feat1} & {feat2}:")
        crosstab = pd.crosstab(train_df[feat1], train_df[feat2])
        print(crosstab)

        for val1 in [0, 1]:
            for val2 in [0, 1]:
                subset = train_df[(train_df[feat1] == val1) & (train_df[feat2] == val2)]
                if len(subset) > 0:
                    mean_paw = subset['Pawpularity'].mean()
                    print(f"  {feat1}={val1}, {feat2}={val2}: Mean Pawpularity = {mean_paw:.2f} (n={len(subset)})")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

idx = 0
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        if idx >= 6:
            break
        feat1, feat2 = top_features[i], top_features[j]

        grouped_data = []
        labels = []
        for val1 in [0, 1]:
            for val2 in [0, 1]:
                subset = train_df[(train_df[feat1] == val1) & (train_df[feat2] == val2)]
                if len(subset) > 10:
                    grouped_data.append(subset['Pawpularity'])
                    labels.append(f'{feat1}={val1}\n{feat2}={val2}')

        bp = axes[idx].boxplot(grouped_data, labels=labels, patch_artist=True)
        axes[idx].set_title(f'{feat1} × {feat2}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Pawpularity', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=8)
        axes[idx].grid(alpha=0.3)

        colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']
        for patch, color in zip(bp['boxes'], colors[:len(grouped_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        idx += 1

plt.tight_layout()
plt.show()


import random

print("="*80)
print("VISUALIZING SAMPLE IMAGES")
print("="*80)

low_paw = train_df.nsmallest(3, 'Pawpularity')
mid_paw = train_df.iloc[(len(train_df)//2 - 1):(len(train_df)//2 + 2)]
high_paw = train_df.nlargest(3, 'Pawpularity')

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

categories = [
    ("LOW PAWPULARITY", low_paw),
    ("MEDIUM PAWPULARITY", mid_paw),
    ("HIGH PAWPULARITY", high_paw)
]

for row_idx, (title, df_subset) in enumerate(categories):
    for col_idx, (_, row) in enumerate(df_subset.iterrows()):
        img_path = f'/kaggle/input/petfinder-pawpularity-score/train/{row["Id"]}.jpg'
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')

            metadata_str = ', '.join([col.replace('Subject ', '') for col in metadata_cols if row[col] == 1])
            if not metadata_str:
                metadata_str = "No special features"

            axes[row_idx, col_idx].set_title(
                f'Score: {row["Pawpularity"]:.1f}\n{metadata_str[:40]}...',
                fontsize=9,
                fontweight='bold'
            )

plt.suptitle('Sample Images by Pawpularity Score', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("="*80)
print("ANALYZING IMAGE DIMENSIONS")
print("="*80)

sample_size = min(1000, len(train_df))
sample_indices = np.random.choice(len(train_df), sample_size, replace=False)

image_dims = []
for idx in tqdm(sample_indices, desc="Loading images"):
    img_id = train_df.iloc[idx]['Id']
    img_path = f'/kaggle/input/petfinder-pawpularity-score/train/{img_id}.jpg'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w, c = img.shape
            image_dims.append({
                'Id': img_id,
                'height': h,
                'width': w,
                'channels': c,
                'aspect_ratio': w/h,
                'total_pixels': h*w,
                'Pawpularity': train_df.iloc[idx]['Pawpularity']
            })

dims_df = pd.DataFrame(image_dims)

print("\nImage Dimension Statistics:")
print(dims_df[['height', 'width', 'aspect_ratio', 'total_pixels']].describe())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(dims_df['height'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Height (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Image Height Distribution')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(dims_df['width'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Width (pixels)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Image Width Distribution')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].hist(dims_df['aspect_ratio'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Aspect Ratio (width/height)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Aspect Ratio Distribution')
axes[0, 2].grid(alpha=0.3)

axes[1, 0].scatter(dims_df['width'], dims_df['height'], alpha=0.5, s=10, c='purple')
axes[1, 0].set_xlabel('Width (pixels)')
axes[1, 0].set_ylabel('Height (pixels)')
axes[1, 0].set_title('Width vs Height')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(dims_df['total_pixels'], dims_df['Pawpularity'], alpha=0.5, s=10, c='orange')
axes[1, 1].set_xlabel('Total Pixels')
axes[1, 1].set_ylabel('Pawpularity')
axes[1, 1].set_title('Image Size vs Pawpularity')
axes[1, 1].grid(alpha=0.3)

axes[1, 2].scatter(dims_df['aspect_ratio'], dims_df['Pawpularity'], alpha=0.5, s=10, c='teal')
axes[1, 2].set_xlabel('Aspect Ratio')
axes[1, 2].set_ylabel('Pawpularity')
axes[1, 2].set_title('Aspect Ratio vs Pawpularity')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()


print("="*80)
print("ANALYZING IMAGE COLORS")
print("="*80)

color_stats = []
for idx in tqdm(sample_indices[:500], desc="Analyzing colors"):
    img_id = train_df.iloc[idx]['Id']
    img_path = f'/kaggle/input/petfinder-pawpularity-score/train/{img_id}.jpg'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            color_stats.append({
                'Id': img_id,
                'mean_r': img_rgb[:,:,0].mean(),
                'mean_g': img_rgb[:,:,1].mean(),
                'mean_b': img_rgb[:,:,2].mean(),
                'std_r': img_rgb[:,:,0].std(),
                'std_g': img_rgb[:,:,1].std(),
                'std_b': img_rgb[:,:,2].std(),
                'brightness': img_rgb.mean(),
                'Pawpularity': train_df.iloc[idx]['Pawpularity']
            })

color_df = pd.DataFrame(color_stats)

print("\nColor Statistics:")
print(color_df[['mean_r', 'mean_g', 'mean_b', 'brightness']].describe())

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].scatter(color_df['brightness'], color_df['Pawpularity'], alpha=0.5, c='gold', s=20)
axes[0, 0].set_xlabel('Image Brightness')
axes[0, 0].set_ylabel('Pawpularity')
axes[0, 0].set_title(f'Brightness vs Pawpularity\n(Corr: {color_df["brightness"].corr(color_df["Pawpularity"]):.3f})')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist([color_df['mean_r'], color_df['mean_g'], color_df['mean_b']],
                bins=30, label=['Red', 'Green', 'Blue'], alpha=0.6, color=['red', 'green', 'blue'])
axes[0, 1].set_xlabel('Mean Channel Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Color Channels')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(color_df['std_r'] + color_df['std_g'] + color_df['std_b'],
                   color_df['Pawpularity'], alpha=0.5, c='purple', s=20)
axes[1, 0].set_xlabel('Total Color Variance')
axes[1, 0].set_ylabel('Pawpularity')
axes[1, 0].set_title('Color Variance vs Pawpularity')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(color_df['mean_r'], color_df['mean_b'],
                   c=color_df['Pawpularity'], cmap='viridis', s=30, alpha=0.6)
axes[1, 1].set_xlabel('Mean Red')
axes[1, 1].set_ylabel('Mean Blue')
axes[1, 1].set_title('Red vs Blue (colored by Pawpularity)')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Pawpularity')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


print("="*80)
print("PREPARING DATA FOR MODELING")
print("="*80)

IMG_SIZE = 224
BATCH_SIZE = 32
USE_KFOLD = True
N_FOLDS = 5

train_df['image_path'] = train_df['Id'].apply(lambda x: f'/kaggle/input/petfinder-pawpularity-score/train/{x}.jpg')
test_df['image_path'] = test_df['Id'].apply(lambda x: f'/kaggle/input/petfinder-pawpularity-score/test/{x}.jpg')

train_df['Pawpularity_normalized'] = train_df['Pawpularity'] / 100.0

feature_cols = metadata_cols

X_test_meta = test_df[feature_cols].values
scaler_test = StandardScaler()
X_test_meta = scaler_test.fit_transform(X_test_meta)

if USE_KFOLD:
    print(f"\nUsing {N_FOLDS}-Fold Cross-Validation")
    print(f"Total training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    train_df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        train_df.iloc[val_idx, train_df.columns.get_loc('fold')] = fold_idx
    
    print("\nFold distribution:")
    print(train_df['fold'].value_counts().sort_index())
    
else:
    print("\nUsing Simple Train/Validation Split")
    
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_df)}")
    
    X_train_meta = train_data[feature_cols].values
    X_val_meta = val_data[feature_cols].values
    
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(X_train_meta)
    X_val_meta = scaler.transform(X_val_meta)
    X_test_meta = scaler.transform(test_df[feature_cols].values)
    
    y_train = train_data['Pawpularity_normalized'].values
    y_val = val_data['Pawpularity_normalized'].values
    
    print("\nMetadata features shape:")
    print(f"X_train_meta: {X_train_meta.shape}")
    print(f"X_val_meta: {X_val_meta.shape}")
    print(f"X_test_meta: {X_test_meta.shape}")
    
    print("\nTarget shape:")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    img = tf.image.resize(img, [img_size, img_size])
    
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def create_dataset(image_paths, metadata, labels=None, batch_size=BATCH_SIZE, shuffle=False, augment=False):
    
    def load_data_with_labels(image_path, meta, label):
        img = load_and_preprocess_image(image_path)
        
        if augment:
            img = tf.image.random_flip_left_right(img)  
            img = tf.image.random_brightness(img, 0.2)  
            img = tf.image.random_contrast(img, 0.8, 1.2) 
            img = tf.image.random_saturation(img, 0.8, 1.2) 
        
        return {'image_input': img, 'metadata_input': meta}, label
    
    def load_data_without_labels(image_path, meta):
        img = load_and_preprocess_image(image_path)
        return {'image_input': img, 'metadata_input': meta}
    
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, metadata, labels))
        dataset = dataset.map(load_data_with_labels, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, metadata))
        dataset = dataset.map(load_data_without_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

print("="*80)
print("CREATING DATASETS")
print("="*80)

if not USE_KFOLD:
    train_dataset = create_dataset(
        train_data['image_path'].values,
        X_train_meta,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_dataset = create_dataset(
        val_data['image_path'].values,
        X_val_meta,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    test_dataset = create_dataset(
        test_df['image_path'].values,
        X_test_meta,
        labels=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    print("Training dataset created")
    print("Validation dataset created")
    print("Test dataset created")
else:
    print("Using K-Fold Cross-Validation")
    print("Datasets will be created within each fold during training")
    print(f"{N_FOLDS} folds configured")


print("="*80)
print("BUILDING MODEL")
print("="*80)

def build_resnet_model(img_size=IMG_SIZE, num_metadata_features=12, use_batch_norm=True, l2_reg=0.01):

    if l2_reg > 0:
        regularizer = keras.regularizers.l2(l2_reg)
    else:
        regularizer = None

    image_input = layers.Input(shape=(img_size, img_size, 3), name='image_input')

    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=image_input,
        pooling='avg'
    )

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x1 = base_model.output

    x1 = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizer,
                      name='image_dense1')(x1)
    if use_batch_norm:
        x1 = layers.BatchNormalization(name='image_bn1')(x1)
    x1 = layers.Dropout(0.3, name='image_dropout1')(x1)

    x1 = layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizer,
                      name='image_dense2')(x1)
    if use_batch_norm:
        x1 = layers.BatchNormalization(name='image_bn2')(x1)
    x1 = layers.Dropout(0.2, name='image_dropout2')(x1)

    metadata_input = layers.Input(shape=(num_metadata_features,), name='metadata_input')

    x2 = layers.Dense(64, activation='relu',
                      kernel_regularizer=regularizer,
                      name='metadata_dense1')(metadata_input)
    if use_batch_norm:
        x2 = layers.BatchNormalization(name='metadata_bn1')(x2)
    x2 = layers.Dropout(0.2, name='metadata_dropout1')(x2)

    x2 = layers.Dense(32, activation='relu',
                      kernel_regularizer=regularizer,
                      name='metadata_dense2')(x2)
    if use_batch_norm:
        x2 = layers.BatchNormalization(name='metadata_bn2')(x2)

    combined = layers.concatenate([x1, x2], name='concatenate')

    z = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizer,
                     name='combined_dense1')(combined)
    if use_batch_norm:
        z = layers.BatchNormalization(name='combined_bn1')(z)
    z = layers.Dropout(0.2, name='combined_dropout1')(z)

    z = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizer,
                     name='combined_dense2')(z)
    if use_batch_norm:
        z = layers.BatchNormalization(name='combined_bn2')(z)

    output = layers.Dense(1, activation='sigmoid', name='output')(z)

    model = keras.Model(inputs=[image_input, metadata_input], outputs=output, name='ResNet50_Pawpularity')

    return model

model = build_resnet_model(use_batch_norm=True, l2_reg=0.01)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
model.summary()

print("\n" + "="*80)
print("MODEL DETAILS")
print("="*80)
print(f"Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")

print("\n" + "="*80)
print("REGULARIZATION TECHNIQUES USED")
print("="*80)
print("Dropout: 20-30% at multiple layers")
print("Batch Normalization: After every dense layer")
print("L2 Regularization: λ=0.01 on all dense layer weights")
print("Early Stopping: Patience = 7 epochs")
print("Learning Rate Reduction: Factor = 0.5, Patience = 3")
print("Data Augmentation: Flip, brightness, contrast, saturation")
print("Transfer Learning: ResNet50 pretrained on ImageNet")
print("Partial Freezing: First 143 layers frozen, last 30 trainable")

print("\n" + "="*80)
print("REGULARIZATION CONFIGURATION")
print("="*80)
print("L2 Regularization Strength: 0.01")
print("  - Penalizes large weights to prevent overfitting")
print("  - Applied to all Dense layers (not BatchNorm or Dropout)")
print("  - Loss = MSE + 0.01 × (sum of squared weights)")
print("\nTo adjust regularization:")
print("  - Increase (0.01 → 0.1): Stronger regularization, may underfit")
print("  - Decrease (0.01 → 0.001): Weaker regularization, may overfit")
print("  - Disable (set l2_reg=0): No L2 penalty")


print("="*80)
print("VISUALIZING MODEL ARCHITECTURE")
print("="*80)

try:
    keras.utils.plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )

    from IPython.display import Image
    display(Image('model_architecture.png'))
except Exception as e:
    print(f"Could not plot model architecture: {e}")
    print("This may be due to missing dependencies like Graphviz.")


print("="*80)
print("SETTING UP CALLBACKS")
print("="*80)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_rmse',
    mode='min',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_rmse',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_rmse',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, early_stop]

print("ModelCheckpoint: Save best model based on validation RMSE")
print("ReduceLROnPlateau: Reduce learning rate when validation RMSE plateaus")
print("EarlyStopping: Stop training if no improvement for 7 epochs")

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("\n" + "="*80)
    print("GPU MEMORY CHECK")
    print("="*80)
    print("GPU is available and will be used for training")
    print("TensorFlow will automatically use GPU for operations")
    print("Monitor GPU usage in Kaggle's system resources panel →")


print("="*80)
print("TRAINING MODEL")
print("="*80)

EPOCHS = 50

if USE_KFOLD:
    print(f"\nTRAINING WITH {N_FOLDS}-FOLD CROSS-VALIDATION\n")

    fold_scores = []
    fold_histories = []
    oof_predictions = np.zeros(len(train_df))  
    test_predictions_all = []

    for fold in range(N_FOLDS):
        print("\n" + "="*80)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("="*80)

        train_idx = train_df[train_df['fold'] != fold].index
        val_idx = train_df[train_df['fold'] == fold].index

        train_fold = train_df.loc[train_idx]
        val_fold = train_df.loc[val_idx]

        print(f"Train samples: {len(train_fold)}")
        print(f"Val samples: {len(val_fold)}")

        X_train_meta = train_fold[feature_cols].values
        X_val_meta = val_fold[feature_cols].values

        scaler = StandardScaler()
        X_train_meta = scaler.fit_transform(X_train_meta)
        X_val_meta = scaler.transform(X_val_meta)

        y_train = train_fold['Pawpularity_normalized'].values
        y_val = val_fold['Pawpularity_normalized'].values

        train_dataset = create_dataset(
            train_fold['image_path'].values,
            X_train_meta,
            y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            augment=True
        )

        val_dataset = create_dataset(
            val_fold['image_path'].values,
            X_val_meta,
            y_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            augment=False
        )

        print(f"\nBuilding model for fold {fold + 1}...")
        model = build_resnet_model(use_batch_norm=True, l2_reg=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        checkpoint = ModelCheckpoint(
            f'best_model_fold{fold}.h5',
            monitor='val_rmse',
            mode='min',
            save_best_only=True,
            verbose=0
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_rmse',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0
        )

        early_stop = EarlyStopping(
            monitor='val_rmse',
            patience=7,
            restore_best_weights=True,
            verbose=0
        )

        callbacks = [checkpoint, reduce_lr, early_stop]

        print(f"Training fold {fold + 1}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0
        )

        fold_histories.append(history)

        val_results = model.evaluate(val_dataset, verbose=0)
        val_loss, val_mae, val_rmse = val_results
        fold_scores.append(val_rmse)

        print(f"\nFold {fold + 1} Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE:  {val_mae:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")

        val_preds = model.predict(val_dataset, verbose=0).flatten()
        oof_predictions[val_idx] = val_preds

        X_test_meta = test_df[feature_cols].values
        X_test_meta = scaler.transform(X_test_meta)

        test_dataset_fold = create_dataset(
            test_df['image_path'].values,
            X_test_meta,
            labels=None,
            batch_size=BATCH_SIZE,
            shuffle=False,
            augment=False
        )

        test_preds = model.predict(test_dataset_fold, verbose=0).flatten()
        test_predictions_all.append(test_preds)

        del model
        tf.keras.backend.clear_session()

    oof_rmse = np.sqrt(mean_squared_error(train_df['Pawpularity_normalized'].values, oof_predictions))
    oof_rmse_original = oof_rmse * 100

    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)

    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold + 1}: RMSE = {score:.4f} (normalized), {score * 100:.4f} (original scale)")

    print(f"\nMean RMSE: {np.mean(fold_scores):.4f} (normalized), {np.mean(fold_scores) * 100:.4f} (original scale)")
    print(f"Std RMSE:  {np.std(fold_scores):.4f}")
    print(f"\nOverall OOF RMSE: {oof_rmse:.4f} (normalized), {oof_rmse_original:.4f} (original scale)")

    test_predictions = np.mean(test_predictions_all, axis=0)
    test_predictions_denorm = test_predictions * 100
    test_predictions_denorm = np.clip(test_predictions_denorm, 0, 100)

    train_df['oof_predictions'] = oof_predictions

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(range(1, N_FOLDS + 1), [s * 100 for s in fold_scores],
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axhline(np.mean(fold_scores) * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(fold_scores) * 100:.2f}')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('RMSE (Original Scale)', fontsize=12)
    axes[0].set_title('RMSE by Fold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for fold, history in enumerate(fold_histories):
        axes[1].plot(history.history['val_rmse'], label=f'Fold {fold + 1}', alpha=0.7)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation RMSE', fontsize=12)
    axes[1].set_title('Validation RMSE Across All Folds', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    history = fold_histories[0]
    val_predictions_denorm = oof_predictions * 100
    y_val_denorm = train_df['Pawpularity'].values

else:
    print("\nTRAINING WITH SINGLE TRAIN/VAL SPLIT\n")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)




print("="*80)
print("TRAINING HISTORY")
print("="*80)

best_epoch = np.argmin(history.history['val_rmse']) + 1
best_val_rmse = np.min(history.history['val_rmse'])
best_train_rmse = history.history['rmse'][best_epoch - 1]

print(f"\nBest Epoch: {best_epoch}")
print(f"Best Validation RMSE: {best_val_rmse:.4f}")
print(f"Training RMSE at best epoch: {best_train_rmse:.4f}")
print(f"Validation Loss at best epoch: {history.history['val_loss'][best_epoch - 1]:.4f}")
print(f"Training Loss at best epoch: {history.history['loss'][best_epoch - 1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='red')
axes[0, 0].axvline(best_epoch - 1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Best Epoch')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(history.history['rmse'], label='Train RMSE', linewidth=2, color='blue')
axes[0, 1].plot(history.history['val_rmse'], label='Val RMSE', linewidth=2, color='red')
axes[0, 1].axvline(best_epoch - 1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Best Epoch')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('RMSE', fontsize=12)
axes[0, 1].set_title('Training and Validation RMSE', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(history.history['mae'], label='Train MAE', linewidth=2, color='blue')
axes[1, 0].plot(history.history['val_mae'], label='Val MAE', linewidth=2, color='red')
axes[1, 0].axvline(best_epoch - 1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Best Epoch')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('MAE', fontsize=12)
axes[1, 0].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Tracked',
                    ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

plt.tight_layout()
plt.show()



print("="*80)
print("MODEL EVALUATION")
print("="*80)

if USE_KFOLD:
    print("\nUsing Out-of-Fold (OOF) predictions for evaluation")
    print(f"Test predictions are ensemble of {N_FOLDS} models")
    
    val_predictions_denorm = train_df['oof_predictions'].values * 100
    y_val_denorm = train_df['Pawpularity'].values
    
    rmse = np.sqrt(mean_squared_error(y_val_denorm, val_predictions_denorm))
    mae = mean_absolute_error(y_val_denorm, val_predictions_denorm)
    r2 = r2_score(y_val_denorm, val_predictions_denorm)
    
    print("\n" + "="*80)
    print("OUT-OF-FOLD EVALUATION METRICS")
    print("="*80)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
else:
    print("\nLoading best model...")
    model = keras.models.load_model('best_model.h5', compile=False)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    print("Best model loaded and recompiled successfully")
    
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    
    val_results = model.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_results[0]:.4f}")
    print(f"Validation MAE: {val_results[1]:.4f}")
    print(f"Validation RMSE: {val_results[2]:.4f}")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("="*80)
print("VALIDATION PREDICTIONS ANALYSIS")
print("="*80)

if not USE_KFOLD:
    val_predictions = model.predict(val_dataset, verbose=1)
    val_predictions = val_predictions.flatten()

    val_predictions_denorm = val_predictions * 100
    y_val_denorm = y_val * 100

print(f"\nPredictions shape: {val_predictions_denorm.shape}")
print(f"Predictions range: [{val_predictions_denorm.min():.2f}, {val_predictions_denorm.max():.2f}]")
print(f"Actual range: [{y_val_denorm.min():.2f}, {y_val_denorm.max():.2f}]")

if not USE_KFOLD:
    rmse = np.sqrt(mean_squared_error(y_val_denorm, val_predictions_denorm))
    mae = mean_absolute_error(y_val_denorm, val_predictions_denorm)
    r2 = r2_score(y_val_denorm, val_predictions_denorm)

print("\n" + "="*80)
print("VALIDATION METRICS (Original Scale 0-100)")
print("="*80)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


print("="*80)
print("PREDICTION ANALYSIS")
print("="*80)

residuals = y_val_denorm - val_predictions_denorm

print(f"Residual Mean: {residuals.mean():.4f}")
print(f"Residual Std: {residuals.std():.4f}")
print(f"Residual Min: {residuals.min():.4f}")
print(f"Residual Max: {residuals.max():.4f}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].scatter(y_val_denorm, val_predictions_denorm, alpha=0.5, s=20, c='blue')
axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Pawpularity', fontsize=12)
axes[0, 0].set_ylabel('Predicted Pawpularity', fontsize=12)
axes[0, 0].set_title(f'Predicted vs Actual\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(val_predictions_denorm, residuals, alpha=0.5, s=20, c='purple')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Pawpularity', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].hist(residuals, bins=50, color='green', edgecolor='black', alpha=0.7)
axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('Residuals', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title(f'Residual Distribution\nMean: {residuals.mean():.2f}, Std: {residuals.std():.2f}',
                     fontsize=14, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

pred_ranges = pd.cut(val_predictions_denorm, bins=10)
error_by_range = pd.DataFrame({
    'range': pred_ranges,
    'abs_error': np.abs(residuals)
}).groupby('range')['abs_error'].mean()

axes[1, 0].bar(range(len(error_by_range)), error_by_range.values,
               color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Predicted Pawpularity Range', fontsize=12)
axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=12)
axes[1, 0].set_title('MAE by Prediction Range', fontsize=14, fontweight='bold')
axes[1, 0].set_xticklabels([f'{i*10}-{(i+1)*10}' for i in range(10)], rotation=45)
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(y_val_denorm, bins=30, alpha=0.5, label='Actual', color='blue', edgecolor='black')
axes[1, 1].hist(val_predictions_denorm, bins=30, alpha=0.5, label='Predicted', color='red', edgecolor='black')
axes[1, 1].set_xlabel('Pawpularity Score', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

sorted_errors = np.sort(np.abs(residuals))
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
axes[1, 2].plot(sorted_errors, cumulative, linewidth=2, color='teal')
axes[1, 2].set_xlabel('Absolute Error', fontsize=12)
axes[1, 2].set_ylabel('Cumulative Percentage (%)', fontsize=12)
axes[1, 2].set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axhline(50, color='red', linestyle='--', alpha=0.5)
axes[1, 2].axvline(np.median(np.abs(residuals)), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


print("="*80)
print("ANALYZING BEST AND WORST PREDICTIONS")
print("="*80)

if USE_KFOLD:
    val_results_df = train_df.copy()
    val_results_df['predicted_pawpularity'] = val_predictions_denorm
    val_results_df['actual_pawpularity'] = y_val_denorm
else:
    val_results_df = val_data.copy()
    val_results_df['predicted_pawpularity'] = val_predictions_denorm
    val_results_df['actual_pawpularity'] = y_val_denorm

residuals = val_results_df['actual_pawpularity'] - val_results_df['predicted_pawpularity']
val_results_df['absolute_error'] = np.abs(residuals)
val_results_df['residual'] = residuals

best_preds = val_results_df.nsmallest(6, 'absolute_error')
print("\nBEST PREDICTIONS (Lowest Error):")
print(best_preds[['Id', 'actual_pawpularity', 'predicted_pawpularity', 'absolute_error']])

worst_preds = val_results_df.nlargest(6, 'absolute_error')
print("\nWORST PREDICTIONS (Highest Error):")
print(worst_preds[['Id', 'actual_pawpularity', 'predicted_pawpularity', 'absolute_error']])

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (_, row) in enumerate(best_preds.iterrows()):
    if idx >= 6:
        break
    img_path = row['image_path']
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        metadata_active = [col.replace('Subject ', '') for col in metadata_cols if row[col] == 1]
        metadata_str = ', '.join(metadata_active) if metadata_active else "None"
        
        axes[idx].set_title(
            f'Actual: {row["actual_pawpularity"]:.1f} | Pred: {row["predicted_pawpularity"]:.1f}\n' +
            f'Error: {row["absolute_error"]:.1f} | {metadata_str[:30]}',
            fontsize=10,
            fontweight='bold',
            color='green'
        )

plt.suptitle('BEST PREDICTIONS (Lowest Error)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (_, row) in enumerate(worst_preds.iterrows()):
    if idx >= 6:
        break
    img_path = row['image_path']
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        metadata_active = [col.replace('Subject ', '') for col in metadata_cols if row[col] == 1]
        metadata_str = ', '.join(metadata_active) if metadata_active else "None"
        
        axes[idx].set_title(
            f'Actual: {row["actual_pawpularity"]:.1f} | Pred: {row["predicted_pawpularity"]:.1f}\n' +
            f'Error: {row["absolute_error"]:.1f} | {metadata_str[:30]}',
            fontsize=10,
            fontweight='bold',
            color='red'
        )

plt.suptitle('WORST PREDICTIONS (Highest Error)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("="*80)
print("ANALYZING METADATA FEATURE IMPACT")
print("="*80)


val_results_with_meta = val_results_df.copy()

print("\nCorrelation of Metadata Features with Absolute Error:")
for col in metadata_cols:
    corr = val_results_with_meta[col].corr(val_results_with_meta['absolute_error'])
    print(f"{col:20s}: {corr:7.4f}")

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.ravel()

for idx, col in enumerate(metadata_cols):
    data_0 = val_results_with_meta[val_results_with_meta[col] == 0]['absolute_error']
    data_1 = val_results_with_meta[val_results_with_meta[col] == 1]['absolute_error']

    bp = axes[idx].boxplot([data_0, data_1], labels=['No (0)', 'Yes (1)'], patch_artist=True)

    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    mean_0 = data_0.mean()
    mean_1 = data_1.mean()

    axes[idx].set_title(f'{col}\nMean Error: No={mean_0:.2f}, Yes={mean_1:.2f}',
                        fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Absolute Error', fontsize=10)
    axes[idx].grid(alpha=0.3)

plt.suptitle('Prediction Error by Metadata Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("="*80)
print("TEST SET PREDICTIONS")
print("="*80)

if not USE_KFOLD:
    print("\nGenerating predictions from single model...")
    
    print("Creating test dataset with correct format...")
    test_dataset = create_dataset(
        test_df['image_path'].values,
        X_test_meta,
        labels=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    test_predictions = model.predict(test_dataset, verbose=1)
    test_predictions = test_predictions.flatten()
    
    test_predictions_denorm = test_predictions * 100
    
    test_predictions_denorm = np.clip(test_predictions_denorm, 0, 100)
else:
    print(f"\nUsing ensemble of {N_FOLDS} models (already computed)")
    print(f"Test predictions are averaged across all folds")

print(f"\nTest Predictions shape: {test_predictions_denorm.shape}")
print(f"Test Predictions range: [{test_predictions_denorm.min():.2f}, {test_predictions_denorm.max():.2f}]")
print(f"Test Predictions mean: {test_predictions_denorm.mean():.2f}")
print(f"Test Predictions std: {test_predictions_denorm.std():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(test_predictions_denorm, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(test_predictions_denorm.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {test_predictions_denorm.mean():.2f}')
axes[0].axvline(np.median(test_predictions_denorm), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(test_predictions_denorm):.2f}')
axes[0].set_xlabel('Predicted Pawpularity Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Test Set - Predicted Pawpularity Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(train_df['Pawpularity'], bins=50, alpha=0.5, label='Train (Actual)', 
             color='blue', edgecolor='black')
axes[1].hist(test_predictions_denorm, bins=50, alpha=0.5, label='Test (Predicted)', 
             color='red', edgecolor='black')
axes[1].set_xlabel('Pawpularity Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution Comparison: Train vs Test Predictions', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


print("="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Pawpularity': test_predictions_denorm
})

print("\nSubmission DataFrame:")
print(submission.head(20))

print(f"\nSubmission shape: {submission.shape}")
print(f"\nSubmission statistics:")
print(submission['Pawpularity'].describe())

submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")

