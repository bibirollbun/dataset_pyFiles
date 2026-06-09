import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
import os
import cv2
from collections import Counter
import warnings
from IPython.display import display



plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


train_df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
print("Training Data Shape:", train_df.shape)
print("\nHead of few csv:")
display(train_df.head())
print("\nData Info:")
display(train_df.info())


class_counts = train_df['label'].value_counts()
print("Class Distribution:")
print(class_counts)
print(f"\nTotal classes: {train_df['label'].nunique()}")
print(f"Class names: {sorted(train_df['label'].unique())}")


print(f"\nClass imbalance ratio: {(class_counts.max() / class_counts.min()):.2f}")
class_percentages = (class_counts / len(train_df) * 100).round(2)
print("\nClass Percentages:")
for label, percentage in class_percentages.items():
    print(f"{label}: {percentage}%")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
class_counts.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8)
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sheep Breeds')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=45)

# Histogram
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
wedges, texts, autotexts = axes[1].pie(class_counts.values, labels=class_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()


train_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/'
test_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/'

if os.path.exists(train_dir):
    train_files = os.listdir(train_dir)
    print(f"Number of images in train directory: {len(train_files)}")
    print(f"Expected from train CSV: {len(train_df)}")
    

if os.path.exists(test_dir):
    test_files = os.listdir(test_dir)
    print(f"\nNumber of images in test directory: {len(test_files)}")


if 'props_df' in locals() and len(props_df) > 0:
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(props_df['width'], props_df['height'], 
                         c=props_df['label'].astype('category').cat.codes, 
                         alpha=0.6, s=50, cmap='tab10')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Image Dimensions by Class', fontweight='bold')
    
    max_dim = max(props_df['width'].max(), props_df['height'].max())
    x_line = np.linspace(0, max_dim, 100)
    plt.plot(x_line, x_line, 'r--', alpha=0.5, label='1:1 (Square)')
    plt.plot(x_line, x_line * 3/4, 'g--', alpha=0.5, label='4:3')
    plt.plot(x_line, x_line * 9/16, 'b--', alpha=0.5, label='16:9')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class')
    
    plt.show()


boxprops = dict(color='blue', linewidth=2)
whiskerprops = dict(color='blue', linewidth=1.5, linestyle='--')
capprops = dict(color='blue', linewidth=1.5)
flierprops = dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none')

if 'props_df' in locals() and len(props_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    props_df.boxplot(column='width', by='label', ax=axes[0],
                     boxprops=boxprops, whiskerprops=whiskerprops,
                     capprops=capprops, flierprops=flierprops)
    axes[0].set_title('Image Width by Class')
    axes[0].set_xlabel('Sheep Breed')
    axes[0].set_ylabel('Width (pixels)')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    props_df.boxplot(column='height', by='label', ax=axes[1],
                     boxprops=boxprops, whiskerprops=whiskerprops,
                     capprops=capprops, flierprops=flierprops)
    axes[1].set_title('Image Height by Class')
    axes[1].set_xlabel('Sheep Breed')
    axes[1].set_ylabel('Height (pixels)')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


figsize= (20, 15)
samples_per_class= 3
classes = sorted(train_df['label'].unique())
n_classes = len(classes)

fig, axes = plt.subplots(n_classes, samples_per_class, figsize=figsize)
if n_classes == 1:
    axes = axes.reshape(1, -1)

for i, class_name in enumerate(classes):
    class_df = train_df[train_df['label'] == class_name]
    sample_files = class_df['filename'].sample(min(samples_per_class, len(class_df))).values
    
    for j, filename in enumerate(sample_files):
        img_path = os.path.join(train_dir, filename)
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{class_name}\n{filename}\n{img.size[0]}x{img.size[1]}', 
                                   fontsize=10)
                axes[i, j].axis('off')
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f'Error loading\n{filename}', 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{class_name} - Error')
                axes[i, j].axis('off')
        else:
            axes[i, j].text(0.5, 0.5, f'File not found\n{filename}', 
                          ha='center', va='center', transform=axes[i, j].transAxes)
            axes[i, j].set_title(f'{class_name} - Missing')
            axes[i, j].axis('off')
    
    for j in range(len(sample_files), samples_per_class):
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()


from PIL import Image
import matplotlib.pyplot as plt

if duplicate_hashes:
    for h in list(duplicate_hashes)[:5]:
        train_file = train_hashes[h]
        test_file = test_hashes[h]
        
        train_path = os.path.join(train_dir, train_file)
        test_path = os.path.join(test_dir, test_file)

        try:
            img_train = Image.open(train_path)
            img_test = Image.open(test_path)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_train)
            axes[0].set_title(f"Train: {train_file}", fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(img_test)
            axes[1].set_title(f"Test: {test_file}", fontsize=10)
            axes[1].axis('off')
            
            plt.suptitle("Duplicate Image Found in Train and Test", fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error displaying {train_file} and {test_file}: {e}")


print("=== DATA QUALITY ===\n")

print("1. FILE INTEGRITY:")
if os.path.exists(train_dir):
    problematic_files = []
    sample_files = train_df['filename'].sample(min(100, len(train_df)))
    
    for filename in sample_files:
        img_path = os.path.join(train_dir, filename)
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            problematic_files.append((filename, str(e)))
    
    print(f"Checked {len(sample_files)} sample files")
    print(f"Problematic files found: {len(problematic_files)}")
    if problematic_files:
        print("Problematic files:", problematic_files[:5])
else:
    print("Train directory not found!")

imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1]

# 2. Class balance 
print(f"\n2. CLASS BALANCE:")
print(f"Most frequent class: {class_counts.index[0]} ({class_counts.iloc[0]} images)")
print(f"Least frequent class: {class_counts.index[-1]} ({class_counts.iloc[-1]} images)")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 3:
    print("âš ï¸�  HIGH CLASS IMBALANCE")
elif imbalance_ratio > 2:
    print("âš ï¸�  MODERATE CLASS IMBALANCE")
else:
    print("âœ… CLASS DISTRIBUTION IS REASONABLE")

# 3. Image quality
if 'props_df' in locals() and len(props_df) > 0:
    print(f"\n3. IMAGE QUALITY:")
    
    small_images = props_df[(props_df['width'] < 224) | (props_df['height'] < 224)]
    print(f"Small images: {len(small_images)} ({len(small_images)/len(props_df)*100:.1f}%)")
    
    large_images = props_df[(props_df['width'] > 2048) | (props_df['height'] > 2048)]
    print(f"Large images: {len(large_images)} ({len(large_images)/len(props_df)*100:.1f}%)")
    
    large_files = props_df[props_df['file_size_kb'] > 1000]
    small_files = props_df[props_df['file_size_kb'] < 10]
    print(f"Large files (>1MB): {len(large_files)} ({len(large_files)/len(props_df)*100:.1f}%)")
    print(f"Small files (<10KB): {len(small_files)} ({len(small_files)/len(props_df)*100:.1f}%)")


def show_random_sheep_image(breed, train_df=train_df, train_dir=train_dir):
    if breed not in train_df['label'].unique():
        print(f"Breed '{breed}' not found in the dataset")
        return
        
    breed_df = train_df[train_df['label'] == breed]
    random_file = breed_df['filename'].sample(1).values[0]
    img_path = os.path.join(train_dir, random_file)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f'{breed}\n{random_file}\n{img.size[0]}x{img.size[1]}', fontsize=12)
        plt.axis('off')
        plt.show()
    else:
        print(f"Image file '{random_file}' not found in directory.")


show_random_sheep_image("Naeimi")







