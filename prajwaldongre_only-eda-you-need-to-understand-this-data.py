import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

import warnings
warnings.simplefilter('ignore')


def load_data(file_path):
    with h5py.File(file_path, "r") as h5file:
        train_images = {k: np.array(v) for k, v in h5file["images/Train"].items()}
        train_spots = {k: np.array(v) for k, v in h5file["spots/Train"].items()}
        test_images = {k: np.array(v) for k, v in h5file["images/Test"].items()}
        test_spots = {k: np.array(v) for k, v in h5file["spots/Test"].items()}
    return train_images, train_spots, test_images, test_spots


file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"  
train_images, train_spots, test_images, test_spots = load_data(file_path)


print("Dataset Overview:")
print(f"  Number of training slides: {len(train_images)}")
print(f"  Number of test slides: {len(test_images)}")


print("\nTraining Images Structure:")
for slide_name, image in train_images.items():
    print(f"  Slide: {slide_name}, Shape: {image.shape}, Data Type: {image.dtype}")


print("\nTraining Spots Structure:")
for slide_name, spots in train_spots.items():
    print(f"  Slide: {slide_name}, Number of Spots: {len(spots)}, Data Type: {spots.dtype}")


print("\nTest Images Structure:")
for slide_name, image in test_images.items():
    print(f"  Slide: {slide_name}, Shape: {image.shape}, Data Type: {image.dtype}")


print("\nTest Spots Structure:")
for slide_name, spots in test_spots.items():
    print(f"  Slide: {slide_name}, Number of Spots: {len(spots)}, Data Type: {spots.dtype}")


# Image Size Distribution
train_widths = [image.shape[1] for image in train_images.values()]
train_heights = [image.shape[0] for image in train_images.values()]
test_widths = [image.shape[1] for image in test_images.values()]
test_heights = [image.shape[0] for image in test_images.values()]


sns.set_style("whitegrid")
sns.set_palette("coolwarm")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

sns.histplot(train_widths, kde=True, bins=30, ax=axes[0], color="#4c72b0", edgecolor="black", alpha=0.8)
axes[0].set_title("Train Image Widths Distribution", fontsize=14, fontweight='bold', color="#333")
axes[0].set_xlabel("Width", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)

sns.histplot(train_heights, kde=True, bins=30, ax=axes[1], color="#dd8452", edgecolor="black", alpha=0.8)
axes[1].set_title("Train Image Heights Distribution", fontsize=14, fontweight='bold', color="#333")
axes[1].set_xlabel("Height", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)

plt.tight_layout()
plt.show()

# Test Image Width Distribution
plt.figure(figsize=(7, 4), dpi=100)
sns.histplot(test_widths, kde=True, bins=30, color="#55a868", edgecolor="black", alpha=0.8)
plt.title("Test Image Widths Distribution", fontsize=14, fontweight='bold', color="#333")
plt.xlabel("Width", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Test Image Height Distribution
plt.figure(figsize=(7, 4), dpi=100)
sns.histplot(test_heights, kde=True, bins=30, color="#c44e52", edgecolor="black", alpha=0.8)
plt.title("Test Image Heights Distribution", fontsize=14, fontweight='bold', color="#333")
plt.xlabel("Height", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3)
plt.show()


def analyze_color_channels(images):
    channel_means = {}
    channel_stds = {}
    for slide_name, image in images.items():
        # Flatten the image to calculate statistics across all pixels
        pixels = image.reshape(-1, 3)
        channel_means[slide_name] = np.mean(pixels, axis=0)
        channel_stds[slide_name] = np.std(pixels, axis=0)
    return channel_means, channel_stds


train_channel_means, train_channel_stds = analyze_color_channels(train_images)
test_channel_means, test_channel_stds = analyze_color_channels(test_images)


sns.set_style("whitegrid")

num_slides = len(train_channel_means)
cols = 3  
rows = (num_slides // cols) + (num_slides % cols > 0)  

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), dpi=100)

axes = axes.flatten() if num_slides > 1 else [axes]

colors = ["#e63946", "#2a9d8f", "#457b9d"]  

for i, (slide_name, means) in enumerate(train_channel_means.items()):
    axes[i].bar(["Red", "Green", "Blue"], means, color=colors, edgecolor="black", alpha=0.85)
    axes[i].set_title(f"Train {slide_name} - Channel Means", fontsize=14, fontweight="bold", color="#333")
    axes[i].set_ylabel("Mean Pixel Value", fontsize=12)
    axes[i].set_ylim(0, 1)  
    axes[i].set_xticklabels(["Red", "Green", "Blue"], fontsize=12)

plt.tight_layout()
plt.show()


sns.set_style("whitegrid")

num_slides = len(train_channel_stds)
cols = 3  
rows = (num_slides // cols) + (num_slides % cols > 0)  

fig, axes = plt.subplots(rows, cols, figsize=(17, 4 * rows), dpi=100)

axes = axes.flatten() if num_slides > 1 else [axes]

colors = ["#e63946", "#2a9d8f", "#457b9d"]

for i, (slide_name, stds) in enumerate(train_channel_stds.items()):
    axes[i].bar(["Red", "Green", "Blue"], stds, color=colors, edgecolor="black", alpha=0.85)
    axes[i].set_title(f"Train {slide_name} - Channel Stds", fontsize=14, fontweight="bold", color="#333")
    axes[i].set_ylabel("Standard Deviation", fontsize=12)
    axes[i].set_ylim(0, 1)  
    axes[i].set_xticklabels(["Red", "Green", "Blue"], fontsize=12)


plt.tight_layout()
plt.show()


spot_counts = {
    slide_name: len(spots) for slide_name, spots in train_spots.items()
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(spot_counts.keys()), y=list(spot_counts.values()))
plt.title("Number of Spots per Training Image")
plt.xlabel("Slide")
plt.ylabel("Number of Spots")
plt.show()


sns.set_style("whitegrid")

num_slides = len(train_spots)
cols = 3  
rows = (num_slides // cols) + (num_slides % cols > 0)  

fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows), dpi=100)

axes = axes.flatten() if num_slides > 1 else [axes]

# Aesthetic color palette
spot_color = "#000000"  # black color for spots

for i, (slide_name, spots) in enumerate(train_spots.items()):
    axes[i].scatter(
        spots["x"], spots["y"], s=5, color=spot_color, alpha=1, edgecolors='w', linewidth=0.5
    )  
    axes[i].set_title(f"Spot Distribution - {slide_name}", fontsize=14, fontweight="bold", color="#ff0808")
    axes[i].set_xlabel("X Coordinate", fontsize=12)
    axes[i].set_ylabel("Y Coordinate", fontsize=12)
    axes[i].invert_yaxis()  
    axes[i].tick_params(axis='both', labelsize=10)  

plt.tight_layout()
plt.show()


train_spot_dfs = {
    slide_name: pd.DataFrame(spots) for slide_name, spots in train_spots.items()
}

all_train_spots_df = pd.concat(train_spot_dfs.values(), ignore_index=True)

cell_type_cols = [
    col for col in all_train_spots_df.columns if col.startswith("C")
]  
cell_type_stats = all_train_spots_df[cell_type_cols].describe()

print("Descriptive Statistics for Cell Type Abundances:")
cell_type_stats


cell_type_corr = all_train_spots_df[cell_type_cols].corr()

plt.figure(figsize=(20, 12))
sns.heatmap(cell_type_corr, annot=False, cmap="Blues")  
plt.title("Correlation Matrix of Cell Type Abundances")
plt.show()


plt.figure(figsize=(12, 6))
for i, slide_name in enumerate(list(train_images.keys())[:3]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(train_images[slide_name])
    plt.title(f"Train - {slide_name}")
    plt.axis("off") 
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
# Display the test image
plt.imshow(test_images["S_7"])
plt.title("Test - S_7")
plt.axis("off")
plt.show()


test_spots_df = pd.DataFrame(test_spots["S_7"])

test_set_counts = test_spots_df["Test_Set"].value_counts()

print("Value Counts for 'Test_Set' Column:")
print(test_set_counts)

plt.figure(figsize=(6, 4))
sns.barplot(x=test_set_counts.index, y=test_set_counts.values)
plt.title("Distribution of Test_Set Values")
plt.xlabel("Test_Set Value")
plt.ylabel("Count")
plt.show()

