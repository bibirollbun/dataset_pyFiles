import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tifffile as tiff  # For loading .tif images
import seaborn as sns
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm  # Import progress bar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam
import pyro.optim as optim
import time
from scipy.stats import shapiro, skew, kurtosis
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import silhouette_score  
from sklearn.cluster import DBSCAN  



# Define main dataset path
DATASET_PATH = r"D:\hpa-single-cell-image-classification"

# Define sub-paths
TRAIN_PATH = os.path.join(DATASET_PATH, "train")  # Path to training images
LABELS_PATH = os.path.join(DATASET_PATH, "train.csv")  # Path to labels CSV

# Print to verify
print("Train Path:", TRAIN_PATH)
print("Labels Path:", LABELS_PATH)


df = pd.read_csv(LABELS_PATH)
print(df.head())  # Check first few rows


# Get first image ID from the CSV
sample_id = df.iloc[0]["ID"]

# Load each channel (CHANGE .tif to .png)
img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{sample_id}_blue.png"), cv2.IMREAD_UNCHANGED)
img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{sample_id}_green.png"), cv2.IMREAD_UNCHANGED)  # Protein of interest
img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{sample_id}_red.png"), cv2.IMREAD_UNCHANGED)
img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{sample_id}_yellow.png"), cv2.IMREAD_UNCHANGED)

# Convert images to RGB format for correct visualization
img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2RGB)
img_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2RGB)
img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2RGB)
img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2RGB)

# Display images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img_blue); axes[0].set_title("Nucleus (Blue)")
axes[1].imshow(img_green); axes[1].set_title("Protein of Interest (Green)")
axes[2].imshow(img_red); axes[2].set_title("Microtubules (Red)")
axes[3].imshow(img_yellow); axes[3].set_title("Endoplasmic Reticulum (Yellow)")
plt.show()


print("Blue Channel Shape:", img_blue.shape)
print("Green Channel Shape:", img_green.shape)  # This is the protein signal
print("Red Channel Shape:", img_red.shape)
print("Yellow Channel Shape:", img_yellow.shape)

# Check pixel range
print("Min Pixel Value:", np.min(img_green))
print("Max Pixel Value:", np.max(img_green))


# Stack original RGB images into a single 12-channel tensor
img_stacked_rgb = np.concatenate([img_blue, img_green, img_red, img_yellow], axis=-1)
print("Non-Grayscale Shape:", img_stacked_rgb.shape)  # Should be (2048, 2048, 12)


# Convert each channel to grayscale
img_blue_gray = cv2.cvtColor(img_blue, cv2.COLOR_RGB2GRAY)
img_green_gray = cv2.cvtColor(img_green, cv2.COLOR_RGB2GRAY)
img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_RGB2GRAY)
img_yellow_gray = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)

# Stack grayscale images into a single 4-channel tensor
img_stacked_gray = np.stack([img_blue_gray, img_green_gray, img_red_gray, img_yellow_gray], axis=-1)
print("Grayscale Shape:", img_stacked_gray.shape)  # Should be (2048, 2048, 4)


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (resize & normalize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

def extract_cnn_features_green(img):
    """Extract features from the Green channel (Protein of Interest) using ResNet."""
    img_green = img[:, :, 1]  # âœ… Extract Green channel (index 1)

    # Convert single-channel grayscale to 3-channel by stacking
    img_green_3ch = np.stack([img_green, img_green, img_green], axis=-1)  # (H, W, 3)

    img_tensor = transform(img_green_3ch).unsqueeze(0).to(device)  # âœ… Move tensor to GPU
    with torch.no_grad():
        features = model(img_tensor)  # âœ… Model is on GPU
    return features.squeeze().cpu().numpy()  # âœ… Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "test_green_channel_features.npy"

# Get the first 10 images for testing
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f][:10]
num_images = len(all_files)
print(f"Processing {num_images} Green channel images...")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_green = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_green = []
    start_index = 0

# Loop through the first 10 images
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four grayscale images (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_rgb = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Extract features for this image
    features = extract_cnn_features_green(img_stacked_rgb)

    # Store in the list
    all_features_green.append(features)

    # Save after every image (since we only have 10, we save frequently)
    np.save(SAVE_PATH, np.array(all_features_green))
    print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_green))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Green-Only):", np.array(all_features_green).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


# Load the saved features
features_green = np.load("test_green_channel_features.npy")

# Print shape
print("Loaded Feature Shape:", features_green.shape)  # Should be (10, 2048)

# Confirm we have exactly 10 vectors
if features_green.shape[0] == 10:
    print("âœ… The file contains 10 feature vectors.")
else:
    print("â�Œ Something is wrong! The file has", features_green.shape[0], "vectors.")


# Load features
features_green = np.load("test_green_channel_features.npy")

# Compute differences between consecutive vectors
diffs = np.diff(features_green, axis=0)

# Count how many vectors are identical
identical_vectors = np.sum(np.all(diffs == 0, axis=1))

if identical_vectors == 0:
    print("âœ… All feature vectors are unique.")
else:
    print(f"â�Œ {identical_vectors} feature vectors are duplicates!")


for i in range(3):  # Print first 3 vectors
    print(f"\nFeature Vector {i + 1} (First 10 values):", features_green[i][:10])


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (normalize and resize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize intensity
])

def create_pseudo_rgb(img):
    """Convert 4-channel image into a 3-channel (R-G-B) image."""
    r = img[:, :, 2]  # Use Red channel (Microtubules)
    g = img[:, :, 3]  # Use Green channel (Protein of Interest)
    b = img[:, :, 0]  # Use Blue channel (Nucleus)
    
    pseudo_rgb = np.stack([r, g, b], axis=-1)  # Stack into (H, W, 3)
    return pseudo_rgb.astype(np.uint8)  # Ensure proper dtype

def extract_cnn_features_pseudo_rgb(img):
    """Extract features from a Pseudo-RGB image using ResNet."""
    img_tensor = transform(img).unsqueeze(0).to(device)  # âœ… Move tensor to GPU
    with torch.no_grad():
        features = model(img_tensor)  # âœ… Model is on GPU
    return features.squeeze().cpu().numpy()  # âœ… Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "test_pseudo_rgb_features.npy"

# Get list of first 10 images for testing
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f][:10]
num_images = len(all_files)
print(f"Processing {num_images} Pseudo-RGB images...\n")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_pseudo_rgb = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_pseudo_rgb = []
    start_index = 0

# Loop through first 10 images
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features (Pseudo-RGB)", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four images as grayscale (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_rgb = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Convert to Pseudo-RGB
    pseudo_rgb_image = create_pseudo_rgb(img_stacked_rgb)

    # Extract features
    features = extract_cnn_features_pseudo_rgb(pseudo_rgb_image)

    # Store in the list
    all_features_pseudo_rgb.append(features)

    # Save after every image (for testing)
    np.save(SAVE_PATH, np.array(all_features_pseudo_rgb))
    print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_pseudo_rgb))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Pseudo-RGB):", np.array(all_features_pseudo_rgb).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


# Load extracted feature files
features_green = np.load("test_green_channel_features.npy")  # Green-Only features
features_pseudo_rgb = np.load("test_pseudo_rgb_features.npy")  # Pseudo-RGB features

print("Loaded Green-Only Feature Shape:", features_green.shape)
print("Loaded Pseudo-RGB Feature Shape:", features_pseudo_rgb.shape)


diff = np.abs(features_green - features_pseudo_rgb)
print("Mean Absolute Feature Difference:", np.mean(diff))
print("Max Absolute Feature Difference:", np.max(diff))


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (normalize and resize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize intensity
])

def convert_4ch_to_3ch(img_4ch):
    """Convert 4-channel grayscale image to 3-channel for ResNet."""
    r = img_4ch[:, :, 0]  # Blue (Nucleus)
    g = img_4ch[:, :, 1]  # Green (Protein of Interest)
    b = (img_4ch[:, :, 2] + img_4ch[:, :, 3]) / 2  # Merge Red (Microtubules) & Yellow (ER)

    grayscale_3ch = np.stack([r, g, b], axis=-1)  # Shape: (H, W, 3)
    return np.clip(grayscale_3ch, 0, 255).astype(np.uint8)  # Ensure valid pixel range

def extract_cnn_features_grayscale(img):
    """Extract features from the 3-channel grayscale image using ResNet."""
    img_tensor = transform(img).unsqueeze(0).to(device)  # Convert to tensor & move to GPU
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().cpu().numpy()  # Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "test_grayscale_3ch_features.npy"

# Get first 10 images for testing
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f][:10]
num_images = len(all_files)

print(f"Processing {num_images} Grayscale 3-Channel images...\n")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_grayscale = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_grayscale = []
    start_index = 0

# Loop through 10 images with a progress bar
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features (Grayscale 3-CH)", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four grayscale images (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_gray = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Convert to 3-channel grayscale
    img_grayscale_3ch = convert_4ch_to_3ch(img_stacked_gray)

    # Extract features for this image
    features = extract_cnn_features_grayscale(img_grayscale_3ch)

    # Store feature vector
    all_features_grayscale.append(features)

    # Save after every image (for testing)
    np.save(SAVE_PATH, np.array(all_features_grayscale))
    print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_grayscale))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Grayscale 3-Channel):", np.array(all_features_grayscale).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


# Load extracted feature files
features_green = np.load("test_green_channel_features.npy")  # Green-Only
features_pseudo_rgb = np.load("test_pseudo_rgb_features.npy")  # Pseudo-RGB
features_grayscale = np.load("test_grayscale_3ch_features.npy")  # Grayscale 3-CH

print("Feature Shapes:")
print("Green-Only:", features_green.shape)
print("Pseudo-RGB:", features_pseudo_rgb.shape)
print("Grayscale 3-CH:", features_grayscale.shape)


diff_green_gray = np.abs(features_green - features_grayscale)
diff_pseudo_gray = np.abs(features_pseudo_rgb - features_grayscale)

print("\nMean Absolute Feature Difference (Green vs Grayscale):", np.mean(diff_green_gray))
print("Max Absolute Feature Difference (Green vs Grayscale):", np.max(diff_green_gray))

print("\nMean Absolute Feature Difference (Pseudo-RGB vs Grayscale):", np.mean(diff_pseudo_gray))
print("Max Absolute Feature Difference (Pseudo-RGB vs Grayscale):", np.max(diff_pseudo_gray))


# Reduce to 2D for visualization
pca = PCA(n_components=2)
features_combined = np.vstack((features_green, features_pseudo_rgb, features_grayscale))  # Stack all feature sets
features_2d = pca.fit_transform(features_combined)

# Plot Green-Only vs Pseudo-RGB vs Grayscale 3-CH
plt.figure(figsize=(8,6))
plt.scatter(features_2d[:10, 0], features_2d[:10, 1], label="Green-Only", alpha=0.7, color="blue")
plt.scatter(features_2d[10:20, 0], features_2d[10:20, 1], label="Pseudo-RGB", alpha=0.7, color="red")
plt.scatter(features_2d[20:, 0], features_2d[20:, 1], label="Grayscale 3-CH", alpha=0.7, color="gray")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection: Green-Only vs Pseudo-RGB vs Grayscale 3-CH Features")
plt.legend()
plt.show()


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (resize & normalize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

def extract_cnn_features_green(img):
    """Extract features from the Green channel (Protein of Interest) using ResNet."""
    img_green = img[:, :, 1]  # âœ… Extract Green channel (index 1)

    # Convert single-channel grayscale to 3-channel by stacking
    img_green_3ch = np.stack([img_green, img_green, img_green], axis=-1)  # (H, W, 3)

    img_tensor = transform(img_green_3ch).unsqueeze(0).to(device)  # âœ… Move tensor to GPU
    with torch.no_grad():
        features = model(img_tensor)  # âœ… Model is on GPU
    return features.squeeze().cpu().numpy()  # âœ… Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "green_channel_features.npy"

# Get all images
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f]
num_images = len(all_files)

print(f"Processing {num_images} Green channel images...\n")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_green = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_green = []
    start_index = 0

# Loop through all images
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four grayscale images (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_rgb = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Extract features for this image
    features = extract_cnn_features_green(img_stacked_rgb)

    # Store in the list
    all_features_green.append(features)

    # Save every 100 images (prevents data loss)
    if (i + start_index + 1) % 100 == 0:
        np.save(SAVE_PATH, np.array(all_features_green))
        print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_green))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Green-Only):", np.array(all_features_green).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (normalize and resize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize intensity
])

def create_pseudo_rgb(img):
    """Convert 4-channel image into a 3-channel (R-G-B) image."""
    r = img[:, :, 2]  # Use Red channel (Microtubules)
    g = img[:, :, 3]  # Use Green channel (Protein of Interest)
    b = img[:, :, 0]  # Use Blue channel (Nucleus)
    
    pseudo_rgb = np.stack([r, g, b], axis=-1)  # Stack into (H, W, 3)
    return pseudo_rgb.astype(np.uint8)  # Ensure proper dtype

def extract_cnn_features_pseudo_rgb(img):
    """Extract features from a Pseudo-RGB image using ResNet."""
    img_tensor = transform(img).unsqueeze(0).to(device)  # âœ… Move tensor to GPU
    with torch.no_grad():
        features = model(img_tensor)  # âœ… Model is on GPU
    return features.squeeze().cpu().numpy()  # âœ… Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "pseudo_rgb_features.npy"

# Get all images
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f]
num_images = len(all_files)

print(f"Processing {num_images} Pseudo-RGB images...\n")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_pseudo_rgb = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_pseudo_rgb = []
    start_index = 0

# Loop through all images
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features (Pseudo-RGB)", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four images as grayscale (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_rgb = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Convert to Pseudo-RGB
    pseudo_rgb_image = create_pseudo_rgb(img_stacked_rgb)

    # Extract features
    features = extract_cnn_features_pseudo_rgb(pseudo_rgb_image)

    # Store in the list
    all_features_pseudo_rgb.append(features)

    # Save every 100 images (to prevent data loss)
    if (i + start_index + 1) % 100 == 0:
        np.save(SAVE_PATH, np.array(all_features_pseudo_rgb))
        print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_pseudo_rgb))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Pseudo-RGB):", np.array(all_features_pseudo_rgb).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


# Load pretrained ResNet model and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Define transform (normalize and resize for CNN)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize intensity
])

def convert_4ch_to_3ch(img_4ch):
    """Convert 4-channel grayscale image to 3-channel for ResNet."""
    r = img_4ch[:, :, 0]  # Blue (Nucleus)
    g = img_4ch[:, :, 1]  # Green (Protein of Interest)
    b = (img_4ch[:, :, 2] + img_4ch[:, :, 3]) / 2  # Merge Red (Microtubules) & Yellow (ER)

    grayscale_3ch = np.stack([r, g, b], axis=-1)  # Shape: (H, W, 3)
    return np.clip(grayscale_3ch, 0, 255).astype(np.uint8)  # Ensure valid pixel range

def extract_cnn_features_grayscale(img):
    """Extract features from the 3-channel grayscale image using ResNet."""
    img_tensor = transform(img).unsqueeze(0).to(device)  # Convert to tensor & move to GPU
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().cpu().numpy()  # Move result back to CPU for NumPy

# Directory where images are stored
TRAIN_PATH = "D:/hpa-single-cell-image-classification/train"
SAVE_PATH = "grayscale_3ch_features.npy"

# Get all images
all_files = [f for f in os.listdir(TRAIN_PATH) if "_green.png" in f]
num_images = len(all_files)

print(f"Processing {num_images} Grayscale 3-Channel images...\n")

# Initialize or resume from existing file
if os.path.exists(SAVE_PATH):
    saved_features = np.load(SAVE_PATH)
    start_index = len(saved_features)
    all_features_grayscale = list(saved_features)  # Convert back to list
    print(f"Resuming from index {start_index}/{num_images}...")
else:
    all_features_grayscale = []
    start_index = 0

# Loop through all images
for i, filename in enumerate(tqdm(all_files[start_index:], desc="Extracting CNN Features (Grayscale 3-CH)", unit="image")):
    base_id = filename.replace("_green.png", "")

    # Load all four grayscale images (single-channel)
    img_blue = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_blue.png"), cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_green.png"), cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_red.png"), cv2.IMREAD_GRAYSCALE)
    img_yellow = cv2.imread(os.path.join(TRAIN_PATH, f"{base_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)

    # Stack into (H, W, 4)
    img_stacked_gray = np.stack([img_blue, img_green, img_red, img_yellow], axis=-1)

    # Convert to 3-channel grayscale
    img_grayscale_3ch = convert_4ch_to_3ch(img_stacked_gray)

    # Extract features
    features = extract_cnn_features_grayscale(img_grayscale_3ch)

    # Store feature vector
    all_features_grayscale.append(features)

    # Save every 100 images (to prevent data loss)
    if (i + start_index + 1) % 100 == 0:
        np.save(SAVE_PATH, np.array(all_features_grayscale))
        print(f"Progress saved at {i + start_index + 1}/{num_images} images.")

# Final save
np.save(SAVE_PATH, np.array(all_features_grayscale))
print("\nFeature extraction complete!")
print("Final Shape of Extracted Features (Grayscale 3-Channel):", np.array(all_features_grayscale).shape)
print(f"Features saved successfully to '{SAVE_PATH}'")


features_green = np.load("green_channel_features.npy")  # Entire dataset features
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")


# Reduce to 2D for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_combined)
plt.figure(figsize=(8,6))
num_images = features_green.shape[0]

plt.scatter(features_2d[:num_images, 0], features_2d[:num_images, 1], label="Green-Only", alpha=0.7, color="blue")
plt.scatter(features_2d[num_images:2*num_images, 0], features_2d[num_images:2*num_images, 1], label="Pseudo-RGB", alpha=0.7, color="red")
plt.scatter(features_2d[2*num_images:, 0], features_2d[2*num_images:, 1], label="Grayscale 3-CH", alpha=0.7, color="gray")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection: Green-Only vs Pseudo-RGB vs Grayscale 3-CH Features")
plt.legend()
plt.show()


# Load all feature sets
features_green = np.load("green_channel_features.npy")  
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# Stack all feature sets into one array
features_combined = np.vstack((features_green, features_pseudo_rgb, features_grayscale))

# Standardize the features to have mean=0 and std=1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_combined)

# Perform PCA reduction
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Define number of images in each feature set
num_images = features_green.shape[0]

# Plot PCA projections
plt.figure(figsize=(8,6))
plt.scatter(features_2d[:num_images, 0], features_2d[:num_images, 1], label="Green-Only", alpha=0.7, color="blue")
plt.scatter(features_2d[num_images:2*num_images, 0], features_2d[num_images:2*num_images, 1], label="Pseudo-RGB", alpha=0.7, color="red")
plt.scatter(features_2d[2*num_images:, 0], features_2d[2*num_images:, 1], label="Grayscale 3-CH", alpha=0.7, color="gray")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection: Green-Only vs Pseudo-RGB vs Grayscale 3-CH Features")
plt.legend()
plt.show()


features_grayscale = np.load("grayscale_3ch_features.npy") # âœ… Load features 

# âœ… Sample a subset (to speed up computation) 
subset_size = 5000 
features_grayscale = features_grayscale[:subset_size] 


# âœ… Pick 5 random features from each dataset 
np.random.seed(42)
random_features_gray = np.random.choice(features_grayscale.shape[1], size=5, replace=False)  

# âœ… Analyze each feature in Grayscale 
print("\nğŸ“Š **Grayscale Features**") 
for feature_idx in random_features_gray:     
    stat, p_value = shapiro(features_grayscale[:, feature_idx])     
    skewness = skew(features_grayscale[:, feature_idx])     
    kurt = kurtosis(features_grayscale[:, feature_idx])      
    print(f"ğŸ”� Feature {feature_idx}:")     
    print(f"   - Shapiro-Wilk p-value = {p_value}")    
    print(f"   - Skewness = {skewness}")    
    print(f"   - Kurtosis = {kurt}\n") 


features_pseudo_rgb = np.load("pseudo_rgb_features.npy") 
 

# âœ… Sample a subset (to speed up computation) 
subset_size = 5000 
features_pseudo_rgb = features_pseudo_rgb[:subset_size] 
features_grayscale = features_grayscale[:subset_size]  

# âœ… Pick 5 random features from each dataset 
np.random.seed(42) 
random_features_pseudo = np.random.choice(features_pseudo_rgb.shape[1], size=5, replace=False) 
random_features_gray = np.random.choice(features_grayscale.shape[1], size=5, replace=False)  
# âœ… Analyze each feature in Pseudo-RGB 
print("\nğŸ“Š **Pseudo-RGB Features**") 
for feature_idx in random_features_pseudo:     
    stat, p_value = shapiro(features_pseudo_rgb[:, feature_idx])     
    skewness = skew(features_pseudo_rgb[:, feature_idx])     
    kurt = kurtosis(features_pseudo_rgb[:, feature_idx])      
    print(f"ğŸ”� Feature {feature_idx}:")     
    print(f"   - Shapiro-Wilk p-value = {p_value}")     
    print(f"   - Skewness = {skewness}")     
    print(f"   - Kurtosis = {kurt}\n")  


# âœ… Load features
features_green = np.load("green_channel_features.npy")

# âœ… Sample a subset
subset_size = 5000
features_green = features_green[:subset_size]

# âœ… Pick 5 random features
np.random.seed(42)
random_features = np.random.choice(features_green.shape[1], size=5, replace=False)

# âœ… Analyze each feature
for feature_idx in random_features:
    stat, p_value = shapiro(features_green[:, feature_idx])
    skewness = skew(features_green[:, feature_idx])
    kurt = kurtosis(features_green[:, feature_idx])

    print(f"\nğŸ”� Feature {feature_idx}:")
    print(f"   - Shapiro-Wilk p-value = {p_value}")
    print(f"   - Skewness = {skewness}")
    print(f"   - Kurtosis = {kurt}")


# âœ… Load features
features_green = np.load("green_channel_features.npy")
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# âœ… Check for NaNs
print("Checking for NaNs...")
features_green = np.nan_to_num(features_green)
features_pseudo_rgb = np.nan_to_num(features_pseudo_rgb)
features_grayscale = np.nan_to_num(features_grayscale)

# âœ… Sample a smaller subset of data
subset_size = 5000
features_green = features_green[:subset_size]
features_pseudo_rgb = features_pseudo_rgb[:subset_size]
features_grayscale = features_grayscale[:subset_size]

# âœ… Standardize features
scaler = StandardScaler()
features_green = scaler.fit_transform(features_green)
features_pseudo_rgb = scaler.transform(features_pseudo_rgb)
features_grayscale = scaler.transform(features_grayscale)

# âœ… DP-GMM Model Setup (Fixes Applied)
dpgmm = BayesianGaussianMixture(
    n_components=50,  
    weight_concentration_prior=1e-2,  
    weight_concentration_prior_type="dirichlet_process",
    covariance_type="full",
    reg_covar=1e-4,  
    init_params="kmeans",  
    n_init=1,  
    max_iter=5,  # â¬† More iterations per fit
    warm_start=True,  
    tol=1e-4,  # â¬‡ Reduce tolerance
    verbose=2,
    random_state=42
)

# âœ… Track ELBO per iteration
num_iterations = 5
elbo_values = []

# âœ… Check if `warm_start` is actually working
print("Warm Start Enabled:", dpgmm.warm_start)

previous_means = None  # Track mean changes

print("\nFitting DP-GMM on Green-Only Features with Manual Iteration Tracking...")
for i in range(num_iterations):
    dpgmm.lower_bound_ = -np.inf  # ğŸ”„ Reset ELBO to force recomputation
    
    dpgmm.fit(features_green)  # Run one step of EM
    elbo_values.append(dpgmm.lower_bound_)
    
    # âœ… Check if means are changing
    if previous_means is not None:
        print(f"Iteration {i+1}, Mean Shift: {np.linalg.norm(dpgmm.means_ - previous_means)}")
    previous_means = np.copy(dpgmm.means_)
    
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

# âœ… Save ELBO values
np.save("dpgmm_elbo_green_debug.npy", elbo_values)

# ğŸ“Š Plot ELBO over iterations
plt.figure(figsize=(8,5))
plt.plot(range(1, num_iterations + 1), elbo_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Over Iterations for Green-Only Features (Debugging)")
plt.grid()
plt.show()

# âœ… Get cluster assignments
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscidebug.npy", clusters_green)


# Load all feature sets
features_green = np.load("green_channel_features.npy")      # (21806, 2048)
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")    # (21806, 2048)
features_grayscale = np.load("grayscale_3ch_features.npy")  # (21806, 2048)

# DP-GMM Model Setup
dpgmm = BayesianGaussianMixture(
    n_components=100,  # Max clusters, DP picks the optimal number
    covariance_type="full",
    weight_concentration_prior_type="dirichlet_process",
    verbose = 2,
    max_iter=5,
    random_state=42
)

# Run DP-GMM on Green Features
print("Fitting DP-GMM on Green-Only Features...")
dpgmm.fit(features_green)
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscimk1.npy", clusters_green)

# Run DP-GMM on Pseudo-RGB Features
print("Fitting DP-GMM on Pseudo-RGB Features...")
dpgmm.fit(features_pseudo_rgb)
clusters_pseudo_rgb = dpgmm.predict(features_pseudo_rgb)
np.save("dpgmm_clusters_pseudo_rgbscimk1.npy", clusters_pseudo_rgb)

# Run DP-GMM on Grayscale Features
print("Fitting DP-GMM on Grayscale Features...")
dpgmm.fit(features_grayscale)
clusters_grayscale = dpgmm.predict(features_grayscale)
np.save("dpgmm_clusters_grayscalescimk1.npy", clusters_grayscale)

print("\nClustering complete! Clusters found:")
print(f"- Green-Only: {len(set(clusters_green))} clusters")
print(f"- Pseudo-RGB: {len(set(clusters_pseudo_rgb))} clusters")
print(f"- Grayscale: {len(set(clusters_grayscale))} clusters")

# ğŸ“Š Plot Cluster Distributions
plt.figure(figsize=(10,5))
plt.hist(clusters_green, bins=50, alpha=0.6, label="Green-Only", color="green")
plt.hist(clusters_pseudo_rgb, bins=50, alpha=0.6, label="Pseudo-RGB", color="red")
plt.hist(clusters_grayscale, bins=50, alpha=0.6, label="Grayscale 3-CH", color="gray")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")
plt.title("Cluster Distributions Across Feature Representations")
plt.legend()
plt.show()


# âœ… Load features
features_green = np.load("green_channel_features.npy")
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# âœ… Check for NaNs
print("Checking for NaNs...")
features_green = np.nan_to_num(features_green)
features_pseudo_rgb = np.nan_to_num(features_pseudo_rgb)
features_grayscale = np.nan_to_num(features_grayscale)

# âœ… Standardize features
scaler = StandardScaler()
features_green = scaler.fit_transform(features_green)
features_pseudo_rgb = scaler.transform(features_pseudo_rgb)
features_grayscale = scaler.transform(features_grayscale)

# âœ… DP-GMM Model Setup
dpgmm = BayesianGaussianMixture(
    n_components=100,  # Maximum possible clusters
    weight_concentration_prior=1e-2,  # Controls sparsity
    weight_concentration_prior_type="dirichlet_process",  # âœ… Enables DP behavior
    covariance_type="full",
    reg_covar=1e-4,  # âœ… Prevent covariance collapse
    init_params="kmeans",  # âœ… Better initialization
    n_init=1,  # Single initialization for tracking
    max_iter=5,  # Run for multiple steps per update
    warm_start=True,  # âœ… Keep improving model instead of restarting
    tol=1e-3,  # âœ… Loosen tolerance for better convergence
    verbose=2,
    random_state=42
)
# âœ… Track ELBO per iteration
elbo_values = []
num_iterations = 5  # Set how many iterations you want to track manually

print("\nFitting DP-GMM on Green-Only Features with Manual Iteration Tracking...")
for i in range(num_iterations):
    dpgmm.fit(features_green)
    elbo_values.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

# âœ… Save ELBO values
np.save("dpgmm_elbo_greenmk2.npy", elbo_values)

# ğŸ“Š Plot ELBO over iterations
plt.figure(figsize=(8,5))
plt.plot(range(1, num_iterations + 1), elbo_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Over Iterations for Green-Only Features")
plt.grid()
plt.show()

# âœ… Get cluster assignments
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscimk2.npy", clusters_green)

# âœ… Repeat for Pseudo-RGB Features
print("\nFitting DP-GMM on Pseudo-RGB Features...")
elbo_values_pseudo = []
for i in range(num_iterations):
    dpgmm.fit(features_pseudo_rgb)
    elbo_values_pseudo.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_pseudomk2.npy", elbo_values_pseudo)
clusters_pseudo_rgb = dpgmm.predict(features_pseudo_rgb)
np.save("dpgmm_clusters_pseudo_rgbscimk2.npy", clusters_pseudo_rgb)

# âœ… Repeat for Grayscale Features
print("\nFitting DP-GMM on Grayscale Features...")
elbo_values_gray = []
for i in range(num_iterations):
    dpgmm.fit(features_grayscale)
    elbo_values_gray.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_graymk2.npy", elbo_values_gray)
clusters_grayscale = dpgmm.predict(features_grayscale)
np.save("dpgmm_clusters_grayscalescimk2.npy", clusters_grayscale)

# ğŸ“Š Compare Cluster Distributions
plt.figure(figsize=(10,5))
plt.hist(clusters_green, bins=50, alpha=0.6, label="Green-Only", color="green")
plt.hist(clusters_pseudo_rgb, bins=50, alpha=0.6, label="Pseudo-RGB", color="red")
plt.hist(clusters_grayscale, bins=50, alpha=0.6, label="Grayscale 3-CH", color="gray")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")
plt.title("Cluster Distributions Across Feature Representations")
plt.legend()
plt.show()


# âœ… Load features
features_green = np.load("green_channel_features.npy")
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# âœ… Check for NaNs
print("Checking for NaNs...")
features_green = np.nan_to_num(features_green)
features_pseudo_rgb = np.nan_to_num(features_pseudo_rgb)
features_grayscale = np.nan_to_num(features_grayscale)

# âœ… Standardize features
scaler = StandardScaler()
features_green = scaler.fit_transform(features_green)
features_pseudo_rgb = scaler.transform(features_pseudo_rgb)
features_grayscale = scaler.transform(features_grayscale)

# âœ… DP-GMM Model Setup
dpgmm = BayesianGaussianMixture(
    n_components=100,  # Maximum possible clusters
    weight_concentration_prior=1e-2,  # Controls sparsity
    weight_concentration_prior_type="dirichlet_process",  # âœ… Enables DP behavior
    covariance_type="full",
    reg_covar=1e-5,  # Prevents singularities
    init_params='kmeans',  # Better initialization
    n_init=1,  # Only one initialization since we're tracking iterations
    max_iter=1,  # Run only one iteration per step
    warm_start=True,  # âœ… Keeps updating the model instead of restarting
    verbose=2,
    random_state=42
)

# âœ… Track ELBO per iteration
elbo_values = []
num_iterations = 10  # Set how many iterations you want to track manually

print("\nFitting DP-GMM on Green-Only Features with Manual Iteration Tracking...")
for i in range(num_iterations):
    dpgmm.fit(features_green)
    elbo_values.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

# âœ… Save ELBO values
np.save("dpgmm_elbo_greenmk2r.npy", elbo_values)

# ğŸ“Š Plot ELBO over iterations
plt.figure(figsize=(8,5))
plt.plot(range(1, num_iterations + 1), elbo_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Over Iterations for Green-Only Features")
plt.grid()
plt.show()

# âœ… Get cluster assignments
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscimk2r.npy", clusters_green)

# âœ… Repeat for Pseudo-RGB Features
print("\nFitting DP-GMM on Pseudo-RGB Features...")
elbo_values_pseudo = []
for i in range(num_iterations):
    dpgmm.fit(features_pseudo_rgb)
    elbo_values_pseudo.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_pseudomk2r.npy", elbo_values_pseudo)
clusters_pseudo_rgb = dpgmm.predict(features_pseudo_rgb)
np.save("dpgmm_clusters_pseudo_rgbscimk2r.npy", clusters_pseudo_rgb)

# âœ… Repeat for Grayscale Features
print("\nFitting DP-GMM on Grayscale Features...")
elbo_values_gray = []
for i in range(num_iterations):
    dpgmm.fit(features_grayscale)
    elbo_values_gray.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_graymk2r.npy", elbo_values_gray)
clusters_grayscale = dpgmm.predict(features_grayscale)
np.save("dpgmm_clusters_grayscalescimk2r.npy", clusters_grayscale)

# ğŸ“Š Compare Cluster Distributions
plt.figure(figsize=(10,5))
plt.hist(clusters_green, bins=50, alpha=0.6, label="Green-Only", color="green")
plt.hist(clusters_pseudo_rgb, bins=50, alpha=0.6, label="Pseudo-RGB", color="red")
plt.hist(clusters_grayscale, bins=50, alpha=0.6, label="Grayscale 3-CH", color="gray")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")
plt.title("Cluster Distributions Across Feature Representations")
plt.legend()
plt.show()


# âœ… Load features
features_green = np.load("green_channel_features.npy")
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# âœ… Check for NaNs
print("Checking for NaNs...")
features_green = np.nan_to_num(features_green)
features_pseudo_rgb = np.nan_to_num(features_pseudo_rgb)
features_grayscale = np.nan_to_num(features_grayscale)

# âœ… Standardize features
scaler = StandardScaler()
features_green = scaler.fit_transform(features_green)
features_pseudo_rgb = scaler.transform(features_pseudo_rgb)
features_grayscale = scaler.transform(features_grayscale)

# âœ… DP-GMM Model Setup (No Restart Between Iterations)
dpgmm = BayesianGaussianMixture(
    n_components=100,  # Maximum possible clusters
    weight_concentration_prior=1e-2,  # Controls sparsity
    weight_concentration_prior_type="dirichlet_process",  # âœ… Enables DP behavior
    covariance_type="full",
    reg_covar=1e-5,  # Prevents singularities
    n_init=1,  # Single initialization for tracking
    max_iter=1,  # Run one iteration at a time
   # warm_start=True,  # âœ… Keeps improving model instead of resetting
    verbose=2,
    random_state=42
)

# âœ… Track ELBO per iteration
num_iterations = 10  # Number of manual iterations
elbo_values = []

print("\nFitting DP-GMM on Green-Only Features with Manual Iteration Tracking...")
for i in range(num_iterations):
    dpgmm.fit(features_green)
    elbo_values.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

# âœ… Save ELBO values
np.save("dpgmm_elbo_green_mk3.npy", elbo_values)

# ğŸ“Š Plot ELBO over iterations
plt.figure(figsize=(8,5))
plt.plot(range(1, num_iterations + 1), elbo_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Over Iterations for Green-Only Features")
plt.grid()
plt.show()

# âœ… Get cluster assignments
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscimk3.npy", clusters_green)

# âœ… Repeat for Pseudo-RGB Features
print("\nFitting DP-GMM on Pseudo-RGB Features...")
elbo_values_pseudo = []
for i in range(num_iterations):
    dpgmm.fit(features_pseudo_rgb)
    elbo_values_pseudo.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_pseudo_mk3.npy", elbo_values_pseudo)
clusters_pseudo_rgb = dpgmm.predict(features_pseudo_rgb)
np.save("dpgmm_clusters_pseudo_rgbscimk3.npy", clusters_pseudo_rgb)

# âœ… Repeat for Grayscale Features
print("\nFitting DP-GMM on Grayscale Features...")
elbo_values_gray = []
for i in range(num_iterations):
    dpgmm.fit(features_grayscale)
    elbo_values_gray.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_gray_mk3.npy", elbo_values_gray)
clusters_grayscale = dpgmm.predict(features_grayscale)
np.save("dpgmm_clusters_grayscalescimk3.npy", clusters_grayscale)

# ğŸ“Š Compare Cluster Distributions
plt.figure(figsize=(10,5))
plt.hist(clusters_green, bins=50, alpha=0.6, label="Green-Only", color="green")
plt.hist(clusters_pseudo_rgb, bins=50, alpha=0.6, label="Pseudo-RGB", color="red")
plt.hist(clusters_grayscale, bins=50, alpha=0.6, label="Grayscale 3-CH", color="gray")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")
plt.title("Cluster Distributions Across Feature Representations")
plt.legend()
plt.show()


# âœ… Load features
features_green = np.load("green_channel_features.npy")
features_pseudo_rgb = np.load("pseudo_rgb_features.npy")
features_grayscale = np.load("grayscale_3ch_features.npy")

# âœ… Check for NaNs
print("Checking for NaNs...")
features_green = np.nan_to_num(features_green)
features_pseudo_rgb = np.nan_to_num(features_pseudo_rgb)
features_grayscale = np.nan_to_num(features_grayscale)

# âœ… Standardize features
scaler = StandardScaler()
features_green = scaler.fit_transform(features_green)
features_pseudo_rgb = scaler.transform(features_pseudo_rgb)
features_grayscale = scaler.transform(features_grayscale)

# âœ… DP-GMM Model Setup (No Restart Between Iterations)
dpgmm = BayesianGaussianMixture(
    n_components=100,  # Maximum possible clusters
    weight_concentration_prior=1e-2,  # Controls sparsity
    weight_concentration_prior_type="dirichlet_process",  # âœ… Enables DP behavior
    covariance_type="full",
    reg_covar=1e-5,  # Prevents singularities
    n_init=1,  # Single initialization for tracking
    max_iter=1,  # Run one iteration at a time
    warm_start=True,  # âœ… Keeps improving model instead of resetting
    verbose=2,
    random_state=42
)

# âœ… Track ELBO per iteration
num_iterations = 10  # Number of manual iterations
elbo_values = []

print("\nFitting DP-GMM on Green-Only Features with Manual Iteration Tracking...")
for i in range(num_iterations):
    dpgmm.fit(features_green)
    elbo_values.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

# âœ… Save ELBO values
np.save("dpgmm_elbo_green_mk3r.npy", elbo_values)

# ğŸ“Š Plot ELBO over iterations
plt.figure(figsize=(8,5))
plt.plot(range(1, num_iterations + 1), elbo_values, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Over Iterations for Green-Only Features")
plt.grid()
plt.show()

# âœ… Get cluster assignments
clusters_green = dpgmm.predict(features_green)
np.save("dpgmm_clusters_greenscimk3r.npy", clusters_green)

# âœ… Repeat for Pseudo-RGB Features
print("\nFitting DP-GMM on Pseudo-RGB Features...")
elbo_values_pseudo = []
for i in range(num_iterations):
    dpgmm.fit(features_pseudo_rgb)
    elbo_values_pseudo.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_pseudo_mk3r.npy", elbo_values_pseudo)
clusters_pseudo_rgb = dpgmm.predict(features_pseudo_rgb)
np.save("dpgmm_clusters_pseudo_rgbscimk3r.npy", clusters_pseudo_rgb)

# âœ… Repeat for Grayscale Features
print("\nFitting DP-GMM on Grayscale Features...")
elbo_values_gray = []
for i in range(num_iterations):
    dpgmm.fit(features_grayscale)
    elbo_values_gray.append(dpgmm.lower_bound_)
    print(f"Iteration {i+1}, ELBO: {dpgmm.lower_bound_}")

np.save("dpgmm_elbo_gray_mk3r.npy", elbo_values_gray)
clusters_grayscale = dpgmm.predict(features_grayscale)
np.save("dpgmm_clusters_grayscalescimk3r.npy", clusters_grayscale)

# ğŸ“Š Compare Cluster Distributions
plt.figure(figsize=(10,5))
plt.hist(clusters_green, bins=50, alpha=0.6, label="Green-Only", color="green")
plt.hist(clusters_pseudo_rgb, bins=50, alpha=0.6, label="Pseudo-RGB", color="red")
plt.hist(clusters_grayscale, bins=50, alpha=0.6, label="Grayscale 3-CH", color="gray")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")
plt.title("Cluster Distributions Across Feature Representations")
plt.legend()
plt.show()


# âœ… Load features (Green Channel) 
features_green = np.load("green_channel_features.npy")  
# âœ… Sample a smaller subset 
subset_size = 5000 
features_green = features_green[:subset_size]  
# âœ… Standardize the features 
scaler = StandardScaler() 
features_green = scaler.fit_transform(features_green)  
# âœ… Try different values of k using the Elbow Method 
wcss = []  # Within-cluster sum of squares 
k_values = range(2, 15)  # Test k from 2 to 15  
for k in k_values:     
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)     
    kmeans.fit(features_green)     
    wcss.append(kmeans.inertia_)  # Store the sum of squared distances  
# ğŸ“Š Plot Elbow Method results 
plt.figure(figsize=(8, 5)) 
plt.plot(k_values, wcss, marker='o', linestyle='-') 
plt.xlabel("Number of Clusters (k)") 
plt.ylabel("WCSS (Within-Cluster Sum of Squares)") 
plt.title("Elbow Method for Optimal k") 
plt.grid() 
plt.show() 


# âœ… Load features 
features_pseudo_rgb = np.load("pseudo_rgb_features.npy") 
features_grayscale = np.load("grayscale_3ch_features.npy")  
# âœ… Sample a subset (to speed up computation) 
subset_size = 5000 
features_pseudo_rgb = features_pseudo_rgb[:subset_size] 
features_grayscale = features_grayscale[:subset_size]  
# âœ… Standardize the features 
scaler = StandardScaler() 
features_pseudo_rgb = scaler.fit_transform(features_pseudo_rgb) 
features_grayscale = scaler.transform(features_grayscale)  
# âœ… Try different values of k using the Elbow Method 
k_values = range(2, 15)  # Testing k from 2 to 15  
# ğŸ“Š Run K-Means for Pseudo-RGB 
wcss_pseudo = [] 
for k in k_values:     
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)     
    kmeans.fit(features_pseudo_rgb)     
    wcss_pseudo.append(kmeans.inertia_)  
# ğŸ“Š Run K-Means for Grayscale 
wcss_gray = [] 
for k in k_values:     
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)     
    kmeans.fit(features_grayscale)     
    wcss_gray.append(kmeans.inertia_)  
# âœ… Plot the Elbow Method results for both 
plt.figure(figsize=(10, 5))  
plt.subplot(1, 2, 1) 
plt.plot(k_values, wcss_pseudo, marker='o', linestyle='-') 
plt.xlabel("Number of Clusters (k)") 
plt.ylabel("WCSS (Within-Cluster Sum of Squares)") 
plt.title("Elbow Method for Pseudo-RGB Features") 
plt.grid()  
plt.subplot(1, 2, 2) 
plt.plot(k_values, wcss_gray, marker='o', linestyle='-') 
plt.xlabel("Number of Clusters (k)") 
plt.ylabel("WCSS (Within-Cluster Sum of Squares)") 
plt.title("Elbow Method for Grayscale Features") 
plt.grid()  
plt.tight_layout() 
plt.show() 


# âœ… Load features 
features_green = np.load("green_channel_features.npy") 
features_pseudo_rgb = np.load("pseudo_rgb_features.npy") 
features_grayscale = np.load("grayscale_3ch_features.npy")  
# âœ… Sample a subset (to speed up computation) 
subset_size = 5000 
features_green = features_green[:subset_size] 
features_pseudo_rgb = features_pseudo_rgb[:subset_size] 
features_grayscale = features_grayscale[:subset_size]  
# âœ… Standardize the features 
scaler = StandardScaler() 
features_green = scaler.fit_transform(features_green) 
features_pseudo_rgb = scaler.transform(features_pseudo_rgb) 
features_grayscale = scaler.transform(features_grayscale)  
# âœ… Run K-Means for Green Channel (k=7) 
kmeans_green = KMeans(n_clusters=7, random_state=42, n_init=10) 
clusters_green = kmeans_green.fit_predict(features_green) 
silhouette_green = silhouette_score(features_green, clusters_green)  
# âœ… Run K-Means for Pseudo-RGB (k=8) 
kmeans_pseudo = KMeans(n_clusters=8, random_state=42, n_init=10) 
clusters_pseudo = kmeans_pseudo.fit_predict(features_pseudo_rgb) 
silhouette_pseudo = silhouette_score(features_pseudo_rgb, clusters_pseudo)  
# âœ… Run K-Means for Grayscale (k=9) 
kmeans_gray = KMeans(n_clusters=9, random_state=42, n_init=10) 
clusters_gray = kmeans_gray.fit_predict(features_grayscale) 
silhouette_gray = silhouette_score(features_grayscale, clusters_gray)  
# âœ… Save cluster assignments 
np.save("kmeans_clusters_greentest.npy", clusters_green) 
np.save("kmeans_clusters_pseudotest.npy", clusters_pseudo) 
np.save("kmeans_clusters_graytest.npy", clusters_gray)  
# âœ… Print Silhouette Scores 
print(f"Silhouette Score (Green): {silhouette_green}") 
print(f"Silhouette Score (Pseudo-RGB): {silhouette_pseudo}") 
print(f"Silhouette Score (Grayscale): {silhouette_gray}")  
# ğŸ“Š Compare Cluster Distributions 
plt.figure(figsize=(10,5)) 
plt.hist(clusters_green, bins=7, alpha=0.6, label="Green-Only", color="green") 
plt.hist(clusters_pseudo, bins=8, alpha=0.6, label="Pseudo-RGB", color="red") 
plt.hist(clusters_gray, bins=9, alpha=0.6, label="Grayscale", color="gray") 
plt.xlabel("Cluster ID") 
plt.ylabel("Number of Samples") 
plt.title("K-Means Cluster Distributions Across Feature Representations") 
plt.legend() 
plt.show() 


features_green = np.load("green_channel_features.npy") 
features_pseudo_rgb = np.load("pseudo_rgb_features.npy") 
features_grayscale = np.load("grayscale_3ch_features.npy")  
# âœ… Sample a subset (to speed up computation) 
subset_size = 5000 
features_green = features_green[:subset_size] 
features_pseudo_rgb = features_pseudo_rgb[:subset_size] 
features_grayscale = features_grayscale[:subset_size]  
# âœ… Standardize the features (DBSCAN is distance-based, so scaling is crucial) 
scaler = StandardScaler() 
features_green = scaler.fit_transform(features_green) 
features_pseudo_rgb = scaler.transform(features_pseudo_rgb) 
features_grayscale = scaler.transform(features_grayscale)  
# âœ… Set DBSCAN parameters (Will fine-tune later if needed) 
eps_value = 0.5  # Controls neighborhood size (will adjust if needed) 
min_samples_value = 10  # Minimum points per dense region  
# âœ… Run DBSCAN for Green Channel 
dbscan_green = DBSCAN(eps=eps_value, min_samples=min_samples_value, n_jobs=-1) 
clusters_green = dbscan_green.fit_predict(features_green) 
silhouette_green = silhouette_score(features_green, clusters_green) if len(set(clusters_green)) > 1 else -1  
# âœ… Run DBSCAN for Pseudo-RGB 
dbscan_pseudo = DBSCAN(eps=eps_value, min_samples=min_samples_value, n_jobs=-1) 
clusters_pseudo = dbscan_pseudo.fit_predict(features_pseudo_rgb) 
silhouette_pseudo = silhouette_score(features_pseudo_rgb, clusters_pseudo) if len(set(clusters_pseudo)) > 1 else -1  
# âœ… Run DBSCAN for Grayscale 
dbscan_gray = DBSCAN(eps=eps_value, min_samples=min_samples_value, n_jobs=-1) 
clusters_gray = dbscan_gray.fit_predict(features_grayscale) 
silhouette_gray = silhouette_score(features_grayscale, clusters_gray) if len(set(clusters_gray)) > 1 else -1  
# âœ… Save cluster assignments 
np.save("dbscan_clusters_green.npy", clusters_green) 
np.save("dbscan_clusters_pseudo.npy", clusters_pseudo) 
np.save("dbscan_clusters_gray.npy", clusters_gray)  
# âœ… Print Silhouette Scores 
print(f"DBSCAN Silhouette Score (Green): {silhouette_green}") 
print(f"DBSCAN Silhouette Score (Pseudo-RGB): {silhouette_pseudo}") 
print(f"DBSCAN Silhouette Score (Grayscale): {silhouette_gray}")  
# ğŸ“Š Compare Cluster Distributions (excluding noise points labeled as -1) 
plt.figure(figsize=(10,5)) 
plt.hist(clusters_green[clusters_green != -1], bins=20, alpha=0.6, label="Green-Only", color="green") 
plt.hist(clusters_pseudo[clusters_pseudo != -1], bins=20, alpha=0.6, label="Pseudo-RGB", color="red") 
plt.hist(clusters_gray[clusters_gray != -1], bins=20, alpha=0.6, label="Grayscale", color="gray") 
plt.xlabel("Cluster ID") 
plt.ylabel("Number of Samples") 
plt.title("DBSCAN Cluster Distributions Across Feature Representations") 
plt.legend() 
plt.show()  

