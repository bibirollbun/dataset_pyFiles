# Step 1.1: Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Step 1.2: Define Paths
data_path = Path('/kaggle/input/image-matching-challenge-2025')
train_path = data_path / 'train'
test_path = data_path / 'test'
sample_submission_path = data_path / 'sample_submission.csv'
train_labels_path = data_path / 'train_labels.csv'
train_thresholds_path = data_path / 'train_thresholds.csv'

# Print paths to confirm
print("Data Path:", data_path)
print("Train Path:", train_path)
print("Test Path:", test_path)
print("Sample Submission Path:", sample_submission_path)
print("Train Labels Path:", train_labels_path)
print("Train Thresholds Path:", train_thresholds_path)


# Step 1.3: Explore the Directory Structure
print("Train directory contents:")
for root, dirs, files in os.walk(train_path):
    print(f"Directory: {root}")
    print(f"Number of files: {len(files)}")
    if len(files) > 0:
        print(f"Sample files: {files[:3]}")
    print()

print("Test directory contents:")
for root, dirs, files in os.walk(test_path):
    print(f"Directory: {root}")
    print(f"Number of files: {len(files)}")
    if len(files) > 0:
        print(f"Sample files: {files[:3]}")
    print()


# Step 1.4: Load and Display CSV Files
# Sample Submission
print("Sample Submission:")
sample_submission = pd.read_csv(sample_submission_path)
print(sample_submission.head())
print(f"Shape: {sample_submission.shape}")
print()

# Train Labels
print("Train Labels:")
train_labels = pd.read_csv(train_labels_path)
print(train_labels.head())
print(f"Shape: {train_labels.shape}")
print()

# Train Thresholds
print("Train Thresholds:")
train_thresholds = pd.read_csv(train_thresholds_path)
print(train_thresholds.head())
print(f"Shape: {train_thresholds.shape}")
print()


# Step 1.5: Visualize Sample Images from Train
# Get a few sample images (use .png since filenames have .png extension)
train_images = list(train_path.glob('**/*.png'))[:4]
print("Sample image paths:")
for img_path in train_images:
    print(img_path)

# Load and visualize images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for idx, img_path in enumerate(train_images):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Display image
    axes[idx].imshow(img)
    axes[idx].set_title(f"Image {idx+1}: {img_path.name}")
    axes[idx].axis('off')
    print(f"Image {idx+1} shape: {img.shape}")
plt.tight_layout()
plt.show()


# Step 1.6: Analyze the Explored Data
# Summarize the directory structure
train_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
test_dirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]

train_image_count = sum(len(files) for root, dirs, files in os.walk(train_path) if files)
test_image_count = sum(len(files) for root, dirs, files in os.walk(test_path) if files)

print("Summary of Directory Structure:")
print(f"Number of train datasets: {len(train_dirs)}")
print(f"Train datasets: {train_dirs}")
print(f"Total train images: {train_image_count}")
print(f"Number of test datasets: {len(test_dirs)}")
print(f"Test datasets: {test_dirs}")
print(f"Total test images: {test_image_count}")
print()

# Summarize CSV files
print("Summary of CSV Files:")
print("Sample Submission Columns:", list(sample_submission.columns))
print("Sample Submission Shape:", sample_submission.shape)
print("Train Labels Columns:", list(train_labels.columns))
print("Train Labels Shape:", train_labels.shape)
print("Train Thresholds Columns:", list(train_thresholds.columns))
print("Train Thresholds Shape:", train_thresholds.shape)
print()

# Summarize image characteristics
print("Image Characteristics:")
print("Sample image paths (from previous step):")
for img_path in train_images:
    print(img_path)
print("Sample image shapes (from previous step):")
print("Image 1 shape: (1024, 576, 3)")
print("Image 2 shape: (768, 1024, 3)")
print("Image 3 shape: (1024, 576, 3)")
print("Image 4 shape: (768, 1024, 3)")
print("Observation: Images are from the 'amy_gardens' dataset, depicting a garden with peach trees, supported by poles and wires, under cloudy skies.")


# Step 2.1: Define the Problem Technically
# Define the tasks
print("Task Definition:")
print("1. Clustering: Assign each test image to a scene (e.g., 'cluster0' in sample_submission.csv).")
print("   - Training data is already clustered (e.g., 'fountain' in imc2023_haiper).")
print("   - Test data requires clustering, handling outliers (e.g., 'outliers_out_et003.png').")
print("2. 3D Reconstruction: For each image, predict its camera pose (rotation_matrix, translation_vector).")
print("   - Requires image matching to find keypoint correspondences between images in the same scene.")
print("   - Use matches to estimate camera poses and reconstruct the 3D scene.")
print()

# Hypothesize the evaluation metric
print("Evaluation Metric (Hypothesized):")
print("- Clustering: Likely evaluated using Adjusted Rand Index (ARI) or purity, comparing predicted clusters to ground truth.")
print("- 3D Reconstruction: Likely evaluated using reprojection error or pose accuracy (error in rotation and translation).")
print("Note: train_thresholds.csv suggests image matching performance may be evaluated at different thresholds (e.g., mAP).")
print()

# Outline the approach
print("Approach Outline:")
print("1. Clustering:")
print("   - Extract image embeddings using a pre-trained CNN (e.g., ResNet).")
print("   - Cluster images using HDBSCAN or K-means to group them into scenes.")
print("2. 3D Reconstruction:")
print("   - Image Matching: Use LoFTR or SuperGlue to find keypoint matches between image pairs.")
print("   - Structure from Motion (SfM): Use COLMAP to estimate camera poses and reconstruct the 3D scene.")
print("   - Post-Processing: Filter matches using thresholds from train_thresholds.csv.")
print()

# Identify challenges
print("Challenges:")
print("- Varying image sizes (e.g., 1024x576, 768x1024) require resizing for consistency.")
print("- Outliers in datasets (e.g., 'outliers_out_et003.png') need robust clustering.")
print("- Diverse scenes (landmarks, heritage, natural) may require domain-specific handling.")
print("- Hidden test set may include new scenes, requiring generalization.")
print("- Notebook must run within Kaggle's time limits (typically 20 minutes).")


# Step 2.2: Extract Image Embeddings for Clustering
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode
model = model.cuda() if torch.cuda.is_available() else model  # Use GPU if available

# Remove the final fully connected layer to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 as required by ResNet
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
])

# Get all test images
test_images = list(test_path.glob('**/*.png'))
print(f"Total test images to process: {len(test_images)}")

# Extract embeddings
embeddings = []
image_paths = []

for img_path in test_images:
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Get embedding
    with torch.no_grad():
        embedding = model(img_tensor)
    embedding = embedding.cpu().numpy().flatten()  # Flatten to 1D array
    embeddings.append(embedding)
    image_paths.append(img_path)

# Convert to numpy array
embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Number of images processed: {len(image_paths)}")


# Step 2.3: Install HDBSCAN and Cluster Test Images
# Install hdbscan
!pip install hdbscan

import hdbscan

# Perform clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
cluster_labels = clusterer.fit_predict(embeddings)

# Analyze clustering results
unique_labels = np.unique(cluster_labels)
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise label (-1)
print(f"Number of clusters found: {num_clusters}")
print(f"Cluster labels: {unique_labels}")

# Count images per cluster
for label in unique_labels:
    if label == -1:
        print(f"Number of noise points (label -1): {np.sum(cluster_labels == label)}")
    else:
        print(f"Number of images in cluster {label}: {np.sum(cluster_labels == label)}")

# Map cluster labels to scene names (e.g., cluster0, cluster1, ...)
scene_mapping = {label: f"cluster{label}" for label in unique_labels if label != -1}
scene_mapping[-1] = "noise"  # Label for outliers

# Assign scene names to images
image_scenes = [scene_mapping[label] for label in cluster_labels]

# Print cluster assignments for the first few images
print("\nCluster assignments for the first 5 images:")
for i in range(min(5, len(image_paths))):
    print(f"Image: {image_paths[i].name}, Scene: {image_scenes[i]}")


# Step 2.4: Visualize Clusters Using Dimensionality Reduction
# Install umap-learn
!pip install umap-learn

import umap
import seaborn as sns

# Reduce dimensionality of embeddings to 2D using UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Create a scatter plot of the 2D embeddings
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=cluster_labels, palette="deep", style=cluster_labels, size=cluster_labels, sizes=(50, 200))
plt.title("2D Visualization of Test Image Clusters (UMAP)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster Label")
plt.show()

# Print summary of clustering results
print("Clustering Summary:")
print(f"Number of clusters found: {num_clusters}")
print(f"Total images: {len(cluster_labels)}")
for label in unique_labels:
    if label == -1:
        print(f"Number of noise points (label -1): {np.sum(cluster_labels == label)}")
    else:
        print(f"Number of images in cluster {label}: {np.sum(cluster_labels == label)}")


# Step 3.1: Correct Image Matching with LoFTR and Visualize Matches
import kornia
from kornia.feature import LoFTR
import kornia as K
import kornia.feature as KF

# Select images from the largest cluster (cluster 0)
cluster_0_indices = [i for i, label in enumerate(cluster_labels) if label == 0]
if len(cluster_0_indices) < 2:
    print("Not enough images in cluster 0 to perform matching.")
else:
    # Select the first two images from cluster 0
    img1_path = image_paths[cluster_0_indices[0]]
    img2_path = image_paths[cluster_0_indices[1]]
    
    # Load images using PIL to ensure correct RGB loading
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # Preprocess images: resize and convert to tensor
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to a smaller size for faster processing
        transforms.ToTensor(),  # Convert to tensor
    ])
    
    img1_tensor = preprocess(img1)
    img2_tensor = preprocess(img2)
    
    # Convert to grayscale for LoFTR
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)  # Should be [1, H, W]
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)  # Should be [1, H, W]
    
    # Add batch dimension explicitly
    img1_gray = img1_gray.unsqueeze(0)  # Shape: [1, 1, H, W]
    img2_gray = img2_gray.unsqueeze(0)  # Shape: [1, 1, H, W]
    
    # Print shapes to debug
    print(f"Image 1 grayscale shape: {img1_gray.shape}")
    print(f"Image 2 grayscale shape: {img2_gray.shape}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_gray = img1_gray.to(device)
    img2_gray = img2_gray.to(device)
    
    # Initialize LoFTR
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher.eval()
    
    # Prepare input for LoFTR
    input_dict = {
        "image0": img1_gray,  # Shape: [1, 1, H, W]
        "image1": img2_gray   # Shape: [1, 1, H, W]
    }
    
    # Perform matching
    with torch.no_grad():
        correspondences = matcher(input_dict)
    
    # Extract keypoints and matches
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    print(f"Number of matches found: {len(mkpts0)}")
    
    # Load images for visualization using OpenCV
    img1_np = cv2.imread(str(img1_path))
    img2_np = cv2.imread(str(img2_path))
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)
    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_BGR2RGB)
    
    # Resize images for visualization to match the input size to LoFTR
    img1_np = cv2.resize(img1_np, (640, 640))
    img2_np = cv2.resize(img2_np, (640, 640))
    
    # Create a combined image to draw matches
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1_np
    combined_img[:h2, w1:w1+w2] = img2_np
    
    # Draw matches
    for (x1, y1), (x2, y2) in zip(mkpts0, mkpts1):
        x2_shifted = x2 + w1  # Shift x-coordinate for the second image
        cv2.circle(combined_img, (int(x1), int(y1)), 5, (0, 255, 0), 2)
        cv2.circle(combined_img, (int(x2_shifted), int(y2)), 5, (0, 255, 0), 2)
        cv2.line(combined_img, (int(x1), int(y1)), (int(x2_shifted), int(y2)), (255, 0, 0), 1)
    
    # Display the result
    plt.figure(figsize=(15, 5))
    plt.imshow(combined_img)
    plt.title(f"Keypoint Matches Between {img1_path.name} and {img2_path.name}")
    plt.axis('off')
    plt.show()


# Install pycolmap if not already installed
!pip install pycolmap

# Step 3.2: Adjust SfM Options for PyCOLMAP and Visualize 3D Point Cloud
import pycolmap
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Create a directory for COLMAP workspace
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# Copy images from cluster 0 to the COLMAP image directory
cluster_0_paths = [image_paths[i] for i in cluster_0_indices]
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# Set up paths for COLMAP
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# Create IncrementalPipelineOptions object and set attributes
options = pycolmap.IncrementalPipelineOptions()

# Inspect available attributes (for debugging)
print("Available attributes in IncrementalPipelineOptions:")
print(dir(options))

# Set available options (adjust based on available attributes)
options.min_num_matches = 15
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False
# Instead of init_min_num_inliers, we can use min_model_size or similar if available
if hasattr(options, 'min_model_size'):
    options.min_model_size = 10  # Minimum number of inliers for initial model
else:
    print("min_model_size not available, proceeding with default options.")

# Run the full COLMAP pipeline: feature extraction, matching, and SfM
try:
    # Extract features
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO
    )
    
    # Perform exhaustive matching
    pycolmap.match_exhaustive(str(database_path))
    
    # Run incremental mapping (SfM)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    
    # Since incremental_mapping returns a dict of reconstructions, get the first one
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]  # Take the first reconstruction
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        # Extract 3D points for visualization
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)
        
        # Visualize the 3D point cloud
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', marker='o')
        ax.set_title("3D Point Cloud of Cluster 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        
        # Store camera poses for submission
        camera_poses = {}
        image_names = [f"image_{i}.png" for i in range(len(cluster_0_paths))]
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            rotation_matrix = img.qvec2rotmat()
            translation_vector = img.tvec
            img_name = image_names[img_id-1]
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)
    else:
        print("Reconstruction failed: No reconstructions returned.")
except Exception as e:
    print(f"Reconstruction failed: {e}")


# Step 3.3: Use COLMAP’s High-Level Pipeline with Relaxed Constraints and Visualize 3D Reconstruction
import pycolmap

# Create a directory for COLMAP workspace
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# Copy images from cluster 0 to the COLMAP image directory
cluster_0_paths = [image_paths[i] for i in cluster_0_indices]
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# Set up paths for COLMAP
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# Create IncrementalPipelineOptions object and set attributes
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 5    # Lowered to allow more matches
options.min_model_size = 3     # Lowered to allow smaller initial models
options.init_num_trials = 2000  # Increased to allow more attempts at initialization
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

# Run the full COLMAP pipeline: feature extraction, matching, and SfM
try:
    # Extract features
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO
    )
    
    # Perform exhaustive matching
    pycolmap.match_exhaustive(str(database_path))
    
    # Run incremental mapping (SfM)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    
    # Since incremental_mapping returns a dict of reconstructions, get the first one
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]  # Take the first reconstruction
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        # Extract 3D points for visualization
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)
        
        # Extract camera positions for visualization
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            # Compute camera position as -R^T * t
            rotation = img.rotmat()  # Use rotmat() to get the rotation matrix
            translation = img.tvec
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)
        
        # Visualize the 3D point cloud and camera positions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot 3D points
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', marker='o', label='3D Points')
        # Plot camera positions
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions of Cluster 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
        
        # Store camera poses for submission
        camera_poses = {}
        image_names = [f"image_{i}.png" for i in range(len(cluster_0_paths))]
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            rotation_matrix = img.rotmat()  # Use rotmat() to get the rotation matrix
            translation_vector = img.tvec
            img_name = image_names[img_id-1]
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)
    else:
        print("Reconstruction failed: No reconstructions returned.")
except Exception as e:
    print(f"Reconstruction failed: {e}")


# Step 3.4: Fix Feature Extraction Argument and Visualize 3D Reconstruction
import pycolmap

# Create a directory for COLMAP workspace
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# Copy images from cluster 0 to the COLMAP image directory
cluster_0_paths = [image_paths[i] for i in cluster_0_indices]
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# Set up paths for COLMAP
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# Create SiftExtractionOptions to adjust feature extraction
sift_options = pycolmap.SiftExtractionOptions()
sift_options.peak_threshold = 0.01  # Lower threshold to extract more features
sift_options.max_num_features = 8192  # Increase the maximum number of features

# Create IncrementalPipelineOptions object and set attributes
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 5    # Lowered to allow more matches
options.min_model_size = 3     # Lowered to allow smaller initial models
options.init_num_trials = 2000  # Increased to allow more attempts at initialization
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

# Run the full COLMAP pipeline: feature extraction, matching, and SfM
try:
    # Extract features with adjusted options
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_options  # Fixed argument name
    )
    
    # Perform exhaustive matching
    pycolmap.match_exhaustive(str(database_path))
    
    # Run incremental mapping (SfM)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    
    # Since incremental_mapping returns a dict of reconstructions, get the first one
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]  # Take the first reconstruction
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        # Extract 3D points for visualization
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)
        
        # Extract camera positions for visualization
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            # Compute camera position as -R^T * t
            rotation = img.rotation  # Use rotation attribute to get the rotation matrix
            translation = img.tvec
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)
        
        # Visualize the 3D point cloud and camera positions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot 3D points
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', marker='o', label='3D Points')
        # Plot camera positions
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions of Cluster 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
        
        # Store camera poses for submission
        camera_poses = {}
        image_names = [f"image_{i}.png" for i in range(len(cluster_0_paths))]
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            rotation_matrix = img.rotation  # Use rotation attribute to get the rotation matrix
            translation_vector = img.tvec
            img_name = image_names[img_id-1]
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)
    else:
        print("Reconstruction failed: No reconstructions returned.")
except Exception as e:
    print(f"Reconstruction failed: {e}")


import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Define function to convert quaternion to rotation matrix
def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1**2 - 2*q2**2]
    ])

# Step 3.5: Fix Matching Argument, Relax Matching Constraints, and Visualize 3D Reconstruction
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# Copy a subset of images from cluster 0 to the COLMAP image directory (to speed up debugging)
cluster_0_paths = [image_paths[i] for i in cluster_0_indices[:10]]  # Use only the first 10 images
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# Set up paths for COLMAP
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# Create SiftExtractionOptions to adjust feature extraction
sift_extraction_options = pycolmap.SiftExtractionOptions()
sift_extraction_options.peak_threshold = 0.01  # Lower threshold to extract more features
sift_extraction_options.max_num_features = 8192  # Increase the maximum number of features

# Create SiftMatchingOptions and set attributes to relax matching constraints
sift_options = pycolmap.SiftMatchingOptions()
sift_options.max_ratio = 0.9  # Relaxed to allow more matches (default is 0.8)
sift_options.max_distance = 0.8  # Relaxed to allow more matches (default is 0.7)
sift_options.cross_check = True  # Keep cross-check enabled for robustness

# Create TwoViewGeometryOptions to relax geometric verification
verification_options = pycolmap.TwoViewGeometryOptions()
verification_options.min_num_inliers = 5  # Lowered to allow more pairs to be considered valid

# Create IncrementalPipelineOptions object and set attributes
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 2    # Further lowered to allow more matches
options.min_model_size = 2     # Further lowered to allow smaller initial models
options.init_num_trials = 10000  # Further increased to allow more attempts at initialization
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

# Run the full COLMAP pipeline: feature extraction, matching, and SfM
try:
    # Extract features with adjusted options
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_extraction_options
    )
    
    # Perform exhaustive matching with adjusted options
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        sift_options=sift_options,
        verification_options=verification_options
    )
    
    # Run incremental mapping (SfM)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    
    # Since incremental_mapping returns a dict of reconstructions, get the first one
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]  # Take the first reconstruction
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        # Extract 3D points for visualization
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)
        
        # Extract camera positions for visualization
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            # Compute camera position as -R^T * t
            rotation = qvec_to_rotmat(img.qvec)  # Use qvec_to_rotmat() to get the rotation matrix
            translation = img.tvec
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)
        
        # Visualize the 3D point cloud and camera positions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot 3D points
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', marker='o', label='3D Points')
        # Plot camera positions
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions of Cluster 0 (First 10 Images)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
        
        # Store camera poses for submission
        camera_poses = {}
        image_names = [f"image_{i}.png" for i in range(len(cluster_0_paths))]
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            rotation_matrix = qvec_to_rotmat(img.qvec)  # Use qvec_to_rotmat() to get the rotation matrix
            translation_vector = img.tvec
            img_name = image_names[img_id-1]
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)
    else:
        print("Reconstruction failed: No reconstructions returned.")
except Exception as e:
    print(f"Reconstruction failed: {e}")



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pycolmap

# === Setup COLMAP workspace ===
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Select and copy a subset of cluster 0 images ===
cluster_0_paths = [image_paths[i] for i in cluster_0_indices[:10]]  # Use only first 10 images
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Define database and output paths ===
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# === Feature extraction options ===
sift_extraction_options = pycolmap.SiftExtractionOptions()
sift_extraction_options.peak_threshold = 0.01
sift_extraction_options.max_num_features = 8192

# === Feature matching options ===
sift_options = pycolmap.SiftMatchingOptions()
sift_options.max_ratio = 0.9
sift_options.max_distance = 0.8
sift_options.cross_check = True

# === Geometric verification options ===
verification_options = pycolmap.TwoViewGeometryOptions()
verification_options.min_num_inliers = 5

# === Incremental pipeline options ===
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 2
options.min_model_size = 2
options.init_num_trials = 10000
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

# === Run the COLMAP SfM pipeline ===
try:
    # Step 1: Extract features
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_extraction_options
    )

    # Step 2: Match features
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        sift_options=sift_options,
        verification_options=verification_options
    )

    # Step 3: Run SfM (incremental mapping)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )

    if reconstructions:
        reconstruction = list(reconstructions.values())[0]
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")

        # === Debug one camera pose ===
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation = cam_from_world.rotation
            print(f"Available attributes in Rotation3d object:")
            print(dir(rotation))
            translation = cam_from_world.translation
            print(f"Type of translation: {type(translation)}")
            break  # Only debug the first image

        # === Extract 3D points for visualization ===
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)

        # === Extract camera positions ===
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)

        # === Visualize the 3D point cloud and camera positions ===
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', label='3D Points')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions (Cluster 0 - First 10 Images)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

        # === Store camera poses ===
        camera_poses = {}
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation_matrix = cam_from_world.rotation.matrix()
            translation_vector = cam_from_world.translation
            img_name = img.name  # ✅ Use image name from pycolmap
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)

    else:
        print("Reconstruction failed: No reconstructions returned.")

except Exception as e:
    print(f"Reconstruction failed: {e}")



# Step 3.7: Fix Image Visualization and Rerun SfM with Sequential Matching
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# === Setup COLMAP workspace ===
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Select and copy a subset of cluster 0 images ===
cluster_0_paths = [image_paths[i] for i in cluster_0_indices[:10]]  # Use only first 10 images
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    saved_path = image_dir / f"image_{i}.png"
    img.save(saved_path)
    print(f"Saved image {i} to: {saved_path}")

# === Visualize the first 10 images ===
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i in range(10):
    img_path = image_dir / f"image_{i}.png"
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)  # Convert PIL image to NumPy array for matplotlib
        axes[i].imshow(img_array)
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')
    except Exception as e:
        print(f"Failed to load image {i}: {e}")
        axes[i].set_title(f"Image {i} (Failed)")
        axes[i].axis('off')
plt.tight_layout()
plt.show()

# === Define database and output paths ===
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# === Feature extraction options ===
sift_extraction_options = pycolmap.SiftExtractionOptions()
sift_extraction_options.peak_threshold = 0.01
sift_extraction_options.max_num_features = 8192

# === Feature matching options ===
sift_options = pycolmap.SiftMatchingOptions()
sift_options.max_ratio = 0.9
sift_options.max_distance = 0.8
sift_options.cross_check = True

# === Geometric verification options ===
verification_options = pycolmap.TwoViewGeometryOptions()
verification_options.min_num_inliers = 3  # Further lowered to allow more pairs

# === Incremental pipeline options ===
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 2
options.min_model_size = 2
options.init_num_trials = 10000
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

# === Run the COLMAP SfM pipeline ===
try:
    # Step 1: Extract features
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_extraction_options
    )

    # Step 2: Match features using sequential matching
    pycolmap.match_sequential(
        database_path=str(database_path),
        sift_options=sift_options,
        verification_options=verification_options
    )

    # Step 3: Run SfM (incremental mapping)
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )

    if reconstructions:
        reconstruction = list(reconstructions.values())[0]
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")

        # === Extract 3D points for visualization ===
        points3d = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
        points3d = np.array(points3d)

        # === Extract camera positions ===
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)

        # === Visualize the 3D point cloud and camera positions ===
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', label='3D Points')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions (Cluster 0 - First 10 Images)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

        # === Store camera poses ===
        camera_poses = {}
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation_matrix = cam_from_world.rotation.matrix()
            translation_vector = cam_from_world.translation
            img_name = img.name
            camera_poses[img_name] = (rotation_matrix.flatten(), translation_vector)

    else:
        print("Reconstruction failed: No reconstructions returned.")

except Exception as e:
    print(f"Reconstruction failed: {e}")


!pip install torch torchvision
!pip install opencv-python


!git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
!wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth -P SuperGluePretrainedNetwork/models/weights/


import sys
sys.path.append('SuperGluePretrainedNetwork')


# Step 3.9: Use SuperPoint and SuperGlue with COLMAP Command-Line Integration (Retry)
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Install COLMAP in Kaggle environment ===
print("Installing COLMAP dependencies...")
try:
    # Install required packages
    subprocess.run("apt-get update", shell=True, check=True)
    subprocess.run("apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev libceres-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev", shell=True, check=True)
    
    # Download and install COLMAP
    print("Downloading and installing COLMAP...")
    subprocess.run("git clone https://github.com/colmap/colmap.git /kaggle/working/colmap-repo", shell=True, check=True)
    subprocess.run("mkdir -p /kaggle/working/colmap-repo/build", shell=True, check=True)
    subprocess.run("cmake .. -DCMAKE_BUILD_TYPE=Release", shell=True, check=True, cwd="/kaggle/working/colmap-repo/build")
    subprocess.run("make -j$(nproc)", shell=True, check=True, cwd="/kaggle/working/colmap-repo/build")
    subprocess.run("make install", shell=True, check=True, cwd="/kaggle/working/colmap-repo/build")
    
    # Verify installation
    result = subprocess.run("colmap --version", shell=True, capture_output=True, text=True)
    print("COLMAP version:", result.stdout)
    if result.returncode != 0:
        raise Exception("COLMAP installation failed.")
except Exception as e:
    print(f"Error installing COLMAP: {e}")
    print("Falling back to default COLMAP installation method...")
    subprocess.run("apt-get install -y colmap", shell=True, check=True)
    result = subprocess.run("colmap --version", shell=True, capture_output=True, text=True)
    print("COLMAP version:", result.stdout)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Define cluster_0_indices (select first 5 images) ===
cluster_0_indices = list(range(min(5, len(image_paths))))
print(f"Cluster 0 indices: {cluster_0_indices}")

# === Setup COLMAP workspace ===
colmap_dir = Path('/kaggle/working/colmap')
colmap_dir.mkdir(exist_ok=True)
image_dir = colmap_dir / 'images'
image_dir.mkdir(exist_ok=True)
features_dir = colmap_dir / 'features'
features_dir.mkdir(exist_ok=True)
matches_dir = colmap_dir / 'matches'
matches_dir.mkdir(exist_ok=True)

# === Select and preprocess a subset of cluster 0 images ===
cluster_0_paths = [image_paths[i] for i in cluster_0_indices]
print("Paths in cluster_0_paths:")
for path in cluster_0_paths:
    print(path)

for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    r, g, b = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)
    img_clahe = cv2.merge([r_clahe, g_clahe, b_clahe])
    img_pil = Image.fromarray(img_clahe)
    img_pil.save(image_dir / f"image_{i}.png")

# === Visualize the preprocessed images ===
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes = axes.flatten()
for i, img_path in enumerate(cluster_0_paths):
    img = Image.open(image_dir / f"image_{i}.png")
    axes[i].imshow(img)
    axes[i].set_title(f"Preprocessed Image {i}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
plt.close('all')

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

# === Define database and output paths ===
database_path = colmap_dir / 'database.db'
output_path = colmap_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# === Step 1: Initialize the COLMAP database with images ===
sift_extraction_options = pycolmap.SiftExtractionOptions()
sift_extraction_options.peak_threshold = 0.001
sift_extraction_options.edge_threshold = 20
sift_extraction_options.max_num_features = 16384

start_time = time.time()
pycolmap.extract_features(
    database_path=str(database_path),
    image_path=str(image_dir),
    camera_mode=pycolmap.CameraMode.AUTO,
    sift_options=sift_extraction_options
)
print(f"Feature extraction (initialization) took {time.time() - start_time:.2f} seconds.")

# === Step 2: Run SuperPoint and SuperGlue for feature detection and matching ===
keypoints_dict = {}
descriptors_dict = {}
matches_dict = {}

start_time = time.time()
image_pairs = [(i, j) for i in range(len(cluster_0_paths)) for j in range(i + 1, len(cluster_0_paths))]
for idx0, idx1 in image_pairs:
    img0 = cv2.imread(str(image_dir / f"image_{idx0}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(image_dir / f"image_{idx1}.png"), cv2.IMREAD_GRAYSCALE)
    
    img0 = cv2.resize(img0, (640, 480))
    img1 = cv2.resize(img1, (640, 480))
    
    inp0 = frame2tensor(img0, device)
    inp1 = frame2tensor(img1, device)
    
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    match_conf = conf[valid]
    
    img0_name = f"image_{idx0}.png"
    img1_name = f"image_{idx1}.png"
    if img0_name not in keypoints_dict:
        keypoints_dict[img0_name] = kpts0
        descriptors_dict[img0_name] = np.zeros((len(kpts0), 128), dtype=np.float32)
    if img1_name not in keypoints_dict:
        keypoints_dict[img1_name] = kpts1
        descriptors_dict[img1_name] = np.zeros((len(kpts1), 128), dtype=np.float32)
    
    matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1, match_conf)
    
    if idx0 == 0 and idx1 == 1:
        color = np.random.randint(0, 255, (len(mkpts0), 3)) / 255.0
        text = [
            f'SuperGlue',
            f'Keypoints: {len(kpts0)}:{len(kpts1)}',
            f'Matches: {len(mkpts0)}'
        ]
        plt.figure(figsize=(12, 6))
        make_matching_plot(img0, img1, kpts0, kpts1, mkpts0, mkpts1, color, text, path=None, show_keypoints=True)
        fig = plt.gcf()
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        plt.figure(figsize=(12, 6))
        plt.imshow(plot_img)
        plt.axis('off')
        plt.show()
        plt.close('all')

print(f"SuperGlue matching took {time.time() - start_time:.2f} seconds.")

# === Step 3: Save SuperPoint keypoints to text files ===
for img_name in keypoints_dict:
    keypoints = keypoints_dict[img_name]
    descriptors = descriptors_dict[img_name]
    keypoints_colmap = np.zeros((len(keypoints), 4), dtype=np.float32)
    keypoints_colmap[:, :2] = keypoints
    keypoints_colmap[:, 2] = 1.0
    keypoints_colmap[:, 3] = 0.0
    
    feature_file = features_dir / f"{img_name}.txt"
    with open(feature_file, 'w') as f:
        f.write(f"{len(keypoints)} 128\n")
        for kp, desc in zip(keypoints_colmap, descriptors):
            f.write(f"{kp[0]} {kp[1]} {kp[2]} {kp[3]} {' '.join(map(str, desc))}\n")

# === Step 4: Save SuperGlue matches to a text file ===
matches_file = matches_dir / "matches.txt"
with open(matches_file, 'w') as f:
    for (img0_name, img1_name), (mkpts0, mkpts1, match_conf) in matches_dict.items():
        kpts0 = keypoints_dict[img0_name]
        kpts1 = keypoints_dict[img1_name]
        matches = []
        for i, (mkpt0, mkpt1) in enumerate(zip(mkpts0, mkpts1)):
            idx0 = np.where((kpts0 == mkpt0).all(axis=1))[0][0]
            idx1 = np.where((kpts1 == mkpt1).all(axis=1))[0][0]
            matches.append((idx0, idx1))
        if matches:
            f.write(f"{img0_name} {img1_name}\n")
            for idx0, idx1 in matches:
                f.write(f"{idx0} {idx1}\n")
            f.write("\n")

# === Step 5: Import features and matches using COLMAP command-line tools ===
start_time = time.time()
try:
    subprocess.run([
        "colmap", "feature_importer",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--import_path", str(features_dir)
    ], check=True)
    
    subprocess.run([
        "colmap", "matches_importer",
        "--database_path", str(database_path),
        "--match_list_path", str(matches_file),
        "--match_type", "pairs"
    ], check=True)
    print(f"Importing features and matches took {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error importing features and matches: {e}")
    print("Falling back to default COLMAP matching in the next step...")

# === Step 6: Incremental mapping ===
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 1
options.min_model_size = 2
options.init_num_trials = 5000
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False

start_time = time.time()
reconstructions = pycolmap.incremental_mapping(
    database_path=str(database_path),
    image_path=str(image_dir),
    output_path=str(output_path),
    options=options
)
print(f"Incremental mapping took {time.time() - start_time:.2f} seconds.")

if reconstructions:
    reconstruction = list(reconstructions.values())[0]
    print(f"Number of images registered: {reconstruction.num_reg_images()}")
    print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")

    points3d = []
    for point3d_id in reconstruction.points3D:
        point3d = reconstruction.points3D[point3d_id]
        points3d.append(point3d.xyz)
    points3d = np.array(points3d)

    camera_positions = []
    for img_id in reconstruction.images:
        img = reconstruction.images[img_id]
        cam_from_world = img.cam_from_world
        rotation = cam_from_world.rotation.matrix()
        translation = cam_from_world.translation
        position = -np.dot(rotation.T, translation)
        camera_positions.append(position)
    camera_positions = np.array(camera_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='b', label='3D Points')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title("3D Point Cloud and Camera Positions (Cluster 0 - First 5 Images)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')
else:
    print("Reconstruction failed: No reconstructions returned.")


# Step 3.10 (Simplified): Custom SfM Pipeline with SuperGlue Matches (5 Images, Single Cluster, No Bundle Adjustment)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Select first 5 images ===
num_images = 5
selected_indices = list(range(min(num_images, len(image_paths))))
print(f"Selected indices: {selected_indices}")

# === Define a single cluster ===
clusters = [selected_indices]  # Single cluster with 5 images
print(f"Clusters: {clusters}")

# === Setup workspace ===
work_dir = Path('/kaggle/working/custom_sfm')
work_dir.mkdir(exist_ok=True)
image_dir = work_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Copy all selected images ===
selected_paths = [image_paths[i] for i in selected_indices]
for i, img_path in enumerate(selected_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 2048  # Reduced to speed up processing
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05  # Increased to reduce matches
    }
}
matching = Matching(config).eval().to(device)

# === Function to run SfM on a single cluster ===
def run_sfm_on_cluster(cluster_indices, image_dir, matching, device):
    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    
    print(f"Processing cluster with indices: {cluster_indices}")
    
    # Run SuperPoint and SuperGlue
    start_time = time.time()
    image_pairs = [(i, j) for i in range(len(cluster_indices)) for j in range(i + 1, len(cluster_indices))]
    edges = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0 = cv2.imread(str(image_dir / f"image_{global_idx0}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(image_dir / f"image_{global_idx1}.png"), cv2.IMREAD_GRAYSCALE)
        
        orig_size0 = img0.shape[::-1]
        orig_size1 = img1.shape[::-1]
        
        img0 = cv2.resize(img0, (640, 480))
        img1 = cv2.resize(img1, (640, 480))
        
        inp0 = frame2tensor(img0, device)
        inp1 = frame2tensor(img1, device)
        
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
        scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
        kpts0[:, 0] *= scale0[0]
        kpts0[:, 1] *= scale0[1]
        kpts1[:, 0] *= scale1[0]
        kpts1[:, 1] *= scale1[1]
        mkpts0[:, 0] *= scale0[0]
        mkpts0[:, 1] *= scale0[1]
        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        if img0_name not in keypoints_dict:
            keypoints_dict[img0_name] = kpts0
            image_sizes[img0_name] = orig_size0
        if img1_name not in keypoints_dict:
            keypoints_dict[img1_name] = kpts1
            image_sizes[img1_name] = orig_size1
        
        matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
        
        edges.append((len(mkpts0), idx0, idx1))
        
        print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")
    
    print(f"SuperGlue matching for cluster took {time.time() - start_time:.2f} seconds.")
    
    # Sort edges by number of matches (descending)
    edges.sort(key=lambda x: -x[0])
    parent = list(range(len(cluster_indices)))
    rank = [0] * len(cluster_indices)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    selected_pairs = []
    for num_matches, idx0, idx1 in edges:
        if num_matches < 8:
            continue
        if find(idx0) != find(idx1):
            union(idx0, idx1)
            selected_pairs.append((idx0, idx1))
    
    if not selected_pairs:
        print("No pairs with sufficient matches found in cluster.")
        return np.array([]), np.array([]), np.array([]), {}
    
    # Initialize poses with the first selected pair
    idx0, idx1 = selected_pairs[0]
    global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
    img0_name = f"image_{global_idx0}.png"
    img1_name = f"image_{global_idx1}.png"
    
    mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
    mkpts0 = mkpts0.astype(np.float32)
    mkpts1 = mkpts1.astype(np.float32)
    
    # Estimate focal length
    image_width, image_height = image_sizes[img0_name]
    focal_length = max(image_width, image_height) * 1.2
    cx, cy = image_width / 2, image_height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        print(f"Failed to estimate essential matrix for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    if R is None or t is None:
        print(f"Failed to recover pose for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}
    
    R = R.astype(np.float32)
    t = t.astype(np.float32).flatten()
    
    # Initialize poses
    poses = {img0_name: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    poses[img1_name] = (R, t)
    registered_indices = {idx0, idx1}
    
    # Process remaining pairs
    start_time = time.time()
    for idx0, idx1 in selected_pairs[1:]:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name in poses and img1_name in poses:
            continue
        elif img0_name not in poses and img1_name not in poses:
            continue
        
        if img1_name in poses and img0_name not in poses:
            img0_name, img1_name = img1_name, img0_name
            idx0, idx1 = idx1, idx0
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name})")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            print(f"Failed to estimate essential matrix between {img0_name} and {img1_name}")
            continue
        
        _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
        if R is None or t is None:
            print(f"Failed to recover pose between {img0_name} and {img1_name}")
            continue
        
        R = R.astype(np.float32)
        t = t.astype(np.float32).flatten()
        
        R0, t0 = poses[img0_name]
        R = R0 @ R
        t = R0 @ t + t0
        
        poses[img1_name] = (R, t)
        registered_indices.add(idx1)
    
    print(f"Pose estimation took {time.time() - start_time:.2f} seconds.")
    
    # Triangulate 3D points
    start_time = time.time()
    points3d = []
    colors = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name not in poses or img1_name not in poses:
            continue
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name}) during triangulation")
            continue
        
        if len(mkpts0) != len(mkpts1):
            print(f"Mismatch in number of points between {img0_name} and {img1_name}: {len(mkpts0)} vs {len(mkpts1)}")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        R0, t0 = poses[img0_name]
        R1, t1 = poses[img1_name]
        
        P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P0 = P0.astype(np.float32)
        P1 = P1.astype(np.float32)
        
        pts0 = mkpts0.T.astype(np.float32)
        pts1 = mkpts1.T.astype(np.float32)
        
        if pts0.shape[1] == 0 or pts1.shape[1] == 0:
            print(f"Skipping triangulation due to zero matches between {img0_name} and {img1_name}")
            continue
        
        points4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
        points3d_h = points4d[:3] / points4d[3]
        points3d.extend(points3d_h.T)
        
        img0 = cv2.imread(str(image_dir / img0_name))
        img0 = cv2.resize(img0, image_sizes[img0_name])
        for pt in mkpts0:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img0.shape[1] and 0 <= y < img0.shape[0]:
                colors.append(img0[y, x] / 255.0)
            else:
                colors.append([0, 0, 0])
    
    points3d = np.array(points3d)
    colors = np.array(colors)
    
    # Remove invalid points
    valid = np.all(np.isfinite(points3d), axis=1) & (np.abs(points3d) < 1e5).all(axis=1)
    points3d = points3d[valid]
    colors = colors[valid]
    
    print(f"Triangulation took {time.time() - start_time:.2f} seconds.")
    
    # Camera positions
    camera_positions = []
    for img_name in poses:
        R, t = poses[img_name]
        pos = -R.T @ t
        camera_positions.append(pos)
    camera_positions = np.array(camera_positions)
    
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1, label='3D Points')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title(f"3D Point Cloud and Camera Positions (Cluster {cluster_indices[0]}-{cluster_indices[-1]})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')
    
    print(f"Cluster {cluster_indices[0]}-{cluster_indices[-1]}: {len(points3d)} 3D points, {len(poses)} cameras")
    
    return points3d, colors, camera_positions, poses

# === Process the single cluster ===
all_points3d = []
all_colors = []
all_camera_positions = []
all_poses = {}

start_time = time.time()
for cluster_idx, cluster_indices in enumerate(clusters):
    points3d, colors, camera_positions, poses = run_sfm_on_cluster(cluster_indices, image_dir, matching, device)
    all_points3d.append(points3d)
    all_colors.append(colors)
    all_camera_positions.append(camera_positions)
    all_poses.update(poses)

# Since there's only one cluster, no merging is needed
points3d = all_points3d[0]
colors = all_colors[0]
camera_positions = all_camera_positions[0]

print(f"Total runtime: {time.time() - start_time:.2f} seconds")
print(f"Total number of 3D points reconstructed: {len(points3d)}")
print(f"Total number of cameras estimated: {len(camera_positions)}")


# Step 3.10 (Step 2): Custom SfM Pipeline with SuperGlue Matches (10 Images, 2 Clusters, No Bundle Adjustment)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Select first 10 images ===
num_images = 10
selected_indices = list(range(min(num_images, len(image_paths))))
print(f"Selected indices: {selected_indices}")

# === Define clusters (5 images per cluster, with overlap of 2 images) ===
cluster_size = 5
overlap = 2
clusters = []
for i in range(0, len(selected_indices), cluster_size - overlap):
    cluster_indices = selected_indices[i:i + cluster_size]
    if len(cluster_indices) >= 2:
        clusters.append(cluster_indices)
print(f"Clusters: {clusters}")

# === Setup workspace ===
work_dir = Path('/kaggle/working/custom_sfm')
work_dir.mkdir(exist_ok=True)
image_dir = work_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Copy all selected images ===
selected_paths = [image_paths[i] for i in selected_indices]
for i, img_path in enumerate(selected_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05
    }
}
matching = Matching(config).eval().to(device)

# === Function to run SfM on a single cluster ===
def run_sfm_on_cluster(cluster_indices, image_dir, matching, device, cluster_idx, clusters):
    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    
    print(f"Processing cluster with indices: {cluster_indices}")
    
    # Run SuperPoint and SuperGlue
    start_time = time.time()
    image_pairs = [(i, j) for i in range(len(cluster_indices)) for j in range(i + 1, len(cluster_indices))]
    edges = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0 = cv2.imread(str(image_dir / f"image_{global_idx0}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(image_dir / f"image_{global_idx1}.png"), cv2.IMREAD_GRAYSCALE)
        
        orig_size0 = img0.shape[::-1]
        orig_size1 = img1.shape[::-1]
        
        img0 = cv2.resize(img0, (640, 480))
        img1 = cv2.resize(img1, (640, 480))
        
        inp0 = frame2tensor(img0, device)
        inp1 = frame2tensor(img1, device)
        
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
        scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
        kpts0[:, 0] *= scale0[0]
        kpts0[:, 1] *= scale0[1]
        kpts1[:, 0] *= scale1[0]
        kpts1[:, 1] *= scale1[1]
        mkpts0[:, 0] *= scale0[0]
        mkpts0[:, 1] *= scale0[1]
        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        if img0_name not in keypoints_dict:
            keypoints_dict[img0_name] = kpts0
            image_sizes[img0_name] = orig_size0
        if img1_name not in keypoints_dict:
            keypoints_dict[img1_name] = kpts1
            image_sizes[img1_name] = orig_size1
        
        matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
        
        priority = 0
        if cluster_idx < len(clusters) - 1:
            overlap_indices = clusters[cluster_idx][-overlap:]
            if global_idx0 in overlap_indices or global_idx1 in overlap_indices:
                priority = 1
        
        edges.append((len(mkpts0), idx0, idx1, priority))
        
        print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")
    
    print(f"SuperGlue matching for cluster took {time.time() - start_time:.2f} seconds.")
    
    # Sort edges
    edges.sort(key=lambda x: (-x[3], -x[0]))
    parent = list(range(len(cluster_indices)))
    rank = [0] * len(cluster_indices)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    selected_pairs = []
    for num_matches, idx0, idx1, _ in edges:
        if num_matches < 8:
            continue
        if find(idx0) != find(idx1):
            union(idx0, idx1)
            selected_pairs.append((idx0, idx1))
    
    if cluster_idx < len(clusters) - 1:
        overlap_indices = clusters[cluster_idx][-overlap:]
        for overlap_idx in overlap_indices:
            overlap_local_idx = cluster_indices.index(overlap_idx)
            if overlap_local_idx not in {idx0 for idx0, _ in selected_pairs} and overlap_local_idx not in {idx1 for _, idx1 in selected_pairs}:
                best_pair = None
                best_num_matches = 0
                for num_matches, idx0, idx1, _ in edges:
                    if num_matches < 8:
                        continue
                    if idx0 == overlap_local_idx or idx1 == overlap_local_idx:
                        if num_matches > best_num_matches:
                            best_num_matches = num_matches
                            best_pair = (idx0, idx1)
                if best_pair:
                    idx0, idx1 = best_pair
                    if find(idx0) != find(idx1):
                        union(idx0, idx1)
                        selected_pairs.append((idx0, idx1))
    
    if not selected_pairs:
        print("No pairs with sufficient matches found in cluster.")
        return np.array([]), np.array([]), np.array([]), {}
    
    # Initialize poses
    idx0, idx1 = selected_pairs[0]
    global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
    img0_name = f"image_{global_idx0}.png"
    img1_name = f"image_{global_idx1}.png"
    
    mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
    mkpts0 = mkpts0.astype(np.float32)
    mkpts1 = mkpts1.astype(np.float32)
    
    image_width, image_height = image_sizes[img0_name]
    focal_length = max(image_width, image_height) * 1.2
    cx, cy = image_width / 2, image_height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        print(f"Failed to estimate essential matrix for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}
    
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    if R is None or t is None:
        print(f"Failed to recover pose for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}
    
    R = R.astype(np.float32)
    t = t.astype(np.float32).flatten()
    
    poses = {img0_name: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    poses[img1_name] = (R, t)
    registered_indices = {idx0, idx1}
    
    # Process remaining pairs
    start_time = time.time()
    for idx0, idx1 in selected_pairs[1:]:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name in poses and img1_name in poses:
            continue
        elif img0_name not in poses and img1_name not in poses:
            continue
        
        if img1_name in poses and img0_name not in poses:
            img0_name, img1_name = img1_name, img0_name
            idx0, idx1 = idx1, idx0
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name})")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            print(f"Failed to estimate essential matrix between {img0_name} and {img1_name}")
            continue
        
        _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
        if R is None or t is None:
            print(f"Failed to recover pose between {img0_name} and {img1_name}")
            continue
        
        R = R.astype(np.float32)
        t = t.astype(np.float32).flatten()
        
        R0, t0 = poses[img0_name]
        R = R0 @ R
        t = R0 @ t + t0
        
        poses[img1_name] = (R, t)
        registered_indices.add(idx1)
    
    print(f"Pose estimation took {time.time() - start_time:.2f} seconds.")
    
    # Triangulate 3D points
    start_time = time.time()
    points3d = []
    colors = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name not in poses or img1_name not in poses:
            continue
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name}) during triangulation")
            continue
        
        if len(mkpts0) != len(mkpts1):
            print(f"Mismatch in number of points between {img0_name} and {img1_name}: {len(mkpts0)} vs {len(mkpts1)}")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        R0, t0 = poses[img0_name]
        R1, t1 = poses[img1_name]
        
        P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P0 = P0.astype(np.float32)
        P1 = P1.astype(np.float32)
        
        pts0 = mkpts0.T.astype(np.float32)
        pts1 = mkpts1.T.astype(np.float32)
        
        if pts0.shape[1] == 0 or pts1.shape[1] == 0:
            print(f"Skipping triangulation due to zero matches between {img0_name} and {img1_name}")
            continue
        
        points4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
        points3d_h = points4d[:3] / points4d[3]
        points3d.extend(points3d_h.T)
        
        img0 = cv2.imread(str(image_dir / img0_name))
        img0 = cv2.resize(img0, image_sizes[img0_name])
        for pt in mkpts0:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img0.shape[1] and 0 <= y < img0.shape[0]:
                colors.append(img0[y, x] / 255.0)
            else:
                colors.append([0, 0, 0])
    
    points3d = np.array(points3d)
    colors = np.array(colors)
    
    valid = np.all(np.isfinite(points3d), axis=1) & (np.abs(points3d) < 1e5).all(axis=1)
    points3d = points3d[valid]
    colors = colors[valid]
    
    print(f"Triangulation took {time.time() - start_time:.2f} seconds.")
    
    # Camera positions
    camera_positions = []
    for img_name in poses:
        R, t = poses[img_name]
        pos = -R.T @ t
        camera_positions.append(pos)
    camera_positions = np.array(camera_positions)
    
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1, label='3D Points')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title(f"3D Point Cloud and Camera Positions (Cluster {cluster_indices[0]}-{cluster_indices[-1]})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')
    
    print(f"Cluster {cluster_indices[0]}-{cluster_indices[-1]}: {len(points3d)} 3D points, {len(poses)} cameras")
    
    return points3d, colors, camera_positions, poses

# === Process each cluster ===
all_points3d = []
all_colors = []
all_camera_positions = []
all_poses = {}

start_time = time.time()
for cluster_idx, cluster_indices in enumerate(clusters):
    points3d, colors, camera_positions, poses = run_sfm_on_cluster(cluster_indices, image_dir, matching, device, cluster_idx, clusters)
    all_points3d.append(points3d)
    all_colors.append(colors)
    all_camera_positions.append(camera_positions)
    all_poses.update(poses)

# Merge reconstructions
if not all_points3d or len(all_points3d[0]) == 0:
    print("No points reconstructed in the first cluster. Cannot proceed with merging.")
else:
    merged_points3d = all_points3d[0]
    merged_colors = all_colors[0]
    merged_camera_positions = all_camera_positions[0]

    for i in range(1, len(clusters)):
        overlap_image = f"image_{clusters[i-1][-1]}.png"
        if overlap_image not in all_poses:
            print(f"Cannot align clusters {i-1} and {i} using overlapping image {overlap_image}.")
            continue
        
        R0_prev, t0_prev = all_poses[overlap_image]
        R0_curr, t0_curr = all_poses[overlap_image]
        
        R_align = R0_prev @ np.linalg.inv(R0_curr)
        t_align = t0_prev - R_align @ t0_curr
        
        points3d_curr = all_points3d[i]
        if len(points3d_curr) == 0:
            print(f"Cluster {i} has no points to merge")
            continue
        points3d_curr = (R_align @ points3d_curr.T).T + t_align
        all_points3d[i] = points3d_curr
        merged_points3d = np.vstack((merged_points3d, points3d_curr))
        merged_colors = np.vstack((merged_colors, all_colors[i]))
        
        camera_positions_curr = all_camera_positions[i]
        if len(camera_positions_curr) == 0:
            print(f"Cluster {i} has no camera positions to merge")
            continue
        camera_positions_curr = (R_align @ camera_positions_curr.T).T + t_align
        all_camera_positions[i] = camera_positions_curr
        merged_camera_positions = np.vstack((merged_camera_positions, camera_positions_curr))
        
        for img_name in list(all_poses.keys()):
            if img_name in all_poses:
                R, t = all_poses[img_name]
                R = R_align @ R
                t = R_align @ t + t_align
                all_poses[img_name] = (R, t)

    # Remove duplicates in camera positions
    unique_camera_positions = []
    seen_images = set()
    for i, pos in enumerate(merged_camera_positions):
        img_idx = i % len(selected_indices)
        img_name = f"image_{img_idx}.png"
        if img_name not in seen_images:
            unique_camera_positions.append(pos)
            seen_images.add(img_name)
    unique_camera_positions = np.array(unique_camera_positions)

    # Visualize merged reconstruction
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(merged_points3d[:, 0], merged_points3d[:, 1], merged_points3d[:, 2], c=merged_colors, s=1, label='3D Points')
    ax.scatter(unique_camera_positions[:, 0], unique_camera_positions[:, 1], unique_camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title("Merged 3D Point Cloud and Camera Positions (First 10 Images)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    print(f"Total number of 3D points reconstructed: {len(merged_points3d)}")
    print(f"Total number of cameras estimated: {len(unique_camera_positions)}")


# Step 3.10 (Step 3): Custom SfM Pipeline with SuperGlue Matches and COLMAP Bundle Adjustment (10 Images)
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Select first 10 images ===
num_images = 10
selected_indices = list(range(min(num_images, len(image_paths))))
print(f"Selected indices: {selected_indices}")

# === Setup workspace ===
work_dir = Path('/kaggle/working/colmap')
work_dir.mkdir(exist_ok=True)
image_dir = work_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Copy all selected images ===
selected_paths = [image_paths[i] for i in selected_indices]
for i, img_path in enumerate(selected_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05
    }
}
matching = Matching(config).eval().to(device)

# === Run SuperPoint and SuperGlue ===
keypoints_dict = {}
matches_dict = {}
image_sizes = {}

start_time = time.time()
image_pairs = [(i, j) for i in range(len(selected_indices)) for j in range(i + 1, len(selected_indices))]
for idx0, idx1 in image_pairs:
    img0 = cv2.imread(str(image_dir / f"image_{idx0}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(image_dir / f"image_{idx1}.png"), cv2.IMREAD_GRAYSCALE)
    
    orig_size0 = img0.shape[::-1]
    orig_size1 = img1.shape[::-1]
    
    img0 = cv2.resize(img0, (640, 480))
    img1 = cv2.resize(img1, (640, 480))
    
    inp0 = frame2tensor(img0, device)
    inp1 = frame2tensor(img1, device)
    
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
    scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
    kpts0[:, 0] *= scale0[0]
    kpts0[:, 1] *= scale0[1]
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]
    mkpts0[:, 0] *= scale0[0]
    mkpts0[:, 1] *= scale0[1]
    mkpts1[:, 0] *= scale1[0]
    mkpts1[:, 1] *= scale1[1]
    
    img0_name = f"image_{idx0}.png"
    img1_name = f"image_{idx1}.png"
    if img0_name not in keypoints_dict:
        keypoints_dict[img0_name] = kpts0
        image_sizes[img0_name] = orig_size0
    if img1_name not in keypoints_dict:
        keypoints_dict[img1_name] = kpts1
        image_sizes[img1_name] = orig_size1
    
    matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
    
    print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")

print(f"SuperGlue matching took {time.time() - start_time:.2f} seconds.")

# === Setup COLMAP database ===
database_path = work_dir / 'database.db'
output_path = work_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# === Import features and matches into COLMAP ===
try:
    db = pycolmap.Database(str(database_path))
    
    camera_ids = {}
    for img_name in keypoints_dict:
        width, height = image_sizes[img_name]
        focal_length = max(width, height) * 1.2
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length, width / 2, height / 2]
        )
        camera_id = db.add_camera(camera)
        camera_ids[img_name] = camera_id
    
    image_ids = {}
    for img_name in keypoints_dict:
        image_id = db.add_image(img_name, camera_ids[img_name])
        image_ids[img_name] = image_id
    
    for img_name, kpts in keypoints_dict.items():
        image_id = image_ids[img_name]
        keypoints = np.zeros((len(kpts), 4), dtype=np.float32)
        keypoints[:, :2] = kpts
        keypoints[:, 2] = 1.0  # Dummy sigma
        db.add_keypoints(image_id, keypoints)
    
    for (img0_name, img1_name), (mkpts0, mkpts1) in matches_dict.items():
        image_id0 = image_ids[img0_name]
        image_id1 = image_ids[img1_name]
        kpts0 = keypoints_dict[img0_name]
        kpts1 = keypoints_dict[img1_name]
        
        matches = []
        for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
            idx0 = np.where((kpts0 == pt0).all(axis=1))[0]
            idx1 = np.where((kpts1 == pt1).all(axis=1))[0]
            if len(idx0) == 1 and len(idx1) == 1:
                matches.append((idx0[0], idx1[0]))
        matches = np.array(matches, dtype=np.uint32)
        if len(matches) > 0:
            db.add_matches(image_id0, image_id1, matches)
    
    db.commit()
    db.close()
    
    # === Run COLMAP SfM ===
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 8
    options.min_model_size = 2
    options.init_num_trials = 50000
    
    start_time = time.time()
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    print(f"Incremental mapping (with bundle adjustment) took {time.time() - start_time:.2f} seconds.")
    
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        points3d = []
        colors = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
            colors.append(point3d.color / 255.0)
        points3d = np.array(points3d)
        colors = np.array(colors)
        
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1, label='3D Points')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions (First 10 Images)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
        plt.close('all')
    else:
        print("Reconstruction failed: No reconstructions returned.")

except AttributeError as e:
    print(f"pycolmap does not support importing features and matches: {e}")
    print("Please proceed without bundle adjustment or use a different environment with COLMAP support.")


# Step 3.10 (Step 3): Custom SfM Pipeline with SuperGlue Matches and COLMAP Bundle Adjustment (10 Images)
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Select first 10 images ===
num_images = 10
selected_indices = list(range(min(num_images, len(image_paths))))
print(f"Selected indices: {selected_indices}")

# === Setup workspace ===
work_dir = Path('/kaggle/working/colmap')
work_dir.mkdir(exist_ok=True)
image_dir = work_dir / 'images'
image_dir.mkdir(exist_ok=True)

# === Copy all selected images ===
selected_paths = [image_paths[i] for i in selected_indices]
for i, img_path in enumerate(selected_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05
    }
}
matching = Matching(config).eval().to(device)

# === Run SuperPoint and SuperGlue ===
keypoints_dict = {}
matches_dict = {}
image_sizes = {}

start_time = time.time()
image_pairs = [(i, j) for i in range(len(selected_indices)) for j in range(i + 1, len(selected_indices))]
for idx0, idx1 in image_pairs:
    img0 = cv2.imread(str(image_dir / f"image_{idx0}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(image_dir / f"image_{idx1}.png"), cv2.IMREAD_GRAYSCALE)
    
    orig_size0 = img0.shape[::-1]
    orig_size1 = img1.shape[::-1]
    
    img0 = cv2.resize(img0, (640, 480))
    img1 = cv2.resize(img1, (640, 480))
    
    inp0 = frame2tensor(img0, device)
    inp1 = frame2tensor(img1, device)
    
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
    scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
    kpts0[:, 0] *= scale0[0]
    kpts0[:, 1] *= scale0[1]
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]
    mkpts0[:, 0] *= scale0[0]
    mkpts0[:, 1] *= scale0[1]
    mkpts1[:, 0] *= scale1[0]
    mkpts1[:, 1] *= scale1[1]
    
    img0_name = f"image_{idx0}.png"
    img1_name = f"image_{idx1}.png"
    if img0_name not in keypoints_dict:
        keypoints_dict[img0_name] = kpts0
        image_sizes[img0_name] = orig_size0
    if img1_name not in keypoints_dict:
        keypoints_dict[img1_name] = kpts1
        image_sizes[img1_name] = orig_size1
    
    matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
    
    print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")

print(f"SuperGlue matching took {time.time() - start_time:.2f} seconds.")

# === Setup COLMAP database ===
database_path = work_dir / 'database.db'
output_path = work_dir / 'sparse'
output_path.mkdir(exist_ok=True)

# === Import features and matches into COLMAP ===
try:
    db = pycolmap.Database(str(database_path))
    
    camera_ids = {}
    for img_name in keypoints_dict:
        width, height = image_sizes[img_name]
        focal_length = max(width, height) * 1.2
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length, width / 2, height / 2]
        )
        camera_id = db.add_camera(camera)
        camera_ids[img_name] = camera_id
    
    image_ids = {}
    for img_name in keypoints_dict:
        image_id = db.add_image(img_name, camera_ids[img_name])
        image_ids[img_name] = image_id
    
    for img_name, kpts in keypoints_dict.items():
        image_id = image_ids[img_name]
        keypoints = np.zeros((len(kpts), 4), dtype=np.float32)
        keypoints[:, :2] = kpts
        keypoints[:, 2] = 1.0  # Dummy sigma
        db.add_keypoints(image_id, keypoints)
    
    for (img0_name, img1_name), (mkpts0, mkpts1) in matches_dict.items():
        image_id0 = image_ids[img0_name]
        image_id1 = image_ids[img1_name]
        kpts0 = keypoints_dict[img0_name]
        kpts1 = keypoints_dict[img1_name]
        
        matches = []
        for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
            idx0 = np.where((kpts0 == pt0).all(axis=1))[0]
            idx1 = np.where((kpts1 == pt1).all(axis=1))[0]
            if len(idx0) == 1 and len(idx1) == 1:
                matches.append((idx0[0], idx1[0]))
        matches = np.array(matches, dtype=np.uint32)
        if len(matches) > 0:
            db.add_matches(image_id0, image_id1, matches)
    
    db.commit()
    db.close()
    
    # === Run COLMAP SfM ===
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 8
    options.min_model_size = 2
    options.init_num_trials = 50000
    
    start_time = time.time()
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=options
    )
    print(f"Incremental mapping (with bundle adjustment) took {time.time() - start_time:.2f} seconds.")
    
    if reconstructions:
        reconstruction = list(reconstructions.values())[0]
        print(f"Number of images registered: {reconstruction.num_reg_images()}")
        print(f"Number of 3D points reconstructed: {reconstruction.num_points3D()}")
        
        points3d = []
        colors = []
        for point3d_id in reconstruction.points3D:
            point3d = reconstruction.points3D[point3d_id]
            points3d.append(point3d.xyz)
            colors.append(point3d.color / 255.0)
        points3d = np.array(points3d)
        colors = np.array(colors)
        
        camera_positions = []
        for img_id in reconstruction.images:
            img = reconstruction.images[img_id]
            cam_from_world = img.cam_from_world
            rotation = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation
            position = -np.dot(rotation.T, translation)
            camera_positions.append(position)
        camera_positions = np.array(camera_positions)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1, label='3D Points')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
        ax.set_title("3D Point Cloud and Camera Positions (First 10 Images)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
        plt.close('all')
    else:
        print("Reconstruction failed: No reconstructions returned.")

except AttributeError as e:
    print(f"pycolmap does not support importing features and matches: {e}")
    print("Please proceed without bundle adjustment or use a different environment with COLMAP support.")


# Step 5: Custom SfM Pipeline for Image Matching Challenge 2025 (All Images, Format Output for Submission)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import torch
import sys
import warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Define image paths ===
stairs_dir = Path('/kaggle/input/image-matching-challenge-2025/train/stairs')
image_paths = [str(stairs_dir / img) for img in os.listdir(stairs_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images in stairs directory: {len(image_paths)}")
print("First few image paths:")
for path in image_paths[:5]:
    print(path)

# === Select all images ===
num_images = len(image_paths)
selected_indices = list(range(num_images))
print(f"Selected indices: {selected_indices}")

# === Define clusters (5 images per cluster, with overlap of 2 images) ===
cluster_size = 5
overlap = 2
clusters = []
for i in range(0, len(selected_indices), cluster_size - overlap):
    cluster_indices = selected_indices[i:i + cluster_size]
    if len(cluster_indices) >= 2:
        clusters.append(cluster_indices)
print(f"Clusters: {clusters}")

# === Setup workspace ===
work_dir = Path('/kaggle/working/custom_sfm')
work_dir.mkdir(exist_ok=True)
image_dir = work_dir / 'images'
image_dir.mkdir(exist_ok=True)
output_dir = work_dir / 'output'
output_dir.mkdir(exist_ok=True)

# === Copy all selected images ===
selected_paths = [image_paths[i] for i in selected_indices]
for i, img_path in enumerate(selected_paths):
    img = Image.open(img_path).convert('RGB')
    img.save(image_dir / f"image_{i}.png")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.03
    }
}
matching = Matching(config).eval().to(device)

# === Function to run SfM on a single cluster ===
def run_sfm_on_cluster(cluster_indices, image_dir, matching, device, cluster_idx, clusters):
    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    point_to_images = []  # Track which images see each 3D point
    
    print(f"Processing cluster with indices: {cluster_indices}")
    
    # Run SuperPoint and SuperGlue
    start_time = time.time()
    image_pairs = [(i, j) for i in range(len(cluster_indices)) for j in range(i + 1, len(cluster_indices))]
    edges = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0 = cv2.imread(str(image_dir / f"image_{global_idx0}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(image_dir / f"image_{global_idx1}.png"), cv2.IMREAD_GRAYSCALE)
        
        orig_size0 = img0.shape[::-1]
        orig_size1 = img1.shape[::-1]
        
        img0 = cv2.resize(img0, (640, 480))
        img1 = cv2.resize(img1, (640, 480))
        
        inp0 = frame2tensor(img0, device)
        inp1 = frame2tensor(img1, device)
        
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
        scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
        kpts0[:, 0] *= scale0[0]
        kpts0[:, 1] *= scale0[1]
        kpts1[:, 0] *= scale1[0]
        kpts1[:, 1] *= scale1[1]
        mkpts0[:, 0] *= scale0[0]
        mkpts0[:, 1] *= scale0[1]
        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        if img0_name not in keypoints_dict:
            keypoints_dict[img0_name] = kpts0
            image_sizes[img0_name] = orig_size0
        if img1_name not in keypoints_dict:
            keypoints_dict[img1_name] = kpts1
            image_sizes[img1_name] = orig_size1
        
        matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
        
        priority = 0
        if cluster_idx < len(clusters) - 1:
            overlap_indices = clusters[cluster_idx][-overlap:]
            if global_idx0 in overlap_indices or global_idx1 in overlap_indices:
                priority = 1
        
        edges.append((len(mkpts0), idx0, idx1, priority))
        
        print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")
    
    print(f"SuperGlue matching for cluster took {time.time() - start_time:.2f} seconds.")
    
    # Sort edges
    edges.sort(key=lambda x: (-x[3], -x[0]))
    parent = list(range(len(cluster_indices)))
    rank = [0] * len(cluster_indices)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    selected_pairs = []
    for num_matches, idx0, idx1, _ in edges:
        if num_matches < 8:
            continue
        if find(idx0) != find(idx1):
            union(idx0, idx1)
            selected_pairs.append((idx0, idx1))
    
    if cluster_idx < len(clusters) - 1:
        overlap_indices = clusters[cluster_idx][-overlap:]
        for overlap_idx in overlap_indices:
            overlap_local_idx = cluster_indices.index(overlap_idx)
            if overlap_local_idx not in {idx0 for idx0, _ in selected_pairs} and overlap_local_idx not in {idx1 for _, idx1 in selected_pairs}:
                best_pair = None
                best_num_matches = 0
                for num_matches, idx0, idx1, _ in edges:
                    if num_matches < 8:
                        continue
                    if idx0 == overlap_local_idx or idx1 == overlap_local_idx:
                        if num_matches > best_num_matches:
                            best_num_matches = num_matches
                            best_pair = (idx0, idx1)
                if best_pair:
                    idx0, idx1 = best_pair
                    if find(idx0) != find(idx1):
                        union(idx0, idx1)
                        selected_pairs.append((idx0, idx1))
    
    if not selected_pairs:
        print("No pairs with sufficient matches found in cluster.")
        return np.array([]), np.array([]), np.array([]), {}, [], {}
    
    # Initialize poses
    idx0, idx1 = selected_pairs[0]
    global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
    img0_name = f"image_{global_idx0}.png"
    img1_name = f"image_{global_idx1}.png"
    
    mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
    mkpts0 = mkpts0.astype(np.float32)
    mkpts1 = mkpts1.astype(np.float32)
    
    image_width, image_height = image_sizes[img0_name]
    focal_length = max(image_width, image_height) * 1.2
    cx, cy = image_width / 2, image_height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        print(f"Failed to estimate essential matrix for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}, [], {}
    
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    if R is None or t is None:
        print(f"Failed to recover pose for initial pair {img0_name} and {img1_name}")
        return np.array([]), np.array([]), np.array([]), {}, [], {}
    
    R = R.astype(np.float32)
    t = t.astype(np.float32).flatten()
    
    poses = {img0_name: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    poses[img1_name] = (R, t)
    registered_indices = {idx0, idx1}
    
    # Process remaining pairs
    start_time = time.time()
    for idx0, idx1 in selected_pairs[1:]:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name in poses and img1_name in poses:
            continue
        elif img0_name not in poses and img1_name not in poses:
            continue
        
        if img1_name in poses and img0_name not in poses:
            img0_name, img1_name = img1_name, img0_name
            idx0, idx1 = idx1, idx0
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name})")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            print(f"Failed to estimate essential matrix between {img0_name} and {img1_name}")
            continue
        
        _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
        if R is None or t is None:
            print(f"Failed to recover pose between {img0_name} and {img1_name}")
            continue
        
        R = R.astype(np.float32)
        t = t.astype(np.float32).flatten()
        
        R0, t0 = poses[img0_name]
        R = R0 @ R
        t = R0 @ t + t0
        
        poses[img1_name] = (R, t)
        registered_indices.add(idx1)
    
    print(f"Pose estimation took {time.time() - start_time:.2f} seconds.")
    
    # Triangulate 3D points
    start_time = time.time()
    points3d = []
    colors = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = f"image_{global_idx0}.png"
        img1_name = f"image_{global_idx1}.png"
        
        if img0_name not in poses or img1_name not in poses:
            continue
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name}) during triangulation")
            continue
        
        if len(mkpts0) != len(mkpts1):
            print(f"Mismatch in number of points between {img0_name} and {img1_name}: {len(mkpts0)} vs {len(mkpts1)}")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        R0, t0 = poses[img0_name]
        R1, t1 = poses[img1_name]
        
        P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P0 = P0.astype(np.float32)
        P1 = P1.astype(np.float32)
        
        pts0 = mkpts0.T.astype(np.float32)
        pts1 = mkpts1.T.astype(np.float32)
        
        if pts0.shape[1] == 0 or pts1.shape[1] == 0:
            print(f"Skipping triangulation due to zero matches between {img0_name} and {img1_name}")
            continue
        
        points4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
        points3d_h = points4d[:3] / points4d[3]
        points3d.extend(points3d_h.T)
        
        for _ in range(pts0.shape[1]):
            point_to_images.append([(img0_name, pts0[:, _]), (img1_name, pts1[:, _])])
        
        img0 = cv2.imread(str(image_dir / img0_name))
        img0 = cv2.resize(img0, image_sizes[img0_name])
        for pt in mkpts0:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img0.shape[1] and 0 <= y < img0.shape[0]:
                colors.append(img0[y, x] / 255.0)
            else:
                colors.append([0, 0, 0])
    
    points3d = np.array(points3d)
    colors = np.array(colors)
    
    valid = np.all(np.isfinite(points3d), axis=1) & (np.abs(points3d) < 1e5).all(axis=1)
    points3d = points3d[valid]
    colors = colors[valid]
    point_to_images = [pt for pt, v in zip(point_to_images, valid) if v]
    
    print(f"Triangulation took {time.time() - start_time:.2f} seconds.")
    
    # Lightweight Bundle Adjustment
    start_time = time.time()
    def project(points3d, R, t, K):
        points3d = points3d.T
        points = R @ points3d + t.reshape(3, 1)
        points = K @ points
        points = points[:2] / points[2]
        return points.T
    
    def reprojection_error(params, points3d, observations, K, img_names):
        num_cameras = len(img_names)
        num_points = len(points3d)
        
        # Unpack parameters (only optimize translations and a single focal length)
        translations = params[:num_cameras * 3].reshape(num_cameras, 3)
        focal_length = params[-1]
        K_opt = K.copy()
        K_opt[0, 0] = K_opt[1, 1] = focal_length
        
        errors = []
        for i, img_name in enumerate(img_names):
            R, _ = poses[img_name]
            t = translations[i]
            for j, obs in enumerate(observations):
                for img_obs, pt in obs:
                    if img_obs == img_name:
                        proj = project(points3d[j:j+1], R, t, K_opt)
                        errors.append(proj[0] - pt)
        
        return np.concatenate(errors)
    
    # Pack parameters for optimization (only optimize translations and focal length)
    img_names = list(poses.keys())
    translations = np.array([t for _, t in poses.values()])
    params = np.hstack((translations.ravel(), focal_length))
    
    # Run bundle adjustment with limited iterations
    result = least_squares(
        reprojection_error,
        params,
        args=(points3d, point_to_images, K, img_names),
        max_nfev=10,
        ftol=1e-4,
        xtol=1e-4
    )
    optimized_params = result.x
    
    # Unpack optimized parameters
    num_cameras = len(img_names)
    translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
    for i, img_name in enumerate(img_names):
        R, _ = poses[img_name]
        poses[img_name] = (R, translations[i])
    
    print(f"Bundle adjustment took {time.time() - start_time:.2f} seconds.")
    
    # Camera positions
    camera_positions = []
    for img_name in poses:
        R, t = poses[img_name]
        pos = -R.T @ t
        camera_positions.append(pos)
    camera_positions = np.array(camera_positions)
    
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1, label='3D Points')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title(f"3D Point Cloud and Camera Positions (Cluster {cluster_indices[0]}-{cluster_indices[-1]})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')
    
    print(f"Cluster {cluster_indices[0]}-{cluster_indices[-1]}: {len(points3d)} 3D points, {len(poses)} cameras")
    
    return points3d, colors, camera_positions, poses, point_to_images, K

# === Process each cluster ===
all_points3d = []
all_colors = []
all_camera_positions = []
all_poses = {}
all_point_to_images = []
all_Ks = {}

start_time = time.time()
for cluster_idx, cluster_indices in enumerate(clusters):
    points3d, colors, camera_positions, poses, point_to_images, K = run_sfm_on_cluster(cluster_indices, image_dir, matching, device, cluster_idx, clusters)
    all_points3d.append(points3d)
    all_colors.append(colors)
    all_camera_positions.append(camera_positions)
    all_poses.update(poses)
    all_point_to_images.append(point_to_images)
    all_Ks.update({img_name: K for img_name in poses})

# Merge reconstructions
if not all_points3d or len(all_points3d[0]) == 0:
    print("No points reconstructed in the first cluster. Cannot proceed with merging.")
else:
    merged_points3d = all_points3d[0]
    merged_colors = all_colors[0]
    merged_camera_positions = all_camera_positions[0]
    merged_point_to_images = all_point_to_images[0]

    for i in range(1, len(clusters)):
        overlap_image = f"image_{clusters[i-1][-1]}.png"
        if overlap_image not in all_poses:
            print(f"Cannot align clusters {i-1} and {i} using overlapping image {overlap_image}.")
            continue
        
        R0_prev, t0_prev = all_poses[overlap_image]
        R0_curr, t0_curr = all_poses[overlap_image]
        
        R_align = R0_prev @ np.linalg.inv(R0_curr)
        t_align = t0_prev - R_align @ t0_curr
        
        points3d_curr = all_points3d[i]
        if len(points3d_curr) == 0:
            print(f"Cluster {i} has no points to merge")
            continue
        points3d_curr = (R_align @ points3d_curr.T).T + t_align
        all_points3d[i] = points3d_curr
        merged_points3d = np.vstack((merged_points3d, points3d_curr))
        merged_colors = np.vstack((merged_colors, all_colors[i]))
        merged_point_to_images.extend(all_point_to_images[i])
        
        camera_positions_curr = all_camera_positions[i]
        if len(camera_positions_curr) == 0:
            print(f"Cluster {i} has no camera positions to merge")
            continue
        camera_positions_curr = (R_align @ camera_positions_curr.T).T + t_align
        all_camera_positions[i] = camera_positions_curr
        merged_camera_positions = np.vstack((merged_camera_positions, camera_positions_curr))
        
        for img_name in list(all_poses.keys()):
            if img_name in all_poses:
                R, t = all_poses[img_name]
                R = R_align @ R
                t = R_align @ t + t_align
                all_poses[img_name] = (R, t)

    # Remove duplicates in camera positions
    unique_camera_positions = []
    seen_images = set()
    for i, pos in enumerate(merged_camera_positions):
        img_idx = i % len(selected_indices)
        img_name = f"image_{img_idx}.png"
        if img_name not in seen_images:
            unique_camera_positions.append(pos)
            seen_images.add(img_name)
    unique_camera_positions = np.array(unique_camera_positions)

    # Visualize merged reconstruction
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(merged_points3d[:, 0], merged_points3d[:, 1], merged_points3d[:, 2], c=merged_colors, s=1, label='3D Points')
    ax.scatter(unique_camera_positions[:, 0], unique_camera_positions[:, 1], unique_camera_positions[:, 2], s=50, c='r', marker='^', label='Cameras')
    ax.set_title("Merged 3D Point Cloud and Camera Positions (All Images)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close('all')

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    print(f"Total number of 3D points reconstructed: {len(merged_points3d)}")
    print(f"Total number of cameras estimated: {len(unique_camera_positions)}")

    # === Save output in COLMAP format ===
    # cameras.txt
    with open(output_dir / 'cameras.txt', 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(all_poses)))
        for i, img_name in enumerate(sorted(all_poses.keys())):
            width, height = all_Ks[img_name][0, 2] * 2, all_Ks[img_name][1, 2] * 2
            focal_length = all_Ks[img_name][0, 0]
            cx, cy = all_Ks[img_name][0, 2], all_Ks[img_name][1, 2]
            f.write(f"{i+1} SIMPLE_PINHOLE {int(width)} {int(height)} {focal_length} {cx} {cy}\n")

    # images.txt
    with open(output_dir / 'images.txt', 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(all_poses)))
        for i, img_name in enumerate(sorted(all_poses.keys())):
            R, t = all_poses[img_name]
            # Convert rotation matrix to quaternion
            w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
            if w < 1e-10:
                w = 1e-10
            x = (R[2, 1] - R[1, 2]) / (4 * w)
            y = (R[0, 2] - R[2, 0]) / (4 * w)
            z = (R[1, 0] - R[0, 1]) / (4 * w)
            f.write(f"{i+1} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} {i+1} {img_name}\n")
            # Add 2D-3D correspondences (simplified, assuming we track them)
            f.write("\n")  # We'll leave POINTS2D empty for now; can be added if needed

    # points3D.txt
    with open(output_dir / 'points3D.txt', 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(merged_points3d)))
        for i, (point, color) in enumerate(zip(merged_points3d, merged_colors)):
            rgb = (color * 255).astype(int)
            f.write(f"{i+1} {point[0]} {point[1]} {point[2]} {rgb[0]} {rgb[1]} {rgb[2]} 1.0\n")


# Step 6: Process All Scenes in Test Set and Generate Submission for Image Matching Challenge 2025
import numpy as np
from pathlib import Path
import cv2
import time
import torch
import sys
import warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot)

# === Function to save point cloud as PLY ===
def save_ply(points3d, output_path):
    header = f"""ply
format ascii 1.0
element vertex {len(points3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    colors = np.random.randint(0, 255, size=(len(points3d), 3))
    with open(output_path, 'w') as f:
        f.write(header)
        for point, color in zip(points3d, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Initialize SuperGlue ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.003,
        'max_keypoints': 8192  # Increased
    },
    'superglue': {
        'weights': 'indoor',  # Test 'outdoor' if needed
        'sinkhorn_iterations': 20,
        'match_threshold': 0.01
    }
}
matching = Matching(config).eval().to(device)
print("Loaded SuperPoint model")
print("Loaded SuperGlue model (\"indoor\" weights)")

# === Function to run SfM on a single cluster ===
def run_sfm_on_cluster(cluster_indices, image_dir, image_paths, matching, device, cluster_idx, clusters):
    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    point_to_images = []
    
    print(f"Processing cluster with indices: {cluster_indices}")
    
    start_time = time.time()
    image_pairs = [(i, j) for i in range(len(cluster_indices)) for j in range(i + 1, len(cluster_indices))]
    edges = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0 = cv2.imread(image_paths[global_idx0], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(image_paths[global_idx1], cv2.IMREAD_GRAYSCALE)
        
        # Apply histogram equalization
        img0 = cv2.equalizeHist(img0)
        img1 = cv2.equalizeHist(img1)
        
        orig_size0 = img0.shape[::-1]
        orig_size1 = img1.shape[::-1]
        
        img0 = cv2.resize(img0, (640, 480))
        img1 = cv2.resize(img1, (640, 480))
        
        inp0 = frame2tensor(img0, device)
        inp1 = frame2tensor(img1, device)
        
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
        scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
        kpts0[:, 0] *= scale0[0]
        kpts0[:, 1] *= scale0[1]
        kpts1[:, 0] *= scale1[0]
        kpts1[:, 1] *= scale1[1]
        mkpts0[:, 0] *= scale0[0]
        mkpts0[:, 1] *= scale0[1]
        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        if img0_name not in keypoints_dict:
            keypoints_dict[img0_name] = kpts0
            image_sizes[img0_name] = orig_size0
        if img1_name not in keypoints_dict:
            keypoints_dict[img1_name] = kpts1
            image_sizes[img1_name] = orig_size1
        
        matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
        
        priority = 0
        if cluster_idx < len(clusters) - 1:
            overlap_indices = clusters[cluster_idx][-overlap:]
            if global_idx0 in overlap_indices or global_idx1 in overlap_indices:
                priority = 1
        
        edges.append((len(mkpts0), idx0, idx1, priority))
        
        print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")
    
    print(f"SuperGlue matching for cluster took {time.time() - start_time:.2f} seconds.")
    
    edges.sort(key=lambda x: (-x[3], -x[0]))
    parent = list(range(len(cluster_indices)))
    rank = [0] * len(cluster_indices)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    selected_pairs = []
    for num_matches, idx0, idx1, _ in edges:
        if num_matches < 8:
            continue
        if find(idx0) != find(idx1):
            union(idx0, idx1)
            selected_pairs.append((idx0, idx1))
    
    if cluster_idx < len(clusters) - 1:
        overlap_indices = clusters[cluster_idx][-overlap:]
        for overlap_idx in overlap_indices:
            overlap_local_idx = cluster_indices.index(overlap_idx)
            if overlap_local_idx not in {idx0 for idx0, _ in selected_pairs} and overlap_local_idx not in {idx1 for _, idx1 in selected_pairs}:
                best_pair = None
                best_num_matches = 0
                for num_matches, idx0, idx1, _ in edges:
                    if num_matches < 8:
                        continue
                    if idx0 == overlap_local_idx or idx1 == overlap_local_idx:
                        if num_matches > best_num_matches:
                            best_num_matches = num_matches
                            best_pair = (idx0, idx1)
                if best_pair:
                    idx0, idx1 = best_pair
                    if find(idx0) != find(idx1):
                        union(idx0, idx1)
                        selected_pairs.append((idx0, idx1))
    
    if not selected_pairs:
        print("No pairs with sufficient matches found in cluster.")
        return {}, [], {}
    
    idx0, idx1 = selected_pairs[0]
    global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
    img0_name = Path(image_paths[global_idx0]).name
    img1_name = Path(image_paths[global_idx1]).name
    
    mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
    mkpts0 = mkpts0.astype(np.float32)
    mkpts1 = mkpts1.astype(np.float32)
    
    image_width, image_height = image_sizes[img0_name]
    focal_length = max(image_width, image_height) * 1.2
    cx, cy = image_width / 2, image_height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        print(f"Failed to estimate essential matrix for initial pair {img0_name} and {img1_name}")
        return {}, [], {}
    
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    if R is None or t is None:
        print(f"Failed to recover pose for initial pair {img0_name} and {img1_name}")
        return {}, [], {}
    
    R = R.astype(np.float32)
    t = t.astype(np.float32).flatten()
    
    poses = {img0_name: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    poses[img1_name] = (R, t)
    registered_indices = {idx0, idx1}
    
    start_time = time.time()
    for idx0, idx1 in selected_pairs[1:]:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        
        if img0_name in poses and img1_name in poses:
            continue
        elif img0_name not in poses and img1_name not in poses:
            continue
        
        if img1_name in poses and img0_name not in poses:
            img0_name, img1_name = img1_name, img0_name
            idx0, idx1 = idx1, idx0
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name})")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            print(f"Failed to estimate essential matrix between {img0_name} and {img1_name}")
            continue
        
        _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
        if R is None or t is None:
            print(f"Failed to recover pose between {img0_name} and {img1_name}")
            continue
        
        R = R.astype(np.float32)
        t = t.astype(np.float32).flatten()
        
        R0, t0 = poses[img0_name]
        R = R0 @ R
        t = R0 @ t + t0
        
        poses[img1_name] = (R, t)
        registered_indices.add(idx1)
    
    print(f"Pose estimation took {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    points3d = []
    for idx0, idx1 in image_pairs:
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        
        if img0_name not in poses or img1_name not in poses:
            continue
        
        key1 = (img0_name, img1_name)
        key2 = (img1_name, img0_name)
        if key1 in matches_dict:
            mkpts0, mkpts1 = matches_dict[key1]
        elif key2 in matches_dict:
            mkpts1, mkpts0 = matches_dict[key2]
        else:
            print(f"Matches not found for pair ({img0_name}, {img1_name}) during triangulation")
            continue
        
        if len(mkpts0) != len(mkpts1):
            print(f"Mismatch in number of points between {img0_name} and {img1_name}: {len(mkpts0)} vs {len(mkpts1)}")
            continue
        
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        R0, t0 = poses[img0_name]
        R1, t1 = poses[img1_name]
        
        P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P0 = P0.astype(np.float32)
        P1 = P1.astype(np.float32)
        
        pts0 = mkpts0.T.astype(np.float32)
        pts1 = mkpts1.T.astype(np.float32)
        
        if pts0.shape[1] == 0 or pts1.shape[1] == 0:
            print(f"Skipping triangulation due to zero matches between {img0_name} and {img1_name}")
            continue
        
        points4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
        points3d_h = points4d[:3] / points4d[3]
        points3d.extend(points3d_h.T)
        
        for _ in range(pts0.shape[1]):
            point_to_images.append([(img0_name, pts0[:, _]), (img1_name, pts1[:, _])])
    
    points3d = np.array(points3d)
    
    valid = np.all(np.isfinite(points3d), axis=1) & (np.abs(points3d) < 1e4).all(axis=1)
    points3d = points3d[valid]
    point_to_images = [pt for pt, v in zip(point_to_images, valid) if v]
    
    print(f"Triangulation took {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    def project(points3d, R, t, K):
        points3d = points3d.T
        points = R @ points3d + t.reshape(3, 1)
        points = K @ points
        points = points[:2] / points[2]
        return points.T
    
    def reprojection_error(params, points3d, observations, K, img_names):
        num_cameras = len(img_names)
        num_points = len(points3d)
        
        translations = params[:num_cameras * 3].reshape(num_cameras, 3)
        focal_length = params[-1]
        K_opt = K.copy()
        K_opt[0, 0] = K_opt[1, 1] = focal_length
        
        errors = []
        for i, img_name in enumerate(img_names):
            R, _ = poses[img_name]
            t = translations[i]
            for j, obs in enumerate(observations):
                for img_obs, pt in obs:
                    if img_obs == img_name:
                        proj = project(points3d[j:j+1], R, t, K_opt)
                        errors.append(proj[0] - pt)
        
        return np.concatenate(errors)
    
    img_names = list(poses.keys())
    translations = np.array([t for _, t in poses.values()])
    params = np.hstack((translations.ravel(), focal_length))
    
    result = least_squares(
        reprojection_error,
        params,
        args=(points3d, point_to_images, K, img_names),
        max_nfev=50,
        ftol=1e-4,
        xtol=1e-4
    )
    optimized_params = result.x
    
    num_cameras = len(img_names)
    translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
    for i, img_name in enumerate(img_names):
        R, _ = poses[img_name]
        poses[img_name] = (R, translations[i])
    
    print(f"Bundle adjustment took {time.time() - start_time:.2f} seconds.")
    
    return poses, points3d, K

# === Process all scenes in the test set ===
test_dir = Path('/kaggle/input/image-matching-challenge-2025/test')
submission_data = []

for scene_dir in test_dir.iterdir():
    if not scene_dir.is_dir():
        continue
    scene_name = scene_dir.name
    print(f"\nProcessing scene: {scene_name}")
    
    image_paths = [str(img) for img in scene_dir.iterdir() if img.suffix in ('.png', '.jpg', '.jpeg')]
    image_paths.sort()
    print(f"Number of images in scene: {len(image_paths)}")
    
    num_images = len(image_paths)
    selected_indices = list(range(num_images))
    
    cluster_size = 4
    overlap = 2
    clusters = []
    for i in range(0, len(selected_indices), cluster_size - overlap):
        cluster_indices = selected_indices[i:i + cluster_size]
        if len(cluster_indices) >= 2:
            clusters.append(cluster_indices)
    print(f"Clusters: {clusters}")
    
    all_poses = {}
    all_points3d = []
    all_Ks = {}
    
    start_time = time.time()
    for cluster_idx, cluster_indices in enumerate(clusters):
        poses, points3d, K = run_sfm_on_cluster(cluster_indices, None, image_paths, matching, device, cluster_idx, clusters)
        all_poses.update(poses)
        all_points3d.append(points3d)
        all_Ks.update({img_name: K for img_name in poses})
    
    if not all_points3d or len(all_points3d[0]) == 0:
        print(f"No points reconstructed in the first cluster for scene {scene_name}. Skipping.")
        continue
    
    merged_points3d = all_points3d[0]
    for i in range(1, len(clusters)):
        overlap_image = Path(image_paths[clusters[i-1][-1]]).name
        if overlap_image not in all_poses:
            print(f"Cannot align clusters {i-1} and {i} using overlapping image {overlap_image}.")
            continue
        
        R0_prev, t0_prev = all_poses[overlap_image]
        R0_curr, t0_curr = all_poses[overlap_image]
        
        R_align = R0_prev @ np.linalg.inv(R0_curr)
        t_align = t0_prev - R_align @ t0_curr
        
        points3d_curr = all_points3d[i]
        if len(points3d_curr) == 0:
            print(f"Cluster {i} has no points to merge")
            continue
        points3d_curr = (R_align @ points3d_curr.T).T + t_align
        all_points3d[i] = points3d_curr
        merged_points3d = np.vstack((merged_points3d, points3d_curr))
        
        for img_name in list(all_poses.keys()):
            if img_name in all_poses:
                R, t = all_poses[img_name]
                R = R_align @ R
                t = R_align @ t + t_align
                all_poses[img_name] = (R, t)
    
    save_ply(merged_points3d, f"/kaggle/working/{scene_name}_point_cloud.ply")
    print(f"Point cloud saved at: /kaggle/working/{scene_name}_point_cloud.ply")
    
    print(f"Scene {scene_name} processed in {time.time() - start_time:.2f} seconds")
    print(f"Total number of 3D points reconstructed: {len(merged_points3d)}")
    print(f"Total number of cameras estimated: {len(all_poses)}")
    
    for img_name in all_poses:
        R, t = all_poses[img_name]
        w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        if w < 1e-10:
            w = 1e-10
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        image_id = f"{scene_name}/{img_name}"
        submission_data.append((image_id, w, x, y, z, t[0], t[1], t[2]))

# === Ensure all images are included in submission ===
all_image_paths = []
for scene_dir in test_dir.iterdir():
    if not scene_dir.is_dir():
        continue
    scene_name = scene_dir.name
    image_paths = [str(img) for img in scene_dir.iterdir() if img.suffix in ('.png', '.jpg', '.jpeg')]
    image_paths.sort()
    for img_path in image_paths:
        img_name = Path(img_path).name
        image_id = f"{scene_name}/{img_name}"
        all_image_paths.append(image_id)

for image_id in all_image_paths:
    if not any(data[0] == image_id for data in submission_data):
        submission_data.append((image_id, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

# === Generate submission file ===
submission_file = Path('/kaggle/working/submission.csv')
with open(submission_file, 'w') as f:
    f.write("image_path,rotation_w,rotation_x,rotation_y,rotation_z,translation_x,translation_y,translation_z\n")
    for data in submission_data:
        image_id, w, x, y, z, tx, ty, tz = data
        f.write(f"{image_id},{w},{x},{y},{z},{tx},{ty},{tz}\n")

print(f"Submission file generated at: {submission_file}")


# === Process all scenes in the test set ===
test_dir = Path('/kaggle/input/image-matching-challenge-2025/test')
submission_data = []

print(f"Test directory: {test_dir}")
print(f"Contents of test directory: {list(test_dir.iterdir())}")

for scene_dir in test_dir.iterdir():
    if not scene_dir.is_dir():
        print(f"Skipping non-directory: {scene_dir}")
        continue
    scene_name = scene_dir.name
    print(f"\nProcessing scene: {scene_name}")

    image_paths = [str(img) for img in scene_dir.iterdir() if img.suffix in ('.png', '.jpg', '.jpeg')]
    image_paths.sort()
    print(f"Number of images in scene: {len(image_paths)}")

    num_images = len(image_paths)
    selected_indices = list(range(num_images))

    cluster_size = 4
    overlap = 2
    clusters = []
    for i in range(0, len(selected_indices), cluster_size - overlap):
        cluster_indices = selected_indices[i:i + cluster_size]
        if len(cluster_indices) >= 2:
            clusters.append(cluster_indices)
    print(f"Clusters: {clusters}")

    all_poses = {}
    all_points3d = []
    all_Ks = {}

    start_time = time.time()
    for cluster_idx, cluster_indices in enumerate(clusters):
        poses, points3d, K = run_sfm_on_cluster(cluster_indices, None, image_paths, matching, device, cluster_idx, clusters)
        print(f"Cluster {cluster_idx}: Number of poses estimated: {len(poses)}")
        all_poses.update(poses)
        all_points3d.append(points3d)
        all_Ks.update({img_name: K for img_name in poses})

    print(f"Total poses after processing all clusters: {len(all_poses)}")

    if not all_points3d or len(all_points3d[0]) == 0:
        print(f"No points reconstructed in the first cluster for scene {scene_name}. Skipping.")
        continue

    merged_points3d = all_points3d[0]
    for i in range(1, len(clusters)):
        overlap_image = Path(image_paths[clusters[i-1][-1]]).name
        if overlap_image not in all_poses:
            print(f"Cannot align clusters {i-1} and {i} using overlapping image {overlap_image}.")
            continue

        R0_prev, t0_prev = all_poses[overlap_image]
        R0_curr, t0_curr = all_poses[overlap_image]

        R_align = R0_prev @ np.linalg.inv(R0_curr)
        t_align = t0_prev - R_align @ t0_curr

        points3d_curr = all_points3d[i]
        if len(points3d_curr) == 0:
            print(f"Cluster {i} has no points to merge")
            continue
        points3d_curr = (R_align @ points3d_curr.T).T + t_align
        all_points3d[i] = points3d_curr
        merged_points3d = np.vstack((merged_points3d, points3d_curr))

        for img_name in list(all_poses.keys()):
            if img_name in all_poses:
                R, t = all_poses[img_name]
                R = R_align @ R
                t = R_align @ t + t_align
                all_poses[img_name] = (R, t)

    save_ply(merged_points3d, f"/kaggle/working/{scene_name}_point_cloud.ply")
    print(f"Point cloud saved at: /kaggle/working/{scene_name}_point_cloud.ply")

    print(f"Scene {scene_name} processed in {time.time() - start_time:.2f} seconds")
    print(f"Total number of 3D points reconstructed: {len(merged_points3d)}")
    print(f"Total number of cameras estimated: {len(all_poses)}")

    print(f"Adding poses to submission_data for scene {scene_name}")
    for img_name in all_poses:
        R, t = all_poses[img_name]
        w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        if w < 1e-10:
            w = 1e-10
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        image_id = f"{scene_name}_{img_name}_public"
        submission_data.append((image_id, w, x, y, z, t[0], t[1], t[2]))
        print(f"Added to submission_data: {image_id}")

print(f"Total entries in submission_data after processing all scenes: {len(submission_data)}")

# === Load the sample submission file to get all expected image paths ===
sample_submission_path = '/kaggle/input/image-matching-challenge-2025/sample_submission.csv'
print(f"Loading sample submission file from: {sample_submission_path}")
sample_submission_df = pd.read_csv(sample_submission_path)
print(f"Total images in sample submission: {len(sample_submission_df)}")
print(f"Columns in sample submission: {sample_submission_df.columns.tolist()}")
print(f"First few rows of sample submission:\n{sample_submission_df.head()}")

# === Debug: Print sample entries in submission_data ===
print("Sample entries in submission_data:")
for entry in submission_data[:5]:
    print(entry)

# === Convert submission_data to a DataFrame ===
# The image_id in submission_data is already in the correct format (e.g., ETs_another_et_another_et001.png_public)
submission_df = pd.DataFrame(
    submission_data,
    columns=['image_path', 'rotation_w', 'rotation_x', 'rotation_y', 'rotation_z', 'translation_x', 'translation_y', 'translation_z']
)
print(f"submission_data has {len(submission_df)} entries before merging")

# === Merge with sample submission to ensure all images are included ===
# The image_path column in submission_df matches the image_id column in sample_submission_df
merged_df = sample_submission_df[['image_id']].merge(
    submission_df,
    left_on='image_id',
    right_on='image_path',
    how='left'
)

# Drop the redundant image_path column after merging
merged_df = merged_df.drop(columns=['image_path'])

# Fill missing values with default poses
default_pose = {'rotation_w': 1.0, 'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation_z': 0.0,
                'translation_x': 0.0, 'translation_y': 0.0, 'translation_z': 0.0}
merged_df.fillna(default_pose, inplace=True)

print(f"After merging, submission has {len(merged_df)} entries")
print(f"Sample rows from merged submission:\n{merged_df.head()}")

# === Generate submission file ===
# Ensure the submission file has the expected column names (Kaggle expects 'image_path')
merged_df = merged_df.rename(columns={'image_id': 'image_path'})
submission_file = Path('/kaggle/working/submission.csv')
merged_df.to_csv(submission_file, index=False)
print(f"Submission file generated at: {submission_file}")

# Verify the submission file
submission_df = pd.read_csv(submission_file)
print(f"Number of rows in submission.csv: {len(submission_df)}")
print(f"First few rows of submission.csv:\n{submission_df.head()}")


# === Import Libraries ===
import numpy as np
from pathlib import Path
import cv2
import time
import torch
import sys
import warnings
import pandas as pd
from scipy.optimize import least_squares
warnings.filterwarnings("ignore", category=FutureWarning)

# Add SuperGlue path
sys.path.append('SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import frame2tensor

# === Function to Save Point Cloud as PLY ===
def save_ply(points3d, output_path):
    header = f"""ply
format ascii 1.0
element vertex {len(points3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    colors = np.random.randint(0, 255, size=(len(points3d), 3))
    with open(output_path, 'w') as f:
        f.write(header)
        for point, color in zip(points3d, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

# === Function to Initialize SuperGlue with Scene-Specific Weights ===
def initialize_superglue(scene_name):
    print(f"Initializing SuperGlue for scene: {scene_name}")
    start_time = time.time()
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 4096
        },
        'superglue': {
            'sinkhorn_iterations': 30 if scene_name.lower() == 'stairs' else 20,  # Increased for stairs
            'match_threshold': 0.03 if scene_name.lower() == 'stairs' else 0.02   # Increased for stairs
        }
    }
    indoor_scenes = ['ETs', 'imc2023_theather_imc2024_church', 'imc2024_dioscuri_baalshamin',
                     'pt_brandenburg_british_buckingham', 'pt_piazzasanmarco_grandplace',
                     'pt_sacrecoeur_trevi_tajmahal', 'pt_stpeters_stpauls']
    if any(scene.lower() in scene_name.lower() for scene in indoor_scenes):
        config['superglue']['weights'] = 'indoor'
        print(f"Using 'indoor' weights for scene: {scene_name}")
    else:
        config['superglue']['weights'] = 'outdoor'
        print(f"Using 'outdoor' weights for scene: {scene_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matching = Matching(config).eval().to(device)
    print(f"SuperGlue initialized in {time.time() - start_time:.2f} seconds")
    return matching, device

# === Function to Recover Pose with Fallback ===
def recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name):
    # First attempt: Essential matrix
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5, maxIters=2000)
    if E is None or E.shape != (3, 3):
        print(f"Essential matrix estimation failed for pair {img0_name} and {img1_name}. Trying homography.")
    else:
        inliers, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
        if inliers >= 10:  # Require at least 10 inliers
            return R, t, inliers
    
    # Fallback: Homography (assuming planar scene, common in stairs)
    H, mask = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=0.5, confidence=0.999, maxIters=2000)
    if H is None or H.shape != (3, 3):
        print(f"Homography estimation failed for pair {img0_name} and {img1_name}. Giving up.")
        return None, None, 0
    
    # Decompose homography to get possible poses
    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
    best_R, best_t, best_inliers = None, None, 0
    for i in range(num):
        R, t = Rs[i], ts[i]
        points3d = cv2.triangulatePoints(K @ np.hstack((np.eye(3), np.zeros((3, 1)))), K @ np.hstack((R, t)), mkpts0.T, mkpts1.T)
        points3d = points3d[:3] / points3d[3]
        points = K @ (R @ points3d + t.reshape(3, 1))
        points = points[:2] / points[2]
        errors = np.linalg.norm(points.T - mkpts1, axis=1)
        inliers = np.sum(errors < 1.0)
        if inliers > best_inliers:
            best_inliers = inliers
            best_R, best_t = R, t
    
    if best_inliers >= 10:
        return best_R, best_t, best_inliers
    else:
        print(f"Insufficient inliers from homography decomposition for pair {img0_name} and {img1_name}.")
        return None, None, 0

# === Function to Run SfM on a Single Cluster ===
def run_sfm_on_cluster(args):
    cluster_indices, image_dir, image_paths, scene_name, cluster_idx, clusters, overlap = args
    print(f"Starting cluster {cluster_idx} for scene {scene_name}")
    
    try:
        # Initialize SuperGlue and device
        matching, device = initialize_superglue(scene_name)
        
        keypoints_dict = {}
        matches_dict = {}
        image_sizes = {}
        point_to_images = []
        
        print(f"Processing cluster {cluster_idx} with indices: {cluster_indices}")
        
        start_time = time.time()
        image_pairs = [(i, j) for i in range(len(cluster_indices)) for j in range(i + 1, len(cluster_indices))]
        edges = []
        for idx0, idx1 in image_pairs:
            global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
            img0 = cv2.imread(image_paths[global_idx0], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(image_paths[global_idx1], cv2.IMREAD_GRAYSCALE)
            
            if img0 is None or img1 is None:
                print(f"Failed to load images: {image_paths[global_idx0]} or {image_paths[global_idx1]}")
                continue
            
            img0 = cv2.equalizeHist(img0)
            img1 = cv2.equalizeHist(img1)
            
            orig_size0 = img0.shape[::-1]
            orig_size1 = img1.shape[::-1]
            
            img0 = cv2.resize(img0, (640, 480))
            img1 = cv2.resize(img1, (640, 480))
            
            inp0 = frame2tensor(img0, device)
            inp1 = frame2tensor(img1, device)
            
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            
            scale0 = (orig_size0[0] / 640, orig_size0[1] / 480)
            scale1 = (orig_size1[0] / 640, orig_size1[1] / 480)
            kpts0[:, 0] *= scale0[0]
            kpts0[:, 1] *= scale0[1]
            kpts1[:, 0] *= scale1[0]
            kpts1[:, 1] *= scale1[1]
            mkpts0[:, 0] *= scale0[0]
            mkpts0[:, 1] *= scale0[1]
            mkpts1[:, 0] *= scale1[0]
            mkpts1[:, 1] *= scale1[1]
            
            img0_name = Path(image_paths[global_idx0]).name
            img1_name = Path(image_paths[global_idx1]).name
            if img0_name not in keypoints_dict:
                keypoints_dict[img0_name] = kpts0
                image_sizes[img0_name] = orig_size0
            if img1_name not in keypoints_dict:
                keypoints_dict[img1_name] = kpts1
                image_sizes[img1_name] = orig_size1
            
            matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
            
            priority = 0
            if cluster_idx < len(clusters) - 1:
                overlap_indices = clusters[cluster_idx][-overlap:]
                if global_idx0 in overlap_indices or global_idx1 in overlap_indices:
                    priority = 1
            
            edges.append((len(mkpts0), idx0, idx1, priority))
            
            print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")
        
        print(f"SuperGlue matching for cluster took {time.time() - start_time:.2f} seconds.")
        
        # Sort edges by priority and number of matches
        edges.sort(key=lambda x: (-x[3], -x[0]))
        parent = list(range(len(cluster_indices)))
        rank = [0] * len(cluster_indices)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        selected_pairs = []
        for num_matches, idx0, idx1, _ in edges:
            if num_matches < 6:
                continue
            if find(idx0) != find(idx1):
                union(idx0, idx1)
                selected_pairs.append((idx0, idx1))
        
        # Ensure overlap images are included
        if cluster_idx < len(clusters) - 1:
            overlap_indices = clusters[cluster_idx][-overlap:]
            for overlap_idx in overlap_indices:
                overlap_local_idx = cluster_indices.index(overlap_idx)
                if overlap_local_idx not in {idx0 for idx0, _ in selected_pairs} and overlap_local_idx not in {idx1 for _, idx1 in selected_pairs}:
                    best_pair = None
                    best_num_matches = 0
                    for num_matches, idx0, idx1, _ in edges:
                        if num_matches < 6:
                            continue
                        if idx0 == overlap_local_idx or idx1 == overlap_local_idx:
                            if num_matches > best_num_matches:
                                best_num_matches = num_matches
                                best_pair = (idx0, idx1)
                    if best_pair:
                        idx0, idx1 = best_pair
                        if find(idx0) != find(idx1):
                            union(idx0, idx1)
                            selected_pairs.append((idx0, idx1))
        
        # Fallback: If no pairs are selected, try pairwise processing
        if not selected_pairs:
            print("No pairs with sufficient matches found in cluster. Trying pairwise processing...")
            selected_pairs = []
            for i in range(len(cluster_indices) - 1):
                idx0, idx1 = i, i + 1
                global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
                img0_name = Path(image_paths[global_idx0]).name
                img1_name = Path(image_paths[global_idx1]).name
                if (img0_name, img1_name) in matches_dict:
                    mkpts0, _ = matches_dict[(img0_name, img1_name)]
                    if len(mkpts0) >= 6:
                        selected_pairs.append((idx0, idx1))
        
        if not selected_pairs:
            print("Pairwise processing also failed. Skipping cluster.")
            return {}, [], {}, []
        
        # Initialize poses with the first pair
        idx0, idx1 = selected_pairs[0]
        global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        
        if (img0_name, img1_name) not in matches_dict:
            print(f"Initial pair ({img0_name}, {img1_name}) not in matches_dict. Skipping cluster.")
            return {}, [], {}, []
        
        mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
        mkpts0 = mkpts0.astype(np.float32)
        mkpts1 = mkpts1.astype(np.float32)
        
        image_width, image_height = image_sizes[img0_name]
        focal_length = max(image_width, image_height) * 1.0
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            print(f"Failed to recover pose for initial pair {img0_name} and {img1_name}. Skipping cluster.")
            return {}, [], {}, []
        
        R = R.astype(np.float32)
        t = t.astype(np.float32).flatten()
        
        poses = {img0_name: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
        poses[img1_name] = (R, t)
        registered_indices = {idx0, idx1}
        
        # Pose estimation for remaining pairs
        start_time = time.time()
        for idx0, idx1 in selected_pairs[1:]:
            global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
            img0_name = Path(image_paths[global_idx0]).name
            img1_name = Path(image_paths[global_idx1]).name
            
            # Skip if both images are already registered
            if img0_name in poses and img1_name in poses:
                continue
            # Skip if neither image is registered (cannot anchor the pair)
            if img0_name not in poses and img1_name not in poses:
                continue
            
            # Ensure img0_name is the registered image
            if img1_name in poses and img0_name not in poses:
                img0_name, img1_name = img1_name, img0_name
                idx0, idx1 = idx1, idx0
            
            # Check if matches exist for this pair
            key1 = (img0_name, img1_name)
            key2 = (img1_name, img0_name)
            if key1 in matches_dict:
                mkpts0, mkpts1 = matches_dict[key1]
            elif key2 in matches_dict:
                mkpts1, mkpts0 = matches_dict[key2]
            else:
                print(f"Matches not found for pair ({img0_name}, {img1_name}) in matches_dict. Skipping pair.")
                continue
            
            # Ensure we have enough matches
            if len(mkpts0) < 6:
                print(f"Insufficient matches ({len(mkpts0)}) for pair ({img0_name}, {img1_name}). Skipping pair.")
                continue
            
            mkpts0 = mkpts0.astype(np.float32)
            mkpts1 = mkpts1.astype(np.float32)
            
            R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
            if R is None or t is None:
                print(f"Failed to recover pose between {img0_name} and {img1_name}. Skipping pair.")
                continue
            
            R = R.astype(np.float32)
            t = t.astype(np.float32).flatten()
            
            R0, t0 = poses[img0_name]
            R = R0 @ R
            t = R0 @ t + t0
            
            poses[img1_name] = (R, t)
            registered_indices.add(idx1)
        
        # Fallback: Register remaining images pairwise
        remaining_indices = set(range(len(cluster_indices))) - registered_indices
        for idx in sorted(remaining_indices):
            global_idx = cluster_indices[idx]
            img_name = Path(image_paths[global_idx]).name
            best_pair = None
            best_num_matches = 0
            for num_matches, idx0, idx1, _ in edges:
                if num_matches < 6:
                    continue
                if idx0 == idx or idx1 == idx:
                    other_idx = idx1 if idx0 == idx else idx0
                    if other_idx in registered_indices:
                        if num_matches > best_num_matches:
                            best_num_matches = num_matches
                            best_pair = (idx0, idx1)
            if best_pair:
                idx0, idx1 = best_pair
                global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
                img0_name = Path(image_paths[global_idx0]).name
                img1_name = Path(image_paths[global_idx1]).name
                
                if img0_name in poses and img1_name in poses:
                    continue
                elif img0_name not in poses and img1_name not in poses:
                    continue
                
                if img1_name in poses and img0_name not in poses:
                    img0_name, img1_name = img1_name, img0_name
                    idx0, idx1 = idx1, idx0
                
                key1 = (img0_name, img1_name)
                key2 = (img1_name, img0_name)
                if key1 in matches_dict:
                    mkpts0, mkpts1 = matches_dict[key1]
                elif key2 in matches_dict:
                    mkpts1, mkpts0 = matches_dict[key2]
                else:
                    print(f"Fallback: Matches not found for pair ({img0_name}, {img1_name}) in matches_dict. Skipping pair.")
                    continue
                
                if len(mkpts0) < 6:
                    print(f"Fallback: Insufficient matches ({len(mkpts0)}) for pair ({img0_name}, {img1_name}). Skipping pair.")
                    continue
                
                mkpts0 = mkpts0.astype(np.float32)
                mkpts1 = mkpts1.astype(np.float32)
                
                R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
                if R is None or t is None:
                    print(f"Fallback: Failed to recover pose between {img0_name} and {img1_name}. Skipping pair.")
                    continue
                
                R = R.astype(np.float32)
                t = t.astype(np.float32).flatten()
                
                R0, t0 = poses[img0_name]
                R = R0 @ R
                t = R0 @ t + t0
                
                poses[img1_name] = (R, t)
                registered_indices.add(idx1)
        
        print(f"Pose estimation took {time.time() - start_time:.2f} seconds.")
        
        # Triangulation
        start_time = time.time()
        points3d = []
        cluster_point_to_images = []
        for idx0, idx1 in image_pairs:
            global_idx0, global_idx1 = cluster_indices[idx0], cluster_indices[idx1]
            img0_name = Path(image_paths[global_idx0]).name
            img1_name = Path(image_paths[global_idx1]).name
            
            if img0_name not in poses or img1_name not in poses:
                continue
            
            key1 = (img0_name, img1_name)
            key2 = (img1_name, img0_name)
            if key1 in matches_dict:
                mkpts0, mkpts1 = matches_dict[key1]
            elif key2 in matches_dict:
                mkpts1, mkpts0 = matches_dict[key2]
            else:
                print(f"Matches not found for pair ({img0_name}, {img1_name}) during triangulation")
                continue
            
            if len(mkpts0) != len(mkpts1):
                print(f"Mismatch in number of points between {img0_name} and {img1_name}: {len(mkpts0)} vs {len(mkpts1)}")
                continue
            
            mkpts0 = mkpts0.astype(np.float32)
            mkpts1 = mkpts1.astype(np.float32)
            
            R0, t0 = poses[img0_name]
            R1, t1 = poses[img1_name]
            
            P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
            P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
            P0 = P0.astype(np.float32)
            P1 = P1.astype(np.float32)
            
            pts0 = mkpts0.T.astype(np.float32)
            pts1 = mkpts1.T.astype(np.float32)
            
            if pts0.shape[1] == 0 or pts1.shape[1] == 0:
                print(f"Skipping triangulation due to zero matches between {img0_name} and {img1_name}")
                continue
            
            points4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
            points3d_h = points4d[:3] / points4d[3]
            points3d.extend(points3d_h.T)
            
            for _ in range(pts0.shape[1]):
                cluster_point_to_images.append([(img0_name, pts0[:, _]), (img1_name, pts1[:, _])])
        
        points3d = np.array(points3d)
        
        valid = np.all(np.isfinite(points3d), axis=1) & (np.abs(points3d) < 1e4).all(axis=1)
        points3d = points3d[valid]
        cluster_point_to_images = [pt for pt, v in zip(cluster_point_to_images, valid) if v]
        
        print(f"Triangulation took {time.time() - start_time:.2f} seconds.")
        
        # Local Bundle Adjustment
        if len(poses) < 3 or len(cluster_point_to_images) == 0:
            print("Skipping bundle adjustment due to insufficient poses or observations.")
        else:
            start_time = time.time()
            def project(points3d, R, t, K):
                points3d = points3d.T
                points = R @ points3d + t.reshape(3, 1)
                points = K @ points
                points = points[:2] / points[2]
                return points.T
            
            def reprojection_error(params, points3d, observations, K, img_names):
                num_cameras = len(img_names)
                num_points = len(points3d)
                
                translations = params[:num_cameras * 3].reshape(num_cameras, 3)
                focal_length = params[-1]
                K_opt = K.copy()
                K_opt[0, 0] = K_opt[1, 1] = focal_length
                
                errors = []
                for i, img_name in enumerate(img_names):
                    R, _ = poses[img_name]
                    t = translations[i]
                    for j, obs in enumerate(observations):
                        for img_obs, pt in obs:
                            if img_obs == img_name:
                                proj = project(points3d[j:j+1], R, t, K_opt)
                                errors.append(proj[0] - pt)
                
                if len(errors) == 0:
                    print("No reprojection errors computed. Returning zeros.")
                    return np.zeros(num_cameras * 3 + 1)
                return np.concatenate(errors)
            
            img_names = list(poses.keys())
            translations = np.array([t for _, t in poses.values()])
            params = np.hstack((translations.ravel(), focal_length))
            
            result = least_squares(
                reprojection_error,
                params,
                args=(points3d, cluster_point_to_images, K, img_names),
                max_nfev=20,  # Reduced for faster processing
                ftol=1e-5,
                xtol=1e-5
            )
            optimized_params = result.x
            
            num_cameras = len(img_names)
            translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
            for i, img_name in enumerate(img_names):
                R, _ = poses[img_name]
                poses[img_name] = (R, translations[i])
            
            print(f"Bundle adjustment took {time.time() - start_time:.2f} seconds.")
        
        return poses, points3d, K, cluster_point_to_images
    
    except Exception as e:
        print(f"Error in cluster {cluster_idx} for scene {scene_name}: {str(e)}")
        return {}, [], {}, []

# === Process All Scenes in the Test Set ===
test_dir = Path('/kaggle/input/image-matching-challenge-2025/test')
submission_data = []

print(f"Test directory: {test_dir}")
print(f"Contents of test directory: {list(test_dir.iterdir())}")

# Load sample submission to get all expected scenes
sample_submission_path = '/kaggle/input/image-matching-challenge-2025/sample_submission.csv'
sample_submission_df = pd.read_csv(sample_submission_path)
print(f"Total images in sample submission: {len(sample_submission_df)}")
unique_scenes = sample_submission_df['dataset'].unique()
print(f"Unique scenes in sample submission: {unique_scenes}")

# Iterate over all scenes in the sample submission
for scene_name in unique_scenes:
    scene_dir = test_dir / scene_name
    if not scene_dir.is_dir():
        print(f"Scene directory {scene_dir} does not exist, skipping (will be available during evaluation).")
        continue
    print(f"\nProcessing scene: {scene_name}")

    image_paths = [str(img) for img in scene_dir.iterdir() if img.suffix in ('.png', '.jpg', '.jpeg')]
    image_paths.sort()
    print(f"Number of images in scene: {len(image_paths)}")

    num_images = len(image_paths)
    selected_indices = list(range(num_images))

    # Adjust cluster size for stairs
    cluster_size = 8 if scene_name.lower() == 'stairs' else 10  # Reduced for stairs
    overlap = 2 if scene_name.lower() == 'stairs' else 5        # Reduced overlap for stairs
    clusters = []
    for i in range(0, len(selected_indices), cluster_size - overlap):
        cluster_indices = selected_indices[i:i + cluster_size]
        if len(cluster_indices) >= 2:
            clusters.append(cluster_indices)
    print(f"Clusters: {clusters}")

    all_poses = {}
    all_points3d = []
    all_Ks = {}
    all_observations = []

    start_time = time.time()
    print(f"Using sequential processing for scene {scene_name}")
    for cluster_idx, cluster_indices in enumerate(clusters):
        result = run_sfm_on_cluster((cluster_indices, None, image_paths, scene_name, cluster_idx, clusters, overlap))
        poses, points3d, K, cluster_point_to_images = result
        all_poses.update(poses)
        if len(points3d) > 0:
            all_points3d.append(points3d)
        all_Ks.update({img_name: K for img_name in poses})
        all_observations.extend(cluster_point_to_images)

    print(f"Total poses after processing all clusters: {len(all_poses)}")

    if not all_points3d or len(all_points3d[0]) == 0:
        print(f"No points reconstructed in the first cluster for scene {scene_name}. Skipping.")
        continue

    # Merge point clouds and align clusters
    merged_points3d = all_points3d[0]
    for i in range(1, len(all_points3d)):
        if i >= len(clusters):
            continue
        overlap_images = [Path(image_paths[idx]).name for idx in clusters[i-1][-overlap:]]
        alignment_found = False
        for overlap_image in overlap_images:
            if overlap_image in all_poses:
                R0_prev, t0_prev = all_poses[overlap_image]
                R0_curr, t0_curr = all_poses[overlap_image]

                R_align = R0_prev @ np.linalg.inv(R0_curr)
                t_align = t0_prev - R_align @ t0_curr

                points3d_curr = all_points3d[i]
                if len(points3d_curr) == 0:
                    print(f"Cluster {i} has no points to merge")
                    continue
                points3d_curr = (R_align @ points3d_curr.T).T + t_align
                all_points3d[i] = points3d_curr
                merged_points3d = np.vstack((merged_points3d, points3d_curr))

                for img_name in list(all_poses.keys()):
                    if img_name in all_poses:
                        R, t = all_poses[img_name]
                        R = R_align @ R
                        t = R_align @ t + t_align
                        all_poses[img_name] = (R, t)
                alignment_found = True
                break
        if not alignment_found:
            print(f"Cannot align clusters {i-1} and {i} due to missing overlap poses.")

    # Global bundle adjustment (simplified)
    if len(all_poses) >= 3 and len(all_observations) > 0:
        start_time = time.time()
        def project(points3d, R, t, K):
            points3d = points3d.T
            points = R @ points3d + t.reshape(3, 1)
            points = K @ points
            points = points[:2] / points[2]
            return points.T
        
        def reprojection_error(params, points3d, observations, K, img_names):
            num_cameras = len(img_names)
            num_points = len(points3d)
            
            translations = params[:num_cameras * 3].reshape(num_cameras, 3)
            focal_length = params[-1]
            K_opt = K.copy()
            K_opt[0, 0] = K_opt[1, 1] = focal_length
            
            errors = []
            for i, img_name in enumerate(img_names):
                R, _ = all_poses[img_name]
                t = translations[i]
                for j, obs in enumerate(observations):
                    for img_obs, pt in obs:
                        if img_obs == img_name:
                            proj = project(points3d[j:j+1], R, t, K_opt)
                            errors.append(proj[0] - pt)
            
            if len(errors) == 0:
                print("No reprojection errors computed in global BA. Returning zeros.")
                return np.zeros(num_cameras * 3 + 1)
            return np.concatenate(errors)
        
        img_names = list(all_poses.keys())
        translations = np.array([t for _, t in all_poses.values()])
        params = np.hstack((translations.ravel(), focal_length))
        
        result = least_squares(
            reprojection_error,
            params,
            args=(merged_points3d, all_observations, K, img_names),
            max_nfev=20,  # Reduced for faster processing
            ftol=1e-5,
            xtol=1e-5
        )
        optimized_params = result.x
        
        num_cameras = len(img_names)
        translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
        for i, img_name in enumerate(img_names):
            R, _ = all_poses[img_name]
            all_poses[img_name] = (R, translations[i])
        
        print(f"Global bundle adjustment took {time.time() - start_time:.2f} seconds.")

    save_ply(merged_points3d, f"/kaggle/working/{scene_name}_point_cloud.ply")
    print(f"Point cloud saved at: /kaggle/working/{scene_name}_point_cloud.ply")

    print(f"Scene {scene_name} processed in {time.time() - start_time:.2f} seconds")
    print(f"Total number of 3D points reconstructed: {len(merged_points3d)}")
    print(f"Total number of cameras estimated: {len(all_poses)}")

    print(f"Adding poses to submission_data for scene {scene_name}")
    for img_name in all_poses:
        R, t = all_poses[img_name]
        w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        if w < 1e-10:
            w = 1e-10
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        image_id = f"{scene_name}/{img_name}"
        submission_data.append((image_id, w, x, y, z, t[0], t[1], t[2]))
        print(f"Added to submission_data: {image_id}")

print(f"Total entries in submission_data after processing all scenes: {len(submission_data)}")

# === Merge with Sample Submission ===
print(f"Loading sample submission file from: {sample_submission_path}")
print(f"Total images in sample submission: {len(sample_submission_df)}")
print(f"Columns in sample submission: {sample_submission_df.columns.tolist()}")
print(f"First few rows of sample submission:\n{sample_submission_df.head()}")

submission_df = pd.DataFrame(
    submission_data,
    columns=['image_path', 'rotation_w', 'rotation_x', 'rotation_y', 'rotation_z', 'translation_x', 'translation_y', 'translation_z']
)
print(f"submission_data has {len(submission_df)} entries before merging")

sample_submission_df['image_path'] = sample_submission_df['dataset'] + '/' + sample_submission_df['image']
merged_df = sample_submission_df[['image_path']].merge(
    submission_df,
    on='image_path',
    how='left'
)

default_pose = {
    'rotation_w': 1.0, 'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation_z': 0.0,
    'translation_x': 0.0, 'translation_y': 0.0, 'translation_z': 0.0
}
merged_df.fillna(default_pose, inplace=True)

print(f"After merging, submission has {len(merged_df)} entries")
print(f"Sample rows from merged submission:\n{merged_df.head()}")

# === Generate Submission File ===
submission_file = Path('/kaggle/working/submission.csv')
merged_df.to_csv(submission_file, index=False)
print(f"Submission file generated at: {submission_file}")

# Verify the submission file
submission_df = pd.read_csv(submission_file)
print(f"Number of rows in submission.csv: {len(submission_df)}")
print(f"First few rows of submission.csv:\n{submission_df.head()}")


import pandas as pd
import numpy as np

# Read the submission.csv file
# If you're on your local machine, replace the path with the location of your downloaded file
# If you're in a Kaggle notebook, upload the file or read it from /kaggle/working/
df = pd.read_csv('/kaggle/working/submission.csv')  # Adjust the path as needed

# Validate the file
print(f"Number of rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Expected values
expected_rows = 1945
expected_columns = ['image_path', 'rotation_w', 'rotation_x', 'rotation_y', 'rotation_z', 
                    'translation_x', 'translation_y', 'translation_z']

# Check row count
assert len(df) == expected_rows, f"Expected {expected_rows} rows, but got {len(df)}"

# Check columns
assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, but got {df.columns.tolist()}"

# Check for NaN or infinity
numeric_cols = ['rotation_w', 'rotation_x', 'rotation_y', 'rotation_z', 
                'translation_x', 'translation_y', 'translation_z']
assert not df[numeric_cols].isnull().any().any(), "Submission contains NaN values"
assert np.isfinite(df[numeric_cols]).all().all(), "Submission contains non-finite values (inf or -inf)"

# Check quaternion normalization
quaternion_cols = ['rotation_w', 'rotation_x', 'rotation_y', 'rotation_z']
quaternion_norm = (df[quaternion_cols]**2).sum(axis=1)
assert np.allclose(quaternion_norm, 1.0, atol=1e-6), "Quaternions are not normalized"

# Check image_path format
for path in df['image_path']:
    assert isinstance(path, str) and len(path.split('/')) == 2, f"Invalid image_path format: {path}"

print("Submission file validation passed!")

# Re-save the file with proper encoding and line endings
df.to_csv('/kaggle/working/submission_clean.csv', index=False, encoding='utf-8', lineterminator='\n')
print("Re-saved submission as submission_clean.csv")


import os

# Rename submission_clean.csv to submission.csv
os.rename('/kaggle/working/submission_clean.csv', '/kaggle/working/submission.csv')
print("Renamed submission_clean.csv to submission.csv")

# Confirm the file exists
print("Files in /kaggle/working/:")
for file in os.listdir('/kaggle/working'):
    print(f"- {file}")


import shutil

# Clean up /kaggle/working/ except for submission.csv
files_to_keep = ['submission.csv']
for file in os.listdir('/kaggle/working'):
    if file not in files_to_keep:
        file_path = os.path.join('/kaggle/working', file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Removed directory {file_path}")

# Confirm remaining files
print("Remaining files in /kaggle/working/:")
for file in os.listdir('/kaggle/working'):
    print(f"- {file}")


import os
import shutil

# Create the weights directory if it doesn't exist
weights_dir = '/kaggle/working/SuperGluePretrainedNetwork/models/weights/'
os.makedirs(weights_dir, exist_ok=True)

# Path to the weights dataset (adjust based on your dataset name)
weights_dataset_path = '/kaggle/input/superglue-weights/'

# List of weight files
weight_files = ['superpoint_v1.pth', 'superglue_indoor.pth', 'superglue_outdoor.pth']

# Copy weights from the dataset to the expected directory
for weight_file in weight_files:
    src_path = os.path.join(weights_dataset_path, weight_file)
    dst_path = os.path.join(weights_dir, weight_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {weight_file} to {dst_path}")
    else:
        raise FileNotFoundError(f"Weight file {src_path} not found in dataset")

# Verify that the files exist
for weight_file in weight_files:
    weight_path = os.path.join(weights_dir, weight_file)
    if os.path.exists(weight_path):
        print(f"Found {weight_path}")
    else:
        raise FileNotFoundError(f"Failed to find {weight_path}")


!git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git


import sys
import os

# Path to the SuperGluePretrainedNetwork dataset
superglue_dataset_path = '/kaggle/input/supergluepretrainednetwork/SuperGluePretrainedNetwork'
superglue_working_path = '/kaggle/working/SuperGluePretrainedNetwork'

if os.path.exists(superglue_dataset_path):
    sys.path.append(superglue_dataset_path)
    print(f"Added {superglue_dataset_path} to sys.path")
elif os.path.exists(superglue_working_path):
    sys.path.append(superglue_working_path)
    print(f"Added {superglue_working_path} to sys.path")
else:
    raise FileNotFoundError("SuperGluePretrainedNetwork directory not found in /kaggle/input/ or /kaggle/working/")

from models.matching import Matching
from models.utils import frame2tensor


# Copy SuperGluePretrainedNetwork to /kaggle/working/
if not os.path.exists(superglue_working_path):
    shutil.copytree(superglue_dataset_path, superglue_working_path)
    print(f"Copied SuperGluePretrainedNetwork to {superglue_working_path}")
else:
    print(f"{superglue_working_path} already exists, skipping copy")


# === Import Libraries ===
import numpy as np
from pathlib import Path
import cv2
import time
import torch
import sys
import warnings
import pandas as pd
from scipy.optimize import least_squares
import os
import shutil
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore", category=FutureWarning)

# === Set Up SuperGluePretrainedNetwork ===
# Path to the SuperGluePretrainedNetwork dataset
superglue_dataset_path = '/kaggle/input/supergluepretrainednetwork/SuperGluePretrainedNetwork'
superglue_working_path = '/kaggle/working/SuperGluePretrainedNetwork'

# Add SuperGluePretrainedNetwork to sys.path
if os.path.exists(superglue_dataset_path):
    sys.path.append(superglue_dataset_path)
    print(f"Added {superglue_dataset_path} to sys.path")
elif os.path.exists(superglue_working_path):
    sys.path.append(superglue_working_path)
    print(f"Added {superglue_working_path} to sys.path")
else:
    raise FileNotFoundError("SuperGluePretrainedNetwork directory not found in /kaggle/input/ or /kaggle/working/")

# Copy SuperGluePretrainedNetwork to /kaggle/working/
if not os.path.exists(superglue_working_path):
    shutil.copytree(superglue_dataset_path, superglue_working_path)
    print(f"Copied SuperGluePretrainedNetwork to {superglue_working_path}")
else:
    print(f"{superglue_working_path} already exists, skipping copy")

# Import SuperGlue modules
from models.matching import Matching
from models.utils import frame2tensor

# === Set Up SuperGlue Weights ===
# Create the weights directory
weights_dir = '/kaggle/working/SuperGluePretrainedNetwork/models/weights/'
os.makedirs(weights_dir, exist_ok=True)

# Path to the weights dataset (adjust based on your dataset name)
weights_dataset_path = '/kaggle/input/superglue-weights/'

# List of weight files
weight_files = ['superpoint_v1.pth', 'superglue_indoor.pth', 'superglue_outdoor.pth']

# Copy weights from the dataset to the expected directory
for weight_file in weight_files:
    src_path = os.path.join(weights_dataset_path, weight_file)
    dst_path = os.path.join(weights_dir, weight_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {weight_file} to {dst_path}")
    else:
        raise FileNotFoundError(f"Weight file {src_path} not found in dataset")

# Verify that the files exist
for weight_file in weight_files:
    weight_path = os.path.join(weights_dir, weight_file)
    if os.path.exists(weight_path):
        print(f"Found {weight_path}")
    else:
        raise FileNotFoundError(f"Failed to find {weight_path}")

# Rest of your code (save_ply, initialize_superglue, cluster_images, etc.) remains the same


# Phase 1: Setup and Initial Functions
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# Utility functions
def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    if not np.all(np.isfinite(R)) or R.shape != (3, 3):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # Reorder to [w, x, y, z]

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    if not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64)
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # Reorder [x, y, z, w]
    return rot.as_matrix()

def project(points3d, R, t, K):
    """Project 3D points to 2D using camera intrinsics and extrinsics."""
    if not (np.all(np.isfinite(points3d)) and np.all(np.isfinite(R)) and np.all(np.isfinite(t)) and np.all(np.isfinite(K))):
        return np.zeros((points3d.shape[0], 2), dtype=np.float32)
    points3d = points3d.T
    P = K @ np.hstack((R, t.reshape(3, 1)))
    points2d = P @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
    points2d = points2d[:2] / (points2d[2] + 1e-8)
    return points2d.T

def validate_pose(R, t, img_name="Unknown"):
    """Validate that R and t have the correct shapes and are finite."""
    if R.shape != (3, 3):
        print(f"Invalid rotation matrix shape for {img_name}: {R.shape}")
        return False
    if t.shape != (3,):
        print(f"Invalid translation vector shape for {img_name}: {t.shape}")
        return False
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        print(f"Non-finite values in pose for {img_name}: R={R}, t={t}")
        return False
    return True

def normalize_translation(t, img_name="Unknown"):
    """Ensure translation vector is a 1D array of shape (3,)."""
    if t is None:
        print(f"Translation vector is None for {img_name}. Using default.")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    t = np.array(t, dtype=np.float32).flatten()
    if t.size != 3:
        print(f"Cannot normalize translation vector for {img_name}: size={t.size}")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return t

# Load sample submission
sample_submission = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
print("Columns in sample_submission.csv:", sample_submission.columns.tolist())

image_path_column = 'image'
if image_path_column not in sample_submission.columns:
    raise ValueError("Expected 'image' column in sample_submission.csv")

# Function to infer the correct scene
def infer_scene_from_image(dataset, image_name):
    if dataset == 'ETs':
        if image_name.startswith('et_et') or image_name.startswith('another_et_another_et') or image_name.startswith('outliers_out_et'):
            return 'et'
    elif dataset == 'amy_gardens':
        if image_name.startswith('peach_'):
            return 'peach'
    elif dataset == 'stairs':
        if image_name.startswith('stairs_split_1_'):
            return 'stairs_split_1'
        elif image_name.startswith('stairs_split_2_'):
            return 'stairs_split_2'
    elif dataset == 'pt_stpeters_stpauls':
        if image_name.startswith('st_peters_square_'):
            return 'st_peters_square'
        return 'pt_stpeters_stpauls'
    elif dataset == 'imc2024_dioscuri_baalshamin':
        return 'dioscuri_baalshamin'
    elif dataset == 'imc2024_lizard_pond':
        return 'lizard_pond'
    elif dataset == 'imc2023_haiper':
        return 'haiper'
    elif dataset == 'imc2023_heritage':
        return 'heritage'
    elif dataset == 'imc2023_theather_imc2024_church':
        return 'theather_imc2024_church'
    elif dataset == 'fbk_vineyard':
        return 'fbk_vineyard'
    elif dataset == 'pt_brandenburg_british_buckingham':
        return 'brandenburg_british_buckingham'
    elif dataset == 'pt_piazzasanmarco_grandplace':
        return 'piazzasanmarco_grandplace'
    elif dataset == 'pt_sacrecoeur_trevi_tajmahal':
        return 'sacrecoeur_trevi_tajmahal'
    return dataset

# Add inferred scene column
sample_submission['inferred_scene'] = sample_submission.apply(
    lambda row: infer_scene_from_image(row['dataset'], row['image']), axis=1
)

# Group images by dataset and inferred scene
grouped_images = sample_submission.groupby(['dataset', 'inferred_scene'])
print("Grouped dataset and inferred scene combinations:", list(grouped_images.groups.keys()))

# Initialize summary tracking
summary = {
    'total_groups': len(grouped_images),
    'successful_groups': 0,
    'failed_groups': [],
    'total_images': len(sample_submission),
    'computed_poses': 0,
    'sequential_poses': 0,
    'images_not_found': 0,
    'high_reprojection_errors': 0
}

# Phase 2: Feature Matching (Using SIFT)
def cluster_images(image_paths, scene_name, min_matches_threshold=10):
    """Cluster images using a sliding window approach for efficiency."""
    print(f"Clustering images for scene: {scene_name}")
    start_time = time.time()
    num_images = len(image_paths)
    if num_images < 2:
        print("Fewer than 2 images. Using sequential clustering.")
        return [list(range(num_images))], []

    window_size = 7
    edges = []

    # Create SIFT and FLANN matcher for clustering
    sift = cv2.SIFT_create(
        nfeatures=5000,  # Increased to get more features
        contrastThreshold=0.01,
        edgeThreshold=15
    )
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for i in range(num_images):
        for j in range(i + 1, min(i + window_size + 1, num_images)):
            img0 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(image_paths[j], cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                print(f"Failed to load images: {image_paths[i]} or {image_paths[j]}")
                continue
            img0 = cv2.equalizeHist(img0)
            img1 = cv2.equalizeHist(img1)
            img0 = cv2.resize(img0, (320, 240))
            img1 = cv2.resize(img1, (320, 240))
            kp0, des0 = sift.detectAndCompute(img0, None)
            kp1, des1 = sift.detectAndCompute(img1, None)
            if des0 is None or des1 is None:
                continue
            matches = flann.knnMatch(des0, des1, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m)
            num_matches = len(good_matches)
            if num_matches >= min_matches_threshold:
                edges.append((num_matches, i, j))

    if not edges:
        print("No matches found between any image pairs. Using sequential clustering.")
        return [list(range(num_images))], []

    # Build a graph and find connected components
    from collections import defaultdict
    graph = defaultdict(list)
    for _, i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    def find_component(start, graph, visited):
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(n for n in graph[node] if n not in visited)
        return component

    visited = set()
    clusters = []
    for i in range(num_images):
        if i not in visited:
            component = find_component(i, graph, visited)
            if len(component) >= 2:
                clusters.append(sorted(component))

    noise_indices = [i for i in range(num_images) if i not in visited]

    if not clusters:
        print("No clusters found. Using sequential clustering.")
        clusters = [list(range(num_images))]

    print(f"Clustering took {time.time() - start_time:.2f} seconds")
    return clusters, noise_indices

# Phase 3: Pose Estimation
def recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name):
    """Recover pose between two images with robust fallbacks."""
    if len(mkpts0) < 5:
        print(f"Too few matches ({len(mkpts0)}) between {img0_name} and {img1_name}. Cannot estimate pose.")
        return None, None, 0

    max_iters = 5000
    E, mask = cv2.findEssentialMat(
        mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5, maxIters=max_iters
    )
    if E is None or E.shape != (3, 3):
        print(f"Essential matrix estimation failed for pair {img0_name} and {img1_name}. Trying homography.")
    else:
        inliers, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K, mask=mask)
        if inliers >= 5:
            P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = K @ np.hstack((R, t.reshape(3, 1)))
            points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
            points3d = points4d[:3] / (points4d[3] + 1e-8)
            points3d_cam2 = R @ points3d + t.reshape(3, 1)
            in_front = (points3d[2, :] > 0) & (points3d_cam2[2, :] > 0)
            if np.sum(in_front) >= 0.5 * inliers:
                t = t.flatten()  # Ensure t is (3,)
                if validate_pose(R, t, f"{img0_name}-{img1_name}"):
                    return R, t, inliers

    print(f"Falling back to homography for pair {img0_name} and {img1_name}")
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 0.5, maxIters=5000)
    if H is None or H.shape != (3, 3):
        print(f"Homography estimation failed for pair {img0_name} and {img1_name}")
        return None, None, 0

    inliers = np.sum(mask)
    if inliers < 5:
        print(f"Too few inliers from homography for pair {img0_name} and {img1_name}: {inliers}")
        return None, None, 0

    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
    best_R, best_t, best_inliers = None, None, 0
    for i in range(num):
        R, t = Rs[i], ts[i]
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R, t))
        points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
        points3d = points4d[:3] / (points4d[3] + 1e-8)
        points3d_cam2 = R @ points3d + t.reshape(3, 1)
        in_front = (points3d[2, :] > 0) & (points3d_cam2[2, :] > 0)
        inliers_count = np.sum(in_front)
        if inliers_count > best_inliers:
            best_inliers = inliers_count
            best_R, best_t = R, t

    if best_R is None or best_t is None:
        print(f"No valid pose from homography for pair {img0_name} and {img1_name}")
        return None, None, 0

    best_t = best_t.flatten()  # Ensure t is (3,)
    if validate_pose(best_R, best_t, f"{img0_name}-{img1_name}"):
        return best_R, best_t, best_inliers
    return None, None, 0

def reprojection_error(params, points3d, observations, K, img_names, poses):
    """Compute reprojection error for bundle adjustment."""
    num_cameras = len(img_names)
    translations = params[:num_cameras * 3].reshape(num_cameras, 3)
    focal_length = params[num_cameras * 3]
    cx = params[num_cameras * 3 + 1]
    cy = params[num_cameras * 3 + 2]

    K_opt = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    errors = []
    for i, img_name in enumerate(img_names):
        R, _ = poses[img_name]
        t = translations[i]
        for j, obs in enumerate(observations):
            for img_obs, pt in obs:
                if img_obs == img_name:
                    proj = project(points3d[j:j+1], R, t, K_opt)
                    if not np.all(np.isfinite(proj)):
                        continue
                    errors.append(proj[0] - pt)

    if len(errors) == 0:
        print("No reprojection errors computed. Returning zeros.")
        return np.zeros(num_cameras * 3 + 3)
    errors = np.concatenate(errors)
    if not np.all(np.isfinite(errors)):
        return np.zeros_like(errors)
    return errors

# Phase 4: Feature Matching
def match_image_pair(args):
    """Match keypoints between a pair of images."""
    idx0, idx1, global_idx0, global_idx1, image_paths = args
    img0 = cv2.imread(image_paths[global_idx0], cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(image_paths[global_idx1], cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Failed to load images: {image_paths[global_idx0]} or {image_paths[global_idx1]}")
        return idx0, idx1, None, None, None, None

    img0 = cv2.equalizeHist(img0)
    img1 = cv2.equalizeHist(img1)

    orig_size0 = img0.shape[::-1]
    orig_size1 = img1.shape[::-1]

    img0_resized = cv2.resize(img0, (320, 240))
    img1_resized = cv2.resize(img1, (320, 240))

    # Create SIFT and FLANN matcher inside the function
    sift = cv2.SIFT_create(
        nfeatures=5000,  # Increased to get more features
        contrastThreshold=0.01,
        edgeThreshold=15
    )
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp0, des0 = sift.detectAndCompute(img0_resized, None)
    kp1, des1 = sift.detectAndCompute(img1_resized, None)
    if des0 is None or des1 is None:
        return idx0, idx1, None, None, None, None

    matches = flann.knnMatch(des0, des1, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    if not good_matches:
        print(f"No good matches between {Path(image_paths[global_idx0]).name} and {Path(image_paths[global_idx1]).name}")
        return idx0, idx1, None, None, None, None

    mkpts0 = np.array([kp0[m.queryIdx].pt for m in good_matches])
    mkpts1 = np.array([kp1[m.trainIdx].pt for m in good_matches])
    kpts0 = np.array([kp.pt for kp in kp0])
    kpts1 = np.array([kp.pt for kp in kp1])

    scale0 = (orig_size0[0] / 320, orig_size0[1] / 240)
    scale1 = (orig_size1[0] / 320, orig_size1[1] / 240)
    kpts0[:, 0] *= scale0[0]
    kpts0[:, 1] *= scale0[1]
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]
    mkpts0[:, 0] *= scale0[0]
    mkpts0[:, 1] *= scale0[1]
    mkpts1[:, 0] *= scale1[0]
    mkpts1[:, 1] *= scale1[1]

    img0_name = Path(image_paths[global_idx0]).name
    img1_name = Path(image_paths[global_idx1]).name

    return idx0, idx1, img0_name, img1_name, (kpts0, kpts1), (mkpts0, mkpts1)

def run_sfm_on_cluster(args):
    """Run Structure-from-Motion (SfM) on a cluster of images using incremental SfM."""
    cluster_indices, image_paths, scene_name, cluster_idx, clusters, overlap = args
    print(f"Starting cluster {cluster_idx} for scene {scene_name}")

    if not cluster_indices or not image_paths:
        print("Empty cluster or image paths. Using sequential poses.")
        poses = {}
        for idx in range(len(cluster_indices)):
            if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
                continue
            img_name = Path(image_paths[cluster_indices[idx]]).name
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return poses, np.array([], dtype=np.float32).reshape(0, 3), K, []

    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    point_to_images = []

    start_time = time.time()
    window_size = 7
    image_pairs = []
    for i in range(len(cluster_indices)):
        for j in range(i + 1, min(i + window_size + 1, len(cluster_indices))):
            image_pairs.append((i, j))

    edges = []
    max_workers = min(multiprocessing.cpu_count(), 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(match_image_pair, (idx0, idx1, cluster_indices[idx0], cluster_indices[idx1], image_paths))
            for idx0, idx1 in image_pairs
        ]
        for future in futures:
            idx0, idx1, img0_name, img1_name, keypoints, matches = future.result()
            if matches is None:
                continue
            kpts0, kpts1 = keypoints
            mkpts0, mkpts1 = matches
            if img0_name not in keypoints_dict:
                keypoints_dict[img0_name] = kpts0
                image_sizes[img0_name] = (kpts0.shape[0] * 320 / mkpts0.shape[0], kpts0.shape[1] * 240 / mkpts0.shape[1]) if mkpts0.shape[0] > 0 else (320, 240)
            if img1_name not in keypoints_dict:
                keypoints_dict[img1_name] = kpts1
                image_sizes[img1_name] = (kpts1.shape[0] * 320 / mkpts1.shape[0], kpts1.shape[1] * 240 / mkpts1.shape[1]) if mkpts1.shape[0] > 0 else (320, 240)
            matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
            priority = 0
            if cluster_idx < len(clusters) - 1:
                overlap_indices = clusters[cluster_idx][-overlap:]
                if cluster_indices[idx0] in overlap_indices or cluster_indices[idx1] in overlap_indices:
                    priority = 1
            edges.append((len(mkpts0), idx0, idx1, priority))
            print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")

    print(f"Feature matching for cluster took {time.time() - start_time:.2f} seconds.")

    edges.sort(reverse=True)
    poses = {}
    added_images = set()

    # Initialize the first pair
    for num_matches, idx0, idx1, priority in edges:
        if num_matches < 5:
            continue
        if idx0 >= len(cluster_indices) or idx1 >= len(cluster_indices):
            continue
        global_idx0 = cluster_indices[idx0]
        global_idx1 = cluster_indices[idx1]
        if global_idx0 >= len(image_paths) or global_idx1 >= len(image_paths):
            continue
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
        image_width, image_height = image_sizes[img0_name]
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            continue
        t = normalize_translation(t, img0_name)
        poses[img0_name] = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        poses[img1_name] = (R, t)
        added_images.update([img0_name, img1_name])
        break

    if not poses:
        print(f"No initial pair found for cluster {cluster_idx}. Using sequential poses.")
        for idx in range(len(cluster_indices)):
            if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
                continue
            img_name = Path(image_paths[cluster_indices[idx]]).name
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
            added_images.add(img_name)
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return poses, np.array([], dtype=np.float32).reshape(0, 3), K, []

    # Incremental SfM: Add images one at a time
    points3d = []
    point_to_images = []
    for idx in range(len(cluster_indices)):
        if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
            continue
        global_idx = cluster_indices[idx]
        img_name = Path(image_paths[global_idx]).name
        if img_name in added_images:
            continue

        # Find the best image to match against
        best_pair = None
        best_num_matches = 0
        best_idx0 = None
        for idx0 in range(len(cluster_indices)):
            if idx0 >= len(cluster_indices) or cluster_indices[idx0] >= len(image_paths):
                continue
            img0_name = Path(image_paths[cluster_indices[idx0]]).name
            if img0_name not in added_images:
                continue
            # Check both (img0_name, img_name) and (img_name, img0_name) for matches
            pair_key = (img0_name, img_name) if (img0_name, img_name) in matches_dict else (img_name, img0_name)
            if pair_key not in matches_dict:
                # Match on-the-fly if not already matched
                _, _, _, _, _, matches = match_image_pair((idx0, idx, cluster_indices[idx0], global_idx, image_paths))
                if matches is None:
                    matches_dict[(img0_name, img_name)] = (np.array([]), np.array([]))
                    num_matches = 0
                else:
                    mkpts0, mkpts1 = matches
                    matches_dict[(img0_name, img_name)] = (mkpts0, mkpts1)
                    num_matches = len(mkpts0)
            else:
                mkpts0, mkpts1 = matches_dict[pair_key]
                num_matches = len(mkpts0)

            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_pair = (img0_name, img_name)
                best_idx0 = idx0

        if best_pair is None or best_num_matches < 5:
            print(f"Could not add {img_name} to cluster {cluster_idx}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
            added_images.add(img_name)
            continue

        img0_name, img1_name = best_pair
        # Use pair_key to access matches_dict
        pair_key = (img0_name, img1_name) if (img0_name, img1_name) in matches_dict else (img1_name, img0_name)
        mkpts0, mkpts1 = matches_dict[pair_key]
        if len(mkpts0) < 5:
            print(f"Too few matches ({len(mkpts0)}) for pair {img0_name} and {img1_name}. Using sequential pose for {img1_name}.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img1_name] = (R, t)
            added_images.add(img1_name)
            continue

        R0, t0 = poses[img0_name]
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            print(f"Failed to estimate pose for {img1_name}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img1_name] = (R, t)
            added_images.add(img1_name)
            continue

        t = normalize_translation(t, img1_name)
        t0 = normalize_translation(t0, img0_name)
        R = R0 @ R
        t = R0 @ t + t0
        t = normalize_translation(t, img1_name)
        if not validate_pose(R, t, img1_name):
            print(f"Invalid pose for {img1_name}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
        poses[img1_name] = (R, t)
        added_images.add(img1_name)

        # Triangulate points with the best pair
        if len(mkpts0) >= 5:
            R1, t1 = poses[img1_name]
            t0 = normalize_translation(t0, img0_name)
            t1 = normalize_translation(t1, img1_name)
            if validate_pose(R0, t0, img0_name) and validate_pose(R1, t1, img1_name):
                P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
                P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
                points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
                points4d = points4d[:3] / (points4d[3] + 1e-8)
                points4d = points4d.T
                if np.all(np.isfinite(points4d)):
                    for i in range(len(points4d)):
                        points3d.append(points4d[i])
                        point_to_images.append([(img0_name, mkpts0[i]), (img1_name, mkpts1[i])])

    points3d = np.array(points3d, dtype=np.float32) if points3d else np.array([], dtype=np.float32).reshape(0, 3)
    print(f"Reconstructed {len(points3d)} 3D points in cluster {cluster_idx}")

    # Local Bundle Adjustment (skip for large clusters to save time)
    if len(poses) >= 3 and len(point_to_images) >= 10 and len(poses) < 20:
        start_time = time.time()
        img_names = list(poses.keys())
        translations = np.array([normalize_translation(poses[img_name][1], img_name) for img_name in img_names], dtype=np.float32)
        if translations.shape != (len(img_names), 3):
            print(f"Mismatch in translations shape: {translations.shape}. Expected ({len(img_names)}, 3). Skipping bundle adjustment.")
        else:
            initial_focal_length = K[0, 0]
            initial_cx, initial_cy = K[0, 2], K[1, 2]
            params = np.hstack((translations.ravel(), initial_focal_length, initial_cx, initial_cy))
            max_nfev = min(20, len(poses) * 5)
            result = least_squares(
                reprojection_error,
                params,
                args=(points3d, point_to_images, K, img_names, poses),
                max_nfev=max_nfev,
                ftol=1e-5,
                xtol=1e-5
            )
            optimized_params = result.x
            num_cameras = len(img_names)
            translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
            focal_length = optimized_params[num_cameras * 3]
            cx = optimized_params[num_cameras * 3 + 1]
            cy = optimized_params[num_cameras * 3 + 2]
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            for i, img_name in enumerate(img_names):
                R, _ = poses[img_name]
                t = translations[i]
                t = normalize_translation(t, img_name)
                if not validate_pose(R, t, img_name):
                    R = np.eye(3, dtype=np.float32)
                    t = np.array([0.0, 0.0, len(poses) * 0.1], dtype=np.float32)
                poses[img_name] = (R, t)
            print(f"Local bundle adjustment took {time.time() - start_time:.2f} seconds.")
    else:
        print("Skipping local bundle adjustment due to insufficient poses/observations or large cluster size.")

    print(f"Cluster {cluster_idx} processed {len(added_images)} cameras")
    return poses, points3d, K, point_to_images

# Phase 5: Reprojection Error
def compute_reprojection_error(img_name, points3d, observations, R, t, K):
    """Compute the average reprojection error for an image."""
    errors = []
    for j, obs in enumerate(observations):
        for img_obs, pt in obs:
            if img_obs == img_name:
                proj = project(points3d[j:j+1], R, t, K)
                if not np.all(np.isfinite(proj)):
                    continue
                error = np.linalg.norm(proj[0] - pt)
                if np.isfinite(error):
                    errors.append(error)
    return np.mean(errors) if errors else float('inf')

# Phase 6: Main Loop and Submission
all_poses = {}
all_Ks = {}
processed_images = set()
pose_status = {}

for (dataset, inferred_scene), group in grouped_images:
    print(f"\nProcessing dataset: {dataset}, inferred scene: {inferred_scene}")
    start_time = time.time()

    image_dir = f"/kaggle/input/image-matching-challenge-2025/test/{dataset}"
    if not os.path.exists(image_dir):
        print(f"Dataset directory {image_dir} does not exist in test set. Skipping.")
        summary['failed_groups'].append((dataset, inferred_scene, "Dataset not found in test directory"))
        summary['images_not_found'] += len(group)
        continue

    image_paths = []
    for _, row in group.iterrows():
        img_name = row['image']
        img_path = f"{image_dir}/{img_name}"
        if Path(img_path).exists():
            image_paths.append(img_path)
            processed_images.add(img_name)
        else:
            print(f"Image not found: {img_path}")
            summary['images_not_found'] += 1

    if not image_paths:
        print(f"No images found for dataset {dataset}, scene {inferred_scene}. Skipping.")
        summary['failed_groups'].append((dataset, inferred_scene, "No images found"))
        continue

    if len(image_paths) < 2:
        print(f"Dataset {dataset}, scene {inferred_scene} has fewer than 2 images. Using sequential poses.")
        for idx, img_path in enumerate(image_paths):
            img_name = Path(img_path).name
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
        summary['failed_groups'].append((dataset, inferred_scene, "Fewer than 2 images"))
        continue

    clusters, noise_indices = cluster_images(image_paths, inferred_scene, min_matches_threshold=10)
    print(f"Clusters: {clusters}")
    print(f"Outlier images: {[Path(image_paths[idx]).name for idx in noise_indices if idx < len(image_paths)]}")

    overlap = 2

    all_points3d = []
    all_observations = []

    print(f"Processing clusters for dataset {dataset}, scene {inferred_scene}")
    for cluster_idx, cluster_indices in enumerate(clusters):
        result = run_sfm_on_cluster((cluster_indices, image_paths, inferred_scene, cluster_idx, clusters, overlap))
        poses, points3d, K, cluster_point_to_images = result
        for img_name in poses:
            R, t = poses[img_name]
            t = normalize_translation(t, img_name)
            if not validate_pose(R, t, img_name):
                print(f"Invalid pose for {img_name} after SfM. Using sequential pose.")
                R = np.eye(3, dtype=np.float32)
                t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            all_Ks[img_name] = K
            pose_status[img_name] = 'computed'
            summary['computed_poses'] += 1
        if len(points3d) > 0:
            all_points3d.append(points3d)
        all_observations.extend(cluster_point_to_images)

    # Handle outliers incrementally
    for idx in noise_indices:
        if idx >= len(image_paths):
            continue
        img_name = Path(image_paths[idx]).name
        best_pair = None
        best_num_matches = 0
        for i in range(max(0, idx - 3), min(len(image_paths), idx + 4)):
            if i == idx or Path(image_paths[i]).name not in all_poses:
                continue
            img0_name = Path(image_paths[i]).name
            pair_key = (img0_name, img_name) if (img0_name, img_name) in matches_dict else (img_name, img0_name)
            if pair_key not in matches_dict:
                _, _, _, _, _, matches = match_image_pair((i, idx, i, idx, image_paths))
                if matches is None:
                    matches_dict[(img0_name, img_name)] = (np.array([]), np.array([]))
                    num_matches = 0
                else:
                    mkpts0, mkpts1 = matches
                    matches_dict[(img0_name, img_name)] = (mkpts0, mkpts1)
                    num_matches = len(mkpts0)
            else:
                mkpts0, mkpts1 = matches_dict[pair_key]
                num_matches = len(mkpts0)
            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_pair = (img0_name, img_name)

        if best_pair is None or best_num_matches < 5:
            print(f"Could not register outlier image {img_name}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
            continue

        img0_name, img1_name = best_pair
        pair_key = (img0_name, img1_name) if (img0_name, img1_name) in matches_dict else (img1_name, img0_name)
        mkpts0, mkpts1 = matches_dict[pair_key]
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            print(f"Failed to register outlier image {img_name}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
            continue
        R0, t0 = all_poses[img0_name]
        t0 = normalize_translation(t0, img0_name)
        t = normalize_translation(t, img1_name)
        R = R0 @ R
        t = R0 @ t + t0
        t = normalize_translation(t, img1_name)
        if not validate_pose(R, t, img1_name):
            print(f"Invalid pose for outlier {img1_name}. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
        all_poses[img_name] = (R, t)
        all_Ks[img_name] = K
        pose_status[img_name] = 'computed'
        summary['computed_poses'] += 1

    if all_points3d:
        merged_points3d = np.concatenate(all_points3d, axis=0)
    else:
        merged_points3d = np.array([], dtype=np.float32).reshape(0, 3)

    print("Skipping global bundle adjustment to save time.")

    reproj_threshold = 10.0
    for img_name in list(all_poses.keys()):
        R, t = all_poses[img_name]
        t = normalize_translation(t, img_name)
        if not validate_pose(R, t, img_name):
            print(f"Invalid pose for {img_name} before reprojection check. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
        if img_name not in all_Ks:
            print(f"Camera intrinsics (K) missing for {img_name}. Assigning default K.")
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
        K = all_Ks[img_name]
        error = compute_reprojection_error(img_name, merged_points3d, all_observations, R, t, K)
        if error > reproj_threshold:
            print(f"Image {img_name} has high reprojection error ({error:.2f}). Resetting to sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
            summary['high_reprojection_errors'] += 1
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K

    for img_path in image_paths:
        img_name = Path(img_path).name
        if img_name not in all_poses:
            print(f"Image {img_name} was not processed. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1

    print(f"Dataset {dataset}, scene {inferred_scene} processed in {time.time() - start_time:.2f} seconds")
    summary['successful_groups'] += 1

# Handle unprocessed images
unprocessed_images = set(sample_submission['image']) - processed_images
if unprocessed_images:
    print(f"Warning: The following images were not processed: {unprocessed_images}")
    for img_name in unprocessed_images:
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
        all_poses[img_name] = (R, t)
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        all_Ks[img_name] = K
        pose_status[img_name] = 'sequential'
        summary['sequential_poses'] += 1

# Generate submission file
submission_rows = []
for _, row in sample_submission.iterrows():
    img_path = row['image_id']
    img_name = row['image']
    if img_name in all_poses:
        R, t = all_poses[img_name]
        t = normalize_translation(t, img_name)
        if not validate_pose(R, t, img_name):
            print(f"Invalid pose for {img_name} in submission. Using sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float32)
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
        q = rotation_matrix_to_quaternion(R)
        t = t.reshape(3)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)):
            print(f"Non-finite pose for {img_name}. Using sequential pose.")
            q = rotation_matrix_to_quaternion(np.eye(3))
            t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float64)
            pose_status[img_name] = 'sequential'
            summary['sequential_poses'] += 1
    else:
        print(f"Image {img_name} not found in poses. Using sequential pose.")
        q = rotation_matrix_to_quaternion(np.eye(3))
        t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float64)
        pose_status[img_name] = 'sequential'
        summary['sequential_poses'] += 1
    submission_rows.append([img_path] + q.tolist() + t.tolist())

submission_columns = [
    'image_id',
    'rotation_w', 'rotation_x', 'rotation_y', 'rotation_z',
    'translation_x', 'translation_y', 'translation_z'
]
submission_df = pd.DataFrame(submission_rows, columns=submission_columns)
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Validate submission
print("Validating submission...")
assert len(submission_df) == len(sample_submission), f"Submission has {len(submission_df)} rows, expected {len(sample_submission)}"
assert set(submission_df.columns) == set(submission_columns), "Submission columns do not match expected columns"

for i, row in submission_df.iterrows():
    q = np.array([
        row['rotation_w'], row['rotation_x'], row['rotation_y'], row['rotation_z']
    ], dtype=np.float64)
    t = np.array([
        row['translation_x'], row['translation_y'], row['translation_z']
    ], dtype=np.float64)
    assert not np.any(np.isnan(q)), f"NaN values in quaternion at row {i}"
    assert not np.any(np.isnan(t)), f"NaN values in translation at row {i}"
    assert not np.any(np.isinf(q)), f"Infinite values in quaternion at row {i}"
    assert not np.any(np.isinf(t)), f"Infinite values in translation at row {i}"
    norm = np.linalg.norm(q)
    assert abs(norm - 1.0) < 1e-6, f"Quaternion not normalized at row {i}: norm={norm}"
    R = quaternion_to_rotation_matrix(q)
    det_R = np.linalg.det(R)
    assert abs(det_R - 1.0) < 1e-6, f"Rotation matrix determinant not 1 at row {i}: det={det_R}"
    orthogonality = np.linalg.norm(R.T @ R - np.eye(3))
    assert orthogonality < 1e-6, f"Rotation matrix not orthogonal at row {i}: orthogonality={orthogonality}"

print("Submission validated successfully.")

# Print summary
print("\n=== Processing Summary ===")
print(f"Total dataset/scene groups: {summary['total_groups']}")
print(f"Successful groups: {summary['successful_groups']}")
print(f"Failed groups: {summary['failed_groups']}")
print(f"Total images: {summary['total_images']}")
print(f"Computed poses: {summary['computed_poses']}")
print(f"Sequential poses: {summary['sequential_poses']}")
print(f"Images not found: {summary['images_not_found']}")
print(f"High reprojection errors: {summary['high_reprojection_errors']}")


# Phase 1: Setup and Initial Functions
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# Utility functions
def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    if not np.all(np.isfinite(R)) or R.shape != (3, 3):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # Reorder to [w, x, y, z]

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    if not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64)
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # Reorder [x, y, z, w]
    return rot.as_matrix()

def project(points3d, R, t, K):
    """Project 3D points to 2D using camera intrinsics and extrinsics."""
    if not (np.all(np.isfinite(points3d)) and np.all(np.isfinite(R)) and np.all(np.isfinite(t)) and np.all(np.isfinite(K))):
        return np.zeros((points3d.shape[0], 2), dtype=np.float32)
    points3d = points3d.T
    P = K @ np.hstack((R, t.reshape(3, 1)))
    points2d = P @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
    points2d = points2d[:2] / (points2d[2] + 1e-8)
    return points2d.T

def validate_pose(R, t, img_name="Unknown"):
    """Validate that R and t have the correct shapes and are finite."""
    if R.shape != (3, 3):
        print(f"Invalid rotation matrix shape for {img_name}: {R.shape}")
        return False
    if t.shape != (3,):
        print(f"Invalid translation vector shape for {img_name}: {t.shape}")
        return False
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        print(f"Non-finite values in pose for {img_name}: R={R}, t={t}")
        return False
    return True

def normalize_translation(t, img_name="Unknown"):
    """Ensure translation vector is a 1D array of shape (3,)."""
    if t is None:
        print(f"Translation vector is None for {img_name}. Using default.")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    t = np.array(t, dtype=np.float32).flatten()
    if t.size != 3:
        print(f"Cannot normalize translation vector for {img_name}: size={t.size}")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return t

# Load sample submission
sample_submission = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
print("Columns in sample_submission.csv:", sample_submission.columns.tolist())

image_path_column = 'image'
if image_path_column not in sample_submission.columns:
    raise ValueError("Expected 'image' column in sample_submission.csv")

# Function to infer the correct scene
def infer_scene_from_image(dataset, image_name):
    if dataset == 'ETs':
        if image_name.startswith('et_et') or image_name.startswith('another_et_another_et') or image_name.startswith('outliers_out_et'):
            return 'et'
    elif dataset == 'amy_gardens':
        if image_name.startswith('peach_'):
            return 'peach'
    elif dataset == 'stairs':
        if image_name.startswith('stairs_split_1_'):
            return 'stairs_split_1'
        elif image_name.startswith('stairs_split_2_'):
            return 'stairs_split_2'
    elif dataset == 'pt_stpeters_stpauls':
        if image_name.startswith('st_peters_square_'):
            return 'st_peters_square'
        return 'pt_stpeters_stpauls'
    elif dataset == 'imc2024_dioscuri_baalshamin':
        return 'dioscuri_baalshamin'
    elif dataset == 'imc2024_lizard_pond':
        return 'lizard_pond'
    elif dataset == 'imc2023_haiper':
        return 'haiper'
    elif dataset == 'imc2023_heritage':
        return 'heritage'
    elif dataset == 'imc2023_theather_imc2024_church':
        return 'theather_imc2024_church'
    elif dataset == 'fbk_vineyard':
        return 'fbk_vineyard'
    elif dataset == 'pt_brandenburg_british_buckingham':
        return 'brandenburg_british_buckingham'
    elif dataset == 'pt_piazzasanmarco_grandplace':
        return 'piazzasanmarco_grandplace'
    elif dataset == 'pt_sacrecoeur_trevi_tajmahal':
        return 'sacrecoeur_trevi_tajmahal'
    return dataset

# Add inferred scene column
sample_submission['inferred_scene'] = sample_submission.apply(
    lambda row: infer_scene_from_image(row['dataset'], row['image']), axis=1
)

# Group images by dataset and inferred scene
grouped_images = sample_submission.groupby(['dataset', 'inferred_scene'])
print("Grouped dataset and inferred scene combinations:", list(grouped_images.groups.keys()))

# Initialize summary tracking
summary = {
    'total_groups': len(grouped_images),
    'successful_groups': 0,
    'failed_groups': [],
    'total_images': len(sample_submission),
    'computed_poses': 0,
    'sequential_poses': 0,
    'sequential_poses_due_to_missing_data': 0,
    'sequential_poses_due_to_processing': 0,
    'images_not_found': 0,
    'high_reprojection_errors': 0
}

# Track processed images to avoid double-counting
counted_images_missing = set()
counted_images_processing = set()

# Phase 2: Feature Matching (Using SIFT)
def cluster_images(image_paths, scene_name, min_matches_threshold=5):
    """Cluster images using a sliding window approach for efficiency."""
    print(f"Clustering images for scene: {scene_name}")
    start_time = time.time()
    num_images = len(image_paths)
    if num_images < 2:
        print("Fewer than 2 images. Using sequential clustering.")
        return [list(range(num_images))], []

    window_size = 10
    edges = []

    sift = cv2.SIFT_create(
        nfeatures=5000,
        contrastThreshold=0.01,
        edgeThreshold=15
    )
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for i in range(num_images):
        for j in range(i + 1, min(i + window_size + 1, num_images)):
            img0 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(image_paths[j], cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                print(f"Failed to load images: {image_paths[i]} or {image_paths[j]}")
                continue
            img0 = cv2.equalizeHist(img0)
            img1 = cv2.equalizeHist(img1)
            img0 = cv2.resize(img0, (320, 240))
            img1 = cv2.resize(img1, (320, 240))
            kp0, des0 = sift.detectAndCompute(img0, None)
            kp1, des1 = sift.detectAndCompute(img1, None)
            if des0 is None or des1 is None:
                continue
            matches = flann.knnMatch(des0, des1, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:  # Reverted to 0.6
                    good_matches.append(m)
            num_matches = len(good_matches)
            if num_matches >= min_matches_threshold:
                edges.append((num_matches, i, j))

    if not edges:
        print("No matches found between any image pairs. Using sequential clustering.")
        return [list(range(num_images))], []

    from collections import defaultdict
    graph = defaultdict(list)
    for _, i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    def find_component(start, graph, visited):
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(n for n in graph[node] if n not in visited)
        return component

    visited = set()
    clusters = []
    for i in range(num_images):
        if i not in visited:
            component = find_component(i, graph, visited)
            if len(component) >= 2:
                clusters.append(sorted(component))

    noise_indices = [i for i in range(num_images) if i not in visited]

    if not clusters:
        print("No clusters found. Using sequential clustering.")
        clusters = [list(range(num_images))]

    print(f"Clustering took {time.time() - start_time:.2f} seconds")
    return clusters, noise_indices

# Phase 3: Pose Estimation
def recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name):
    """Recover pose between two images with robust fallbacks."""
    if len(mkpts0) < 5:
        print(f"Too few matches ({len(mkpts0)}) between {img0_name} and {img1_name}. Cannot estimate pose.")
        return None, None, 0

    max_iters = 5000
    E, mask = cv2.findEssentialMat(
        mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5, maxIters=max_iters
    )
    if E is None or E.shape != (3, 3):
        print(f"Essential matrix estimation failed for pair {img0_name} and {img1_name}. Trying homography.")
    else:
        inliers, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K, mask=mask)
        if inliers >= 5:
            P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = K @ np.hstack((R, t.reshape(3, 1)))
            points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
            points3d = points4d[:3] / (points4d[3] + 1e-8)
            points3d_cam2 = R @ points3d + t.reshape(3, 1)
            in_front = (points3d[2, :] > 0) & (points3d_cam2[2, :] > 0)
            if np.sum(in_front) >= 0.5 * inliers:
                t = t.flatten()
                if validate_pose(R, t, f"{img0_name}-{img1_name}"):
                    return R, t, inliers

    print(f"Falling back to homography for pair {img0_name} and {img1_name}")
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 0.5, maxIters=5000)
    if H is None or H.shape != (3, 3):
        print(f"Homography estimation failed for pair {img0_name} and {img1_name}")
        return None, None, 0

    inliers = np.sum(mask)
    if inliers < 5:
        print(f"Too few inliers from homography for pair {img0_name} and {img1_name}: {inliers}")
        return None, None, 0

    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
    best_R, best_t, best_inliers = None, None, 0
    for i in range(num):
        R, t = Rs[i], ts[i]
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R, t))
        points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
        points3d = points4d[:3] / (points4d[3] + 1e-8)
        points3d_cam2 = R @ points3d + t.reshape(3, 1)
        in_front = (points3d[2, :] > 0) & (points3d_cam2[2, :] > 0)
        inliers_count = np.sum(in_front)
        if inliers_count > best_inliers:
            best_inliers = inliers_count
            best_R, best_t = R, t

    if best_R is None or best_t is None:
        print(f"No valid pose from homography for pair {img0_name} and {img1_name}")
        return None, None, 0

    best_t = best_t.flatten()
    if validate_pose(best_R, best_t, f"{img0_name}-{img1_name}"):
        return best_R, best_t, best_inliers
    return None, None, 0

def reprojection_error(params, points3d, observations, K, img_names, poses):
    """Compute reprojection error for bundle adjustment."""
    num_cameras = len(img_names)
    translations = params[:num_cameras * 3].reshape(num_cameras, 3)
    focal_length = params[num_cameras * 3]
    cx = params[num_cameras * 3 + 1]
    cy = params[num_cameras * 3 + 2]

    K_opt = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    errors = []
    for i, img_name in enumerate(img_names):
        R, _ = poses[img_name]
        t = translations[i]
        for j, obs in enumerate(observations):
            for img_obs, pt in obs:
                if img_obs == img_name:
                    proj = project(points3d[j:j+1], R, t, K_opt)
                    if not np.all(np.isfinite(proj)):
                        continue
                    errors.append(proj[0] - pt)

    if len(errors) == 0:
        print("No reprojection errors computed. Returning zeros.")
        return np.zeros(num_cameras * 3 + 3)
    errors = np.concatenate(errors)
    if not np.all(np.isfinite(errors)):
        return np.zeros_like(errors)
    return errors

# Phase 4: Feature Matching
def match_image_pair(args):
    """Match keypoints between a pair of images."""
    idx0, idx1, global_idx0, global_idx1, image_paths = args
    img0 = cv2.imread(image_paths[global_idx0], cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(image_paths[global_idx1], cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Failed to load images: {image_paths[global_idx0]} or {image_paths[global_idx1]}")
        return idx0, idx1, None, None, None, None

    img0 = cv2.equalizeHist(img0)
    img1 = cv2.equalizeHist(img1)

    orig_size0 = img0.shape[::-1]
    orig_size1 = img1.shape[::-1]

    img0_resized = cv2.resize(img0, (320, 240))
    img1_resized = cv2.resize(img1, (320, 240))

    sift = cv2.SIFT_create(
        nfeatures=5000,
        contrastThreshold=0.01,
        edgeThreshold=15
    )
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp0, des0 = sift.detectAndCompute(img0_resized, None)
    kp1, des1 = sift.detectAndCompute(img1_resized, None)
    if des0 is None or des1 is None:
        return idx0, idx1, None, None, None, None

    matches = flann.knnMatch(des0, des1, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # Reverted to 0.6
            good_matches.append(m)

    if not good_matches:
        print(f"No good matches between {Path(image_paths[global_idx0]).name} and {Path(image_paths[global_idx1]).name}")
        return idx0, idx1, None, None, None, None

    mkpts0 = np.array([kp0[m.queryIdx].pt for m in good_matches])
    mkpts1 = np.array([kp1[m.trainIdx].pt for m in good_matches])
    kpts0 = np.array([kp.pt for kp in kp0])
    kpts1 = np.array([kp.pt for kp in kp1])

    scale0 = (orig_size0[0] / 320, orig_size0[1] / 240)
    scale1 = (orig_size1[0] / 320, orig_size1[1] / 240)
    kpts0[:, 0] *= scale0[0]
    kpts0[:, 1] *= scale0[1]
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]
    mkpts0[:, 0] *= scale0[0]
    mkpts0[:, 1] *= scale0[1]
    mkpts1[:, 0] *= scale1[0]
    mkpts1[:, 1] *= scale1[1]

    img0_name = Path(image_paths[global_idx0]).name
    img1_name = Path(image_paths[global_idx1]).name

    return idx0, idx1, img0_name, img1_name, (kpts0, kpts1), (mkpts0, mkpts1)

def run_sfm_on_cluster(args):
    """Run Structure-from-Motion (SfM) on a cluster of images using incremental SfM."""
    cluster_indices, image_paths, scene_name, cluster_idx, clusters, overlap = args
    print(f"Starting cluster {cluster_idx} for scene {scene_name}")

    if not cluster_indices or not image_paths:
        print("Empty cluster or image paths. Using sequential poses.")
        poses = {}
        for idx in range(len(cluster_indices)):
            if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
                continue
            img_name = Path(image_paths[cluster_indices[idx]]).name
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return poses, np.array([], dtype=np.float32).reshape(0, 3), K, []

    keypoints_dict = {}
    matches_dict = {}
    image_sizes = {}
    point_to_images = []

    start_time = time.time()
    window_size = 7
    image_pairs = []
    for i in range(len(cluster_indices)):
        for j in range(i + 1, min(i + window_size + 1, len(cluster_indices))):
            image_pairs.append((i, j))

    edges = []
    max_workers = min(multiprocessing.cpu_count(), 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(match_image_pair, (idx0, idx1, cluster_indices[idx0], cluster_indices[idx1], image_paths))
            for idx0, idx1 in image_pairs
        ]
        for future in futures:
            idx0, idx1, img0_name, img1_name, keypoints, matches = future.result()
            if matches is None:
                continue
            kpts0, kpts1 = keypoints
            mkpts0, mkpts1 = matches
            if img0_name not in keypoints_dict:
                keypoints_dict[img0_name] = kpts0
                image_sizes[img0_name] = (kpts0.shape[0] * 320 / mkpts0.shape[0], kpts0.shape[1] * 240 / mkpts0.shape[1]) if mkpts0.shape[0] > 0 else (320, 240)
            if img1_name not in keypoints_dict:
                keypoints_dict[img1_name] = kpts1
                image_sizes[img1_name] = (kpts1.shape[0] * 320 / mkpts1.shape[0], kpts1.shape[1] * 240 / mkpts1.shape[1]) if mkpts1.shape[0] > 0 else (320, 240)
            matches_dict[(img0_name, img1_name)] = (mkpts0, mkpts1)
            priority = 0
            if cluster_idx < len(clusters) - 1:
                overlap_indices = clusters[cluster_idx][-overlap:]
                if cluster_indices[idx0] in overlap_indices or cluster_indices[idx1] in overlap_indices:
                    priority = 1
            edges.append((len(mkpts0), idx0, idx1, priority))
            print(f"Matches between {img0_name} and {img1_name}: {len(mkpts0)}")

    print(f"Feature matching for cluster took {time.time() - start_time:.2f} seconds.")

    edges.sort(reverse=True)
    poses = {}
    added_images = set()

    for num_matches, idx0, idx1, priority in edges:
        if num_matches < 5:
            continue
        if idx0 >= len(cluster_indices) or idx1 >= len(cluster_indices):
            continue
        global_idx0 = cluster_indices[idx0]
        global_idx1 = cluster_indices[idx1]
        if global_idx0 >= len(image_paths) or global_idx1 >= len(image_paths):
            continue
        img0_name = Path(image_paths[global_idx0]).name
        img1_name = Path(image_paths[global_idx1]).name
        mkpts0, mkpts1 = matches_dict[(img0_name, img1_name)]
        image_width, image_height = image_sizes[img0_name]
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            continue
        t = normalize_translation(t, img0_name)
        poses[img0_name] = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        poses[img1_name] = (R, t)
        added_images.update([img0_name, img1_name])
        break

    if not poses:
        print(f"No initial pair found for cluster {cluster_idx}. Using sequential poses.")
        for idx in range(len(cluster_indices)):
            if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
                continue
            img_name = Path(image_paths[cluster_indices[idx]]).name
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
            added_images.add(img_name)
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return poses, np.array([], dtype=np.float32).reshape(0, 3), K, []

    points3d = []
    point_to_images = []
    for idx in range(len(cluster_indices)):
        if idx >= len(cluster_indices) or cluster_indices[idx] >= len(image_paths):
            continue
        global_idx = cluster_indices[idx]
        img_name = Path(image_paths[global_idx]).name
        if img_name in added_images:
            continue

        best_pair = None
        best_num_matches = 0
        best_idx0 = None
        for idx0 in range(len(cluster_indices)):
            if idx0 >= len(cluster_indices) or cluster_indices[idx0] >= len(image_paths):
                continue
            img0_name = Path(image_paths[cluster_indices[idx0]]).name
            if img0_name not in added_images:
                continue
            pair_key = (img0_name, img_name) if (img0_name, img_name) in matches_dict else (img_name, img0_name)
            if pair_key not in matches_dict:
                _, _, _, _, _, matches = match_image_pair((idx0, idx, cluster_indices[idx0], global_idx, image_paths))
                if matches is None:
                    matches_dict[(img0_name, img_name)] = (np.array([]), np.array([]))
                    num_matches = 0
                else:
                    mkpts0, mkpts1 = matches
                    matches_dict[(img0_name, img_name)] = (mkpts0, mkpts1)
                    num_matches = len(mkpts0)
            else:
                mkpts0, mkpts1 = matches_dict[pair_key]
                num_matches = len(mkpts0)

            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_pair = (img0_name, img_name)
                best_idx0 = idx0

        if best_pair is None or best_num_matches < 5:
            print(f"Could not add {img_name} to cluster {cluster_idx}. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img_name] = (R, t)
            added_images.add(img_name)
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
            continue

        img0_name, img1_name = best_pair
        pair_key = (img0_name, img1_name) if (img0_name, img1_name) in matches_dict else (img1_name, img0_name)
        mkpts0, mkpts1 = matches_dict[pair_key]
        if len(mkpts0) < 5:
            print(f"Too few matches ({len(mkpts0)}) for pair {img0_name} and {img1_name}. Using sequential pose for {img1_name}.")
            if img1_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img1_name] = (R, t)
            added_images.add(img1_name)
            counted_images_processing.add(img1_name)
            summary['sequential_poses_due_to_processing'] += 1
            continue

        R0, t0 = poses[img0_name]
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            print(f"Failed to estimate pose for {img1_name}. Using sequential pose.")
            if img1_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            poses[img1_name] = (R, t)
            added_images.add(img1_name)
            counted_images_processing.add(img1_name)
            summary['sequential_poses_due_to_processing'] += 1
            continue

        t = normalize_translation(t, img1_name)
        t0 = normalize_translation(t0, img0_name)
        R = R0 @ R
        t = R0 @ t + t0
        t = normalize_translation(t, img1_name)
        if not validate_pose(R, t, img1_name):
            print(f"Invalid pose for {img1_name}. Using sequential pose.")
            if img1_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(added_images) * 0.1], dtype=np.float32)
            counted_images_processing.add(img1_name)
            summary['sequential_poses_due_to_processing'] += 1
        poses[img1_name] = (R, t)
        added_images.add(img1_name)

        if len(mkpts0) >= 5:
            R1, t1 = poses[img1_name]
            t0 = normalize_translation(t0, img0_name)
            t1 = normalize_translation(t1, img1_name)
            if validate_pose(R0, t0, img0_name) and validate_pose(R1, t1, img1_name):
                P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
                P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
                points4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
                points4d = points4d[:3] / (points4d[3] + 1e-8)
                points4d = points4d.T
                if np.all(np.isfinite(points4d)):
                    for i in range(len(points4d)):
                        points3d.append(points4d[i])
                        point_to_images.append([(img0_name, mkpts0[i]), (img1_name, mkpts1[i])])

    points3d = np.array(points3d, dtype=np.float32) if points3d else np.array([], dtype=np.float32).reshape(0, 3)
    print(f"Reconstructed {len(points3d)} 3D points in cluster {cluster_idx}")

    if len(poses) >= 3 and len(point_to_images) >= 10 and len(poses) < 20:
        start_time = time.time()
        img_names = list(poses.keys())
        translations = np.array([normalize_translation(poses[img_name][1], img_name) for img_name in img_names], dtype=np.float32)
        if translations.shape != (len(img_names), 3):
            print(f"Mismatch in translations shape: {translations.shape}. Expected ({len(img_names)}, 3). Skipping bundle adjustment.")
        else:
            initial_focal_length = K[0, 0]
            initial_cx, initial_cy = K[0, 2], K[1, 2]
            params = np.hstack((translations.ravel(), initial_focal_length, initial_cx, initial_cy))
            max_nfev = min(20, len(poses) * 5)
            result = least_squares(
                reprojection_error,
                params,
                args=(points3d, point_to_images, K, img_names, poses),
                max_nfev=max_nfev,
                ftol=1e-5,
                xtol=1e-5
            )
            optimized_params = result.x
            num_cameras = len(img_names)
            translations = optimized_params[:num_cameras * 3].reshape(num_cameras, 3)
            focal_length = optimized_params[num_cameras * 3]
            cx = optimized_params[num_cameras * 3 + 1]
            cy = optimized_params[num_cameras * 3 + 2]
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            for i, img_name in enumerate(img_names):
                R, _ = poses[img_name]
                t = translations[i]
                t = normalize_translation(t, img_name)
                if not validate_pose(R, t, img_name):
                    if img_name in counted_images_processing:
                        continue
                    R = np.eye(3, dtype=np.float32)
                    t = np.array([0.0, 0.0, len(poses) * 0.1], dtype=np.float32)
                    counted_images_processing.add(img_name)
                    summary['sequential_poses_due_to_processing'] += 1
                poses[img_name] = (R, t)
            print(f"Local bundle adjustment took {time.time() - start_time:.2f} seconds.")
    else:
        print("Skipping local bundle adjustment due to insufficient poses/observations or large cluster size.")

    print(f"Cluster {cluster_idx} processed {len(added_images)} cameras")
    return poses, points3d, K, point_to_images

# Phase 5: Reprojection Error
def compute_reprojection_error(img_name, points3d, observations, R, t, K):
    """Compute the average reprojection error for an image."""
    errors = []
    for j, obs in enumerate(observations):
        for img_obs, pt in obs:
            if img_obs == img_name:
                proj = project(points3d[j:j+1], R, t, K)
                if not np.all(np.isfinite(proj)):
                    continue
                error = np.linalg.norm(proj[0] - pt)
                if np.isfinite(error):
                    errors.append(error)
    return np.mean(errors) if errors else float('inf')

# Phase 6: Main Loop and Submission
all_poses = {}
all_Ks = {}
processed_images = set()
pose_status = {}

for (dataset, inferred_scene), group in grouped_images:
    print(f"\nProcessing dataset: {dataset}, inferred scene: {inferred_scene}")
    start_time = time.time()

    image_dir = f"/kaggle/input/image-matching-challenge-2025/test/{dataset}"
    if not os.path.exists(image_dir):
        print(f"Dataset directory {image_dir} does not exist in test set. Skipping.")
        summary['failed_groups'].append((dataset, inferred_scene, "Dataset not found in test directory"))
        for _, row in group.iterrows():
            img_name = row['image']
            if img_name not in counted_images_missing:
                summary['images_not_found'] += 1
                summary['sequential_poses_due_to_missing_data'] += 1
                counted_images_missing.add(img_name)
        continue

    image_paths = []
    for _, row in group.iterrows():
        img_name = row['image']
        img_path = f"{image_dir}/{img_name}"
        if Path(img_path).exists():
            image_paths.append(img_path)
            processed_images.add(img_name)
        else:
            print(f"Image not found: {img_path}")
            if img_name not in counted_images_missing:
                summary['images_not_found'] += 1
                summary['sequential_poses_due_to_missing_data'] += 1
                counted_images_missing.add(img_name)

    if not image_paths:
        print(f"No images found for dataset {dataset}, scene {inferred_scene}. Skipping.")
        summary['failed_groups'].append((dataset, inferred_scene, "No images found"))
        continue

    if len(image_paths) < 2:
        print(f"Dataset {dataset}, scene {inferred_scene} has fewer than 2 images. Using sequential poses.")
        for idx, img_path in enumerate(image_paths):
            img_name = Path(img_path).name
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, idx * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
        summary['failed_groups'].append((dataset, inferred_scene, "Fewer than 2 images"))
        continue

    clusters, noise_indices = cluster_images(image_paths, inferred_scene, min_matches_threshold=5)
    print(f"Clusters: {clusters}")
    print(f"Outlier images: {[Path(image_paths[idx]).name for idx in noise_indices if idx < len(image_paths)]}")

    overlap = 2

    all_points3d = []
    all_observations = []

    print(f"Processing clusters for dataset {dataset}, scene {inferred_scene}")
    for cluster_idx, cluster_indices in enumerate(clusters):
        result = run_sfm_on_cluster((cluster_indices, image_paths, inferred_scene, cluster_idx, clusters, overlap))
        poses, points3d, K, cluster_point_to_images = result
        for img_name in poses:
            R, t = poses[img_name]
            t = normalize_translation(t, img_name)
            if not validate_pose(R, t, img_name):
                print(f"Invalid pose for {img_name} after SfM. Using sequential pose.")
                if img_name in counted_images_processing:
                    continue
                R = np.eye(3, dtype=np.float32)
                t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
                counted_images_processing.add(img_name)
                summary['sequential_poses_due_to_processing'] += 1
            all_poses[img_name] = (R, t)
            all_Ks[img_name] = K
            if img_name not in pose_status:
                pose_status[img_name] = 'computed'
                summary['computed_poses'] += 1
        if len(points3d) > 0:
            all_points3d.append(points3d)
        all_observations.extend(cluster_point_to_images)

    for idx in noise_indices:
        if idx >= len(image_paths):
            continue
        img_name = Path(image_paths[idx]).name
        best_pair = None
        best_num_matches = 0
        for i in range(max(0, idx - 3), min(len(image_paths), idx + 4)):  # Reduced range to ±3
            if i == idx or Path(image_paths[i]).name not in all_poses:
                continue
            img0_name = Path(image_paths[i]).name
            pair_key = (img0_name, img_name) if (img0_name, img_name) in matches_dict else (img_name, img0_name)
            if pair_key not in matches_dict:
                _, _, _, _, _, matches = match_image_pair((i, idx, i, idx, image_paths))
                if matches is None:
                    matches_dict[(img0_name, img_name)] = (np.array([]), np.array([]))
                    num_matches = 0
                else:
                    mkpts0, mkpts1 = matches
                    matches_dict[(img0_name, img_name)] = (mkpts0, mkpts1)
                    num_matches = len(mkpts0)
            else:
                mkpts0, mkpts1 = matches_dict[pair_key]
                num_matches = len(mkpts0)
            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_pair = (img0_name, img_name)

        if best_pair is None or best_num_matches < 5:
            print(f"Could not register outlier image {img_name}. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
            continue

        img0_name, img1_name = best_pair
        pair_key = (img0_name, img1_name) if (img0_name, img1_name) in matches_dict else (img1_name, img0_name)
        mkpts0, mkpts1 = matches_dict[pair_key]
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        R, t, inliers = recover_pose_with_fallback(mkpts0, mkpts1, K, img0_name, img1_name)
        if R is None or t is None:
            print(f"Failed to register outlier image {img_name}. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1
            continue
        R0, t0 = all_poses[img0_name]
        t0 = normalize_translation(t0, img0_name)
        t = normalize_translation(t, img1_name)
        R = R0 @ R
        t = R0 @ t + t0
        t = normalize_translation(t, img1_name)
        if not validate_pose(R, t, img1_name):
            print(f"Invalid pose for outlier {img1_name}. Using sequential pose.")
            if img1_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            pose_status[img_name] = 'sequential'
            counted_images_processing.add(img1_name)
            summary['sequential_poses_due_to_processing'] += 1
        all_poses[img_name] = (R, t)
        all_Ks[img_name] = K
        if img_name not in pose_status:
            pose_status[img_name] = 'computed'
            summary['computed_poses'] += 1

    if all_points3d:
        merged_points3d = np.concatenate(all_points3d, axis=0)
    else:
        merged_points3d = np.array([], dtype=np.float32).reshape(0, 3)

    print("Skipping global bundle adjustment to save time.")

    reproj_threshold = 20.0  # Increased to 20.0
    for img_name in list(all_poses.keys()):
        R, t = all_poses[img_name]
        t = normalize_translation(t, img_name)
        if not validate_pose(R, t, img_name):
            print(f"Invalid pose for {img_name} before reprojection check. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            if pose_status.get(img_name) == 'computed':
                summary['computed_poses'] -= 1
                counted_images_processing.add(img_name)
                summary['sequential_poses_due_to_processing'] += 1
            pose_status[img_name] = 'sequential'
        if img_name not in all_Ks:
            print(f"Camera intrinsics (K) missing for {img_name}. Assigning default K.")
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
        K = all_Ks[img_name]
        error = compute_reprojection_error(img_name, merged_points3d, all_observations, R, t, K)
        if error > reproj_threshold and pose_status.get(img_name) == 'computed':
            print(f"Image {img_name} has high reprojection error ({error:.2f}). Resetting to sequential pose.")
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            pose_status[img_name] = 'sequential'
            summary['computed_poses'] -= 1
            if img_name not in counted_images_processing:
                counted_images_processing.add(img_name)
                summary['sequential_poses_due_to_processing'] += 1
            summary['high_reprojection_errors'] += 1
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K

    for img_path in image_paths:
        img_name = Path(img_path).name
        if img_name not in all_poses:
            print(f"Image {img_name} was not processed. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
            all_poses[img_name] = (R, t)
            image_width, image_height = 320, 240
            focal_length = max(image_width, image_height) * 1.2
            cx, cy = image_width / 2, image_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            all_Ks[img_name] = K
            pose_status[img_name] = 'sequential'
            counted_images_processing.add(img_name)
            summary['sequential_poses_due_to_processing'] += 1

    print(f"Dataset {dataset}, scene {inferred_scene} processed in {time.time() - start_time:.2f} seconds")
    summary['successful_groups'] += 1

# Handle unprocessed images
unprocessed_images = set(sample_submission['image']) - processed_images
if unprocessed_images:
    print(f"Warning: The following images were not processed: {unprocessed_images}")
    for img_name in unprocessed_images:
        if img_name in counted_images_missing:
            continue
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, len(all_poses) * 0.1], dtype=np.float32)
        all_poses[img_name] = (R, t)
        image_width, image_height = 320, 240
        focal_length = max(image_width, image_height) * 1.2
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        all_Ks[img_name] = K
        pose_status[img_name] = 'sequential'
        counted_images_missing.add(img_name)
        summary['sequential_poses_due_to_missing_data'] += 1

# Generate submission file
submission_rows = []
for _, row in sample_submission.iterrows():
    img_path = row['image_id']
    img_name = row['image']
    if img_name in all_poses:
        R, t = all_poses[img_name]
        t = normalize_translation(t, img_name)
        if not validate_pose(R, t, img_name):
            print(f"Invalid pose for {img_name} in submission. Using sequential pose.")
            if img_name in counted_images_processing:
                continue
            R = np.eye(3, dtype=np.float32)
            t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float32)
            if pose_status.get(img_name) == 'computed':
                summary['computed_poses'] -= 1
                counted_images_processing.add(img_name)
                summary['sequential_poses_due_to_processing'] += 1
            pose_status[img_name] = 'sequential'
        q = rotation_matrix_to_quaternion(R)
        t = t.reshape(3)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)):
            print(f"Non-finite pose for {img_name}. Using sequential pose.")
            q = rotation_matrix_to_quaternion(np.eye(3))
            t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float64)
            if pose_status.get(img_name) == 'computed':
                summary['computed_poses'] -= 1
                counted_images_processing.add(img_name)
                summary['sequential_poses_due_to_processing'] += 1
            pose_status[img_name] = 'sequential'
    else:
        print(f"Image {img_name} not found in poses. Using sequential pose.")
        if img_name in counted_images_missing:
            continue
        q = rotation_matrix_to_quaternion(np.eye(3))
        t = np.array([0.0, 0.0, len(submission_rows) * 0.1], dtype=np.float64)
        pose_status[img_name] = 'sequential'
        counted_images_missing.add(img_name)
        summary['sequential_poses_due_to_missing_data'] += 1
    submission_rows.append([img_path] + q.tolist() + t.tolist())

submission_columns = [
    'image_id',
    'rotation_w', 'rotation_x', 'rotation_y', 'rotation_z',
    'translation_x', 'translation_y', 'translation_z'
]
submission_df = pd.DataFrame(submission_rows, columns=submission_columns)
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Validate submission
print("Validating submission...")
assert len(submission_df) == len(sample_submission), f"Submission has {len(submission_df)} rows, expected {len(sample_submission)}"
assert set(submission_df.columns) == set(submission_columns), "Submission columns do not match expected columns"

for i, row in submission_df.iterrows():
    q = np.array([
        row['rotation_w'], row['rotation_x'], row['rotation_y'], row['rotation_z']
    ], dtype=np.float64)
    t = np.array([
        row['translation_x'], row['translation_y'], row['translation_z']
    ], dtype=np.float64)
    assert not np.any(np.isnan(q)), f"NaN values in quaternion at row {i}"
    assert not np.any(np.isnan(t)), f"NaN values in translation at row {i}"
    assert not np.any(np.isinf(q)), f"Infinite values in quaternion at row {i}"
    assert not np.any(np.isinf(t)), f"Infinite values in translation at row {i}"
    norm = np.linalg.norm(q)
    assert abs(norm - 1.0) < 1e-6, f"Quaternion not normalized at row {i}: norm={norm}"
    R = quaternion_to_rotation_matrix(q)
    det_R = np.linalg.det(R)
    assert abs(det_R - 1.0) < 1e-6, f"Rotation matrix determinant not 1 at row {i}: det={det_R}"
    orthogonality = np.linalg.norm(R.T @ R - np.eye(3))
    assert orthogonality < 1e-6, f"Rotation matrix not orthogonal at row {i}: orthogonality={orthogonality}"

print("Submission validated successfully.")

# Compute total sequential poses
summary['sequential_poses'] = summary['sequential_poses_due_to_missing_data'] + summary['sequential_poses_due_to_processing']

# Print summary
print("\n=== Processing Summary ===")
print(f"Total dataset/scene groups: {summary['total_groups']}")
print(f"Successful groups: {summary['successful_groups']}")
print(f"Failed groups: {summary['failed_groups']}")
print(f"Total images: {summary['total_images']}")
print(f"Computed poses: {summary['computed_poses']}")
print(f"Sequential poses: {summary['sequential_poses']}")
print(f"  - Due to missing data: {summary['sequential_poses_due_to_missing_data']}")
print(f"  - Due to processing: {summary['sequential_poses_due_to_processing']}")
print(f"Images not found: {summary['images_not_found']}")
print(f"High reprojection errors: {summary['high_reprojection_errors']}")




