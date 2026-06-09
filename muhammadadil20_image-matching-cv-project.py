# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#       print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")


!pip install --no-deps git+https://github.com/cvg/LightGlue.git


import cv2
import torch
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from kornia.feature import LightGlue


class Config:
    # Paths - MODIFY THESE ACCORDING TO YOUR KAGGLE SETUP
    DATA_PATH = "/kaggle/input/image-matching-challenge-2025"
    TRAIN_PATH = f"{DATA_PATH}/train"
    TEST_PATH = f"{DATA_PATH}/test"
    OUTPUT_PATH = "/kaggle/working/output"
    
    # Which dataset to use
    USE_TRAIN = True  # Set False to use test data
    PROCESS_ALL_SCENES = True  # Set False to process single scene
    SCENE_NAME = None  # Set to specific scene name or None to auto-detect
    
    # Pipeline parameters
    TOP_K_SIMILAR = 15  # Number of most similar images to match per image
    MIN_MATCHES = 15  # Minimum matches required for a valid pair
    MATCH_CONFIDENCE = 0.2  # LightGlue confidence threshold
    MAX_IMAGES = 50  # Maximum images to process per scene (set None for all)
    
    # Visualization
    DISPLAY_MATCHES = True
    NUM_MATCH_EXAMPLES = 2  # Per scene


config = Config()
# Create output directory
os.makedirs(config.OUTPUT_PATH, exist_ok=True)
print(f"Configuration loaded. Output directory: {config.OUTPUT_PATH}")
print(f"Process all scenes: {config.PROCESS_ALL_SCENES}")



class Config:
    # Paths - MODIFY THESE ACCORDING TO YOUR KAGGLE SETUP
    DATA_PATH = "/kaggle/input/image-matching-challenge-2025"
    TRAIN_PATH = f"{DATA_PATH}/train"
    TEST_PATH = f"{DATA_PATH}/test"
    OUTPUT_PATH = "/kaggle/working/output"
    
    # Which dataset to use
    USE_TRAIN = True  # Set False to use test data
    PROCESS_ALL_SCENES = True  # Set False to process single scene
    SCENE_NAME = None  # Set to specific scene name or None to auto-detect
    
    # Pipeline parameters
    TOP_K_SIMILAR = 15  # Number of most similar images to match per image
    MIN_MATCHES = 15  # Minimum matches required for a valid pair
    MATCH_CONFIDENCE = 0.2  # LightGlue confidence threshold
    MAX_IMAGES = 50  # Maximum images to process per scene (set None for all)
    
    # Visualization
    DISPLAY_MATCHES = True
    NUM_MATCH_EXAMPLES = 2  # Per scene
    
    # COLMAP settings
    USE_COLMAP = True  # Use actual COLMAP for reconstruction
    COLMAP_VOCAB_TREE = "/kaggle/input/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin"  # Optional
    
config = Config()

# Create output directory
os.makedirs(config.OUTPUT_PATH, exist_ok=True)
print(f"Configuration loaded. Output directory: {config.OUTPUT_PATH}")
print(f"Process all scenes: {config.PROCESS_ALL_SCENES}")



def load_images_from_scene(scene_path, max_images=None):
    """Load images from a scene directory"""
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene path does not exist: {scene_path}")
    
    # Find all image files
    for ext in valid_extensions:
        image_files.extend(list(scene_path.glob(f"*{ext}")))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {scene_path}")
    
    print(f"  Found {len(image_files)} images in scene")
    
    # Load images
    images = []
    image_names = []
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                image_names.append(img_path.name)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
    
    print(f"  Successfully loaded {len(images)} images")
    return images, image_names, image_files

# Get all scenes to process
base_path = config.TRAIN_PATH if config.USE_TRAIN else config.TEST_PATH

if config.PROCESS_ALL_SCENES:
    # Get all scenes
    all_scenes = sorted([d for d in Path(base_path).iterdir() if d.is_dir()])
    if len(all_scenes) == 0:
        raise ValueError(f"No scenes found in {base_path}")
    print(f"Found {len(all_scenes)} scenes to process:")
    for scene in all_scenes:
        print(f"  - {scene.name}")
else:
    # Single scene
    if config.SCENE_NAME is None:
        scenes = [d for d in Path(base_path).iterdir() if d.is_dir()]
        if len(scenes) == 0:
            raise ValueError(f"No scenes found in {base_path}")
        all_scenes = [scenes[0]]
        print(f"Auto-detected scene: {all_scenes[0].name}")
    else:
        all_scenes = [Path(base_path) / config.SCENE_NAME]

# Store all scene data
scenes_data = {}

print(f"\n{'='*70}")
print("LOADING ALL SCENES")
print(f"{'='*70}\n")

for scene_path in all_scenes:
    scene_name = scene_path.name
    print(f"Loading scene: {scene_name}")
    
    try:
        images, image_names, image_paths = load_images_from_scene(
            scene_path, 
            max_images=config.MAX_IMAGES
        )
        
        scenes_data[scene_name] = {
            'images': images,
            'image_names': image_names,
            'image_paths': image_paths,
            'scene_path': scene_path
        }
        
        print(f"  ✓ Loaded {len(images)} images from '{scene_name}'\n")
    
    except Exception as e:
        print(f"  ✗ Error loading scene '{scene_name}': {e}\n")
        continue

print(f"\n✓ Successfully loaded {len(scenes_data)} scenes")
print(f"  Total images: {sum(len(data['images']) for data in scenes_data.values())}")

# Display sample from first scene
if scenes_data:
    first_scene = list(scenes_data.keys())[0]
    sample_images = scenes_data[first_scene]['images']
    sample_names = scenes_data[first_scene]['image_names']
    
    fig, axes = plt.subplots(1, min(4, len(sample_images)), figsize=(15, 4))
    if len(sample_images) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i < len(sample_images):
            ax.imshow(sample_images[i])
            ax.set_title(f"{sample_names[i]}")
            ax.axis('off')
    plt.suptitle(f"Sample from scene: {first_scene}")
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/sample_images.png", dpi=150, bbox_inches='tight')
    plt.show()


print("\n" + "="*70)
print("PROCESSING ALL SCENES")
print("="*70 + "\n")

# Initialize models once (reuse across scenes)
print("Loading models...")

# Load DINOv2
try:
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = dinov2_model.to(device).eval()
    print("✓ DINOv2 loaded")
except Exception as e:
    print(f"✗ Error loading DINOv2: {e}")
    dinov2_model = None

aliked = ALIKED(
        max_num_keypoints=2048,
        detection_threshold=0.01,
        resize=1024
    ).to(device).eval()
    
print("✓ ALIKED loaded successfully")

lightglue = LightGlue(
        features='aliked',
        depth_confidence=0.95,
        width_confidence=0.95
    ).to(device).eval()
    
print("✓ LightGlue loaded successfully")

print()

def preprocess_for_dinov2(image, size=224):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = size - new_h, size - new_w
    padded = np.pad(resized, ((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2), (0, 0)), mode='constant')
    img_tensor = torch.from_numpy(padded).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor - mean) / std

def extract_features_opencv(image, max_features=2048):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        detector = cv2.SIFT_create(nfeatures=max_features)
    except:
        detector = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        return np.array([]), np.array([])
    return np.array([kp.pt for kp in keypoints]), descriptors

def match_features_opencv(desc1, desc2):
    if len(desc1) == 0 or len(desc2) == 0:
        return np.array([])
    bf = cv2.BFMatcher(cv2.NORM_L2 if desc1.dtype == np.float32 else cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches)

def match_image_pair(img1, img2):
    if aliked is not None and lightglue is not None:
        try:
            with torch.no_grad():
                img1_t = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                img2_t = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                feats1 = aliked(img1_t)
                feats2 = aliked(img2_t)
                matches_dict = lightglue({'image0': feats1, 'image1': feats2})
                matches = matches_dict['matches0'][0].cpu().numpy()
                valid = matches > -1
                kpts0 = feats1['keypoints'][0].cpu().numpy()
                kpts1 = feats2['keypoints'][0].cpu().numpy()
                return kpts0[valid], kpts1[matches[valid].astype(int)]
        except Exception as e:
            pass
    kpts1, desc1 = extract_features_opencv(img1)
    kpts2, desc2 = extract_features_opencv(img2)
    matches = match_features_opencv(desc1, desc2)
    if len(matches) == 0:
        return np.array([]), np.array([])
    return kpts1[matches[:, 0]], kpts2[matches[:, 1]]

# Process each scene
all_results = {}

for scene_idx, (scene_name, scene_data) in enumerate(scenes_data.items()):
    print(f"\n{'='*70}")
    print(f"SCENE {scene_idx+1}/{len(scenes_data)}: {scene_name}")
    print(f"{'='*70}\n")
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    
    # Extract embeddings
    if dinov2_model is not None:
        print("Extracting DINOv2 embeddings...")
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(images), 8):
                batch = torch.stack([preprocess_for_dinov2(img) for img in images[i:i+8]]).to(device)
                embeddings.append(dinov2_model(batch).cpu())
                print(f"  {min(i+8, len(images))}/{len(images)}", end='\r')
        embeddings = F.normalize(torch.cat(embeddings, dim=0), p=2, dim=1)
        similarity_matrix = torch.mm(embeddings, embeddings.t()).numpy()
        print(f"\n✓ Extracted embeddings")
    else:
        print("⚠ Skipping DINOv2, using sequential pairs")
        similarity_matrix = np.eye(len(images))
    
    # Create pairs
    image_pairs = []
    for i in range(len(images)):
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1
        top_k = np.argsort(similarities)[::-1][:config.TOP_K_SIMILAR]
        for j in top_k:
            if j > i:
                image_pairs.append((i, j))
    print(f"✓ Created {len(image_pairs)} pairs")
    
    # Match pairs
    print("Matching pairs...")
    all_matches = {}
    for idx, (i, j) in enumerate(image_pairs):
        try:
            mkpts0, mkpts1 = match_image_pair(images[i], images[j])
            if len(mkpts0) >= config.MIN_MATCHES:
                all_matches[(i, j)] = {'mkpts0': mkpts0, 'mkpts1': mkpts1, 'num_matches': len(mkpts0)}
            if (idx+1) % 20 == 0:
                print(f"  {idx+1}/{len(image_pairs)}", end='\r')
        except:
            continue
    print(f"\n✓ Matched {len(all_matches)} pairs")
    
    # Geometric verification
    print("Geometric verification...")
    verified_matches = {}
    for (i, j), match_data in all_matches.items():
        mkpts0, mkpts1 = match_data['mkpts0'], match_data['mkpts1']
        if len(mkpts0) < 8:
            continue
        try:
            E, mask = cv2.findEssentialMat(mkpts0, mkpts1, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and mask is not None:
                inliers = mask.ravel() == 1
                if np.sum(inliers) >= config.MIN_MATCHES:
                    verified_matches[(i, j)] = {
                        'mkpts0': mkpts0[inliers],
                        'mkpts1': mkpts1[inliers],
                        'num_inliers': np.sum(inliers)
                    }
        except:
            continue
    print(f"✓ Verified {len(verified_matches)} pairs")
    
    all_results[scene_name] = {
        'verified_matches': verified_matches,
        'similarity_matrix': similarity_matrix,
        'images': images,
        'image_names': image_names,
        'scene_path': scene_data['scene_path']
    }
    
    torch.cuda.empty_cache()

print(f"\n✓ Processed all {len(all_results)} scenes")



def preprocess_for_dinov2(image, size=224):
    """Preprocess image for DINOv2"""
    # Resize
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to square
    pad_h = size - new_h
    pad_w = size - new_w
    padded = np.pad(resized, ((pad_h//2, pad_h - pad_h//2), 
                               (pad_w//2, pad_w - pad_w//2), 
                               (0, 0)), mode='constant')
    
    # Normalize
    img_tensor = torch.from_numpy(padded).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor

print("Extracting DINOv2 embeddings...")

embeddings = []
batch_size = 8

with torch.no_grad():
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_tensors = torch.stack([
            preprocess_for_dinov2(img) for img in batch_images
        ]).to(device)
        
        batch_embeddings = dinov2_model(batch_tensors)
        embeddings.append(batch_embeddings.cpu())
        
        print(f"Processed {min(i+batch_size, len(images))}/{len(images)} images", end='\r')

embeddings = torch.cat(embeddings, dim=0)
embeddings = F.normalize(embeddings, p=2, dim=1)

print(f"\n✓ Extracted embeddings: {embeddings.shape}")

# Free up memory
del dinov2_model
torch.cuda.empty_cache()


print("Computing image similarity matrix...")

# Compute similarity matrix
similarity_matrix = torch.mm(embeddings, embeddings.t()).numpy()

# Create image pairs based on similarity
image_pairs = []
pair_similarities = []

for i in range(len(images)):
    # Get top-k most similar images (excluding self)
    similarities = similarity_matrix[i].copy()
    similarities[i] = -1  # Exclude self
    
    top_k_indices = np.argsort(similarities)[::-1][:config.TOP_K_SIMILAR]
    
    for j in top_k_indices:
        if j > i:  # Avoid duplicates
            image_pairs.append((i, j))
            pair_similarities.append(similarities[j])

print(f"✓ Created {len(image_pairs)} image pairs for matching")

# Visualize similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Cosine Similarity')
plt.title('Image Similarity Matrix (DINOv2)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.tight_layout()
plt.savefig(f"{config.OUTPUT_PATH}/similarity_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# Show top pairs
print("\nTop 5 most similar image pairs:")
sorted_pairs = sorted(zip(image_pairs, pair_similarities), key=lambda x: x[1], reverse=True)
for (i, j), sim in sorted_pairs[:5]:
    print(f"  {image_names[i]} <-> {image_names[j]} (similarity: {sim:.3f})")



def extract_features_opencv(image, max_features=2048):
    """Fallback feature extraction using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Try SIFT first, fallback to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=max_features)
    except:
        detector = cv2.ORB_create(nfeatures=max_features)
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None:
        return np.array([]), np.array([])
    
    kpts = np.array([kp.pt for kp in keypoints])
    return kpts, descriptors

def match_features_opencv(desc1, desc2):
    """Fallback matching using OpenCV"""
    if len(desc1) == 0 or len(desc2) == 0:
        return np.array([])
    
    # Use BFMatcher with cross-check
    if desc1.dtype == np.float32:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])
    
    return np.array(good_matches)

def match_image_pair(img1, img2, use_aliked=True):
    """Match a pair of images"""
    if use_aliked and aliked is not None and lightglue is not None:
        # Use ALIKED + LightGlue
        try:
            with torch.no_grad():
                # Prepare images
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                img1_t = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img2_t = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                img1_t = img1_t.to(device)
                img2_t = img2_t.to(device)
                
                # Extract features
                feats1 = aliked(img1_t)
                feats2 = aliked(img2_t)
                
                # Match with LightGlue
                matches_dict = lightglue({'image0': feats1, 'image1': feats2})
                
                # Extract matched keypoints - CORRECT indexing for Kornia ALIKED
                # Check the actual structure first
                if 'matches0' in matches_dict:
                    matches = matches_dict['matches0']
                    # Handle batch dimension properly
                    if matches.dim() == 2:  # Already unbatched or shape [N]
                        matches = matches.squeeze() if matches.dim() > 1 else matches
                    elif matches.dim() == 1:  # Already 1D
                        pass
                    else:  # Has batch dimension
                        matches = matches[0]
                    
                    matches = matches.cpu().numpy()
                    valid = matches > -1
                    
                    # Get keypoints - handle different possible structures
                    if 'keypoints' in feats1:
                        kpts0 = feats1['keypoints']
                        if kpts0.dim() == 3:  # [B, N, 2]
                            kpts0 = kpts0[0]
                        kpts0 = kpts0.cpu().numpy()
                    else:
                        raise KeyError("No keypoints in features")
                    
                    if 'keypoints' in feats2:
                        kpts1 = feats2['keypoints']
                        if kpts1.dim() == 3:  # [B, N, 2]
                            kpts1 = kpts1[0]
                        kpts1 = kpts1.cpu().numpy()
                    else:
                        raise KeyError("No keypoints in features")
                    
                    # Get matched points
                    mkpts0 = kpts0[valid]
                    mkpts1 = kpts1[matches[valid].astype(int)]
                    
                    return mkpts0, mkpts1
                else:
                    raise KeyError("No matches0 in output")
        
        except Exception as e:
            # Silently fall back to OpenCV - comment this line to debug
            # print(f"\nALIKED failed, using OpenCV fallback: {e}")
            pass
    
    # Use OpenCV fallback
    kpts1, desc1 = extract_features_opencv(img1)
    kpts2, desc2 = extract_features_opencv(img2)
    
    matches = match_features_opencv(desc1, desc2)
    
    if len(matches) == 0:
        return np.array([]), np.array([])
    
    mkpts0 = kpts1[matches[:, 0]]
    mkpts1 = kpts2[matches[:, 1]]
    
    return mkpts0, mkpts1

print("Matching image pairs...")

all_matches = {}
match_counts = []

for idx, (i, j) in enumerate(image_pairs):
    try:
        mkpts0, mkpts1 = match_image_pair(images[i], images[j])
        
        if len(mkpts0) >= config.MIN_MATCHES:
            all_matches[(i, j)] = {
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'num_matches': len(mkpts0)
            }
            match_counts.append(len(mkpts0))
        
        if (idx + 1) % 10 == 0:
            print(f"Matched {idx+1}/{len(image_pairs)} pairs", end='\r')
            
    except Exception as e:
        print(f"\nError matching pair ({i}, {j}): {e}")
        continue

print(f"\n✓ Successfully matched {len(all_matches)} pairs")
print(f"  Average matches per pair: {np.mean(match_counts):.1f}")
print(f"  Max matches: {np.max(match_counts) if match_counts else 0}")
print(f"  Min matches: {np.min(match_counts) if match_counts else 0}")



def draw_matches(img1, img2, mkpts0, mkpts1, num_display=50):
    """Draw matches between two images"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    h = max(h1, h2)
    w = w1 + w2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    
    # Randomly sample matches to display
    if len(mkpts0) > num_display:
        indices = np.random.choice(len(mkpts0), num_display, replace=False)
        mkpts0_display = mkpts0[indices]
        mkpts1_display = mkpts1[indices]
    else:
        mkpts0_display = mkpts0
        mkpts1_display = mkpts1
    
    # Draw matches
    for pt1, pt2 in zip(mkpts0_display, mkpts1_display):
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple((pt2 + np.array([w1, 0])).astype(int))
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)
    
    return canvas

if config.DISPLAY_MATCHES and len(all_matches) > 0:
    print("\nVisualizing matches...")
    
    # Get top matches by count
    sorted_matches = sorted(all_matches.items(), 
                           key=lambda x: x[1]['num_matches'], 
                           reverse=True)
    
    num_to_display = min(config.NUM_MATCH_EXAMPLES, len(sorted_matches))
    
    fig, axes = plt.subplots(num_to_display, 1, figsize=(20, 6*num_to_display))
    if num_to_display == 1:
        axes = [axes]
    
    for idx, ((i, j), match_data) in enumerate(sorted_matches[:num_to_display]):
        match_img = draw_matches(
            images[i], images[j],
            match_data['mkpts0'], match_data['mkpts1']
        )
        
        axes[idx].imshow(match_img)
        axes[idx].set_title(
            f"{image_names[i]} ↔ {image_names[j]} "
            f"({match_data['num_matches']} matches)",
            fontsize=14
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/feature_matches.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualized {num_to_display} match examples")



print("Performing geometric verification...")

verified_matches = {}

for (i, j), match_data in all_matches.items():
    mkpts0 = match_data['mkpts0']
    mkpts1 = match_data['mkpts1']
    
    if len(mkpts0) < 8:
        continue
    
    try:
        # Estimate essential matrix
        E, inlier_mask = cv2.findEssentialMat(
            mkpts0, mkpts1,
            focal=1.0,
            pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is not None and inlier_mask is not None:
            inliers = inlier_mask.ravel() == 1
            num_inliers = np.sum(inliers)
            
            if num_inliers >= config.MIN_MATCHES:
                verified_matches[(i, j)] = {
                    'mkpts0': mkpts0[inliers],
                    'mkpts1': mkpts1[inliers],
                    'num_inliers': num_inliers,
                    'E': E
                }
    
    except Exception as e:
        continue

print(f"✓ Verified {len(verified_matches)} pairs with geometric consistency")

if len(verified_matches) == 0:
    print("⚠ Warning: No geometrically verified matches found!")
    print("  The reconstruction may fail. Consider:")
    print("  - Reducing MIN_MATCHES threshold")
    print("  - Increasing TOP_K_SIMILAR")
    print("  - Using more/different images")



print("Preparing data for 3D reconstruction...")

# Create COLMAP workspace
colmap_workspace = Path(config.OUTPUT_PATH) / "colmap_workspace"
colmap_workspace.mkdir(exist_ok=True)

images_dir = colmap_workspace / "images"
images_dir.mkdir(exist_ok=True)

# Copy/save images to workspace
for idx, (img, name) in enumerate(zip(images, image_names)):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(images_dir / name), img_bgr)

# Save matches to text file for COLMAP
matches_file = colmap_workspace / "matches.txt"
with open(matches_file, 'w') as f:
    for (i, j), match_data in verified_matches.items():
        f.write(f"{image_names[i]} {image_names[j]}\n")
        mkpts0 = match_data['mkpts0']
        mkpts1 = match_data['mkpts1']
        for pt0, pt1 in zip(mkpts0, mkpts1):
            f.write(f"{pt0[0]:.2f} {pt0[1]:.2f} {pt1[0]:.2f} {pt1[1]:.2f}\n")

print(f"✓ Prepared {len(images)} images and {len(verified_matches)} match pairs")
print(f"  Workspace: {colmap_workspace}")



print("\n" + "="*70)
print("RUNNING COLMAP 3D RECONSTRUCTION")
print("="*70 + "\n")

# Check if COLMAP is available
try:
    import subprocess
    result = subprocess.run(['colmap', '--help'], capture_output=True, timeout=5)
    colmap_available = result.returncode == 0
    print("✓ COLMAP is available")
except:
    colmap_available = False
    print("⚠ COLMAP not available, will use simple triangulation")

def run_colmap_reconstruction(scene_name, scene_data, output_base):
    """Run COLMAP reconstruction for a scene"""
    
    # Create workspace
    workspace = Path(output_base) / scene_name
    workspace.mkdir(parents=True, exist_ok=True)
    
    images_dir = workspace / "images"
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    images_dir.mkdir(exist_ok=True)
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    verified_matches = scene_data['verified_matches']
    
    print(f"  Scene: {scene_name}")
    print(f"    Images: {len(images)}")
    print(f"    Verified pairs: {len(verified_matches)}")
    
    # Save images
    for img, name in zip(images, image_names):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / name), img_bgr)
    
    # Write matches in COLMAP format
    matches_import_dir = workspace / "matches_import"
    matches_import_dir.mkdir(exist_ok=True)
    
    # Write image pairs
    pairs_file = matches_import_dir / "image_pairs.txt"
    with open(pairs_file, 'w') as f:
        for (i, j) in verified_matches.keys():
            f.write(f"{image_names[i]} {image_names[j]}\n")
    
    # Write matches
    matches_file = matches_import_dir / "matches.txt"
    with open(matches_file, 'w') as f:
        for (i, j), match_data in verified_matches.items():
            f.write(f"{image_names[i]} {image_names[j]}\n")
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            for pt0, pt1 in zip(mkpts0, mkpts1):
                f.write(f"{pt0[0]} {pt0[1]} {pt1[0]} {pt1[1]}\n")
    
    if not colmap_available:
        print(f"    ⚠ COLMAP not available, skipping reconstruction")
        return None
    
    try:
        # Feature extraction
        print(f"    Running feature extraction...")
        subprocess.run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--ImageReader.single_camera', '1',
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--SiftExtraction.max_num_features', '8192'
        ], check=True, capture_output=True, timeout=300)
        
        # Feature matching
        print(f"    Running feature matching...")
        subprocess.run([
            'colmap', 'exhaustive_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.guided_matching', '1'
        ], check=True, capture_output=True, timeout=300)
        
        # Mapper (reconstruction)
        print(f"    Running mapper...")
        subprocess.run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.ba_refine_focal_length', '1',
            '--Mapper.ba_refine_extra_params', '1',
            '--Mapper.min_num_matches', str(config.MIN_MATCHES)
        ], check=True, capture_output=True, timeout=600)
        
        print(f"    ✓ COLMAP reconstruction complete")
        
        # Find the reconstruction (usually in sparse/0)
        recon_dir = sparse_dir / "0"
        if recon_dir.exists():
            return recon_dir
        else:
            print(f"    ⚠ No reconstruction found")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ COLMAP timeout")
        return None
    except subprocess.CalledProcessError as e:
        print(f"    ✗ COLMAP error: {e}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None

# Run COLMAP for each scene
colmap_results = {}

if config.USE_COLMAP and colmap_available:
    for scene_name, scene_data in all_results.items():
        recon_dir = run_colmap_reconstruction(
            scene_name, 
            scene_data, 
            config.OUTPUT_PATH + "/colmap_scenes"
        )
        colmap_results[scene_name] = recon_dir
        print()
else:
    print("Skipping COLMAP reconstruction (not available or disabled)")

print(f"\n✓ Completed reconstructions for {len(colmap_results)} scenes")



print("\n" + "="*70)
print("LOADING COLMAP RECONSTRUCTIONS")
print("="*70 + "\n")

def read_colmap_points3D(points3D_path):
    """Read COLMAP points3D.txt file"""
    points = []
    colors = []
    
    with open(points3D_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def read_colmap_binary_points3D(points3D_path):
    """Read COLMAP points3D.bin file"""
    try:
        import struct
        points = []
        colors = []
        
        with open(points3D_path, 'rb') as f:
            num_points = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack('Q', f.read(8))[0]
                xyz = struct.unpack('ddd', f.read(24))
                rgb = struct.unpack('BBB', f.read(3))
                error = struct.unpack('d', f.read(8))[0]
                track_length = struct.unpack('Q', f.read(8))[0]
                f.read(8 * track_length)  # Skip track elements
                
                points.append(xyz)
                colors.append(rgb)
        
        return np.array(points), np.array(colors)
    except:
        return np.array([]), np.array([])

# Load all COLMAP reconstructions
reconstructions = {}

for scene_name, recon_dir in colmap_results.items():
    if recon_dir is None or not Path(recon_dir).exists():
        continue
    
    print(f"Loading reconstruction: {scene_name}")
    
    # Try binary format first
    points3D_bin = Path(recon_dir) / "points3D.bin"
    points3D_txt = Path(recon_dir) / "points3D.txt"
    
    if points3D_bin.exists():
        points, colors = read_colmap_binary_points3D(points3D_bin)
    elif points3D_txt.exists():
        points, colors = read_colmap_points3D(points3D_txt)
    else:
        print(f"  ⚠ No points3D file found")
        continue
    
    if len(points) > 0:
        # Remove outliers
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        mask = np.all(np.abs(points - mean) < 3 * std, axis=1)
        points = points[mask]
        colors = colors[mask]
        
        reconstructions[scene_name] = {
            'points': points,
            'colors': colors,
            'recon_dir': recon_dir
        }
        print(f"  ✓ Loaded {len(points)} 3D points")
    else:
        print(f"  ⚠ No points loaded")

print(f"\n✓ Loaded {len(reconstructions)} reconstructions")



print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70 + "\n")

import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# Create visualizations for each reconstruction
for scene_name, recon_data in reconstructions.items():
    print(f"Visualizing: {scene_name}")
    
    points = recon_data['points']
    colors = recon_data['colors']
    
    if len(points) == 0:
        continue
    
    # Create output directory for this scene
    scene_output = Path(config.OUTPUT_PATH) / "visualizations" / scene_name
    scene_output.mkdir(parents=True, exist_ok=True)
    
    # Plotly 3D interactive visualization
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=colors if len(colors) > 0 else 'blue',
                opacity=0.8
            ),
            name='3D Points'
        )
    ])
    
    fig.update_layout(
        title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    html_path = scene_output / "reconstruction_3d.html"
    fig.write_html(str(html_path))
    print(f"  Saved interactive: {html_path}")
    
    # Matplotlib 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for faster rendering
    subsample = min(10000, len(points))
    indices = np.random.choice(len(points), subsample, replace=False)
    
    ax.scatter(
        points[indices, 0],
        points[indices, 1],
        points[indices, 2],
        c=colors[indices] / 255.0 if len(colors) > 0 else 'blue',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction - {scene_name}\n{len(points)} points')
    
    png_path = scene_output / "reconstruction_3d.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved image: {png_path}")
    
    # Show first reconstruction
    if scene_name == list(reconstructions.keys())[0]:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color=colors if len(colors) > 0 else 'blue', opacity=0.8),
                name='3D Points'
            )
        ])
        fig.update_layout(
            title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            width=1000, height=800
        )
        fig.show()
    
    print()

print(f"✓ Created visualizations for {len(reconstructions)} scenes")



print("\n" + "="*70)
print("FINAL PIPELINE SUMMARY")
print("="*70 + "\n")

total_images = sum(len(data['images']) for data in scenes_data.values())
total_matches = sum(len(data['verified_matches']) for data in all_results.values())
total_points = sum(len(data['points']) for data in reconstructions.values())

print(f"{'Metric':<40} {'Value':>20}")
print("-" * 62)
print(f"{'Total scenes processed':<40} {len(scenes_data):>20}")
print(f"{'Total images loaded':<40} {total_images:>20}")
print(f"{'Total verified match pairs':<40} {total_matches:>20}")
print(f"{'Successful reconstructions':<40} {len(reconstructions):>20}")
print(f"{'Total 3D points reconstructed':<40} {total_points:>20,}")
print()

print("Per-Scene Statistics:")
print("-" * 62)
print(f"{'Scene Name':<30} {'Images':>10} {'Matches':>10} {'3D Points':>10}")
print("-" * 62)

for scene_name in scenes_data.keys():
    num_images = len(scenes_data[scene_name]['images'])
    num_matches = len(all_results[scene_name]['verified_matches'])
    num_points = len(reconstructions[scene_name]['points']) if scene_name in reconstructions else 0
    print(f"{scene_name:<30} {num_images:>10} {num_matches:>10} {num_points:>10,}")

print()
print("Output Locations:")
print(f"  Main output: {config.OUTPUT_PATH}")
print(f"  COLMAP workspaces: {config.OUTPUT_PATH}/colmap_scenes/")
print(f"  Visualizations: {config.OUTPUT_PATH}/visualizations/")
print()

# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Images per scene
scene_names = list(scenes_data.keys())
image_counts = [len(scenes_data[s]['images']) for s in scene_names]
axes[0, 0].barh(range(len(scene_names)), image_counts, color='steelblue')
axes[0, 0].set_yticks(range(len(scene_names)))
axes[0, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 0].set_xlabel('Number of Images')
axes[0, 0].set_title('Images per Scene')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Verified matches per scene
match_counts = [len(all_results[s]['verified_matches']) for s in scene_names]
axes[0, 1].barh(range(len(scene_names)), match_counts, color='forestgreen')
axes[0, 1].set_yticks(range(len(scene_names)))
axes[0, 1].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 1].set_xlabel('Number of Verified Matches')
axes[0, 1].set_title('Verified Matches per Scene')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: 3D points per scene
point_counts = [len(reconstructions[s]['points']) if s in reconstructions else 0 for s in scene_names]
axes[1, 0].barh(range(len(scene_names)), point_counts, color='coral')
axes[1, 0].set_yticks(range(len(scene_names)))
axes[1, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[1, 0].set_xlabel('Number of 3D Points')
axes[1, 0].set_title('3D Points per Scene')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Pipeline overview
pipeline_stages = ['Images\nLoaded', 'Pairs\nCreated', 'Pairs\nMatched', 'Pairs\nVerified', '3D Points\nReconstructed']
pipeline_counts = [
    total_images,
    sum(len(all_results[s]['verified_matches']) * 2 for s in scene_names),  # Approximate pairs created
    total_matches,
    total_matches,
    total_points
]
axes[1, 1].plot(pipeline_stages, pipeline_counts, marker='o', linewidth=2, markersize=10, color='darkviolet')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Pipeline Flow')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_PATH}/pipeline_summary.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Pipeline complete!")
print(f"\n{'='*70}")


print("\n" + "="*70)
print("RUNNING 3D RECONSTRUCTION")
print("="*70 + "\n")

# Check if pycolmap is available
try:
    import pycolmap
    print("✓ pycolmap is available")
    use_pycolmap = True
except ImportError:
    print("⚠ pycolmap not available, will use simple triangulation")
    use_pycolmap = False


def reconstruct_with_pycolmap(scene_name, scene_data, output_base):
    """Reconstruct using pycolmap (Python-based)"""
    import pycolmap
    
    workspace = Path(output_base) / scene_name
    workspace.mkdir(parents=True, exist_ok=True)
    
    images_dir = workspace / "images"
    images_dir.mkdir(exist_ok=True)
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    verified_matches = scene_data['verified_matches']
    
    print(f"  Scene: {scene_name}")
    print(f"    Images: {len(images)}")
    print(f"    Verified pairs: {len(verified_matches)}")
    
    # Save images to disk
    for img, name in zip(images, image_names):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / name), img_bgr)
    
    # Create pycolmap reconstruction
    output_path = workspace / "sparse"
    output_path.mkdir(exist_ok=True)
    
    try:
        # Setup reconstruction
        reconstruction = pycolmap.Reconstruction()
        
        # Add camera (assume all images use same camera with approximate parameters)
        camera = pycolmap.Camera(
            model='SIMPLE_RADIAL',
            width=images[0].shape[1],
            height=images[0].shape[0],
            params=[max(images[0].shape[:2]), images[0].shape[1]/2, images[0].shape[0]/2, 0]
        )
        camera_id = reconstruction.add_camera(camera)
        
        # Add images
        image_id_map = {}
        for idx, name in enumerate(image_names):
            image = pycolmap.Image(
                id=idx + 1,
                name=name,
                camera_id=camera_id
            )
            image_id = reconstruction.add_image(image)
            image_id_map[idx] = image_id
        
        # Add keypoints and matches
        for (i, j), match_data in verified_matches.items():
            img_id1 = image_id_map[i]
            img_id2 = image_id_map[j]
            
            # Add keypoints to images
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            
            # This is simplified - in production you'd need proper feature management
            # For now, we'll use a simpler triangulation approach
        
        print(f"    ⚠ pycolmap setup complete but needs full implementation")
        print(f"    Falling back to simple triangulation...")
        return None
        
    except Exception as e:
        print(f"    ✗ pycolmap error: {e}")
        return None

def reconstruct_simple_triangulation(scene_name, scene_data, output_base):
    """Simple reconstruction using triangulation (fallback)"""
    
    workspace = Path(output_base) / scene_name
    workspace.mkdir(parents=True, exist_ok=True)
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    verified_matches = scene_data['verified_matches']
    
    print(f"  Scene: {scene_name}")
    print(f"    Images: {len(images)}")
    print(f"    Verified pairs: {len(verified_matches)}")
    
    if len(verified_matches) == 0:
        print(f"    ⚠ No matches to reconstruct")
        return None
    
    # Simple triangulation approach
    all_points_3d = []
    all_colors = []
    
    # Process multiple pairs
    for pair_idx, ((i, j), match_data) in enumerate(list(verified_matches.items())[:10]):
        try:
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            
            if len(mkpts0) < 8:
                continue
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                mkpts0, mkpts1,
                focal=max(images[i].shape[:2]),
                pp=(images[i].shape[1]/2, images[i].shape[0]/2),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None:
                continue
            
            # Recover pose
            _, R, t, mask_pose = cv2.recoverPose(E, mkpts0, mkpts1)
            
            # Create projection matrices
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t])
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(P1, P2, mkpts0.T, mkpts1.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Get colors from first image
            colors = []
            for pt in mkpts0.astype(int):
                y, x = np.clip(pt[1], 0, images[i].shape[0]-1), np.clip(pt[0], 0, images[i].shape[1]-1)
                colors.append(images[i][y, x])
            
            all_points_3d.append(points_3d)
            all_colors.append(np.array(colors))
            
        except Exception as e:
            continue
    
    if len(all_points_3d) == 0:
        print(f"    ⚠ No 3D points reconstructed")
        return None
    
    # Combine all points
    points_3d = np.vstack(all_points_3d)
    colors = np.vstack(all_colors)
    
    # Remove outliers (points too far from median)
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)
    threshold = np.percentile(distances, 95)  # Keep 95% of points
    mask = distances < threshold
    
    points_3d = points_3d[mask]
    colors = colors[mask]
    
    print(f"    ✓ Reconstructed {len(points_3d)} 3D points")
    
    # Save as simple format
    output_file = workspace / "points3D.txt"
    with open(output_file, 'w') as f:
        f.write("# 3D point list with RGB colors\n")
        f.write("# Format: X Y Z R G B\n")
        for pt, col in zip(points_3d, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    
    return {
        'points': points_3d,
        'colors': colors,
        'output_file': output_file,
        'workspace': workspace
    }

# Run reconstruction for each scene
reconstructions = {}

for scene_name, scene_data in all_results.items():
    print(f"\n{'─'*60}")
    
    # Try pycolmap first, fallback to simple triangulation
    if use_pycolmap and config.USE_COLMAP:
        recon = reconstruct_with_pycolmap(
            scene_name, 
            scene_data, 
            config.OUTPUT_PATH + "/reconstructions"
        )
    else:
        recon = None
    
    # Fallback to simple triangulation
    if recon is None:
        recon = reconstruct_simple_triangulation(
            scene_name,
            scene_data,
            config.OUTPUT_PATH + "/reconstructions"
        )
    
    if recon is not None:
        reconstructions[scene_name] = recon

print(f"\n{'='*70}")
print(f"✓ Completed {len(reconstructions)}/{len(all_results)} reconstructions")
print(f"{'='*70}\n")


print("\n" + "="*70)
print("LOADING RECONSTRUCTIONS")
print("="*70 + "\n")

# Reconstructions are already loaded from Cell 15!
# They're in the 'reconstructions' dictionary

if len(reconstructions) > 0:
    print(f"✓ Loaded {len(reconstructions)} reconstructions\n")
    
    for scene_name, recon_data in reconstructions.items():
        num_points = len(recon_data['points'])
        print(f"  {scene_name}: {num_points:,} 3D points")
else:
    print("⚠ No reconstructions available")
    print("  This might be because:")
    print("  - No verified matches were found")
    print("  - Triangulation failed")
    print("  - Try adjusting MIN_MATCHES or TOP_K_SIMILAR in Cell 2")


print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70 + "\n")

import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# Create visualizations for each reconstruction
for scene_name, recon_data in reconstructions.items():
    print(f"Visualizing: {scene_name}")
    
    points = recon_data['points']
    colors = recon_data['colors']
    
    if len(points) == 0:
        continue
    
    # Create output directory for this scene
    scene_output = Path(config.OUTPUT_PATH) / "visualizations" / scene_name
    scene_output.mkdir(parents=True, exist_ok=True)
    
    # Plotly 3D interactive visualization
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=colors if len(colors) > 0 else 'blue',
                opacity=0.8
            ),
            name='3D Points'
        )
    ])
    
    fig.update_layout(
        title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    html_path = scene_output / "reconstruction_3d.html"
    fig.write_html(str(html_path))
    print(f"  Saved interactive: {html_path}")
    
    # Matplotlib 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for faster rendering
    subsample = min(10000, len(points))
    indices = np.random.choice(len(points), subsample, replace=False)
    
    ax.scatter(
        points[indices, 0],
        points[indices, 1],
        points[indices, 2],
        c=colors[indices] / 255.0 if len(colors) > 0 else 'blue',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction - {scene_name}\n{len(points)} points')
    
    png_path = scene_output / "reconstruction_3d.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved image: {png_path}")
    
    # Show first reconstruction
    if scene_name == list(reconstructions.keys())[0]:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color=colors if len(colors) > 0 else 'blue', opacity=0.8),
                name='3D Points'
            )
        ])
        fig.update_layout(
            title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            width=1000, height=800
        )
        fig.show()
    
    print()

print(f"✓ Created visualizations for {len(reconstructions)} scenes")



print("\n" + "="*70)
print("FINAL PIPELINE SUMMARY")
print("="*70 + "\n")

total_images = sum(len(data['images']) for data in scenes_data.values())
total_matches = sum(len(data['verified_matches']) for data in all_results.values())
total_points = sum(len(data['points']) for data in reconstructions.values())

print(f"{'Metric':<40} {'Value':>20}")
print("-" * 62)
print(f"{'Total scenes processed':<40} {len(scenes_data):>20}")
print(f"{'Total images loaded':<40} {total_images:>20}")
print(f"{'Total verified match pairs':<40} {total_matches:>20}")
print(f"{'Successful reconstructions':<40} {len(reconstructions):>20}")
print(f"{'Total 3D points reconstructed':<40} {total_points:>20,}")
print()

print("Per-Scene Statistics:")
print("-" * 62)
print(f"{'Scene Name':<30} {'Images':>10} {'Matches':>10} {'3D Points':>10}")
print("-" * 62)

for scene_name in scenes_data.keys():
    num_images = len(scenes_data[scene_name]['images'])
    num_matches = len(all_results[scene_name]['verified_matches'])
    num_points = len(reconstructions[scene_name]['points']) if scene_name in reconstructions else 0
    print(f"{scene_name:<30} {num_images:>10} {num_matches:>10} {num_points:>10,}")

print()
print("Output Locations:")
print(f"  Main output: {config.OUTPUT_PATH}")
print(f"  COLMAP workspaces: {config.OUTPUT_PATH}/colmap_scenes/")
print(f"  Visualizations: {config.OUTPUT_PATH}/visualizations/")
print()

# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Images per scene
scene_names = list(scenes_data.keys())
image_counts = [len(scenes_data[s]['images']) for s in scene_names]
axes[0, 0].barh(range(len(scene_names)), image_counts, color='steelblue')
axes[0, 0].set_yticks(range(len(scene_names)))
axes[0, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 0].set_xlabel('Number of Images')
axes[0, 0].set_title('Images per Scene')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Verified matches per scene
match_counts = [len(all_results[s]['verified_matches']) for s in scene_names]
axes[0, 1].barh(range(len(scene_names)), match_counts, color='forestgreen')
axes[0, 1].set_yticks(range(len(scene_names)))
axes[0, 1].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 1].set_xlabel('Number of Verified Matches')
axes[0, 1].set_title('Verified Matches per Scene')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: 3D points per scene
point_counts = [len(reconstructions[s]['points']) if s in reconstructions else 0 for s in scene_names]
axes[1, 0].barh(range(len(scene_names)), point_counts, color='coral')
axes[1, 0].set_yticks(range(len(scene_names)))
axes[1, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[1, 0].set_xlabel('Number of 3D Points')
axes[1, 0].set_title('3D Points per Scene')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Pipeline overview
pipeline_stages = ['Images\nLoaded', 'Pairs\nCreated', 'Pairs\nMatched', 'Pairs\nVerified', '3D Points\nReconstructed']
pipeline_counts = [
    total_images,
    sum(len(all_results[s]['verified_matches']) * 2 for s in scene_names),  # Approximate pairs created
    total_matches,
    total_matches,
    total_points
]
axes[1, 1].plot(pipeline_stages, pipeline_counts, marker='o', linewidth=2, markersize=10, color='darkviolet')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Pipeline Flow')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_PATH}/pipeline_summary.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Pipeline complete!")
print(f"\n{'='*70}")


! pip install pycolmap


print("Preparing data for 3D reconstruction...")

# Create COLMAP workspace
colmap_workspace = Path(config.OUTPUT_PATH) / "colmap_workspace"
colmap_workspace.mkdir(exist_ok=True)

images_dir = colmap_workspace / "images"
images_dir.mkdir(exist_ok=True)

# Copy/save images to workspace
for idx, (img, name) in enumerate(zip(images, image_names)):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(images_dir / name), img_bgr)

# Save matches to text file for COLMAP
matches_file = colmap_workspace / "matches.txt"
with open(matches_file, 'w') as f:
    for (i, j), match_data in verified_matches.items():
        f.write(f"{image_names[i]} {image_names[j]}\n")
        mkpts0 = match_data['mkpts0']
        mkpts1 = match_data['mkpts1']
        for pt0, pt1 in zip(mkpts0, mkpts1):
            f.write(f"{pt0[0]:.2f} {pt0[1]:.2f} {pt1[0]:.2f} {pt1[1]:.2f}\n")

print(f"✓ Prepared {len(images)} images and {len(verified_matches)} match pairs")
print(f"  Workspace: {colmap_workspace}")


print("\n" + "="*70)
print("RUNNING 3D RECONSTRUCTION")
print("="*70 + "\n")

# Check if pycolmap is available
try:
    import pycolmap
    print("✓ pycolmap is available")
    use_pycolmap = True
except ImportError:
    print("⚠ pycolmap not available, will use simple triangulation")
    use_pycolmap = False


def reconstruct_with_pycolmap(scene_name, scene_data, output_base):
    """Reconstruct using pycolmap (Python-based)"""
    import pycolmap
    
    workspace = Path(output_base) / scene_name
    workspace.mkdir(parents=True, exist_ok=True)
    
    images_dir = workspace / "images"
    images_dir.mkdir(exist_ok=True)
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    verified_matches = scene_data['verified_matches']
    
    print(f"  Scene: {scene_name}")
    print(f"    Images: {len(images)}")
    print(f"    Verified pairs: {len(verified_matches)}")
    
    # Save images to disk
    for img, name in zip(images, image_names):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / name), img_bgr)
    
    # Create pycolmap reconstruction
    output_path = workspace / "sparse"
    output_path.mkdir(exist_ok=True)
    
    try:
        # Setup reconstruction
        reconstruction = pycolmap.Reconstruction()
        
        # Add camera (assume all images use same camera with approximate parameters)
        camera = pycolmap.Camera(
            model='SIMPLE_RADIAL',
            width=images[0].shape[1],
            height=images[0].shape[0],
            params=[max(images[0].shape[:2]), images[0].shape[1]/2, images[0].shape[0]/2, 0]
        )
        camera_id = reconstruction.add_camera(camera)
        
        # Add images
        image_id_map = {}
        for idx, name in enumerate(image_names):
            image = pycolmap.Image(
                id=idx + 1,
                name=name,
                camera_id=camera_id
            )
            image_id = reconstruction.add_image(image)
            image_id_map[idx] = image_id
        
        # Add keypoints and matches
        for (i, j), match_data in verified_matches.items():
            img_id1 = image_id_map[i]
            img_id2 = image_id_map[j]
            
            # Add keypoints to images
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            
            # This is simplified - in production you'd need proper feature management
            # For now, we'll use a simpler triangulation approach
        
        print(f"    ⚠ pycolmap setup complete but needs full implementation")
        print(f"    Falling back to simple triangulation...")
        return None
        
    except Exception as e:
        print(f"    ✗ pycolmap error: {e}")
        return None

def reconstruct_simple_triangulation(scene_name, scene_data, output_base):
    """Simple reconstruction using triangulation (fallback)"""
    
    workspace = Path(output_base) / scene_name
    workspace.mkdir(parents=True, exist_ok=True)
    
    images = scene_data['images']
    image_names = scene_data['image_names']
    verified_matches = scene_data['verified_matches']
    
    print(f"  Scene: {scene_name}")
    print(f"    Images: {len(images)}")
    print(f"    Verified pairs: {len(verified_matches)}")
    
    if len(verified_matches) == 0:
        print(f"    ⚠ No matches to reconstruct")
        return None
    
    # Simple triangulation approach
    all_points_3d = []
    all_colors = []
    
    # Process multiple pairs
    for pair_idx, ((i, j), match_data) in enumerate(list(verified_matches.items())[:10]):
        try:
            mkpts0 = match_data['mkpts0']
            mkpts1 = match_data['mkpts1']
            
            if len(mkpts0) < 8:
                continue
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                mkpts0, mkpts1,
                focal=max(images[i].shape[:2]),
                pp=(images[i].shape[1]/2, images[i].shape[0]/2),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None:
                continue
            
            # Recover pose
            _, R, t, mask_pose = cv2.recoverPose(E, mkpts0, mkpts1)
            
            # Create projection matrices
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t])
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(P1, P2, mkpts0.T, mkpts1.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Get colors from first image
            colors = []
            for pt in mkpts0.astype(int):
                y, x = np.clip(pt[1], 0, images[i].shape[0]-1), np.clip(pt[0], 0, images[i].shape[1]-1)
                colors.append(images[i][y, x])
            
            all_points_3d.append(points_3d)
            all_colors.append(np.array(colors))
            
        except Exception as e:
            continue
    
    if len(all_points_3d) == 0:
        print(f"    ⚠ No 3D points reconstructed")
        return None
    
    # Combine all points
    points_3d = np.vstack(all_points_3d)
    colors = np.vstack(all_colors)
    
    # Remove outliers (points too far from median)
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)
    threshold = np.percentile(distances, 95)  # Keep 95% of points
    mask = distances < threshold
    
    points_3d = points_3d[mask]
    colors = colors[mask]
    
    print(f"    ✓ Reconstructed {len(points_3d)} 3D points")
    
    # Save as simple format
    output_file = workspace / "points3D.txt"
    with open(output_file, 'w') as f:
        f.write("# 3D point list with RGB colors\n")
        f.write("# Format: X Y Z R G B\n")
        for pt, col in zip(points_3d, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    
    return {
        'points': points_3d,
        'colors': colors,
        'output_file': output_file,
        'workspace': workspace
    }

# Run reconstruction for each scene
reconstructions = {}

for scene_name, scene_data in all_results.items():
    print(f"\n{'─'*60}")
    
    # Try pycolmap first, fallback to simple triangulation
    if use_pycolmap and config.USE_COLMAP:
        recon = reconstruct_with_pycolmap(
            scene_name, 
            scene_data, 
            config.OUTPUT_PATH + "/reconstructions"
        )
    else:
        recon = None
    
    # Fallback to simple triangulation
    if recon is None:
        recon = reconstruct_simple_triangulation(
            scene_name,
            scene_data,
            config.OUTPUT_PATH + "/reconstructions"
        )
    
    if recon is not None:
        reconstructions[scene_name] = recon

print(f"\n{'='*70}")
print(f"✓ Completed {len(reconstructions)}/{len(all_results)} reconstructions")
print(f"{'='*70}\n")



print("\n" + "="*70)
print("LOADING RECONSTRUCTIONS")
print("="*70 + "\n")

# Reconstructions are already loaded from Cell 15!
# They're in the 'reconstructions' dictionary

if len(reconstructions) > 0:
    print(f"✓ Loaded {len(reconstructions)} reconstructions\n")
    
    for scene_name, recon_data in reconstructions.items():
        num_points = len(recon_data['points'])
        print(f"  {scene_name}: {num_points:,} 3D points")
else:
    print("⚠ No reconstructions available")
    print("  This might be because:")
    print("  - No verified matches were found")
    print("  - Triangulation failed")
    print("  - Try adjusting MIN_MATCHES or TOP_K_SIMILAR in Cell 2")



print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70 + "\n")

import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# Create visualizations for each reconstruction
for scene_name, recon_data in reconstructions.items():
    print(f"Visualizing: {scene_name}")
    
    points = recon_data['points']
    colors = recon_data['colors']
    
    if len(points) == 0:
        continue
    
    # Create output directory for this scene
    scene_output = Path(config.OUTPUT_PATH) / "pycolmap_visualizations" / scene_name
    scene_output.mkdir(parents=True, exist_ok=True)
    
    # Plotly 3D interactive visualization
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=colors if len(colors) > 0 else 'blue',
                opacity=0.8
            ),
            name='3D Points'
        )
    ])
    
    fig.update_layout(
        title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    html_path = scene_output / "reconstruction_3d.html"
    fig.write_html(str(html_path))
    print(f"  Saved interactive: {html_path}")
    
    # Matplotlib 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for faster rendering
    subsample = min(10000, len(points))
    indices = np.random.choice(len(points), subsample, replace=False)
    
    ax.scatter(
        points[indices, 0],
        points[indices, 1],
        points[indices, 2],
        c=colors[indices] / 255.0 if len(colors) > 0 else 'blue',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction - {scene_name}\n{len(points)} points')
    
    png_path = scene_output / "reconstruction_3d.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved image: {png_path}")
    
    # Show first reconstruction
    if scene_name == list(reconstructions.keys())[0]:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color=colors if len(colors) > 0 else 'blue', opacity=0.8),
                name='3D Points'
            )
        ])
        fig.update_layout(
            title=f'3D Reconstruction - {scene_name}<br>{len(points)} points',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            width=1000, height=800
        )
        fig.show()
    
    print()

print(f"✓ Created visualizations for {len(reconstructions)} scenes")



print("\n" + "="*70)
print("FINAL PIPELINE SUMMARY")
print("="*70 + "\n")

total_images = sum(len(data['images']) for data in scenes_data.values())
total_matches = sum(len(data['verified_matches']) for data in all_results.values())
total_points = sum(len(data['points']) for data in reconstructions.values())

print(f"{'Metric':<40} {'Value':>20}")
print("-" * 62)
print(f"{'Total scenes processed':<40} {len(scenes_data):>20}")
print(f"{'Total images loaded':<40} {total_images:>20}")
print(f"{'Total verified match pairs':<40} {total_matches:>20}")
print(f"{'Successful reconstructions':<40} {len(reconstructions):>20}")
print(f"{'Total 3D points reconstructed':<40} {total_points:>20,}")
print()

print("Per-Scene Statistics:")
print("-" * 62)
print(f"{'Scene Name':<30} {'Images':>10} {'Matches':>10} {'3D Points':>10}")
print("-" * 62)

for scene_name in scenes_data.keys():
    num_images = len(scenes_data[scene_name]['images'])
    num_matches = len(all_results[scene_name]['verified_matches'])
    num_points = len(reconstructions[scene_name]['points']) if scene_name in reconstructions else 0
    print(f"{scene_name:<30} {num_images:>10} {num_matches:>10} {num_points:>10,}")

print()
print("Output Locations:")
print(f"  Main output: {config.OUTPUT_PATH}")
print(f"  COLMAP workspaces: {config.OUTPUT_PATH}/colmap_scenes/")
print(f"  Visualizations: {config.OUTPUT_PATH}/visualizations/")
print()

# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Images per scene
scene_names = list(scenes_data.keys())
image_counts = [len(scenes_data[s]['images']) for s in scene_names]
axes[0, 0].barh(range(len(scene_names)), image_counts, color='steelblue')
axes[0, 0].set_yticks(range(len(scene_names)))
axes[0, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 0].set_xlabel('Number of Images')
axes[0, 0].set_title('Images per Scene')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Verified matches per scene
match_counts = [len(all_results[s]['verified_matches']) for s in scene_names]
axes[0, 1].barh(range(len(scene_names)), match_counts, color='forestgreen')
axes[0, 1].set_yticks(range(len(scene_names)))
axes[0, 1].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[0, 1].set_xlabel('Number of Verified Matches')
axes[0, 1].set_title('Verified Matches per Scene')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: 3D points per scene
point_counts = [len(reconstructions[s]['points']) if s in reconstructions else 0 for s in scene_names]
axes[1, 0].barh(range(len(scene_names)), point_counts, color='coral')
axes[1, 0].set_yticks(range(len(scene_names)))
axes[1, 0].set_yticklabels([s[:20] for s in scene_names], fontsize=8)
axes[1, 0].set_xlabel('Number of 3D Points')
axes[1, 0].set_title('3D Points per Scene')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Pipeline overview
pipeline_stages = ['Images\nLoaded', 'Pairs\nCreated', 'Pairs\nMatched', 'Pairs\nVerified', '3D Points\nReconstructed']
pipeline_counts = [
    total_images,
    sum(len(all_results[s]['verified_matches']) * 2 for s in scene_names),  # Approximate pairs created
    total_matches,
    total_matches,
    total_points
]
axes[1, 1].plot(pipeline_stages, pipeline_counts, marker='o', linewidth=2, markersize=10, color='darkviolet')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Pipeline Flow')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_PATH}/pipeline_summary.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Pipeline complete!")
print(f"\n{'='*70}")





