


"""
COLMAP Vocab Tree Performance Verification Notebook
Using the actual vocab_tree_flickr100K_words256K.bin for image similarity search
"""

import os
import sys
import subprocess
import urllib.request
import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import glob


def setup_environment():
    """
    Setup environment with clean NumPy installation at the beginning
    """
    print("Setting up environment for Kaggle...")
    WORK_DIR = '/kaggle/working/gaussian_splatting'

    # ========================================================================
    # STEP 0: Clean NumPy installation BEFORE importing anything
    # ========================================================================
    print("="*70)
    print("STEP 0: Fixing NumPy compatibility (clean install)")
    print("="*70)
    
    try:
        # Uninstall NumPy completely
        print("Uninstalling NumPy 2.x...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', '-y', 'numpy'
        ], check=True, capture_output=True)
        print("âœ“ NumPy uninstalled")
        
        # Install NumPy 1.x
        print("Installing NumPy 1.x...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'numpy<2'
        ], check=True, capture_output=True)
        print("âœ“ NumPy 1.x installed")
        
        # Reinstall key packages that depend on NumPy
        print("Reinstalling NumPy-dependent packages...")
        packages_to_reinstall = [
            'scikit-learn',
            'scipy',
            'matplotlib',
            'pandas'
        ]
        
        for pkg in packages_to_reinstall:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '--force-reinstall',
                    '--no-deps', pkg
                ], check=True, capture_output=True)
                print(f"âœ“ Reinstalled {pkg}")
            except subprocess.CalledProcessError:
                print(f"âš  Failed to reinstall {pkg} (may not be critical)")
        
        # Verify NumPy version
        result = subprocess.run([
            sys.executable, '-c', 'import numpy; print(numpy.__version__)'
        ], capture_output=True, text=True)
        numpy_version = result.stdout.strip()
        print(f"\nâœ“ NumPy version now: {numpy_version}")
        
        if numpy_version.startswith('1.'):
            print("âœ“ NumPy fix successful!")
        else:
            print(f"âš  Warning: NumPy version is {numpy_version}, expected 1.x")
            
    except subprocess.CalledProcessError as e:
        print(f"âš  NumPy fix encountered issues: {e}")
        print("Continuing anyway...")

    # ========================================================================
    # STEP 1: System packages and dependencies
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Installing system packages")
    print("="*70)
    
    # Virtual display setup
    try:
        print("Setting up virtual display...")
        subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'xvfb'], 
                      check=True, capture_output=True)
        
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ':99'
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("âœ“ Virtual display setup")
    except Exception as e:
        print(f"âš  Virtual display skipped: {e}")

    # Install COLMAP
    print("\nInstalling COLMAP...")
    try:
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'colmap'], 
                       check=True, capture_output=True)
        print("âœ“ COLMAP installed")
    except subprocess.CalledProcessError as e:
        print(f"âš  COLMAP warning: {e}")

    # Install build dependencies
    print("\nInstalling build dependencies...")
    try:
        subprocess.run([
            'apt-get', 'install', '-y', '-qq',
            'build-essential', 'cmake', 'git', 'libopenblas-dev'
        ], check=True, capture_output=True)
        print("âœ“ Build dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"âš  Build dependencies warning: {e}")

setup_environment()


print("="*70)
print("COLMAP Vocab Tree Performance Verification System")
print("="*70)

# ========================================
# 1. Download Vocab Tree
# ========================================

def download_vocab_tree():
    """Download official Vocab Tree file (500MB)"""
    vocab_url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
    vocab_path = "vocab_tree_flickr100K_words256K.bin"
    
    if os.path.exists(vocab_path):
        file_size = os.path.getsize(vocab_path) / (1024**2)
        print(f"âœ“ Vocab tree already exists: {vocab_path} ({file_size:.1f}MB)")
        return vocab_path
    
    print(f"\nğŸ“¥ Downloading Vocab Tree... (~500MB)")
    print(f"URL: {vocab_url}")
    print("â�±ï¸�  This may take 5-10 minutes...")
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024**2)
            mb_total = total_size / (1024**2)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='')
        
        urllib.request.urlretrieve(vocab_url, vocab_path, reporthook=report_progress)
        print(f"\nâœ… Download complete: {vocab_path}")
        return vocab_path
    except Exception as e:
        print(f"\nâ�Œ Download failed: {e}")
        raise

# Download vocab tree
vocab_tree_path = download_vocab_tree()
print(vocab_tree_path)



def use_local_images(image_folder):
    """Use local images instead"""
    if not os.path.exists(image_folder):
        raise ValueError(f"Folder does not exist: {image_folder}")
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    image_files.sort()
    print(f"âœ“ Found {len(image_files)} images in {image_folder}/")
    return image_folder, image_files

# Choose image source
USE_SAMPLE_IMAGES = True  # Set to False to use your own images
image_dir, image_paths = use_local_images("/kaggle/input/image-matching-challenge-2025/train/imc2023_theather_imc2024_church")

# ========================================
# 3. Display Images
# ========================================

def display_images(image_paths, cols=5, figsize=(15, 8)):
    """Display images in a grid"""
    n_images = len(image_paths)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, img_path in enumerate(image_paths):
        if idx < len(axes):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(os.path.basename(img_path), fontsize=9)
            axes[idx].axis('off')
    
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

print("\nğŸ“· Test Images:")
display_images(image_paths[0:20])


# ========================================
# 4. Setup COLMAP Environment
# ========================================

def setup_colmap_environment():
    """Setup environment for COLMAP"""
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Check if COLMAP is installed
    try:
        result = subprocess.run(['colmap', '-h'], 
                              capture_output=True, 
                              text=True, 
                              env=env)
        print("âœ“ COLMAP is available")
        return env
    except FileNotFoundError:
        print("â�Œ COLMAP not found!")
        print("Please install COLMAP:")
        print("  Ubuntu/Debian: sudo apt-get install colmap")
        print("  macOS: brew install colmap")
        print("  Or build from source: https://colmap.github.io/install.html")
        raise

env = setup_colmap_environment()

# ========================================
# 5. Run COLMAP Feature Extraction
# ========================================

def extract_features(image_dir, database_path, env):
    """Extract SIFT features using COLMAP"""
    print("\nğŸ”� Extracting SIFT features with COLMAP...")
    
    # Remove existing database
    if os.path.exists(database_path):
        os.remove(database_path)
    
    try:
        subprocess.run([
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', image_dir,
            '--ImageReader.single_camera', '0',
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--SiftExtraction.max_num_features', '8192',
            '--SiftExtraction.num_threads', '4'
        ], check=True, env=env, capture_output=True, text=True)
        
        print("âœ… Feature extraction complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"â�Œ Feature extraction failed: {e.stderr}")
        return False

colmap_workspace = "colmap_workspace"
os.makedirs(colmap_workspace, exist_ok=True)
database_path = os.path.join(colmap_workspace, "database.db")


if extract_features(image_dir, database_path, env):
    # ========================================
    # 6. Vocab Tree Matching (FIXED)
    # ========================================
    
    def vocab_tree_matching(database_path, vocab_tree_path, env, num_images=10):
        """Use vocab tree to match similar images"""
        print(f"\nğŸŒ³ Running Vocab Tree matching...")
        print(f"   Finding top {num_images} similar images for each query")
        
        try:
            # Use vocab_tree_matcher instead of vocab_tree_retriever
            subprocess.run([
                'colmap', 'vocab_tree_matcher',
                '--database_path', database_path,
                '--VocabTreeMatching.vocab_tree_path', vocab_tree_path,
                '--VocabTreeMatching.num_images', str(num_images),
                '--VocabTreeMatching.num_nearest_neighbors', '10',
                '--VocabTreeMatching.num_checks', '256',
                '--VocabTreeMatching.num_images_after_verification', str(num_images)
            ], check=True, env=env, capture_output=True, text=True)
            
            print("âœ… Vocab tree matching complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"â�Œ Vocab tree matching failed: {e.stderr}")
            return False
    
    if vocab_tree_matching(database_path, vocab_tree_path, env, num_images=10):
        # ========================================
        # 7. Extract and Visualize Results
        # ========================================
        
        def decode_pair_id(pair_id):
            """Decode COLMAP pair_id to image_id1 and image_id2"""
            # COLMAP encodes: pair_id = image_id1 * 2147483647 + image_id2
            # where image_id1 < image_id2
            image_id2 = pair_id % 2147483647
            image_id1 = pair_id // 2147483647
            return int(image_id1), int(image_id2)
        
        def extract_image_pairs(database_path):
            """Extract image pair matches from COLMAP database"""
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            
            # Get image names
            cursor.execute("SELECT image_id, name FROM images")
            images = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get number of matches for each pair (using pair_id)
            cursor.execute("""
                SELECT pair_id, rows
                FROM two_view_geometries
                WHERE rows > 0
                ORDER BY rows DESC
            """)
            
            # Decode pair_ids to get image_id1 and image_id2
            match_counts = []
            for pair_id, num_matches in cursor.fetchall():
                img_id1, img_id2 = decode_pair_id(pair_id)
                match_counts.append((img_id1, img_id2, num_matches))
            
            conn.close()
            
            print(f"\nğŸ“Š Database Statistics:")
            print(f"   Total images: {len(images)}")
            print(f"   Image pairs with matches: {len(match_counts)}")
            
            return images, match_counts
        
        def visualize_similar_images(database_path, image_dir, top_n=5):
            """Visualize similar image pairs"""
            images, match_counts = extract_image_pairs(database_path)
            
            if not match_counts:
                print("âš ï¸�  No matches found")
                return
            
            # Group by query image
            query_matches = {}
            for img_id1, img_id2, num_matches in match_counts:
                if img_id1 not in query_matches:
                    query_matches[img_id1] = []
                query_matches[img_id1].append((img_id2, num_matches))
            
            # Visualize for each query image (show first 3 queries)
            for query_id in list(query_matches.keys())[:min(3, len(query_matches))]:
                matches = sorted(query_matches[query_id], 
                               key=lambda x: x[1], reverse=True)[:top_n]
                
                print(f"\nğŸ”� Query Image: {images[query_id]}")
                
                fig, axes = plt.subplots(1, top_n + 1, figsize=(20, 4))
                
                # Display query image
                query_path = os.path.join(image_dir, images[query_id])
                query_img = cv2.imread(query_path)
                if query_img is not None:
                    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                    axes[0].imshow(query_img)
                    axes[0].set_title(f"Query\n{images[query_id]}", 
                                    fontweight='bold', fontsize=10)
                    axes[0].axis('off')
                
                # Display similar images
                for idx, (match_id, num_matches) in enumerate(matches):
                    match_path = os.path.join(image_dir, images[match_id])
                    match_img = cv2.imread(match_path)
                    if match_img is not None:
                        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                        axes[idx + 1].imshow(match_img)
                        axes[idx + 1].set_title(
                            f"Match #{idx+1}\n{images[match_id]}\n"
                            f"({num_matches} matches)",
                            fontsize=9
                        )
                        axes[idx + 1].axis('off')
                
                plt.tight_layout()
                plt.show()
        
        # ========================================
        # 8. Display Results
        # ========================================
        
        print("\n" + "="*70)
        print("ğŸ“ˆ VOCAB TREE PERFORMANCE RESULTS")
        print("="*70)
        
        visualize_similar_images(database_path, image_dir, top_n=5)
        
        # ========================================
        # 9. Create Similarity Matrix
        # ========================================
        
        def create_similarity_matrix(database_path):
            """Create similarity matrix from match counts"""
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT image_id FROM images ORDER BY image_id")
            image_ids = [row[0] for row in cursor.fetchall()]
            n = len(image_ids)
            
            # Initialize matrix
            similarity_matrix = np.zeros((n, n))
            
            # Fill with match counts (decode pair_id)
            cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
            for pair_id, matches in cursor.fetchall():
                img_id1, img_id2 = decode_pair_id(pair_id)
                try:
                    idx1 = image_ids.index(img_id1)
                    idx2 = image_ids.index(img_id2)
                    similarity_matrix[idx1, idx2] = matches
                    similarity_matrix[idx2, idx1] = matches
                except ValueError:
                    # Skip if image_id not found in list
                    continue
            
            conn.close()
            
            return similarity_matrix, image_ids
        
        print("\nğŸ“Š Creating similarity matrix...")
        sim_matrix, img_ids = create_similarity_matrix(database_path)
        
        # Normalize for visualization
        max_matches = np.max(sim_matrix)
        if max_matches > 0:
            sim_matrix_norm = sim_matrix / max_matches
        else:
            sim_matrix_norm = sim_matrix
        
        # Visualize
        plt.figure(figsize=(14, 12))
        plt.imshow(sim_matrix_norm, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Normalized Match Count')
        plt.title('Image Similarity Matrix (Vocab Tree Results)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Image Index')
        plt.ylabel('Image Index')
        
        # Get image names for labels
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
        image_names = [row[1][:15] for row in cursor.fetchall()]
        conn.close()
        
        plt.xticks(range(len(image_names)), image_names, rotation=90, ha='right', fontsize=8)
        plt.yticks(range(len(image_names)), image_names, fontsize=8)
        plt.tight_layout()
        plt.show()
        
        # ========================================
        # 10. Summary Statistics
        # ========================================
        
        print("\n" + "="*70)
        print("ğŸ“Š SUMMARY STATISTICS")
        print("="*70)
        
        total_matches = int(np.sum(sim_matrix) / 2)  # Divide by 2 because matrix is symmetric
        avg_matches = np.mean(sim_matrix[sim_matrix > 0]) if np.any(sim_matrix > 0) else 0
        max_match = int(np.max(sim_matrix))
        
        print(f"Vocab Tree File: {vocab_tree_path}")
        print(f"File Size: {os.path.getsize(vocab_tree_path) / (1024**2):.1f} MB")
        print(f"Number of Images: {len(image_paths)}")
        print(f"Total Image Pairs Matched: {total_matches}")
        print(f"Average Feature Matches per Pair: {avg_matches:.1f}")
        print(f"Maximum Feature Matches: {max_match}")
        
        # Get feature counts per image
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rows FROM keypoints")
        feature_counts = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"Average Features per Image: {np.mean(feature_counts):.1f}")
        print(f"Total Features Extracted: {sum(feature_counts)}")
        
        print("\nâœ… Vocab Tree performance verification complete!")

else:
    print("\nâ�Œ Feature extraction failed. Cannot proceed with vocab tree matching.")

print("\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)

