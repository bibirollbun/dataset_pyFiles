# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install --no-index /kaggle/input/imc2025-packages-pycolmap-lightglue-rerun-kornia/* --no-deps
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth


import sys, os, gc, h5py, pycolmap
import numpy as np
import pandas as pd
import torch
import kornia as K
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from lightglue import ALIKED, LightGlue
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc
import warnings
warnings.filterwarnings('ignore')


def optimize_memory():
    """Free up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"ğŸ§¹ Cleared GPU cache")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

optimize_memory()


print("\nğŸ§ª Testing model loading...")

# Test ALIKED
try:
    from lightglue import ALIKED
    extractor = ALIKED(model_name='aliked-n16', max_num_keypoints=1024).eval()
    print("âœ… ALIKED loaded successfully")
    del extractor
    optimize_memory()
except Exception as e:
    print(f"â�Œ ALIKED error: {e}")

# Test LightGlue
try:
    from lightglue import LightGlue
    matcher = LightGlue(features='aliked').eval()
    print("âœ… LightGlue loaded successfully")
    del matcher
    optimize_memory()
except Exception as e:
    print(f"â�Œ LightGlue error: {e}")

# Test DINOv2 (this will download ~330MB)
try:
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    print("âœ… DINOv2 loaded successfully")
    del model
    optimize_memory()
except Exception as e:
    print(f"â�Œ DINOv2 error: {e}")

print("\nâœ… All models tested successfully!")
print("\nğŸš€ Ready to run the pipeline!")


data_path = Path('/kaggle/input/image-matching-challenge-2025')

# Check train data
train_labels = pd.read_csv(data_path / 'train_labels.csv')
print(f"\nğŸ“Š Training Data:")
print(f"   Total images: {len(train_labels)}")
print(f"   Datasets: {train_labels['dataset'].nunique()}")
print(f"   Scenes: {train_labels['scene'].nunique()}")

print(f"\nğŸ“‚ Available datasets:")
for dataset in train_labels['dataset'].unique():
    scenes = train_labels[train_labels['dataset'] == dataset]['scene'].unique()
    n_images = len(train_labels[train_labels['dataset'] == dataset])
    print(f"   â€¢ {dataset}: {len(scenes)} scenes, {n_images} images")

print(f"\nğŸ’¡ Start with a small scene first (e.g., 'fountain' from imc2023_haiper)")


class Config:
    # Paths
    DATA_PATH = Path('/kaggle/input/image-matching-challenge-2025')
    TRAIN_PATH = DATA_PATH / 'train'
    TEST_PATH = DATA_PATH / 'test'
    
    # Processing settings
    IS_TRAIN = True
    DATASETS_TO_PROCESS = ['imc2023_haiper']  # Start with one, expand later
    
    # Image processing
    RESIZE_MAX = 1280
    TILE_SIZE = 1280
    TILE_OVERLAP = 50
    
    # Feature extraction
    ALIKED_MODEL = 'aliked-n16'  # Fast and accurate
    NUM_FEATURES = 2048
    
    # Matching
    LIGHTGLUE_FEATURES = 'aliked'
    MATCH_THRESHOLD = 0.2
    
    # Image retrieval
    DINO_MODEL = 'facebook/dinov2-small'
    RETRIEVAL_TOP_K = 20  # Match top K similar images
    SIM_THRESHOLD = 0.6
    
    # 3D reconstruction
    MIN_MATCHES = 15
    RANSAC_THRESH = 4.0
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1  # Process one image pair at a time

config = Config()
print(f"ğŸš€ Running on {config.DEVICE} | Mode: {'TRAIN' if config.IS_TRAIN else 'TEST'}")



def load_image(img_path, resize_max=None):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    
    if resize_max:
        w, h = img.size
        scale = min(resize_max / max(w, h), 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return np.array(img)

def to_tensor(img):
    """Convert image to tensor"""
    if len(img.shape) == 2:
        img = img[..., None]
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

def compute_rotation_error(R_pred, R_gt):
    """Compute rotation error in degrees"""
    R_err = R_pred @ R_gt.T
    trace = np.trace(R_err)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle)

def compute_translation_error(t_pred, t_gt):
    """Compute translation error (angular error)"""
    t_pred_norm = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)
    angle = np.arccos(np.clip(np.dot(t_pred_norm, t_gt_norm), -1, 1))
    return np.degrees(angle)


class ImageRetrieval:
    def __init__(self, model_name=config.DINO_MODEL, device=config.DEVICE):
        print("ğŸ“¸ Initializing DINOv2 for image retrieval...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.features_cache = {}
    
    @torch.no_grad()
    def extract_features(self, img_path):
        """Extract global image features"""
        if str(img_path) in self.features_cache:
            return self.features_cache[str(img_path)]
        
        img = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Use CLS token as global feature
        features = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
        self.features_cache[str(img_path)] = features
        
        return features
    
    def find_similar_images(self, query_path, candidate_paths, top_k=20):
        """Find top-k similar images using cosine similarity"""
        query_feat = self.extract_features(query_path)
        
        similarities = []
        for cand_path in candidate_paths:
            if cand_path == query_path:
                continue
            cand_feat = self.extract_features(cand_path)
            sim = 1 - cosine(query_feat, cand_feat)
            similarities.append((cand_path, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cluster_images(self, img_paths, eps=0.3, min_samples=2):
        """Cluster images using DBSCAN on feature space"""
        features = np.array([self.extract_features(p) for p in img_paths])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(features)
        
        # Group images by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(img_paths[idx])
        
        return clusters


class FeatureMatcher:
    """ALIKED + LightGlue feature matching - DEBUGGED VERSION"""
    
    def __init__(self, device=config.DEVICE):
        print("ğŸ”� Initializing ALIKED + LightGlue...")
        self.device = device
        
        # Load models
        self.extractor = ALIKED(
            model_name=config.ALIKED_MODEL,
            max_num_keypoints=config.NUM_FEATURES,
            detection_threshold=0.2,
            resize=config.RESIZE_MAX
        ).to(device).eval()
        
        self.matcher = LightGlue(
            features=config.LIGHTGLUE_FEATURES,
            depth_confidence=-1,
            width_confidence=-1
        ).to(device).eval()
    
    @torch.no_grad()
    def extract_features(self, img_path):
        """Extract keypoints and descriptors"""
        img = load_image(img_path, config.RESIZE_MAX)
        img_tensor = to_tensor(img).unsqueeze(0).to(self.device)

        # Extract features - pass image tensor directly, not in dict
        features = self.extractor.extract(img_tensor)
        
        kpts = features['keypoints']
        desc = features['descriptors']
        
        # Handle different descriptor formats
        if desc.ndim == 4:
            # Dense descriptors - extract at keypoint locations
            B, C, H, W = desc.shape
            kpts_norm = kpts.clone()
            kpts_norm[..., 0] = kpts_norm[..., 0] / img_tensor.shape[-1] * (W - 1)
            kpts_norm[..., 1] = kpts_norm[..., 1] / img_tensor.shape[-2] * (H - 1)
            
            # Normalize to [-1, 1] for grid_sample
            grid = kpts_norm.clone()
            grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
            grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
            grid = grid.unsqueeze(1)  # (B, 1, K, 2)
            
            # Sample descriptors at keypoint locations
            desc = F.grid_sample(desc, grid, mode='bilinear', align_corners=True)
            desc = desc.squeeze(2).permute(0, 2, 1).contiguous()  # (B, K, C)
        
        # Ensure contiguous memory
        kpts = kpts.contiguous()
        desc = desc.contiguous()
        
        # Add image_size field (required by LightGlue)
        return {
            'keypoints': kpts,           # (B, K, 2)
            'descriptors': desc,         # (B, K, C)
            'image': img_tensor,         # (B, 3, H, W)
            'image_size': torch.tensor([[img_tensor.shape[-2], img_tensor.shape[-1]]], 
                                       device=self.device, dtype=torch.float32)  # (B, 2) = [H, W]
        }
    
    @torch.no_grad()
    def match_pair(self, feat0, feat1):
        """Match features between two images"""
        
        # Prepare data for LightGlue - use exact format expected
        data = {
            'image0': {
                'image': feat0['image'],
                'keypoints': feat0['keypoints'],
                'descriptors': feat0['descriptors'],
                'image_size': feat0['image_size']
            },
            'image1': {
                'image': feat1['image'],
                'keypoints': feat1['keypoints'],
                'descriptors': feat1['descriptors'],
                'image_size': feat1['image_size']
            }
        }
        
        # Run matcher
        try:
            matches = self.matcher(data)
        except Exception as e:
            print(f"âš ï¸� Matcher error: {e}")
            # Print debug info
            print("Debug shapes:")
            print(f"  image0: {feat0['image'].shape}")
            print(f"  image1: {feat1['image'].shape}")
            print(f"  keypoints0: {feat0['keypoints'].shape}")
            print(f"  keypoints1: {feat1['keypoints'].shape}")
            print(f"  descriptors0: {feat0['descriptors'].shape}")
            print(f"  descriptors1: {feat1['descriptors'].shape}")
            print(f"  image_size0: {feat0['image_size'].shape if 'image_size' in feat0 else 'N/A'}")
            print(f"  image_size1: {feat1['image_size'].shape if 'image_size' in feat1 else 'N/A'}")
            import traceback
            traceback.print_exc()
            return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
        
        # Extract matches
        matches0 = matches['matches0']  # (B, K0)
        if matches0.ndim == 2:
            matches0 = matches0[0]  # Remove batch dim -> (K0,)
        
        valid = matches0 > -1
        
        if valid.sum() == 0:
            return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
        
        # Get matched keypoints
        kpts0 = feat0['keypoints'][0] if feat0['keypoints'].ndim == 3 else feat0['keypoints']
        kpts1 = feat1['keypoints'][0] if feat1['keypoints'].ndim == 3 else feat1['keypoints']
        
        mkpts0 = kpts0[valid].cpu().numpy()
        mkpts1 = kpts1[matches0[valid]].cpu().numpy()
        
        # Get confidence scores
        if 'matching_scores0' in matches:
            scores = matches['matching_scores0']
            if scores.ndim == 2:
                scores = scores[0]
            confidence = scores[valid].cpu().numpy()
        else:
            confidence = np.ones(len(mkpts0))
        
        return mkpts0, mkpts1, confidence


class SceneReconstructor:
    def __init__(self):
        print("ğŸ—ºï¸� Initializing PyColmap reconstructor...")
        self.reconstruction = None
    
    def add_matches(self, img_pairs, matches_dict, img_paths):
        """Prepare matches for reconstruction - MEMORY OPTIMIZED"""
        database_path = 'reconstruction.db'
        
        # Create database
        if os.path.exists(database_path):
            os.remove(database_path)
        
        # Use correct PyColmap API
        db = pycolmap.Database(database_path)
        
        # Create camera model
        camera = pycolmap.Camera(
            model='SIMPLE_RADIAL',
            width=1280,
            height=720,
            params=[1000.0, 640.0, 360.0, 0.0]
        )
        
        # Write camera to database
        camera_id = db.write_camera(camera)
        
        # Build image ID map first (minimal memory)
        img_id_map = {}
        img_to_kpts = {}  # Track which images have keypoints
        
        for idx, img_path in enumerate(img_paths):
            image = pycolmap.Image(
                name=str(img_path.name),
                camera_id=camera_id,
                points2D=[]
            )
            img_id = db.write_image(image)
            img_id_map[str(img_path)] = img_id
        
        # Process matches in batches to avoid memory issues
        print(f"ğŸ“� Writing {len(matches_dict)} image pair matches...")
        
        for idx, ((path0, path1), (mkpts0, mkpts1, conf)) in enumerate(matches_dict.items()):
            if len(mkpts0) < config.MIN_MATCHES:
                continue
            
            img_id0 = img_id_map.get(str(path0))
            img_id1 = img_id_map.get(str(path1))
            
            if img_id0 is None or img_id1 is None:
                continue
            
            # Only write keypoints ONCE per image (not for every pair!)
            if img_id0 not in img_to_kpts:
                db.write_keypoints(img_id0, mkpts0.astype(np.float32))
                img_to_kpts[img_id0] = len(mkpts0)
            
            if img_id1 not in img_to_kpts:
                db.write_keypoints(img_id1, mkpts1.astype(np.float32))
                img_to_kpts[img_id1] = len(mkpts1)
            
            # Create match array (indices of corresponding points)
            matches_array = np.column_stack([
                np.arange(len(mkpts0)), 
                np.arange(len(mkpts1))
            ]).astype(np.uint32)
            
            db.write_matches(img_id0, img_id1, matches_array)
            
            # Free memory every 10 pairs
            if (idx + 1) % 10 == 0:
                del matches_array
                gc.collect()
        
        db.close()
        print(f"âœ… Database created with {len(img_to_kpts)} images")
        return database_path
    
    def reconstruct(self, database_path, output_path='sparse'):
        """Run incremental reconstruction - MEMORY OPTIMIZED"""
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Run mapper with memory-conscious options
        options = pycolmap.IncrementalPipelineOptions()
        options.min_num_matches = config.MIN_MATCHES
        options.init_min_num_inliers = 50
        options.abs_pose_min_num_inliers = 20
        
        print("ğŸ”¨ Running COLMAP reconstruction...")
        
        try:
            maps = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path='.',
                output_path=output_path,
                options=options
            )
        except Exception as e:
            print(f"âš ï¸� Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if not maps or len(maps) == 0:
            print("âš ï¸� Reconstruction produced no models!")
            return None
        
        self.reconstruction = maps[0]
        print(f"âœ… Reconstructed {len(self.reconstruction.images)} images")
        print(f"   {len(self.reconstruction.points3D)} 3D points")
        
        return self.reconstruction


class PipelineEvaluator:
    def __init__(self, labels_df):
        self.labels_df = labels_df
    
    def parse_matrix(self, matrix_str):
        if pd.isna(matrix_str) or matrix_str == 'nan':
            return None
        
        # Remove 'nan;' prefix and split
        matrix_str = str(matrix_str).replace('nan;', '')
        values = [float(x) for x in matrix_str.split(';') if x]
        
        if len(values) == 9:  # Rotation matrix
            return np.array(values).reshape(3, 3)
        elif len(values) == 3:  # Translation vector
            return np.array(values)
        return None
    
    def evaluate_scene(self, scene_name, reconstruction):
        """Evaluate reconstruction for a scene"""
        scene_labels = self.labels_df[self.labels_df['scene'] == scene_name]
        
        results = {
            'rotation_errors': [],
            'translation_errors': [],
            'num_images': len(reconstruction.images) if reconstruction else 0,
            'num_points': len(reconstruction.points3D) if reconstruction else 0
        }
        
        if not reconstruction:
            return results
        
        # Compare reconstructed poses with ground truth
        for img_id, image in reconstruction.images.items():
            img_name = image.name
            
            # Find ground truth
            gt_row = scene_labels[scene_labels['image'] == img_name]
            if gt_row.empty:
                continue
            
            gt_row = gt_row.iloc[0]
            R_gt = self.parse_matrix(gt_row['rotation_matrix'])
            t_gt = self.parse_matrix(gt_row['translation_vector'])
            
            if R_gt is None or t_gt is None:
                continue
            
            # Get reconstructed pose
            R_pred = image.rotmat()
            t_pred = image.projection_center()
            
            # Compute errors
            rot_err = compute_rotation_error(R_pred, R_gt)
            trans_err = compute_translation_error(t_pred, t_gt)
            
            results['rotation_errors'].append(rot_err)
            results['translation_errors'].append(trans_err)
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("ğŸ“Š EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nğŸ�—ï¸� Reconstruction Stats:")
        print(f"   â€¢ Images reconstructed: {results['num_images']}")
        print(f"   â€¢ 3D points: {results['num_points']}")
        
        if results['rotation_errors']:
            rot_errs = results['rotation_errors']
            trans_errs = results['translation_errors']
            
            print(f"\nğŸ“� Pose Accuracy:")
            print(f"   â€¢ Rotation error (mean): {np.mean(rot_errs):.2f}Â°")
            print(f"   â€¢ Rotation error (median): {np.median(rot_errs):.2f}Â°")
            print(f"   â€¢ Translation error (mean): {np.mean(trans_errs):.2f}Â°")
            print(f"   â€¢ Translation error (median): {np.median(trans_errs):.2f}Â°")
            
            # Compute accuracy at thresholds
            thresholds = [(5, 10), (10, 20), (20, 30)]
            print(f"\nğŸ�¯ Accuracy at thresholds:")
            for rot_th, trans_th in thresholds:
                correct = sum(1 for r, t in zip(rot_errs, trans_errs) 
                            if r < rot_th and t < trans_th)
                acc = 100 * correct / len(rot_errs)
                print(f"   â€¢ <{rot_th}Â°/<{trans_th}Â°: {acc:.1f}%")
        
        print("="*60 + "\n")


def run_pipeline(dataset_name, scene_name):
    print(f"\n{'='*60}")
    print(f"ğŸ�¬ Processing: {dataset_name}/{scene_name}")
    print(f"{'='*60}\n")
    
    # Load labels
    labels_path = '/kaggle/input/image-matching-challenge-2025/train_labels.csv'
    labels_df = pd.read_csv(labels_path)
    
    # Filter for this scene
    scene_df = labels_df[
        (labels_df['dataset'] == dataset_name) & 
        (labels_df['scene'] == scene_name)
    ]
    
    if scene_df.empty:
        print(f"â�Œ No data found for {dataset_name}/{scene_name}")
        return
    
    # Get image paths
    img_dir = '/kaggle/input/image-matching-challenge-2025/train/imc2023_haiper'
    img_dir = Path(img_dir)
    img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
    
    print(f"ğŸ“‚ Found {len(img_paths)} images")
    
    # Phase 1: Image Retrieval (MEMORY OPTIMIZED)
    print("\n" + "-"*60)
    print("PHASE 1: Image Retrieval")
    print("-"*60)
    
    retriever = ImageRetrieval()
    
    # Limit to first 8 images to avoid memory issues (reduced from 10)
    sample_imgs = img_paths[:8]
    
    # Find image pairs to match (smart pairing)
    pairs_to_match = []
    for img_path in tqdm(sample_imgs, desc="Finding pairs"):
        similar = retriever.find_similar_images(
            img_path, sample_imgs, top_k=min(3, len(sample_imgs)-1)  # Reduced from 5
        )
        for sim_path, sim_score in similar:
            if sim_score >= config.SIM_THRESHOLD:
                pairs_to_match.append((img_path, sim_path, sim_score))
    
    print(f"âœ… Found {len(pairs_to_match)} promising image pairs")
    
    # Clear retriever completely
    retriever.features_cache.clear()
    del retriever.model, retriever.processor, retriever
    gc.collect()
    torch.cuda.empty_cache()
    
    # Phase 2: Feature Matching (AGGRESSIVELY MEMORY OPTIMIZED)
    print("\n" + "-"*60)
    print("PHASE 2: Feature Matching")
    print("-"*60)
    
    matcher = FeatureMatcher()
    matches_dict = {}
    
    # Drastically limit pairs to avoid OOM
    max_pairs = min(20, len(pairs_to_match))  # Reduced from 30
    print(f"âš™ï¸� Processing {max_pairs} pairs (out of {len(pairs_to_match)} found)")
    
    for idx, (img0_path, img1_path, _) in enumerate(tqdm(pairs_to_match[:max_pairs], desc="Matching")):
        try:
            # Extract features
            feat0 = matcher.extract_features(img0_path)
            feat1 = matcher.extract_features(img1_path)
            
            # Match
            mkpts0, mkpts1, conf = matcher.match_pair(feat0, feat1)
            
            # CRITICAL: Free tensors IMMEDIATELY
            del feat0['image'], feat0['keypoints'], feat0['descriptors'], feat0['image_size']
            del feat1['image'], feat1['keypoints'], feat1['descriptors'], feat1['image_size']
            del feat0, feat1
            
            if len(mkpts0) >= config.MIN_MATCHES:
                matches_dict[(img0_path, img1_path)] = (mkpts0, mkpts1, conf)
            else:
                # Free even if not saving
                del mkpts0, mkpts1, conf
            
            # AGGRESSIVE cleanup every 3 matches
            if (idx + 1) % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"âš ï¸� Error matching {img0_path.name} - {img1_path.name}: {e}")
            # Emergency cleanup
            gc.collect()
            torch.cuda.empty_cache()
            continue
    
    print(f"âœ… Matched {len(matches_dict)} image pairs")
    
    # Free matcher completely
    del matcher.extractor, matcher.matcher, matcher
    gc.collect()
    torch.cuda.empty_cache()
    
    # Phase 3: 3D Reconstruction
    print("\n" + "-"*60)
    print("PHASE 3: 3D Reconstruction")
    print("-"*60)
    
    reconstructor = SceneReconstructor()
    database_path = reconstructor.add_matches(
        list(matches_dict.keys()), matches_dict, sample_imgs
    )
    
    # Free matches_dict after writing to database
    del matches_dict
    gc.collect()
    
    reconstruction = reconstructor.reconstruct(database_path)
    
    # Phase 4: Evaluation
    print("\n" + "-"*60)
    print("PHASE 4: Evaluation")
    print("-"*60)
    
    evaluator = PipelineEvaluator(labels_df)
    results = evaluator.evaluate_scene(scene_name, reconstruction)
    evaluator.print_results(results)
    
    # Visualize (simple 3D plot) - only if points exist
    if reconstruction and len(reconstruction.points3D) > 0:
        visualize_3d_points(reconstruction)
    
    # Cleanup
    del reconstructor, evaluator
    if reconstruction:
        del reconstruction
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def visualize_3d_points(reconstruction, max_points=500):  # Reduced from 1000
    """Visualize 3D reconstruction - ULTRA MEMORY OPTIMIZED"""
    print("\nğŸ“Š Generating 3D visualization...")
    
    # Extract 3D points (limit to avoid memory issues)
    points = []
    colors = []
    
    for point_id, point3D in reconstruction.points3D.items():
        if len(points) >= max_points:
            break
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)
    
    if len(points) == 0:
        print("âš ï¸� No 3D points to visualize")
        return
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))  # Further reduced
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=0.5, alpha=0.6)  # Smaller points
    
    # Camera positions
    cam_positions = []
    for img_id, image in reconstruction.images.items():
        cam_positions.append(image.projection_center())
    
    if cam_positions:
        cam_positions = np.array(cam_positions)
        ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
                   c='red', s=80, marker='^', label='Cameras')  # Smaller cameras
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction ({len(points)} points)', fontsize=10)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reconstruction_3d.png', dpi=80, bbox_inches='tight')  # Lower DPI
    print("âœ… Saved visualization to reconstruction_3d.png")
    plt.show()
    plt.close(fig)  # Free memory
    
    # Cleanup
    del points, colors, cam_positions, fig, ax
    gc.collect()


# MODIFIED: Only run image retrieval + matching (Phases 1-2)
# Reconstruction moved to separate cells for debugging

dataset = 'imc2023_haiper'
scene = 'fountain'

print(f"\n{'='*60}")
print(f"ğŸ�¬ Processing: {dataset}/{scene}")
print(f"{'='*60}\n")

# Load labels
labels_path = '/kaggle/input/image-matching-challenge-2025/train_labels.csv'
labels_df = pd.read_csv(labels_path)

# Filter for this scene
scene_df = labels_df[
    (labels_df['dataset'] == dataset) & 
    (labels_df['scene'] == scene)
]

# Get image paths
img_dir = '/kaggle/input/image-matching-challenge-2025/train/imc2023_haiper'
img_dir = Path(img_dir)
img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))

print(f"ğŸ“‚ Found {len(img_paths)} images")

# Phase 1: Image Retrieval
print("\n" + "-"*60)
print("PHASE 1: Image Retrieval")
print("-"*60)

retriever = ImageRetrieval()

# Limit to first 8 images
sample_imgs = img_paths[:8]

# Find image pairs to match
pairs_to_match = []
for img_path in tqdm(sample_imgs, desc="Finding pairs"):
    similar = retriever.find_similar_images(
        img_path, sample_imgs, top_k=min(3, len(sample_imgs)-1)
    )
    for sim_path, sim_score in similar:
        if sim_score >= config.SIM_THRESHOLD:
            pairs_to_match.append((img_path, sim_path, sim_score))

print(f"âœ… Found {len(pairs_to_match)} promising image pairs")

# Clear retriever completely
retriever.features_cache.clear()
del retriever.model, retriever.processor, retriever
gc.collect()
torch.cuda.empty_cache()

# Phase 2: Feature Matching
print("\n" + "-"*60)
print("PHASE 2: Feature Matching")
print("-"*60)

matcher = FeatureMatcher()
matches_dict = {}

# Process limited pairs
max_pairs = min(20, len(pairs_to_match))
print(f"âš™ï¸� Processing {max_pairs} pairs (out of {len(pairs_to_match)} found)")

for idx, (img0_path, img1_path, _) in enumerate(tqdm(pairs_to_match[:max_pairs], desc="Matching")):
    try:
        # Extract features
        feat0 = matcher.extract_features(img0_path)
        feat1 = matcher.extract_features(img1_path)
        
        # Match
        mkpts0, mkpts1, conf = matcher.match_pair(feat0, feat1)
        
        # Free tensors IMMEDIATELY
        del feat0['image'], feat0['keypoints'], feat0['descriptors'], feat0['image_size']
        del feat1['image'], feat1['keypoints'], feat1['descriptors'], feat1['image_size']
        del feat0, feat1
        
        if len(mkpts0) >= config.MIN_MATCHES:
            matches_dict[(img0_path, img1_path)] = (mkpts0, mkpts1, conf)
        else:
            del mkpts0, mkpts1, conf
        
        # Cleanup every 3 matches
        if (idx + 1) % 3 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"âš ï¸� Error matching {img0_path.name} - {img1_path.name}: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        continue

print(f"âœ… Matched {len(matches_dict)} image pairs")

# Free matcher completely
del matcher.extractor, matcher.matcher, matcher
gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*60)
print("âœ… PHASES 1-2 COMPLETED")
print("="*60)
print(f"ğŸ“Š Summary:")
print(f"   â€¢ Images: {len(sample_imgs)}")
print(f"   â€¢ Matched pairs: {len(matches_dict)}")
print(f"   â€¢ Total matches: {sum(len(v[0]) for v in matches_dict.values())}")
print(f"\nğŸ’¾ Variables saved:")
print(f"   â€¢ sample_imgs: List of {len(sample_imgs)} image paths")
print(f"   â€¢ matches_dict: {len(matches_dict)} image pairs with matches")
print(f"   â€¢ labels_df: Ground truth labels")
print(f"   â€¢ scene: '{scene}'")
print(f"\nâ–¶ï¸� Run next cell to analyze reconstruction code")


print("\n" + "-"*60)
print("PHASE 3: Evaluate Matches Against Ground Truth")
print("-"*60)

print("\nğŸ’¡ Skipping expensive COLMAP reconstruction...")
print("   Instead, evaluating feature matches directly using ground truth poses")

# Detect actual scene from image names
sample_scene_names = [img.name for img in sample_imgs]
print(f"\nğŸ”� Detected images: {sample_scene_names[:3]}... (showing first 3)")

# Try to auto-detect scene from filenames
detected_scenes = set()
for img_name in sample_scene_names:
    # Extract scene prefix (e.g., 'bike', 'fountain', 'chairs')
    scene_prefix = img_name.split('_')[0]
    detected_scenes.add(scene_prefix)

print(f"   Detected scene prefixes: {detected_scenes}")

# Use all available ground truth (don't filter by scene if mismatch)
scene_labels = labels_df.copy()
print(f"   Loaded {len(scene_labels)} ground truth entries")

# Helper to parse ground truth matrices
def parse_matrix(matrix_str):
    """Parse rotation/translation matrix from string"""
    if pd.isna(matrix_str) or matrix_str == 'nan':
        return None
    
    matrix_str = str(matrix_str).replace('nan;', '')
    values = [float(x) for x in matrix_str.split(';') if x]
    
    if len(values) == 9:  # Rotation matrix
        return np.array(values).reshape(3, 3)
    elif len(values) == 3:  # Translation vector
        return np.array(values)
    return None

# Evaluate match quality using epipolar geometry
print("\nğŸ“� Computing epipolar errors for matched pairs...")

epipolar_errors = []
match_stats = []

for (img0_path, img1_path), (mkpts0, mkpts1, conf) in matches_dict.items():
    # Get ground truth for both images
    gt0 = scene_labels[scene_labels['image'] == img0_path.name]
    gt1 = scene_labels[scene_labels['image'] == img1_path.name]
    
    if gt0.empty or gt1.empty:
        print(f"âš ï¸�  Missing ground truth for: {img0_path.name} or {img1_path.name}")
        continue
    
    gt0 = gt0.iloc[0]
    gt1 = gt1.iloc[0]
    
    # Parse ground truth poses
    R0 = parse_matrix(gt0['rotation_matrix'])
    t0 = parse_matrix(gt0['translation_vector'])
    R1 = parse_matrix(gt1['rotation_matrix'])
    t1 = parse_matrix(gt1['translation_vector'])
    
    if R0 is None or t0 is None or R1 is None or t1 is None:
        print(f"âš ï¸�  Invalid pose data for: {img0_path.name} or {img1_path.name}")
        continue
    
    # Compute relative pose (1 w.r.t 0)
    R_rel = R1 @ R0.T
    t_rel = t1 - R_rel @ t0
    
    # Normalize translation
    t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-8)
    
    # Compute essential matrix
    t_cross = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    E = t_cross @ R_rel
    
    # Compute epipolar errors (assuming calibrated coordinates)
    # Using simple pinhole: K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    f = 1000.0
    cx, cy = 640.0, 360.0
    
    # Normalize keypoints
    pts0_norm = np.column_stack([
        (mkpts0[:, 0] - cx) / f,
        (mkpts0[:, 1] - cy) / f,
        np.ones(len(mkpts0))
    ])
    
    pts1_norm = np.column_stack([
        (mkpts1[:, 0] - cx) / f,
        (mkpts1[:, 1] - cy) / f,
        np.ones(len(mkpts1))
    ])
    
    # Compute symmetric epipolar error
    errors = []
    for p0, p1 in zip(pts0_norm, pts1_norm):
        # Epipolar constraint: p1^T * E * p0 = 0
        # Error is the distance from p1 to the epipolar line
        Ep0 = E @ p0  # Epipolar line in image 1
        
        # Sampson distance (approximation of geometric distance)
        numerator = (p1 @ Ep0) ** 2
        denominator = Ep0[0]**2 + Ep0[1]**2 + 1e-8
        err = np.sqrt(numerator / denominator)
        errors.append(err)
    
    errors = np.array(errors)
    epipolar_errors.extend(errors)
    
    # Compute inlier ratio (using 1e-3 threshold for normalized coords)
    inlier_ratio = np.mean(errors < 1e-3)
    
    match_stats.append({
        'pair': f"{img0_path.name} â†” {img1_path.name}",
        'num_matches': len(mkpts0),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'inlier_ratio': inlier_ratio,
        'mean_confidence': np.mean(conf)
    })

epipolar_errors = np.array(epipolar_errors)

# Print detailed results
print("\n" + "="*70)
print("ğŸ“Š MATCHING QUALITY EVALUATION")
print("="*70)

print(f"\nğŸ�¯ Overall Statistics:")
print(f"   â€¢ Total matched pairs: {len(matches_dict)}")
print(f"   â€¢ Evaluated pairs: {len(match_stats)}")
print(f"   â€¢ Total match points: {len(epipolar_errors)}")

if len(epipolar_errors) > 0:
    print(f"\nğŸ“� Epipolar Error Analysis:")
    print(f"   â€¢ Mean error: {np.mean(epipolar_errors):.6f}")
    print(f"   â€¢ Median error: {np.median(epipolar_errors):.6f}")
    print(f"   â€¢ Std dev: {np.std(epipolar_errors):.6f}")
    print(f"   â€¢ Min error: {np.min(epipolar_errors):.6f}")
    print(f"   â€¢ Max error: {np.max(epipolar_errors):.6f}")
    
    # Inlier rates at different thresholds
    thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    print(f"\nâœ… Inlier Rates (% matches below threshold):")
    for thresh in thresholds:
        inlier_pct = 100 * np.mean(epipolar_errors < thresh)
        print(f"   â€¢ Error < {thresh:.0e}: {inlier_pct:.1f}%")
    
    # Per-pair statistics
    print(f"\nğŸ“‹ Top 10 Best Matched Pairs (by inlier ratio):")
    match_stats_sorted = sorted(match_stats, key=lambda x: x['inlier_ratio'], reverse=True)
    for i, stats in enumerate(match_stats_sorted[:10], 1):
        print(f"\n   {i}. {stats['pair']}")
        print(f"      Matches: {stats['num_matches']:4d} | Inliers: {stats['inlier_ratio']*100:5.1f}% | "
              f"Mean error: {stats['mean_error']:.6f} | Conf: {stats['mean_confidence']:.3f}")

print("\n" + "="*70)

reconstruction = None

del scene_labels, match_stats, epipolar_errors
gc.collect()

