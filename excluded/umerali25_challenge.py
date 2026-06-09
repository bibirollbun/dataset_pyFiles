# --- CELL 1: THE "FIX EVERYTHING" INSTALLER ---
# 1. Force NumPy < 2 (Critical for hloc/pycolmap)
# 2. Install LightGlue (Critical for ALIKED)
# 3. Install Kornia/Transformers (Critical for the pipeline)
!pip install "numpy<2" --force-reinstall git+https://github.com/cvg/LightGlue.git kornia pycolmap transformers h5py

# Clone repositories if not already present
!git clone --recursive https://github.com/cvg/Hierarchical-Localization/
!git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

print("âœ… Installation Complete.")
print("âš ï¸� CRITICAL: You MUST restart the session now for the NumPy downgrade to apply.")
print("Go to: 'Run' (or 'Session') -> 'Restart Session' (or 'Restart Kernel').")


import os
os._exit(00)


import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import shutil
import warnings

# --- 1. CONFIGURATION & SETUP ---
warnings.filterwarnings("ignore")

# Setup Paths
current_dir = Path.cwd()
if 'Hierarchical-Localization' not in sys.path:
    sys.path.append(str(current_dir / 'Hierarchical-Localization'))
if 'SuperGluePretrainedNetwork' not in sys.path:
    sys.path.append(str(current_dir / 'SuperGluePretrainedNetwork'))

# Manual LightGlue import fix
try:
    import lightglue
except ImportError:
    import site
    sys.path.append(site.getsitepackages()[0])

import pycolmap
from hloc import extract_features, match_features, reconstruction
from hloc.utils.read_write_model import read_model

# Kaggle Directories
KAGGLE_INPUT_DIR = Path('/kaggle/input/image-matching-challenge-2025')
# POINT TO TRAIN DIRECTORY TO RUN ALL TRAIN DATASETS
TRAIN_TEST_DIR = KAGGLE_INPUT_DIR / 'train' 
OUTPUT_DIR = Path('/kaggle/working/outputs')

# Clean Start
if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Models
CONF_EXTRACT = {
    'model': {
        'name': 'aliked',
        'model_name': 'aliked-n16',
        'max_num_keypoints': 4096,
        'detection_threshold': 0.01,
        'nms_radius': 2,
    },
    'output': 'feats-aliked',
    'preprocessing': {'resize_max': 2048, 'grayscale': False},
}
CONF_MATCH = match_features.confs['aliked+lightglue']

# --- 2. HELPER CLASSES & FUNCTIONS ---

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class AdaptivePairSelector:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"   [Selector] Loading DINOv2 on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_feat(self, img):
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img_t)
        return F.normalize(feat, p=2, dim=1)

    def correct_rotations(self, image_paths):
        # Rotation Logic (Only run if > 10 images)
        if len(image_paths) < 10: return
        
        # Consensus
        ref_feats = []
        limit = min(5, len(image_paths))
        for p in image_paths[:limit]:
            img = Image.open(p).convert('RGB')
            ref_feats.append(self.extract_feat(img))
        scene_center = torch.mean(torch.cat(ref_feats), dim=0, keepdim=True)
        
        count = 0
        for p in tqdm(image_paths, desc="   [Selector] Checking Rotation"):
            img_org = Image.open(p).convert('RGB')
            best_score, best_rot, best_img = -1, 0, img_org
            
            for rot in [0, 90, 180, 270]:
                if rot == 0: i_rot = img_org
                elif rot == 90: i_rot = img_org.rotate(90, expand=True)
                elif rot == 180: i_rot = img_org.rotate(180, expand=True)
                elif rot == 270: i_rot = img_org.rotate(270, expand=True)
                
                feat = self.extract_feat(i_rot)
                score = torch.mm(feat, scene_center.t()).item()
                if score > best_score:
                    best_score, best_rot, best_img = score, rot, i_rot
            
            if best_rot != 0:
                best_img.save(p) # Overwrite in temp dir
                count += 1
        print(f"   [Selector] Fixed {count} rotated images.")

    def get_pairs(self, image_paths, dataset_name):
        # Run Rotation Check first
        self.correct_rotations(image_paths)
        
        num_images = len(image_paths)
        
        # LOGIC: Force Exhaustive for high accuracy if under 300 images
        if num_images < 300:
            print(f"   [Selector] Using EXHAUSTIVE matching ({num_images} images)")
            pairs = []
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    pairs.append((image_paths[i], image_paths[j]))
            return pairs

        # Strategy 2: DINOv2 for > 300
        print(f"   [Selector] Using DINOv2 filtering ({num_images} images)")
        features = []
        bs = 16
        with torch.no_grad():
            for i in range(0, len(image_paths), bs):
                batch = []
                for p in image_paths[i:i+bs]:
                    img = Image.open(p).convert('RGB')
                    batch.append(self.transform(img))
                batch_t = torch.stack(batch).to(self.device)
                feat = self.model(batch_t)
                features.append(F.normalize(feat, p=2, dim=1).cpu())
        
        all_feats = torch.cat(features)
        sim = torch.mm(all_feats, all_feats.t())
        
        pairs = []
        for i in range(num_images):
            scores = sim[i]
            valid = torch.where(scores > 0.15)[0]
            valid = valid[valid != i]
            if len(valid) > 0:
                topk = torch.topk(scores[valid], min(50, len(valid))).indices
                real_idx = valid[topk].tolist()
                for j in real_idx:
                    if i < j: pairs.append((image_paths[i], image_paths[j]))
                    elif j < i: pairs.append((image_paths[j], image_paths[i]))
        return sorted(list(set(pairs)))

def process_dataset(original_dataset_path, output_dir, pair_selector):
    dataset_name = original_dataset_path.name
    print(f"\nğŸš€ Processing {dataset_name}...")
    
    # 1. Setup Temp Directory (Writable)
    temp_dir = Path(f'/kaggle/working/temp_{dataset_name}')
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    shutil.copytree(original_dataset_path, temp_dir)
    
    images = sorted([p for p in temp_dir.glob('**/*') if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}])
    if len(images) < 3: return False

    # 2. Setup Outputs
    ds_out = output_dir / dataset_name
    ds_out.mkdir(exist_ok=True)
    pairs_path = ds_out / 'pairs.txt'
    feats_path = ds_out / 'feats.h5'
    matches_path = ds_out / 'matches.h5'
    
    # Clean zombies
    if feats_path.exists(): 
        if feats_path.is_dir(): shutil.rmtree(feats_path)
        else: feats_path.unlink()
    if matches_path.exists():
        if matches_path.is_dir(): shutil.rmtree(matches_path)
        else: matches_path.unlink()
    
    # 3. Generate Pairs
    # Pass paths to temp images
    raw_pairs = pair_selector.get_pairs(images, dataset_name)
    
    with open(pairs_path, 'w') as f:
        for p1, p2 in raw_pairs:
            rel1 = p1.relative_to(temp_dir).as_posix()
            rel2 = p2.relative_to(temp_dir).as_posix()
            f.write(f"{rel1} {rel2}\n")

    # 4. Run HLOC
    img_list = [p.relative_to(temp_dir).as_posix() for p in images]
    
    extract_features.main(CONF_EXTRACT, temp_dir, feature_path=feats_path, image_list=img_list)
    match_features.main(CONF_MATCH, pairs=pairs_path, features=feats_path, matches=matches_path)
    
    model_path = ds_out / 'colmap'
    model = reconstruction.main(model_path, temp_dir, pairs=pairs_path, features=feats_path, matches=matches_path, camera_mode='AUTO', verbose=False)
    
    # Cleanup to save space
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    
    if model:
        print(f"âœ… {dataset_name}: Reconstruction successful.")
        return True
    else:
        print(f"âš ï¸� {dataset_name}: Failed to reconstruct.")
        return False

def merge_and_save_csv():
    print("\nğŸ› ï¸� STARTING MULTI-MODEL MERGE & SAVE...")
    submission_data = {}
    solved_count = 0
    
    # Scan all outputs
    for dataset_dir in OUTPUT_DIR.iterdir():
        if not dataset_dir.is_dir(): continue
        dataset_name = dataset_dir.name
        
        # Find ALL binary models (handling fragmentation)
        model_files = list(dataset_dir.glob('**/images.bin'))
        
        if model_files:
            print(f"   ğŸ“‚ {dataset_name}: Found {len(model_files)} model fragments.")
            
            for m_file in model_files:
                try:
                    # Safe Binary Read
                    _, images, _ = read_model(m_file.parent, ext='.bin')
                    for img_id, img in images.items():
                        if (dataset_name, img.name) in submission_data: continue
                        
                        # Pose Math
                        R_mat = qvec2rotmat(img.qvec)
                        tvec = img.tvec
                        
                        submission_data[(dataset_name, img.name)] = (
                            ";".join(map(str, R_mat.flatten())),
                            ";".join(map(str, tvec.flatten()))
                        )
                        solved_count += 1
                except: pass
    
    # Write CSV
    sample_path = KAGGLE_INPUT_DIR / 'sample_submission.csv'
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        out_rot, out_trans = [], []
        for idx, row in df.iterrows():
            key = (row['dataset'], row['image'])
            if key in submission_data:
                r, t = submission_data[key]
                out_rot.append(r)
                out_trans.append(t)
            else:
                out_rot.append("1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0")
                out_trans.append("0.0;0.0;0.0")
        df['rotation_matrix'] = out_rot
        df['translation_vector'] = out_trans
        df.to_csv('submission.csv', index=False)
        print(f"ğŸ�‰ FINAL SUBMISSION SAVED! Total Solved Images: {solved_count}")
    else:
        print("No sample submission found (Dry Run). Saving raw matches.")
        # Optional: Save raw df if no sample exists
        # pd.DataFrame(...).to_csv()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"ğŸ“‚ Source Directory: {TRAIN_TEST_DIR}")
    
    # Initialize DINO
    dino = AdaptivePairSelector()
    
    # --- GET ALL DATASETS AUTOMATICALLY ---
    all_datasets = sorted([d for d in TRAIN_TEST_DIR.iterdir() if d.is_dir()])
    print(f"ğŸš€ Starting Sequential Execution on ALL {len(all_datasets)} datasets...")
    
    for dataset_folder in all_datasets:
        ds_name = dataset_folder.name
        
        # Skip 'stairs' because it takes forever and guarantees 0% score
        if 'stairs' in ds_name:
            print(f"â�­ï¸� Skipping {ds_name} (Known failure case/High computation)")
            continue
            
        try:
            process_dataset(dataset_folder, OUTPUT_DIR, dino)
        except Exception as e:
            print(f"â�Œ Critical Error on {ds_name}: {e}")
            
    # Final Step: Merge and Save
    merge_and_save_csv()


# THE "MULTI-MODEL MERGER" FIX (train accuracy 5 MODEL)---
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go

# Setup HLOC path
current_dir = Path.cwd()
hloc_path = current_dir / 'Hierarchical-Localization'
if str(hloc_path) not in sys.path:
    sys.path.append(str(hloc_path))

# Import Pure Python Reader
from hloc.utils.read_write_model import read_model

# Config
OUTPUT_DIR = Path('/kaggle/working/outputs')
KAGGLE_INPUT_DIR = Path('/kaggle/input/image-matching-challenge-2025') 

print("ğŸ› ï¸� STARTING MULTI-MODEL MERGING...")

submission_data = {}
debug_counts = {}

# 1. LOOP THROUGH DATASETS
for dataset_dir in OUTPUT_DIR.iterdir():
    if not dataset_dir.is_dir(): continue
    dataset_name = dataset_dir.name
    
    print(f"\nğŸ“‚ Scanning {dataset_name}...")
    
    # 2. FIND ALL SUB-MODELS (0, 1, 2...)
    # COLMAP saves splits as separate folders inside 'colmap'
    # We look recursively for ANY file named 'images.bin'
    model_files = list(dataset_dir.glob('**/images.bin'))
    
    if not model_files:
        print(f"   âš ï¸� No models found for {dataset_name}")
        continue
        
    print(f"   âœ… Found {len(model_files)} fragmented models! Merging...")
    
    total_imgs_for_ds = 0
    
    # 3. MERGE THEM
    for m_file in model_files:
        try:
            # Read the model
            _, images, points3D = read_model(m_file.parent, ext='.bin')
            
            # Count images in this fragment
            n_fragment = len(images)
            if n_fragment == 0: continue
            
            print(f"      - Merging fragment from '{m_file.parent.name}': {n_fragment} images")
            
            # Extract poses
            for img_id, img in images.items():
                # If image already exists (from a larger model), skip it
                # We prioritize the first model we see (usually the largest 0)
                if (dataset_name, img.name) in submission_data:
                    continue
                
                # Pose Math
                qvec = img.qvec
                tvec = img.tvec
                w, x, y, z = qvec
                R_mat = np.array([
                    [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                    [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                    [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
                ])
                
                submission_data[(dataset_name, img.name)] = (
                    ";".join(map(str, R_mat.flatten())),
                    ";".join(map(str, tvec.flatten()))
                )
                total_imgs_for_ds += 1
                
        except Exception as e:
            print(f"      â�Œ Error reading {m_file}: {e}")
            
    debug_counts[dataset_name] = total_imgs_for_ds
    print(f"   ğŸ‘‰ Total merged count for {dataset_name}: {total_imgs_for_ds}")

# --- 4. SAVE SUBMISSION CSV ---
sample_path = KAGGLE_INPUT_DIR / 'sample_submission.csv'
if sample_path.exists():
    df = pd.read_csv(sample_path)
    out_rot, out_trans = [], []
    filled_count = 0
    
    for idx, row in df.iterrows():
        key = (row['dataset'], row['image'])
        if key in submission_data:
            r, t = submission_data[key]
            out_rot.append(r)
            out_trans.append(t)
            filled_count += 1
        else:
            # Outlier / Failed
            out_rot.append("1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0")
            out_trans.append("0.0;0.0;0.0")
    
    df['rotation_matrix'] = out_rot
    df['translation_vector'] = out_trans
    df.to_csv('submission.csv', index=False)
    
    print("\n" + "="*40)
    print(f"ğŸ�‰ FINAL RESULT: Solved {filled_count} images!")
    print("="*40)
    
    # Recalculate your specific score
    for ds, count in debug_counts.items():
        total_in_ds = len(df[df['dataset'] == ds])
        if total_in_ds > 0:
            rate = (count / total_in_ds) * 100
            print(f"Dataset: {ds}")
            print(f"  Accuracy: {rate:.2f}% ( {count} / {total_in_ds} )")
            if rate > 60: print("  Verdict: â­� EXCELLENT")
            elif rate > 30: print("  Verdict: âœ… IMPROVED")
            else: print("  Verdict: âš ï¸� STILL LOW")
            print("-" * 20)

else:
    print("No sample submission found.")

# --- 5. VISUALIZATION (ALL MERGED MODELS) ---
if debug_counts:
    print("\nGeneratng Visualization for ALL datasets...")
    
    # Loop through all datasets that have results
    for ds_name in debug_counts.keys():
        print(f"\nVisualizing {ds_name}...")
        search_dir = OUTPUT_DIR / ds_name
        
        # Find the largest sub-model for this dataset to plot
        candidates = list(search_dir.glob('**/images.bin'))
        # Sort by file size (rough proxy for model size)
        candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        if candidates:
            # We visualize the largest fragment
            try:
                _, _, points3D = read_model(candidates[0].parent, ext='.bin')
                pts, cols = [], []
                for p3d in points3D.values():
                    pts.append(p3d.xyz)
                    cols.append(p3d.rgb)
                
                if len(pts) > 0:
                    pts = np.array(pts)
                    cols = np.array(cols) / 255.0
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers', marker=dict(size=1.5, color=cols),
                        name='Scene'
                    ))
                    fig.update_layout(title=f"Reconstruction: {ds_name}", height=600, template='plotly_dark')
                    fig.show()
                else:
                    print(f"   âš ï¸� Model found but has no 3D points.")
            except Exception as e:
                print(f"   â�Œ Error visualizing {ds_name}: {e}")




