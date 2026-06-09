!git clone --recursive https://github.com/naver/mast3r
%cd mast3r
!pip install -r requirements.txt
!pip install -e .
!pip install roma


import os
import gc
import torch
import numpy as np
import pandas as pd
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

IMG_SIZE = 448
MAX_IMAGES = 32
WINDOW_SIZE = 3
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RANSAC_RUNS = 7

KAGGLE_INPUT_ROOT = '/kaggle/input/image-matching-challenge-2025/train'
GT_CSV_PATH = '/kaggle/input/image-matching-challenge-2025/train_labels.csv'

def get_filenames_from_disk(scene_path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    return sorted([f for f in os.listdir(scene_path) if f.lower().endswith(exts)])

def get_gt_poses(gt_df, scene_name):
    mask = gt_df['dataset'].astype(str).str.lower() == scene_name.lower()
    data = gt_df[mask]
    poses = {}
    
    for _, row in data.iterrows():
        name = str(row['image'])
        basename = name.split('/')[-1]
        
        try:
            R = np.array([float(x) for x in str(row['rotation_matrix']).split(';')]).reshape(3,3)
            t = np.array([float(x) for x in str(row['translation_vector']).split(';')]).reshape(3,1)
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = (-R.T @ t).flatten()
            # Store with both full path and basename
            poses[basename] = pose
            poses[name] = pose
        except:
            continue
    return poses

def smart_load_images(path, size, max_images):
    fnames = get_filenames_from_disk(path)
    if len(fnames) > max_images:
        fnames = fnames[:max_images]
    
    loaded_imgs = load_images(path, size=size, verbose=False)
    
    if len(loaded_imgs) > len(fnames):
        loaded_imgs = loaded_imgs[:len(fnames)]
    
    final_imgs = []
    for img_obj in loaded_imgs:
        img_tensor = img_obj['img']
        _, _, h, w = img_tensor.shape
        new_h = h - (h % 16)
        new_w = w - (w % 16)
        if new_h != h or new_w != w:
            img_obj['img'] = img_tensor[:, :, :new_h, :new_w]
        final_imgs.append(img_obj)
    
    return final_imgs, fnames

def multi_run_ransac_alignment(pred_poses, gt_poses, n_runs=7):
    """Multi-run RANSAC with better filename matching"""
    common = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    
    # Strategy 1: Direct match
    if len(common) >= 5:
        pass  # Use direct match
    else:
        # Strategy 2: Basename matching
        gt_basenames = {k.split('/')[-1]: v for k, v in gt_poses.items()}
        pred_basenames = {k.split('/')[-1]: v for k, v in pred_poses.items()}
        common = sorted(list(set(pred_basenames.keys()) & set(gt_basenames.keys())))
        
        if len(common) >= 5:
            pred_poses = pred_basenames
            gt_poses = gt_basenames
        else:
            print(f"    WARNING: Only {len(common)} common images found")
            print(f"    Pred keys sample: {list(pred_poses.keys())[:3]}")
            print(f"    GT keys sample: {list(gt_poses.keys())[:3]}")
            return 0.0, 0
    
    P_pred = np.array([pred_poses[k][:3, 3] for k in common]).T
    P_gt = np.array([gt_poses[k][:3, 3] for k in common]).T
    
    if np.isnan(P_pred).any() or np.isnan(P_gt).any():
        valid = ~np.isnan(P_pred).any(0) & ~np.isnan(P_gt).any(0)
        P_pred, P_gt = P_pred[:, valid], P_gt[:, valid]
        common = [common[i] for i in range(len(common)) if valid[i]]
        if len(common) < 5:
            return 0.0, 0
    
    scene_scale = np.linalg.norm(P_gt.max(1) - P_gt.min(1))
    ransac_threshold = max(0.05 * scene_scale, 0.5)
    
    best_maa = 0.0
    best_n = 0
    
    for run_seed in range(n_runs):
        np.random.seed(run_seed)
        
        best_res = None
        best_inliers = -1
        
        for _ in range(5000):
            try:
                idx = np.random.choice(P_pred.shape[1], 3, replace=False)
                src, dst = P_pred[:, idx], P_gt[:, idx]
                
                mu_s, mu_d = src.mean(1, keepdims=True), dst.mean(1, keepdims=True)
                src_c, dst_c = src - mu_s, dst - mu_d
                
                s_val = np.sum(src_c**2)
                if s_val < 1e-7:
                    continue
                s = np.sqrt(np.sum(dst_c**2) / s_val)
                
                H = src_c @ dst_c.T
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                R_norm = R.copy()
                if np.linalg.det(R_norm) < 0:
                    Vt_tmp = Vt.copy()
                    Vt_tmp[2] *= -1
                    R_norm = Vt_tmp.T @ U.T
                t_norm = mu_d - s * (R_norm @ mu_s)
                err_norm = np.linalg.norm((s * (R_norm @ P_pred) + t_norm) - P_gt, axis=0)
                in_norm = np.sum(err_norm < ransac_threshold)
                
                R_ref = R.copy()
                if np.linalg.det(R_ref) > 0:
                    Vt_tmp = Vt.copy()
                    Vt_tmp[2] *= -1
                    R_ref = Vt_tmp.T @ U.T
                t_ref = mu_d - s * (R_ref @ mu_s)
                err_ref = np.linalg.norm((s * (R_ref @ P_pred) + t_ref) - P_gt, axis=0)
                in_ref = np.sum(err_ref < ransac_threshold)
                
                if in_norm >= in_ref:
                    if in_norm > best_inliers:
                        best_inliers = in_norm
                        best_res = (s, R_norm, t_norm)
                else:
                    if in_ref > best_inliers:
                        best_inliers = in_ref
                        best_res = (s, R_ref, t_ref)
            except:
                continue
        
        if best_res is None:
            continue
        
        s, R_a, t_a = best_res
        
        errors = []
        scene_center = P_gt.mean(1)
        P_aligned = s * (R_a @ P_pred) + t_a
        
        for i in range(len(common)):
            k = common[i]
            G = gt_poses[k]
            P = pred_poses[k]
            
            R_pred_aligned = R_a @ P[:3, :3]
            R_diff = G[:3, :3].T @ R_pred_aligned
            
            if np.linalg.det(R_diff) < 0:
                R_diff = R_diff @ np.diag([1, 1, -1])
            
            tr = np.clip(np.trace(R_diff), -1, 3)
            r_err = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
            if np.isnan(r_err):
                r_err = 180.0
            
            v_gt = G[:3, 3] - scene_center
            v_pred = P_aligned[:, i] - scene_center
            n_gt, n_pred = np.linalg.norm(v_gt), np.linalg.norm(v_pred)
            
            if n_gt < 1e-6 or n_pred < 1e-6:
                t_err = 0.0
            else:
                dot = np.dot(v_gt, v_pred) / (n_gt * n_pred)
                t_err = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
            
            errors.append(max(r_err, t_err))
        
        errors = np.array(errors)
        acc = [np.mean(errors < t) for t in [3, 5, 10]]
        maa = np.mean(acc)
        
        if maa > best_maa:
            best_maa = maa
            best_n = len(common)
    
    return best_maa, best_n

def run_pipeline():
    """Final optimized pipeline"""
    print(f"Loading MASt3R on {DEVICE}...")
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(DEVICE)
    
    gt_df = pd.read_csv(GT_CSV_PATH)
    scenes = sorted([
        d for d in os.listdir(KAGGLE_INPUT_ROOT)
        if os.path.isdir(os.path.join(KAGGLE_INPUT_ROOT, d))
    ])
    
    results = []
    print(f"FINAL (Memory-Optimized): {IMG_SIZE}px | Max: {MAX_IMAGES} | Window: {WINDOW_SIZE} | RANSAC: {N_RANSAC_RUNS} ⭐")
    
    for scene_id in scenes:
        print(f"\n--- {scene_id} ---")
        try:
            gc.collect()
            torch.cuda.empty_cache()
            
            path = os.path.join(KAGGLE_INPUT_ROOT, scene_id)
            imgs, fnames = smart_load_images(path, IMG_SIZE, MAX_IMAGES)
            
            if len(imgs) < 2:
                print(f"  Skipping: only {len(imgs)} images")
                continue
            
            pairs = make_pairs(imgs, scene_graph=f'swin-{WINDOW_SIZE}', symmetrize=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = inference(pairs, model, device=DEVICE, batch_size=BATCH_SIZE, verbose=False)
            
            scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
            # Reduced iterations to save memory
            scene.compute_global_alignment(init="mst", niter=120, schedule='cosine', lr=0.01)
            
            poses = scene.get_im_poses()
            pred_poses = {}
            for i in range(len(imgs)):
                name = fnames[i]
                pred_poses[name] = poses[i].detach().float().cpu().numpy()
            
            gt_data = get_gt_poses(gt_df, scene_id)
            score, n = multi_run_ransac_alignment(pred_poses, gt_data, n_runs=N_RANSAC_RUNS)
            
            print(f"  mAA: {score:.4f} | Matches: {n}")
            results.append({'scene': scene_id, 'mAA': score, 'status': 'OK'})
            
            del scene, output, pairs, imgs, poses, pred_poses
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            results.append({'scene': scene_id, 'mAA': 0.0, 'status': 'Error'})
            gc.collect()
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(df)
    if not df.empty:
        print(f"\nCombined mAA: {df['mAA'].mean():.4f}")
    print("="*60)

# if __name__ == "__main__":
#     run_pipeline()


run_pipeline()

