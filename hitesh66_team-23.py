# HMRF Probabilistic Model applied on PANDA Kaggle dataset

# Importing all the necessary Python libraries
import os, sys, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import openslide
import cv2
import mahotas
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, cohen_kappa_score
from scipy.stats import entropy
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

os.system('apt-get update -qq')
os.system('apt-get install -y -qq openslide-tools > /dev/null')
os.system('pip install -q openslide-python scikit-image mahotas tqdm > /dev/null')

# Loading the dataset
data_path = Path("/kaggle/input/prostate-cancer-grade-assessment")
train_csv = data_path / "train.csv"

# Handling if the dataset is missing 
if not train_csv.exists():
    raise FileNotFoundError(f"train.csv not found at {train_csv}. Make sure dataset is mounted to /kaggle/input/...")

train_df = pd.read_csv(train_csv)
print(f"Loaded {len(train_df)} training samples")
print("ISUP grade distribution:")
print(train_df['isup_grade'].value_counts().sort_index())


def safe_array(x):
    return np.nan_to_num(np.array(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

EPS = 1e-10

#Patch Extraction from the slides
def open_slide(path):
    return openslide.OpenSlide(str(path))

# Function to find the tissue regions from the slides
    def find_tissue_regions(slide, thumb_size=2048, min_tissue_area=0.01):
    thumb = np.array(slide.get_thumbnail((thumb_size, thumb_size)))[:,:,:3]
    gray = rgb2gray(thumb)
    gray = gaussian(gray, sigma=2)
    try:
        thresh = threshold_otsu(gray)
        mask = gray < thresh
    except Exception:
        mask = gray < 0.9
    from skimage.measure import label, regionprops
    labeled = label(mask)
    regions = regionprops(labeled)
    W, H = slide.dimensions
    scale_x, scale_y = W / thumb_size, H / thumb_size
    tissue_coords = []
    total_area = thumb_size * thumb_size
    for region in regions:
        if region.area / total_area > min_tissue_area:
            cy, cx = region.centroid
            full_x = int(cx * scale_x)
            full_y = int(cy * scale_y)
            tissue_coords.append((full_x, full_y))
    if not tissue_coords:
        tissue_coords = [(W//2, H//2)]
    return tissue_coords


# Finction to extract adaptive patches
def extract_adaptive_patches(slide, tissue_coords, n_patches=25, patch_size=256):
    W, H = slide.dimensions
    patches_list = []
    coords_list = []
    grid_size = int(np.sqrt(n_patches))
    for center_x, center_y in tissue_coords[:3]:
        stride = 128
        radius = grid_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = center_x + dx * stride
                y = center_y + dy * stride
                if x < 0 or y < 0 or x + patch_size > W or y + patch_size > H:
                    continue
                try:
                    region = slide.read_region((x, y), 0, (patch_size, patch_size))
                    patch = np.array(region.convert('RGB'))
                except Exception:
                    continue
                if np.mean(patch) < 240:
                    patches_list.append(patch)
                    coords_list.append((x, y))
    return patches_list, coords_list

# Feature Extraction
from skimage.filters import gaussian as sk_gaussian

# Function to extract the enhanced features
def extract_enhanced_features(patch):
    features = {}
    def safe_stat(arr, stat_func, default=0.0):
        try:
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return default
            val = stat_func(arr)
            if np.isnan(val) or np.isinf(val):
                return default
            return float(val)
        except Exception:
            return default

    for i, channel in enumerate(['r','g','b']):
        ch = patch[:,:,i].astype(float)
        features[f'{channel}_mean'] = safe_stat(ch, np.mean, 128.0)
        features[f'{channel}_std']  = safe_stat(ch, np.std, 10.0)
        features[f'{channel}_median'] = safe_stat(ch, np.median, 128.0)
        features[f'{channel}_q25'] = safe_stat(ch, lambda x: np.percentile(x,25), 100.0)
        features[f'{channel}_q75'] = safe_stat(ch, lambda x: np.percentile(x,75), 150.0)

    try:
        hsv = rgb2hsv(patch)
        for i, channel in enumerate(['h','s','v']):
            ch = hsv[:,:,i]
            features[f'{channel}_mean'] = safe_stat(ch, np.mean, 0.5)
            features[f'{channel}_std'] = safe_stat(ch, np.std, 0.1)
    except Exception:
        for channel in ['h','s','v']:
            features[f'{channel}_mean'] = 0.5
            features[f'{channel}_std']  = 0.1

    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    except:
        gray = np.mean(patch, axis=2).astype(np.uint8)

    try:
        haralick_feats = mahotas.features.haralick(gray).mean(axis=0)
        for i, val in enumerate(haralick_feats[:10]):
            features[f'haralick_{i}'] = safe_stat(val, lambda x: x, 0.0)
    except:
        for i in range(10):
            features[f'haralick_{i}'] = 0.0

    try:
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + EPS)
        for i, val in enumerate(lbp_hist[:10]):
            features[f'lbp_{i}'] = safe_stat(val, lambda x: x, 0.1)
    except:
        for i in range(10):
            features[f'lbp_{i}'] = 0.1

    # Edges
    try:
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = safe_stat(edges/255.0, np.mean, 0.1)
    except:
        features['edge_density'] = 0.1

    # Focus measure
    try:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['focus_lap'] = safe_stat(lap_var, lambda x: x, 100.0)
    except:
        features['focus_lap'] = 100.0

    # Tissue density
    try:
        tissue_mask = np.mean(patch, axis=2) < 240
        features['tissue_density'] = safe_stat(tissue_mask, np.mean, 0.5)
    except:
        features['tissue_density'] = 0.5

    # Color contrast
    try:
        features['color_contrast'] = safe_stat(patch, lambda x: np.std(x, axis=(0,1)).mean(), 20.0)
    except:
        features['color_contrast'] = 20.0

    # Spatial gradient
    try:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features['grad_mean'] = safe_stat(grad_mag, np.mean, 10.0)
        features['grad_std'] = safe_stat(grad_mag, np.std, 5.0)
    except:
        features['grad_mean'] = 10.0
        features['grad_std'] = 5.0

    return features


# Function to build the feature matrix
def build_feature_matrix(patches):
    feature_list = []
    for patch in tqdm(patches, desc="Extracting features"):
        features = extract_enhanced_features(patch)
        feature_list.append(features)
    df = pd.DataFrame(feature_list)

    # Handling NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        col_mean = df[col].mean()
        if np.isnan(col_mean) or np.isinf(col_mean):
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(col_mean)

    df = df.fillna(0.0)

    X_raw = df.values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    if X_raw.shape[0] == 0:
        return np.zeros((0,0)), [], scaler
    try:
        X = scaler.fit_transform(X_raw)
    except Exception:
        X = X_raw - np.mean(X_raw, axis=0, keepdims=True)
        X = np.nan_to_num(X, nan=0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df.columns.tolist(), scaler

# Building a spatial graph
def build_spatial_graph(coords, patch_size=512, connectivity=4):
    N = len(coords)
    neighbors = [[] for _ in range(N)]
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    for i, (x, y) in enumerate(coords):
        candidates = [
            (x + patch_size, y),
            (x - patch_size, y),
            (x, y + patch_size),
            (x, y - patch_size)
        ]
        if connectivity == 8:
            candidates += [
                (x + patch_size, y + patch_size),
                (x + patch_size, y - patch_size),
                (x - patch_size, y + patch_size),
                (x - patch_size, y - patch_size)
            ]
        for coord in candidates:
            j = coord_to_idx.get(coord, None)
            if j is not None and j != i:
                neighbors[i].append(j)
    return neighbors

# Initialization of Gaussian Log Likelihood
def initialize_gmm(X, K, method='kmeans'):
    N, D = X.shape
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mus = np.zeros((K, D), dtype=float)
    Sigmas = np.zeros((K, D, D), dtype=float)

    if method == 'kmeans' and N >= K:
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) > 1:
                mu = X[idx].mean(axis=0)
                mu = np.nan_to_num(mu, nan=0.0)
                mus[k] = mu
                cov = np.cov(X[idx].T)
                if cov.ndim == 0:
                    cov = np.eye(D) * max(float(cov), 1e-6)
                cov = np.nan_to_num(cov, nan=0.0)
                cov = cov + 1e-4 * np.eye(D)
                Sigmas[k] = cov
            elif len(idx) == 1:
                mus[k] = np.nan_to_num(X[idx[0]], nan=0.0)
                Sigmas[k] = np.eye(D) * 1e-3
            else:
                mus[k] = np.zeros(D)
                Sigmas[k] = np.eye(D)
    else:
        rng = np.random.RandomState(42)
        if N >= K:
            indices = rng.choice(N, K, replace=False)
        else:
            indices = rng.choice(N, K, replace=True)
        base_cov = np.cov(X.T) if N > 1 else np.eye(X.shape[1])
        base_cov = np.nan_to_num(base_cov, nan=0.0)
        base_cov = base_cov + 1e-4 * np.eye(D)
        for k, idx in enumerate(indices):
            mus[k] = np.nan_to_num(X[idx], nan=0.0)
            Sigmas[k] = base_cov.copy()

    pi = np.ones(K) / K
    return mus, Sigmas, pi


def gaussian_log_likelihood(x, mu, Sigma, min_loglik=-1e10):
    x = safe_array(x)
    mu = safe_array(mu)
    Sigma = np.array(Sigma, dtype=float)
    if np.any(np.isnan(x)) or np.any(np.isnan(mu)) or np.any(np.isnan(Sigma)):
        return min_loglik
    D = x.shape[0]
    Sigma = Sigma + 1e-6 * np.eye(D)
    try:
        # Use slogdet for determinant and solve for Mahalanobis
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0 or np.isnan(logdet) or np.isinf(logdet):
            Sigma = Sigma + 1e-3 * np.eye(D)
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return min_loglik
        diff = x - mu
        sol = np.linalg.solve(Sigma, diff)
        mahalanobis = float(diff @ sol)
        if np.isnan(mahalanobis) or np.isinf(mahalanobis):
            return min_loglik
        loglik = -0.5 * (mahalanobis + logdet + D * np.log(2 * np.pi))
        loglik = np.maximum(loglik, min_loglik)
        return float(loglik)
    except Exception:
        return min_loglik


# Creating a robust HMRF Model
def hmrf_em_icm(X, neighbors, K=6, beta=1.0, max_iter=30, tol=1e-4, verbose=True):
    N, D = X.shape
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if N == 0:
        return np.array([], dtype=int), np.zeros((0, K)), np.zeros((K, D)), np.zeros((K, D, D))
    mus, Sigmas, pi = initialize_gmm(X, K, method='kmeans')
    labels = np.random.randint(0, K, size=N)
    prev_energy = -np.inf

    for it in range(max_iter):
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) > 1:
                mu = X[idx].mean(axis=0)
                mu = np.nan_to_num(mu, nan=0.0)
                mus[k] = mu
                cov = np.cov(X[idx].T)
                if cov.ndim == 0:
                    cov = np.eye(D) * max(float(cov), 1e-6)
                cov = np.nan_to_num(cov, nan=0.0)
                cov = cov + 1e-4 * np.eye(D)
                Sigmas[k] = cov
            elif len(idx) == 1:
                mus[k] = np.nan_to_num(X[idx[0]], nan=0.0)
                Sigmas[k] = np.eye(D) * 1e-3
            else:
                Sigmas[k] = Sigmas[k] + 1e-4 * np.eye(D)
        changed = 0
        energy = 0.0
        order = np.random.permutation(N)
        for i in order:
            best_label = labels[i]
            best_score = -np.inf
            for k in range(K):
                data_ll = gaussian_log_likelihood(X[i], mus[k], Sigmas[k])
                spatial_prior = 0
                for j in neighbors[i]:
                    if labels[j] == k:
                        spatial_prior += 1
                score = data_ll + beta * spatial_prior
                if score > best_score:
                    best_score = score
                    best_label = k
            if labels[i] != best_label:
                labels[i] = best_label
                changed += 1
            energy += best_score

        if verbose:
            print(f"Iter {it+1}/{max_iter}: changed={changed}, energy={energy:.4f}")
        if abs(energy - prev_energy) < tol * (abs(prev_energy) + 1e-10):
            if verbose:
                print(f"Converged at iter {it+1}")
            break
        prev_energy = energy
        if changed == 0:
            if verbose:
                print("No label changes, stopping early")
            break

    log_post = np.full((N, K), -1e12, dtype=float)
    for k in range(K):
        for i in range(N):
            log_post[i, k] = gaussian_log_likelihood(X[i], mus[k], Sigmas[k], min_loglik=-1e12)

    max_ll = np.max(log_post, axis=1, keepdims=True)
    max_ll = np.nan_to_num(max_ll, nan=0.0, posinf=0.0, neginf=0.0)
    exp_ll = np.exp(log_post - max_ll)
    exp_ll = np.nan_to_num(exp_ll, nan=0.0, posinf=0.0, neginf=0.0)

    row_sums = exp_ll.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.squeeze() == 0)
    if np.any(zero_rows):
        exp_ll[zero_rows, :] = 1.0 / K
        row_sums = exp_ll.sum(axis=1, keepdims=True)

    posteriors = exp_ll / (row_sums + EPS)
    posteriors = np.nan_to_num(posteriors, nan=1.0/K, posinf=1.0/K, neginf=0.0)

    return labels, posteriors, mus, Sigmas

# Function to generate the aggregation metrics
def aggregate_patch_posteriors(posteriors, method='weighted_mean', weights=None):
    posteriors = np.nan_to_num(posteriors, nan=0.0)
    row_sums = posteriors.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    posteriors = posteriors / row_sums

    K = posteriors.shape[1]
    if method == 'mean':
        result = posteriors.mean(axis=0)
    elif method == 'weighted_mean':
        if weights is None:
            weights = posteriors.max(axis=1)
        weights = np.nan_to_num(weights, nan=1.0)
        weight_sum = weights.sum()
        if weight_sum == 0:
            weight_sum = 1.0
        weights = weights / weight_sum
        result = (posteriors.T @ weights)
    elif method == 'max':
        votes = posteriors.argmax(axis=1)
        result = np.bincount(votes, minlength=K)
        result = result / (result.sum() + EPS)
    else:
        result = posteriors.mean(axis=0)

    result = np.nan_to_num(result, nan=1.0/K)
    result = result / (result.sum() + EPS)
    return result

def compute_metrics(y_true, y_pred_probs, verbose=True):
    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.nan_to_num(y_pred_probs, nan=0.0)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    y_pred_probs = y_pred_probs / row_sums

    K = y_pred_probs.shape[1]
    y_pred_map = y_pred_probs.argmax(axis=1)
    valid_mask = (y_true >= 0) & (y_true < K)
    y_true = y_true[valid_mask]
    y_pred_probs = y_pred_probs[valid_mask]
    y_pred_map = y_pred_map[valid_mask]

    if len(y_true) == 0:
        if verbose:
            print("No valid samples for evaluation")
        return None

    acc = accuracy_score(y_true, y_pred_map)
    qwk = cohen_kappa_score(y_true, y_pred_map, weights='quadratic')
    epsilon = 1e-12
    y_pred_probs_clipped = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    y_pred_probs_clipped = y_pred_probs_clipped / y_pred_probs_clipped.sum(axis=1, keepdims=True)
    try:
        ll = log_loss(y_true, y_pred_probs_clipped, labels=list(range(K)))
    except ValueError as e:
        print("Warning in log_loss:", e)
        probs_uniform = np.ones_like(y_pred_probs_clipped) / K
        ll = log_loss(y_true, probs_uniform, labels=list(range(K)))

    N = len(y_true)
    onehot = np.zeros((N, K))
    onehot[np.arange(N), y_true] = 1
    brier = np.mean(np.sum((y_pred_probs - onehot)**2, axis=1))
    cm = confusion_matrix(y_true, y_pred_map, labels=list(range(K)))

    # Print all the evaluation metrics
    if verbose:
        print(f"Samples evaluated: {len(y_true)}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Quadratic Weighted Kappa: {qwk:.4f}")
        print(f"Log Loss: {ll:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")

    return {
        'accuracy': acc,
        'qwk': qwk,
        'log_loss': ll,
        'brier': brier,
        'confusion_matrix': cm,
        'n_samples': len(y_true)
    }

# Process all the slides
def process_slide(slide_id, data_path, n_patches=49, patch_size=512):
    slide_path = data_path / "train_images" / f"{slide_id}.tiff"
    if not slide_path.exists():
        print(f"Slide {slide_id} not found at {slide_path}")
        return None
    try:
        slide = open_slide(slide_path)
        tissue_coords = find_tissue_regions(slide)
        patches, coords = extract_adaptive_patches(slide, tissue_coords, n_patches=n_patches, patch_size=patch_size)
        if len(patches) < 3:
            slide.close()
            return None
        X, feature_names, scaler = build_feature_matrix(patches)
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)
        neighbors = build_spatial_graph(coords, patch_size=patch_size)
        slide.close()
        return {
            'X': X,
            'coords': coords,
            'neighbors': neighbors,
            'n_patches': len(patches),
            'scaler': scaler
        }
    except Exception as e:
        print(f"Error processing {slide_id}: {e}")
        return None

# Running the main HMRF Pipeline
def run_hmrf_pipeline(n_slides=50, K=6, beta=1.0, aggregation='weighted_mean', verbose=True):
    print("\n" + "="*60)
    print(f"Running HMRF Pipeline: K={K}, beta={beta}, n_slides={n_slides}")
    print("="*60 + "\n")
    results = []
    skipped = 0
    y_true_list = []
    y_pred_probs_list = []

    max_idx = min(n_slides, len(train_df))
    for idx in tqdm(range(max_idx), desc="Processing slides"):
        row = train_df.iloc[idx]
        slide_id = row['image_id']
        true_grade = int(row['isup_grade'])
        if true_grade < 0 or true_grade >= K:
            skipped += 1
            continue
        slide_data = process_slide(slide_id, data_path)
        if slide_data is None:
            skipped += 1
            continue
        try:
            labels, posteriors, mus, Sigmas = hmrf_em_icm(
                slide_data['X'], slide_data['neighbors'], K=K, beta=beta, max_iter=20, verbose=False)

            posteriors = np.nan_to_num(posteriors, nan=0.0)
            row_sums = posteriors.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            posteriors = posteriors / row_sums

            if posteriors.size == 0:
                print(f"Warning: empty posteriors for {slide_id}, skipping")
                skipped += 1
                continue

            slide_posterior = aggregate_patch_posteriors(posteriors, method=aggregation)
            if np.any(np.isnan(slide_posterior)) or slide_posterior.sum() == 0:
                slide_posterior = np.ones(K) / K

            map_grade = int(np.argmax(slide_posterior))
            expected_grade = float(np.sum(np.arange(K) * slide_posterior))
            uncertainty = float(entropy(slide_posterior))

            results.append({
                'slide_id': slide_id,
                'true_grade': true_grade,
                'posterior': slide_posterior,
                'map_grade': map_grade,
                'expected_grade': expected_grade,
                'uncertainty': uncertainty,
                'n_patches': slide_data['n_patches']
            })

            # for metrics
            y_true_list.append(true_grade)
            y_pred_probs_list.append(slide_posterior)

        except Exception as e:
            print(f"Error in HMRF for {slide_id}: {e}")
            skipped += 1
            continue

    if len(y_true_list) > 0:
        y_pred_probs_arr = np.vstack(y_pred_probs_list)
        metrics = compute_metrics(y_true_list, y_pred_probs_arr, verbose=verbose)
    else:
        metrics = None

    print(f"\nProcessed: {len(results)}, Skipped: {skipped}")
    return results, metrics

# Performing the grid search
def grid_search(n_slides=30, Ks=[4,5,6], betas=[0.5,1.0,1.5], aggregation='weighted_mean'):
    best_qwk = -np.inf
    best_params = None
    grid_results = []
    for K in Ks:
        for beta in betas:
            print(f"\n--- Grid try K={K}, beta={beta} ---")
            results, metrics = run_hmrf_pipeline(n_slides=n_slides, K=K, beta=beta, aggregation=aggregation, verbose=False)
            qwk = metrics['qwk'] if metrics is not None else -np.inf
            grid_results.append({'K':K, 'beta':beta, 'qwk':qwk, 'metrics': metrics})
            print(f"Result: qwk={qwk:.4f}")
            if qwk > best_qwk:
                best_qwk = qwk
                best_params = {'K':K, 'beta':beta}
    print(f"\nBest params: {best_params}, best_qwk={best_qwk:.4f}")
    return best_params, grid_results

# Running an example
if __name__ == "__main__":
    best_params, grid_results = grid_search(
        n_slides=100,            
        Ks=[4,5,6,7],            
        betas=[0.6,0.8,1.0,1.2], 
        aggregation='weighted_mean'
    )
    print("Quick grid best:", best_params)

    
    results, metrics = run_hmrf_pipeline(
        n_slides=1000,                 
        K=best_params['K'],
        beta=best_params['beta'],
        aggregation='weighted_mean',
        verbose=True
    )
    print("\nSummary metrics:")
    print(metrics)

