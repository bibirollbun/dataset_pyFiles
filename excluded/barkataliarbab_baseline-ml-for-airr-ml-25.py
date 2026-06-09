# ==============================================================
# AIRR-ML-25 — DEBUGGED & IMPROVED NOTEBOOK
# ==============================================================

import gc, warnings
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif

import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ========================= CONFIG ===========================
class Config:
    DATA_ROOT = Path("/kaggle/input/adaptive-immune-profiling-challenge-2025/")
    TRAIN_DIR = DATA_ROOT / "train_datasets" / "train_datasets"
    TEST_DIR  = DATA_ROOT / "test_datasets" / "test_datasets"
    SAMPLE_SUB = DATA_ROOT / "sample_submissions.csv"

    # Feature extraction
    K_LIST = [2, 3, 4, 5]  # Added more k-mer sizes
    MAX_SEQS = 10000       # Smaller but more focused
    TOP_FEATURES = 2000    # Much larger to capture diversity
    MIN_DF = 3             # Minimum samples a feature must appear in
    N_SPLITS = 3           # Reduced for speed but still effective
    EARLY_STOP = 50
    RANDOM_STATE = 42
    
    # XGBoost parameters
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "random_state": RANDOM_STATE
    }
    
    # Class weights
    CLASS_WEIGHTS = {
        1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5,
        7: 3.0, 8: 2.0
    }

OUTPUT_NAME = "submission.csv"

# ====================== AA PROPERTIES =======================
AA = {
    "A": (1.8, 0, 89.09, 1),   # hydro, charge, volume, polarity
    "R": (-4.5, 1, 174.20, 1),
    "N": (-3.5, 0, 132.12, 1),
    "D": (-3.5, -1, 133.10, 1),
    "C": (2.5, 0, 121.15, 1),
    "Q": (-3.5, 0, 146.15, 1),
    "E": (-3.5, -1, 147.13, 1),
    "G": (-0.4, 0, 75.07, 1),
    "H": (-3.2, 0.5, 155.16, 1),
    "I": (4.5, 0, 131.17, 0),
    "L": (3.8, 0, 131.17, 0),
    "K": (-3.9, 1, 146.19, 1),
    "M": (1.9, 0, 149.21, 0),
    "F": (2.8, 0, 165.19, 0),
    "P": (-1.6, 0, 115.13, 1),
    "S": (-0.8, 0, 105.09, 1),
    "T": (-0.7, 0, 119.12, 1),
    "W": (-0.9, 0, 204.23, 0),
    "Y": (-1.3, 0, 181.19, 1),
    "V": (4.2, 0, 117.15, 0)
}

# ====================== DEBUGGED HELPERS ====================
def dataset_id(name: str) -> int:
    """Extract dataset ID from filename."""
    for t in name.replace("_", " ").split():
        if t.isdigit():
            return int(t)
    return 1

def read_rep_debug(path, max_n):
    """Debug version to see what's in the files."""
    try:
        # First, try to read the file
        df = pd.read_csv(path, sep="\t")
        
        print(f"    File: {path.name}, Columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        if "junction_aa" not in df.columns:
            print(f"    WARNING: No 'junction_aa' column in {path.name}")
            print(f"    Available columns: {df.columns.tolist()}")
            
            # Try to find alternative column names
            aa_cols = [c for c in df.columns if 'aa' in c.lower() or 'junction' in c.lower()]
            if aa_cols:
                print(f"    Possible AA columns: {aa_cols}")
                # Use the first alternative
                df = df.rename(columns={aa_cols[0]: "junction_aa"})
        
        # Use only needed columns
        if "templates" in df.columns:
            df = df[["junction_aa", "templates"]]
        else:
            df = df[["junction_aa"]]
            df["templates"] = 1.0
        
    except Exception as e:
        print(f"    ERROR reading {path.name}: {str(e)}")
        return pd.DataFrame(columns=["junction_aa", "templates"])
    
    # Handle empty data
    if len(df) == 0:
        print(f"    WARNING: Empty file {path.name}")
        return pd.DataFrame(columns=["junction_aa", "templates"])
    
    # Clean and sample
    df["junction_aa"] = df["junction_aa"].fillna("").astype(str)
    
    # Filter out invalid sequences
    valid_mask = df["junction_aa"].str.len() >= 5
    df = df[valid_mask]
    
    if len(df) == 0:
        print(f"    WARNING: No valid sequences in {path.name}")
        return pd.DataFrame(columns=["junction_aa", "templates"])
    
    # Sample if needed
    if max_n and len(df) > max_n:
        # Use template counts as weights for sampling
        if "templates" in df.columns:
            w = pd.to_numeric(df["templates"], errors="coerce").fillna(1.0)
            w = w / w.sum()
            if w.sum() > 0:
                idx = np.random.choice(len(df), min(max_n, len(df)), replace=False, p=w)
            else:
                idx = np.random.choice(len(df), min(max_n, len(df)), replace=False)
        else:
            idx = np.random.choice(len(df), min(max_n, len(df)), replace=False)
        df = df.iloc[idx]
    
    print(f"    Loaded {len(df)} sequences from {path.name}")
    return df

def read_rep(path, max_n):
    """Production version - optimized."""
    try:
        # Try to read with different column requirements
        try:
            df = pd.read_csv(path, sep="\t", usecols=["junction_aa", "templates"])
        except:
            # Try reading all columns and then selecting
            df = pd.read_csv(path, sep="\t")
            
            # Find AA column
            aa_cols = [c for c in df.columns if any(x in c.lower() for x in ['aa', 'junction', 'cdr3'])]
            if not aa_cols:
                return pd.DataFrame(columns=["junction_aa", "templates"])
            
            aa_col = aa_cols[0]
            
            # Find template/count column
            count_cols = [c for c in df.columns if any(x in c.lower() for x in ['template', 'count', 'frequency'])]
            if count_cols:
                count_col = count_cols[0]
                df = df[[aa_col, count_col]]
                df.columns = ["junction_aa", "templates"]
            else:
                df = df[[aa_col]]
                df.columns = ["junction_aa"]
                df["templates"] = 1.0
        
        # Clean data
        df["junction_aa"] = df["junction_aa"].fillna("").astype(str)
        
        # Filter invalid sequences
        mask = df["junction_aa"].str.len() >= 5
        df = df[mask]
        
        if len(df) == 0:
            return pd.DataFrame(columns=["junction_aa", "templates"])
        
        # Sample if needed
        if max_n and len(df) > max_n:
            if "templates" in df.columns:
                w = pd.to_numeric(df["templates"], errors="coerce").fillna(1.0)
                w = w / max(w.sum(), 1)
                if w.sum() > 0 and not np.any(np.isnan(w)):
                    idx = np.random.choice(len(df), max_n, replace=False, p=w)
                else:
                    idx = np.random.choice(len(df), max_n, replace=False)
            else:
                idx = np.random.choice(len(df), max_n, replace=False)
            df = df.iloc[idx]
        
        return df
        
    except Exception:
        return pd.DataFrame(columns=["junction_aa", "templates"])

# ====================== ROBUST FEATURE EXTRACTION ============
class RobustFeatureExtractor:
    def __init__(self, k_list):
        self.k_list = k_list
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    
    def extract_sequence_features(self, seqs):
        """Extract features from a list of sequences."""
        feats = {}
        
        if not seqs:
            return {"bias": 1.0, "num_seqs": 0}
        
        # Basic statistics
        feats["num_seqs"] = len(seqs)
        lengths = [len(s) for s in seqs]
        feats["len_mean"] = np.mean(lengths)
        feats["len_std"] = np.std(lengths)
        feats["len_min"] = np.min(lengths)
        feats["len_max"] = np.max(lengths)
        
        # Amino acid composition
        all_aas = "".join(seqs)
        if all_aas:
            aa_counts = Counter(all_aas)
            total_aas = len(all_aas)
            
            # Individual AA frequencies
            for aa in self.aa_list:
                count = aa_counts.get(aa, 0)
                feats[f"aa_{aa}"] = count / total_aas
            
            # AA group frequencies
            hydrophobic = "AVILMFYW"
            hydrophilic = "RNDQEHKST"
            charged = "DEKRH"
            feats["pct_hydrophobic"] = sum(aa_counts.get(aa, 0) for aa in hydrophobic) / total_aas
            feats["pct_hydrophilic"] = sum(aa_counts.get(aa, 0) for aa in hydrophilic) / total_aas
            feats["pct_charged"] = sum(aa_counts.get(aa, 0) for aa in charged) / total_aas
        
        # k-mer features
        for k in self.k_list:
            kmer_counts = Counter()
            total_kmers = 0
            
            for s in seqs:
                if len(s) >= k:
                    for i in range(len(s) - k + 1):
                        kmer = s[i:i+k]
                        if all(aa in self.aa_list for aa in kmer):
                            kmer_counts[kmer] += 1
                            total_kmers += 1
            
            # Add top kmers
            if total_kmers > 0:
                top_kmers = kmer_counts.most_common(50)  # Limit to top 50 per k
                for kmer, count in top_kmers:
                    feats[f"k{k}_{kmer}"] = count / total_kmers
        
        # Position-specific features (first and last 5 positions)
        if seqs:
            max_len = min(5, min(lengths))
            for pos in range(max_len):
                # First positions
                aas_at_pos = [s[pos] for s in seqs if len(s) > pos]
                if aas_at_pos:
                    counter = Counter(aas_at_pos)
                    total = len(aas_at_pos)
                    for aa in self.aa_list[:5]:  # Top 5 AAs
                        feats[f"first{pos}_{aa}"] = counter.get(aa, 0) / total
                
                # Last positions
                aas_at_end = [s[-pos-1] for s in seqs if len(s) > pos]
                if aas_at_end:
                    counter = Counter(aas_at_end)
                    total = len(aas_at_end)
                    for aa in self.aa_list[:5]:  # Top 5 AAs
                        feats[f"last{pos}_{aa}"] = counter.get(aa, 0) / total
        
        # Physicochemical properties
        hydro_scores, charge_scores, volume_scores = [], [], []
        for s in seqs:
            h_sum, c_sum, v_sum, n = 0, 0, 0, 0
            for aa in s:
                if aa in AA:
                    h, c, v, _ = AA[aa]
                    h_sum += h
                    c_sum += c
                    v_sum += v
                    n += 1
            if n > 0:
                hydro_scores.append(h_sum / n)
                charge_scores.append(c_sum / n)
                volume_scores.append(v_sum / n)
        
        if hydro_scores:
            feats["hydro_mean"] = np.mean(hydro_scores)
            feats["hydro_std"] = np.std(hydro_scores)
            feats["charge_mean"] = np.mean(charge_scores)
            feats["charge_std"] = np.std(charge_scores)
            feats["volume_mean"] = np.mean(volume_scores)
        
        # Sequence diversity
        if len(seqs) > 1:
            unique_seqs = len(set(seqs))
            feats["seq_diversity"] = unique_seqs / len(seqs)
        
        return feats
    
    def extract(self, df):
        """Main extraction function."""
        if df.empty:
            return {"bias": 1.0, "num_seqs": 0}
        
        seqs = [s for s in df.junction_aa if isinstance(s, str) and len(s) >= 5]
        
        if not seqs:
            return {"bias": 1.0, "num_seqs": 0}
        
        feats = self.extract_sequence_features(seqs)
        feats["bias"] = 1.0  # Always include bias
        
        return feats

# ====================== SIMPLE FEATURE SELECTION ============
def simple_feature_selection(X, y, topk):
    """Simple but effective feature selection."""
    if X.shape[1] <= topk:
        return list(X.columns)
    
    # Method 1: Variance threshold
    variances = X.var()
    high_var_features = variances.nlargest(min(topk * 2, len(variances))).index.tolist()
    X_filtered = X[high_var_features]
    
    # Method 2: Correlation with target (for binary)
    if len(X_filtered) > 0:
        try:
            # Simple univariate feature selection
            correlations = []
            for col in X_filtered.columns:
                if X_filtered[col].var() > 1e-6:
                    corr = np.abs(np.corrcoef(X_filtered[col], y)[0, 1])
                    correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in correlations[:topk]]
            return selected
        except:
            pass
    
    # Fallback: variance only
    return variances.nlargest(topk).index.tolist()

# ====================== MAIN TRAINING =======================
print("=" * 60)
print("STARTING TRAINING WITH DEBUGGED FEATURE EXTRACTION")
print("=" * 60)

# First, let's debug one dataset to see what's happening
print("\nDEBUGGING DATASET 1:")
test_ds = Config.TRAIN_DIR / "train_dataset_1"
test_meta = pd.read_csv(test_ds / "metadata.csv")
print(f"Metadata shape: {test_meta.shape}")
print(f"Columns: {test_meta.columns.tolist()}")
print(f"First few files: {test_meta.filename.head(3).tolist()}")

# Test reading one file
test_file = test_ds / test_meta.filename.iloc[0]
print(f"\nTesting file: {test_file}")
test_df = read_rep_debug(test_file, Config.MAX_SEQS)

# Initialize extractor
extractor = RobustFeatureExtractor(Config.K_LIST)

# Test feature extraction on the sample
print("\nTesting feature extraction on sample file:")
sample_feats = extractor.extract(test_df)
print(f"Number of features extracted: {len(sample_feats)}")
print(f"First 10 features: {list(sample_feats.keys())[:10]}")

# Now train all datasets
print("\n" + "=" * 60)
print("TRAINING ALL DATASETS")
print("=" * 60)

bundles = {}

for ds in sorted(Config.TRAIN_DIR.glob("train_dataset_*")):
    dsid = dataset_id(ds.name)
    print(f"\n{'='*40}")
    print(f"TRAINING DATASET {dsid}")
    print(f"{'='*40}")
    
    # Load metadata
    meta = pd.read_csv(ds / "metadata.csv")
    print(f"Samples: {len(meta)}")
    print(f"Positive labels: {meta.label_positive.sum()} ({meta.label_positive.mean():.1%})")
    
    # Extract features
    print("Extracting features...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 50
    all_rows = []
    
    for i in range(0, len(meta), batch_size):
        batch = meta.iloc[i:i+batch_size]
        batch_rows = Parallel(4, prefer="threads")(
            delayed(lambda r: {
                **extractor.extract(read_rep(ds / r.filename, Config.MAX_SEQS)),
                "y": int(r.label_positive)
            })(r)
            for _, r in batch.iterrows()
        )
        all_rows.extend(batch_rows)
        
        if i == 0:
            # Show sample features from first batch
            print(f"  Sample row has {len(batch_rows[0])} features")
    
    # Create feature matrix
    df = pd.DataFrame(all_rows).fillna(0)
    y = df.pop("y").values
    X = df
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features with variance > 0: {(X.var() > 0).sum()}")
    
    if X.shape[1] <= 1:
        print("WARNING: Very few features extracted!")
        print("Adding random features as fallback...")
        # Add some random (but structured) features
        for i in range(100):
            X[f"rand_feat_{i}"] = np.random.randn(len(X)) * 0.1
    
    # Basic feature filtering
    nonzero_features = (X.var() > 1e-6)
    if nonzero_features.sum() > 0:
        X = X.loc[:, nonzero_features]
    
    print(f"After variance filtering: {X.shape}")
    
    # Simple feature selection
    if X.shape[1] > Config.TOP_FEATURES:
        print("Selecting top features...")
        selected_feats = simple_feature_selection(X, y, Config.TOP_FEATURES)
        X = X[selected_feats]
        print(f"Selected {len(selected_feats)} features")
    else:
        selected_feats = list(X.columns)
    
    # Train model
    print(f"\nTraining XGBoost on {X.shape[1]} features...")
    
    # Adjust scale_pos_weight based on class imbalance
    pos_ratio = y.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / max(pos_ratio, 1e-6))
    scale_pos_weight = min(scale_pos_weight, 10.0)  # Cap at 10
    
    params = Config.XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    
    # Cross-validation
    kf = StratifiedKFold(Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
    aucs, models = [], []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold+1}/{Config.N_SPLITS}")
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        
        evals_result = {}
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=Config.EARLY_STOP,
            verbose_eval=False,
            evals_result=evals_result
        )
        
        preds = bst.predict(dvalid)
        auc = roc_auc_score(y_va, preds)
        aucs.append(auc)
        models.append(bst)
        
        print(f"    AUC: {auc:.4f}")
    
    print(f"\nDataset {dsid} CV Results:")
    print(f"  AUCs: {[f'{a:.4f}' for a in aucs]}")
    print(f"  Mean: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # Store for prediction
    bundles[ds.name] = (models, selected_feats, np.mean(aucs))
    
    # Cleanup
    del df, X
    gc.collect()

# ====================== PREDICTION =========================
print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

sub = pd.read_csv(Config.SAMPLE_SUB)

for td in sorted(Config.TEST_DIR.glob("test_dataset_*")):
    dsid = dataset_id(td.name)
    print(f"\nProcessing test dataset {dsid}...")
    
    # Find matching model
    train_key = f"train_dataset_{dsid}"
    if train_key not in bundles:
        print(f"  No model found for dataset {dsid}, using first available")
        train_key = list(bundles.keys())[0]
    
    models, feats, cv_score = bundles[train_key]
    print(f"  Using model from {train_key} (CV AUC: {cv_score:.4f})")
    
    # Process test files
    test_files = list(td.glob("*.tsv"))
    print(f"  Found {len(test_files)} test files")
    
    # Extract features in batches
    all_preds = []
    batch_size = 100
    
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        
        batch_rows = []
        for f in batch_files:
            df = read_rep(f, Config.MAX_SEQS)
            features = extractor.extract(df)
            features["ID"] = f.stem
            batch_rows.append(features)
        
        X_batch = pd.DataFrame(batch_rows).fillna(0)
        
        # Add missing features
        for feat in feats:
            if feat not in X_batch.columns:
                X_batch[feat] = 0
        
        # Make predictions with ensemble
        if X_batch.shape[0] > 0:
            X_batch = X_batch[feats]  # Ensure correct order
            
            # Ensemble predictions from all CV models
            fold_preds = []
            for model in models:
                dtest = xgb.DMatrix(X_batch)
                pred = model.predict(dtest)
                fold_preds.append(pred)
            
            # Average predictions
            batch_pred = np.mean(fold_preds, axis=0)
            all_preds.extend(batch_pred)
    
    # Assign predictions
    mask = sub.dataset == td.name
    if len(all_preds) == mask.sum():
        sub.loc[mask, "label_positive_probability"] = all_preds
        print(f"  Assigned {len(all_preds)} predictions")
    else:
        print(f"  WARNING: Mismatch in prediction count!")
        # Use simple average as fallback
        sub.loc[mask, "label_positive_probability"] = 0.5
    
    # Print stats
    if mask.sum() > 0:
        preds = sub.loc[mask, "label_positive_probability"].values
        print(f"  Prediction stats: mean={preds.mean():.4f}, "
              f"min={preds.min():.4f}, max={preds.max():.4f}")

# Final processing
print("\n" + "=" * 60)
print("FINALIZING SUBMISSION")
print("=" * 60)

# Clip predictions to reasonable range
sub["label_positive_probability"] = sub["label_positive_probability"].clip(0.001, 0.999)

# Ensure no NaN values
if sub["label_positive_probability"].isna().any():
    print(f"WARNING: Found {sub['label_positive_probability'].isna().sum()} NaN predictions")
    sub["label_positive_probability"] = sub["label_positive_probability"].fillna(0.5)

print(f"\nSubmission stats:")
print(f"  Shape: {sub.shape}")
print(f"  Range: [{sub.label_positive_probability.min():.4f}, "
      f"{sub.label_positive_probability.max():.4f}]")
print(f"  Mean: {sub.label_positive_probability.mean():.4f}")

# Save
sub.to_csv(OUTPUT_NAME, index=False)
print(f"\nSaved to: {OUTPUT_NAME}")
print("=" * 60)

