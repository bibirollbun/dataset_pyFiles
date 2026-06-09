# MPEG-G Microbiome Classification Challenge
# Can you classify microbiome samples by body site and individual using compressed data?
# https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge

# BY ğŸ§¬_BYTE
# PUBLIC LEADERBOARD 0.004670137
# PRIVATE LEADERBOARD ---
# RANK 6


import time
start_time = time.time()


!pip install -q --upgrade pip
!pip install -q cupy-cuda11x cudf-cu11 dask-cudf-cu11 cuml-cu11 xgboost-cu11 \
    pylibraft-cu11 raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
!pip install -q Bio


!pip install scikit-posthocs


import os
import gc
import cupy as cp
import cudf
import xgboost as xgb
import numpy as np
from pathlib import Path
from cuml import LabelEncoder
from sklearn.model_selection import KFold
from cuml.metrics import log_loss
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor
from numba import cuda
from functools import partial
from itertools import product
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

import random
SEED = 87
def seed_everything(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)

 
seed_everything(SEED)

import warnings
warnings.filterwarnings('ignore')


# Check GPU Availability
assert cuda.is_available(), "â�Œ GPU not available!"
num_gpus = len(cuda.gpus)
print(f"ğŸŸ¢ {num_gpus} GPUs Detected: {[cuda.gpus[i].name.decode() for i in range(num_gpus)]}")


# Define motifs

MOTIFS = ['AGCG', 'TCCG', 'AATC', 'GTTG', 'CCTT', 'CAGT', 'ATCA', 'ATGG',
          'GCGG', 'GGCC', 'ACCA', 'CGCG', 'CCAG', 'TAAC', 'TCTA', 'GCGA',
          'TGCC', 'ATAA', 'GAAT', 'GTAA', 'TCTT', 'CCGT', 'GATT', 'GGCA',
          'CTAT', 'TCAT', 'GGAG', 'AGGA', 'ACTC', 'TGGA', 'CACG', 'TCGC',
          'GCTA', 'TAAT', 'AAAC', 'GACC', 'ATTG', 'GGCT', 'GGGG', 'TAAG',
          'ATCT', 'TCCT', 'GTAG', 'TCGG', 'AGAA', 'AACA', 'GCAC', 'TTGA',
          'CGTT', 'TCAG', 'GTTC', 'GCGT', 'CCCG', 'ACCT', 'GTCA', 'CCTA',
          'CGAA', 'AAGT', 'TGAC', 'TTCC', 'TAGC', 'ATTT', 'CACT', 'CGGC' ] 

# Reverse complement strands
def revcomp(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([comp[base] for base in seq[::-1]])

# Ensure deterministic order
expanded_motifs = sorted(set(MOTIFS + [revcomp(m) for m in MOTIFS]))  # Sort the list

print("Unique expanded motifs:", expanded_motifs)

def process_file_on_gpu(args, motifs=expanded_motifs):
    """
    Process FASTQ file on GPU: extract specified features and body site-associated motifs.
    
    Parameters:
    - args (tuple): (file_path, gpu_id) where file_path is a Path object and gpu_id is the GPU device ID.
    - motifs (list): List of motif sequences to search for.
    
    Returns:
    - dict: Features including num_reads, avg_read_len, gc_content, A, T, G, C, nt_entropy,
            and motif features (e.g., motif_TTAATT, motif_CAGCAT, etc.).
    """
    file_path, gpu_id = args
    try:
        cp.cuda.Device(gpu_id).use()
        with open(file_path, 'r') as file:
            lines = file.readlines()
        seqs = [lines[i].strip().upper() for i in range(1, len(lines), 4) if i < len(lines)]
        if not seqs:
            return None

        # Compute basic features
        lengths = cp.asarray([len(s) for s in seqs], dtype=cp.int32)
        gc_counts = cp.asarray([s.count('G') + s.count('C') for s in seqs], dtype=cp.int32)
        a_counts = cp.asarray([s.count('A') for s in seqs], dtype=cp.int32)
        t_counts = cp.asarray([s.count('T') for s in seqs], dtype=cp.int32)
        g_counts = cp.asarray([s.count('G') for s in seqs], dtype=cp.int32)
        c_counts = cp.asarray([s.count('C') for s in seqs], dtype=cp.int32)

        total_len = lengths.sum()
        num_reads = len(seqs)

        # Nucleotide diversity (Shannon entropy)
        nt_freqs = cp.array([a_counts.sum(), t_counts.sum(), g_counts.sum(), c_counts.sum()], dtype=cp.float32) / total_len
        nt_entropy = -cp.sum(nt_freqs * cp.log2(nt_freqs + cp.finfo(cp.float32).eps))

        # Motif counts
        motif_counts = {motif: 0 for motif in motifs}
        for seq in seqs:
            for motif in motifs:
                if motif in seq:
                    motif_counts[motif] += seq.count(motif)
        motif_features = {f"motif_{motif}": float(motif_counts[motif] / num_reads) if num_reads > 0 else 0
                         for motif in motifs}

        return {
            'filename': file_path.name,
            'num_reads': int(num_reads),
            'avg_read_len': float(lengths.mean()),
            'gc_content': float(gc_counts.sum() / total_len),
            'A': float(a_counts.sum() / total_len),
            'T': float(t_counts.sum() / total_len),
            'G': float(g_counts.sum() / total_len),
            'C': float(c_counts.sum() / total_len),
            'nt_entropy': float(nt_entropy),
            **motif_features
        }
    except Exception as e:
        print(f"Error on {file_path.name} (GPU {gpu_id}): {str(e)}")
        return None



def parallel_extract_features(path, num_workers=num_gpus):
    """Parallel feature extraction using available GPUs"""
    files = list(Path(path).glob('*.fastq'))
    print(f"ğŸ”� Found {len(files)} FASTQ files in {path}")

    # Create partial function to include k and motifs
    process_func = partial(process_file_on_gpu, motifs=expanded_motifs)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(f, i % num_gpus) for i, f in enumerate(files)]
        results = list(executor.map(process_func, args))

    features = [r for r in results if r is not None]
    return cudf.DataFrame(features)


# === MAIN EXECUTION ===
print("ğŸš€ Loading training data...")

train_dirs = [
    "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TrainFiles/TrainFiles",
    "/kaggle/input/secondbatchoffastqfiles/TrainFiles"
]

df_train = cudf.DataFrame()
for folder in train_dirs:
    df_part = parallel_extract_features(folder)
    df_train = cudf.concat([df_train, df_part], ignore_index=True)

train_labels = cudf.read_csv("/kaggle/input/extrafiles/Train.csv")
train_labels['fastq_filename'] = train_labels['filename'].str.replace('.mgb', '.fastq')
df_train = df_train.merge(
    train_labels[['fastq_filename', 'SampleType']],
    left_on='filename',
    right_on='fastq_filename'
)


# from scipy.stats import kruskal
# motif_p_values = {}
# for motif in expanded_motifs:
#     site_groups = [df_train[df_train['SampleType'].str.contains(site)][f'motif_{motif}'].to_pandas()
#                    for site in ['Nasal', 'Mouth', 'Stool', 'Skin']]
#     stat, p = kruskal(*site_groups)
#     motif_p_values[motif] = p
# print("Motif p-values:", sorted(motif_p_values.items(), key=lambda x: x[1]))


motif_p_values = {}
motif_pairwise = {}  # Store pairwise p-values

for motif in expanded_motifs:
    # Group data by sample type
    groups = {
        site: df_train[df_train['SampleType'].str.contains(site)][f'motif_{motif}'].to_pandas()
        for site in ['Nasal', 'Mouth', 'Stool', 'Skin']
    }
    group_data = list(groups.values())
    
    # Kruskal-Wallis test
    stat, p = kruskal(*group_data)
    motif_p_values[motif] = p
    
    # Dunn's test (if KW p < 0.05)
    if p < 0.05:
        dunn_results = posthoc_dunn(group_data, p_adjust='bonferroni')
        motif_pairwise[motif] = dunn_results

# Print top significant motifs with pairwise comparisons
top_motifs = sorted(motif_p_values.items(), key=lambda x: x[1])[:10]
for motif, p in top_motifs:
    print(f"\nMotif: {motif}, Kruskal p = {p:.3e}")
    if p < 0.05 and motif in motif_pairwise:
        print("Pairwise differences (Dunn's test):")
        print(motif_pairwise[motif])


for motif, p in top_motifs:
    print(f"\nMotif: {motif}")
    for site in ['Nasal', 'Mouth', 'Stool', 'Skin']:
        data = df_train[df_train['SampleType'].str.contains(site)][f'motif_{motif}']
        print(f"{site}: Median = {data.median():.2f}, Mean = {data.mean():.2f}")


# Encode labels
le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['SampleType'])


# Define feature columns
motif_features = [f'motif_{motif}' for motif in expanded_motifs]
feature_columns = ['num_reads', 'avg_read_len', 'gc_content', 'A', 'T', 'G', 'C', 
                   'nt_entropy'] + motif_features

# Features + Labels
X = df_train[feature_columns]
y = df_train['label']


# XGBoost params
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': len(le.classes_),
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'learning_rate': 0.1,
    'max_depth': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 70
}

# K-Fold Training
print("ğŸ”¥ Preparing folds...")
kfold = KFold(n_splits=10, shuffle=True, random_state=SEED)
oof_preds = cp.zeros((len(X), len(le.classes_)), dtype=cp.float32)
models = []


for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\nğŸŒ€ Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params, dtrain,
        num_boost_round=7000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    models.append(model)
    oof_preds[val_idx] = model.predict(dval)

    # Free memory
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


# Log Loss
oof_loss = log_loss(
    y.astype(np.int32).to_numpy(),
    oof_preds.astype(np.float32).get()
)
print(f"\nğŸ“Š OOF Log Loss: {oof_loss:.5f}")

# === TEST PROCESSING ===
print("\nâš¡ Extracting test features...")
test_dir = "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TestFiles/TestFiles"
df_test = parallel_extract_features(test_dir)
X_test = df_test[feature_columns]


print("\nğŸ”® Predicting test set...")
test_preds = cp.zeros((len(X_test), len(le.classes_)), dtype=cp.float32)
for model in models:
    dtest = xgb.DMatrix(X_test)
    test_preds += cp.array(model.predict(dtest), dtype=cp.float32)

test_preds /= len(models)


submission = cudf.DataFrame({
    'filename': df_test['filename'].str.replace('.fastq', ''),
    'Mouth': test_preds[:, le.transform(['Mouth'])[0]].get(),
    'Nasal': test_preds[:, le.transform(['Nasal'])[0]].get(),
    'Skin': test_preds[:, le.transform(['Skin'])[0]].get(),
    'Stool': test_preds[:, le.transform(['Stool'])[0]].get()
})

submission = submission[['filename', 'Mouth', 'Nasal', 'Skin', 'Stool']]
submission.to_csv('submission__32.csv', index=False)
print("\nâœ… Submission saved to `submission.csv`")


end_time = time.time()
elapsed = end_time - start_time

mins, secs = divmod(elapsed, 60)
print(f"Notebook runtime: {int(mins)} min {secs:.2f} sec")

