!pip install -q --upgrade pip
!pip install -q cupy-cuda11x cudf-cu11 dask-cudf-cu11 cuml-cu11 xgboost-cu11 \
    pylibraft-cu11 raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
!pip install -q Bio

import os
import cupy as cp
import cudf
import xgboost as xgb
import numpy as np
from pathlib import Path
from collections import Counter
from cuml import LabelEncoder
from sklearn.model_selection import KFold
from cuml.metrics import log_loss
from Bio import SeqIO
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

# Check GPU Availability
from numba import cuda
assert cuda.is_available(), "â�Œ GPU not available!"
num_gpus = len(cuda.gpus)
print(f"ğŸŸ¢ {num_gpus} GPUs Detected: {[cuda.gpus[i].name.decode() for i in range(num_gpus)]}")

def process_file_on_gpu(args):
    """Process a single file on GPU using thread-safe approach"""
    file_path, gpu_id = args
    try:
        # Set the GPU device for this thread
        cp.cuda.Device(gpu_id).use()
        
        # Load data
        with open(file_path, 'rb') as file:
            data = cp.asarray(bytearray(file.read()))
        
        # Process FASTQ data
        seq_data = data.tobytes().decode('utf-8').split('\n')
        seqs = [seq_data[i] for i in range(1, len(seq_data), 4) if i < len(seq_data)]
        
        if not seqs:
            return None
            
        # GPU-accelerated calculations
        lengths = cp.array([len(s) for s in seqs])
        gc_counts = cp.array([s.count('G') + s.count('C') for s in seqs])
        a_counts = cp.array([s.count('A') for s in seqs])
        t_counts = cp.array([s.count('T') for s in seqs])
        g_counts = cp.array([s.count('G') for s in seqs])
        c_counts = cp.array([s.count('C') for s in seqs])
        
        total_len = lengths.sum()
        return {
            'filename': file_path.name,
            'num_reads': len(lengths),
            'avg_read_len': float(lengths.mean()),
            'gc_content': float(gc_counts.sum() / total_len),
            'A': float(a_counts.sum() / total_len),
            'T': float(t_counts.sum() / total_len),
            'G': float(g_counts.sum() / total_len),
            'C': float(c_counts.sum() / total_len)
        }
    except Exception as e:
        print(f"Error processing {file_path.name} on GPU {gpu_id}: {str(e)}")
        return None

def parallel_extract_features(path, num_workers=num_gpus):
    """Parallel feature extraction using all GPUs with thread-safe approach"""
    files = list(Path(path).glob('*.fastq'))
    print(f"ğŸ”� Found {len(files)} files to process")
    
    # Process files in parallel with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Assign files round-robin to GPUs
        results = list(executor.map(
            process_file_on_gpu, 
            [(f, i % num_gpus) for i, f in enumerate(files)]
        ))
    
    # Filter out None results and convert to DataFrame
    features = [f for f in results if f is not None]
    return cudf.DataFrame(features)

# Main execution
if __name__ == '__main__':
    # Load and process training data in parallel
    print("ğŸš€ Loading and processing training data across all GPUs...")
    train_dirs = [
        "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TrainFiles/TrainFiles",
        "/kaggle/input/secondbatchoffastqfiles/TrainFiles"
    ]

    # Process each directory in sequence (but files within in parallel)
    df_train = cudf.DataFrame()
    for folder in train_dirs:
        df_part = parallel_extract_features(folder)
        df_train = cudf.concat([df_train, df_part], ignore_index=True)

    # Merge with labels from original CSV
    train_labels = cudf.read_csv("/kaggle/input/extrafiles/Train.csv")
    train_labels['fastq_filename'] = train_labels['filename'].str.replace('.mgb', '.fastq')
    df_train = df_train.merge(
        train_labels[['fastq_filename', 'SampleType']],
        left_on='filename',
        right_on='fastq_filename'
    )

    # Encode labels on GPU
    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['SampleType'])

    # Prepare GPU Data
    X = df_train[['num_reads', 'avg_read_len', 'gc_content', 'A', 'T', 'G', 'C']]
    y = df_train['label']

    # XGBoost parameters for multi-class classification
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(le.classes_),
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }

    # KFold training with GPU distribution
    print("ğŸ”¥ Starting XGBoost training...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cp.zeros((len(X), len(le.classes_)))
    models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\nğŸŒ€ Processing Fold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        models.append(model)
        oof_preds[val_idx] = model.predict(dval)

# Calculate OOF metrics with proper type conversion
oof_loss = log_loss(
    y.astype(np.int32).to_numpy(),  # Convert labels to int32
    oof_preds.astype(np.float32).get()  # Convert predictions to float32
)
print(f"\nğŸ“Š Out-of-Fold Log Loss: {oof_loss:.5f}")

# Process test data in parallel
print("\nâš¡ Processing test data across GPUs...")
test_dir = "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TestFiles/TestFiles"
df_test = parallel_extract_features(test_dir)
X_test = df_test[['num_reads', 'avg_read_len', 'gc_content', 'A', 'T', 'G', 'C']]



# Predict on test set using all models
print("\nğŸ”® Generating test predictions...")
test_preds = cp.zeros((len(X_test), len(le.classes_)), dtype=cp.float32)  # Use cp.float32 instead of np.float32

for model in models:
    dtest = xgb.DMatrix(X_test)
    # Convert predictions to CuPy array before addition
    preds = cp.array(model.predict(dtest), dtype=cp.float32)
    test_preds += preds

test_preds /= len(models)  # Average predictions

# Prepare submission with proper type conversion
submission = cudf.DataFrame({
    'filename': df_test['filename'].str.replace('.fastq', ''),
    'Mouth': test_preds[:, le.transform(['Mouth'])[0]].get(),
    'Nasal': test_preds[:, le.transform(['Nasal'])[0]].get(),
    'Skin': test_preds[:, le.transform(['Skin'])[0]].get(),
    'Stool': test_preds[:, le.transform(['Stool'])[0]].get()
})

# Ensure correct column order
submission = submission[['filename', 'Mouth', 'Nasal', 'Skin', 'Stool']]

# Save results
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission saved!")












# # Predict on test set using all models
# print("\nğŸ”® Generating test predictions...")
# test_preds = cp.zeros((len(X_test), len(le.classes_)), dtype=cp.float32)  # Use cp.float32 instead of np.float32

# for model in models:
#     dtest = xgb.DMatrix(X_test)
#     # Convert predictions to CuPy array before addition
#     preds = cp.array(model.predict(dtest), dtype=cp.float32)
#     test_preds += preds

# test_preds /= len(models)  # Average predictions

# # Prepare submission with proper type conversion
# submission = cudf.DataFrame({
#     'filename': df_test['filename'].str.replace('.fastq', ''),
#     'Mouth': test_preds[:, le.transform(['Mouth'])[0]].get(),
#     'Nasal': test_preds[:, le.transform(['Nasal'])[0]].get(),
#     'Skin': test_preds[:, le.transform(['Skin'])[0]].get(),
#     'Stool': test_preds[:, le.transform(['Stool'])[0]].get()
# })

# # Ensure correct column order
# submission = submission[['filename', 'Mouth', 'Nasal', 'Skin', 'Stool']]

# # Save results
# submission.to_csv('submission.csv', index=False)
# print("\nâœ… Submission saved!")







