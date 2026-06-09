# # âœ… Install required GPU-accelerated libraries
# !pip install -q --upgrade pip
# !pip install -q cupy-cuda11x cudf-cu11 dask-cudf-cu11 cuml-cu11 xgboost-cu11 \
#     pylibraft-cu11 raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
# !pip install -q Bio

# # âœ… Imports
# import os
# import cupy as cp
# import cudf
# import xgboost as xgb
# import numpy as np
# from pathlib import Path
# from cuml import LabelEncoder
# from sklearn.model_selection import KFold
# from cuml.metrics import log_loss
# import warnings
# from concurrent.futures import ThreadPoolExecutor
# from numba import cuda
# warnings.filterwarnings('ignore')

# # âœ… Check GPU availability
# assert cuda.is_available(), "â�Œ GPU not available!"
# num_gpus = len(cuda.gpus)
# print(f"ğŸŸ¢ {num_gpus} GPUs Detected: {[cuda.gpus[i].name.decode() for i in range(num_gpus)]}")

# # âœ… Helper: Encode sequence to numeric
# def encode_sequence_to_numeric(seq, max_len=300):
#     """Encode ACGTN -> 0-4 and pad/truncate to max_len"""
#     base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
#     encoded = cp.array([base_map.get(base, 4) for base in seq[:max_len]], dtype=cp.int8)
#     if len(encoded) < max_len:
#         padding = cp.full((max_len - len(encoded),), 4, dtype=cp.int8)
#         encoded = cp.concatenate((encoded, padding))
#     return encoded

# # âœ… Feature Extraction per File
# def process_file_on_gpu(args, max_seqs=100, seq_len=300):
#     file_path, gpu_id = args
#     try:
#         cp.cuda.Device(gpu_id).use()

#         with open(file_path, 'r') as file:
#             lines = file.read().split('\n')
#             seqs = [lines[i] for i in range(1, len(lines), 4) if lines[i]]

#         if not seqs:
#             return None

#         seqs = seqs[:max_seqs]
#         numeric_seqs = cp.stack([encode_sequence_to_numeric(seq, seq_len) for seq in seqs])
#         feature_vector = numeric_seqs.mean(axis=0).astype(cp.float32)

#         return {
#             'filename': file_path.name,
#             **{f'base_{i}': float(feature_vector[i]) for i in range(seq_len)}
#         }

#     except Exception as e:
#         print(f"â�Œ Error in {file_path.name} on GPU {gpu_id}: {e}")
#         return None

# # âœ… Parallel Feature Extraction
# def parallel_extract_features(path, num_workers=num_gpus, max_seqs=100, seq_len=300):
#     files = list(Path(path).glob('*.fastq'))
#     print(f"ğŸ”� Found {len(files)} files in {path}")
#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         results = list(executor.map(
#             lambda args: process_file_on_gpu(args, max_seqs, seq_len),
#             [(f, i % num_gpus) for i, f in enumerate(files)]
#         ))
#     features = [f for f in results if f is not None]
#     return cudf.DataFrame(features)

# # âœ… Main Execution
# if __name__ == '__main__':
#     print("ğŸš€ Starting numeric feature extraction from FASTQ files...")

#     train_dirs = [
#         "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TrainFiles/TrainFiles",
#         "/kaggle/input/secondbatchoffastqfiles/TrainFiles"
#     ]

#     df_train = cudf.DataFrame()
#     for folder in train_dirs:
#         df_part = parallel_extract_features(folder, max_seqs=6000, seq_len=9000)
#         df_train = cudf.concat([df_train, df_part], ignore_index=True)

#     # âœ… Load and merge training labels
#     train_labels = cudf.read_csv("/kaggle/input/extrafiles/Train.csv")
#     train_labels['fastq_filename'] = train_labels['filename'].str.replace('.mgb', '.fastq')
#     df_train = df_train.merge(
#         train_labels[['fastq_filename', 'SampleType']],
#         left_on='filename',
#         right_on='fastq_filename'
#     )

#     # âœ… Label Encoding
#     le = LabelEncoder()
#     df_train['label'] = le.fit_transform(df_train['SampleType'])

#     # âœ… Feature columns and target
#     feature_cols = [col for col in df_train.columns if col.startswith('base_')]
#     X = df_train[feature_cols]
#     y = df_train['label']

#     # âœ… XGBoost Parameters
#     params = {
#         'objective': 'multi:softprob',
#         'eval_metric': 'mlogloss',
#         'num_class': len(le.classes_),
#         'tree_method': 'gpu_hist',
#         'predictor': 'gpu_predictor',
#         'learning_rate': 0.05,
#         'max_depth': 8,
#         'subsample': 0.9,
#         'colsample_bytree': 0.9,
#         'reg_alpha': 0.1,
#         'reg_lambda': 0.1,
#         'random_state': 42
#     }

#     # âœ… Training with K-Fold
#     print("ğŸ”¥ Training XGBoost with numeric sequence vectors...")
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     oof_preds = cp.zeros((len(X), len(le.classes_)), dtype=cp.float32)
#     models = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
#         print(f"\nğŸŒ€ Fold {fold + 1}")
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         dtrain = xgb.DMatrix(X_train, label=y_train)
#         dval = xgb.DMatrix(X_val, label=y_val)

#         model = xgb.train(
#             params,
#             dtrain,
#             num_boost_round=16000,
#             evals=[(dtrain, 'train'), (dval, 'val')],
#             # early_stopping_rounds=50,
#             verbose_eval=100
#         )

#         models.append(model)
#         oof_preds[val_idx] = model.predict(dval)

#     # âœ… OOF Evaluation
#     oof_loss = log_loss(
#         y.astype(np.int32).to_numpy(),
#         oof_preds.astype(cp.float32).get()
#     )
#     print(f"\nğŸ“Š Out-of-Fold Log Loss: {oof_loss:.5f}")

#     # âœ… Test Data Feature Extraction
#     print("\nâš¡ Extracting test features...")
#     test_dir = "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TestFiles/TestFiles"
#     df_test = parallel_extract_features(test_dir, max_seqs=6000, seq_len=9000)
#     X_test = df_test[feature_cols]

#     # âœ… Inference on Test Set
#     print("\nğŸ”® Generating predictions...")
#     test_preds = cp.zeros((len(X_test), len(le.classes_)), dtype=cp.float32)

#     for model in models:
#         dtest = xgb.DMatrix(X_test)
#         preds = cp.array(model.predict(dtest), dtype=cp.float32)
#         test_preds += preds

#     test_preds /= len(models)

#     # âœ… Prepare Submission
#     submission = cudf.DataFrame({
#         'filename': df_test['filename'].str.replace('.fastq', ''),
#         'Mouth': test_preds[:, le.transform(['Mouth'])[0]].get(),
#         'Nasal': test_preds[:, le.transform(['Nasal'])[0]].get(),
#         'Skin': test_preds[:, le.transform(['Skin'])[0]].get(),
#         'Stool': test_preds[:, le.transform(['Stool'])[0]].get()
#     })

#     submission = submission[['filename', 'Mouth', 'Nasal', 'Skin', 'Stool']]
#     submission.to_csv('submission.csv', index=False)
#     print("\nâœ… Submission saved successfully as `submission.csv`")


# âœ… Install Required Packages
!pip install -q Bio 

# âœ… Imports
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# âœ… TF GPU Strategy
strategy = tf.distribute.MirroredStrategy()
print("ğŸŸ¢ Number of GPUs:", strategy.num_replicas_in_sync)

# âœ… Helper: Encode ACGTN to 0-4 and pad
def encode_sequence(seq, max_len=9000):
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = [base_map.get(base, 4) for base in seq[:max_len]]
    if len(encoded) < max_len:
        encoded += [4] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int8)

# âœ… Read and encode sequences from FASTQ
def load_fastq_sequences(folder, max_seqs=6000, seq_len=9000):
    print(f"ğŸ”� Reading FASTQ from {folder}")
    X, filenames = [], []
    for path in Path(folder).glob("*.fastq"):
        try:
            records = list(SeqIO.parse(str(path), "fastq"))[:max_seqs]
            if not records:
                continue
            seqs = [encode_sequence(str(rec.seq), seq_len) for rec in records]
            X.append(np.mean(seqs, axis=0))
            filenames.append(path.name)
        except Exception as e:
            print(f"âš ï¸� Failed to process {path.name}: {e}")
    return np.array(X, dtype=np.float32), filenames

# âœ… Load Training Data
train_dirs = [
    "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TrainFiles/TrainFiles",
    "/kaggle/input/secondbatchoffastqfiles/TrainFiles"
]
X_list, fname_list = [], []
for folder in train_dirs:
    x, fnames = load_fastq_sequences(folder)
    X_list.append(x)
    fname_list.extend(fnames)
X = np.vstack(X_list)
filenames = np.array(fname_list)

# âœ… Labels
import pandas as pd
df_labels = pd.read_csv("/kaggle/input/extrafiles/Train.csv")
df_labels["fastq_filename"] = df_labels["filename"].str.replace(".mgb", ".fastq")
df_train = pd.DataFrame({"filename": filenames})
df_train = df_train.merge(df_labels, left_on="filename", right_on="fastq_filename")
le = LabelEncoder()
df_train["label"] = le.fit_transform(df_train["SampleType"])
y = df_train["label"].values
num_classes = len(le.classes_)

# âœ… Reshape for LSTM input [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# âœ… Define LSTM Model Inside Strategy Scope
def create_lstm_model(input_shape, num_classes):
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Masking(mask_value=4.0),  # For padding
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

# âœ… Train with Stratified K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸŒ€ Fold {fold+1}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = create_lstm_model(input_shape=(X.shape[1], 1), num_classes=num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )
    models.append(model)

# âœ… Load Test Set
test_dir = "/kaggle/input/mpeg-g-microbiomeclassificationconvertedfastqfiles/TestFiles/TestFiles"
X_test, test_filenames = load_fastq_sequences(test_dir)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# âœ… Predict on Test
preds = np.zeros((len(X_test), num_classes), dtype=np.float32)
for model in models:
    preds += model.predict(X_test, verbose=0)
preds /= len(models)

# âœ… Save Submission
df_submission = pd.DataFrame({
    "filename": [f.replace(".fastq", "") for f in test_filenames],
    "Mouth": preds[:, le.transform(["Mouth"])[0]],
    "Nasal": preds[:, le.transform(["Nasal"])[0]],
    "Skin": preds[:, le.transform(["Skin"])[0]],
    "Stool": preds[:, le.transform(["Stool"])[0]]
})
df_submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission saved as `submission.csv`")


