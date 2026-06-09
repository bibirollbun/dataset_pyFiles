import os
import shutil
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot
from typing import List, Optional
import numpy as np


PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
PATH_TRAIN_DATASETS = os.path.join(PATH_DATASET, 'train_datasets', 'train_datasets')
train_datasets = sorted(os.listdir(PATH_TRAIN_DATASETS))
print(train_datasets)
PATH_TEST_DATASETS = os.path.join(PATH_DATASET, 'test_datasets', 'test_datasets')
test_datasets = sorted(os.listdir(PATH_TEST_DATASETS))
print(test_datasets)


def parse_tsv_files(folder_path: str, feature_colums = ('v_call', 'j_call')):
    folder = os.path.basename(folder_path) # Derive folder name from folder_path
    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter for .tsv files
    tsv_files = [f for f in files if f.endswith('.tsv')]
    other_files = [f.name for f in os.scandir(folder_path) if not f.name.endswith('.tsv')]
    print(f'Loading {len(tsv_files)} .tsv files from {folder} (remaining: {other_files}).')

    metadata = None
    if "metadata.csv" in files:
        metadata = pd.read_csv(os.path.join(folder_path, "metadata.csv"))
        metadata.set_index("filename", inplace=True)

    # Iterate through each TSV file, load it into a DataFrame, and print column names
    dataset = []
    for tsv_file in tqdm(tsv_files, desc="Loading TSV files"):
        file_path = os.path.join(folder_path, tsv_file)
        file_name, _ = os.path.splitext(tsv_file)
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
        
        one_case = {"ID": file_name, "dataset": folder}
        if metadata is not None:
            one_case = {"label_positive": int(metadata.at[tsv_file, "label_positive"])}
        for col in feature_colums:
            counts = df[col].value_counts() / len(df)
            one_case.update(counts.to_dict())
            # print(one_case)
            dataset.append(one_case)

    return dataset


from collections import Counter

def extract_kmers(seq: str, k: int = 5):
    """Return all k-mers from an amino-acid sequence."""
    if not isinstance(seq, str):
        return []
    seq = seq.strip()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def parse_tsv_files(
        folder_path: str, 
        feature_columns=('v_call', 'j_call'),
        kmer_column='junction_aa',
        k=3  # length of kmers
):
    folder = os.path.basename(folder_path)
    files = os.listdir(folder_path)

    # Only .tsv files
    tsv_files = [f for f in files if f.endswith('.tsv')]
    other_files = [f.name for f in os.scandir(folder_path) if not f.name.endswith('.tsv')]
    print(f'Loading {len(tsv_files)} .tsv files from {folder} (remaining: {other_files}).')

    # Metadata
    metadata = None
    if "metadata.csv" in files:
        metadata = pd.read_csv(os.path.join(folder_path, "metadata.csv"))
        metadata.set_index("filename", inplace=True)

    dataset = []

    for tsv_file in tqdm(tsv_files):
        path = os.path.join(folder_path, tsv_file)
        file_name, _ = os.path.splitext(tsv_file)

        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
            continue

        # Collect all features here
        one_case = {"ID": file_name, "dataset": folder}

        if metadata is not None:
            one_case["label_positive"] = int(metadata.at[tsv_file, "label_positive"])

        # === 1) Count V / J genes ===
        for col in feature_columns:
            counts = df[col].value_counts(normalize=True)  # normalized frequency %
            one_case.update({f"{col}__{k}": v for k, v in counts.to_dict().items()})

        # === 2) Count k-mers from junction_aa ===
        if kmer_column in df.columns:
            all_kmers = []

            for seq in df[kmer_column].dropna():
                all_kmers.extend(extract_kmers(seq, k=k))

            kmer_counts = Counter(all_kmers)

            # convert to relative frequencies
            total = sum(kmer_counts.values())
            if total > 0:
                kmer_freqs = {f"kmer{k}_{mer}": count / total for mer, count in kmer_counts.items()}
            else:
                kmer_freqs = {}

            one_case.update(kmer_freqs)

        dataset.append(one_case)

    return dataset


# Iterate over all sub-datasets
dataset = []
for folder in tqdm(train_datasets, disable=True):
    path_dataset_ = os.path.join(PATH_TRAIN_DATASETS, folder)
    dataset += parse_tsv_files(path_dataset_)

dataset_train = pd.DataFrame(dataset)
#display(dataset_train)


dataset_train.head()


f_cols = [col for col in dataset_train.columns if col not in ['ID', 'dataset']]

cols_na = dataset_train[f_cols].isnull().sum()

cols_na = cols_na[cols_na > 3100]
excluded_features = cols_na.index.to_list()

len(excluded_features)


# Iterate over all sub-datasets
dataset = []
for folder in tqdm(test_datasets, disable=True):
    path_dataset_ = os.path.join(PATH_TEST_DATASETS, folder)
    dataset += parse_tsv_files(path_dataset_)

dataset_test = pd.DataFrame(dataset)
#display(dataset_test)


f_cols = [col for col in dataset_test.columns if col not in ['ID', 'dataset']]

# 2. Sum of missing values
cols_na = dataset_test[f_cols].isnull().sum()

# 3. Just an attempt to reduce variables with many missing values
cols_na = cols_na[cols_na > 4000]
excluded_features_test = cols_na.index.to_list()

excluded_features =  excluded_features  +  excluded_features_test
excluded_features = list(set(excluded_features))
len(excluded_features)


extra_excluded = ["TCRBV22.X","TCRBV29.X","TCRBV25.or09_02", "TCRBV29.or09_02", 
                      "X","TCRBV24"]
excluded_features +=  extra_excluded
excluded_features = list(set(excluded_features))
print(f"n_excluded: {len(excluded_features)}")
excluded_features[:10]


X_train0 = dataset_train.drop(
    ['label_positive', 'ID', 'dataset'], axis=1, errors='ignore').fillna(0)
y_train = dataset_train['label_positive']

X_test = dataset_test.drop(
    ['label_positive', 'ID', 'dataset'], axis=1, errors='ignore').fillna(0)
y_test = pd.Series([]) # Initialize y_test as an empty Series as label_positive is not in test data

if len(excluded_features)>0:
    X_train = X_train0.drop(excluded_features, axis=1, errors='ignore').copy()
# Align columns - crucial for XGBoost to avoid 'feature_names mismatch'
# Ensure X_test only has columns present in X_train
#X_train = X_train0
missing_in_test_but_in_train = set(X_train.columns) - set(X_test.columns)
for c in missing_in_test_but_in_train:
    X_test[c] = 0

X_test = X_test[X_train.columns] # Crop columns in X_test that are not in X_train and ensure order

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# ------------------- Scale the data (required for PCA) -------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------- PCA (keep 95% variance or set fixed components) -------------------
pca = PCA(n_components=0.95, random_state=42)   # or: n_components=300
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)


print("PCA components:", X_train_pca.shape[1])



X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_pca, y_train, test_size=0.2,
    random_state=42, stratify=y_train
)




base_models = [
    ("xgb", xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, eval_metric='logloss'
    )),

    ("lgb", lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.03,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbosity=-1
    )),

    ("cat", CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        verbose=False
    )),

]



meta_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function='Logloss',
    verbose=False
)



stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',
    passthrough=True,
    n_jobs=-1
)



stack_model.fit(X_tr, y_tr)

val_pred = stack_model.predict_proba(X_val)[:, 1]
print("Stacking + CatBoost AUC:", roc_auc_score(y_val, val_pred))



stack_model.fit(X_train_pca, y_train)


test_pred = stack_model.predict_proba(X_test_pca)[:, 1]
print("Test prediction shape:", test_pred.shape)



predictions_df = pd.DataFrame({
    'ID': dataset_test['ID'],
    'dataset': dataset_test['dataset'],
    'label_positive_pred': test_pred, # Store probabilities here
})
print(f"Prediction has {len(predictions_df)} rows")
display(predictions_df.head())


# Construct the path to the sample submission file
sample_submission_path = os.path.join(PATH_DATASET, 'sample_submissions.csv')

# Load the sample submission file
sample_submission_df = pd.read_csv(sample_submission_path)
print(f"Sample submissions has {len(sample_submission_df)} rows")
print("Sample Submission DataFrame Head:")
display(sample_submission_df.head())

# Merge predictions_df with sample_submission_df
# First, drop the existing 'label_positive' from sample_submission_df if it exists, as we will replace it
sample_submission_df = sample_submission_df.drop(columns=['label_positive_probability'])

# Now merge the predictions_df, which contains 'filename' and 'label_positive_pred'
submission_df = pd.merge(
    sample_submission_df,
    predictions_df,
    on=['ID', 'dataset'],
    how='left',
)

# Rename 'label_positive_pred' to 'label_positive' for the final submission format
submission_df = submission_df.rename(columns={'label_positive_pred': 'label_positive_probability'})
submission_df = submission_df.fillna(0.5)
# Remove the duplicates from the final result, keeping the first occurrence
submission_df = submission_df.drop_duplicates(subset=['ID', 'dataset'], keep='first')

print(f"Sample submissions has {len(submission_df)} rows")
# Display the head of the final submission DataFrame
print("Final Submission DataFrame Head:")
display(submission_df.head())

# You can save this to a CSV file if needed
submission_df.to_csv('submission.csv', index=False)

