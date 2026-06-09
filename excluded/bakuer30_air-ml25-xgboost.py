import os
import shutil
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot
from typing import List, Optional
import numpy as np

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # Import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance

import optuna



PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
PATH_TRAIN_DATASETS = os.path.join(PATH_DATASET, 'train_datasets', 'train_datasets')
train_datasets = sorted(os.listdir(PATH_TRAIN_DATASETS))
print(train_datasets)
PATH_TEST_DATASETS = os.path.join(PATH_DATASET, 'test_datasets', 'test_datasets')
test_datasets = sorted(os.listdir(PATH_TEST_DATASETS))
print(test_datasets)


RANDOM_SEED = 42
N_SPLITS = 5



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


import os
import pandas as pd
from tqdm import tqdm
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




# Prepare the data
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



dt_all = X_train.copy()
dt_all['target'] = y_train

f_cols = [c for c in dt_all.columns if c not in ["target"]]

# Boolean mask of duplicated rows based only on f_cols
#dupl = dt_all.duplicated(subset=f_cols, keep="first")

# Keep only duplicated rows
#dt_all = dt_all[dupl].copy()
X_train = dt_all[f_cols]
y_train = dt_all['target']

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


X_train.head()


XGB_PARAMS  = {
    # fixed
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'random_state': RANDOM_SEED,
    'importance_type':'gain',
    ## variable
    'colsample_bytree': 0.8541925755064158,
    'learning_rate': 0.03270545286179225,
    'max_depth': 5,
    'n_estimators': 73,
    'reg_alpha': 0.007228681890108715,
    'reg_lambda': 0.11046982689903566,
    'subsample': 1.0
}



kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
val_log_loss = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
        #
    
        # --- XGBoost ---
        print("Training XGBoost...")
        # Note: XGBoost early stopping uses eval_set parameter directly in fit
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False) # verbose=False keeps output clean
        #best_iter_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else XGB_N_ESTIMATORS # Get best iteration if early stopping triggered
        val_preds_xgb = xgb_model.predict_proba(X_va)[:, 1]
        val_log_loss_iter = log_loss(y_va, val_preds_xgb)
        val_auc_iter = roc_auc_score(y_va, val_preds_xgb)
        val_log_loss.append(val_log_loss_iter)
        print(f"LogLoss = {val_log_loss_iter}")
        print(f"AUC = {val_auc_iter}")

print(f"Mean LofLoss ={np.mean(val_log_loss)} ")
print(f"Median LofLoss ={np.median(val_log_loss)} ")


LGBM_PARAMS = {
    "device": "cpu",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbose":-1,
    'objective': 'binary',
    "n_estimators":100,
    "num_leaves":31,
    "max_depth":-1,
    "learning_rate": 0.1,
    'min_child_weight':0.001,
    "subsample":1.0,
    'colsample_bytree':1.0,
    "reg_alpha":1e-3, 
    "reg_lambda":1e-3,
}




## LGBM
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
val_log_loss = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
        #
    
       
        print("Training LGMBoost...")
 
        # Note: XGBoost early stopping uses eval_set parameter directly in fit
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)]) # verbose=False keeps output clean
        #best_iter_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else XGB_N_ESTIMATORS # Get best iteration if early stopping triggered
        val_preds_lgbm = lgbm_model.predict_proba(X_va)[:, 1]
        val_log_loss_iter = log_loss(y_va, val_preds_lgbm)
        val_auc_iter = roc_auc_score(y_va, val_preds_lgbm)
        val_log_loss.append(val_log_loss_iter)
        print(f"LogLoss = {val_log_loss_iter}")
        print(f"AUC = {val_auc_iter}")

print(f"Mean LofLoss ={np.mean(val_log_loss)} ")
print(f"Median LofLoss ={np.median(val_log_loss)} ")


feature_important = xgb_model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

feature_important_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
feature_important_df.nlargest(20, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 20 features


feature_important_df.nsmallest(20, columns="score").plot(kind='barh', figsize = (20,10)) ## plot worst 20 features



# Initialize XGBoost Classifier
model_xgb = xgb.XGBClassifier(**XGB_PARAMS)
model_lgb = lgb.LGBMClassifier(**LGBM_PARAMS)

# Train the model with PCA-transformed data
print("\nTraining XGBoost model...")
model_xgb.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set with PCA-transformed data
print("\nMaking predictions on the test set...")
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1] # Get probabilities for the positive class
y_pred_proba_lgb = model_lgb.predict_proba(X_test)[:, 1] # Get probabilities for the positive class
y_pred_proba = y_pred_proba_xgb*0.8 + y_pred_proba_lgb*0.2
predictions_df = pd.DataFrame({
    'ID': dataset_test['ID'],
    'dataset': dataset_test['dataset'],
    'label_positive_pred': y_pred_proba, # Store probabilities here
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


!head 'submission.csv'

