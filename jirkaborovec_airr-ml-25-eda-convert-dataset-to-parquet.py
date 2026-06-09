import os

PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
PATH_TRAIN_DATASETS = os.path.join(PATH_DATASET, 'train_datasets', 'train_datasets')
train_datasets = sorted(os.listdir(PATH_TRAIN_DATASETS))
print(train_datasets)
PATH_TEST_DATASETS = os.path.join(PATH_DATASET, 'test_datasets', 'test_datasets')
test_datasets = sorted(os.listdir(PATH_TEST_DATASETS))
print(test_datasets)


import os
import shutil
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional


def load_tsv_files_export_parquet(folder_path: str, output_path: str, show_hist: Optional[List[str]] = None):
    folder = os.path.basename(folder_path) # Derive folder name from folder_path
    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter for .tsv files
    tsv_files = [f for f in files if f.endswith('.tsv')]
    other_files = [f.name for f in os.scandir(folder_path) if not f.name.endswith('.tsv')]
    print(f'Loading {len(tsv_files)} .tsv files from {folder} (remaining: {other_files}).')

    # Iterate through each TSV file, load it into a DataFrame, and print column names
    dfs = []
    for tsv_file in tqdm(tsv_files, desc="Loading TSV files"):
        file_path = os.path.join(folder_path, tsv_file)
        file_name, _ = os.path.splitext(tsv_file)
        try:
            df = pd.read_csv(file_path, sep='\t')
            df['repertoire_id'] = file_name
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")

    merged_df = pd.concat(dfs, ignore_index=True)
    del dfs # Free up memory

    print(f"Merged DataFrame shape: {merged_df.shape}")
    for col in merged_df.columns:
        print(f"Unique values in column '{col}': {len(merged_df[col].unique())}")
    print("Merged DataFrame head:")
    display(merged_df.head())

    os.makedirs(output_path, exist_ok=True)
    merged_df.to_parquet(f'{output_path}/{folder}.parquet')

    # Plot histograms for specified columns if show_hist is provided
    if not isinstance(show_hist, list) and not show_hist:
        return
    print(f"Plotting histograms for columns: {', '.join(show_hist)}")
    for col in show_hist:
        if col not in merged_df.columns:
            print(f"Warning: Column '{col}' not found in the DataFrame for {folder}.")
            continue
        # Get all value counts
        all_counts = merged_df[col].value_counts()
        
        plt.figure(figsize=(min(12, len(all_counts) * 0.3), 4)) # Adjust figure size dynamically
        sns.barplot(x=all_counts.index, y=all_counts.values, palette='viridis')
        plt.title(f'Value Counts for {col} in {folder}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=90, ha='center') # Rotate labels more for many categories
        plt.grid(True)
        plt.tight_layout()
    plt.show()


# Iterate over all sub-datasets
for folder in tqdm(train_datasets):
    path_dataset_ = os.path.join(PATH_TRAIN_DATASETS, folder)
    load_tsv_files_export_parquet(
        path_dataset_, output_path='train_dataset', show_hist=['v_call', 'j_call', 'd_call'])
    new_meta_csv = os.path.join("train_dataset", f"{folder}-metadata.csv")
    shutil.copy(os.path.join(path_dataset_, "metadata.csv"), new_meta_csv)
    df_meta = pd.read_csv(new_meta_csv)
    display(df_meta)


# Iterate over all sub-datasets
for folder in tqdm(test_datasets):
    path_dataset_ = os.path.join(PATH_TEST_DATASETS, folder)
    load_tsv_files_export_parquet(
        path_dataset_, output_path='test_dataset', show_hist=['v_call', 'j_call', 'd_call'])


# Construct the path to the sample submission file
sample_submission_path = os.path.join(PATH_DATASET, 'sample_submissions.csv')

# Load the sample submission file
sample_submission_df = pd.read_csv(sample_submission_path)

# You can save this to a CSV file if needed
sample_submission_df.to_csv('submission.csv', index=False)

